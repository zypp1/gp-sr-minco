"""Random collision-free goal publisher for SUPER.

Picks collision-free goals in a ring around the drone,
then advances either from the collector's saved-result event or directly from
odometry when running inference without a collector.

    python3 scripts/data_collect/goals.py --map output/maps/mixed_forest_200_dense1800_seed1.npz
    python3 scripts/data_collect/goals.py --map <map.npz> --completion-source odom
"""

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from neural_minco.geometry import build_esdf, query_esdf

GOAL_CLEARANCE_M = 1.0  # minimum ESDF clearance for a candidate goal
PATH_BUILDING_CLEARANCE_M = 0.3  # robot radius plus collection safety margin
PUBLISH_RATE_HZ = 5.0  # main loop rate
NEXT_GOAL_DELAY_S = 0.5
ODOM_REACH_DISTANCE_M = 0.1
ODOM_REACH_SPEED_MPS = 0.2
EPISODE_SUCCESS_TOPIC = "/planning/episode_success"
GOAL_FAILURE_TOPIC = "/planning/goal_failed"


@dataclass(frozen=True)
class BuildingVolume:
    """Represent one oriented building volume from static-scene metadata."""

    center_x: float
    center_y: float
    width: float
    depth: float
    height: float
    yaw: float

    def contains(self, xyz, margin: float = 0.0) -> bool:
        """Return whether a world point lies inside the building plus a safety margin."""
        point = np.asarray(xyz, dtype=np.float64)
        if point.shape != (3,) or margin < 0.0:
            raise ValueError("building containment requires one XYZ point and a nonnegative margin")
        if point[2] < -margin or point[2] > self.height + margin:
            return False
        delta_x = float(point[0] - self.center_x)
        delta_y = float(point[1] - self.center_y)
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        return abs(local_x) <= 0.5 * self.width + margin and abs(local_y) <= 0.5 * self.depth + margin

    def intersects_segment(self, start, end, margin: float = 0.0) -> bool:
        """Return whether a world-frame segment enters the expanded solid building."""
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or margin < 0.0:
            raise ValueError("building intersection requires two XYZ points and a nonnegative margin")

        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)

        def to_local(point):
            delta_x = float(point[0] - self.center_x)
            delta_y = float(point[1] - self.center_y)
            return np.asarray([cosine * delta_x + sine * delta_y, -sine * delta_x + cosine * delta_y, float(point[2])])

        local_start = to_local(start)
        direction = to_local(end) - local_start
        lower = np.asarray([-0.5 * self.width - margin, -0.5 * self.depth - margin, -margin])
        upper = np.asarray([0.5 * self.width + margin, 0.5 * self.depth + margin, self.height + margin])

        enter = 0.0
        leave = 1.0
        for axis in range(3):
            if abs(direction[axis]) <= 1e-12:
                if local_start[axis] < lower[axis] or local_start[axis] > upper[axis]:
                    return False
                continue
            first = (lower[axis] - local_start[axis]) / direction[axis]
            second = (upper[axis] - local_start[axis]) / direction[axis]
            enter = max(enter, min(first, second))
            leave = min(leave, max(first, second))
            if enter > leave:
                return False
        return True


def _load_building_volumes(
    npz_path, point_count: int, lower: np.ndarray, upper: np.ndarray, metadata_path: Optional[str] = None
) -> Tuple[Tuple[BuildingVolume, ...], Path]:
    """Load validated building volumes from the generator's required JSON sidecar."""
    source = (
        Path(metadata_path).expanduser()
        if metadata_path is not None
        else Path(npz_path).expanduser().with_suffix(".json")
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is required for building-safe goal sampling; "
            "keep pcl_generate.py's JSON sidecar or pass --map-metadata"
        )
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read scene metadata {source}: {exc}") from exc
    if metadata.get("format") != "static_xyz_occupancy" or metadata.get("frame") != "world":
        raise ValueError(f"{source} must describe a world-frame static_xyz_occupancy map")
    if metadata.get("point_count") != point_count:
        raise ValueError(f"{source} point_count does not match {npz_path}")
    bounds = metadata.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError(f"{source} is missing map bounds")
    try:
        metadata_lower = np.asarray(bounds["min"], dtype=np.float64)
        metadata_upper = np.asarray(bounds["max"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} contains invalid map bounds") from exc
    if (
        metadata_lower.shape != (3,)
        or metadata_upper.shape != (3,)
        or not np.allclose(metadata_lower, lower, atol=1e-5, rtol=0.0)
        or not np.allclose(metadata_upper, upper, atol=1e-5, rtol=0.0)
    ):
        raise ValueError(f"{source} bounds do not match {npz_path}")

    placements = metadata.get("building_placements")
    objects = metadata.get("objects")
    declared_count = objects.get("buildings") if isinstance(objects, dict) else None
    if not isinstance(placements, list) or declared_count != len(placements):
        raise ValueError(f"{source} contains inconsistent building metadata")
    buildings = []
    for index, placement in enumerate(placements):
        try:
            center = np.asarray(placement["center_m"], dtype=np.float64)
            size = np.asarray(placement["size_m"], dtype=np.float64)
            yaw = float(placement["yaw_rad"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source} building {index} is malformed") from exc
        if center.shape != (2,) or size.shape != (3,):
            raise ValueError(f"{source} building {index} contains invalid geometry")
        values = np.concatenate([center, size, np.asarray([yaw])])
        if not np.isfinite(values).all():
            raise ValueError(f"{source} building {index} contains invalid geometry")
        if np.any(size <= 0.0):
            raise ValueError(f"{source} building {index} dimensions must be positive")
        buildings.append(
            BuildingVolume(
                center_x=float(center[0]),
                center_y=float(center[1]),
                width=float(size[0]),
                depth=float(size[1]),
                height=float(size[2]),
                yaw=yaw,
            )
        )
    return tuple(buildings), source


def _load_map(npz_path, margin, metadata_path=None):
    """Build the ESDF, map bounds, and exact building exclusion volumes."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "points" not in data:
            raise ValueError(f"{npz_path} is missing the required 'points' array")
        pts = np.asarray(data["points"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError(f"{npz_path} points must be finite with shape [N,3]")
    lower = pts.min(axis=0).astype(np.float64)
    upper = pts.max(axis=0).astype(np.float64)
    buildings, metadata_source = _load_building_volumes(npz_path, len(pts), lower, upper, metadata_path)
    xr = (float(lower[0] + margin), float(upper[0] - margin))
    yr = (float(lower[1] + margin), float(upper[1] - margin))
    if xr[0] >= xr[1] or yr[0] >= yr[1]:
        raise ValueError("map margin leaves no valid horizontal goal-sampling area")
    esdf = build_esdf(pts, voxel=0.2)
    return esdf, xr, yr, buildings, metadata_source


def _clearance(esdf, xyz):
    return float(query_esdf(esdf, torch.tensor(xyz, dtype=torch.float32).view(1, 3)))


def _sample(esdf, origin, z, dist_min, dist_max, xr, yr, max_tries=200, buildings=()):
    """Return a free in-bounds point in the requested distance annulus."""
    for _ in range(max_tries):
        a = random.uniform(0, 2 * math.pi)
        r = random.uniform(dist_min, dist_max)
        x = origin[0] + r * math.cos(a)
        y = origin[1] + r * math.sin(a)
        if not (xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]):
            continue
        c = np.array([x, y, z])
        if any(building.contains(c, margin=GOAL_CLEARANCE_M) for building in buildings):
            continue
        if any(building.intersects_segment(origin, c, margin=PATH_BUILDING_CLEARANCE_M) for building in buildings):
            continue
        if _clearance(esdf, c) >= GOAL_CLEARANCE_M:
            return c
    return None


def _same_goal(first, second, tolerance=1e-3):
    """Return whether two complete world-frame goal positions identify the same target."""
    return (
        first is not None and second is not None and np.linalg.norm(np.asarray(first) - np.asarray(second)) <= tolerance
    )


def _has_reached_state(position, velocity, goal, distance_tolerance, speed_tolerance):
    """Return whether world-frame odometry is close to and settled at the goal."""
    if min(distance_tolerance, speed_tolerance) < 0.0 or not np.isfinite([distance_tolerance, speed_tolerance]).all():
        raise ValueError("goal tolerances must be finite and nonnegative")
    if position is None or velocity is None or goal is None:
        return False
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if (
        position.shape != (3,)
        or velocity.shape != (3,)
        or goal.shape != (3,)
        or not np.isfinite(np.concatenate([position, velocity, goal])).all()
    ):
        raise ValueError("goal reach check requires finite XYZ position, velocity, and goal")
    return bool(np.linalg.norm(position - goal) <= distance_tolerance and np.linalg.norm(velocity) <= speed_tolerance)


def _replacement_reason(current, success_at, failed, now, next_goal_delay):
    """Return why the current goal must be replaced, or ``None`` while it remains active."""
    if current is None:
        return "initial"
    if failed:
        return "failure"
    if success_at is not None:
        return "success" if now - success_at >= next_goal_delay else None
    return None


def main():
    """Parse args and publish random collision-free goals for SUPER."""
    import rospy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry

    parser = argparse.ArgumentParser(description="Random goal publisher for SUPER")
    parser.add_argument("--map", required=True, help="map point cloud .npz (key 'points')")
    parser.add_argument(
        "--map-metadata", help="matching pcl_generate.py JSON metadata (default: --map path with .json suffix)"
    )
    parser.add_argument("--z", type=float, default=1.5, help="goal height (m)")
    parser.add_argument("--margin", type=float, default=3.0, help="inset from map bounds (m)")
    parser.add_argument("--dist-min", type=float, default=8.0, help="min goal distance from drone")
    parser.add_argument("--dist-max", type=float, default=30.0, help="max goal distance from drone")
    parser.add_argument("--init-delay", type=float, default=2.0, help="hover time before first goal (s)")
    parser.add_argument("--max-successes", type=int, default=0, help="stop after N saved successes (0 = unlimited)")
    parser.add_argument(
        "--completion-source",
        choices=("collector", "odom"),
        default="collector",
        help=(
            "collector: wait for a matched saved Episode result; "
            "odom: advance when odometry reaches the goal (for infer.sh)"
        ),
    )
    parser.add_argument(
        "--reach-distance",
        type=float,
        default=ODOM_REACH_DISTANCE_M,
        help="3D goal distance used only with --completion-source odom",
    )
    parser.add_argument(
        "--reach-speed",
        type=float,
        default=ODOM_REACH_SPEED_MPS,
        help="maximum 3D speed at goal completion with --completion-source odom",
    )
    parser.add_argument(
        "--next-goal-delay",
        type=float,
        default=NEXT_GOAL_DELAY_S,
        help="delay after a saved successful episode before publishing the next goal",
    )
    # parse_known_args: ignore the __name/__log args that roslaunch injects.
    args, _ = parser.parse_known_args()
    if args.max_successes < 0:
        parser.error("--max-successes must be nonnegative")
    if args.next_goal_delay < 0:
        parser.error("--next-goal-delay must be nonnegative")
    if (
        not np.isfinite(
            [
                args.z,
                args.margin,
                args.dist_min,
                args.dist_max,
                args.reach_distance,
                args.reach_speed,
                args.next_goal_delay,
            ]
        ).all()
        or args.margin < 0.0
        or args.dist_min < 0.0
        or args.dist_max < args.dist_min
        or args.reach_distance < 0.0
        or args.reach_speed < 0.0
    ):
        parser.error("goal geometry arguments are invalid")

    try:
        esdf, xr, yr, buildings, metadata_source = _load_map(args.map, args.margin, args.map_metadata)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    rospy.init_node("random_goal", anonymous=True)
    rospy.loginfo(f"loaded {len(buildings)} building exclusion volumes from {metadata_source}")
    goal_pub = rospy.Publisher("/planning/click_goal", PoseStamped, queue_size=1, latch=True)
    odom_pos = None
    odom_vel = None
    current = None
    success_received_at = None
    goal_failed = False
    result_queue = deque()
    success_count = 0

    def _odom_cb(msg):
        nonlocal odom_pos, odom_vel
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        odom_pos = np.array([p.x, p.y, p.z])
        odom_vel = np.array([v.x, v.y, v.z])

    def _store_result(msg, success):
        goal = np.asarray([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        result_queue.append((goal, success, rospy.Time.now().to_sec()))

    def _success_cb(msg):
        _store_result(msg, success=True)

    def _failure_cb(msg):
        _store_result(msg, success=False)

    rospy.Subscriber("/lidar_slam/odom", Odometry, _odom_cb, queue_size=1)
    if args.completion_source == "collector":
        rospy.Subscriber(EPISODE_SUCCESS_TOPIC, PoseStamped, _success_cb, queue_size=10)
        rospy.Subscriber(GOAL_FAILURE_TOPIC, PoseStamped, _failure_cb, queue_size=10)
    else:
        rospy.loginfo(
            f"goal completion uses odometry within {args.reach_distance:.2f} m "
            f"at no more than {args.reach_speed:.2f} m/s"
        )

    while odom_pos is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    if rospy.is_shutdown():
        return
    rospy.loginfo(f"odom ready, hover {args.init_delay:.0f}s")
    rospy.sleep(args.init_delay)

    rate = rospy.Rate(PUBLISH_RATE_HZ)

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        if (
            args.completion_source == "odom"
            and current is not None
            and success_received_at is None
            and _has_reached_state(odom_pos, odom_vel, current, args.reach_distance, args.reach_speed)
        ):
            success_count += 1
            if args.max_successes and success_count >= args.max_successes:
                rospy.loginfo(f"reached {success_count} goals; stopping goals")
                return
            success_received_at = now.to_sec()
            rospy.loginfo(f"goal reached from odometry; waiting {args.next_goal_delay:.1f}s before the next goal")
        while result_queue:
            result_goal, succeeded, received_at = result_queue.popleft()
            if _same_goal(current, result_goal):
                if succeeded:
                    success_count += 1
                    if args.max_successes and success_count >= args.max_successes:
                        rospy.loginfo(f"collected {success_count} successful episodes; stopping goals")
                        return
                    success_received_at = received_at
                    rospy.loginfo(
                        f"goal reached and episode saved; waiting {args.next_goal_delay:.1f}s before the next goal"
                    )
                else:
                    goal_failed = True
                    rospy.logwarn("collector rejected or timed out the current goal; sampling a replacement")
        replacement_reason = _replacement_reason(
            current, success_received_at, goal_failed, now.to_sec(), args.next_goal_delay
        )
        if replacement_reason is not None:
            current = None
            success_received_at = None
            goal_failed = False
            goal = _sample(esdf, odom_pos, args.z, args.dist_min, args.dist_max, xr, yr, buildings=buildings)
            if goal is None:
                rospy.logwarn_throttle(5.0, "no free goal found")
                rate.sleep()
                continue
            current = goal
            rospy.loginfo(
                f"goal ({current[0]:.1f}, {current[1]:.1f}), dist {np.linalg.norm(odom_pos[:2] - current[:2]):.1f}m"
            )
            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = "world"
            msg.pose.position.x = current[0]
            msg.pose.position.y = current[1]
            msg.pose.position.z = current[2]
            msg.pose.orientation.w = 1.0
            goal_pub.publish(msg)

        rate.sleep()


if __name__ == "__main__":
    main()
