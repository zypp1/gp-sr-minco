"""Collect LiDAR, odometry, and command frames for complete goal episodes."""

import argparse
import os
import sys
import threading
from collections import deque
from datetime import datetime

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PolynomialTrajectory, PositionCommand
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.dirname(_HERE), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data_collect.collect_utils.sync import (
    first_synchronized_command,
    frame_stream_metrics,
    has_reached_goal,
    is_post_goal_position_trajectory,
    nearest_frame,
    resolve_nearest_frame,
    trajectory_command_cutoff,
)

from neural_minco.config import SENSOR_SYNC_TOLERANCE_S, default_config
from neural_minco.data import next_episode_sequence, save_episode_env
from neural_minco.geometry import quaternion_matrix_numpy
from neural_minco.lidar import RangeImageRenderer

ODOM_BUFFER_DEPTH = 512
COMMAND_BUFFER_DEPTH = 512
PENDING_CLOUD_DEPTH = 8
MAX_FRAME_DELTA = SENSOR_SYNC_TOLERANCE_S
TRAJECTORY_COMMAND_GUARD = 0.02
MIN_LIDAR_RATE_HZ = 9.5
MAX_LIDAR_GAP_S = 0.15
GOAL_TOLERANCE_M = 0.1
GOAL_SPEED_TOLERANCE_MPS = 0.2
EPISODE_SUCCESS_TOPIC = "/planning/episode_success"
GOAL_FAILURE_TOPIC = "/planning/goal_failed"
TERMINATION_SUCCESS = 0
TERMINATION_GOAL_REPLACED = 1
TERMINATION_TIMEOUT = 2
TERMINATION_SHUTDOWN = 3
TERMINATION_INSUFFICIENT_MOTION = 4
TERMINATION_INSUFFICIENT_LIDAR_RATE = 5
TERMINATION_NAMES = {
    TERMINATION_SUCCESS: "success",
    TERMINATION_GOAL_REPLACED: "goal_replaced",
    TERMINATION_TIMEOUT: "timeout",
    TERMINATION_SHUTDOWN: "shutdown",
    TERMINATION_INSUFFICIENT_MOTION: "insufficient_motion",
    TERMINATION_INSUFFICIENT_LIDAR_RATE: "insufficient_lidar_rate",
}


class EpisodeCollectNode:
    """Record one fixed goal with its LiDAR and executed-odometry streams."""

    def __init__(
        self,
        cfg: dict,
        out_dir: str,
        map_id: str,
        file_tag: str,
        max_episodes: int,
        goal_tolerance: float,
        goal_speed_tolerance: float,
        goal_change_tolerance: float,
        timeout: float,
        min_arc_length: float,
        max_odom_dt: float,
        max_command_dt: float,
        min_lidar_rate: float,
        max_lidar_gap: float,
    ):
        lidar = cfg["lidar"]
        self.renderer = RangeImageRenderer(
            r_max=lidar["r_max"],
            n_horizon=lidar["n_horizon"],
            n_vertical=lidar["n_vertical"],
            horizon_fov_deg=lidar["horizon_fov_deg"],
            vertical_fov_min_deg=lidar["vertical_fov_min_deg"],
            vertical_fov_max_deg=lidar["vertical_fov_max_deg"],
        )
        self.lidar_geometry = np.asarray(
            [
                lidar["r_max"],
                lidar["horizon_fov_deg"] / lidar["n_horizon"],
                lidar["n_horizon"],
                lidar["n_vertical"],
                lidar["horizon_fov_deg"],
                lidar["vertical_fov_min_deg"],
                lidar["vertical_fov_max_deg"],
                1,
            ],
            dtype=np.float64,
        )
        self.out_dir = os.path.abspath(out_dir)
        self.map_id = str(map_id)
        self.file_tag = str(file_tag)
        self.max_episodes = int(max_episodes)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_speed_tolerance = float(goal_speed_tolerance)
        self.goal_change_tolerance = float(goal_change_tolerance)
        self.timeout = float(timeout)
        self.min_arc_length = float(min_arc_length)
        self.max_odom_dt = float(max_odom_dt)
        self.max_command_dt = float(max_command_dt)
        self.min_lidar_rate = float(min_lidar_rate)
        self.max_lidar_gap = float(max_lidar_gap)
        values = (
            self.goal_tolerance,
            self.goal_speed_tolerance,
            self.goal_change_tolerance,
            self.timeout,
            self.min_arc_length,
            self.max_odom_dt,
            self.max_command_dt,
            self.min_lidar_rate,
            self.max_lidar_gap,
        )
        if (
            self.max_episodes < 0
            or any(value < 0.0 for value in values)
            or self.goal_tolerance == 0.0
            or self.timeout == 0.0
            or self.max_odom_dt == 0.0
            or self.max_command_dt == 0.0
            or self.min_lidar_rate == 0.0
            or self.max_lidar_gap == 0.0
            or self.max_odom_dt > MAX_FRAME_DELTA
            or self.max_command_dt > MAX_FRAME_DELTA
            or not self.map_id
        ):
            raise ValueError("episode collection thresholds are invalid")

        self.lock = threading.Lock()
        self.odom_buf = deque(maxlen=ODOM_BUFFER_DEPTH)
        self.command_buf = deque(maxlen=COMMAND_BUFFER_DEPTH)
        self.pending_clouds = deque(maxlen=PENDING_CLOUD_DEPTH)
        self.last_odom_stamp = None
        self.last_command_stamp = None
        self.goal_position = None
        self.pending_goal = None
        self.active = None
        self.episode_count = 0
        self.success_count = 0
        self.drop_counts = {}
        self.saved = False
        self.fatal_error = False
        os.makedirs(self.out_dir, exist_ok=True)
        self.next_sequence = next_episode_sequence(self.out_dir)

        self.success_pub = rospy.Publisher(EPISODE_SUCCESS_TOPIC, PoseStamped, queue_size=1, latch=True)
        self.failure_pub = rospy.Publisher(GOAL_FAILURE_TOPIC, PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber("/lidar_slam/odom", Odometry, self._odom_cb, queue_size=100)
        rospy.Subscriber("/planning/pos_cmd", PositionCommand, self._command_cb, queue_size=100)
        rospy.Subscriber("/planning_cmd/poly_traj", PolynomialTrajectory, self._trajectory_cb, queue_size=100)
        rospy.Subscriber("/planning/click_goal", PoseStamped, self._goal_cb, queue_size=10)
        rospy.Subscriber("/cloud_registered", PointCloud2, self._cloud_cb, queue_size=2)
        rospy.on_shutdown(self._save)

    def _count_drop_locked(self, reason: str):
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1

    def _goal_cb(self, msg):
        stamp = msg.header.stamp.to_sec() or rospy.Time.now().to_sec()
        goal = np.asarray([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        stop_requested = False
        with self.lock:
            same_goal = (
                self.goal_position is not None
                and np.linalg.norm(goal - self.goal_position) <= self.goal_change_tolerance
            )
            if same_goal:
                return
            if self.active is not None:
                stop_requested = self._finish_locked(TERMINATION_GOAL_REPLACED)
            if not stop_requested:
                self.pending_clouds.clear()
                self.goal_position = goal
                self.pending_goal = {"goal_stamp": float(stamp), "goal": goal, "command_not_before": None}
        if stop_requested:
            reason = "episode save failed" if self.fatal_error else "reached max_episodes"
            rospy.signal_shutdown(reason)

    def _trajectory_cb(self, msg):
        """Accept the first complete SUPER trajectory committed after the pending goal."""
        message_stamp = msg.header.stamp.to_sec()
        trajectory_start_stamp = msg.start_WT_pos.to_sec()
        with self.lock:
            if self.pending_goal is None or self.active is not None:
                return
            goal_stamp = self.pending_goal["goal_stamp"]
            if self.pending_goal["command_not_before"] is not None:
                return
            if not is_post_goal_position_trajectory(
                goal_stamp,
                message_stamp,
                trajectory_start_stamp,
                int(msg.piece_num_pos),
                int(msg.type),
                int(msg.POSITION_TRAJ),
            ):
                return
            self.pending_goal["command_not_before"] = trajectory_command_cutoff(
                message_stamp, trajectory_start_stamp, TRAJECTORY_COMMAND_GUARD
            )
            self._try_start_locked(message_stamp)

    def _cloud_cb(self, msg):
        stamp = msg.header.stamp.to_sec()
        with self.lock:
            active = self.active
            if active is None or stamp < active["command_start_stamp"] - 1e-6:
                return
            if stamp <= active["last_candidate_stamp"] + 1e-9:
                self._count_drop_locked("nonmonotonic_cloud")
                return
            active["last_candidate_stamp"] = float(stamp)
        points = np.asarray(
            list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64
        )
        with self.lock:
            if self.active is not active:
                return
            self.pending_clouds.append((float(stamp), points))

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        stamp = msg.header.stamp.to_sec()
        velocity = np.asarray([v.x, v.y, v.z], dtype=np.float64)
        stop_requested = False
        with self.lock:
            if self.last_odom_stamp is not None and stamp <= self.last_odom_stamp + 1e-9:
                self._count_drop_locked("nonmonotonic_odom")
                return
            entry = (stamp, np.asarray([p.x, p.y, p.z], dtype=np.float64), (q.w, q.x, q.y, q.z), velocity)
            self.odom_buf.append(entry)
            self.last_odom_stamp = stamp
            self._try_start_locked(stamp)
            if self.active is not None and stamp > self.active["start_stamp"] + 1e-6:
                self.active["execution_odom"].append(entry)
                self._process_pending_clouds_locked()
                if has_reached_goal(
                    entry[1], entry[3], self.active["goal"], self.goal_tolerance, self.goal_speed_tolerance
                ):
                    stop_requested = self._finish_locked(TERMINATION_SUCCESS)
                elif stamp - self.active["start_stamp"] >= self.timeout:
                    stop_requested = self._finish_locked(TERMINATION_TIMEOUT)
        if stop_requested:
            rospy.sleep(0.1)
            reason = "episode save failed" if self.fatal_error else "reached max_episodes"
            rospy.signal_shutdown(reason)

    def _command_cb(self, msg):
        """Buffer each analytic command frame without changing its timestamp or values."""
        stamp = msg.header.stamp.to_sec()
        position = msg.position
        velocity = msg.velocity
        acceleration = msg.acceleration
        jerk = msg.jerk
        entry = (
            stamp,
            np.asarray([position.x, position.y, position.z], dtype=np.float64),
            np.asarray([velocity.x, velocity.y, velocity.z], dtype=np.float64),
            np.asarray([acceleration.x, acceleration.y, acceleration.z], dtype=np.float64),
            np.asarray([jerk.x, jerk.y, jerk.z], dtype=np.float64),
        )
        with self.lock:
            if self.last_command_stamp is not None and stamp <= self.last_command_stamp + 1e-9:
                self._count_drop_locked("nonmonotonic_command")
                return
            self.command_buf.append(entry)
            self.last_command_stamp = stamp
            self._try_start_locked(stamp)
            if self.active is not None and stamp >= self.active["command_start_stamp"] - 1e-9:
                frames = self.active["command_frames"]
                if not frames or stamp > frames[-1][0] + 1e-9:
                    frames.append(entry)
                self._process_pending_clouds_locked()

    def _try_start_locked(self, current_stamp: float):
        """Start from the first synchronized command after the trajectory commit guard."""
        if self.pending_goal is None or self.active is not None:
            return
        goal_stamp = self.pending_goal["goal_stamp"]
        goal = self.pending_goal["goal"]
        command_not_before = self.pending_goal["command_not_before"]
        pair = None
        if command_not_before is not None:
            pair = first_synchronized_command(self.command_buf, self.odom_buf, command_not_before, self.max_odom_dt)
        command, odom = (None, None) if pair is None else pair
        if odom is None or command is None:
            if current_stamp - goal_stamp > self.timeout:
                if command_not_before is None:
                    reason = "missing_start_trajectory"
                elif not any(entry[0] >= command_not_before for entry in self.command_buf):
                    reason = "missing_start_command"
                else:
                    reason = "missing_start_odom"
                self._count_drop_locked(reason)
                self.pending_goal = None
                self._publish_goal_result_locked(goal, success=False)
            return
        self.active = {
            "goal": goal.astype(np.float32),
            "goal_stamp": np.float64(goal_stamp),
            "start_stamp": float(odom[0]),
            "command_start_stamp": float(command[0]),
            "last_candidate_stamp": -np.inf,
            "observations": [],
            "observation_stamps": [],
            "execution_odom": [odom],
            "command_frames": sorted(
                (entry for entry in self.command_buf if entry[0] >= command[0]), key=lambda entry: entry[0]
            ),
        }
        self.pending_goal = None
        rospy.loginfo(f"[episode] start goal ({goal[0]:.1f}, {goal[1]:.1f})")

    def _process_pending_clouds_locked(self):
        """Pair each cloud with the nearest real odom and command frames."""
        if self.active is None:
            self.pending_clouds.clear()
            return
        remaining = deque(maxlen=PENDING_CLOUD_DEPTH)
        for stamp, points_world in self.pending_clouds:
            # Only pair against odometry rows that belong to this Episode. A pre-start
            # buffer row may be nearer to the first cloud but cannot be persisted.
            odom, odom_resolved = resolve_nearest_frame(self.active["execution_odom"], stamp, self.max_odom_dt)
            if not odom_resolved:
                remaining.append((stamp, points_world))
                continue
            if odom is None:
                self._count_drop_locked("missing_observation_odom")
                continue
            command, command_resolved = resolve_nearest_frame(
                self.active["command_frames"], odom[0], self.max_command_dt
            )
            if not command_resolved:
                remaining.append((stamp, points_world))
                continue
            if command is None:
                self._count_drop_locked("missing_observation_command")
                continue
            if not len(points_world):
                self._count_drop_locked("empty_cloud")
                continue
            if self.active["observation_stamps"] and stamp <= self.active["observation_stamps"][-1] + 1e-9:
                self._count_drop_locked("nonmonotonic_cloud")
                continue
            R_wb = quaternion_matrix_numpy(odom[2])
            observation = self.renderer((points_world - odom[1]) @ R_wb).numpy()
            self.active["observations"].append(observation)
            self.active["observation_stamps"].append(float(stamp))
        self.pending_clouds = remaining

    def _publish_goal_result_locked(self, goal, success: bool):
        """Publish one latched world-frame result for the goal publisher."""
        event = PoseStamped()
        event.header.stamp = rospy.Time.now()
        event.header.frame_id = "world"
        event.pose.position.x = goal[0]
        event.pose.position.y = goal[1]
        event.pose.position.z = goal[2]
        event.pose.orientation.w = 1.0
        (self.success_pub if success else self.failure_pub).publish(event)

    def _execution_arrays_locked(self) -> dict:
        """Pair executed odometry with command acceleration for the saved Episode."""
        matched = []
        for odom in self.active["execution_odom"]:
            command = nearest_frame(self.active["command_frames"], odom[0], self.max_command_dt, wait_for_target=False)
            if command is None:
                self._count_drop_locked("missing_execution_command")
                continue
            matched.append((odom, command))
        if not matched:
            return {}
        return {
            "odom_stamps": np.asarray([float(odom[0]) for odom, _ in matched], dtype=np.float64),
            "odom_positions": np.asarray([odom[1] for odom, _ in matched], dtype=np.float32),
            "odom_velocities": np.asarray([odom[3] for odom, _ in matched], dtype=np.float32),
            "odom_orientations_wxyz": np.asarray([odom[2] for odom, _ in matched], dtype=np.float32),
            "acceleration_stamps": np.asarray([float(cmd[0]) for _, cmd in matched], dtype=np.float64),
            "accelerations": np.asarray([cmd[3] for _, cmd in matched], dtype=np.float32),
        }

    def _finish_locked(self, termination_code: int) -> bool:
        """Close the active episode and return whether the successful-episode limit was reached."""
        if self.active is None:
            return False
        execution = self._execution_arrays_locked()
        if not execution:
            self._count_drop_locked("empty_matched_execution")
            completed_goal = np.asarray(self.active["goal"], dtype=np.float64)
            self.active = None
            self.pending_clouds.clear()
            if termination_code not in (TERMINATION_GOAL_REPLACED, TERMINATION_SHUTDOWN):
                self._publish_goal_result_locked(completed_goal, success=False)
            return False

        odom_stamps = execution["odom_stamps"]
        odom_positions = execution["odom_positions"]
        arc_length = float(np.linalg.norm(np.diff(odom_positions, axis=0), axis=1).sum())
        if termination_code == TERMINATION_SUCCESS and (
            arc_length < self.min_arc_length or not self.active["observations"]
        ):
            termination_code = TERMINATION_INSUFFICIENT_MOTION
        observation_stamps = np.asarray(self.active["observation_stamps"], dtype=np.float64)
        lidar_rate, largest_lidar_gap = frame_stream_metrics(observation_stamps)
        if termination_code == TERMINATION_SUCCESS and (
            lidar_rate < self.min_lidar_rate or largest_lidar_gap > self.max_lidar_gap
        ):
            termination_code = TERMINATION_INSUFFICIENT_LIDAR_RATE
            self._count_drop_locked("insufficient_lidar_rate")
            rospy.logwarn(
                f"[episode] rejecting nominal success: LiDAR={lidar_rate:.2f} Hz, "
                f"largest gap={largest_lidar_gap:.3f}s; require >= "
                f"{self.min_lidar_rate:.2f} Hz and <= {self.max_lidar_gap:.3f}s"
            )

        wall_stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        label = f"-{self.file_tag}" if self.file_tag else ""
        sequence = self.next_sequence
        base_name = f"episode-{sequence:06d}-{wall_stamp}{label}.npz"
        episode_path = os.path.join(self.out_dir, base_name)
        episode_id = os.path.splitext(os.path.basename(episode_path))[0]
        obs_shape = (0, 1, self.renderer.v_fov, self.renderer.n_horizon)
        payload = {
            "episode_id": np.asarray(episode_id),
            "map_id": np.asarray(self.map_id),
            "start_stamp": np.asarray(odom_stamps[0], dtype=np.float64),
            "end_stamp": np.asarray(odom_stamps[-1], dtype=np.float64),
            "goal_stamp": np.asarray(self.active["goal_stamp"], dtype=np.float64),
            "goal_world": np.asarray(self.active["goal"], dtype=np.float32),
            "success": np.asarray(termination_code == TERMINATION_SUCCESS, dtype=np.uint8),
            "lidar_geometry": self.lidar_geometry,
            "lidar_range_images": (
                np.stack(self.active["observations"])
                if self.active["observations"]
                else np.empty(obs_shape, dtype=np.float32)
            ),
            "lidar_stamps": observation_stamps,
            **execution,
        }
        try:
            save_episode_env(episode_path, payload)
        except Exception as exc:
            self._count_drop_locked("episode_save_failed")
            self.saved = True
            self.fatal_error = True
            self.active = None
            self.pending_clouds.clear()
            rospy.logfatal(f"[episode] cannot save {episode_path}: {exc}")
            return True

        is_success = bool(payload["success"])
        completed_goal = np.asarray(self.active["goal"], dtype=np.float64)
        self.episode_count += 1
        self.next_sequence += 1
        self.success_count += int(is_success)
        self.active = None
        self.pending_clouds.clear()
        rospy.loginfo(
            f"[episode] saved {episode_id}: {TERMINATION_NAMES[termination_code]} "
            f"({self.success_count} successful / {self.episode_count} total), "
            f"LiDAR={lidar_rate:.2f} Hz, max_gap={largest_lidar_gap:.3f}s"
        )
        if is_success:
            self._publish_goal_result_locked(completed_goal, success=True)
        elif termination_code not in (TERMINATION_GOAL_REPLACED, TERMINATION_SHUTDOWN):
            self._publish_goal_result_locked(completed_goal, success=False)
        return bool(self.max_episodes and self.success_count >= self.max_episodes)

    def _save(self):
        with self.lock:
            if self.saved:
                return
            self.saved = True
            if self.active is not None:
                self._finish_locked(TERMINATION_SHUTDOWN)
            drop_counts = dict(self.drop_counts)
            episode_count = self.episode_count
            success_count = self.success_count
        if not episode_count:
            rospy.logwarn("[episode] no episodes collected; nothing saved")
        else:
            rospy.loginfo(
                f"[episode] collection complete: {success_count} successful / {episode_count} total -> {self.out_dir}"
            )
        if drop_counts:
            rospy.loginfo(f"[episode] drops: {drop_counts}")


def main():
    """Parse arguments and collect complete multi-rate goal episodes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", required=True, help="map cloud used by the simulator")
    parser.add_argument("--map-id", default=None, help="stable map identifier (default: map filename stem)")
    parser.add_argument("--out", default=None, help="directory for independent episode-*.npz files")
    parser.add_argument("--tag", default=None, help="optional label in generated episode IDs")
    parser.add_argument("--max-episodes", type=int, default=0, help="stop after N successful episodes")
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_TOLERANCE_M, help="3D success distance in metres")
    parser.add_argument(
        "--goal-speed-tolerance",
        type=float,
        default=GOAL_SPEED_TOLERANCE_MPS,
        help="maximum 3D speed at success in metres per second",
    )
    parser.add_argument("--goal-change-tolerance", type=float, default=1e-3, help="duplicate-goal tolerance")
    parser.add_argument("--timeout", type=float, default=30.0, help="episode timeout in seconds")
    parser.add_argument("--min-arc-length", type=float, default=0.1, help="minimum successful path length")
    parser.add_argument(
        "--max-odom-dt",
        type=float,
        default=MAX_FRAME_DELTA,
        help="maximum LiDAR-to-nearest-odom time difference in seconds",
    )
    parser.add_argument(
        "--max-command-dt",
        type=float,
        default=MAX_FRAME_DELTA,
        help="maximum odom-to-nearest-pos_cmd time difference in seconds",
    )
    parser.add_argument(
        "--min-lidar-rate",
        type=float,
        default=MIN_LIDAR_RATE_HZ,
        help="minimum effective LiDAR rate required for a successful Episode",
    )
    parser.add_argument(
        "--max-lidar-gap",
        type=float,
        default=MAX_LIDAR_GAP_S,
        help="largest allowed gap between retained LiDAR frames in a successful Episode",
    )
    args = parser.parse_args()

    rospy.init_node("nmp_episode_collect", anonymous=True)
    if not os.path.exists(args.map):
        raise SystemExit(f"map does not exist: {args.map}")
    file_tag = args.tag or ""
    out_dir = os.path.abspath(args.out or os.path.join(_ROOT, "data"))
    map_id = args.map_id or os.path.splitext(os.path.basename(args.map))[0]
    rospy.loginfo(f"[episode] map={map_id} ({args.map}) -> {out_dir}")
    EpisodeCollectNode(
        default_config(),
        out_dir,
        map_id,
        file_tag,
        args.max_episodes,
        args.goal_tolerance,
        args.goal_speed_tolerance,
        args.goal_change_tolerance,
        args.timeout,
        args.min_arc_length,
        args.max_odom_dt,
        args.max_command_dt,
        args.min_lidar_rate,
        args.max_lidar_gap,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
