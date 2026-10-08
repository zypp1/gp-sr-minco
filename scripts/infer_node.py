#!/usr/bin/env python3
"""Thin ROS/Noetic wrapper for current GP-SR inference and execution.

One registered LiDAR cloud is paired with its nearest real odometry row and
decoded through the same final-z3 path as offline inference. Command publication
is opt-in with ``--execute``.
"""

import argparse
import threading
import time
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

from neural_minco.geometry import world_to_body_numpy
from neural_minco.lidar import RangeImageRenderer
from neural_minco.runtime import (
    COMMAND_RATE_HZ,
    MAX_PLANNING_LATENCY_S,
    MAX_SENSOR_DELTA_S,
    TERMINAL_ENGAGE_DISTANCE_M,
    CurrentScanClearance,
    InferenceWorker,
    InferenceWorkerError,
    OdometrySample,
    PlanExecutor,
    select_nearest_odometry,
)
from utils.viz import Visualizer

CLOUD_TOPIC = "/cloud_registered"
ODOM_TOPIC = "/lidar_slam/odom"
GOAL_TOPIC = "/planning/click_goal"
COMMAND_TOPIC = "/planning/pos_cmd"
CALLBACK_BUDGET_MS = 100.0


def _horizontal_goal_distance(goal_body):
    goal = np.asarray(goal_body, dtype=np.float64)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("terminal goal distance requires finite XYZ")
    return float(np.linalg.norm(goal[:2]))


class InferNode:
    """Adapt ROS messages to the ROS-free current-model runtime."""

    def __init__(
        self,
        checkpoint_path,
        *,
        device="cuda",
        viz=True,
        lidar_viz=True,
        execute=False,
        command_rate_hz=COMMAND_RATE_HZ,
    ):
        self.command_rate_hz = float(command_rate_hz)
        if not np.isfinite(self.command_rate_hz) or self.command_rate_hz <= 0.0:
            raise ValueError("command rate must be finite and positive")
        self.worker = InferenceWorker(checkpoint_path, device)
        try:
            self._initialize_from_worker(checkpoint_path, viz=viz, lidar_viz=lidar_viz, execute=execute)
        except BaseException:
            self.close()
            raise

    def _initialize_from_worker(self, checkpoint_path, *, viz, lidar_viz, execute):
        self.config = self.worker.config
        self.device = self.worker.device
        self.execute = bool(execute)
        lidar = self.config["lidar"]
        if int(lidar.get("n_frames", -1)) != 1:
            raise ValueError("current GP-SR runtime requires one LiDAR frame")
        self.renderer = RangeImageRenderer(
            r_max=lidar["r_max"],
            n_horizon=lidar["n_horizon"],
            n_vertical=lidar["n_vertical"],
            horizon_fov_deg=lidar["horizon_fov_deg"],
            vertical_fov_min_deg=lidar["vertical_fov_min_deg"],
            vertical_fov_max_deg=lidar["vertical_fov_max_deg"],
        )
        self.executor = PlanExecutor(command_rate_hz=self.command_rate_hz)
        self.lock = threading.RLock()
        self.odom_buffer = deque(maxlen=256)
        self.goal_world = None
        self.goal_generation = 0
        self.previous_velocity_world = None
        self.previous_odom_stamp = None
        self.command_acceleration_world = np.zeros(3, dtype=np.float64)
        self.viz = (
            Visualizer(
                float(lidar["r_max"]),
                np.radians(float(lidar["vertical_fov_min_deg"])),
                np.radians(float(lidar["vertical_fov_max_deg"])),
            )
            if viz
            else None
        )
        self.lidar_viz = bool(lidar_viz and self.viz is not None)
        self.command_pub = rospy.Publisher(COMMAND_TOPIC, PositionCommand, queue_size=10) if self.execute else None
        self._warmup_clearance()
        rospy.Subscriber(CLOUD_TOPIC, PointCloud2, self._cloud_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=50)
        rospy.Subscriber(GOAL_TOPIC, PoseStamped, self._goal_cb, queue_size=5)
        self.command_timer = (
            rospy.Timer(rospy.Duration(1.0 / self.command_rate_hz), self._command_timer_cb) if self.execute else None
        )
        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "[gp_sr] ready checkpoint=%s device=%s warmup=%.1fms execute=%s rate=%.1fHz sync<=%.1fms"
            % (
                checkpoint_path,
                self.device,
                self.worker.warmup_ms,
                self.execute,
                self.command_rate_hz,
                1e3 * MAX_SENSOR_DELTA_S,
            )
        )

    def _warmup_clearance(self):
        if self.execute:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.zeros((1, 3)), balanced_tree=False, compact_nodes=False)
            tree.query(np.zeros((1, 3)), k=1)

    def _run_inference(self, observation, state_body, state_stamp):
        now = float(rospy.Time.now().to_sec())
        timeout_s = float(state_stamp) + MAX_PLANNING_LATENCY_S - now
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise InferenceWorkerError("planning deadline expired before inference")
        worker = self.worker
        if worker is None:
            raise InferenceWorkerError("inference worker is unavailable")
        return worker.predict(observation[None], state_body[None], timeout_s=timeout_s)

    def close(self):
        worker = getattr(self, "worker", None)
        if worker is not None:
            self.worker = None
            worker.close()

    def _odom_cb(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        v = message.twist.twist.linear
        stamp = float(message.header.stamp.to_sec())
        position = np.asarray([p.x, p.y, p.z], dtype=np.float64)
        quaternion = np.asarray([q.w, q.x, q.y, q.z], dtype=np.float64)
        velocity = np.asarray([v.x, v.y, v.z], dtype=np.float64)
        with self.lock:
            if self.odom_buffer and stamp <= self.odom_buffer[-1].stamp + 1e-9:
                rospy.logwarn_throttle(1.0, "[gp_sr] dropping nonmonotonic odometry")
                return
            if self.execute:
                acceleration = self.command_acceleration_world.copy()
            elif self.previous_velocity_world is not None and stamp > self.previous_odom_stamp + 1e-9:
                acceleration = (velocity - self.previous_velocity_world) / (stamp - self.previous_odom_stamp)
            else:
                acceleration = np.zeros(3, dtype=np.float64)
            try:
                sample = OdometrySample(stamp, position, quaternion, velocity, acceleration)
            except ValueError as error:
                rospy.logwarn_throttle(1.0, "[gp_sr] invalid odometry: %s" % error)
                return
            self.odom_buffer.append(sample)
            self.previous_velocity_world = velocity.copy()
            self.previous_odom_stamp = stamp

    def _goal_cb(self, message):
        goal = np.asarray([message.pose.position.x, message.pose.position.y, message.pose.position.z], dtype=np.float64)
        if not np.isfinite(goal).all():
            rospy.logwarn_throttle(1.0, "[gp_sr] rejecting non-finite goal")
            return
        with self.lock:
            changed = self.goal_world is None or np.linalg.norm(goal - self.goal_world) > 1e-3
            self.goal_world = goal
            if changed:
                self.goal_generation += 1
        if changed:
            self.executor.clear_terminal_latch()

    def _goal_generation_is_current(self, generation):
        with self.lock:
            return generation == self.goal_generation

    def _current_clearance(self, observation, odometry, scan_stamp, points_world):
        lidar = self.config["lidar"]
        return CurrentScanClearance(
            observation,
            odometry.position_world,
            odometry.quaternion_wxyz,
            scan_stamp,
            points_world,
            r_max=lidar["r_max"],
            n_horizon=lidar["n_horizon"],
            n_vertical=lidar["n_vertical"],
            vertical_fov_min_deg=lidar["vertical_fov_min_deg"],
            vertical_fov_max_deg=lidar["vertical_fov_max_deg"],
        )

    def _publish_prediction_viz(self, prediction, goal_body, odometry):
        if self.viz is None:
            return
        self.viz.publish_traj(
            prediction.position[0].detach().cpu().numpy(),
            prediction.waypoints[0].detach().cpu().numpy(),
            prediction.endpoint[0].detach().cpu().numpy(),
            goal_body,
            odometry.position_world,
            odometry.quaternion_wxyz,
            rospy.Time.from_sec(odometry.stamp),
        )

    def _cloud_cb(self, message):
        started = time.perf_counter()
        scan_stamp = float(message.header.stamp.to_sec())
        if not np.isfinite(scan_stamp):
            rospy.logwarn_throttle(1.0, "[gp_sr] rejecting non-finite cloud timestamp")
            return
        points_world = np.asarray(
            list(point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64
        ).reshape(-1, 3)
        if not len(points_world) or not np.isfinite(points_world).all():
            rospy.logwarn_throttle(1.0, "[gp_sr] rejecting empty/non-finite registered cloud")
            return
        with self.lock:
            odometry = select_nearest_odometry(tuple(self.odom_buffer), scan_stamp)
            goal = None if self.goal_world is None else self.goal_world.copy()
            goal_generation = self.goal_generation
        if odometry is None or goal is None:
            rospy.logwarn_throttle(
                1.0, "[gp_sr] waiting for goal and odometry within %.1fms" % (1e3 * MAX_SENSOR_DELTA_S)
            )
            return
        now = float(rospy.Time.now().to_sec())
        if now < odometry.stamp - 1e-6 or now - odometry.stamp > MAX_PLANNING_LATENCY_S:
            rospy.logwarn_throttle(1.0, "[gp_sr] rejecting stale/future synchronized state")
            return

        points_body = world_to_body_numpy(points_world - odometry.position_world[None], odometry.quaternion_wxyz)
        observation = self.renderer(points_body).numpy()
        clearance = self._current_clearance(observation, odometry, scan_stamp, points_world)
        if self.lidar_viz:
            hits, ranges = self.renderer.backproject(observation)
            self.viz.publish_lidar(
                hits, ranges, odometry.position_world, odometry.quaternion_wxyz, rospy.Time.from_sec(odometry.stamp)
            )
        velocity_body = world_to_body_numpy(odometry.velocity_world, odometry.quaternion_wxyz)
        acceleration_body = world_to_body_numpy(odometry.acceleration_world, odometry.quaternion_wxyz)
        goal_body = world_to_body_numpy(goal - odometry.position_world, odometry.quaternion_wxyz)
        state_body = np.concatenate([velocity_body, acceleration_body, goal_body]).astype(np.float32)

        terminal_goal = self.executor.terminal_goal_world
        if self.execute and terminal_goal is not None and np.linalg.norm(terminal_goal - goal) <= 1e-3:
            decision = self.executor.refresh_clearance(clearance, float(rospy.Time.now().to_sec()))
            rospy.loginfo_throttle(2.0, "[gp_sr] terminal refresh: %s" % decision.reason)
            return
        if self.execute and _horizontal_goal_distance(goal_body) <= TERMINAL_ENGAGE_DISTANCE_M:
            decision = self.executor.consider_terminal(
                velocity_body,
                acceleration_body,
                goal_body,
                goal,
                odometry.position_world,
                odometry.quaternion_wxyz,
                odometry.stamp,
                float(rospy.Time.now().to_sec()),
                clearance,
                install_guard=lambda: self._goal_generation_is_current(goal_generation),
                install_lock=self.lock,
            )
            if decision.reason == "install_guard_rejected":
                return
            if decision.accepted:
                rospy.loginfo("[gp_sr] installed terminal plan")
                return

        try:
            prediction = self._run_inference(observation, state_body, odometry.stamp)
        except (InferenceWorkerError, RuntimeError, ValueError, FloatingPointError) as error:
            rospy.logerr_throttle(1.0, "[gp_sr] inference rejected: %s" % error)
            if self.execute:
                self.executor.refresh_clearance(clearance, float(rospy.Time.now().to_sec()))
            return
        if self.execute:
            decision = self.executor.consider_prediction(
                prediction,
                odometry.position_world,
                odometry.quaternion_wxyz,
                odometry.stamp,
                float(rospy.Time.now().to_sec()),
                clearance,
                install_guard=lambda: self._goal_generation_is_current(goal_generation),
                install_lock=self.lock,
            )
            if decision.reason == "install_guard_rejected":
                return
            if not decision.accepted:
                self.executor.refresh_clearance(clearance, float(rospy.Time.now().to_sec()))
        else:
            decision = None
        self._publish_prediction_viz(prediction, goal_body, odometry)
        elapsed_ms = 1e3 * (time.perf_counter() - started)
        outcome = "visualized" if decision is None else decision.reason
        if elapsed_ms > CALLBACK_BUDGET_MS:
            rospy.logwarn_throttle(1.0, "[gp_sr] %.1fms OVER_BUDGET outcome=%s" % (elapsed_ms, outcome))
        else:
            rospy.loginfo_throttle(2.0, "[gp_sr] %.1fms outcome=%s" % (elapsed_ms, outcome))

    @staticmethod
    def _set_vector(target, value):
        target.x = float(value[0])
        target.y = float(value[1])
        target.z = float(value[2])

    def _command_timer_cb(self, _event):
        with self.lock:
            odometry = None if not self.odom_buffer else self.odom_buffer[-1]
        if odometry is None:
            return
        try:
            command = self.executor.sample_command(
                float(rospy.Time.now().to_sec()), odometry.position_world, odometry.quaternion_wxyz
            )
        except (ValueError, FloatingPointError) as error:
            rospy.logfatal("[gp_sr] refusing invalid command: %s" % error)
            rospy.signal_shutdown("invalid GP-SR command")
            return
        with self.lock:
            self.command_acceleration_world = command.acceleration.copy()
        message = PositionCommand()
        message.header.stamp = rospy.Time.from_sec(command.stamp)
        message.header.frame_id = "world"
        message.trajectory_id = command.trajectory_id
        message.trajectory_flag = message.TRAJECTORY_STATUS_READY
        self._set_vector(message.position, command.position)
        self._set_vector(message.velocity, command.velocity)
        self._set_vector(message.acceleration, command.acceleration)
        self._set_vector(message.jerk, command.jerk)
        message.yaw = command.yaw
        message.yaw_dot = command.yaw_rate
        message.vel_norm = float(np.linalg.norm(command.velocity))
        message.acc_norm = float(np.linalg.norm(command.acceleration))
        self.command_pub.publish(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="current GP-SR checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true", help="publish PerfectDrone commands")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-lidar-viz", action="store_true")
    parser.add_argument("--command-rate", type=float, default=COMMAND_RATE_HZ)
    arguments, _ = parser.parse_known_args()
    rospy.init_node("gp_sr_infer", anonymous=True)
    try:
        node = InferNode(
            arguments.checkpoint,
            device=arguments.device,
            viz=not arguments.no_viz,
            lidar_viz=not arguments.no_lidar_viz,
            execute=arguments.execute,
            command_rate_hz=arguments.command_rate,
        )
    except Exception as error:
        rospy.logfatal("[gp_sr] startup failed: %s" % error)
        rospy.signal_shutdown("GP-SR startup failed")
        return
    try:
        rospy.spin()
    finally:
        node.close()


if __name__ == "__main__":
    main()
