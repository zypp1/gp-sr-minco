"""Functional clean-checkpoint contract test for the ROS adapter."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from neural_minco.runtime import InferenceWorkerError, OdometrySample


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_adapter(monkeypatch, now=10.025):
    shutdown_callbacks = []

    class Stamp:
        def __init__(self, value):
            self.value = float(value)

        def to_sec(self):
            return self.value

    class Time:
        @staticmethod
        def now():
            return Stamp(now)

        @staticmethod
        def from_sec(value):
            return Stamp(value)

    rospy = _module(
        "rospy",
        Subscriber=lambda *args, **kwargs: (args, kwargs),
        Publisher=lambda *args, **kwargs: (args, kwargs),
        Duration=lambda value: value,
        Timer=lambda *args, **kwargs: (args, kwargs),
        Time=Time,
        on_shutdown=shutdown_callbacks.append,
        shutdown_callbacks=shutdown_callbacks,
        loginfo=lambda *args, **kwargs: None,
        loginfo_throttle=lambda *args, **kwargs: None,
        logwarn_throttle=lambda *args, **kwargs: None,
        logerr_throttle=lambda *args, **kwargs: None,
        logfatal=lambda *args, **kwargs: None,
        signal_shutdown=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    message_packages = {
        "geometry_msgs.msg": "PoseStamped",
        "nav_msgs.msg": "Odometry",
        "quadrotor_msgs.msg": "PositionCommand",
        "sensor_msgs.msg": "PointCloud2",
    }
    for qualified_name, class_name in message_packages.items():
        package_name = qualified_name.split(".")[0]
        package = _module(package_name)
        messages = _module(qualified_name, **{class_name: type(class_name, (), {})})
        package.msg = messages
        monkeypatch.setitem(sys.modules, package_name, package)
        monkeypatch.setitem(sys.modules, qualified_name, messages)

    sensor_msgs = sys.modules["sensor_msgs"]
    sensor_msgs.point_cloud2 = _module("sensor_msgs.point_cloud2", read_points=lambda *args, **kwargs: ())
    monkeypatch.setitem(sys.modules, "sensor_msgs.point_cloud2", sensor_msgs.point_cloud2)
    utilities = _module("utils")
    visualization = _module("utils.viz", Visualizer=type("Visualizer", (), {}))
    utilities.viz = visualization
    monkeypatch.setitem(sys.modules, "utils", utilities)
    monkeypatch.setitem(sys.modules, "utils.viz", visualization)

    path = Path(__file__).parents[1] / "scripts" / "infer_node.py"
    specification = importlib.util.spec_from_file_location("current_infer_node_test", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _lidar_config():
    return {
        "lidar": {
            "r_max": 17.0,
            "n_horizon": 180,
            "n_vertical": 90,
            "horizon_fov_deg": 360.0,
            "vertical_fov_min_deg": -7.0,
            "vertical_fov_max_deg": 52.0,
            "n_frames": 1,
        }
    }


class _FakeWorker:
    def __init__(self, config, prediction=None, error=None):
        self.config = config
        self.device = "cpu"
        self.warmup_ms = 1.0
        self.prediction = prediction
        self.error = error
        self.calls = []
        self.closed = False

    def predict(self, observation, state_body, *, timeout_s):
        self.calls.append((observation.copy(), state_body.copy(), timeout_s))
        if self.error is not None:
            raise self.error
        return self.prediction

    def close(self):
        self.closed = True


def test_infer_node_reads_lidar_from_clean_checkpoint_config(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    config = _lidar_config()
    worker = _FakeWorker(config)
    calls = []

    def inference_worker(path, device):
        calls.append((path, device))
        assert path == "clean.pth"
        assert device == "cpu"
        return worker

    monkeypatch.setattr(adapter, "InferenceWorker", inference_worker, raising=False)
    node = adapter.InferNode("clean.pth", device="cpu", viz=False, lidar_viz=False, execute=False)

    assert calls == [("clean.pth", "cpu")]
    assert node.config == config
    assert not hasattr(node, "model")
    assert node.renderer.r_max == 17.0
    assert node.renderer.n_horizon == 180
    assert node.renderer.n_vertical == 90
    assert node.renderer.v_fov == 30

    observation = node.renderer(np.empty((0, 3))).numpy()
    odometry = OdometrySample(1.0, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    clearance = node._current_clearance(observation, odometry, 1.0, np.empty((0, 3)))
    assert clearance.r_max == 17.0
    assert clearance.n_horizon == 180
    assert clearance.n_vertical == 90
    assert adapter.rospy.shutdown_callbacks == [node.close]
    node.close()
    assert worker.closed


def test_infer_node_delegates_with_remaining_state_deadline_and_closes(monkeypatch):
    adapter = _load_adapter(monkeypatch, now=10.025)
    worker = _FakeWorker(_lidar_config(), prediction=object())
    monkeypatch.setattr(adapter, "InferenceWorker", lambda path, device: worker, raising=False)
    node = adapter.InferNode("clean.pth", device="cpu", viz=False, lidar_viz=False, execute=False)

    prediction = node._run_inference(np.ones((1, 30, 180), dtype=np.float32), np.zeros(9, dtype=np.float32), 10.0)

    assert prediction is worker.prediction
    observation, state_body, timeout_s = worker.calls[0]
    assert observation.shape == (1, 1, 30, 180)
    assert state_body.shape == (1, 9)
    assert timeout_s == pytest.approx(0.075)
    call_count = len(worker.calls)
    with pytest.raises(adapter.InferenceWorkerError, match="deadline expired"):
        node._run_inference(np.ones((1, 30, 180), dtype=np.float32), np.zeros(9, dtype=np.float32), 9.8)
    assert len(worker.calls) == call_count

    node.close()
    node.close()
    assert worker.closed


def test_inference_failure_refreshes_existing_execution_prefix(monkeypatch):
    adapter = _load_adapter(monkeypatch, now=10.025)
    worker = _FakeWorker(_lidar_config(), error=InferenceWorkerError("request timed out"))
    monkeypatch.setattr(adapter, "InferenceWorker", lambda path, device: worker, raising=False)
    monkeypatch.setattr(adapter.point_cloud2, "read_points", lambda *args, **kwargs: [(2.0, 0.0, 0.0)])
    node = adapter.InferNode("clean.pth", device="cpu", viz=False, lidar_viz=False, execute=True)
    odometry = OdometrySample(10.0, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    node.odom_buffer.append(odometry)
    node.goal_world = np.asarray([10.0, 0.0, 0.0])
    refreshes = []
    node.executor = types.SimpleNamespace(
        terminal_goal_world=None, refresh_clearance=lambda clearance, now: refreshes.append((clearance, now))
    )
    message = types.SimpleNamespace(header=types.SimpleNamespace(stamp=adapter.rospy.Time.from_sec(10.0)))

    node._cloud_cb(message)

    assert len(worker.calls) == 1
    assert len(refreshes) == 1
    assert refreshes[0][1] == pytest.approx(10.025)
    node.close()


def test_goal_generation_changes_only_for_a_new_goal(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    node = object.__new__(adapter.InferNode)
    node.lock = adapter.threading.RLock()
    node.goal_world = None
    node.goal_generation = 0
    cleared = []
    node.executor = types.SimpleNamespace(clear_terminal_latch=lambda: cleared.append(True))

    def message(x, y, z):
        position = types.SimpleNamespace(x=x, y=y, z=z)
        return types.SimpleNamespace(pose=types.SimpleNamespace(position=position))

    node._goal_cb(message(1.0, 2.0, 3.0))
    assert node.goal_generation == 1
    assert node._goal_generation_is_current(1)
    node._goal_cb(message(1.0, 2.0, 3.0))
    assert node.goal_generation == 1
    node._goal_cb(message(2.0, 2.0, 3.0))
    assert node.goal_generation == 2
    assert not node._goal_generation_is_current(1)
    assert len(cleared) == 2
