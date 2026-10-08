"""Focused tests for the current ROS-free online runtime."""

import time

import numpy as np
import pytest
import torch

import neural_minco.runtime as runtime
from neural_minco.config import default_config
from neural_minco.lidar import RangeImageRenderer
from neural_minco.model import build_model

_PREDICTION_FIELDS = (
    "waypoints",
    "durations",
    "endpoint",
    "terminal_velocity",
    "terminal_acceleration",
    "coefficients",
    "position",
    "velocity",
    "acceleration",
    "jerk",
)


class _DeterministicPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.waypoint_scale = 10.0
        self.v_end_scale = 6.0
        self.acceleration_scale = 6.0
        self.horizon = 2.0

    def forward(self, observation, state_body, return_stages=False):
        del state_body, return_stages
        batch_size = len(observation)
        zeros = observation.new_zeros
        return {
            "Q": zeros(batch_size, 2, 3),
            "p_end": zeros(batch_size, 3),
            "v_end": zeros(batch_size, 3),
            "a_end": zeros(batch_size, 3),
            "ratio_logits": observation.new_tensor([[0.0, 1.0, -1.0]]).expand(batch_size, -1),
        }


def _linear_prediction(speed=1.0):
    durations = torch.full((1, 3), 2.0 / 3.0, dtype=torch.float64)
    coefficients = torch.zeros(1, 3, 6, 3, dtype=torch.float64)
    for segment in range(3):
        coefficients[0, segment, 0, 0] = segment * 2.0 / 3.0 * speed
        coefficients[0, segment, 1, 0] = speed
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    sampled = torch.zeros(1, 20, 3, dtype=torch.float64)
    return runtime.TrajectoryPrediction(
        waypoints=torch.zeros(1, 2, 3, dtype=torch.float64),
        durations=durations,
        endpoint=zeros,
        terminal_velocity=zeros,
        terminal_acceleration=zeros,
        coefficients=coefficients,
        position=sampled,
        velocity=sampled,
        acceleration=sampled,
        jerk=sampled,
    )


def _fake_inference_process(connection, checkpoint_path, device, samples):
    try:
        if checkpoint_path == "startup-error":
            connection.send(("error", None, "ValueError", "bad checkpoint"))
            return
        if checkpoint_path == "startup-timeout":
            time.sleep(10.0)
            return
        resolved_device = "cpu" if device == "auto" else device
        connection.send(("ready", default_config(), resolved_device, 1.25))
        while True:
            message = connection.recv()
            if message[0] == "stop":
                return
            _, request_id, observation, state_body = message
            if checkpoint_path == "timeout":
                continue
            if checkpoint_path == "exit":
                return
            if checkpoint_path == "runtime-error":
                connection.send(("error", request_id, "RuntimeError", "worker failed"))
                return
            assert observation.shape == (1, 1, 30, 180)
            assert state_body.shape == (1, 9)
            assert samples == 20
            payload = runtime.prediction_to_payload(_linear_prediction())
            if checkpoint_path == "mismatch":
                connection.send(("prediction", request_id + 1, payload))
            elif checkpoint_path == "invalid-payload":
                connection.send(("prediction", request_id, {}))
            else:
                connection.send(("prediction", request_id, payload))
    finally:
        connection.close()


def _free_clearance(stamp=10.0):
    return runtime.CurrentScanClearance(
        np.ones((1, 30, 180), dtype=np.float32), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], stamp, np.empty((0, 3))
    )


def test_prediction_payload_round_trip_preserves_every_tensor():
    prediction = _linear_prediction()
    restored = runtime.prediction_from_payload(runtime.prediction_to_payload(prediction))
    for field in _PREDICTION_FIELDS:
        torch.testing.assert_close(getattr(restored, field), getattr(prediction, field))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durations", np.ones((1, 2), dtype=np.float64)),
        ("durations", np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64)),
        ("durations", np.asarray([[1.0, -0.5, 1.5]], dtype=np.float64)),
        ("position", np.ones((1, 19, 3), dtype=np.float64)),
        ("coefficients", np.full((1, 3, 6, 3), np.nan, dtype=np.float64)),
    ],
)
def test_prediction_payload_rejects_wrong_shape_and_nonfinite_values(field, value):
    payload = runtime.prediction_to_payload(_linear_prediction())
    values = {name: getattr(payload, name) for name in _PREDICTION_FIELDS}
    values[field] = value
    with pytest.raises(ValueError):
        runtime.PredictionPayload(**values)


def test_inference_worker_spawn_round_trip_and_clean_shutdown(monkeypatch):
    monkeypatch.setattr(runtime, "_inference_worker_main", _fake_inference_process)
    worker = runtime.InferenceWorker("success", "cpu")
    try:
        assert worker.device == "cpu"
        assert worker.config["lidar"]["n_frames"] == 1
        assert worker.warmup_ms == pytest.approx(1.25)
        prediction = worker.predict(
            np.ones((1, 1, 30, 180), dtype=np.float32), np.zeros((1, 9), dtype=np.float32), timeout_s=1.0
        )
        assert prediction.coefficients.shape == (1, 3, 6, 3)
    finally:
        worker.close()
    worker.close()
    assert not worker.is_alive


def test_inference_worker_reports_startup_error(monkeypatch):
    monkeypatch.setattr(runtime, "_inference_worker_main", _fake_inference_process)
    with pytest.raises(runtime.InferenceWorkerError, match="ValueError: bad checkpoint"):
        runtime.InferenceWorker("startup-error", "cpu")


def test_inference_worker_bounds_startup_wait(monkeypatch):
    monkeypatch.setattr(runtime, "_inference_worker_main", _fake_inference_process)
    with pytest.raises(runtime.InferenceWorkerError, match="startup timed out"):
        runtime.InferenceWorker("startup-timeout", "cpu", startup_timeout_s=0.02)


def test_inference_worker_timeout_disables_future_requests(monkeypatch):
    monkeypatch.setattr(runtime, "_inference_worker_main", _fake_inference_process)
    worker = runtime.InferenceWorker("timeout", "cpu")
    observation = np.ones((1, 1, 30, 180), dtype=np.float32)
    state_body = np.zeros((1, 9), dtype=np.float32)
    try:
        with pytest.raises(runtime.InferenceWorkerError, match="timed out"):
            worker.predict(observation, state_body, timeout_s=0.02)
        with pytest.raises(runtime.InferenceWorkerError, match="unavailable"):
            worker.predict(observation, state_body, timeout_s=0.02)
    finally:
        worker.close()
    assert not worker.is_alive


@pytest.mark.parametrize(
    ("checkpoint_path", "message"),
    [
        ("mismatch", "mismatched inference response"),
        ("invalid-payload", "invalid inference prediction"),
        ("runtime-error", "RuntimeError: worker failed"),
        ("exit", "connection closed"),
    ],
)
def test_inference_worker_rejects_failed_or_malformed_response(monkeypatch, checkpoint_path, message):
    monkeypatch.setattr(runtime, "_inference_worker_main", _fake_inference_process)
    worker = runtime.InferenceWorker(checkpoint_path, "cpu")
    try:
        with pytest.raises(runtime.InferenceWorkerError, match=message):
            worker.predict(
                np.ones((1, 1, 30, 180), dtype=np.float32), np.zeros((1, 9), dtype=np.float32), timeout_s=1.0
            )
    finally:
        worker.close()
    assert not worker.is_alive


@pytest.mark.parametrize(
    ("observation", "state_body", "timeout_s", "message"),
    [
        (np.ones((1, 30, 180), dtype=np.float32), np.zeros((1, 9), dtype=np.float32), 1.0, "observation"),
        (np.ones((1, 1, 30, 180), dtype=np.float32), np.zeros(9, dtype=np.float32), 1.0, "physical state"),
        (np.ones((1, 1, 30, 180), dtype=np.float32), np.zeros((1, 9), dtype=np.float32), 0.0, "timeout"),
    ],
)
def test_inference_worker_validates_request_before_sending(monkeypatch, observation, state_body, timeout_s, message):
    monkeypatch.setattr(runtime, "_inference_worker_main", _fake_inference_process)
    worker = runtime.InferenceWorker("success", "cpu")
    try:
        with pytest.raises(ValueError, match=message):
            worker.predict(observation, state_body, timeout_s=timeout_s)
    finally:
        worker.close()


def test_predict_is_fixed_two_second_m3_minimum_jerk():
    planner = _DeterministicPlanner().eval()
    prediction = runtime.predict(
        planner, torch.ones(1, 1, 30, 180), torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]])
    )
    assert prediction.coefficients.shape == (1, 3, 6, 3)
    assert prediction.position.shape == (1, 20, 3)
    torch.testing.assert_close(prediction.durations.sum(dim=1), torch.tensor([2.0]))
    assert not torch.allclose(prediction.durations, torch.full((1, 3), 2.0 / 3.0))


def test_network_skips_minco_and_runtime_decodes_final_stage_once(monkeypatch):
    model = build_model().eval()
    calls = []
    original = runtime.MinimumJerkDecoder.forward

    def record(decoder, *args, **kwargs):
        calls.append(len(args[0]))
        return original(decoder, *args, **kwargs)

    monkeypatch.setattr(runtime.MinimumJerkDecoder, "forward", record)
    observation = torch.rand(1, 1, 30, 180)
    state_body = torch.tensor([[1.0, 0.1, 0.0, 0.1, 0.0, 0.0, 7.0, 1.0, 0.4]])
    with torch.inference_mode():
        raw = model(observation, state_body)
        assert calls == []
        runtime.decode_prediction(model, raw, state_body)
    assert calls == [1]


def test_nearest_odometry_enforces_five_millisecond_contract():
    def sample(stamp):
        return runtime.OdometrySample(stamp, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))

    rows = [sample(0.990), sample(1.004), sample(1.020)]
    assert runtime.select_nearest_odometry(rows, 1.0) is rows[1]
    assert runtime.select_nearest_odometry([rows[0]], 1.0) is None


def test_world_trajectory_uses_full_quaternion_and_true_state_stamp():
    coefficients = np.zeros((1, 6, 3), dtype=np.float64)
    coefficients[0, 1, 1] = 1.0
    roll_90 = np.asarray([2.0**-0.5, 2.0**-0.5, 0.0, 0.0])
    trajectory = runtime.WorldMincoTrajectory(coefficients, [2.0], [1.0, 2.0, 3.0], roll_90, 10.0)
    sample = trajectory.sample(10.5)
    np.testing.assert_allclose(sample.position, [1.0, 2.0, 3.5], atol=1e-12)
    np.testing.assert_allclose(sample.velocity, [0.0, 0.0, 1.0], atol=1e-12)


def test_current_scan_clearance_detects_points_behind_a_return():
    renderer = RangeImageRenderer()
    obstacle = np.asarray([[5.0, 0.0, 0.0]])
    observation = renderer(obstacle).numpy()
    clearance = runtime.CurrentScanClearance(observation, np.zeros(3), [1.0, 0.0, 0.0, 0.0], 1.0, obstacle)
    assert clearance.minimum([[4.0, 0.0, 0.0]]) == pytest.approx(1.0)
    assert clearance.minimum([[6.0, 0.0, 0.0]]) == pytest.approx(-1.0)


def test_executor_installs_prediction_then_commits_brake():
    executor = runtime.PlanExecutor(prefix_duration=0.30)
    prediction = _linear_prediction()
    decision = executor.consider_prediction(
        prediction, np.zeros(3), [1.0, 0.0, 0.0, 0.0], 10.0, 10.01, _free_clearance()
    )
    assert decision.accepted and decision.verified
    np.testing.assert_allclose(executor.active_plan.durations, prediction.durations[0].numpy())
    assert executor.active_plan.total_duration == pytest.approx(2.0)
    command = executor.sample_command(executor.verified_until, np.zeros(3), [1.0, 0.0, 0.0, 0.0])
    assert command.mode == "brake"
    values = np.concatenate([command.position, command.velocity, command.acceleration, command.jerk])
    assert np.isfinite(values).all()


def test_stale_prediction_does_not_replace_active_plan():
    executor = runtime.PlanExecutor()
    prediction = _linear_prediction()
    accepted = executor.consider_prediction(
        prediction, np.zeros(3), [1.0, 0.0, 0.0, 0.0], 10.0, 10.01, _free_clearance()
    )
    assert accepted.accepted
    active = executor.active_plan
    stale = executor.consider_prediction(prediction, np.ones(3), [1.0, 0.0, 0.0, 0.0], 10.0, 10.2, _free_clearance())
    assert not stale.accepted
    assert stale.reason == "stale_planning_state"
    assert executor.active_plan is active


def test_terminal_plan_stops_exactly_at_goal():
    terminal = runtime.plan_terminal_trajectory(np.zeros(3), np.zeros(3), np.asarray([1.0, -0.5, 0.25]))
    assert terminal is not None
    trajectory = runtime.WorldMincoTrajectory(
        terminal.coefficients_body, [terminal.duration], np.zeros(3), [1.0, 0.0, 0.0, 0.0], 0.0
    )
    end = trajectory.sample(terminal.duration)
    np.testing.assert_allclose(end.position, [1.0, -0.5, 0.25], atol=1e-9)
    np.testing.assert_allclose(end.velocity, 0.0, atol=1e-9)
    np.testing.assert_allclose(end.acceleration, 0.0, atol=1e-9)


def test_completed_terminal_plan_holds_goal_without_reinstallation():
    executor = runtime.PlanExecutor()
    goal = np.asarray([1.0, -0.5, 0.25])
    decision = executor.consider_terminal(
        np.zeros(3), np.zeros(3), goal, goal, np.zeros(3), [1.0, 0.0, 0.0, 0.0], 10.0, 10.01, _free_clearance()
    )
    assert decision.accepted
    trajectory_id = executor.sample_command(10.01, np.zeros(3), [1.0, 0.0, 0.0, 0.0]).trajectory_id
    finish_stamp = executor.active_plan.start_stamp + executor.active_plan.total_duration
    for refresh_stamp in np.arange(10.2, finish_stamp - 0.05, 0.2):
        assert executor.refresh_clearance(_free_clearance(refresh_stamp), refresh_stamp).accepted
    assert executor.refresh_clearance(_free_clearance(finish_stamp - 0.05), finish_stamp - 0.05).accepted

    finished = executor.sample_command(finish_stamp + 1e-9, goal, [1.0, 0.0, 0.0, 0.0])
    repeated = executor.sample_command(finish_stamp + 0.1, goal, [1.0, 0.0, 0.0, 0.0])
    refreshed = executor.refresh_clearance(_free_clearance(), finish_stamp + 0.1)

    assert finished.mode == repeated.mode == "terminal_hold"
    assert finished.trajectory_id == repeated.trajectory_id == trajectory_id
    np.testing.assert_allclose(finished.position, goal, atol=1e-9)
    np.testing.assert_allclose(finished.velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(executor.terminal_goal_world, goal, atol=1e-12)
    assert refreshed.accepted and refreshed.verified
    assert refreshed.reason == "terminal_hold"

    executor.clear_terminal_latch()
    assert executor.terminal_goal_world is None
    assert executor.sample_command(finish_stamp + 0.2, goal, [1.0, 0.0, 0.0, 0.0]).mode == "hold"


def _constant_acceleration_trajectory(initial_velocity, acceleration):
    initial_velocity = np.asarray(initial_velocity, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    durations = np.full(3, 2.0 / 3.0, dtype=np.float64)
    coefficients = np.zeros((3, 6, 3), dtype=np.float64)
    for segment in range(3):
        start = segment * 2.0 / 3.0
        coefficients[segment, 0] = initial_velocity * start + 0.5 * acceleration * start**2
        coefficients[segment, 1] = initial_velocity + acceleration * start
        coefficients[segment, 2] = 0.5 * acceleration
    return runtime.WorldMincoTrajectory(coefficients, durations, np.zeros(3), [1.0, 0.0, 0.0, 0.0], 10.0)


@pytest.mark.parametrize(
    ("initial_velocity", "acceleration"), [([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]), ([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0])]
)
def test_finished_nonstationary_boundary_requires_verified_brake(initial_velocity, acceleration):
    trajectory = _constant_acceleration_trajectory(initial_velocity, acceleration)
    result = runtime.inspect_trajectory_prefix(trajectory, _free_clearance(), 11.8)
    assert result.plan is not None
    assert result.plan.fallback is not None
    terminal = trajectory.sample(12.0)
    brake_start = result.plan.fallback.sample(12.0)
    np.testing.assert_allclose(brake_start.position, terminal.position, atol=1e-12)
    np.testing.assert_allclose(brake_start.velocity, terminal.velocity, atol=1e-12)
    np.testing.assert_allclose(brake_start.acceleration, terminal.acceleration, atol=1e-12)


def test_finished_stationary_boundary_can_hold_directly():
    trajectory = _constant_acceleration_trajectory(np.zeros(3), np.zeros(3))
    result = runtime.inspect_trajectory_prefix(trajectory, _free_clearance(), 11.8)
    assert result.reason == "accepted_finished_prefix"
    assert result.plan is not None
    assert result.plan.fallback is None


def test_executor_crosses_endpoint_into_brake_then_holds_continuously():
    executor = runtime.PlanExecutor(prefix_duration=0.30)
    decision = executor.consider_prediction(
        _linear_prediction(), np.zeros(3), [1.0, 0.0, 0.0, 0.0], 10.0, 10.01, _free_clearance()
    )
    assert decision.accepted
    refreshed = executor.refresh_clearance(_free_clearance(), 11.8)
    assert refreshed.accepted
    terminal = executor.active_plan.sample(12.0)

    brake_start = executor.sample_command(12.0, np.zeros(3), [1.0, 0.0, 0.0, 0.0])
    assert brake_start.mode == "brake"
    np.testing.assert_allclose(brake_start.position, terminal.position, atol=1e-12)
    np.testing.assert_allclose(brake_start.velocity, terminal.velocity, atol=1e-12)
    np.testing.assert_allclose(brake_start.acceleration, terminal.acceleration, atol=1e-12)
    brake_finish = 12.0 + executor.active_plan.total_duration
    hold = executor.sample_command(brake_finish + 1e-9, np.zeros(3), [1.0, 0.0, 0.0, 0.0])
    assert hold.mode == "hold"
    np.testing.assert_allclose(hold.velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(hold.acceleration, 0.0, atol=1e-12)
    repeated = executor.sample_command(brake_finish + 0.01, np.zeros(3), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(repeated.position, hold.position, atol=1e-12)


def test_install_guard_rejects_obsolete_prediction():
    executor = runtime.PlanExecutor()
    decision = executor.consider_prediction(
        _linear_prediction(),
        np.zeros(3),
        [1.0, 0.0, 0.0, 0.0],
        10.0,
        10.01,
        _free_clearance(),
        install_guard=lambda: False,
    )
    assert not decision.accepted
    assert decision.reason == "install_guard_rejected"
    assert executor.active_plan is None
