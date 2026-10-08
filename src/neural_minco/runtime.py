"""ROS-free inference and guarded execution for the current GP-SR planner.

A physical body-frame ``[v, a, goal]`` state and one normalized
``[1, 30, 180]`` range image decode to a fixed-two-second, three-segment
minimum-jerk trajectory. The execution utilities turn it into guarded commands
and safety fallbacks without importing ROS.
"""

import copy
import math
import multiprocessing
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import ACCELERATION_LIMIT_MPS2, JERK_LIMIT_MPS3, SENSOR_SYNC_TOLERANCE_S, VELOCITY_LIMIT_MPS
from .geometry import body_to_world_numpy, quaternion_matrix_numpy, world_to_body_numpy
from .minco import MinimumJerkDecoder, sample_trajectory_batch_with_derivatives
from .model import TrajectoryAttentionPlanner, decode_action, load_model

MAX_SENSOR_DELTA_S = SENSOR_SYNC_TOLERANCE_S
MAX_PLANNING_LATENCY_S = 0.10
COMMAND_RATE_HZ = 100.0
INFERENCE_STARTUP_TIMEOUT_S = 120.0
INFERENCE_SHUTDOWN_TIMEOUT_S = 2.0
PREDICTION_SAMPLES = 20
YAW_RATE_MAX = 2.5
PREFIX_DURATION_S = 0.30
PREFIX_SAMPLES = 101
VISIBLE_CLEARANCE_M = 0.40
V_MAX = VELOCITY_LIMIT_MPS
A_MAX = ACCELERATION_LIMIT_MPS2
J_MAX = JERK_LIMIT_MPS3
BRAKING_MIN_DURATION_S = 0.5
BRAKING_MAX_DURATION_S = 5.0
BRAKING_DURATION_STEP_S = 0.1
BRAKING_SAMPLES = 1001
HOLD_VELOCITY_TOLERANCE_MPS = 1e-3
HOLD_ACCELERATION_TOLERANCE_MPS2 = 1e-3
TERMINAL_ENGAGE_DISTANCE_M = 6.0
TERMINAL_MIN_DURATION_S = 0.5
TERMINAL_MAX_DURATION_S = 5.0
TERMINAL_DURATION_STEP_S = 0.1
TERMINAL_SAMPLES = 1001
TERMINAL_MAX_GOAL_DEPARTURE_M = 0.1
TERMINAL_MAX_PATH_EXCESS_M = 0.5


@dataclass(frozen=True)
class TrajectoryPrediction:
    """Physical body-frame parameters and samples from the final GP-SR stage."""

    waypoints: torch.Tensor
    durations: torch.Tensor
    endpoint: torch.Tensor
    terminal_velocity: torch.Tensor
    terminal_acceleration: torch.Tensor
    coefficients: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor


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


@dataclass(frozen=True)
class PredictionPayload:
    """Finite CPU arrays transferred across the inference process boundary."""

    waypoints: np.ndarray
    durations: np.ndarray
    endpoint: np.ndarray
    terminal_velocity: np.ndarray
    terminal_acceleration: np.ndarray
    coefficients: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "waypoints": (1, 2, 3),
            "durations": (1, 3),
            "endpoint": (1, 3),
            "terminal_velocity": (1, 3),
            "terminal_acceleration": (1, 3),
            "coefficients": (1, 3, 6, 3),
            "position": (1, 20, 3),
            "velocity": (1, 20, 3),
            "acceleration": (1, 20, 3),
            "jerk": (1, 20, 3),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError("prediction payload %s must be finite floating data with shape %s" % (name, shape))
            array = np.ascontiguousarray(value).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if np.any(self.durations <= 0.0) or not math.isclose(
            float(self.durations.sum()), 2.0, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError("prediction payload durations must be positive and sum to 2 seconds")


def prediction_to_payload(prediction: TrajectoryPrediction) -> PredictionPayload:
    """Copy one decoded prediction into validated CPU arrays."""
    if not isinstance(prediction, TrajectoryPrediction):
        raise TypeError("prediction must be a TrajectoryPrediction")
    values = {}
    for name in _PREDICTION_FIELDS:
        tensor = getattr(prediction, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("prediction %s must be a tensor" % name)
        values[name] = tensor.detach().cpu().contiguous().numpy().copy()
    return PredictionPayload(**values)


def prediction_from_payload(payload: PredictionPayload) -> TrajectoryPrediction:
    """Rebuild the existing prediction value from validated CPU arrays."""
    if not isinstance(payload, PredictionPayload):
        raise TypeError("payload must be a PredictionPayload")
    tensors = {name: torch.from_numpy(getattr(payload, name).copy()) for name in _PREDICTION_FIELDS}
    return TrajectoryPrediction(**tensors)


def decode_prediction(
    planner: TrajectoryAttentionPlanner, raw: Mapping[str, torch.Tensor], state_body: torch.Tensor, *, samples: int = 20
) -> TrajectoryPrediction:
    """Decode one current-model output through fixed-total M3 minimum jerk."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if state_body.ndim != 2 or state_body.shape[1] != 9:
        raise ValueError("physical state must have shape (B,9)")
    waypoints, durations, endpoint, velocity_end, acceleration_end = decode_action(
        raw,
        waypoint_scale=planner.waypoint_scale,
        terminal_velocity_scale=planner.v_end_scale,
        terminal_acceleration_scale=planner.acceleration_scale,
        horizon=planner.horizon,
    )
    coefficients = MinimumJerkDecoder()(
        waypoints, durations, endpoint, velocity_end, state_body, a_end=acceleration_end
    )
    position, velocity, acceleration, jerk = sample_trajectory_batch_with_derivatives(
        coefficients, durations, samples + 1, derivs=(0, 1, 2, 3)
    )
    return TrajectoryPrediction(
        waypoints=waypoints,
        durations=durations,
        endpoint=endpoint,
        terminal_velocity=velocity_end,
        terminal_acceleration=acceleration_end,
        coefficients=coefficients,
        position=position[:, 1:],
        velocity=velocity[:, 1:],
        acceleration=acceleration[:, 1:],
        jerk=jerk[:, 1:],
    )


@torch.inference_mode()
def predict(
    planner: TrajectoryAttentionPlanner, observation: torch.Tensor, state_body: torch.Tensor, *, samples: int = 20
) -> TrajectoryPrediction:
    """Run the final z3 planner from normalized LiDAR and physical body state."""
    device = next(planner.parameters()).device
    observation = observation.to(device=device, dtype=torch.float32)
    state_body = state_body.to(device=device, dtype=torch.float32)
    raw = planner(observation, state_body)
    return decode_prediction(planner, raw, state_body, samples=samples)


class InferenceWorkerError(RuntimeError):
    """A persistent inference child failed its bounded protocol."""


def _inference_worker_main(connection, checkpoint_path: str, device: str, samples: int) -> None:
    """Own model/device state and serve one synchronous prediction at a time."""
    request_id = None
    try:
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        planner, checkpoint = load_model(checkpoint_path, resolved_device)
        parameter = next(planner.parameters())
        observation = torch.ones(1, 1, 30, 180, dtype=torch.float32, device=parameter.device)
        state_body = torch.zeros(1, 9, dtype=torch.float32, device=parameter.device)
        state_body[:, 6] = 1.0
        started = time.perf_counter()
        predict(planner, observation, state_body, samples=samples)
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)
        warmup_ms = 1e3 * (time.perf_counter() - started)
        connection.send(("ready", copy.deepcopy(checkpoint["config"]), str(resolved_device), warmup_ms))
        while True:
            message = connection.recv()
            if not isinstance(message, tuple) or not message:
                raise ValueError("invalid inference request")
            if message[0] == "stop":
                return
            if len(message) != 4 or message[0] != "predict":
                raise ValueError("invalid inference request")
            _, request_id, observation_array, state_array = message
            prediction = predict(
                planner, torch.from_numpy(observation_array), torch.from_numpy(state_array), samples=samples
            )
            if parameter.is_cuda:
                torch.cuda.synchronize(parameter.device)
            connection.send(("prediction", request_id, prediction_to_payload(prediction)))
    except (EOFError, BrokenPipeError):
        return
    except BaseException as error:
        try:
            connection.send(("error", request_id, type(error).__name__, str(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class InferenceWorker:
    """Persistent spawned model process with bounded synchronous requests."""

    def __init__(
        self, checkpoint_path: str, device: str = "auto", startup_timeout_s: float = INFERENCE_STARTUP_TIMEOUT_S
    ):
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("checkpoint path must be a nonempty string")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("worker device must be auto, cpu, or cuda")
        if not math.isfinite(float(startup_timeout_s)) or startup_timeout_s <= 0.0:
            raise ValueError("startup timeout must be finite and positive")
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection = parent_connection
        self._process = context.Process(
            target=_inference_worker_main,
            args=(child_connection, checkpoint_path, device, PREDICTION_SAMPLES),
            name="gp-sr-inference",
            daemon=True,
        )
        self._lock = threading.RLock()
        self._request_id = 0
        self._failed = False
        self._closed = False
        self._started = False
        self._config = None
        self._device = None
        self._warmup_ms = None
        try:
            self._process.start()
            self._started = True
            child_connection.close()
            if not self._connection.poll(float(startup_timeout_s)):
                self._failed = True
                raise InferenceWorkerError("inference worker startup timed out")
            response = self._connection.recv()
            if isinstance(response, tuple) and len(response) == 4 and response[0] == "error":
                self._failed = True
                raise InferenceWorkerError("%s: %s" % (response[2], response[3]))
            if not self._valid_ready_response(response):
                self._failed = True
                raise InferenceWorkerError("invalid inference worker startup response")
            self._config = copy.deepcopy(response[1])
            self._device = response[2]
            self._warmup_ms = float(response[3])
        except BaseException as error:
            if not self._started:
                child_connection.close()
            self.close()
            if isinstance(error, InferenceWorkerError):
                raise
            raise InferenceWorkerError("inference worker startup failed: %s" % error) from error

    @staticmethod
    def _valid_ready_response(response) -> bool:
        return bool(
            isinstance(response, tuple)
            and len(response) == 4
            and response[0] == "ready"
            and isinstance(response[1], Mapping)
            and response[2] in {"cpu", "cuda"}
            and math.isfinite(float(response[3]))
            and response[3] >= 0.0
        )

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def device(self) -> str:
        return self._device

    @property
    def warmup_ms(self) -> float:
        return self._warmup_ms

    @property
    def is_alive(self) -> bool:
        return self._started and self._process.is_alive()

    def predict(self, observation, state_body, *, timeout_s: float) -> TrajectoryPrediction:
        """Return one current prediction before the caller's remaining deadline."""
        observation_array = np.asarray(observation, dtype=np.float32)
        state_array = np.asarray(state_body, dtype=np.float32)
        if observation_array.shape != (1, 1, 30, 180) or not np.isfinite(observation_array).all():
            raise ValueError("worker observation must be finite with shape (1,1,30,180)")
        if state_array.shape != (1, 9) or not np.isfinite(state_array).all():
            raise ValueError("worker physical state must be finite with shape (1,9)")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0.0:
            raise ValueError("worker timeout must be finite and positive")
        with self._lock:
            if self._closed or self._failed or not self._process.is_alive():
                raise InferenceWorkerError("inference worker is unavailable")
            self._request_id += 1
            request_id = self._request_id
            try:
                self._connection.send(
                    ("predict", request_id, np.ascontiguousarray(observation_array), np.ascontiguousarray(state_array))
                )
                if not self._connection.poll(float(timeout_s)):
                    self._failed = True
                    raise InferenceWorkerError("inference request %d timed out" % request_id)
                response = self._connection.recv()
            except InferenceWorkerError:
                raise
            except (BrokenPipeError, EOFError, OSError) as error:
                self._failed = True
                raise InferenceWorkerError("inference worker connection closed: %s" % error) from error
            if isinstance(response, tuple) and len(response) == 4 and response[0] == "error":
                self._failed = True
                raise InferenceWorkerError("%s: %s" % (response[2], response[3]))
            if not isinstance(response, tuple) or len(response) != 3 or response[:2] != ("prediction", request_id):
                self._failed = True
                raise InferenceWorkerError("mismatched inference response")
            try:
                return prediction_from_payload(response[2])
            except (TypeError, ValueError) as error:
                self._failed = True
                raise InferenceWorkerError("invalid inference prediction: %s" % error) from error

    def close(self) -> None:
        """Close IPC and stop only this instance's owned child process."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._started and self._process.is_alive():
                try:
                    self._connection.send(("stop",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._connection.close()
            if self._started:
                self._process.join(timeout=INFERENCE_SHUTDOWN_TIMEOUT_S)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=INFERENCE_SHUTDOWN_TIMEOUT_S)


@dataclass(frozen=True)
class OdometrySample:
    """One finite world-frame odometry state and matched command acceleration."""

    stamp: float
    position_world: np.ndarray
    quaternion_wxyz: np.ndarray
    velocity_world: np.ndarray
    acceleration_world: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            ("position", "position_world", self.position_world, (3,)),
            ("quaternion", "quaternion_wxyz", self.quaternion_wxyz, (4,)),
            ("velocity", "velocity_world", self.velocity_world, (3,)),
            ("acceleration", "acceleration_world", self.acceleration_world, (3,)),
        )
        if not math.isfinite(float(self.stamp)):
            raise ValueError("odometry stamp must be finite")
        for label, name, value, shape in arrays:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("odometry %s must be finite with shape %s" % (label, shape))
            object.__setattr__(self, name, array.copy())
        if np.linalg.norm(self.quaternion_wxyz) <= 1e-12:
            raise ValueError("odometry quaternion must be nonzero")


def select_nearest_odometry(
    samples: Sequence[OdometrySample], scan_stamp: float, *, max_delta_s: float = MAX_SENSOR_DELTA_S
) -> Optional[OdometrySample]:
    """Return the nearest real odometry row when it meets the Episode time contract."""
    if not math.isfinite(float(scan_stamp)):
        raise ValueError("scan stamp must be finite")
    if not math.isfinite(float(max_delta_s)) or max_delta_s < 0.0:
        raise ValueError("maximum sensor delta must be finite and nonnegative")
    if not samples:
        return None
    nearest = min(samples, key=lambda sample: abs(sample.stamp - float(scan_stamp)))
    if abs(nearest.stamp - float(scan_stamp)) > float(max_delta_s) + 1e-12:
        return None
    return nearest


@dataclass(frozen=True)
class TrajectorySample:
    """Position and first three derivatives at one trajectory time."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    finished: bool


@dataclass(frozen=True)
class TrajectorySamples:
    """Vectorized position and derivatives on one timestamp array."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    finished: np.ndarray


class WorldMincoTrajectory:
    """Body-frame polynomial rigidly anchored to one world-frame odometry pose."""

    def __init__(self, coefficients_body, durations, anchor_position_world, anchor_quaternion_wxyz, start_stamp: float):
        coefficients = np.asarray(coefficients_body, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        position = np.asarray(anchor_position_world, dtype=np.float64)
        quaternion = np.asarray(anchor_quaternion_wxyz, dtype=np.float64)
        if (
            coefficients.ndim != 3
            or coefficients.shape[1] < 4
            or coefficients.shape[2] != 3
            or durations.shape != (coefficients.shape[0],)
            or position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(durations).all()
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or not math.isfinite(float(start_stamp))
            or np.any(durations <= 0.0)
            or np.linalg.norm(quaternion) <= 1e-12
        ):
            raise ValueError("world MINCO trajectory inputs are invalid")
        rotation = quaternion_matrix_numpy(quaternion)
        coefficients_world = coefficients @ rotation.T
        coefficients_world[:, 0] += position
        self.coefficients = coefficients_world
        self.durations = durations.copy()
        self.segment_ends = np.cumsum(self.durations)
        self.start_stamp = float(start_stamp)
        self.total_duration = float(self.segment_ends[-1])

    @staticmethod
    def _evaluate(coefficients: np.ndarray, time_s, derivative: int) -> np.ndarray:
        time_s = np.asarray(time_s, dtype=np.float64)
        values = np.zeros(time_s.shape + (3,), dtype=np.float64)
        for power in range(derivative, coefficients.shape[-2]):
            factor = math.prod(range(power - derivative + 1, power + 1))
            values += coefficients[..., power, :] * factor * time_s[..., None] ** (power - derivative)
        return values

    def sample(self, stamp: float) -> TrajectorySample:
        """Evaluate one finite timestamp, clamped to the represented interval."""
        if not math.isfinite(float(stamp)):
            raise ValueError("trajectory sample stamp must be finite")
        raw_elapsed = float(stamp) - self.start_stamp
        finished = raw_elapsed >= self.total_duration
        elapsed = min(max(0.0, raw_elapsed), self.total_duration)
        segment = min(int(np.searchsorted(self.segment_ends, elapsed, side="right")), len(self.durations) - 1)
        start = 0.0 if segment == 0 else float(self.segment_ends[segment - 1])
        local = elapsed - start
        coefficients = self.coefficients[segment]
        return TrajectorySample(
            position=self._evaluate(coefficients, local, 0),
            velocity=self._evaluate(coefficients, local, 1),
            acceleration=self._evaluate(coefficients, local, 2),
            jerk=self._evaluate(coefficients, local, 3),
            finished=finished,
        )

    def sample_many(self, stamps) -> TrajectorySamples:
        """Evaluate one nonempty one-dimensional finite timestamp array."""
        stamps = np.asarray(stamps, dtype=np.float64)
        if stamps.ndim != 1 or not len(stamps) or not np.isfinite(stamps).all():
            raise ValueError("trajectory sample timestamps must be finite [N]")
        raw_elapsed = stamps - self.start_stamp
        finished = raw_elapsed >= self.total_duration
        elapsed = np.clip(raw_elapsed, 0.0, self.total_duration)
        segment = np.searchsorted(self.segment_ends, elapsed, side="right")
        segment = np.minimum(segment, len(self.durations) - 1)
        starts = np.concatenate(([0.0], self.segment_ends[:-1]))
        local = elapsed - starts[segment]
        coefficients = self.coefficients[segment]
        return TrajectorySamples(
            position=self._evaluate(coefficients, local, 0),
            velocity=self._evaluate(coefficients, local, 1),
            acceleration=self._evaluate(coefficients, local, 2),
            jerk=self._evaluate(coefficients, local, 3),
            finished=finished,
        )


_NORMALIZED_TAIL_MATRIX = np.asarray(
    [[1.0, 1.0, 1.0, 1.0], [4.0, 5.0, 6.0, 7.0], [12.0, 20.0, 30.0, 42.0], [24.0, 60.0, 120.0, 210.0]], dtype=np.float64
)


def _terminal_coefficients(velocity, acceleration, goal, duration: float) -> np.ndarray:
    normalized = np.zeros((8, 3), dtype=np.float64)
    normalized[1] = velocity * duration
    normalized[2] = 0.5 * acceleration * duration**2
    right_hand_side = np.stack(
        [
            goal - normalized[:4].sum(axis=0),
            -(normalized[1] + 2.0 * normalized[2] + 3.0 * normalized[3]),
            -(2.0 * normalized[2] + 6.0 * normalized[3]),
            -6.0 * normalized[3],
        ],
        axis=0,
    )
    normalized[4:] = np.linalg.solve(_NORMALIZED_TAIL_MATRIX, right_hand_side)
    powers = duration ** np.arange(8, dtype=np.float64)
    return (normalized / powers[:, None])[None]


def _sample_single_polynomial(coefficients, duration, samples, derivative=0):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (1, 8, 3):
        raise ValueError("terminal coefficients must have shape [1,8,3]")
    times = np.linspace(0.0, float(duration), int(samples), dtype=np.float64)
    values = np.zeros((len(times), 3), dtype=np.float64)
    for power in range(int(derivative), 8):
        factor = math.prod(range(power - int(derivative) + 1, power + 1))
        values += coefficients[0, power] * factor * times[:, None] ** (power - int(derivative))
    return values


@dataclass(frozen=True)
class TerminalTrajectoryPlan:
    coefficients_body: np.ndarray
    duration: float
    sampled_points_body: np.ndarray
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    maximum_goal_departure: float
    path_excess: float


def plan_terminal_trajectory(
    velocity_body,
    acceleration_body,
    goal_body,
    *,
    min_duration: float = TERMINAL_MIN_DURATION_S,
    max_duration: float = TERMINAL_MAX_DURATION_S,
    duration_step: float = TERMINAL_DURATION_STEP_S,
    samples: int = TERMINAL_SAMPLES,
    v_max: float = V_MAX,
    a_max: float = A_MAX,
    j_max: float = J_MAX,
    max_goal_departure: float = TERMINAL_MAX_GOAL_DEPARTURE_M,
    max_path_excess: float = TERMINAL_MAX_PATH_EXCESS_M,
) -> Optional[TerminalTrajectoryPlan]:
    """Find the shortest feasible analytic trajectory that stops at the goal."""
    velocity = np.asarray(velocity_body, dtype=np.float64)
    acceleration = np.asarray(acceleration_body, dtype=np.float64)
    goal = np.asarray(goal_body, dtype=np.float64)
    scalars = np.asarray(
        [min_duration, max_duration, duration_step, v_max, a_max, j_max, max_goal_departure, max_path_excess],
        dtype=np.float64,
    )
    if (
        velocity.shape != (3,)
        or acceleration.shape != (3,)
        or goal.shape != (3,)
        or not np.isfinite(np.concatenate([velocity, acceleration, goal, scalars])).all()
        or min_duration <= 0.0
        or max_duration < min_duration
        or duration_step <= 0.0
        or samples < 2
        or min(v_max, a_max, j_max) <= 0.0
        or min(max_goal_departure, max_path_excess) < 0.0
    ):
        raise ValueError("terminal trajectory inputs are invalid")
    count = int(math.floor((max_duration - min_duration) / duration_step + 1e-12)) + 1
    durations = min_duration + duration_step * np.arange(count, dtype=np.float64)
    if durations[-1] < max_duration - 1e-12:
        durations = np.append(durations, max_duration)
    goal_distance = float(np.linalg.norm(goal))
    for duration in durations:
        coefficients = _terminal_coefficients(velocity, acceleration, goal, float(duration))
        derivatives = [
            _sample_single_polynomial(coefficients, duration, samples, derivative=index) for index in range(4)
        ]
        maxima = [float(np.linalg.norm(value, axis=-1).max()) for value in derivatives[1:]]
        departure = max(0.0, float(np.linalg.norm(derivatives[0] - goal[None], axis=-1).max()) - goal_distance)
        path_excess = max(0.0, float(np.linalg.norm(np.diff(derivatives[0], axis=0), axis=-1).sum()) - goal_distance)
        if (
            maxima[0] <= v_max + 1e-9
            and maxima[1] <= a_max + 1e-9
            and maxima[2] <= j_max + 1e-9
            and departure <= max_goal_departure + 1e-9
            and path_excess <= max_path_excess + 1e-9
        ):
            return TerminalTrajectoryPlan(
                coefficients_body=coefficients,
                duration=float(duration),
                sampled_points_body=derivatives[0],
                max_velocity=maxima[0],
                max_acceleration=maxima[1],
                max_jerk=maxima[2],
                maximum_goal_departure=departure,
                path_excess=path_excess,
            )
    return None


@dataclass(frozen=True)
class BrakingTrajectoryPlan:
    coefficients_frame: np.ndarray
    duration: float
    endpoint_frame: np.ndarray
    sampled_points_frame: np.ndarray
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    within_limits: bool


def plan_braking_trajectory(
    velocity_frame,
    acceleration_frame,
    *,
    min_duration: float = BRAKING_MIN_DURATION_S,
    max_duration: float = BRAKING_MAX_DURATION_S,
    duration_step: float = BRAKING_DURATION_STEP_S,
    samples: int = BRAKING_SAMPLES,
    v_max: float = V_MAX,
    a_max: float = A_MAX,
    j_max: float = J_MAX,
    allow_limit_violation: bool = False,
) -> Optional[BrakingTrajectoryPlan]:
    """Find a continuous free-endpoint minimum-snap stop in the current frame."""
    velocity = np.asarray(velocity_frame, dtype=np.float64)
    acceleration = np.asarray(acceleration_frame, dtype=np.float64)
    scalars = np.asarray([min_duration, max_duration, duration_step, v_max, a_max, j_max], dtype=np.float64)
    if (
        velocity.shape != (3,)
        or acceleration.shape != (3,)
        or not np.isfinite(np.concatenate([velocity, acceleration, scalars])).all()
        or min_duration <= 0.0
        or max_duration < min_duration
        or duration_step <= 0.0
        or samples < 2
        or min(v_max, a_max, j_max) <= 0.0
    ):
        raise ValueError("braking trajectory inputs are invalid")
    count = int(math.floor((max_duration - min_duration) / duration_step + 1e-12)) + 1
    durations = min_duration + duration_step * np.arange(count, dtype=np.float64)
    if durations[-1] < max_duration - 1e-12:
        durations = np.append(durations, max_duration)
    unit = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    u2, u3 = unit * unit, unit * unit * unit
    u4, u5, u6 = u2 * u2, u2 * u2 * unit, u3 * u3
    basis = (
        unit - 2.5 * u4 + 3.0 * u5 - u6,
        0.5 * u2 - 1.5 * u4 + 1.6 * u5 - 0.5 * u6,
        1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5,
        unit - 6.0 * u3 + 8.0 * u4 - 3.0 * u5,
        -30.0 * u2 + 60.0 * u3 - 30.0 * u4,
        1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4,
        -60.0 * unit + 180.0 * u2 - 120.0 * u3,
        -36.0 * unit + 96.0 * u2 - 60.0 * u3,
    )
    duration = durations[:, None, None]
    velocity_row = velocity[None, None]
    acceleration_row = acceleration[None, None]
    position = (
        duration * basis[0][None, :, None] * velocity_row + duration**2 * basis[1][None, :, None] * acceleration_row
    )
    profiles = (
        basis[2][None, :, None] * velocity_row + duration * basis[3][None, :, None] * acceleration_row,
        basis[4][None, :, None] / duration * velocity_row + basis[5][None, :, None] * acceleration_row,
        basis[6][None, :, None] / duration**2 * velocity_row + basis[7][None, :, None] / duration * acceleration_row,
    )
    maxima = np.stack([np.linalg.norm(profile, axis=-1).max(axis=1) for profile in profiles], axis=-1)
    within = (maxima[:, 0] <= v_max + 1e-9) & (maxima[:, 1] <= a_max + 1e-9) & (maxima[:, 2] <= j_max + 1e-9)
    feasible = np.flatnonzero(within)
    if len(feasible):
        selected = int(feasible[0])
    elif allow_limit_violation:
        selected = int(np.argmin(np.max(maxima / np.asarray([v_max, a_max, j_max])[None], axis=1)))
    else:
        return None
    selected_duration = float(durations[selected])
    endpoint = 0.5 * velocity * selected_duration + 0.1 * acceleration * selected_duration**2
    coefficients = np.zeros((1, 8, 3), dtype=np.float64)
    coefficients[0, 1] = velocity
    coefficients[0, 2] = 0.5 * acceleration
    coefficients[0, 4] = -2.5 * velocity / selected_duration**3 - 1.5 * acceleration / selected_duration**2
    coefficients[0, 5] = 3.0 * velocity / selected_duration**4 + 1.6 * acceleration / selected_duration**3
    coefficients[0, 6] = -velocity / selected_duration**5 - 0.5 * acceleration / selected_duration**4
    return BrakingTrajectoryPlan(
        coefficients_frame=coefficients,
        duration=selected_duration,
        endpoint_frame=endpoint,
        sampled_points_frame=position[selected],
        max_velocity=float(maxima[selected, 0]),
        max_acceleration=float(maxima[selected, 1]),
        max_jerk=float(maxima[selected, 2]),
        within_limits=bool(within[selected]),
    )


class PointCloudClearance:
    """Nearest Euclidean clearance to the current finite return cloud."""

    def __init__(self, obstacle_points):
        points = np.asarray(obstacle_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("obstacle points must be finite [N,3]")
        self.points = points.copy()
        if len(points):
            from scipy.spatial import cKDTree

            self.tree = cKDTree(points, balanced_tree=False, compact_nodes=False)
        else:
            self.tree = None

    def minimum(self, trajectory_points) -> float:
        points = np.asarray(trajectory_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError("trajectory points must be finite [N,3]")
        if self.tree is None:
            return float("inf")
        distance, _ = self.tree.query(points, k=1)
        return float(np.min(distance))


class CurrentScanClearance:
    """Observed-return and radial-occlusion check for exactly one LiDAR scan."""

    def __init__(
        self,
        observation,
        position_world,
        quaternion_wxyz,
        stamp: float,
        obstacle_points_world,
        *,
        r_max: float = 20.0,
        n_horizon: int = 180,
        n_vertical: int = 90,
        vertical_fov_min_deg: float = -7.0,
        vertical_fov_max_deg: float = 52.0,
        row_radius: int = 1,
        column_radius: int = 2,
    ):
        elevation = -math.pi / 2.0 + (np.arange(int(n_vertical), dtype=np.float64) + 0.5) * math.pi / int(n_vertical)
        kept = np.flatnonzero(
            (elevation >= math.radians(vertical_fov_min_deg) - 1e-9)
            & (elevation <= math.radians(vertical_fov_max_deg) + 1e-9)
        )
        normalized = np.asarray(observation, dtype=np.float64)
        if normalized.shape == (1, len(kept), int(n_horizon)):
            normalized = normalized[0]
        position = np.asarray(position_world, dtype=np.float64)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        scalars = np.asarray([stamp, r_max, vertical_fov_min_deg, vertical_fov_max_deg], dtype=np.float64)
        if (
            normalized.shape != (len(kept), int(n_horizon))
            or not np.isfinite(normalized).all()
            or np.any((normalized < 0.0) | (normalized > 1.0 + 1e-6))
            or position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) <= 1e-12
            or not np.isfinite(scalars).all()
            or r_max <= 0.0
            or n_horizon <= 0
            or n_vertical <= 0
            or vertical_fov_min_deg >= vertical_fov_max_deg
            or row_radius < 0
            or column_radius < 0
            or 2 * int(row_radius) + 1 > len(kept)
            or 2 * int(column_radius) + 1 > int(n_horizon)
        ):
            raise ValueError("current scan clearance inputs are invalid")
        self.observation = np.minimum(normalized, 1.0).copy()
        self.position_world = position.copy()
        self.quaternion_wxyz = quaternion.copy()
        self.stamp = float(stamp)
        self.r_max = float(r_max)
        self.n_horizon = int(n_horizon)
        self.n_vertical = int(n_vertical)
        self.first_fov_row = int(kept[0])
        self.v_fov = int(len(kept))
        self.row_radius = int(row_radius)
        self.column_radius = int(column_radius)
        self.obstacle_clearance = PointCloudClearance(obstacle_points_world)

    def _headroom(self, points_world) -> np.ndarray:
        points_body = world_to_body_numpy(points_world - self.position_world[None], self.quaternion_wxyz)
        radius = np.linalg.norm(points_body, axis=1)
        theta = np.arctan2(points_body[:, 1], points_body[:, 0])
        phi = np.arcsin(np.clip(points_body[:, 2] / np.maximum(radius, 1e-12), -1.0, 1.0))
        columns = np.floor((theta + math.pi) / (2.0 * math.pi) * self.n_horizon).astype(np.int64) % self.n_horizon
        rows = np.floor((phi + math.pi / 2.0) / math.pi * self.n_vertical).astype(np.int64) - self.first_fov_row
        valid = (
            (radius > 1e-9)
            & (radius <= self.r_max)
            & (rows - self.row_radius >= 0)
            & (rows + self.row_radius < self.v_fov)
        )
        headroom = np.full(len(points_world), -np.inf, dtype=np.float64)
        headroom[radius <= 1e-9] = np.inf
        if valid.any():
            minimum_depth = np.full(len(points_world), np.inf, dtype=np.float64)
            for row_offset in range(-self.row_radius, self.row_radius + 1):
                for column_offset in range(-self.column_radius, self.column_radius + 1):
                    depth = (
                        self.r_max
                        * self.observation[
                            np.clip(rows + row_offset, 0, self.v_fov - 1), (columns + column_offset) % self.n_horizon
                        ]
                    )
                    minimum_depth = np.minimum(minimum_depth, depth)
            headroom[valid] = minimum_depth[valid] - radius[valid]
        return headroom

    def minimum(self, trajectory_points) -> float:
        """Return a current-visible check, not a global safety certificate."""
        points = np.asarray(trajectory_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError("trajectory points must be finite [N,3]")
        headroom = self._headroom(points)
        headroom[np.isneginf(headroom)] = np.inf
        return min(float(np.min(headroom)), self.obstacle_clearance.minimum(points))


@dataclass(frozen=True)
class VerifiedPrefixPlan:
    trajectory: WorldMincoTrajectory
    verified_until: float
    fallback: Optional[WorldMincoTrajectory]
    prefix_clearance: float
    fallback_clearance: float
    prefix_max_velocity: float
    prefix_max_acceleration: float
    prefix_max_jerk: float


@dataclass(frozen=True)
class PrefixVerificationResult:
    plan: Optional[VerifiedPrefixPlan]
    reason: str


def inspect_trajectory_prefix(
    trajectory: WorldMincoTrajectory,
    clearance,
    now: float,
    *,
    prefix_duration: float = PREFIX_DURATION_S,
    clearance_margin: float = VISIBLE_CLEARANCE_M,
    prefix_samples: int = PREFIX_SAMPLES,
    braking_min_duration: float = BRAKING_MIN_DURATION_S,
    braking_max_duration: float = BRAKING_MAX_DURATION_S,
    braking_duration_step: float = BRAKING_DURATION_STEP_S,
    braking_samples: int = BRAKING_SAMPLES,
    v_max: float = V_MAX,
    a_max: float = A_MAX,
    j_max: float = J_MAX,
) -> PrefixVerificationResult:
    """Check the next executable prefix and a complete continuous stop."""
    if (
        not isinstance(trajectory, WorldMincoTrajectory)
        or not callable(getattr(clearance, "minimum", None))
        or not math.isfinite(float(now))
        or prefix_duration <= 0.0
        or clearance_margin < 0.0
        or prefix_samples < 2
        or braking_samples < 2
    ):
        raise ValueError("verified-prefix inputs are invalid")
    finish = trajectory.start_stamp + trajectory.total_duration
    if float(now) >= finish:
        return PrefixVerificationResult(None, "trajectory_finished")
    verified_until = min(float(now) + float(prefix_duration), finish)
    times = np.linspace(float(now), verified_until, int(prefix_samples), dtype=np.float64)
    states = trajectory.sample_many(times)
    maxima = np.asarray(
        [np.linalg.norm(getattr(states, field), axis=-1).max() for field in ("velocity", "acceleration", "jerk")],
        dtype=np.float64,
    )
    if np.any(maxima > np.asarray([v_max, a_max, j_max]) + 1e-9):
        return PrefixVerificationResult(None, "prefix_dynamics=%.2f/%.2f/%.2f" % tuple(maxima))
    prefix_clearance = float(clearance.minimum(states.position))
    if prefix_clearance < float(clearance_margin):
        return PrefixVerificationResult(None, "prefix_clearance=%.2fm" % prefix_clearance)
    boundary = trajectory.sample(verified_until)
    terminal_is_stationary = (
        np.linalg.norm(boundary.velocity) <= HOLD_VELOCITY_TOLERANCE_MPS
        and np.linalg.norm(boundary.acceleration) <= HOLD_ACCELERATION_TOLERANCE_MPS2
    )
    if boundary.finished and terminal_is_stationary:
        return PrefixVerificationResult(
            VerifiedPrefixPlan(
                trajectory=trajectory,
                verified_until=verified_until,
                fallback=None,
                prefix_clearance=prefix_clearance,
                fallback_clearance=float("inf"),
                prefix_max_velocity=float(maxima[0]),
                prefix_max_acceleration=float(maxima[1]),
                prefix_max_jerk=float(maxima[2]),
            ),
            "accepted_finished_prefix",
        )
    braking = plan_braking_trajectory(
        boundary.velocity,
        boundary.acceleration,
        min_duration=braking_min_duration,
        max_duration=braking_max_duration,
        duration_step=braking_duration_step,
        samples=braking_samples,
        v_max=v_max,
        a_max=a_max,
        j_max=j_max,
    )
    if braking is None:
        return PrefixVerificationResult(None, "no_feasible_brake")
    fallback = WorldMincoTrajectory(
        braking.coefficients_frame, [braking.duration], boundary.position, [1.0, 0.0, 0.0, 0.0], verified_until
    )
    fallback_clearance = float(clearance.minimum(braking.sampled_points_frame + boundary.position[None]))
    if fallback_clearance < float(clearance_margin):
        return PrefixVerificationResult(None, "brake_clearance=%.2fm" % fallback_clearance)
    return PrefixVerificationResult(
        VerifiedPrefixPlan(
            trajectory=trajectory,
            verified_until=verified_until,
            fallback=fallback,
            prefix_clearance=prefix_clearance,
            fallback_clearance=fallback_clearance,
            prefix_max_velocity=float(maxima[0]),
            prefix_max_acceleration=float(maxima[1]),
            prefix_max_jerk=float(maxima[2]),
        ),
        "accepted",
    )


def yaw_from_quaternion_wxyz(quaternion) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 1e-12:
        raise ValueError("yaw extraction requires one finite nonzero quaternion")
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class VelocityYawTracker:
    def __init__(self, yaw: float, max_rate: float = YAW_RATE_MAX, min_speed: float = 0.2):
        if not np.isfinite([yaw, max_rate, min_speed]).all() or max_rate <= 0.0 or min_speed < 0.0:
            raise ValueError("yaw tracker inputs are invalid")
        self.yaw = float(yaw)
        self.max_rate = float(max_rate)
        self.min_speed = float(min_speed)

    @staticmethod
    def _wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def step(self, velocity_world, dt: float) -> Tuple[float, float]:
        velocity = np.asarray(velocity_world, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("yaw update requires finite XYZ velocity and positive dt")
        if np.linalg.norm(velocity[:2]) < self.min_speed:
            return self.yaw, 0.0
        desired = math.atan2(float(velocity[1]), float(velocity[0]))
        delta = self._wrap(desired - self.yaw)
        yaw_rate = float(np.clip(delta / dt, -self.max_rate, self.max_rate))
        self.yaw = self._wrap(self.yaw + yaw_rate * dt)
        return self.yaw, yaw_rate


@dataclass(frozen=True)
class PlanDecision:
    accepted: bool
    verified: bool
    reason: str
    mode: str
    verified_until: Optional[float]


@dataclass(frozen=True)
class RuntimeCommand:
    stamp: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    yaw: float
    yaw_rate: float
    trajectory_id: int
    mode: str


class PlanExecutor:
    """Thread-safe prefix-and-brake state machine for one current GP-SR plan."""

    def __init__(
        self,
        *,
        max_planning_latency: float = MAX_PLANNING_LATENCY_S,
        prefix_duration: float = PREFIX_DURATION_S,
        prefix_samples: int = PREFIX_SAMPLES,
        clearance_margin: float = VISIBLE_CLEARANCE_M,
        v_max: float = V_MAX,
        a_max: float = A_MAX,
        j_max: float = J_MAX,
        yaw_rate_max: float = YAW_RATE_MAX,
        command_rate_hz: float = COMMAND_RATE_HZ,
    ):
        scalars = np.asarray(
            [
                max_planning_latency,
                prefix_duration,
                clearance_margin,
                v_max,
                a_max,
                j_max,
                yaw_rate_max,
                command_rate_hz,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(scalars).all() or np.any(scalars <= 0.0) or prefix_samples < 2:
            raise ValueError("plan executor settings are invalid")
        self.max_planning_latency = float(max_planning_latency)
        self.prefix_duration = float(prefix_duration)
        self.prefix_samples = int(prefix_samples)
        self.clearance_margin = float(clearance_margin)
        self.v_max, self.a_max, self.j_max = float(v_max), float(a_max), float(j_max)
        self.yaw_rate_max = float(yaw_rate_max)
        self.command_rate_hz = float(command_rate_hz)
        self._lock = threading.RLock()
        self._active_plan = None
        self._verified_until = None
        self._fallback = None
        self._mode = "hold"
        self._hold_position = None
        self._terminal_goal_world = None
        self._trajectory_id = 0
        self._yaw_tracker = None
        self._last_command_stamp = None

    @property
    def active_plan(self) -> Optional[WorldMincoTrajectory]:
        with self._lock:
            return self._active_plan

    @property
    def verified_until(self) -> Optional[float]:
        with self._lock:
            return self._verified_until

    @property
    def terminal_goal_world(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._terminal_goal_world is None else self._terminal_goal_world.copy()

    def clear_terminal_latch(self) -> None:
        with self._lock:
            self._terminal_goal_world = None
            if self._mode == "terminal_hold":
                self._mode = "hold"

    def _decision(self, accepted: bool, verified: bool, reason: str) -> PlanDecision:
        with self._lock:
            return PlanDecision(accepted, verified, reason, self._mode, self._verified_until)

    def _install_verified_plan(
        self,
        plan: VerifiedPrefixPlan,
        mode: str,
        reason: str,
        install_guard: Optional[Callable[[], bool]],
        install_lock,
        goal_world=None,
    ) -> PlanDecision:
        """Commit a verified prefix and its fallback under the installation guard."""
        with self._lock, nullcontext() if install_lock is None else install_lock:
            if install_guard is not None and not install_guard():
                return self._decision(False, False, "install_guard_rejected")
            self._active_plan = plan.trajectory
            self._verified_until = plan.verified_until
            self._fallback = plan.fallback
            self._mode = mode
            self._hold_position = None
            self._terminal_goal_world = np.asarray(goal_world, dtype=np.float64).copy() if mode == "terminal" else None
            self._trajectory_id += 1
        return self._decision(True, True, reason)

    def _inspect(self, trajectory, clearance, now):
        return inspect_trajectory_prefix(
            trajectory,
            clearance,
            now,
            prefix_duration=self.prefix_duration,
            clearance_margin=self.clearance_margin,
            prefix_samples=self.prefix_samples,
            v_max=self.v_max,
            a_max=self.a_max,
            j_max=self.j_max,
        )

    @staticmethod
    def _prediction_arrays(prediction: TrajectoryPrediction):
        if not isinstance(prediction, TrajectoryPrediction):
            raise TypeError("prediction must be a TrajectoryPrediction")
        tensors = (
            prediction.waypoints,
            prediction.durations,
            prediction.endpoint,
            prediction.terminal_velocity,
            prediction.terminal_acceleration,
            prediction.coefficients,
        )
        if not all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("prediction tensors must be finite")
        if prediction.durations.shape != (1, 3) or prediction.coefficients.shape != (1, 3, 6, 3):
            raise ValueError("online GP-SR prediction must be one M3 minimum-jerk trajectory")
        durations = prediction.durations[0].detach().cpu().double().numpy()
        coefficients = prediction.coefficients[0].detach().cpu().double().numpy()
        if np.any(durations <= 0.0) or not math.isclose(float(durations.sum()), 2.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("online GP-SR durations must be positive and sum to 2 seconds")
        return coefficients, durations

    def consider_prediction(
        self,
        prediction: TrajectoryPrediction,
        anchor_position_world,
        anchor_quaternion_wxyz,
        state_stamp: float,
        now: float,
        clearance: CurrentScanClearance,
        install_guard: Optional[Callable[[], bool]] = None,
        install_lock=None,
    ) -> PlanDecision:
        """Install one unchanged model trajectory only when its prefix and stop pass."""
        if not np.isfinite([state_stamp, now]).all() or now < state_stamp - 1e-9:
            return self._decision(False, False, "invalid_planning_time")
        if now - state_stamp > self.max_planning_latency + 1e-9:
            return self._decision(False, False, "stale_planning_state")
        try:
            coefficients, durations = self._prediction_arrays(prediction)
            trajectory = WorldMincoTrajectory(
                coefficients, durations, anchor_position_world, anchor_quaternion_wxyz, state_stamp
            )
            inspected = self._inspect(trajectory, clearance, now)
        except (TypeError, ValueError):
            return self._decision(False, False, "invalid_prediction")
        if inspected.plan is None:
            return self._decision(False, False, inspected.reason)
        return self._install_verified_plan(inspected.plan, "network", inspected.reason, install_guard, install_lock)

    def consider_terminal(
        self,
        velocity_body,
        acceleration_body,
        goal_body,
        goal_world,
        anchor_position_world,
        anchor_quaternion_wxyz,
        state_stamp: float,
        now: float,
        clearance: CurrentScanClearance,
        install_guard: Optional[Callable[[], bool]] = None,
        install_lock=None,
    ) -> PlanDecision:
        """Install a visible-clear analytic terminal stop for the current goal."""
        if not np.isfinite([state_stamp, now]).all() or now < state_stamp - 1e-9:
            return self._decision(False, False, "invalid_planning_time")
        if now - state_stamp > self.max_planning_latency + 1e-9:
            return self._decision(False, False, "stale_planning_state")
        terminal = plan_terminal_trajectory(
            velocity_body, acceleration_body, goal_body, v_max=self.v_max, a_max=self.a_max, j_max=self.j_max
        )
        if terminal is None:
            return self._decision(False, False, "outside_terminal_capture_set")
        points_world = body_to_world_numpy(
            terminal.sampled_points_body, np.asarray(anchor_position_world), anchor_quaternion_wxyz
        )
        if clearance.minimum(points_world) < self.clearance_margin:
            return self._decision(False, False, "terminal_clearance")
        trajectory = WorldMincoTrajectory(
            terminal.coefficients_body, [terminal.duration], anchor_position_world, anchor_quaternion_wxyz, state_stamp
        )
        inspected = self._inspect(trajectory, clearance, now)
        if inspected.plan is None:
            return self._decision(False, False, inspected.reason)
        return self._install_verified_plan(
            inspected.plan, "terminal", "terminal_accepted", install_guard, install_lock, goal_world=goal_world
        )

    def refresh_clearance(self, clearance: CurrentScanClearance, now: float) -> PlanDecision:
        """Recheck the active prefix, or replace it by one continuous brake."""
        with self._lock:
            trajectory = self._active_plan
            mode = self._mode
            terminal_goal = self._terminal_goal_world
        if trajectory is None and mode == "terminal_hold" and terminal_goal is not None:
            return self._decision(True, True, "terminal_hold")
        if trajectory is None:
            return self._decision(False, False, "no_active_plan")
        if mode == "brake":
            return self._decision(True, False, "emergency_brake_active")
        if now >= trajectory.start_stamp + trajectory.total_duration:
            return self._decision(True, True, "trajectory_finished")
        inspected = self._inspect(trajectory, clearance, now)
        if inspected.plan is not None:
            with self._lock:
                if self._active_plan is trajectory:
                    self._verified_until = inspected.plan.verified_until
                    self._fallback = inspected.plan.fallback
            return self._decision(True, True, inspected.reason)
        boundary = trajectory.sample(now)
        braking = plan_braking_trajectory(
            boundary.velocity, boundary.acceleration, v_max=self.v_max, a_max=self.a_max, j_max=self.j_max
        )
        verified = braking is not None
        if braking is not None:
            clearance_value = clearance.minimum(braking.sampled_points_frame + boundary.position[None])
            verified = clearance_value >= self.clearance_margin
        if not verified:
            braking = plan_braking_trajectory(
                boundary.velocity,
                boundary.acceleration,
                v_max=self.v_max,
                a_max=self.a_max,
                j_max=self.j_max,
                allow_limit_violation=True,
            )
        if braking is None:
            return self._decision(False, False, "no_continuous_brake")
        emergency = WorldMincoTrajectory(
            braking.coefficients_frame, [braking.duration], boundary.position, [1.0, 0.0, 0.0, 0.0], now
        )
        with self._lock:
            if self._active_plan is trajectory:
                self._active_plan = emergency
                self._verified_until = now + braking.duration
                self._fallback = None
                self._mode = "brake"
                self._terminal_goal_world = None
                self._trajectory_id += 1
        reason = "verified_emergency_brake" if verified else "uncertified_best_effort_brake"
        return self._decision(True, verified, reason)

    def sample_command(self, now: float, odom_position_world, odom_quaternion_wxyz) -> RuntimeCommand:
        """Sample the active polynomial; an expired prefix commits its stored stop."""
        if not math.isfinite(float(now)):
            raise ValueError("command time must be finite")
        odom_position = np.asarray(odom_position_world, dtype=np.float64)
        if odom_position.shape != (3,) or not np.isfinite(odom_position).all():
            raise ValueError("odometry position must be finite XYZ")
        initial_yaw = yaw_from_quaternion_wxyz(odom_quaternion_wxyz)
        with self._lock:
            if (
                self._active_plan is not None
                and self._verified_until is not None
                and now >= self._verified_until
                and self._fallback is not None
            ):
                self._active_plan = self._fallback
                self._verified_until = self._fallback.start_stamp + self._fallback.total_duration
                self._fallback = None
                self._mode = "brake"
                self._terminal_goal_world = None
                self._trajectory_id += 1
            trajectory = self._active_plan
            mode = self._mode
            hold = None if self._hold_position is None else self._hold_position.copy()
            if trajectory is None:
                position = odom_position if hold is None else hold
                velocity = acceleration = jerk = np.zeros(3, dtype=np.float64)
                mode = "terminal_hold" if mode == "terminal_hold" and self._terminal_goal_world is not None else "hold"
            else:
                sampled = trajectory.sample(now)
                position = sampled.position
                velocity = sampled.velocity
                acceleration = sampled.acceleration
                jerk = sampled.jerk
                if sampled.finished:
                    completed_terminal = mode == "terminal"
                    self._active_plan = None
                    self._verified_until = None
                    self._fallback = None
                    self._hold_position = position.copy()
                    if not completed_terminal:
                        self._terminal_goal_world = None
                    self._mode = "terminal_hold" if completed_terminal else "hold"
                    mode = self._mode
                    velocity = acceleration = jerk = np.zeros(3, dtype=np.float64)
            values = np.concatenate([position, velocity, acceleration, jerk])
            if not np.isfinite(values).all():
                raise FloatingPointError("refusing to publish a non-finite command")
            dt = (
                1.0 / self.command_rate_hz
                if self._last_command_stamp is None or now <= self._last_command_stamp
                else min(now - self._last_command_stamp, 0.1)
            )
            if self._yaw_tracker is None:
                self._yaw_tracker = VelocityYawTracker(initial_yaw, max_rate=self.yaw_rate_max)
            yaw, yaw_rate = self._yaw_tracker.step(velocity, dt)
            self._last_command_stamp = float(now)
            return RuntimeCommand(
                stamp=float(now),
                position=np.asarray(position, dtype=np.float64).copy(),
                velocity=np.asarray(velocity, dtype=np.float64).copy(),
                acceleration=np.asarray(acceleration, dtype=np.float64).copy(),
                jerk=np.asarray(jerk, dtype=np.float64).copy(),
                yaw=yaw,
                yaw_rate=yaw_rate,
                trajectory_id=self._trajectory_id,
                mode=mode,
            )


__all__ = [
    "A_MAX",
    "COMMAND_RATE_HZ",
    "CurrentScanClearance",
    "HOLD_ACCELERATION_TOLERANCE_MPS2",
    "HOLD_VELOCITY_TOLERANCE_MPS",
    "InferenceWorker",
    "InferenceWorkerError",
    "J_MAX",
    "MAX_PLANNING_LATENCY_S",
    "MAX_SENSOR_DELTA_S",
    "OdometrySample",
    "PlanDecision",
    "PlanExecutor",
    "PredictionPayload",
    "PREFIX_DURATION_S",
    "RuntimeCommand",
    "TERMINAL_ENGAGE_DISTANCE_M",
    "TrajectoryPrediction",
    "V_MAX",
    "VISIBLE_CLEARANCE_M",
    "VelocityYawTracker",
    "WorldMincoTrajectory",
    "decode_prediction",
    "inspect_trajectory_prefix",
    "plan_braking_trajectory",
    "plan_terminal_trajectory",
    "predict",
    "prediction_from_payload",
    "prediction_to_payload",
    "select_nearest_odometry",
    "yaw_from_quaternion_wxyz",
]
