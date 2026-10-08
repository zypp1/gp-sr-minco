"""Differentiable M=3 minimum-jerk decoding and trajectory sampling.

Coefficients use ascending powers and shape ``(B,3,6,3)``. The interpolation
system is solved in float64 and cast back to the input dtype.
"""

import math
from typing import List, Sequence, Tuple

import torch
from torch import nn

NUM_SEGMENTS = 3
NUM_COEFFICIENTS = 6
NUM_AXES = 3


class MinimumJerkDecoder(nn.Module):
    """Decode physical body-frame boundaries into one M3 trajectory."""

    def forward(
        self,
        Q: torch.Tensor,
        T: torch.Tensor,
        p_end: torch.Tensor,
        v_end: torch.Tensor,
        state_body: torch.Tensor,
        a_end: torch.Tensor,
    ) -> torch.Tensor:
        """Return coefficients with shape ``(B,3,6,3)``.

        ``Q`` is ``(B,2,3)``, ``T`` is ``(B,3)``, and endpoint tensors are
        ``(B,3)``. ``state_body`` is the physical ``[v0,a0,goal]`` tensor with
        shape ``(B,9)``; the current position is the body-frame origin. All
        tensors must be finite, share dtype/device, and durations must be
        strictly positive.
        """
        if not isinstance(state_body, torch.Tensor):
            raise TypeError("state_body must be a torch.Tensor")
        if state_body.ndim != 2 or state_body.shape[1] != 9:
            raise ValueError("state_body must have shape (B,9) as [v0,a0,goal]")
        if state_body.device.type == "cpu" and not bool(torch.isfinite(state_body).all()):
            raise ValueError("state_body must contain only finite values")

        start = torch.zeros_like(state_body[:, :3])
        head = torch.stack((start, state_body[:, :3], state_body[:, 3:6]), dim=1)
        tail = torch.stack((p_end, v_end, a_end), dim=1)
        return solve_minimum_jerk_batch(T, Q, head, tail)


def solve_minimum_jerk_batch(T: torch.Tensor, Q: torch.Tensor, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Solve the PVA-constrained M3 minimum-jerk interpolation system.

    Args:
        T: Segment durations with shape ``(B,3)`` in seconds.
        Q: Interior positions with shape ``(B,2,3)``.
        head: Initial position, velocity, acceleration with shape ``(B,3,3)``.
        tail: Terminal position, velocity, acceleration with shape ``(B,3,3)``.

    Returns:
        Ascending fifth-order coefficients with shape ``(B,3,6,3)``.

    Inputs support float32 or float64, must share dtype/device, must be finite,
    and each duration must be strictly positive. CUDA solves explicitly use
    cuSOLVER; all solves use float64 internally for numerical stability. CPU
    calls validate tensor values immediately. CUDA callers enforce value bounds
    at their input boundary to avoid synchronizing the training hot path.
    """
    _validate_problem(T, Q, head, tail)
    batch_size = T.shape[0]
    output_dtype = T.dtype
    device = T.device

    durations = T.double()
    waypoints = Q.double()
    head = head.double()
    tail = tail.double()
    t2 = durations.square()
    t3 = t2 * durations
    t4 = t2.square()
    t5 = t4 * durations

    dimension = NUM_SEGMENTS * NUM_COEFFICIENTS
    matrix = torch.zeros(batch_size, dimension, dimension, dtype=torch.float64, device=device)
    rhs = torch.zeros(batch_size, dimension, NUM_AXES, dtype=torch.float64, device=device)

    # Initial position, velocity, and acceleration at local time zero.
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = 2.0
    rhs[:, :3] = head

    for segment in range(NUM_SEGMENTS - 1):
        base = NUM_COEFFICIENTS * segment
        t = durations[:, segment]
        t_sq = t2[:, segment]
        t_cu = t3[:, segment]
        t_four = t4[:, segment]
        t_five = t5[:, segment]

        # Natural minimum-jerk junction conditions: jerk and snap continuity.
        matrix[:, base + 3, base + 3] = 6.0
        matrix[:, base + 3, base + 4] = 24.0 * t
        matrix[:, base + 3, base + 5] = 60.0 * t_sq
        matrix[:, base + 3, base + 9] = -6.0
        matrix[:, base + 4, base + 4] = 24.0
        matrix[:, base + 4, base + 5] = 120.0 * t
        matrix[:, base + 4, base + 10] = -24.0

        # Segment endpoint equals Q; position, velocity, acceleration continue.
        end_terms = (1.0, t, t_sq, t_cu, t_four, t_five)
        for power, value in enumerate(end_terms):
            matrix[:, base + 5, base + power] = value
            matrix[:, base + 6, base + power] = value
        matrix[:, base + 6, base + 6] = -1.0

        matrix[:, base + 7, base + 1] = 1.0
        matrix[:, base + 7, base + 2] = 2.0 * t
        matrix[:, base + 7, base + 3] = 3.0 * t_sq
        matrix[:, base + 7, base + 4] = 4.0 * t_cu
        matrix[:, base + 7, base + 5] = 5.0 * t_four
        matrix[:, base + 7, base + 7] = -1.0

        matrix[:, base + 8, base + 2] = 2.0
        matrix[:, base + 8, base + 3] = 6.0 * t
        matrix[:, base + 8, base + 4] = 12.0 * t_sq
        matrix[:, base + 8, base + 5] = 20.0 * t_cu
        matrix[:, base + 8, base + 8] = -2.0
        rhs[:, base + 5] = waypoints[:, segment]

    # Terminal position, velocity, and acceleration.
    offset = NUM_COEFFICIENTS * (NUM_SEGMENTS - 1)
    t = durations[:, -1]
    t_sq = t2[:, -1]
    t_cu = t3[:, -1]
    t_four = t4[:, -1]
    t_five = t5[:, -1]
    end_terms = (1.0, t, t_sq, t_cu, t_four, t_five)
    for power, value in enumerate(end_terms):
        matrix[:, -3, offset + power] = value
    matrix[:, -2, offset + 1] = 1.0
    matrix[:, -2, offset + 2] = 2.0 * t
    matrix[:, -2, offset + 3] = 3.0 * t_sq
    matrix[:, -2, offset + 4] = 4.0 * t_cu
    matrix[:, -2, offset + 5] = 5.0 * t_four
    matrix[:, -1, offset + 2] = 2.0
    matrix[:, -1, offset + 3] = 6.0 * t
    matrix[:, -1, offset + 4] = 12.0 * t_sq
    matrix[:, -1, offset + 5] = 20.0 * t_cu
    rhs[:, -3:] = tail

    if device.type == "cuda":
        torch.backends.cuda.preferred_linalg_library("cusolver")
    coefficients = torch.linalg.solve(matrix, rhs)
    if device.type == "cpu" and not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError(
            "minimum-jerk solve produced non-finite coefficients; check durations and boundary magnitudes"
        )
    return coefficients.reshape(batch_size, NUM_SEGMENTS, NUM_COEFFICIENTS, NUM_AXES).to(output_dtype)


def _validate_problem(T: torch.Tensor, Q: torch.Tensor, head: torch.Tensor, tail: torch.Tensor) -> None:
    values = {"T": T, "Q": Q, "head": head, "tail": tail}
    if not all(isinstance(value, torch.Tensor) for value in values.values()):
        raise TypeError("T, Q, head, and tail must be torch.Tensor objects")
    if T.ndim != 2:
        raise ValueError("T must have shape (B,3)")
    batch_size = T.shape[0]
    expected = {
        "T": (batch_size, NUM_SEGMENTS),
        "Q": (batch_size, NUM_SEGMENTS - 1, NUM_AXES),
        "head": (batch_size, 3, NUM_AXES),
        "tail": (batch_size, 3, NUM_AXES),
    }
    for name, value in values.items():
        if value.shape != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(value.shape)}")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use float32 or float64")
        if value.dtype != T.dtype or value.device != T.device:
            raise ValueError("T, Q, head, and tail must share dtype and device")
        if value.device.type == "cpu" and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    if batch_size < 1:
        raise ValueError("minimum-jerk solving requires a non-empty batch")
    if T.device.type == "cpu" and not bool((T > 0.0).all()):
        raise ValueError("all segment durations must be strictly positive")


def poly_eval(coeffs: torch.Tensor, t: torch.Tensor, deriv: int = 0) -> torch.Tensor:
    """Evaluate a derivative of ascending-power polynomial coefficients."""
    if not isinstance(coeffs, torch.Tensor) or not isinstance(t, torch.Tensor):
        raise TypeError("coeffs and t must be torch.Tensor objects")
    if coeffs.ndim < 1 or coeffs.shape[-1] < 1:
        raise ValueError("coeffs must have a non-empty coefficient axis")
    if coeffs.dtype != t.dtype or coeffs.device != t.device:
        raise ValueError("coeffs and t must share dtype and device")
    if coeffs.device.type == "cpu" and not bool(torch.isfinite(coeffs).all() & torch.isfinite(t).all()):
        raise ValueError("coeffs and t must contain only finite values")
    if isinstance(deriv, bool) or not isinstance(deriv, int) or deriv < 0:
        raise ValueError("deriv must be a non-negative integer")
    return _poly_eval(coeffs, t, deriv)


def _poly_eval(coeffs: torch.Tensor, t: torch.Tensor, deriv: int) -> torch.Tensor:
    result = torch.zeros(coeffs.shape[:-1], dtype=coeffs.dtype, device=coeffs.device)
    t_power = torch.ones_like(t)
    for power in range(deriv, coeffs.shape[-1]):
        factor = math.prod(range(power - deriv + 1, power + 1))
        result = result + coeffs[..., power] * factor * t_power
        t_power = t_power * t
    return result


def sample_trajectory_batch(coeffs: torch.Tensor, T: torch.Tensor, n_samples: int, deriv: int = 0) -> torch.Tensor:
    """Uniformly sample from time zero through the final endpoint."""
    segment_coeffs, local_time = _prepare_uniform_samples(coeffs, T, n_samples)
    _validate_derivative(deriv)
    return _poly_eval(segment_coeffs, local_time, deriv)


def sample_future_trajectory_batch(
    coeffs: torch.Tensor, T: torch.Tensor, n_samples: int, deriv: int = 0
) -> torch.Tensor:
    """Uniformly sample future points while excluding time zero."""
    _validate_sample_count(n_samples)
    return sample_trajectory_batch(coeffs, T, n_samples + 1, deriv=deriv)[:, 1:]


def sample_trajectory_at_times_batch(
    coeffs: torch.Tensor, T: torch.Tensor, times: torch.Tensor, deriv: int = 0
) -> torch.Tensor:
    """Sample ``(B,3,6,3)`` trajectories at elapsed times ``(B,S)``.

    Times are clamped to each trajectory's closed time interval. The returned
    tensor has shape ``(B,S,3)``.
    """
    batch_size = _validate_trajectory(coeffs, T)
    if not isinstance(times, torch.Tensor):
        raise TypeError("times must be a torch.Tensor")
    if times.ndim != 2 or times.shape[0] != batch_size:
        raise ValueError("times must have shape (B,S) matching coeffs")
    if times.dtype != coeffs.dtype or times.device != coeffs.device:
        raise ValueError("times, coeffs, and T must share dtype and device")
    if times.device.type == "cpu" and not bool(torch.isfinite(times).all()):
        raise ValueError("times must contain only finite values")
    _validate_derivative(deriv)

    segment_ends = torch.cumsum(T, dim=1)
    times = torch.minimum(times.clamp(min=0.0), segment_ends[:, -1:])
    segment_index = torch.searchsorted(segment_ends.contiguous(), times.contiguous()).clamp(max=NUM_SEGMENTS - 1)
    local_time = times - torch.gather(segment_ends - T, 1, segment_index)
    segment_coeffs = _gather_segments(coeffs, segment_index)
    return _poly_eval(segment_coeffs, local_time[..., None], deriv)


def sample_trajectory_batch_with_derivatives(
    coeffs: torch.Tensor, T: torch.Tensor, n_samples: int, derivs: Sequence[int] = (0, 1, 2, 3)
) -> List[torch.Tensor]:
    """Uniformly sample several derivatives with one segment lookup."""
    if not derivs:
        raise ValueError("derivs must contain at least one derivative order")
    for deriv in derivs:
        _validate_derivative(deriv)
    segment_coeffs, local_time = _prepare_uniform_samples(coeffs, T, n_samples)
    return [_poly_eval(segment_coeffs, local_time, deriv) for deriv in derivs]


def _prepare_uniform_samples(
    coeffs: torch.Tensor, T: torch.Tensor, n_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_trajectory(coeffs, T)
    _validate_sample_count(n_samples)
    segment_ends = torch.cumsum(T, dim=1)
    unit_time = torch.linspace(0.0, 1.0, n_samples, dtype=coeffs.dtype, device=coeffs.device)
    times = unit_time[None] * segment_ends[:, -1:]
    segment_index = torch.searchsorted(segment_ends.contiguous(), times.contiguous()).clamp(max=NUM_SEGMENTS - 1)
    local_time = times - torch.gather(segment_ends - T, 1, segment_index)
    return _gather_segments(coeffs, segment_index), local_time[..., None]


def _validate_trajectory(coeffs: torch.Tensor, T: torch.Tensor) -> int:
    if not isinstance(coeffs, torch.Tensor) or not isinstance(T, torch.Tensor):
        raise TypeError("coeffs and T must be torch.Tensor objects")
    if coeffs.ndim != 4 or coeffs.shape[1:] != (NUM_SEGMENTS, NUM_COEFFICIENTS, NUM_AXES):
        raise ValueError("coeffs must have shape (B,3,6,3)")
    batch_size = coeffs.shape[0]
    if T.shape != (batch_size, NUM_SEGMENTS):
        raise ValueError("T must have shape (B,3) matching coeffs")
    if coeffs.dtype not in (torch.float32, torch.float64):
        raise TypeError("coeffs and T must use float32 or float64")
    if coeffs.dtype != T.dtype or coeffs.device != T.device:
        raise ValueError("coeffs and T must share dtype and device")
    if coeffs.device.type == "cpu" and not bool(torch.isfinite(coeffs).all() & torch.isfinite(T).all()):
        raise ValueError("coeffs and T must contain only finite values")
    if T.device.type == "cpu" and not bool((T > 0.0).all()):
        raise ValueError("all segment durations must be strictly positive")
    return batch_size


def _validate_sample_count(n_samples: int) -> None:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")


def _validate_derivative(deriv: int) -> None:
    if isinstance(deriv, bool) or not isinstance(deriv, int) or deriv < 0 or deriv >= NUM_COEFFICIENTS:
        raise ValueError("deriv must be an integer in [0,5]")


def _gather_segments(coeffs: torch.Tensor, segment_index: torch.Tensor) -> torch.Tensor:
    batch_size, sample_count = segment_index.shape
    gather_index = segment_index[:, :, None, None].expand(batch_size, sample_count, NUM_COEFFICIENTS, NUM_AXES)
    return torch.gather(coeffs, 1, gather_index).transpose(2, 3)
