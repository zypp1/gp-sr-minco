import pytest
import torch

from neural_minco.minco import (
    MinimumJerkDecoder,
    poly_eval,
    sample_future_trajectory_batch,
    sample_trajectory_at_times_batch,
    sample_trajectory_batch,
    sample_trajectory_batch_with_derivatives,
)


def _state(dtype=torch.float64):
    state = torch.zeros(1, 9, dtype=dtype)
    state[:, :3] = torch.tensor([1.2, -0.3, 0.4], dtype=dtype)
    state[:, 3:6] = torch.tensor([0.2, 0.1, -0.15], dtype=dtype)
    return state


def _inputs(dtype=torch.float64):
    return {
        "Q": torch.tensor([[[0.7, 0.1, 0.2], [1.8, -0.2, 0.5]]], dtype=dtype),
        "T": torch.tensor([[0.5, 0.8, 0.7]], dtype=dtype),
        "p_end": torch.tensor([[2.8, 0.0, 0.7]], dtype=dtype),
        "v_end": torch.tensor([[0.5, 0.0, 0.1]], dtype=dtype),
        "state_body": _state(dtype),
        "a_end": torch.tensor([[0.1, -0.1, 0.0]], dtype=dtype),
    }


def _segment_end(coefficients, duration, derivative):
    return poly_eval(coefficients.transpose(1, 2), duration[:, None], derivative)


def test_m3_solution_matches_boundaries_and_internal_continuity():
    inputs = _inputs()
    coefficients = MinimumJerkDecoder()(**inputs)
    assert coefficients.shape == (1, 3, 6, 3)

    for segment in range(2):
        assert torch.allclose(
            _segment_end(coefficients[:, segment], inputs["T"][:, segment], 0), inputs["Q"][:, segment], atol=1e-9
        )
        right = coefficients[:, segment + 1].transpose(1, 2)
        for derivative in range(5):
            assert torch.allclose(
                _segment_end(coefficients[:, segment], inputs["T"][:, segment], derivative),
                poly_eval(right, torch.zeros(1, 1, dtype=torch.float64), derivative),
                atol=2e-8,
            )

    start = coefficients[:, 0].transpose(1, 2)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    assert torch.equal(poly_eval(start, zero, 0), torch.zeros_like(inputs["p_end"]))
    assert torch.allclose(poly_eval(start, zero, 1), inputs["state_body"][:, :3])
    assert torch.allclose(poly_eval(start, zero, 2), inputs["state_body"][:, 3:6])
    assert torch.allclose(_segment_end(coefficients[:, -1], inputs["T"][:, -1], 0), inputs["p_end"], atol=1e-9)
    assert torch.allclose(_segment_end(coefficients[:, -1], inputs["T"][:, -1], 1), inputs["v_end"], atol=1e-9)
    assert torch.allclose(_segment_end(coefficients[:, -1], inputs["T"][:, -1], 2), inputs["a_end"], atol=1e-9)


def test_minimum_jerk_backpropagates_through_all_outputs():
    state = _state(torch.float32)
    waypoints = torch.tensor([[[0.8, 0.2, 0.1], [1.7, -0.1, 0.4]]], requires_grad=True)
    logits = torch.zeros(1, 3, requires_grad=True)
    endpoint = torch.tensor([[2.6, 0.1, 0.6]], requires_grad=True)
    velocity_end = torch.tensor([[0.5, 0.0, 0.1]], requires_grad=True)
    acceleration_end = torch.tensor([[0.1, 0.0, 0.0]], requires_grad=True)
    duration = 2.0 * torch.softmax(logits, dim=-1)
    coefficients = MinimumJerkDecoder()(waypoints, duration, endpoint, velocity_end, state, acceleration_end)
    points = sample_trajectory_at_times_batch(coefficients, duration, torch.linspace(0.1, 2.0, 20)[None])
    points.square().mean().backward()
    for value in (waypoints, logits, endpoint, velocity_end, acceleration_end):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_uniform_sampling_contract_and_shared_derivative_lookup():
    inputs = _inputs(torch.float32)
    coefficients = MinimumJerkDecoder()(**inputs)
    including_start = sample_trajectory_batch(coefficients, inputs["T"], n_samples=21)
    future = sample_future_trajectory_batch(coefficients, inputs["T"], n_samples=20)
    assert torch.equal(future, including_start[:, 1:])
    assert torch.equal(including_start[:, 0], torch.zeros_like(inputs["p_end"]))
    assert torch.allclose(future[:, -1], inputs["p_end"], atol=2e-5)

    derivatives = sample_trajectory_batch_with_derivatives(coefficients, inputs["T"], 21)
    assert len(derivatives) == 4
    assert all(value.shape == (1, 21, 3) for value in derivatives)
    assert all(torch.isfinite(value).all() for value in derivatives)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("T", torch.tensor([[0.5, 0.0, 1.5]]), "strictly positive"),
        ("T", torch.tensor([[0.5, float("nan"), 1.5]]), "finite"),
        ("Q", torch.tensor([[[0.7, 0.1, float("inf")], [1.8, -0.2, 0.5]]]), "finite"),
    ],
)
def test_decoder_rejects_invalid_cpu_inputs(field, value, message):
    inputs = _inputs(torch.float32)
    inputs[field] = value
    with pytest.raises(ValueError, match=message):
        MinimumJerkDecoder()(**inputs)


def test_decoder_rejects_non_m3_shapes():
    inputs = _inputs(torch.float32)
    inputs["T"] = torch.full((1, 4), 0.5)
    with pytest.raises(ValueError, match="shape"):
        MinimumJerkDecoder()(**inputs)

    inputs = _inputs(torch.float32)
    inputs["Q"] = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match="shape"):
        MinimumJerkDecoder()(**inputs)
