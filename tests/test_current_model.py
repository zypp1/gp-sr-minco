import math

import pytest
import torch

from neural_minco.config import default_config
from neural_minco.model import TrajectoryAttentionPlanner, _update_action, build_model, load_model


def _inputs(batch=1):
    observation = torch.rand(batch, 1, 30, 180)
    state_body = torch.tensor([[1.2, -0.3, 0.1, 0.2, 0.0, -0.1, 8.0, 2.0, 0.5]]).expand(batch, -1)
    return observation, state_body


def _rotate_xy(value, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = value.clone()
    rotated[..., 0] = cosine * value[..., 0] - sine * value[..., 1]
    rotated[..., 1] = sine * value[..., 0] + cosine * value[..., 1]
    return rotated


def test_current_structure_and_memory_shapes():
    model = build_model().eval()
    observation, _ = _inputs()
    with torch.no_grad():
        memory = model.encoder(observation)
    assert memory["local_tokens"].shape == (1, 30 * 180, 256)
    assert memory["global_tokens"].shape == (1, 15 * 90, 256)
    assert len(model.global_proposal.decoder_layers) == 3
    assert len({id(layer) for layer in model.global_proposal.decoder_layers}) == 3
    assert model.refiner_windows == ((9, 17), (5, 9), (3, 5))
    assert model.shared_refiner.proxy_samples == 10
    assert model.shared_refiner.local_attention.position_role.shape == (10, 256)
    assert model.reference_times.shape == (20,)
    assert not hasattr(model, "decoder")
    assert not hasattr(model.shared_refiner, "trajectory_encoder")
    for head in (model.global_proposal.output_head, model.shared_refiner.output_head):
        assert isinstance(head, torch.nn.Sequential)
        assert list(head.state_dict()) == ["0.weight", "2.weight"]
        assert head[0].weight.shape == (256, 256)
        assert head[2].weight.shape == (18, 256)


def test_refiner_uses_one_query_over_ten_fresh_position_windows():
    model = build_model().eval()
    # Exercise nonzero corrections so reuse of a stale support cannot pass.
    torch.nn.init.normal_(model.shared_refiner.output_head[-1].weight, std=1e-3)
    calls = []
    positions = []
    state_calls = []
    actions = []
    corrections = []

    def capture(_module, args):
        query, position, tangent, descriptor, memory, window = args
        index, _, _, _ = model.shared_refiner.local_attention._window_indices(
            position, tangent, memory["ray_elevation"], window
        )
        calls.append(
            (
                tuple(query.shape),
                tuple(position.shape),
                tuple(descriptor.shape),
                position.shape[1] * index.shape[-1],
                window,
            )
        )
        positions.append(position.clone())

    hooks = [
        model.shared_refiner.local_attention.register_forward_pre_hook(capture),
        model.state_encoder.register_forward_hook(lambda *_args: state_calls.append(1)),
        model.query_encoder.register_forward_pre_hook(lambda _module, args: actions.append(args[0].clone())),
        model.global_proposal.register_forward_hook(lambda _module, _args, output: corrections.append(output)),
        model.shared_refiner.register_forward_hook(lambda _module, _args, output: corrections.append(output)),
    ]
    with torch.inference_mode():
        model(*_inputs(batch=2))
    for hook in hooks:
        hook.remove()

    assert state_calls == [1]
    assert calls == [
        ((2, 1, 256), (2, 10, 3), (2, 10, 7), 1530, (9, 17)),
        ((2, 1, 256), (2, 10, 3), (2, 10, 7), 450, (5, 9)),
        ((2, 1, 256), (2, 10, 3), (2, 10, 7), 150, (3, 5)),
    ]
    assert len(actions) == len(corrections) == 4
    for index in range(3):
        expected = _update_action(actions[index], corrections[index], model.ratio_correction_step)
        torch.testing.assert_close(actions[index + 1], expected, rtol=0, atol=0)
        if index:
            assert not torch.equal(positions[index - 1], positions[index])


def test_local_attention_normalizes_jointly_over_all_position_window_pairs():
    model = build_model().eval()
    probabilities = []
    hook = model.shared_refiner.local_attention.dropout.register_forward_hook(
        lambda _module, _args, output: probabilities.append(output.detach())
    )
    with torch.inference_mode():
        model(*_inputs())
    hook.remove()
    attention = model.shared_refiner.local_attention
    assert attention.output.in_features == attention.output.out_features == 256
    assert len(probabilities) == 3
    for probability, key_count in zip(probabilities, (1530, 450, 150)):
        assert probability.shape == (1, 8, 1, key_count)
        torch.testing.assert_close(probability.sum(dim=-1), torch.ones(1, 8, 1))


def test_hermite_proxy_matches_terminal_position_and_velocity():
    model = build_model().eval()
    action = torch.zeros(2, 18)
    action[:, :9] = torch.tensor([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, -0.8, 0.9])
    action[:, 9:12] = torch.tensor([0.2, -0.1, 0.05])
    action[:, 15:18] = torch.tensor([0.2, -0.3, 0.1])
    state = torch.zeros(2, 9)
    state[:, :3] = torch.tensor([1.0, 0.2, -0.1])
    position, tangent, descriptor = model.shared_refiner.position_context(action, state)
    assert position.shape == tangent.shape == (2, 10, 3)
    assert descriptor.shape == (2, 10, 7)
    torch.testing.assert_close(position[:, -1], model.waypoint_scale * action[:, 6:9], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tangent[:, -1], model.v_end_scale * action[:, 9:12], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(descriptor[:, :, -1], (torch.arange(1, 11) / 10.0)[None].expand(2, -1))


def test_hermite_proxy_reproduces_constant_velocity_reference():
    model = build_model().eval()
    _, state = _inputs(batch=2)
    action = model.reference(state)
    position, tangent, _ = model.shared_refiner.position_context(action, state)
    times = torch.arange(1, 11) * 0.2
    torch.testing.assert_close(position, times[None, :, None] * state[:, None, :3])
    torch.testing.assert_close(tangent, state[:, None, :3].expand(-1, 10, -1))


def test_forward_accepts_physical_state_and_returns_four_stages():
    model = build_model().eval()
    observation, state_body = _inputs()
    with torch.no_grad():
        raw, auxiliary = model(observation, state_body, return_stages=True)
        final_only = model(observation, state_body)
    assert len(auxiliary["raw_stages"]) == 4
    assert auxiliary["correction_norms"].shape == (1, 4)
    assert raw is auxiliary["raw_stages"][-1]
    assert set(raw) == {"Q", "p_end", "v_end", "a_end", "ratio_logits"}
    assert raw["Q"].shape == (1, 2, 3)
    for name in ("p_end", "v_end", "a_end", "ratio_logits"):
        assert raw[name].shape == (1, 3)
    for name in raw:
        torch.testing.assert_close(raw[name], final_only[name], rtol=0, atol=0)
    assert torch.isfinite(auxiliary["correction_norms"]).all()
    assert torch.count_nonzero(auxiliary["correction_norms"][:, 1:]) == 0


def test_clean_checkpoint_round_trip_is_strict(tmp_path):
    config = default_config()
    source = build_model(config).eval()
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"config": config, "state_dict": source.state_dict()}, checkpoint_path)
    loaded, checkpoint = load_model(str(checkpoint_path))
    assert isinstance(loaded, TrajectoryAttentionPlanner)
    assert checkpoint["config"] == config
    assert source.state_dict().keys() == loaded.state_dict().keys()
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(expected, loaded.state_dict()[name], rtol=0, atol=0)
    inputs = _inputs()
    with torch.inference_mode():
        expected = source(*inputs)
        actual = loaded(*inputs)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_strict_loading_rejects_incomplete_or_extra_weights(tmp_path, mutation):
    config = default_config()
    weights = build_model(config).state_dict()
    if mutation == "missing":
        del weights["global_proposal.output_head.2.weight"]
    else:
        weights["unused.weight"] = torch.zeros(1)
    path = tmp_path / "invalid.pth"
    torch.save({"config": config, "state_dict": weights}, path)
    with pytest.raises(RuntimeError, match="state_dict"):
        load_model(str(path))


def test_frozen_h10_metadata_defaults_preserve_initialization_and_forward(tmp_path):
    explicit = default_config()
    torch.manual_seed(17)
    source = build_model(explicit).eval()
    rng_after = torch.get_rng_state()
    config = default_config()
    # These two fields did not exist when the frozen H10 was trained.
    del config["model"]["action_readout"]
    del config["model"]["refiner_query_mode"]
    torch.manual_seed(17)
    implicit = build_model(config).eval()
    assert torch.equal(torch.get_rng_state(), rng_after)
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(implicit.state_dict()[name], expected, rtol=0, atol=0)
    path = tmp_path / "h10.pth"
    torch.save({"config": config, "state_dict": source.state_dict()}, path)
    loaded, _ = load_model(str(path))
    inputs = _inputs()
    with torch.inference_mode():
        expected = source(*inputs)
        actual = loaded(*inputs)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("action_readout", "position_conditioned"),
        ("action_readout", "unknown"),
        ("refiner_query_mode", "carry"),
        ("refiner_query_mode", "carry_detached"),
        ("refiner_attention_normalization", "per_position"),
        ("refiner_attention_normalization", "per_position_ordered"),
        ("refiner_position_scheme", "action_positions"),
        ("refiner_position_scheme", "hermite20"),
    ),
)
def test_unsupported_experiment_metadata_is_rejected_before_weights(tmp_path, key, value):
    config = default_config()
    config["model"][key] = value
    path = tmp_path / "unsupported.pth"
    # Mode validation must precede weight loading: shapes alone cannot detect
    # carry/per-position semantics.
    torch.save({"config": config, "state_dict": {}}, path)
    with pytest.raises(ValueError, match=key):
        load_model(str(path))


def test_missing_support_metadata_is_not_silently_reinterpreted():
    config = default_config()
    del config["model"]["refiner_position_scheme"]
    with pytest.raises(ValueError, match="refiner_position_scheme"):
        build_model(config)


def test_config_is_independent_and_dimensions_are_checked():
    config = default_config()
    assert set(config) == {"model", "lidar", "data", "loss", "training"}
    config["model"]["state_dim"] = 64
    assert default_config()["model"]["state_dim"] == 128
    with pytest.raises(ValueError, match="state_dim"):
        build_model(config)


def test_integer_column_yaw_equivariance():
    torch.manual_seed(20260904)
    model = build_model().eval()
    observation = torch.rand(1, 1, 30, 180)
    state_body = torch.tensor([[1.2, -0.3, 0.1, 0.2, 0.1, -0.1, 8.0, 0.0, 0.5]])
    column_shift = 12
    angle = 2.0 * math.pi * column_shift / observation.shape[-1]
    rotated_observation = torch.roll(observation, shifts=column_shift, dims=-1)
    rotated_state = _rotate_xy(state_body.reshape(1, 3, 3), angle).reshape(1, 9)
    with torch.inference_mode():
        reference = model(observation, state_body)
        rotated = model(rotated_observation, rotated_state)
    for name in ("Q", "p_end", "v_end", "a_end"):
        torch.testing.assert_close(rotated[name], _rotate_xy(reference[name], angle), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(rotated["ratio_logits"], reference["ratio_logits"], rtol=2e-5, atol=2e-6)


def test_forward_rejects_nonfinite_or_unnormalized_inputs():
    model = build_model().eval()
    observation, state_body = _inputs()
    invalid_observations = (
        observation.clone().fill_(float("nan")),
        observation.clone().fill_(-0.01),
        observation.clone().fill_(1.01),
    )
    for invalid in invalid_observations:
        with pytest.raises(ValueError, match="finite and normalized"):
            model(invalid, state_body)
    invalid_state = state_body.clone()
    invalid_state[0, 0] = float("inf")
    with pytest.raises(ValueError, match="physical state must be finite"):
        model(observation, invalid_state)


def test_refiner_is_shared_with_detached_actions_and_no_minco(monkeypatch):
    from neural_minco.minco import MinimumJerkDecoder

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the network must not solve MINCO")

    monkeypatch.setattr(MinimumJerkDecoder, "forward", forbidden)
    model = build_model()
    torch.nn.init.normal_(model.shared_refiner.output_head[-1].weight, std=1e-3)
    calls = []
    hook = model.shared_refiner.register_forward_pre_hook(
        lambda module, args: calls.append((id(module), args[1].requires_grad, args[4]))
    )
    raw = model(*_inputs())
    hook.remove()
    sum(value.square().sum() for value in raw.values()).backward()
    assert calls == [(id(model.shared_refiner), False, window) for window in model.refiner_windows]
    assert model.global_proposal.decoder_layers[0].cross_attention.in_proj_weight.grad is None
    assert model.global_proposal.output_head[-1].weight.grad is None
    assert torch.isfinite(model.shared_refiner.output_head[-1].weight.grad).all()
