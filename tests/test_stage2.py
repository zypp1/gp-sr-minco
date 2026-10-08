import copy
import json

import numpy as np
import pytest
import torch

from neural_minco.config import default_config, stage2_config
from neural_minco.data import CURRENT_LIDAR_GEOMETRY
from neural_minco.model import build_model, load_model
from neural_minco.stage2 import MapClearance, MapDataset, Stage2Loss, freeze_proposal, physical_losses, train_stage2
from neural_minco.train import Stage1Loss


def _batch():
    state = torch.tensor([[1.0, 0.1, 0.0, 0.1, 0.0, 0.0, 7.0, 1.0, 0.4]])
    t = torch.arange(1, 21) / 10
    return dict(
        obs=torch.rand(1, 1, 30, 180),
        state_body=state,
        dense_traj=torch.stack([t, t * 0, t * 0], dim=-1)[None],
        dense_velocity=torch.tensor([1.0, 0.0, 0.0]).expand(1, 20, 3),
        dense_traj_valid=torch.ones(1, 20, dtype=torch.bool),
        a_end=torch.zeros(1, 3),
        state=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        body_to_world_rotation=torch.eye(3)[None],
        map_index=torch.tensor([0]),
    )


def _esdf(value=2.0):
    return dict(sdf=torch.full((11, 11, 11), value), origin=np.array([-5.0, -5.0, -5.0]), voxel=1.0, shape=(11, 11, 11))


def test_stage2_config_preserves_scales_loss_architecture_and_source():
    source = default_config()
    source["model"]["waypoint_scale"] = 11.5
    original = copy.deepcopy(source)
    config = stage2_config(source, dynamics_weight=0.1, collision_weight=0.2)
    assert source == original
    for name in ("model", "loss", "lidar", "data"):
        assert config[name] == source[name]
    assert config["training"]["epochs"] == 10
    assert config["stage2"]["primary_checkpoint"] == "ckpt_last.pth"


def test_zero_physics_is_exact_stage1_loss_and_gradient_with_one_decode(monkeypatch):
    torch.manual_seed(17)
    config = stage2_config(default_config())
    model = build_model(config)
    batch = _batch()
    _, aux = model(batch["obs"], batch["state_body"], return_stages=True)
    expected, report = Stage1Loss()(model, aux, batch, batch["state_body"])
    parameters = list(model.parameters())
    expected_gradient = torch.autograd.grad(expected, parameters, retain_graph=True)
    criterion = Stage2Loss(config)
    calls = []
    original = criterion.decoder.forward

    def recorded(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(criterion.decoder, "forward", recorded)
    actual, terms = criterion(model, aux, batch, batch["state_body"])
    actual_gradient = torch.autograd.grad(actual, parameters)
    assert calls == [4]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name in report:
        torch.testing.assert_close(terms[name], report[name], atol=0, rtol=0)
    for a, b in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_freeze_leaves_z0_bitwise_unchanged_after_refiner_step():
    torch.manual_seed(7)
    model = build_model(default_config()).train()
    parameters = freeze_proposal(model)
    assert all(name.startswith("shared_refiner.") == value.requires_grad for name, value in model.named_parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    batch = _batch()
    _, aux = model(batch["obs"], batch["state_body"], return_stages=True)
    loss, _ = Stage2Loss(stage2_config(default_config(), dynamics_weight=0.1))(model, aux, batch, batch["state_body"])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    torch.optim.AdamW(parameters, lr=3e-5).step()
    _, after = model(batch["obs"], batch["state_body"], return_stages=True)
    for key in aux["raw_stages"][0]:
        torch.testing.assert_close(aux["raw_stages"][0][key], after["raw_stages"][0][key], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if not name.startswith("shared_refiner."):
            assert torch.equal(value, before[name]), name
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.shared_refiner.state_dict(prefix="shared_refiner.").items()
    )


def test_absolute_dynamics_catches_equal_unsafe_jerk_and_weights_refiners_equally():
    coefficients = torch.zeros(4, 1, 3, 6, 3, dtype=torch.float64)
    coefficients[:, :, :, 3, 0] = 3.0  # constant jerk 18 > 12 even with no inter-stage worsening
    coefficients.requires_grad_()
    durations = torch.full((1, 3), 0.01, dtype=torch.float64)
    decoded = [dict(coefficients=value, T=durations) for value in coefficients]
    config = stage2_config(default_config(), dynamics_weight=1.0)["stage2"]
    dynamics, collision = physical_losses(decoded, {}, config)
    assert dynamics.item() == pytest.approx(0.5)
    assert collision.item() == 0
    dynamics.backward()
    assert coefficients.grad[0].count_nonzero() == 0
    assert coefficients.grad[1].abs().sum() > 0
    torch.testing.assert_close(coefficients.grad[1], coefficients.grad[2], rtol=0, atol=0)
    torch.testing.assert_close(coefficients.grad[2], coefficients.grad[3], rtol=0, atol=0)


def test_map_selection_solid_ground_and_outside_gradient():
    clearance = MapClearance([_esdf(2.0), _esdf(-0.4)], ground_height=0.0)
    world = torch.tensor(
        [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]], [[6.0, 0.0, 1.0]]], requires_grad=True
    )
    distance = clearance(world, torch.tensor([0, 1, 0, 0]))
    torch.testing.assert_close(distance[:, 0], torch.tensor([1.0, -0.4, -1.0, -1.0]))
    torch.relu(1 - distance / 0.3).sum().backward()
    assert world.grad[2, 0, 2] < 0  # gradient descent moves up, out of solid ground
    assert world.grad[3, 0, 0] > 0  # gradient descent moves back into mapped domain


def test_collision_uses_true_minco_points_full_pose_and_each_map():
    coefficients = torch.zeros(4, 2, 3, 6, 3)
    coefficients[:, :, :, 0, 0] = 2.0
    coefficients.requires_grad_()
    durations = torch.full((2, 3), 2 / 3)
    decoded = [dict(coefficients=value, T=durations) for value in coefficients]
    batch = dict(
        state=torch.zeros(2, 13),
        body_to_world_rotation=torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]]).repeat(2, 1, 1),
        map_index=torch.tensor([0, 1]),
    )
    batch["state"][:, 2] = 1
    seen = []

    def clearance(world, ids):
        seen.append(ids.tolist())
        torch.testing.assert_close(world[..., 2], torch.full_like(world[..., 2], -1.0))
        return world[..., 2]

    config = stage2_config(default_config(), collision_weight=1.0)["stage2"]
    _, collision = physical_losses(decoded, batch, config, clearance)
    assert collision.item() == pytest.approx(1 + 1 / 0.3)
    assert seen == [[0, 1, 0, 1, 0, 1]]
    collision.backward()
    assert coefficients.grad[0].count_nonzero() == 0
    assert all(coefficients.grad[k].abs().sum() > 0 for k in (1, 2, 3))


class _TinyDataset:
    lidar_geometry = CURRENT_LIDAR_GEOMETRY

    def __init__(self):
        self.batch = _batch()
        self.episode_index = np.array([0, 0])
        self.map_id = np.array(["map", "map"])
        self.state = self.batch["state"].repeat(2, 1).numpy()

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            **{key: value[0].clone() for key, value in self.batch.items()},
            "episode_index": 0,
            "sample_index": index,
        }


def test_unknown_map_is_rejected():
    with pytest.raises(ValueError):
        MapDataset(_TinyDataset(), ["wrong"])


def test_training_writes_complete_schedule_checkpoint_and_tensorboard(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    config = stage2_config(default_config())
    config["training"].update(epochs=2, batch_size=2, num_workers=0)
    model = build_model(config)
    freeze_proposal(model)
    original = {
        name: value.clone() for name, value in model.state_dict().items() if not name.startswith("shared_refiner.")
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    run = tmp_path / "run"
    result = train_stage2(
        model, config, _TinyDataset(), _TinyDataset(), run, manifest, "initial.pth", ["map.npz"], "cpu"
    )
    assert result == str(run / "ckpt_last.pth")
    loaded, checkpoint = load_model(result)
    assert checkpoint["epoch"] == 1 and checkpoint["step"] == 2
    assert checkpoint["scheduler"]["last_epoch"] == checkpoint["scheduler"]["T_max"] == 2
    assert json.loads((run / "params.json").read_text()) == checkpoint["config"]
    for name, value in original.items():
        assert torch.equal(loaded.state_dict()[name], value)
    tags = EventAccumulator(str(run / "tensorboard")).Reload().Tags()["scalars"]
    assert "train/dynamics" in tags and "train/collision" in tags and "val/ade_m" in tags
