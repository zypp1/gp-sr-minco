import numpy as np
import pytest
import torch

from neural_minco.config import default_config
from neural_minco.data import CURRENT_LIDAR_GEOMETRY
from neural_minco.evaluate import evaluate_model, lidar_clearance, minco_derivative_maxima
from neural_minco.lidar import get_ray_geometry
from neural_minco.model import build_model
from neural_minco.runtime import predict


def test_predict_returns_fixed_two_second_m3_trajectory():
    model = build_model().eval()
    observation = torch.ones(1, 1, 30, 180)
    state_body = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]])
    prediction = predict(model, observation, state_body, samples=20)

    assert prediction.waypoints.shape == (1, 2, 3)
    assert prediction.coefficients.shape == (1, 3, 6, 3)
    assert prediction.position.shape == (1, 20, 3)
    torch.testing.assert_close(prediction.durations.sum(dim=1), torch.tensor([2.0]))
    maxima = minco_derivative_maxima(prediction.coefficients, prediction.durations, samples_per_segment=33)
    assert maxima.shape == (1, 3)
    assert torch.isfinite(maxima).all()


def test_visible_clearance_ignores_free_rays_and_detects_hit():
    direction, _, _ = get_ray_geometry()
    directions = torch.from_numpy(direction).float()
    observation = torch.ones(1, 1, 30, 180)
    ray = 15 * 180 + 90
    observation.reshape(1, 1, -1)[0, 0, ray] = 0.25
    hit = 5.0 * directions[ray]
    points = torch.stack((hit, hit + torch.tensor([1.0, 0.0, 0.0])))[None]

    clearance = lidar_clearance(points, observation, directions, 20.0)
    assert torch.isclose(clearance[0, 0], torch.tensor(0.0), atol=1e-5)
    assert clearance[0, 1] > 0.5
    free = lidar_clearance(points, torch.ones_like(observation), directions, 20.0)
    np.testing.assert_allclose(free.numpy(), 40.0)


class _MapIdentityDataset:
    def __init__(self, map_ids):
        self.map_id = np.asarray(map_ids)
        self.lidar_geometry = CURRENT_LIDAR_GEOMETRY.copy()

    def __len__(self):
        return len(self.map_id)


@pytest.mark.parametrize("map_ids", (("different_map",), ("expected_map", "second_map")))
def test_map_evaluation_rejects_dataset_identity_mismatch(tmp_path, map_ids):
    with pytest.raises(ValueError, match="does not match"):
        evaluate_model(
            torch.nn.Identity(), _MapIdentityDataset(map_ids), device="cpu", map_path=str(tmp_path / "expected_map.npz")
        )


def test_evaluation_rejects_checkpoint_lidar_range_mismatch_before_forward():
    model = torch.nn.Identity()
    model.lidar_config = default_config()["lidar"]
    model.lidar_config["r_max"] = 30.0
    dataset = _MapIdentityDataset(("expected_map",))

    with pytest.raises(ValueError, match="LiDAR geometry disagrees"):
        evaluate_model(model, dataset, device="cpu")
