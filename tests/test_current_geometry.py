"""Focused tests for the current shared frame and ESDF geometry."""

import math

import numpy as np
import pytest
import torch

from neural_minco.geometry import (
    body_to_world,
    body_to_world_numpy,
    build_esdf,
    quaternion_matrix,
    quaternion_matrix_numpy,
    query_esdf,
    world_to_body,
    world_to_body_numpy,
)


def _quaternion_and_rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    quaternion = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]])
    ry = np.asarray(
        [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]]
    )
    rz = np.asarray([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return quaternion, rz @ ry @ rx


def test_quaternion_matrices_use_full_roll_pitch_yaw_and_normalize():
    quaternion, expected = _quaternion_and_rotation(0.4, -0.3, 0.7)
    scaled = 3.5 * quaternion

    np.testing.assert_allclose(quaternion_matrix_numpy(scaled), expected, atol=1e-12)
    batched = quaternion_matrix_numpy(np.stack([scaled, -scaled]))
    np.testing.assert_allclose(batched[0], expected, atol=1e-12)
    np.testing.assert_allclose(batched[1], expected, atol=1e-12)

    torch_matrix = quaternion_matrix(torch.tensor(np.stack([scaled, -scaled]), dtype=torch.float64))
    torch.testing.assert_close(
        torch_matrix, torch.tensor(np.stack([expected, expected]), dtype=torch.float64), rtol=0.0, atol=1e-12
    )


def test_torch_and_numpy_body_world_round_trips_match():
    quaternion, rotation = _quaternion_and_rotation(-0.5, 0.2, 0.8)
    body = np.asarray([[1.0, -2.0, 0.5], [-0.4, 0.7, 1.3]], dtype=np.float64)
    origin = np.asarray([4.0, -3.0, 2.0], dtype=np.float64)
    expected_world = body @ rotation.T + origin

    world_numpy = body_to_world_numpy(body, origin, quaternion)
    np.testing.assert_allclose(world_numpy, expected_world, atol=1e-12)
    np.testing.assert_allclose(world_to_body_numpy(world_numpy - origin, quaternion), body, atol=1e-12)

    body_torch = torch.tensor(body[None], dtype=torch.float64)
    origin_torch = torch.tensor(origin[None], dtype=torch.float64)
    quaternion_torch = torch.tensor(quaternion[None], dtype=torch.float64)
    world_torch = body_to_world(body_torch, origin_torch, quaternion_torch)
    torch.testing.assert_close(
        world_torch, torch.tensor(expected_world[None], dtype=torch.float64), rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        world_to_body(world_torch - origin_torch[:, None], quaternion_torch), body_torch, rtol=0.0, atol=1e-12
    )


def test_frame_transforms_reject_zero_quaternions():
    with pytest.raises(ValueError, match="norm"):
        quaternion_matrix(torch.zeros(1, 4))
    with pytest.raises(ValueError, match="norm"):
        quaternion_matrix_numpy(np.zeros(4))


def test_esdf_query_is_signed_differentiable_and_finite_outside_grid():
    cloud = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    esdf = build_esdf(cloud, voxel=0.25, pad=1.0)
    query = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.75, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)

    distance = query_esdf(esdf, query)

    assert distance.shape == (3,)
    assert distance[0] < 0.0
    assert distance[1] > 0.0
    assert distance[2] > distance[1]
    distance.sum().backward()
    assert torch.isfinite(query.grad).all()


def test_esdf_grid_finiteness_is_checked_once_per_cache_entry(monkeypatch):
    esdf = build_esdf(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    sdf = esdf["sdf"]
    original_isfinite = torch.isfinite
    grid_checks = 0

    def tracked_isfinite(value):
        nonlocal grid_checks
        if value.shape == sdf.shape and value.data_ptr() == sdf.data_ptr():
            grid_checks += 1
        return original_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", tracked_isfinite)
    query = torch.zeros(1, 3, dtype=sdf.dtype)
    query_esdf(esdf, query)
    query_esdf(esdf, query)
    assert grid_checks == 1


def test_esdf_invalid_grid_is_rejected_on_first_query():
    esdf = build_esdf(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    esdf["sdf"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid ESDF grid"):
        query_esdf(esdf, torch.zeros(1, 3))
