"""Tests for RangeImageRenderer: single-channel, FOV-cropped range image [1, V_fov, H].

Single channel (normalized range); elevation rows outside the sensor FOV are cropped away.
"""

import numpy as np
import pytest
import torch

from neural_minco.lidar import RangeImageRenderer, get_ray_geometry


def test_shape_is_single_channel_fov_cropped():
    R = RangeImageRenderer(r_max=10.0)
    img = R(np.array([[5.0, 0.0, 0.0], [5.0, 0.05, 0.0], [5.0, -0.05, 0.0]]))
    assert img.shape == (1, R.v_fov, R.n_horizon)
    assert R.v_fov == 30  # -7..+52 deg at 2 deg = 30 rows
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_empty_cloud_is_all_free():
    R = RangeImageRenderer(r_max=10.0)
    img = R(np.zeros((0, 3)))
    assert img.shape == (1, R.v_fov, R.n_horizon)
    assert torch.all(img == 1.0)  # no returns -> free (1.0) everywhere


def test_hit_registers_normalized_range():
    R = RangeImageRenderer(r_max=10.0)
    rng = R(np.array([[5.0, 0.0, 0.0]]))[0]  # 5 m ahead, elevation 0 (in FOV)
    hit = rng < 1.0
    assert hit.any()
    assert torch.isclose(rng[hit].min(), torch.tensor(0.5), atol=0.05)  # 5 m / 10 m


def test_narrower_vfov_has_fewer_rows():
    wide = RangeImageRenderer(vertical_fov_min_deg=-7.0, vertical_fov_max_deg=52.0)
    narrow = RangeImageRenderer(vertical_fov_min_deg=-5.0, vertical_fov_max_deg=20.0)
    assert narrow.v_fov < wide.v_fov  # smaller FOV -> fewer rows


def test_fov_elevations_within_bounds():
    R = RangeImageRenderer(vertical_fov_min_deg=-7.0, vertical_fov_max_deg=52.0)
    el_deg = np.degrees(R._fov_elevation)
    assert el_deg.min() >= -7.0 - 2.0 and el_deg.max() <= 52.0 + 2.0  # bin centers inside FOV (± one bin)


def test_out_of_fov_point_is_cropped_away():
    R = RangeImageRenderer(r_max=10.0, vertical_fov_min_deg=-7.0, vertical_fov_max_deg=52.0)
    el = np.radians(-60.0)  # well below the FOV floor
    pt = np.array([[5.0 * np.cos(el), 0.0, 5.0 * np.sin(el)]])
    img = R(pt)
    assert torch.all(img == 1.0)  # the hit fell outside the crop -> no return


def test_backproject_is_inverse_of_render():
    # Place one hit at the center of each of 200 distinct FOV bins; render -> backproject recovers them.
    r_max = 20.0
    R = RangeImageRenderer(r_max=r_max)
    H, V = R.n_horizon, R.n_vertical
    az = -np.pi + (np.arange(H) + 0.5) / H * 2 * np.pi
    el = -np.pi / 2 + (np.arange(V) + 0.5) / V * np.pi

    gen = np.random.default_rng(0)
    rows = gen.choice(R.fov_rows, size=200)
    cols = gen.integers(0, H, size=200)
    # keep only distinct (row,col) bins so no two points collide into one bin
    seen, keep = set(), []
    for i, (r, c) in enumerate(zip(rows, cols)):
        if (int(r), int(c)) not in seen:
            seen.add((int(r), int(c)))
            keep.append(i)
    rows, cols = rows[keep], cols[keep]
    ranges = gen.uniform(1.0, 19.5, size=len(rows))
    pts = np.stack(
        [
            ranges * np.cos(el[rows]) * np.cos(az[cols]),
            ranges * np.cos(el[rows]) * np.sin(az[cols]),
            ranges * np.sin(el[rows]),
        ],
        axis=1,
    )

    bp_pts, bp_r = R.backproject(R(pts))
    assert len(bp_pts) == len(pts)  # one hit per placed bin
    az_bp = np.arctan2(bp_pts[:, 1], bp_pts[:, 0])
    el_bp = np.arcsin(np.clip(bp_pts[:, 2] / np.maximum(np.linalg.norm(bp_pts, axis=1), 1e-9), -1, 1))
    col_bp = np.floor((az_bp + np.pi) / (2 * np.pi) * H).astype(int) % H
    row_bp = np.clip(np.floor((el_bp + np.pi / 2) / np.pi * V).astype(int), 0, V - 1)
    placed = {(int(r), int(c)): ranges[k] for k, (r, c) in enumerate(zip(rows, cols))}
    errs = [
        abs(bp_r[k] - placed[(int(row_bp[k]), int(col_bp[k]))])
        for k in range(len(bp_pts))
        if (int(row_bp[k]), int(col_bp[k])) in placed
    ]
    assert len(errs) == len(bp_pts) and max(errs) < 1e-3  # float32 round-trip


def test_backproject_empty_cloud_no_hits():
    R = RangeImageRenderer(r_max=10.0)
    pts, ranges = R.backproject(R(np.zeros((0, 3))))
    assert len(pts) == 0 and len(ranges) == 0


def test_backproject_accepts_exact_2d_and_single_channel_3d_shapes():
    renderer = RangeImageRenderer(r_max=10.0)
    obs_2d = np.ones((renderer.v_fov, renderer.n_horizon), dtype=np.float32)
    obs_2d[4, 9] = 0.25
    points_2d, ranges_2d = renderer.backproject(obs_2d)
    points_3d, ranges_3d = renderer.backproject(obs_2d[None])
    np.testing.assert_allclose(points_2d, points_3d, atol=0.0)
    np.testing.assert_allclose(ranges_2d, ranges_3d, atol=0.0)


@pytest.mark.parametrize("shape", [(30, 179), (1, 29, 180), (2, 30, 180)])
def test_backproject_rejects_malformed_range_image_shape(shape):
    renderer = RangeImageRenderer()
    with pytest.raises(ValueError) as error:
        renderer.backproject(np.ones(shape, dtype=np.float32))
    message = str(error.value)
    assert "[1,30,180]" in message and "[30,180]" in message
    assert str(shape) in message


def test_get_ray_geometry_has_exact_default_centers():
    direction, theta, phi = get_ray_geometry()
    assert direction.shape == (5400, 3)
    assert theta.shape == (5400,) and phi.shape == (5400,)
    np.testing.assert_allclose(np.degrees(theta[:180]), np.arange(-179.0, 180.0, 2.0), atol=1e-10)
    np.testing.assert_allclose(np.degrees(phi[::180]), np.arange(-7.0, 52.0, 2.0), atol=1e-10)


def test_get_ray_geometry_is_row_major_and_unit_length():
    direction, theta, phi = get_ray_geometry()
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(theta.reshape(30, 180)[0], theta.reshape(30, 180)[-1], atol=0.0)
    assert np.all(np.diff(phi.reshape(30, 180)[:, 0]) > 0.0)


def test_renderer_backproject_uses_shared_direction_table():
    renderer = RangeImageRenderer(r_max=10.0)
    obs = torch.ones(1, renderer.v_fov, renderer.n_horizon)
    obs[0, 7, 179] = 0.4
    points, ranges = renderer.backproject(obs)
    direction, _, _ = get_ray_geometry()
    expected = 4.0 * direction.reshape(renderer.v_fov, renderer.n_horizon, 3)[7, 179]
    np.testing.assert_allclose(points[0], expected, atol=1e-6)
    np.testing.assert_allclose(ranges, [4.0], atol=1e-6)


def test_non_full_horizontal_fov_is_rejected_explicitly():
    with np.testing.assert_raises_regex(ValueError, "360"):
        get_ray_geometry(horizon_fov_deg=180.0)
    with np.testing.assert_raises_regex(ValueError, "360"):
        RangeImageRenderer(horizon_fov_deg=180.0)
