"""LiDAR sensor model: body-frame point cloud -> FOV-cropped range image [1, V_fov, H].

Single channel (normalized range).  Pixels outside the sensor FOV are cropped away;
inside-FOV bins without a hit default to 1.0 (free space).
"""

import math
from typing import Tuple

import numpy as np
import torch


def _vertical_crop(
    n_vertical: int, vertical_fov_min_deg: float, vertical_fov_max_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    elevation = -math.pi / 2 + (np.arange(n_vertical) + 0.5) * math.pi / n_vertical
    keep = (elevation >= math.radians(vertical_fov_min_deg) - 1e-9) & (
        elevation <= math.radians(vertical_fov_max_deg) + 1e-9
    )
    rows = np.flatnonzero(keep)
    return rows, elevation[rows]


def get_ray_geometry(
    n_horizon: int = 180,
    n_vertical: int = 90,
    horizon_fov_deg: float = 360.0,
    vertical_fov_min_deg: float = -7.0,
    vertical_fov_max_deg: float = 52.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-major body-ray direction, azimuth, and elevation for the cropped range image."""
    if not math.isclose(float(horizon_fov_deg), 360.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ray geometry currently requires a 360-degree horizontal FOV")
    if n_horizon <= 0 or n_vertical <= 0:
        raise ValueError("n_horizon and n_vertical must be positive")
    _, elevation = _vertical_crop(n_vertical, vertical_fov_min_deg, vertical_fov_max_deg)
    azimuth = -math.pi + (np.arange(n_horizon) + 0.5) * 2.0 * math.pi / n_horizon
    theta = np.broadcast_to(azimuth, (len(elevation), n_horizon)).reshape(-1).copy()
    phi = np.broadcast_to(elevation[:, None], (len(elevation), n_horizon)).reshape(-1).copy()
    direction = np.stack([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)], axis=-1)
    return direction, theta, phi


class RangeImageRenderer:
    """Render body-frame LiDAR points into a single-channel, FOV-cropped range image."""

    def __init__(
        self,
        r_max: float = 20.0,
        n_horizon: int = 180,
        n_vertical: int = 90,
        horizon_fov_deg: float = 360.0,
        vertical_fov_min_deg: float = -7.0,
        vertical_fov_max_deg: float = 52.0,
    ):
        self.r_max = float(r_max)
        self.n_horizon = int(n_horizon)
        self.n_vertical = int(n_vertical)

        self.fov_rows, self._fov_elevation = _vertical_crop(self.n_vertical, vertical_fov_min_deg, vertical_fov_max_deg)
        direction, theta, phi = get_ray_geometry(
            self.n_horizon, self.n_vertical, horizon_fov_deg, vertical_fov_min_deg, vertical_fov_max_deg
        )
        self._ray_direction = direction.reshape(len(self.fov_rows), self.n_horizon, 3)
        self._ray_theta = theta.reshape(len(self.fov_rows), self.n_horizon)
        self._ray_phi = phi.reshape(len(self.fov_rows), self.n_horizon)

    @property
    def v_fov(self) -> int:
        return len(self.fov_rows)

    def __call__(self, points: np.ndarray) -> torch.Tensor:
        """points [N,3] body-frame cloud -> range image [1, V_fov, H]."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        distance = np.sqrt(x * x + y * y + z * z)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(np.clip(z / np.maximum(distance, 1e-9), -1.0, 1.0))
        valid = (distance > 1e-3) & (distance <= self.r_max)

        range_grid = np.full(self.n_vertical * self.n_horizon, self.r_max, dtype=np.float64)
        if valid.any():
            col = np.floor((azimuth[valid] + math.pi) / (2.0 * math.pi) * self.n_horizon).astype(int) % self.n_horizon
            row = np.floor((elevation[valid] + math.pi / 2) / math.pi * self.n_vertical).astype(int)
            row = np.clip(row, 0, self.n_vertical - 1)
            np.minimum.at(range_grid, row * self.n_horizon + col, distance[valid])

        range_grid = range_grid.reshape(self.n_vertical, self.n_horizon)
        range_channel = (range_grid / self.r_max).astype(np.float32)
        # Crop to FOV rows, keep channel dim.
        return torch.from_numpy(range_channel[self.fov_rows][None])

    def backproject(self, obs):
        """Inverse of __call__: `[1,V_fov,H]` or `[V_fov,H]` -> body-frame hits and ranges.

        Hits = FOV pixels with a real return (range < 1.0);
        one nearest point per beam (a 2.5D shell, not the full cloud).
        """
        obs = np.asarray(obs)
        expected_2d = (self.v_fov, self.n_horizon)
        expected_3d = (1, self.v_fov, self.n_horizon)
        if obs.shape == expected_3d:
            rng = obs[0]
        elif obs.shape == expected_2d:
            rng = obs
        else:
            raise ValueError(
                f"backproject expects [1,{self.v_fov},{self.n_horizon}] or "
                f"[{self.v_fov},{self.n_horizon}], got {tuple(obs.shape)}"
            )
        hit = rng < 1.0
        if not hit.any():
            return np.zeros((0, 3)), np.zeros((0,))
        rows, cols = np.nonzero(hit)
        r = rng[rows, cols] * self.r_max
        points = r[:, None] * self._ray_direction[rows, cols]
        return points, r
