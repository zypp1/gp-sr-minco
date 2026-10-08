"""Evaluation metrics for the current GP-SR planner."""

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ACCELERATION_LIMIT_MPS2, COLLISION_RADIUS_M, JERK_LIMIT_MPS3, VELOCITY_LIMIT_MPS
from .data import EpisodeWindowDataset, validate_lidar_compatibility
from .geometry import body_to_world, build_esdf, query_esdf
from .lidar import get_ray_geometry
from .minco import poly_eval, sample_future_trajectory_batch
from .runtime import decode_prediction

COLLISION_SAMPLES = 128
DYNAMICS_SAMPLES_PER_SEGMENT = 1001


def lidar_clearance(
    points: torch.Tensor,
    observation: torch.Tensor,
    ray_direction: torch.Tensor,
    r_max: float,
    *,
    point_chunk_size: int = 16,
) -> torch.Tensor:
    """Nearest Euclidean distance to an observed LiDAR return."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (B,S,3)")
    if observation.ndim != 4 or observation.shape[:2] != (len(points), 1):
        raise ValueError("observation must have shape (B,1,30,180)")
    ray_count = observation.shape[2] * observation.shape[3]
    if ray_direction.shape != (ray_count, 3):
        raise ValueError(f"ray_direction must have shape ({ray_count},3)")
    if not math.isfinite(float(r_max)) or r_max <= 0.0:
        raise ValueError("r_max must be finite and positive")
    if point_chunk_size < 1:
        raise ValueError("point_chunk_size must be positive")
    ranges = observation[:, 0].reshape(len(points), ray_count).to(points)
    if not bool(torch.isfinite(ranges).all()) or bool(((ranges < 0.0) | (ranges > 1.0)).any()):
        raise ValueError("normalized range values must lie in [0,1]")
    direction = ray_direction.to(points)
    hit = ranges < 1.0
    hit_points = float(r_max) * ranges[..., None] * direction[None]
    no_hit = points.new_tensor(2.0 * float(r_max))
    chunks = []
    for start in range(0, points.shape[1], point_chunk_size):
        distance = torch.cdist(points[:, start : start + point_chunk_size], hit_points)
        chunks.append(torch.where(hit[:, None], distance, no_hit).amin(dim=-1))
    return torch.cat(chunks, dim=1)


def minco_derivative_maxima(
    coefficients: torch.Tensor, durations: torch.Tensor, samples_per_segment: int = DYNAMICS_SAMPLES_PER_SEGMENT
) -> torch.Tensor:
    """Return sampled continuous-curve max speed, acceleration, and jerk."""
    if (
        coefficients.ndim != 4
        or coefficients.shape[-1] != 3
        or coefficients.shape[2] < 4
        or durations.shape != coefficients.shape[:2]
        or samples_per_segment < 2
    ):
        raise ValueError("invalid coefficients, durations, or sample count")
    unit_time = torch.linspace(0.0, 1.0, samples_per_segment, dtype=coefficients.dtype, device=coefficients.device)
    local_time = durations[:, :, None, None].to(coefficients) * unit_time[None, None, :, None]
    segment_coefficients = coefficients.permute(0, 1, 3, 2)[:, :, None]
    maxima = []
    for derivative in (1, 2, 3):
        value = poly_eval(segment_coefficients, local_time, deriv=derivative)
        maxima.append(torch.linalg.vector_norm(value, dim=-1).amax(dim=(1, 2)))
    return torch.stack(maxima, dim=-1)


def trajectory_displacement_errors(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-window ADE/FDE for the valid future target prefix."""
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    ade = (distance * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    last = valid.long().sum(dim=1).sub(1).clamp_min(0)
    fde = distance.gather(1, last[:, None])[:, 0]
    return ade, fde


def episode_balanced_mean(values: np.ndarray, episodes: np.ndarray) -> float:
    """Average aligned window values within Episodes, then across Episodes."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    episodes = np.asarray(episodes, dtype=np.int64).reshape(-1)
    if values.shape != episodes.shape or not len(values):
        raise ValueError("metric values and Episode indices must align")
    return float(np.mean([values[episodes == episode].mean() for episode in np.unique(episodes)]))


def _metric(values: np.ndarray, episodes: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(values, (0.5, 0.9, 0.95, 0.99))
    return {
        "window_mean": float(values.mean()),
        "episode_balanced_mean": episode_balanced_mean(values, episodes),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "maximum": float(values.max()),
    }


def _rate(flags: np.ndarray, episodes: np.ndarray) -> dict:
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    return {
        "count": int(flags.sum()),
        "windows": int(len(flags)),
        "window_rate": float(flags.mean()),
        "episode_balanced_rate": episode_balanced_mean(flags, episodes),
    }


def _summarize(store: Mapping[str, list], episodes: np.ndarray) -> dict:
    values = {name: np.concatenate(chunks) for name, chunks in store.items()}
    metric_names = (
        "ade_m",
        "fde_m",
        "minimum_visible_clearance_m",
        "maximum_velocity_mps",
        "maximum_acceleration_mps2",
        "maximum_jerk_mps3",
        "segment_1_s",
        "segment_2_s",
        "segment_3_s",
        "correction_norm",
    )
    rate_names = ("visible_collision", "velocity_violation", "acceleration_violation", "jerk_violation")
    result = {
        "metrics": {name: _metric(values[name], episodes) for name in metric_names},
        "rates": {name: _rate(values[name], episodes) for name in rate_names},
    }
    if "minimum_global_clearance_m" in values:
        result["metrics"]["minimum_global_clearance_m"] = _metric(values["minimum_global_clearance_m"], episodes)
        result["rates"]["global_collision"] = _rate(values["global_collision"], episodes)
        result["rates"]["occupied_penetration"] = _rate(values["occupied_penetration"], episodes)
    return result


def load_fixed_esdf(map_path: str, voxel: float = 0.2) -> tuple:
    """Load the fixed point-cloud map and build its deterministic ESDF."""
    resolved = os.path.abspath(map_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != {"points"}:
            raise ValueError("map artifact must contain only a points array")
        points = np.asarray(archive["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("map points must be finite with shape (N,3)")
    return build_esdf(points, voxel=voxel, pad=2.0), {
        "path": resolved,
        "points": int(len(points)),
        "voxel_m": float(voxel),
        "padding_m": 2.0,
    }


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataset: EpisodeWindowDataset,
    *,
    device: str,
    batch_size: int = 32,
    workers: int = 4,
    map_path: Optional[str] = None,
) -> dict:
    """Evaluate all four outputs; visible collision is the primary metric."""
    torch_device = torch.device(device)
    map_ids = sorted({str(value) for value in np.asarray(dataset.map_id).reshape(-1)})
    if not map_ids:
        raise ValueError("evaluation dataset has no map_id")
    expected_map_id = Path(map_path).stem if map_path is not None else None
    if expected_map_id is not None and map_ids != [expected_map_id]:
        raise ValueError("evaluation map does not match the dataset map_id")
    validate_lidar_compatibility(dataset, model.lidar_config)

    model.to(torch_device).eval()
    loader_options = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": torch_device.type == "cuda",
    }
    if workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    loader = DataLoader(dataset, **loader_options)
    direction_np, _, _ = get_ray_geometry()
    ray_direction = torch.as_tensor(direction_np, device=torch_device, dtype=torch.float32)
    esdf = None
    map_metadata = None
    if map_path is not None:
        esdf, map_metadata = load_fixed_esdf(map_path)
        map_metadata["map_id"] = expected_map_id

    stores = [defaultdict(list) for _ in range(4)]
    episode_chunks = []
    for batch in tqdm(loader, desc="evaluate"):
        observation = batch["obs"].to(torch_device, non_blocking=True)
        state_body = batch["state_body"].to(torch_device, non_blocking=True)
        state_world = batch["state"].to(torch_device, non_blocking=True)
        target = batch["dense_traj"].to(torch_device, non_blocking=True)
        valid = batch["dense_traj_valid"].to(torch_device, non_blocking=True)
        _, auxiliary = model(observation, state_body, return_stages=True)
        episode_chunks.append(np.asarray(batch["episode_index"], dtype=np.int64))
        for layer_index, raw in enumerate(auxiliary["raw_stages"]):
            decoded = decode_prediction(model, raw, state_body, samples=20)
            ade, fde = trajectory_displacement_errors(decoded.position, target, valid)
            collision_points = sample_future_trajectory_batch(
                decoded.coefficients, decoded.durations, COLLISION_SAMPLES
            )
            visible = lidar_clearance(collision_points, observation, ray_direction, model.encoder.r_max).amin(dim=1)
            dynamics = minco_derivative_maxima(decoded.coefficients, decoded.durations)
            store = stores[layer_index]
            for name, value in (
                ("ade_m", ade),
                ("fde_m", fde),
                ("minimum_visible_clearance_m", visible),
                ("visible_collision", visible < COLLISION_RADIUS_M),
                ("maximum_velocity_mps", dynamics[:, 0]),
                ("maximum_acceleration_mps2", dynamics[:, 1]),
                ("maximum_jerk_mps3", dynamics[:, 2]),
                ("velocity_violation", dynamics[:, 0] > VELOCITY_LIMIT_MPS),
                ("acceleration_violation", dynamics[:, 1] > ACCELERATION_LIMIT_MPS2),
                ("jerk_violation", dynamics[:, 2] > JERK_LIMIT_MPS3),
                ("segment_1_s", decoded.durations[:, 0]),
                ("segment_2_s", decoded.durations[:, 1]),
                ("segment_3_s", decoded.durations[:, 2]),
                ("correction_norm", auxiliary["correction_norms"][:, layer_index]),
            ):
                store[name].append(value.cpu().numpy())
            if esdf is not None:
                world = body_to_world(collision_points, state_world[:, :3], state_world[:, 9:13])
                global_clearance = query_esdf(esdf, world).amin(dim=1)
                store["minimum_global_clearance_m"].append(global_clearance.cpu().numpy())
                store["global_collision"].append((global_clearance < COLLISION_RADIUS_M).cpu().numpy())
                store["occupied_penetration"].append((global_clearance < 0.0).cpu().numpy())

    episodes = np.concatenate(episode_chunks)
    layers = [{"layer": index, **_summarize(store, episodes)} for index, store in enumerate(stores)]
    return {
        "protocol": {
            "windows": len(dataset),
            "episodes": len(np.unique(episodes)),
            "map_ids": map_ids,
            "collision_samples": COLLISION_SAMPLES,
            "dynamics_samples_per_segment": DYNAMICS_SAMPLES_PER_SEGMENT,
            "primary_collision_metric": "episode_balanced_visible_collision_0.3m",
            "thresholds": {
                "collision_radius_m": COLLISION_RADIUS_M,
                "velocity_limit_mps": VELOCITY_LIMIT_MPS,
                "acceleration_limit_mps2": ACCELERATION_LIMIT_MPS2,
                "jerk_limit_mps3": JERK_LIMIT_MPS3,
            },
            "map": map_metadata,
        },
        "layers": layers,
    }


__all__ = [
    "episode_balanced_mean",
    "evaluate_model",
    "lidar_clearance",
    "load_fixed_esdf",
    "minco_derivative_maxima",
    "trajectory_displacement_errors",
]
