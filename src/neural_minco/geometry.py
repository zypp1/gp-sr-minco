"""Full-quaternion frame transforms and world-map distance geometry."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def quaternion_matrix(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return body-to-world rotation matrices for quaternions shaped (...,4)."""
    if (
        not isinstance(quaternion_wxyz, torch.Tensor)
        or quaternion_wxyz.ndim < 1
        or quaternion_wxyz.shape[-1] != 4
        or not quaternion_wxyz.is_floating_point()
    ):
        raise ValueError("quaternion must be a floating tensor shaped (...,4)")
    if not bool(torch.isfinite(quaternion_wxyz).all()):
        raise ValueError("quaternion must be finite")
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    if bool((norm <= 1e-12).any()):
        raise ValueError("quaternion norm must be positive")
    qw, qx, qy, qz = (quaternion_wxyz / norm).unbind(dim=-1)
    return torch.stack(
        (
            torch.stack(
                (1 - 2 * (qy.square() + qz.square()), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)), dim=-1
            ),
            torch.stack(
                (2 * (qx * qy + qw * qz), 1 - 2 * (qx.square() + qz.square()), 2 * (qy * qz - qw * qx)), dim=-1
            ),
            torch.stack(
                (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx.square() + qy.square())), dim=-1
            ),
        ),
        dim=-2,
    )


def quaternion_matrix_numpy(quaternion_wxyz) -> np.ndarray:
    """Return body-to-world matrices for NumPy quaternions shaped (...,4)."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.ndim < 1 or quaternion.shape[-1] != 4 or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite with shape (...,4)")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("quaternion norm must be positive")
    qw, qx, qy, qz = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)), axis=-1),
            np.stack((2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)), axis=-1),
            np.stack((2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)), axis=-1),
        ),
        axis=-2,
    )


def _torch_frame_inputs(vectors: torch.Tensor, quaternion_wxyz: torch.Tensor) -> None:
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 3 or vectors.shape[-1] != 3:
        raise ValueError("vectors must have shape (B,N,3)")
    if not isinstance(quaternion_wxyz, torch.Tensor) or quaternion_wxyz.shape != (len(vectors), 4):
        raise ValueError("quaternion must have shape (B,4)")
    if vectors.device != quaternion_wxyz.device:
        raise ValueError("vectors and quaternion must share a device")
    if vectors.dtype != quaternion_wxyz.dtype:
        raise ValueError("vectors and quaternion must share a dtype")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite")


def body_to_world(
    points_body: torch.Tensor, position_world: torch.Tensor, quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    """Rotate body points shaped (B,N,3) and translate them into world."""
    _torch_frame_inputs(points_body, quaternion_wxyz)
    if (
        not isinstance(position_world, torch.Tensor)
        or position_world.shape != (len(points_body), 3)
        or position_world.device != points_body.device
        or position_world.dtype != points_body.dtype
        or not bool(torch.isfinite(position_world).all())
    ):
        raise ValueError("position must be finite (B,3) on the same dtype/device")
    rotation = quaternion_matrix(quaternion_wxyz)
    return torch.bmm(points_body, rotation.transpose(1, 2)) + position_world[:, None]


def world_to_body(vectors_world: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors shaped (B,N,3) into the body frame."""
    _torch_frame_inputs(vectors_world, quaternion_wxyz)
    return torch.bmm(vectors_world, quaternion_matrix(quaternion_wxyz))


def body_to_world_numpy(points_body, position_world, quaternion_wxyz) -> np.ndarray:
    """Rotate body points shaped (...,3) and translate them to world."""
    points = np.asarray(points_body, dtype=np.float64)
    position = np.asarray(position_world, dtype=np.float64)
    if (
        points.ndim < 1
        or points.shape[-1] != 3
        or position.shape != (3,)
        or not np.isfinite(points).all()
        or not np.isfinite(position).all()
    ):
        raise ValueError("points must end in XYZ and position must be finite (3,)")
    rotation = quaternion_matrix_numpy(quaternion_wxyz)
    if rotation.shape != (3, 3):
        raise ValueError("NumPy point transforms require one quaternion")
    return points @ rotation.T + position


def world_to_body_numpy(vectors_world, quaternion_wxyz) -> np.ndarray:
    """Rotate world vectors shaped (...,3) into the body frame.

    For world points, subtract the world-frame origin before calling this
    function.
    """
    vectors = np.asarray(vectors_world, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be finite and end in XYZ")
    rotation = quaternion_matrix_numpy(quaternion_wxyz)
    if rotation.shape != (3, 3):
        raise ValueError("NumPy vector transforms require one quaternion")
    return vectors @ rotation


def build_esdf(points, voxel: float = 0.2, pad: float = 2.0) -> dict:
    """Build a signed distance grid from finite world obstacle points (N,3)."""
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError("ESDF points must be finite with shape (N,3)")
    if not math.isfinite(float(voxel)) or float(voxel) <= 0.0 or not math.isfinite(float(pad)) or float(pad) < 0.0:
        raise ValueError("ESDF voxel must be positive and pad nonnegative")

    origin = points.min(axis=0) - float(pad)
    grid_max = points.max(axis=0) + float(pad)
    shape = np.ceil((grid_max - origin) / float(voxel)).astype(int) + 1
    nx, ny, nz = shape.tolist()
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    indices = np.floor((points - origin) / float(voxel)).astype(int)
    indices = np.clip(indices, 0, np.asarray([nx - 1, ny - 1, nz - 1]))
    occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    distance_to_obstacle = distance_transform_edt(~occupied, sampling=float(voxel))
    distance_to_free = distance_transform_edt(occupied, sampling=float(voxel))
    sdf = (distance_to_obstacle - distance_to_free).astype(np.float32)
    return {
        "sdf": torch.from_numpy(sdf),
        "origin": origin.astype(np.float32),
        "voxel": float(voxel),
        "shape": (nx, ny, nz),
    }


def query_esdf(esdf: dict, query: torch.Tensor) -> torch.Tensor:
    """Differentiably query signed distance at world points shaped (...,3)."""
    if (
        not isinstance(query, torch.Tensor)
        or query.ndim < 1
        or query.shape[-1] != 3
        or not query.is_floating_point()
        or not bool(torch.isfinite(query).all())
    ):
        raise ValueError("ESDF query must be a finite floating tensor shaped (...,3)")
    try:
        sdf_source = esdf["sdf"]
        origin_source = esdf["origin"]
        voxel = float(esdf["voxel"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid ESDF mapping") from error
    if not isinstance(sdf_source, torch.Tensor) or sdf_source.ndim != 3 or not math.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("invalid ESDF grid")

    cache = esdf.setdefault("_query_cache", {})
    cache_key = (id(sdf_source), query.device, query.dtype)
    cached = cache.get(cache_key)
    if cached is None:
        sdf = sdf_source.to(device=query.device, dtype=query.dtype)
        if not bool(torch.isfinite(sdf).all()):
            raise ValueError("invalid ESDF grid")
        origin = torch.as_tensor(origin_source, device=query.device, dtype=query.dtype)
        if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
            raise ValueError("invalid ESDF origin")
        cached = {
            "volume": sdf.permute(2, 1, 0)[None, None],
            "origin": origin,
            "shape": torch.as_tensor(sdf.shape, device=query.device, dtype=query.dtype),
            "voxel": voxel,
        }
        cache[cache_key] = cached

    volume = cached["volume"]
    origin = cached["origin"]
    shape = cached["shape"]
    voxel_coordinate = (query - origin) / cached["voxel"]
    clamped_coordinate = torch.minimum(torch.clamp(voxel_coordinate, min=0.0), shape - 1.0)
    outside_offset = (voxel_coordinate - clamped_coordinate) * cached["voxel"]

    denominator = (shape - 1.0).clamp(min=1.0)
    grid = torch.where(shape > 1.0, 2.0 * clamped_coordinate / denominator - 1.0, torch.zeros_like(clamped_coordinate))
    sampled = F.grid_sample(
        volume, grid.reshape(1, -1, 1, 1, 3), mode="bilinear", align_corners=True, padding_mode="border"
    )
    distance = sampled.reshape(query.shape[:-1])
    return distance + outside_offset.norm(dim=-1)


__all__ = [
    "body_to_world",
    "body_to_world_numpy",
    "build_esdf",
    "quaternion_matrix",
    "quaternion_matrix_numpy",
    "query_esdf",
    "world_to_body",
    "world_to_body_numpy",
]
