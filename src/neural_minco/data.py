"""Current raw-Episode storage, fixed window conversion, and datasets.

The supported training contract is deliberately singular: one current LiDAR
range image, the nearest real odometry state as the common time origin, and 20
future targets at 0.1 s intervals over a complete 2 s horizon.
"""

import json
import os
import re
import tempfile
from collections import defaultdict
from typing import Iterator, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import SENSOR_SYNC_TOLERANCE_S, default_config
from .geometry import world_to_body_numpy

WINDOW_TIME_ORIGIN = "nearest_odom_v2"
EPISODE_NAME_PATTERN = re.compile(r"^episode-(\d+)-\d{6}-\d{6}(?:-[^.]+)?\.npz$")

_CONFIG = default_config()
_DATA_CONFIG = _CONFIG["data"]
_LIDAR_CONFIG = _CONFIG["lidar"]
MAX_SENSOR_DELTA_S = SENSOR_SYNC_TOLERANCE_S
HORIZON_S = float(_DATA_CONFIG["horizon"])
TRAJECTORY_DT_S = float(_DATA_CONFIG["trajectory_dt"])
TRAJECTORY_SAMPLES = int(_DATA_CONFIG["trajectory_samples"])
ANCHOR_STRIDE_FRAMES = int(_DATA_CONFIG["anchor_stride"])

_N_HORIZON = int(_LIDAR_CONFIG["n_horizon"])
_N_VERTICAL = int(_LIDAR_CONFIG["n_vertical"])
_HORIZON_FOV_DEG = float(_LIDAR_CONFIG["horizon_fov_deg"])
_VERTICAL_MIN_DEG = float(_LIDAR_CONFIG["vertical_fov_min_deg"])
_VERTICAL_MAX_DEG = float(_LIDAR_CONFIG["vertical_fov_max_deg"])
_ELEVATION_DEG = -90.0 + (np.arange(_N_VERTICAL) + 0.5) * 180.0 / _N_VERTICAL
_V_FOV = int(
    np.count_nonzero((_ELEVATION_DEG >= _VERTICAL_MIN_DEG - 1e-9) & (_ELEVATION_DEG <= _VERTICAL_MAX_DEG + 1e-9))
)
_LIDAR_SHAPE = (1, _V_FOV, _N_HORIZON)
CURRENT_LIDAR_GEOMETRY = np.asarray(
    [
        float(_LIDAR_CONFIG["r_max"]),
        _HORIZON_FOV_DEG / _N_HORIZON,
        _N_HORIZON,
        _N_VERTICAL,
        _HORIZON_FOV_DEG,
        _VERTICAL_MIN_DEG,
        _VERTICAL_MAX_DEG,
        1.0,
    ],
    dtype=np.float64,
)

_REQUIRED_ARRAYS = {
    "episode_id",
    "map_id",
    "start_stamp",
    "end_stamp",
    "goal_stamp",
    "goal_world",
    "success",
    "lidar_geometry",
    "lidar_range_images",
    "lidar_stamps",
    "odom_stamps",
    "odom_positions",
    "odom_velocities",
    "odom_orientations_wxyz",
    "acceleration_stamps",
    "accelerations",
}


def next_episode_sequence(out_dir: str) -> int:
    """Return the next global filename sequence in an Episode directory."""
    largest = 0
    for name in os.listdir(out_dir):
        match = EPISODE_NAME_PATTERN.match(name)
        if match:
            largest = max(largest, int(match.group(1)))
    return largest + 1


def _finite(name: str, value: np.ndarray, path: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{path} array {name} contains NaN or infinity")


def _scalar(name: str, data: Mapping, path: str) -> np.ndarray:
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"{path} {name} must be a scalar")
    return value


def _nearest_indices(query: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """Select nearest real frames, breaking exact ties toward the earlier one."""
    right = np.searchsorted(stamps, query, side="left")
    right = np.clip(right, 0, len(stamps) - 1)
    left = np.clip(right - 1, 0, len(stamps) - 1)
    choose_left = np.abs(query - stamps[left]) <= np.abs(stamps[right] - query)
    return np.where(choose_left, left, right)


def validate_episode_env(path: str, data: Mapping) -> None:
    """Validate one raw Episode against the current collection contract."""
    missing = _REQUIRED_ARRAYS - data.keys()
    if missing:
        raise ValueError(f"{path} missing required arrays {sorted(missing)}")

    episode_id = _scalar("episode_id", data, path)
    map_id = _scalar("map_id", data, path)
    if episode_id.dtype.kind not in "SU" or not str(episode_id.item()):
        raise ValueError(f"{path} episode_id must be a non-empty string")
    if map_id.dtype.kind not in "SU" or not str(map_id.item()):
        raise ValueError(f"{path} map_id must be a non-empty string")

    for name in ("start_stamp", "end_stamp", "goal_stamp", "success"):
        _scalar(name, data, path)
    goal = np.asarray(data["goal_world"])
    if goal.shape != (3,):
        raise ValueError(f"{path} goal_world must have shape [3]")

    lidar_geometry = np.asarray(data["lidar_geometry"], dtype=np.float64)
    if lidar_geometry.shape != (8,) or not np.allclose(lidar_geometry, CURRENT_LIDAR_GEOMETRY, rtol=0.0, atol=1e-9):
        raise ValueError(f"{path} lidar_geometry must match the current single-frame sensor")

    lidar_images = np.asarray(data["lidar_range_images"])
    if lidar_images.ndim != 4 or tuple(lidar_images.shape[1:]) != _LIDAR_SHAPE:
        raise ValueError(
            f"{path} lidar_range_images must have shape [K,{_LIDAR_SHAPE[0]},{_LIDAR_SHAPE[1]},{_LIDAR_SHAPE[2]}]"
        )
    n_lidar = len(lidar_images)
    n_odom = len(np.asarray(data["odom_stamps"]))
    expected_shapes = {
        "lidar_stamps": (n_lidar,),
        "odom_stamps": (n_odom,),
        "odom_positions": (n_odom, 3),
        "odom_velocities": (n_odom, 3),
        "odom_orientations_wxyz": (n_odom, 4),
        "acceleration_stamps": (n_odom,),
        "accelerations": (n_odom, 3),
    }
    for name, expected_shape in expected_shapes.items():
        value = np.asarray(data[name])
        if value.shape != expected_shape:
            raise ValueError(f"{path} array {name} expected shape {expected_shape}, got {value.shape}")
        _finite(name, value, path)
    _finite("lidar_geometry", lidar_geometry, path)
    _finite("lidar_range_images", lidar_images, path)
    for name in ("start_stamp", "end_stamp", "goal_stamp", "goal_world"):
        _finite(name, np.asarray(data[name]), path)
    if n_lidar and (lidar_images.min() < 0.0 or lidar_images.max() > 1.0):
        raise ValueError(f"{path} lidar_range_images must be normalized to [0,1]")

    if n_odom < 1:
        raise ValueError(f"{path} episode needs at least one odom/acceleration pair")
    odom_stamps = np.asarray(data["odom_stamps"], dtype=np.float64)
    acceleration_stamps = np.asarray(data["acceleration_stamps"], dtype=np.float64)
    lidar_stamps = np.asarray(data["lidar_stamps"], dtype=np.float64)
    orientations = np.asarray(data["odom_orientations_wxyz"], dtype=np.float64)
    if np.any(np.linalg.norm(orientations, axis=1) < 1e-8):
        raise ValueError(f"{path} odom_orientations_wxyz contains a zero-norm quaternion")
    if len(odom_stamps) > 1 and np.any(np.diff(odom_stamps) <= 0.0):
        raise ValueError(f"{path} odom_stamps are not strictly increasing")
    if len(lidar_stamps) > 1 and np.any(np.diff(lidar_stamps) <= 0.0):
        raise ValueError(f"{path} lidar_stamps are not strictly increasing")
    if np.any(np.abs(acceleration_stamps - odom_stamps) > MAX_SENSOR_DELTA_S + 1e-9):
        raise ValueError(f"{path} odom/acceleration pair exceeds 5 ms")

    start = float(np.asarray(data["start_stamp"]))
    end = float(np.asarray(data["end_stamp"]))
    if end < start:
        raise ValueError(f"{path} episode end predates its start")
    if not np.isclose(odom_stamps[0], start, atol=1e-6):
        raise ValueError(f"{path} odom stream does not begin at episode start")
    if not np.isclose(odom_stamps[-1], end, atol=1e-6):
        raise ValueError(f"{path} odom stream does not end at episode end")
    if len(lidar_stamps) and (
        lidar_stamps[0] < start - MAX_SENSOR_DELTA_S - 1e-9 or lidar_stamps[-1] > end + MAX_SENSOR_DELTA_S + 1e-9
    ):
        raise ValueError(f"{path} lidar frame lies outside its episode interval")

    success = np.asarray(data["success"])
    if not np.issubdtype(success.dtype, np.integer) or int(success) not in (0, 1):
        raise ValueError(f"{path} success must be zero or one")
    if bool(success) and not n_lidar:
        raise ValueError(f"{path} successful episode has no LiDAR frame")
    if len(lidar_stamps):
        nearest = _nearest_indices(lidar_stamps, odom_stamps)
        if np.any(np.abs(lidar_stamps - odom_stamps[nearest]) > MAX_SENSOR_DELTA_S + 1e-9):
            raise ValueError(f"{path} lidar/odom pair exceeds 5 ms")


def save_episode_env(path: str, payload: Mapping) -> None:
    """Validate and atomically create one complete ``episode-*.npz`` file.

    Existing Episodes are never replaced. Collection must choose a new sequence
    number instead of silently destroying a completed flight.
    """
    arrays = {name: np.asarray(value) for name, value in payload.items()}
    validate_episode_env(path, arrays)

    target_path = os.fspath(path)
    target_dir = os.path.dirname(os.path.abspath(target_path))
    target_name = os.path.basename(target_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target_name}.", suffix=".tmp", dir=target_dir, delete=False
        ) as temporary:
            temporary_path = temporary.name
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())

        load_episode_env(temporary_path)
        os.link(temporary_path, target_path)
        os.unlink(temporary_path)
        temporary_path = None

        directory_fd = os.open(target_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def load_episode_env(path: str) -> dict:
    """Load one raw Episode without pickle and validate the complete archive."""
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    validate_episode_env(path, data)
    return data


def _interp_columns(query: np.ndarray, stamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(query, stamps, values[:, axis]) for axis in range(values.shape[1])], axis=-1)


def convert_episode_windows(data: Mapping) -> dict:
    """Build fixed windows from an Episode already validated by the loader."""
    if not bool(np.asarray(data["success"])):
        raise ValueError("episode data contains no complete trajectory window")

    observations = np.asarray(data["lidar_range_images"], dtype=np.float32)
    lidar_stamps = np.asarray(data["lidar_stamps"], dtype=np.float64)
    odom_stamps = np.asarray(data["odom_stamps"], dtype=np.float64)
    acceleration_stamps = np.asarray(data["acceleration_stamps"], dtype=np.float64)
    positions = np.asarray(data["odom_positions"], dtype=np.float64)
    velocities = np.asarray(data["odom_velocities"], dtype=np.float64)
    accelerations = np.asarray(data["accelerations"], dtype=np.float64)
    quaternions = np.asarray(data["odom_orientations_wxyz"], dtype=np.float64)
    goal_world = np.asarray(data["goal_world"], dtype=np.float64)
    elapsed = (np.arange(TRAJECTORY_SAMPLES, dtype=np.float64) + 1.0) * TRAJECTORY_DT_S

    rows = defaultdict(list)
    for observation_index in range(0, len(lidar_stamps), ANCHOR_STRIDE_FRAMES):
        observation_stamp = lidar_stamps[observation_index]
        odom_index = int(_nearest_indices(np.asarray([observation_stamp]), odom_stamps)[0])
        anchor_stamp = float(odom_stamps[odom_index])
        if anchor_stamp + HORIZON_S > odom_stamps[-1] + 1e-9:
            continue

        query = anchor_stamp + elapsed
        quaternion_wxyz = quaternions[odom_index]
        position_world = positions[odom_index]
        state_world = np.concatenate(
            [position_world, velocities[odom_index], accelerations[odom_index], quaternion_wxyz]
        )
        target_position_world = _interp_columns(query, odom_stamps, positions)
        target_velocity_world = _interp_columns(query, odom_stamps, velocities)
        target_acceleration_world = _interp_columns(query, acceleration_stamps, accelerations)
        target_position_body = world_to_body_numpy(target_position_world - position_world, quaternion_wxyz)
        target_velocity_body = world_to_body_numpy(target_velocity_world, quaternion_wxyz)
        target_acceleration_body = world_to_body_numpy(target_acceleration_world, quaternion_wxyz)
        state_body = np.concatenate(
            [
                world_to_body_numpy(velocities[odom_index], quaternion_wxyz),
                world_to_body_numpy(accelerations[odom_index], quaternion_wxyz),
                world_to_body_numpy(goal_world - position_world, quaternion_wxyz),
            ]
        )

        rows["obs"].append(observations[observation_index])
        rows["state"].append(state_world.astype(np.float32))
        rows["state_body"].append(state_body.astype(np.float32))
        rows["dense_traj"].append(target_position_body.astype(np.float32))
        rows["dense_velocity"].append(target_velocity_body.astype(np.float32))
        rows["dense_acceleration"].append(target_acceleration_body.astype(np.float32))
        rows["v_end"].append(target_velocity_body[-1].astype(np.float32))
        rows["a_end"].append(target_acceleration_body[-1].astype(np.float32))
        rows["observation_index"].append(observation_index)
        rows["anchor_stamp"].append(anchor_stamp)

    if not rows["obs"]:
        raise ValueError("episode data contains no complete trajectory window")

    count = len(rows["obs"])
    episode_id = str(np.asarray(data["episode_id"]).item())
    map_id = str(np.asarray(data["map_id"]).item())
    return {
        "obs": np.stack(rows["obs"]),
        "state": np.stack(rows["state"]),
        "state_body": np.stack(rows["state_body"]),
        "dense_traj": np.stack(rows["dense_traj"]),
        "dense_velocity": np.stack(rows["dense_velocity"]),
        "dense_acceleration": np.stack(rows["dense_acceleration"]),
        "dense_traj_valid": np.ones((count, TRAJECTORY_SAMPLES), dtype=bool),
        "v_end": np.stack(rows["v_end"]),
        "a_end": np.stack(rows["a_end"]),
        "episode_id": np.full(count, episode_id),
        "map_id": np.full(count, map_id),
        "observation_index": np.asarray(rows["observation_index"], dtype=np.int64),
        "anchor_stamp": np.asarray(rows["anchor_stamp"], dtype=np.float64),
    }


def _validated_manifest_splits(manifest: Mapping[str, object]):
    """Validate the fixed flight-level split and safe Episode basenames."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON object")
    split = manifest.get("split")
    split_names = ("train", "val", "test")
    if not isinstance(split, dict) or not all(name in split for name in split_names):
        raise ValueError("manifest must define train, val, and test splits")

    seen = set()
    for split_name in split_names:
        entries = split[split_name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"manifest split.{split_name} must be a non-empty list")
        invalid = [
            entry
            for entry in entries
            if not isinstance(entry, str)
            or os.path.isabs(entry)
            or os.path.basename(entry) != entry
            or "\\" in entry
            or EPISODE_NAME_PATTERN.fullmatch(entry) is None
        ]
        if invalid:
            raise ValueError(f"manifest split.{split_name} entries must be Episode basenames")
        if len(set(entries)) != len(entries):
            raise ValueError(f"manifest split.{split_name} contains duplicates")
        overlap = seen.intersection(entries)
        if overlap:
            raise ValueError("manifest splits overlap at %s" % sorted(overlap)[0])
        seen.update(entries)
    return split


def load_manifest(path):
    """Read a JSON flight-level split manifest."""
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    _validated_manifest_splits(manifest)
    return manifest


def split_episode_files(manifest: Mapping[str, object], data_root: str, split: str) -> Tuple[str, ...]:
    """Resolve one flight-level split to absolute Episode paths."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    names = _validated_manifest_splits(manifest)[split]
    paths = tuple(os.path.join(os.path.abspath(data_root), name) for name in names)
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} Episodes are missing; first is {missing[0]}")
    return paths


def validate_lidar_compatibility(dataset: "EpisodeWindowDataset", lidar_config: Mapping[str, object]) -> None:
    """Reject a dataset whose normalized ranges have different physical meaning."""
    try:
        n_horizon = int(lidar_config["n_horizon"])
        expected = np.asarray(
            [
                float(lidar_config["r_max"]),
                float(lidar_config["horizon_fov_deg"]) / n_horizon,
                n_horizon,
                int(lidar_config["n_vertical"]),
                float(lidar_config["horizon_fov_deg"]),
                float(lidar_config["vertical_fov_min_deg"]),
                float(lidar_config["vertical_fov_max_deg"]),
                int(lidar_config["n_frames"]),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("model/checkpoint LiDAR config is incomplete or invalid") from error

    if not np.allclose(dataset.lidar_geometry, expected, rtol=0.0, atol=1e-9):
        raise ValueError("dataset LiDAR geometry disagrees with the model/checkpoint config")


_WINDOW_TENSOR_FIELDS = (
    "obs",
    "state",
    "state_body",
    "dense_traj",
    "dense_velocity",
    "dense_acceleration",
    "dense_traj_valid",
    "v_end",
    "a_end",
)


class EpisodeWindowDataset(Dataset):
    """Materialize current two-second, stride-five windows from raw Episodes."""

    def __init__(self, files: Sequence[str]):
        if not files:
            raise ValueError("EpisodeWindowDataset needs at least one Episode")
        arrays = defaultdict(list)
        episode_ids = []
        map_ids = []
        retained_files = []
        lidar_geometry = None

        for path in files:
            absolute_path = os.path.abspath(os.fspath(path))
            episode = load_episode_env(absolute_path)
            if not bool(np.asarray(episode["success"])):
                raise ValueError(f"manifest contains unsuccessful Episode {absolute_path}")
            geometry = np.asarray(episode["lidar_geometry"], dtype=np.float64)
            if lidar_geometry is None:
                lidar_geometry = geometry.copy()
            elif not np.array_equal(geometry, lidar_geometry):
                raise ValueError("Episode files use different LiDAR geometry")
            try:
                windows = convert_episode_windows(episode)
            except ValueError as error:
                if "no complete trajectory window" in str(error):
                    continue
                raise
            episode_index = len(retained_files)
            retained_files.append(absolute_path)
            count = len(windows["obs"])
            for name in (*_WINDOW_TENSOR_FIELDS, "observation_index", "anchor_stamp"):
                arrays[name].append(windows[name])
            arrays["episode_index"].append(np.full(count, episode_index, dtype=np.int64))
            episode_ids.extend(windows["episode_id"].tolist())
            map_ids.extend(windows["map_id"].tolist())

        if not retained_files:
            raise ValueError("split has no complete two-second trajectory window")
        self.files = tuple(retained_files)
        self.lidar_geometry = lidar_geometry
        for name in _WINDOW_TENSOR_FIELDS:
            dtype = bool if name == "dense_traj_valid" else np.float32
            setattr(self, name, np.concatenate(arrays[name]).astype(dtype, copy=False))
        self.episode_index = np.concatenate(arrays["episode_index"])
        self.observation_index = np.concatenate(arrays["observation_index"])
        self.anchor_stamp = np.concatenate(arrays["anchor_stamp"])
        self.episode_id = np.asarray(episode_ids)
        self.map_id = np.asarray(map_ids)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, index: int) -> dict:
        return {
            "sample_index": int(index),
            **{name: torch.from_numpy(getattr(self, name)[index]) for name in _WINDOW_TENSOR_FIELDS},
            "observation_index": int(self.observation_index[index]),
            "episode_index": int(self.episode_index[index]),
            "episode_id": str(self.episode_id[index]),
            "map_id": str(self.map_id[index]),
        }


class EpisodeUniformSampler(Sampler):
    """Draw an Episode uniformly, then one of its windows uniformly."""

    def __init__(self, episode_index: np.ndarray, seed: int = 42):
        labels = np.asarray(episode_index, dtype=np.int64)
        if labels.ndim != 1 or not len(labels):
            raise ValueError("episode_index must be a nonempty vector")
        self.groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(labels)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        random = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_samples):
            group = self.groups[int(random.integers(len(self.groups)))]
            yield int(group[int(random.integers(len(group)))])

    def __len__(self) -> int:
        return self.num_samples


def load_datasets(
    data_root: str, manifest_path: str, *, include_test: bool = False
) -> Tuple[EpisodeWindowDataset, ...]:
    """Build train/validation and optional test window datasets."""
    manifest = load_manifest(manifest_path)
    splits = ("train", "val", "test") if include_test else ("train", "val")
    return tuple(EpisodeWindowDataset(split_episode_files(manifest, data_root, split)) for split in splits)


__all__ = [
    "ANCHOR_STRIDE_FRAMES",
    "CURRENT_LIDAR_GEOMETRY",
    "EpisodeUniformSampler",
    "EpisodeWindowDataset",
    "HORIZON_S",
    "MAX_SENSOR_DELTA_S",
    "TRAJECTORY_DT_S",
    "TRAJECTORY_SAMPLES",
    "WINDOW_TIME_ORIGIN",
    "convert_episode_windows",
    "load_datasets",
    "load_episode_env",
    "load_manifest",
    "next_episode_sequence",
    "save_episode_env",
    "split_episode_files",
    "validate_lidar_compatibility",
    "validate_episode_env",
]
