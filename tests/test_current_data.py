"""Focused tests for the sole supported raw-Episode and window contract."""

import json
import math

import numpy as np
import pytest

import neural_minco.data as current_data
from neural_minco.config import default_config
from neural_minco.data import (
    CURRENT_LIDAR_GEOMETRY,
    EpisodeUniformSampler,
    EpisodeWindowDataset,
    convert_episode_windows,
    load_episode_env,
    load_manifest,
    save_episode_env,
    split_episode_files,
    validate_lidar_compatibility,
)
from scripts.data_collect.validate import audit_episode_file


def _quaternion_and_rotation(roll: float, pitch: float, yaw: float):
    """Return independent wxyz and body-to-world matrices for XYZ Euler angles."""
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
        dtype=np.float32,
    )
    rx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]],
        dtype=np.float64,
    )
    ry = np.asarray(
        [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return quaternion, rz @ ry @ rx


def _episode(*, lidar_offset_s: float = 0.0) -> dict:
    stamps = 10.0 + 0.1 * np.arange(26, dtype=np.float64)
    elapsed = stamps - stamps[0]
    positions = np.zeros((len(stamps), 3), dtype=np.float32)
    positions[:, 0] = elapsed
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 1.0
    quaternions = np.zeros((len(stamps), 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    return {
        "episode_id": np.asarray("episode-test"),
        "map_id": np.asarray("mixed_forest_200_dense1800_seed1"),
        "start_stamp": np.asarray(stamps[0], dtype=np.float64),
        "end_stamp": np.asarray(stamps[-1], dtype=np.float64),
        "goal_stamp": np.asarray(stamps[0], dtype=np.float64),
        "goal_world": np.asarray([8.0, 0.0, 0.0], dtype=np.float32),
        "success": np.asarray(1, dtype=np.uint8),
        "lidar_geometry": CURRENT_LIDAR_GEOMETRY.copy(),
        "lidar_range_images": np.ones((len(stamps), 1, 30, 180), dtype=np.float32),
        "lidar_stamps": stamps + lidar_offset_s,
        "odom_stamps": stamps,
        "odom_positions": positions,
        "odom_velocities": velocities,
        "odom_orientations_wxyz": quaternions,
        "acceleration_stamps": stamps.copy(),
        "accelerations": np.zeros_like(positions),
    }


def test_atomic_episode_write_validates_same_directory_archive(tmp_path, monkeypatch):
    path = tmp_path / "episode-000001-000000-000000.npz"
    real_link = current_data.os.link
    observed = {}

    def inspect_link(source, destination):
        observed["source"] = source
        observed["destination"] = destination
        assert tmp_path == type(tmp_path)(source).parent
        assert load_episode_env(source)["episode_id"].item() == "episode-test"
        real_link(source, destination)

    monkeypatch.setattr(current_data.os, "link", inspect_link)
    save_episode_env(str(path), _episode())

    assert observed["destination"] == str(path)
    assert not type(tmp_path)(observed["source"]).exists()
    assert load_episode_env(str(path))["episode_id"].item() == "episode-test"


def test_atomic_episode_write_never_overwrites_existing_episode(tmp_path):
    path = tmp_path / "episode-000001-000000-000000.npz"
    original = b"existing episode"
    path.write_bytes(original)

    with pytest.raises(FileExistsError):
        save_episode_env(str(path), _episode())

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_episode_write_failure_preserves_target(tmp_path, monkeypatch):
    path = tmp_path / "episode-000001-000000-000000.npz"
    original = b"existing episode"
    path.write_bytes(original)

    def fail_after_partial_write(target, **_arrays):
        target.write(b"partial replacement")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(current_data.np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_episode_env(str(path), _episode())

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_four_ms_lidar_offset_uses_nearest_odom_as_common_time_origin():
    windows = convert_episode_windows(_episode(lidar_offset_s=0.004))

    assert windows["observation_index"].tolist() == [0, 5]
    np.testing.assert_allclose(windows["anchor_stamp"], [10.0, 10.5])
    np.testing.assert_allclose(windows["dense_traj"][0, :, 0], 0.1 * np.arange(1, 21), atol=1e-6)
    np.testing.assert_allclose(windows["dense_traj"][0, :, 1:], 0.0)


def test_data_defaults_define_the_single_frame_twenty_point_contract():
    config = default_config()
    assert config["data"]["trajectory_samples"] == 20
    assert config["data"]["trajectory_dt"] * 20 == config["data"]["horizon"] == 2.0
    assert config["lidar"]["n_frames"] == 1


def test_dataset_validates_each_episode_once_at_loading(tmp_path, monkeypatch):
    path = tmp_path / "episode-000001-000000-000000.npz"
    save_episode_env(str(path), _episode())
    validated_paths = []
    validate = current_data.validate_episode_env

    def record_validation(path, data):
        validated_paths.append(path)
        validate(path, data)

    monkeypatch.setattr(current_data, "validate_episode_env", record_validation)
    dataset = EpisodeWindowDataset((str(path),))

    assert len(dataset) == 2
    assert validated_paths == [str(path)]


def test_windows_use_full_roll_pitch_yaw_body_rotation():
    payload = _episode()
    quaternion, rotation = _quaternion_and_rotation(math.radians(30.0), math.radians(-20.0), math.radians(40.0))
    stamps = payload["odom_stamps"]
    elapsed = stamps - stamps[0]
    origin_world = np.asarray([5.0, -2.0, 1.5])
    path_direction_body = np.asarray([0.7, -0.3, 0.25])
    velocity_body = np.asarray([1.1, -0.4, 0.2])
    acceleration_body = np.asarray([0.3, 0.1, -0.35])
    goal_body = np.asarray([6.0, 1.5, -0.8])

    payload["odom_positions"] = (origin_world + (elapsed[:, None] * path_direction_body[None]) @ rotation.T).astype(
        np.float32
    )
    payload["odom_velocities"] = np.broadcast_to(velocity_body @ rotation.T, (len(stamps), 3)).astype(np.float32).copy()
    payload["accelerations"] = (
        np.broadcast_to(acceleration_body @ rotation.T, (len(stamps), 3)).astype(np.float32).copy()
    )
    payload["odom_orientations_wxyz"] = np.broadcast_to(quaternion, (len(stamps), 4)).copy()
    payload["goal_world"] = (origin_world + goal_body @ rotation.T).astype(np.float32)

    windows = convert_episode_windows(payload)

    np.testing.assert_allclose(
        windows["state_body"][0], np.concatenate([velocity_body, acceleration_body, goal_body]), atol=2e-6
    )
    np.testing.assert_allclose(
        windows["dense_traj"][0], np.arange(1, 21)[:, None] * 0.1 * path_direction_body[None], atol=2e-6
    )
    np.testing.assert_allclose(windows["dense_velocity"][0], np.broadcast_to(velocity_body, (20, 3)), atol=2e-6)
    np.testing.assert_allclose(windows["dense_acceleration"][0], np.broadcast_to(acceleration_body, (20, 3)), atol=2e-6)


def test_dataset_shapes_and_episode_sampler_are_current(tmp_path):
    path = tmp_path / "episode-000001-000000-000000.npz"
    save_episode_env(str(path), _episode())
    dataset = EpisodeWindowDataset((str(path),))

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["obs"].shape == (1, 30, 180)
    assert sample["state"].shape == (13,)
    assert sample["state_body"].shape == (9,)
    assert sample["dense_traj"].shape == (20, 3)
    assert sample["dense_velocity"].shape == (20, 3)
    assert sample["dense_acceleration"].shape == (20, 3)
    assert sample["dense_traj_valid"].shape == (20,)
    assert sample["dense_traj_valid"].all()
    assert sample["v_end"].shape == (3,)
    assert sample["a_end"].shape == (3,)

    validate_lidar_compatibility(dataset, default_config()["lidar"])
    incompatible = default_config()["lidar"]
    incompatible["r_max"] = 30.0
    with pytest.raises(ValueError, match="disagrees"):
        validate_lidar_compatibility(dataset, incompatible)

    labels = np.asarray([0, 0, 0, 1, 2, 2])
    first = EpisodeUniformSampler(labels, seed=42)
    second = EpisodeUniformSampler(labels, seed=42)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)


def _split_manifest(train=None, val=None, test=None):
    return {
        "split": {
            "train": train or ["episode-000001-000000-000000.npz"],
            "val": val or ["episode-000002-000000-000000.npz"],
            "test": test or ["episode-000003-000000-000000.npz"],
        }
    }


def test_manifest_requires_nonempty_disjoint_episode_basenames(tmp_path):
    first = "episode-000001-000000-000000.npz"
    invalid = [
        {"split": {"train": [], "val": [first], "test": [first]}},
        {"split": {"train": first, "val": [first], "test": [first]}},
        _split_manifest(train=[1]),
        _split_manifest(train=["not-an-episode.npz"]),
        _split_manifest(train=["/tmp/" + first]),
        _split_manifest(train=["../" + first]),
        _split_manifest(train=[first, first]),
        _split_manifest(train=[first], val=[first]),
    ]
    for index, payload in enumerate(invalid):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_manifest(path)


def test_split_episode_files_revalidates_and_resolves_basenames(tmp_path):
    manifest = _split_manifest()
    for names in manifest["split"].values():
        (tmp_path / names[0]).touch()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_manifest(path)
    assert split_episode_files(loaded, tmp_path, "train") == (str(tmp_path / manifest["split"]["train"][0]),)
    unsafe = _split_manifest(train=["../episode-000001-000000-000000.npz"])
    with pytest.raises(ValueError, match="basenames"):
        split_episode_files(unsafe, tmp_path, "train")


def test_fixed_episode_audit_uses_current_boundary_grid(tmp_path):
    path = tmp_path / "episode-000001-000000-000000.npz"
    save_episode_env(str(path), _episode())

    report = audit_episode_file(str(path))

    assert report["windows"] == 2
    np.testing.assert_allclose(report["boundary_implied_jerk"], 0.0, atol=2e-4)
