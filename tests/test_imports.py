"""Import the current package surface and exercise every non-ROS CLI help path."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import torch

import neural_minco.evaluate as evaluation
import neural_minco.runtime as runtime
from neural_minco.config import (
    ACCELERATION_LIMIT_MPS2,
    COLLISION_RADIUS_M,
    JERK_LIMIT_MPS3,
    SENSOR_SYNC_TOLERANCE_S,
    VELOCITY_LIMIT_MPS,
    default_config,
)
from neural_minco.data import MAX_SENSOR_DELTA_S
from neural_minco.model import build_model, decode_action

CURRENT_MODULES = ("config", "model", "minco", "lidar", "geometry", "data", "train", "evaluate", "runtime")


def test_current_modules_import_explicitly():
    for module in CURRENT_MODULES:
        importlib.import_module(f"neural_minco.{module}")


def test_default_config_and_physical_state_forward():
    config = default_config()
    assert set(config) == {"model", "lidar", "data", "loss", "training"}
    assert config["model"]["refiner_windows"] == [[9, 17], [5, 9], [3, 5]]
    assert config["model"]["horizon_s"] == 2.0
    assert config["training"]["epochs"] == 10
    assert set(config["data"]) == {"horizon", "trajectory_dt", "trajectory_samples", "anchor_stride"}
    assert config["model"]["clearance_radius"] == COLLISION_RADIUS_M
    assert config["model"]["velocity_scale"] == VELOCITY_LIMIT_MPS
    assert config["model"]["acceleration_scale"] == ACCELERATION_LIMIT_MPS2
    assert config["model"]["jerk_scale"] == JERK_LIMIT_MPS3
    assert evaluation.COLLISION_RADIUS_M == COLLISION_RADIUS_M
    assert (runtime.V_MAX, runtime.A_MAX, runtime.J_MAX) == (
        VELOCITY_LIMIT_MPS,
        ACCELERATION_LIMIT_MPS2,
        JERK_LIMIT_MPS3,
    )
    assert runtime.MAX_SENSOR_DELTA_S == MAX_SENSOR_DELTA_S == SENSOR_SYNC_TOLERANCE_S

    model = build_model(config).eval()
    observation = torch.ones(1, 1, 30, 180)
    physical_state = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]])
    with torch.inference_mode():
        raw = model(observation, physical_state)
        Q, T, p_end, v_end, a_end = decode_action(
            raw,
            waypoint_scale=model.waypoint_scale,
            terminal_velocity_scale=model.v_end_scale,
            terminal_acceleration_scale=model.acceleration_scale,
            horizon=model.horizon,
        )
    assert Q.shape == (1, 2, 3)
    assert T.shape == (1, 3)
    assert p_end.shape == v_end.shape == a_end.shape == (1, 3)
    assert torch.isfinite(Q).all()
    torch.testing.assert_close(T.sum(dim=1), torch.tensor([2.0]))


def test_data_module_import_does_not_load_model_or_runtime():
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    python_path = str(root / "src")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    code = (
        "import sys; import neural_minco.data; "
        "assert 'neural_minco.model' not in sys.modules; "
        "assert 'neural_minco.runtime' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=environment, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_current_cli_help_paths_do_not_import_ros_or_matplotlib(tmp_path):
    root = Path(__file__).parents[1]
    (tmp_path / "matplotlib.py").write_text(
        "raise ImportError('matplotlib intentionally unavailable')\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    python_path = os.pathsep.join((str(tmp_path), str(root / "src")))
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path

    for name in ("train.py", "evaluate.py", "infer.py", "visualize.py"):
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / name), "--help"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
