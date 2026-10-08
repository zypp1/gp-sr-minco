"""Episode cleaning defaults and command-timestamp jerk estimates."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data_collect import clean


def test_cleaning_defaults_match_collection_flight_envelope(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["clean.py", "--in", "data", "--out", "data-clean"])

    args = clean._parse_args()

    assert args.max_vel == 6.0
    assert args.max_acc == 6.0
    assert args.max_jerk is None


def _metrics(odom_stamps, command_stamps, acceleration, *, max_jerk=12.0, min_jerk_dt=1e-3):
    args = SimpleNamespace(max_jerk=max_jerk, min_jerk_dt=min_jerk_dt)
    count = len(odom_stamps)
    episode = {
        "odom_stamps": np.asarray(odom_stamps),
        "acceleration_stamps": np.asarray(command_stamps),
        "accelerations": np.asarray(acceleration),
        "odom_positions": np.zeros((count, 3)),
        "odom_velocities": np.zeros((count, 3)),
        "lidar_range_images": np.ones((1, 1, 30, 180)),
        "goal_world": np.zeros(3),
    }
    return clean._episode_metrics(episode, args)


def test_jerk_uses_command_intervals_not_odometry_intervals():
    acceleration = [[0, 0, 0], [0.12, 0, 0]]
    metrics = _metrics([0.003, 0.007], [0, 0.01], acceleration)

    assert metrics["jerk_max"] == pytest.approx(12.0)
    assert metrics["jerk_max"] != pytest.approx(30.0)


def test_jerk_does_not_hide_shorter_command_intervals():
    metrics = _metrics([0, 0.01], [0.002, 0.007], [[0, 0, 0], [0.12, 0, 0]])

    assert metrics["jerk_max"] == pytest.approx(24.0)


def test_identical_duplicate_commands_do_not_divide_by_zero():
    metrics = _metrics([0, 0.004, 0.01], [0, 0, 0.01], [[0, 0, 0], [0, 0, 0], [0.12, 0, 0]])

    assert metrics["jerk_max"] == pytest.approx(12.0)


@pytest.mark.parametrize("command_stamps", [[0, 0], [0.004, 0.002]])
def test_ambiguous_command_timing_aborts_instead_of_rejecting_episode(command_stamps):
    with pytest.raises(ValueError, match="ambiguous acceleration_stamps"):
        _metrics([0, 0.004], command_stamps, [[0, 0, 0], [0.01, 0, 0]])


def test_tiny_command_intervals_are_excluded():
    metrics = _metrics([0, 0.001], [0, 0.0005], [[0, 0, 0], [0.12, 0, 0]])

    assert metrics["jerk_max"] == 0.0


@pytest.mark.parametrize("min_jerk_dt", [0, -1, float("nan"), float("inf")])
def test_jerk_minimum_interval_must_be_positive_and_finite(min_jerk_dt):
    with pytest.raises(ValueError, match="min_jerk_dt"):
        _metrics([0, 0.01], [0, 0.01], [[0, 0, 0], [0.12, 0, 0]], min_jerk_dt=min_jerk_dt)


def test_disabled_jerk_filter_does_not_interpret_ambiguous_timing():
    metrics = _metrics([0, 0.004], [0, 0], [[0, 0, 0], [0.01, 0, 0]], max_jerk=None)

    assert metrics["jerk_max"] == 0.0


def test_jerk_is_the_norm_of_the_world_acceleration_difference():
    metrics = _metrics([0, 0.01], [0, 0.01], [[1, 2, 3], [1.03, 2.04, 3]])

    assert metrics["jerk_max"] == pytest.approx(5.0)


def test_jerk_limit_comparison_is_strict(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["clean.py", "--in", "data", "--out", "data-clean", "--max-jerk", "12"])
    args = clean._parse_args()
    metrics = _metrics([0, 0.01], [0, 0.01], [[0, 0, 0], [0.12, 0, 0]])

    assert clean._drop_reasons({}, metrics, args) == []
    metrics["jerk_max"] = 12.001
    assert clean._drop_reasons({}, metrics, args) == ["max_jerk"]
