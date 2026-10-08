"""Focused tests for causal planning-event sensor synchronization."""

import numpy as np

from scripts.data_collect.collect_utils.sync import (
    first_synchronized_command,
    frame_stream_metrics,
    has_reached_goal,
    is_post_goal_position_trajectory,
    nearest_frame,
    resolve_nearest_frame,
    trajectory_command_cutoff,
)


def _odom(stamp, x, quat=(1.0, 0.0, 0.0, 0.0)):
    return (
        float(stamp),
        np.asarray([x, 0.0, 0.0], dtype=np.float64),
        quat,
        np.asarray([2.0 * x, 0.0, 0.0], dtype=np.float64),
        np.asarray([3.0 * x, 0.0, 0.0], dtype=np.float64),
    )


def test_nearest_frame_returns_one_unmodified_message():
    before = _odom(9.98, 1.0)
    after = _odom(10.01, 3.0)

    selected = nearest_frame([before, after], 10.0, max_delta=0.02)

    assert selected is after
    np.testing.assert_array_equal(selected[1], after[1])


def test_nearest_frame_waits_until_stream_reaches_target():
    before = _odom(9.99, 1.0)

    assert nearest_frame([before], 10.0, max_delta=0.02) is None
    assert nearest_frame([before], 10.0, max_delta=0.02, wait_for_target=False) is before
    assert nearest_frame([before], 10.1, max_delta=0.02, wait_for_target=False) is None


def test_resolve_nearest_frame_waits_for_its_own_stream_watermark():
    before = _odom(9.997, 1.0)

    assert resolve_nearest_frame([before], 10.0, max_delta=0.005) == (None, False)
    frame, resolved = resolve_nearest_frame([before, _odom(10.007, 2.0)], 10.0, max_delta=0.005)
    assert resolved
    assert frame is before


def test_resolve_nearest_frame_distinguishes_resolved_mismatch():
    frame, resolved = resolve_nearest_frame([_odom(9.99, 1.0), _odom(10.01, 2.0)], 10.0, max_delta=0.005)

    assert resolved
    assert frame is None


def test_frame_stream_metrics_distinguishes_ten_hz_from_degraded_stream():
    rate, largest_gap = frame_stream_metrics([10.0, 10.1, 10.2, 10.3])

    np.testing.assert_allclose(rate, 10.0, atol=1e-12)
    np.testing.assert_allclose(largest_gap, 0.1, atol=1e-12)
    degraded_rate, degraded_gap = frame_stream_metrics([10.0, 11.0, 12.0])
    np.testing.assert_allclose(degraded_rate, 1.0, atol=1e-12)
    np.testing.assert_allclose(degraded_gap, 1.0, atol=1e-12)
    assert frame_stream_metrics([10.0]) == (0.0, float("inf"))


def test_post_goal_position_trajectory_requires_complete_post_goal_message():
    common = {
        "goal_stamp": 10.0,
        "message_stamp": 10.2,
        "trajectory_start_stamp": 10.1,
        "piece_count": 3,
        "message_type": 3,
        "position_type": 2,
    }

    assert is_post_goal_position_trajectory(**common)
    assert not is_post_goal_position_trajectory(**{**common, "piece_count": 0})
    assert not is_post_goal_position_trajectory(**{**common, "message_type": 1})
    assert not is_post_goal_position_trajectory(**{**common, "message_stamp": 9.99})
    assert not is_post_goal_position_trajectory(**{**common, "trajectory_start_stamp": 9.99})


def test_trajectory_command_cutoff_uses_later_source_stamp_and_guard():
    np.testing.assert_allclose(trajectory_command_cutoff(10.2, 10.1, 0.02), 10.22)
    np.testing.assert_allclose(trajectory_command_cutoff(10.1, 10.2, 0.02), 10.22)


def test_trajectory_command_cutoff_rejects_negative_guard():
    with np.testing.assert_raises(ValueError):
        trajectory_command_cutoff(10.1, 10.2, -0.01)


def test_first_synchronized_command_skips_precommit_and_unmatched_entries():
    def command(stamp):
        return (stamp, None, None, None, None)

    commands = [command(10.23), command(10.22), command(10.21), command(9.99)]
    odom = [_odom(10.234, 1.0), _odom(10.40, 2.0)]

    selected = first_synchronized_command(commands, odom, not_before=10.22, max_delta=0.005)

    assert selected[0] is commands[0]
    assert selected[1] is odom[0]
    assert first_synchronized_command(commands, odom, not_before=10.24, max_delta=0.005) is None


def test_goal_reach_requires_3d_position_and_low_speed():
    goal = np.asarray([1.0, 2.0, 3.0])

    assert has_reached_goal([1.05, 2.0, 3.05], [0.10, 0.0, 0.10], goal, distance_tolerance=0.1, speed_tolerance=0.2)
    assert not has_reached_goal([1.0, 2.0, 3.0], [0.21, 0.0, 0.0], goal, distance_tolerance=0.1, speed_tolerance=0.2)
    assert not has_reached_goal([1.0, 2.0, 3.11], [0.0, 0.0, 0.0], goal, distance_tolerance=0.1, speed_tolerance=0.2)
