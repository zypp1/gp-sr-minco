"""Timestamp selection for event-driven collection."""

import numpy as np


def nearest_frame(entries, target_stamp: float, max_delta: float, *, wait_for_target: bool = True):
    """Return one real frame nearest to a timestamp without interpolating its values."""
    if max_delta < 0.0:
        raise ValueError("max_delta must be nonnegative")
    if not entries:
        return None
    ordered = sorted(entries, key=lambda entry: entry[0])
    if wait_for_target and ordered[-1][0] < target_stamp:
        return None
    selected = min(ordered, key=lambda entry: (abs(entry[0] - target_stamp), entry[0]))
    if abs(selected[0] - target_stamp) > max_delta:
        return None
    return selected


def resolve_nearest_frame(entries, target_stamp: float, max_delta: float):
    """Return ``(frame, resolved)`` once a monotonic stream has reached the target time."""
    if not entries:
        return None, False
    if max(entry[0] for entry in entries) < target_stamp:
        return None, False
    return nearest_frame(entries, target_stamp, max_delta, wait_for_target=False), True


def frame_stream_metrics(stamps):
    """Return effective frame rate and largest gap for a monotonic timestamp stream."""
    stamps = np.asarray(stamps, dtype=np.float64)
    if stamps.ndim != 1:
        raise ValueError("frame timestamps must be one-dimensional")
    if len(stamps) < 2:
        return 0.0, float("inf")
    gaps = np.diff(stamps)
    if not np.isfinite(stamps).all() or np.any(gaps <= 0.0):
        raise ValueError("frame timestamps must be finite and strictly increasing")
    return float((len(stamps) - 1) / (stamps[-1] - stamps[0])), float(gaps.max())


def is_post_goal_position_trajectory(
    goal_stamp: float,
    message_stamp: float,
    trajectory_start_stamp: float,
    piece_count: int,
    message_type: int,
    position_type: int,
) -> bool:
    """Return whether a complete position trajectory was committed after a goal."""
    return (
        piece_count > 0
        and bool(message_type & position_type)
        and message_stamp >= goal_stamp
        and trajectory_start_stamp >= goal_stamp
    )


def trajectory_command_cutoff(message_stamp: float, trajectory_start_stamp: float, guard_time: float) -> float:
    """Return the first safe command time after a committed trajectory publication."""
    if guard_time < 0.0:
        raise ValueError("guard_time must be nonnegative")
    return max(float(message_stamp), float(trajectory_start_stamp)) + float(guard_time)


def first_synchronized_command(command_entries, odom_entries, not_before: float, max_delta: float):
    """Return the earliest post-boundary command with a nearby real odometry frame."""
    eligible = sorted((entry for entry in command_entries if entry[0] >= not_before), key=lambda entry: entry[0])
    for command in eligible:
        odom = nearest_frame(odom_entries, command[0], max_delta)
        if odom is not None:
            return command, odom
    return None


def has_reached_goal(position, velocity, goal, distance_tolerance: float, speed_tolerance: float) -> bool:
    """Return whether the vehicle is both spatially and dynamically settled at a goal."""
    if distance_tolerance < 0.0 or speed_tolerance < 0.0:
        raise ValueError("goal tolerances must be nonnegative")
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return bool(np.linalg.norm(position - goal) <= distance_tolerance and np.linalg.norm(velocity) <= speed_tolerance)
