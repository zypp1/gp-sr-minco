"""Report quality and synchronization metrics for independent episode files."""

import argparse
import glob
import os

import numpy as np

from neural_minco.config import JERK_LIMIT_MPS3
from neural_minco.data import (
    ANCHOR_STRIDE_FRAMES,
    HORIZON_S,
    TRAJECTORY_DT_S,
    TRAJECTORY_SAMPLES,
    WINDOW_TIME_ORIGIN,
    convert_episode_windows,
    load_episode_env,
)


def _percentiles(values, percentiles=(0, 10, 50, 90, 99, 100)) -> str:
    """Format selected percentiles from one numeric sequence."""
    values = np.asarray(values)
    if not len(values):
        return "none"
    return " ".join(f"p{p}={np.percentile(values, p):.2f}" for p in percentiles)


def _nearest_deltas(query: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(stamps, query, side="left"), 0, len(stamps) - 1)
    left = np.clip(right - 1, 0, len(stamps) - 1)
    return np.minimum(np.abs(query - stamps[left]), np.abs(stamps[right] - query))


def audit_episode_file(path: str) -> dict:
    """Return one episode's metrics and eligible observation-window count."""
    data = load_episode_env(path)
    positions = np.asarray(data["odom_positions"], dtype=np.float64)
    velocities = np.asarray(data["odom_velocities"], dtype=np.float64)
    accelerations = np.asarray(data["accelerations"], dtype=np.float64)
    odom_stamps = np.asarray(data["odom_stamps"], dtype=np.float64)
    accel_stamps = np.asarray(data["acceleration_stamps"], dtype=np.float64)
    lidar_stamps = np.asarray(data["lidar_stamps"], dtype=np.float64)
    lidar = np.asarray(data["lidar_range_images"])
    try:
        windows = convert_episode_windows(data)
        n_windows = len(windows["obs"])
        dt = TRAJECTORY_DT_S
        residual = (
            windows["dense_traj"][:, 0]
            - windows["state_body"][:, :3] * dt
            - 0.5 * windows["state_body"][:, 3:6] * dt**2
        )
        boundary_implied_jerk = 6.0 * np.linalg.norm(residual, axis=-1) / dt**3
    except ValueError as error:
        if "no complete trajectory window" not in str(error):
            raise
        n_windows = 0
        boundary_implied_jerk = np.empty(0, dtype=np.float64)
    hit_fraction = (lidar < 1.0).reshape(len(lidar), -1).mean(axis=1) if len(lidar) else np.empty(0, dtype=np.float64)
    lidar_gaps = np.diff(lidar_stamps)
    lidar_rate = float((len(lidar_stamps) - 1) / (lidar_stamps[-1] - lidar_stamps[0])) if len(lidar_stamps) > 1 else 0.0
    return {
        "success": np.asarray([bool(data["success"])]),
        "arc_length": np.asarray([np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()]),
        "duration": np.asarray([odom_stamps[-1] - odom_stamps[0]]),
        "endpoint_gap": np.asarray([np.linalg.norm(positions[-1, :2] - data["goal_world"][:2])]),
        "vmax": np.asarray([np.linalg.norm(velocities, axis=1).max()]),
        "amax": np.asarray([np.linalg.norm(accelerations, axis=1).max()]),
        "odom_samples": np.asarray([len(odom_stamps)]),
        "lidar_frames": np.asarray([len(lidar_stamps)]),
        "lidar_rate": np.asarray([lidar_rate]),
        "lidar_max_gap": np.asarray([float(lidar_gaps.max()) if len(lidar_gaps) else float("inf")]),
        "lidar_dt": lidar_gaps,
        "odom_dt": np.diff(odom_stamps),
        "lidar_odom_delta": _nearest_deltas(lidar_stamps, odom_stamps),
        "odom_acceleration_delta": np.abs(odom_stamps - accel_stamps),
        "hit_fraction": hit_fraction,
        "windows": n_windows,
        "boundary_implied_jerk": boundary_implied_jerk,
    }


def main():
    """Parse arguments and print the combined episode and window-conversion report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data", help="directory containing episode-*.npz")
    parser.add_argument(
        "--max-jerk", type=float, default=JERK_LIMIT_MPS3, help="first-step implied-jerk threshold in m/s^3"
    )
    parser.add_argument(
        "--min-lidar-rate", type=float, default=9.5, help="minimum acceptable effective LiDAR rate in Hz"
    )
    parser.add_argument(
        "--max-lidar-gap", type=float, default=0.15, help="maximum acceptable gap between LiDAR frames in seconds"
    )
    args = parser.parse_args()
    if args.min_lidar_rate <= 0.0 or args.max_lidar_gap <= 0.0 or args.max_jerk <= 0.0:
        raise SystemExit("LiDAR rate and gap thresholds must be positive")
    files = sorted(glob.glob(os.path.join(args.data, "episode-*.npz")))
    if not files:
        raise SystemExit(f"no episode-*.npz in {args.data}")
    reports = [audit_episode_file(path) for path in files]
    fields = (
        "success",
        "arc_length",
        "duration",
        "endpoint_gap",
        "vmax",
        "amax",
        "odom_samples",
        "lidar_frames",
        "lidar_rate",
        "lidar_max_gap",
        "lidar_dt",
        "odom_dt",
        "hit_fraction",
        "lidar_odom_delta",
        "odom_acceleration_delta",
        "boundary_implied_jerk",
    )
    combined = {name: np.concatenate([report[name] for report in reports]) for name in fields}
    success = combined["success"]
    print(f"\n===== Episode trajectory quality report =====  ({len(files)} files)")
    print(f"successful: {success.sum()}/{len(success)} ({success.mean() * 100:.1f}%)")
    for name, unit in (
        ("arc_length", "m"),
        ("duration", "s"),
        ("endpoint_gap", "m"),
        ("vmax", "m/s"),
        ("amax", "m/s^2"),
        ("odom_samples", "samples"),
        ("lidar_frames", "frames"),
        ("lidar_rate", "Hz"),
        ("lidar_max_gap", "s"),
    ):
        print(f"successful {name} ({unit}): {_percentiles(combined[name][success])}")
    lidar_quality = (combined["lidar_rate"] >= args.min_lidar_rate) & (combined["lidar_max_gap"] <= args.max_lidar_gap)
    print(
        f"successful LiDAR quality: {(success & lidar_quality).sum()}/{success.sum()} "
        f"(rate >= {args.min_lidar_rate:.2f} Hz, max gap <= {args.max_lidar_gap:.3f}s)"
    )
    print(f"LiDAR period (ms): {_percentiles(1e3 * combined['lidar_dt'])}")
    print(f"odom period (ms): {_percentiles(1e3 * combined['odom_dt'])}")
    print(f"LiDAR-nearest odom delta (ms): {_percentiles(1e3 * combined['lidar_odom_delta'])}")
    print(f"odom-nearest acceleration delta (ms): {_percentiles(1e3 * combined['odom_acceleration_delta'])}")
    print(f"LiDAR hit fraction: {_percentiles(combined['hit_fraction'])}")
    boundary_jerk = combined["boundary_implied_jerk"]
    print(f"first-step implied jerk (m/s^3): {_percentiles(boundary_jerk)}")
    if len(boundary_jerk):
        print(
            f"first-step implied jerk > {args.max_jerk:.2f}: "
            f"{int((boundary_jerk > args.max_jerk).sum())}/{len(boundary_jerk)} "
            f"({100.0 * np.mean(boundary_jerk > args.max_jerk):.2f}%)"
        )
    else:
        print(f"first-step implied jerk > {args.max_jerk:.2f}: N/A")
    n_windows = sum(report["windows"] for report in reports)
    print(
        f"eligible windows: {n_windows} "
        f"(horizon={HORIZON_S:.2f}s, stride={ANCHOR_STRIDE_FRAMES}, "
        f"traj_samples={TRAJECTORY_SAMPLES}, traj_dt={TRAJECTORY_DT_S:.3f}s, "
        f"time_origin={WINDOW_TIME_ORIGIN}, "
        "require_full_horizon=True)\n"
    )


if __name__ == "__main__":
    main()
