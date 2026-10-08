#!/usr/bin/env python3
"""Filter independent episode files by dynamics and trajectory-quality thresholds."""

import argparse
import glob
import os
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from neural_minco.config import ACCELERATION_LIMIT_MPS2, VELOCITY_LIMIT_MPS
from neural_minco.data import load_episode_env, save_episode_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, help="episode file or directory")
    parser.add_argument("--out", required=True, help="output directory for retained episodes")
    parser.add_argument("--max-vel", type=float, default=VELOCITY_LIMIT_MPS, help="maximum speed norm")
    parser.add_argument("--max-acc", type=float, default=ACCELERATION_LIMIT_MPS2, help="maximum acceleration norm")
    parser.add_argument(
        "--max-jerk", type=float, default=None, help="maximum adjacent-command average jerk norm; disabled by default"
    )
    parser.add_argument("--min-lidar", type=int, default=1, help="minimum LiDAR frame count")
    parser.add_argument("--min-hit-frac", type=float, default=0.0, help="minimum mean fraction of occupied LiDAR bins")
    parser.add_argument("--keep-success-only", action="store_true", help="retain only successful episodes")
    parser.add_argument("--max-duration", type=float, default=None, help="maximum duration in seconds")
    parser.add_argument(
        "--max-endpoint-gap", type=float, default=None, help="maximum final planar distance from the global goal"
    )
    parser.add_argument(
        "--min-jerk-dt", type=float, default=1e-3, help="ignore jerk estimates below this time interval"
    )
    return parser.parse_args()


def _episode_metrics(data: dict, args: argparse.Namespace) -> dict:
    stamps = np.asarray(data["odom_stamps"], dtype=np.float64)
    positions = np.asarray(data["odom_positions"], dtype=np.float64)
    velocities = np.asarray(data["odom_velocities"], dtype=np.float64)
    accelerations = np.asarray(data["accelerations"], dtype=np.float64)
    lidar = np.asarray(data["lidar_range_images"])
    speed = np.linalg.norm(velocities, axis=1)
    acceleration = np.linalg.norm(accelerations, axis=1)
    arc_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    endpoint_gap = float(np.linalg.norm(positions[-1, :2] - data["goal_world"][:2]))
    duration = float(stamps[-1] - stamps[0])
    hit_fraction = float((lidar < 1.0).mean()) if len(lidar) else 0.0
    jerk_max = 0.0
    if args.max_jerk is not None and len(accelerations) >= 2:
        if not np.isfinite(args.min_jerk_dt) or args.min_jerk_dt <= 0:
            raise ValueError("min_jerk_dt must be positive and finite")
        command_stamps = np.asarray(data["acceleration_stamps"], dtype=np.float64)
        dt = np.diff(command_stamps)
        delta_acceleration = np.linalg.norm(np.diff(accelerations, axis=0), axis=1)
        # A repeated command may be paired with multiple odometry rows. A
        # conflicting repeat or reversed timestamp is not evidence of high jerk.
        if np.any(dt < 0) or np.any((dt == 0) & (delta_acceleration > 1e-6)):
            raise ValueError("ambiguous acceleration_stamps; inspect timing before jerk filtering")
        valid = dt >= args.min_jerk_dt
        if np.any(valid):
            jerk = delta_acceleration[valid] / dt[valid]
            jerk_max = float(jerk.max())
    return {
        "vmax": float(speed.max()),
        "amax": float(acceleration.max()),
        "jerk_max": jerk_max,
        "duration": duration,
        "arc_length": arc_length,
        "endpoint_gap": endpoint_gap,
        "lidar_count": len(lidar),
        "hit_fraction": hit_fraction,
    }


def _drop_reasons(data: dict, metrics: dict, args: argparse.Namespace) -> List[str]:
    reasons = []
    if args.keep_success_only and not bool(data["success"]):
        reasons.append("not_success")
    if args.max_vel is not None and metrics["vmax"] > args.max_vel:
        reasons.append("max_vel")
    if args.max_acc is not None and metrics["amax"] > args.max_acc:
        reasons.append("max_acc")
    if args.max_jerk is not None and metrics["jerk_max"] > args.max_jerk:
        reasons.append("max_jerk")
    if args.max_duration is not None and metrics["duration"] > args.max_duration:
        reasons.append("max_duration")
    if args.max_endpoint_gap is not None and metrics["endpoint_gap"] > args.max_endpoint_gap:
        reasons.append("max_endpoint_gap")
    if metrics["lidar_count"] < args.min_lidar:
        reasons.append("min_lidar")
    if metrics["hit_fraction"] < args.min_hit_frac:
        reasons.append("min_hit")
    return reasons


def _input_files(source: str) -> List[str]:
    path = Path(source)
    if path.is_file():
        return [str(path)]
    return sorted(glob.glob(str(path / "episode-*.npz")))


def main() -> None:
    args = _parse_args()
    files = _input_files(args.source)
    if not files:
        raise SystemExit(f"no episode-*.npz found at {args.source}")
    os.makedirs(args.out, exist_ok=True)

    counts = Counter()
    kept = 0
    for source in files:
        data = load_episode_env(source)
        metrics = _episode_metrics(data, args)
        reasons = _drop_reasons(data, metrics, args)
        if reasons:
            counts.update(reasons)
            print(f"drop {data['episode_id'].item()}: {','.join(reasons)}")
            continue
        output = os.path.join(args.out, f"{data['episode_id'].item()}.npz")
        save_episode_env(output, data)
        kept += 1

    print(f"episodes: kept={kept} dropped={len(files) - kept}/{len(files)}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    if not kept:
        raise SystemExit("no episodes remain after cleaning")


if __name__ == "__main__":
    main()
