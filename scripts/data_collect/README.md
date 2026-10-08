# Episode Data Collection

One `episode-*.npz` stores one complete fixed-goal flight. Collection uses SUPER
as the expert and does not import the neural planner model.

| Path | Responsibility |
| --- | --- |
| `collect.py` | Synchronized Episode recorder |
| `goals.py` | Collision-aware goal publisher |
| `validate.py` / `clean.py` | Read-only audit / explicit copy-based filtering |
| `collect_utils/sync.py` | Timestamp matching |
| `pcl_tools/` | Map generation and conversion |
| `super_config/` | Simulator, SUPER and ROS launch configuration |

## Episode contract

Files contain normalized range images and LiDAR timestamps, world
position/velocity, full `wxyz` attitude, command acceleration and its timestamps,
goal, success flag and map identity. Writes use a new
`episode-<sequence>-<YYMMDD>-<HHMMSS>[-tag].npz` name atomically; existing
Episodes are never replaced.

LiDAR, odometry and command rows retain their real timestamps. Collection does
not interpolate them; matching uses a `5 ms` tolerance. Dataset conversion
anchors state and future targets at the nearest odometry timestamp. See the
[fixed window contract](../../docs/training.md#episode-windows) for the
two-second, stride-five, 20-point targets.

## Simulator setup

Apply `super_config/super_mid360.patch` once to SUPER and rebuild its workspace.
The retained Mid-360 configuration produces `10 Hz` scans. Sensor geometry
comes from `default_config()["lidar"]`; the native image has `30x180` cells,
`20 m` range and one frame.

`super_config/nmp.yaml` defines the expert command envelope: `6 m/s`,
`6 m/s²`, `12 m/s³`, `2.0 rad/s` body rate and `2.5 rad/s` yaw rate.
PerfectDrone performs ideal command playback; these limits do not model
tracking error, latency, wind or actuator dynamics.

Install matching simulator PCD and goal-sampling NPZ/JSON files as described in
the [simulation handoff](../../docs/runtime.md#simulation-handoff). The launcher
uses the versioned simulator and SUPER YAMLs unless explicitly overridden.

## Collect

Run from the repository root after sourcing ROS/SUPER and installing the
[planner environment](../../README.md#installation):

```bash
export SUPER_WS=/absolute/path/to/super_ws
export PLANNER_MAP=/absolute/path/to/example_scene.npz
export EPISODE_OUT=/absolute/path/to/episodes
export EPISODE_MAX=500
bash scripts/run/collect.sh
```

Keep the NPZ and JSON sidecar together; the latter supplies bounds and building
volumes for goal sampling. Launcher overrides include `PLANNER_SIM_CONFIG`,
`PLANNER_SUPER_CONFIG`, `EPISODE_TIMEOUT`, `GOAL_TOLERANCE`,
`GOAL_SPEED_TOLERANCE`, `NEXT_GOAL_DELAY`, `MIN_LIDAR_RATE` and
`MAX_LIDAR_GAP`. Use each CLI's `--help` for quality and goal-generation controls.

Success requires reaching the full 3D goal with low speed and passing the
saved LiDAR rate/gap checks. Failed or low-quality flights are retained as
diagnostics and are not eligible training Episodes.

## Validate and clean

Audit without modifying raw files, then optionally create a filtered copy in
a distinct destination:

```bash
PYTHONPATH=src python scripts/data_collect/validate.py \
  --data /absolute/path/to/episodes

PYTHONPATH=src python scripts/data_collect/clean.py \
  --in /absolute/path/to/episodes \
  --out /absolute/path/to/episodes-clean \
  --keep-success-only
```

The audit uses the same fixed window conversion as training. Cleaning defaults
enforce the current speed/acceleration envelope. Jerk filtering is opt-in:
it differentiates world command acceleration on `acceleration_stamps`, skips
identical repeated commands, and stops on conflicting repeats or reversed
timestamps. This interval-average estimate does not certify continuous peak
jerk. Cleaning neither deletes sources nor changes a training manifest.
Data-contract changes must update collection, validation, cleaning, loading,
documentation and tests together.

## Map tools

Generate a deterministic PCD/NPZ/JSON set:

```bash
python scripts/data_collect/pcl_tools/pcl_generate.py \
  --scene mixed --output output/maps/example_scene
```

PCD and NPZ are written from the same ordered float32 point array. MARSIM
consumes the PCD; goal sampling and offline evaluation consume the NPZ.
The JSON sidecar records generation settings, bounds, building volumes and
point count.

Convert existing clouds when needed:

```bash
PYTHONPATH=src python scripts/data_collect/pcl_tools/pcl_convert.py \
  --in input.ply \
  --pcd-out output/maps/converted.pcd \
  --npz-out output/maps/converted.npz
```

Use `--help` for formats and generation controls. Replacing existing map
outputs requires `--force`.
