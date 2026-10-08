# Online Runtime

`src/neural_minco/runtime.py` owns ROS-free prediction, the persistent spawned
inference worker, world trajectories, execution state and command sampling.
`scripts/infer_node.py` handles ROS transport, visualization and opt-in
publication. It uses the same final-`z3` prediction as offline inference, with
one scan and no online optimizer.

## ROS interfaces

| Topic | Type | Purpose |
| --- | --- | --- |
| `/cloud_registered` | `sensor_msgs/PointCloud2` | Registered world-frame cloud; drives planning |
| `/lidar_slam/odom` | `nav_msgs/Odometry` | World pose and velocity |
| `/planning/click_goal` | `geometry_msgs/PoseStamped` | World-frame goal |
| `/planning/pos_cmd` | `quadrotor_msgs/PositionCommand` | Optional world-frame command |

The direct node publishes commands only with `--execute`. When enabled,
SUPER's planner FSM must be stopped: one process must own
`/planning/pos_cmd`.

## Direct node invocation

Prepare the [simulator workspace](../scripts/data_collect/README.md#simulator-setup).
From the repository root, source ROS/SUPER and select this Python checkout:

```bash
source /opt/ros/noetic/setup.bash
source /absolute/path/to/super_ws/devel/setup.bash
conda activate planner
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Shadow mode: inference and visualization, no command publication.
python scripts/infer_node.py \
  --checkpoint /absolute/path/to/ckpt_best.pth --device auto
```

After the handoff checks below, enable closed-loop simulator commands explicitly:

```bash
python scripts/infer_node.py \
  --checkpoint /absolute/path/to/ckpt_best.pth \
  --device cuda --execute --command-rate 100
```

`--no-viz` disables RViz output; `--no-lidar-viz` keeps trajectories while
disabling the LiDAR panel.

## Integrated launcher

`scripts/run/infer.sh` opens simulator, goals and inference tabs and passes
`--execute`. Use the direct node above for shadow mode.

```bash
export PLANNER_CHECKPOINT=/absolute/path/to/ckpt_best.pth
export SUPER_WS=/absolute/path/to/super_ws
export PLANNER_MAP=/absolute/path/to/example_scene.npz
bash scripts/run/infer.sh
```

| Variable | Meaning |
| --- | --- |
| `PLANNER_CHECKPOINT` | Required current GP-SR checkpoint; no fallback |
| `SUPER_WS` | Built workspace containing `devel/setup.bash` |
| `PLANNER_MAP` | Goal-sampling NPZ; its generated JSON sidecar must accompany it |
| `PLANNER_SIM_CONFIG` | Optional PerfectDrone/MARSIM YAML |
| `PLANNER_DEVICE` | Inference device; defaults to `cuda` |
| `PLANNER_COMMAND_RATE` | Command rate; defaults to `100 Hz` |

The GUI launcher needs `gnome-terminal` and initialized Conda. Its
`nmp_infer.launch` starts PerfectDrone without the SUPER planner FSM.
Map installation and matching are part of the handoff below.

## Input construction

Each registered cloud is paired with the nearest real odometry row within
`5 ms`, matching `nearest_odom_v2`. That odometry timestamp is the physical
origin of state and trajectory. Sensor/state values are not interpolated.

The node transforms world points with the full quaternion, renders one
normalized `[1,30,180]` scan and constructs physical `[v0,a0,goal]`. At startup
it creates one persistent spawned, ROS-free inference worker. The child strictly
loads the checkpoint, owns the selected PyTorch device, warms the current GP-SR
path and returns validated CPU prediction arrays. The ROS parent retains the
renderer, clearance checks, `PlanExecutor`, visualization and the sole optional
`/planning/pos_cmd` publisher.

Each request uses only the current scan and physical state. The parent waits no
longer than the time remaining before the odometry timestamp plus the existing
`0.10 s` planning deadline. A timeout, child exit, protocol mismatch or invalid
payload rejects that plan and uses the existing prefix refresh/brake/hold path;
inference is never retried inside the ROS process. The decoded body polynomial
is anchored rigidly in world coordinates. Future state and malformed or
non-finite inputs are also rejected at this entry.

In execute mode, `a0` uses the last published command acceleration. Shadow
mode estimates it from successive odometry velocities, initially zero. Check
this input difference during simulator preparation; command rate, planning rate
and sensor rate are separate measurements.

## Execution

`PlanExecutor` admits an unchanged network trajectory after checking its next
`0.30 s` prefix, visible clearance of at least `0.40 m`, sampled `6/6/12`
dynamics and a continuous braking plan from the prefix boundary.
`CurrentScanClearance` combines observed-return distance and radial occlusion;
it does not query the global evaluation map.

| Mode | Command source |
| --- | --- |
| `network` | Accepted two-second GP-SR trajectory |
| `terminal` | Analytic goal capture, attempted within `6 m` horizontal goal distance |
| `brake` | Stored stop after prefix expiry, or an emergency stop after failed refresh |
| `terminal_hold` | Exact completed terminal endpoint, retained for the unchanged goal |
| `hold` | Finite ordinary hold after a network/brake stop or without an active plan |

Terminal capture also checks sampled dynamics and visible clearance. A completed
terminal trajectory keeps its goal latch and trajectory ID; only a genuinely
changed goal clears the latch and permits new planning. New scans can refresh a
prefix and its stored brake. Invalid predictions, durations, latency or prefix
checks reject a new plan; fatal command-construction errors stop publication.
`RuntimeCommand` carries world position, velocity, acceleration, jerk, yaw, yaw
rate, trajectory ID and mode. Yaw follows horizontal velocity with a
`2.5 rad/s` rate limit.

Two paths remain unresolved: `refresh_clearance` does not itself reject an
old scan, and failed refresh can select `uncertified_best_effort_brake` with
`verified=False`. Brake mode therefore does not guarantee clearance or dynamic
feasibility. These paths require separate safety fixes and simulator validation
before command-enabled use.

## ROS-free interfaces

| Responsibility | Interface |
| --- | --- |
| Prediction | `TrajectoryPrediction`, `decode_prediction()`, `predict()` |
| Process isolation | `InferenceWorker`, `PredictionPayload` |
| Synchronization | `OdometrySample`, `select_nearest_odometry()` |
| World trajectory / clearance | `WorldMincoTrajectory`, `CurrentScanClearance` |
| Analytic stops / prefix check | `plan_terminal_trajectory()`, `plan_braking_trajectory()`, `inspect_trajectory_prefix()` |
| Execution / yaw | `PlanExecutor`, `PlanDecision`, `RuntimeCommand`, `VelocityYawTracker` |

These interfaces are testable without importing ROS.

## Simulation handoff

Transfer the selected H10 checkpoint, its `params.json`, the fixed evaluation
manifest and matching Episodes for replay. Keep the checkpointed scales.
Data and weights are not included in Git.

Transfer a same-source map PCD/NPZ/JSON set. MARSIM reads the PCD identified by
`pcd_name` in `PLANNER_SIM_CONFIG` from its configured map directory.
`PLANNER_MAP` supplies only the goal-sampling NPZ; changing it does not change
the simulated world. Verify identical geometry before running; see
[map tools](../scripts/data_collect/README.md#map-tools).

The destination needs ROS Noetic, the built SUPER workspace and the supported
planner environment. Validate in this order:

1. Verify `neural_minco.__file__` belongs to the destination checkout. Replay
   identical windows and compare raw actions and physical trajectories with the
   source machine, retaining checkpoint scales and float64 MINCO.
2. Run the direct node without `--execute`. Check scan rendering, full-quaternion
   frames, synchronization, acceleration semantics and end-to-end P50/P95/P99
   latency. Network-only timing does not establish command or planning rate.
3. Enable simulator execution on fixed starts/goals and map seeds. Report
   arrival, collision, timeout/stall, execution jerk, plan-to-plan continuity,
   rejected plans and time spent in each execution mode.

Network-forward latency and command cadence are separate measurements. For a
10-second integration probe, report command count, median/P95/maximum period,
counts above 15 ms and 25 ms, trajectory-ID changes, LiDAR rate, odometry rate
and cloud-callback duration. The pre-isolation reference was 48 command periods
above 15 ms with a 31.6 ms maximum. Approximately 1000 commands, a 10 ms median,
no period above 25 ms and a material reduction in periods above 15 ms form the
simulator acceptance target; they are not deterministic unit-test guarantees.

Report raw network quality separately from executed behavior, braking and
analytic terminal capture. Frequent braking or permanent hold is not successful
planning. Cross-machine agreement and closed-loop stability are acceptance
checks, not results established by offline cleanup or unit tests.
Current-scan coverage and ideal simulation do not cover occlusion, tracking,
estimator, actuator, disturbance or communication failures.

## Expert data collection

Collection runs SUPER as the command owner and records one complete goal flight
per Episode. Use the separate [collection guide](../scripts/data_collect/README.md)
for its launcher, command envelope, sensor configuration, validation and
explicit copy-based cleaning. Window construction and supervision are defined
in [training](training.md).
