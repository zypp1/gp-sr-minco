# Training and Evaluation

This document specifies the Stage-1 imitation protocol and the offline
evaluation protocol; Refiner-only physical fine-tuning is described in
[Stage 2](stage2.md). Install the supported stack and run the basic commands
from the [README](../README.md#installation).
`src/neural_minco/config.py` is the configuration source.

## Episode windows

One `episode-*.npz` stores a complete flight with LiDAR, world pose/velocity,
command acceleration, full attitude, goal and map metadata. Collection and
explicit validation/cleaning are described in the
[collection guide](../scripts/data_collect/README.md); training never rewrites
the raw files.

`EpisodeWindowDataset` uses these fixed settings:

| Field | Value |
| --- | --- |
| Time origin | `nearest_odom_v2` |
| Input | One current `[1,30,180]` scan; no history or previous plan |
| Horizon | `2.0 s`, complete horizon required |
| Target grid | 20 future times `0.1,0.2,...,2.0 s`; excludes `t=0` |
| LiDAR anchor stride | `5` |

The nearest real odometry row supplies both state and the future grid's time
origin. State and LiDAR are not interpolated. Future position and velocity
labels are linearly interpolated on odometry timestamps; acceleration labels
use the recorded command-acceleration timestamps. All targets use the anchor's
full-quaternion body frame.

## Flight-level manifest

Split complete flights before constructing windows. The external JSON manifest
must contain nonempty, disjoint `train`, `val` and `test` lists:

```json
{
  "split": {
    "train": ["episode-000001-260101-120000.npz"],
    "val": ["episode-000002-260101-120100.npz"],
    "test": ["episode-000003-260101-120200.npz"]
  }
}
```

Use actual basenames in the collection format
`episode-<sequence>-<YYMMDD>-<HHMMSS>[-tag].npz`, resolved relative to `--data`.
Requested files must exist, be successful Episodes and use compatible LiDAR
geometry. Flights without a complete two-second window are skipped; each loaded
split must retain at least one window.

Freeze the manifest and Episode contents before a controlled comparison.
Validation and test remain unaugmented. Training and evaluation neither
repartition nor clean the selected files implicitly.

## Stage-1 objective

Decode `z0,z1,z2,z3` outside the network and supervise each on the same 20
physical future targets. Let `k=0..3` index stages and `r=softmax(logits)`:

```text
L = mean_k(Lpos_k + 0.25 Lvel_k + 0.1 Laend_k + 0.01 Lratio_k)
    + 0.01 Ldelta + 0.1 Lz0_ratio + 0.03 Ljerk
```

| Term | Definition with current defaults |
| --- | --- |
| `Lpos` | Masked mean of Smooth L1 applied to position-error norm, beta `0.3` |
| `Lvel` | Masked mean of velocity-error norm divided by `6 m/s` |
| `Laend` | Componentwise mean Smooth L1 of normalized terminal acceleration, scale `6 m/s²` and beta `0.02` |
| `Lratio` | Mean `relu(0.05 - r)^2` over batch and segments |
| `Ldelta` | Mean squared correction norm over the three Refiner calls |
| `Lz0_ratio` | Mean squared z0 logits after subtracting their per-row mean |

Position/velocity terms reduce over valid batch/time points with equal time
weights. Four stage imitation losses have equal weight. The stored 20-point
acceleration labels are not an extra imitation term; terminal acceleration has
its separate endpoint term.

Adjacent-stage jerk non-regression uses 21 uniform samples including `t=0`:

```text
e_k = relu(max_t ||jerk_k(t)|| / 12 - 1)
Ljerk = mean_(batch,k=1..3) relu(e_k - detach(e_(k-1)))
```

The preceding excess is detached. This penalizes worsening excess and can be
zero for an unsafe trajectory. Stage 1 has no absolute collision/dynamics cost,
progress or time-optimality objective, or offline parameter teacher; absolute
physical terms enter only in [Stage 2](stage2.md).

## Optimizer and output scales

| Setting | Default |
| --- | --- |
| Seed / epochs | `42` / `10` |
| Batch size / workers | `32` / `8` |
| Optimizer | AdamW, weight decay `1e-4` |
| Learning rate | Per-step cosine, `3e-4` to `1e-5` |
| Global gradient clip | `1.0` |
| Dropout | `0`, required for current training/resume |
| Monitoring interval | Every `20` optimizer steps and the last batch of each epoch |

The sampler draws a flight uniformly, then one of its windows uniformly.
Validation is deterministic and unaugmented. Episode-balanced validation ADE
selects the best checkpoint; it does not stop the default ten-epoch run early.

Before constructing a fresh model, fit output scales from selected raw training
window labels only:

```text
waypoint_scale = max(1.0, 1.05 * max ||dense_traj||)
terminal_velocity_scale = max(1.0, 1.05 * max ||v_end||)
```

Validation/test labels are excluded. Scales, normalization and the resolved
configuration are saved in `params.json` and checkpoints. An explicit
`Stage1Trainer(..., config=config)` receives an independent configuration copy.

## Runs and exact resume

Smoke-test one real window, then start a fresh run:

```bash
python scripts/train.py --data <episodes> --manifest <split.json> --device cpu --dry-run
python scripts/train.py --data <episodes> --manifest <split.json> --output saved/gp-sr-run --device cuda
```

`--dry-run` fits scales on the first training Episode and executes one window's
forward/loss/backward without optimizer steps or run files. Repeat it on CUDA
before a GPU run. A fresh output directory must not already exist.

Each run writes `params.json`, `split_manifest.json`, `history.csv`,
`tensorboard/`, `ckpt_best.pth` and `ckpt_last.pth`. Only `ckpt_last.pth` includes optimizer
and scheduler state for epoch-end recovery:

```bash
python scripts/train.py \
  --data /absolute/path/to/episodes \
  --manifest /absolute/path/to/split_manifest.json \
  --resume saved/gp-sr-run/ckpt_last.pth \
  --device cuda
```

Resume in the original run directory with unchanged Episodes, manifest,
architecture, losses, seed, epochs, batching and LR schedule. The trainer checks
the manifest and `params.json` against the checkpoint, along with epoch,
optimizer-step and scheduler positions. It restores checkpointed scales without
fitting them again and rejects configuration overrides. A changed protocol
requires a new run.

TensorBoard records the existing running-mean training loss components,
gradient norm and learning rate, plus epoch-end validation loss and ADE/FDE.
The `tqdm` bars show training/validation progress; training also shows running
loss and learning rate. `training.log_interval` controls how often those
scalars are copied to CPU together, avoiding per-batch logging synchronization.
Epoch-end flush and writer closure preserve completed logs. Resume hides events
beyond the checkpoint's completed step without resetting the optimizer schedule.
No TensorBoard files are created by `--dry-run`.

## Evaluation

The [README command](../README.md#evaluation) evaluates an
explicit checkpoint on `val`. Select `--split test` for the test split.
Reports include every stage's ADE/FDE, visible clearance/collision, maximum
speed/acceleration/jerk, segment durations and correction norms.

| Measurement | Sampling / threshold |
| --- | --- |
| ADE/FDE | Same 20 future target times as imitation |
| Collision | 128 uniform future samples, excluding `t=0` |
| Dynamics | 1,001 samples per segment, including boundaries |
| Visible collision | Nearest observed-return clearance `< 0.3 m` |
| Velocity / acceleration / jerk failure | Sampled maximum `> 6 / 6 / 12` in SI units |

Visible-return collision is the primary development metric. Episode-balanced
statistics average window values within each flight before averaging flights;
window-weighted statistics give every window equal weight.

Add a matching world-frame map for global collision and occupied penetration:

```bash
python scripts/evaluate.py \
  --checkpoint saved/gp-sr-run/ckpt_best.pth \
  --data /absolute/path/to/episodes \
  --manifest /absolute/path/to/split_manifest.json --split test \
  --map /absolute/path/to/example_scene.npz \
  --output saved/gp-sr-run/evaluation_test.json
```

The map NPZ contains only finite `points: [N,3]`. Its filename stem must match
the dataset's single `map_id`. ESDF uses `0.2 m` voxels and `2 m` padding.
Global collision is ESDF `< 0.3 m`; occupied penetration is ESDF `< 0 m`.

Without `--output`, the CLI writes `evaluation_val.json` or
`evaluation_test.json` next to the checkpoint. Keep reports local. Offline
metrics and the [online simulator checks](runtime.md#simulation-handoff)
measure different behavior.
