# Stage 2: Refiner-only physical fine-tuning

Stage 2 is a controlled study on top of a fixed Stage-1 checkpoint. The Stage-2 results reported in the
[README](../README.md#results) come from the dynamics + collision arm, with both weights set to 0.1.

## Question and controlled comparison

Does absolute physical supervision improve the frozen mixed-data H10 model without losing imitation
quality? Compare three complete ten-epoch continuations from the same Stage-1 best checkpoint:

| Arm | Original Stage-1 loss | Absolute dynamics | Map collision |
| --- | --- | --- | --- |
| S1 continuation | unchanged | off | off |
| Dynamics | unchanged | on | off |
| Dynamics + collision | unchanged | same weight | on |

All arms freeze the encoder, state encoder, query encoder, and global proposal. Only `shared_refiner`
is trainable. The single query, three global blocks, three shared refinement calls, Hermite lookups,
window sizes, detached previous action, learned time ratios, terminal acceleration, fixed two-second
M3 minimum-jerk decoder, and runtime code are unchanged.

## Objectives

Keep all original losses on the same 20 P/V targets at z0..z3, including terminal acceleration, ratio
floor, correction norm, z0 ratio prior, and adjacent-stage jerk non-regression. The z0 terms are
constant under the frozen proposal; they are not removed or reweighted.

Absolute dynamics is the sum of normalized peak-speed, peak-acceleration, and peak-jerk hinge
violations of 6 m/s, 6 m/s², and 12 m/s³. Use 1,001 samples per MINCO segment, including both ends,
matching the evaluator. This is a sampled peak objective, not a continuous feasibility certificate.

Collision is the trajectory-maximum normalized clearance deficit below 0.3 m, sampled at the same
128 future times as evaluation. Each training window queries its matching 0.2 m voxel ESDF after
the full-quaternion body-to-world transform. Both supplied forest maps have a physical z=0 ground:
the training field includes that solid half-space. Leaving the finite padded distance grid is also
penalized, not treated as increasing free-space distance. These are training map semantics, not
network features or runtime modifications. The existing evaluation ESDF metric is not redefined;
world-height and map-boundary diagnostics must be reported separately.

Both physical terms use the actual decoded MINCO curve, never the Hermite lookup proxy, and are
averaged equally across z1/z2/z3. Imitation and physics share one packed MINCO solve. Loss weights
are fixed before training using training-only loss/gradient probes, recorded in the run-local protocol.
No validation/test-driven weight search or offline parameter teacher is added.

Training-only differentiable map-cost feedback has precedent in
[YOPO](https://wayhome316.github.io/projects/onlyplanonce/youonlyplanonce.pdf); its motion-primitive
candidates and objective are not copied. The event-maximum reduction follows this project's earlier
controlled safety studies; their numerical weights do not transfer automatically to H10.

## Schedule and assessment

Use the fixed mixed-data Episode split, batch 32, seed 42, ten full epochs, fresh AdamW and cosine
3e-5 → 1e-5, weight decay 1e-4, and gradient clipping at 1. Preserve all fitted physical scales from
the checkpoint; never refit on this continuation. Use two workers and prefetch two per loader to
bound shared memory with three concurrent jobs. TensorBoard and per-epoch CSV remain enabled.

Primary comparison is the final epoch of every arm, not the checkpoint favored by a changed loss.
Also save ADE-best as a secondary artifact. Evaluate old and new validation maps separately, then
Episode-balance the combination. Keep the test split sealed. Report visible collision first, global
clearance as supplementary, jerk and v/a violations and tails, ADE/FDE, per-stage changes, endpoint
progress, below-ground and finite-map exits, and inference timing. Compare the warm-start checkpoint
with the same diagnostics. Verify all frozen state tensors and z0 remain unchanged.

Do not promote a checkpoint based on a tiny collision-rate change that worsens jerk, imitation, or
progress. No online flight or command publication is part of this experiment; offline gains cannot
establish flight safety. Launch only after CPU tests, real CPU/CUDA forward/backward, map-gradient,
frozen-z0, loss-equivalence, timing, and memory gates pass.
