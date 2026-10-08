# Architecture

The selected single-frame Global Proposal + Shared Refiner (GP-SR) planner uses
ten Hermite lookup positions (H10). Training supervises all four outputs on the
same 20 future points at `0.1..2.0 s`.

## Computation graph

```text
range [B,1,30,180]                 physical body state [B,9]
        |                                    |
        +--------- yaw canonicalization -----+
                             |
                    dual-scale encoder
                   /                  \
       coarse [B,1350,256]      fine [B,5400,256]
                 |                       |
constant-velocity reference             |
+ one state/action query                |
                 |                       |
       Global Decoder x3 -> z0          |
                             |           |
                   Hermite10 lookup -----+
                             |
                 shared Refiner (9x17) -> z1
                   fresh lookup
                 same Refiner   (5x9)  -> z2
                   fresh lookup
                 same Refiner   (3x5)  -> z3
                                          |
                               final MINCO trajectory
```

Every Refiner receives the preceding detached action and rebuilds its query
and local support. All three calls share the same parameters, including their
joint output head; a learned stage embedding distinguishes the calls.
The three global decoder layers are independent.

The network performs no MINCO solve. Training decodes all four outputs
externally for supervision; inference decodes only the final output.

## Dimensions and connections

| Component | Selected dimensions |
| --- | --- |
| Encoder input | hit-gated closeness, return-point coordinates, hit: 5 channels |
| Fine trunk | `5 -> 64` stem; two residual blocks at `30x180` |
| Coarse trunk | `2x2` average pool; `64 -> 128`; three residual blocks at `15x90` |
| Global content | `128 -> 256`, flattened to 1,350 tokens |
| Local content | fine `64 -> 64` + upsampled coarse `128 -> 64`; convolution then `64 -> 256` |
| State MLP | `9 -> 128 -> 128`, evaluated once |
| Action MLP | `18 -> 128 -> 128`, evaluated for each current action |
| Query | concatenate action/state features, LayerNorm: `[B,1,256]` |
| Global decoder | three layers, each 8-head cross-attention and FFN `256 -> 1024 -> 256` |
| Lookup descriptor | normalized position, unit tangent, progress: `7 -> 64 -> 256` plus position role |
| Local attention | 8 heads, dimension 32 per head, joint softmax over position/window pairs |
| Refiner FFN | `256 -> 1024 -> 256` |
| Joint correction head | `256 -> 256 -> 18`; separate proposal/refiner heads |

MLPs use SiLU. State/action encoders have LayerNorm, and the concatenated query
is normalized again. Proposal attention normalizes Q, K and V separately.
The Refiner normalizes its stage-conditioned query. FFNs are pre-normalized
residual blocks; encoder residual blocks use GroupNorm. Selected dropout is zero.

## Inputs and coordinates

The range image is normalized to `[0,1]`; `1` means no measured return inside
`r_max=20 m`. Its 180 azimuth bins span `360°` at `2°` per bin. Elevation starts
with 90 full-sphere bins and retains centers in `[-7°,52°]`, producing 30 rows.
Azimuth is periodic; elevation is not. Geometry is configured in
`default_config()["lidar"]`.

State is `[v0,a0,goal]` in the current true body frame. The model divides these
vectors by `6 m/s`, `6 m/s²`, and `30 m` for learned features, without
clipping. MINCO receives the original physical state.

Goal direction defines a continuous yaw-aligned frame, with velocity,
acceleration, then body +x as fallbacks. The LiDAR image uses an integer-column
roll; state and predicted vectors use the continuous rotation and its inverse.
This internal canonicalization is not a substitute for body/world conversion,
which always uses the full quaternion.

## LiDAR memory

The shared backbone uses circular azimuth padding and non-periodic vertical
padding. Fine content is fused with nearest-upsampled coarse content.

Each token is the sum of CNN content, a learned ray-direction embedding
`3 -> 256`, and an observation embedding `5 -> 64 -> 256` over hit-gated
closeness, normalized return coordinates and hit state. Directions remain
available even without a return; an all-free scan need not produce zero tokens.

Each coarse cell represents a `2x2` native cell group. Its physical ray uses
the nearest observed return, or the normalized mean direction if all four
rays are free. The fine memory preserves the native 5,400-ray geometry.
Ray directions and angles are persistent checkpoint buffers.

## Initial action and Global Proposal

The constant-velocity reference uses current velocity:

```text
Q1 = v0 * (2/3)        Q2 = v0 * (4/3)        p_end = v0 * 2
v_end = v0            a_end = 0              ratios = [1/3,1/3,1/3]
```

Positions and terminal velocity are normalized and bounded before forming the
initial action. Current acceleration is still part of state context and the
MINCO start boundary; it is not integrated into this reference.

The action MLP receives all 15 normalized geometric/boundary values and the
three softmax ratios, not unbounded ratio logits. Together with state context,
it produces the single query. There is no query self-attention.

Global K/V use all 1,350 coarse tokens. A relative encoding describes each
observed return against its nearest point on the 20-point constant-velocity
reference: displacement, distance, signed body clearance and progress.
A shared `6 -> 64` hidden projection has independent `64 -> 256` K/V
outputs, initially zero. All three global decoder layers read this memory.
Only after the third layer does the proposal head emit the correction for z0.

## Hermite lookup support

The proxy has knots `[0,Q1,Q2,p_end]` and the predicted segment durations.
Its boundary velocities are current v0 and predicted v_end. Internal velocities
are neighboring-knot displacement divided by the combined neighboring durations.

A cubic Hermite segment gives position and tangent without a linear solve.
Evaluate it at ten fixed physical times `0.2,0.4,...,2.0 s`. Each descriptor
contains position/waypoint_scale, unit tangent, and time/2.

This lookup proxy does not enforce MINCO acceleration boundaries and is never
executed. Its ten positions do not change the 20-point supervision grid.

## Local attention and refinement

Each lookup position is projected into the native range-image grid. Its complete
window is gathered; azimuth wraps, and invalid elevation rows are masked.

| Call | Window | Maximum position/window pairs |
| --- | --- | ---: |
| z0 -> z1 | 9x17 | 1,530 |
| z1 -> z2 | 5x9 | 450 |
| z2 -> z3 | 3x5 | 150 |

Overlapping windows may read a ray multiple times with different relative
descriptors. There is no TopK pruning or per-position softmax.

Position descriptors and a learned `[10,256]` role embedding are added to
local tokens before K/V projection. Additional local relations contain:

- hit-relative displacement, radial range margin and window offsets;
- hit status and signed clearance relative to the `0.3 m` body radius;
- obstacle direction and its alignment with the proxy tangent.

The eight displacement/window features use `8 -> 64`; the four direction
features use `4 -> 64`. Their sum passes through SiLU and `64 -> 256`.
The resulting message contributes both to attention scores (`256 -> 8`) and
to attended values. Direction weights start at zero but remain learned.

One softmax per head spans all valid position/window pairs. The attended update,
query residual and FFN feed the joint correction head. The next call rebuilds
everything from the updated detached action; no content vector is carried.

## Action updates and MINCO

An action has 18 values:

```text
Q1 (3) | Q2 (3) | p_end (3) | v_end (3) | a_end (3) | ratio logits (3)
```

Geometry and terminal velocity use componentwise bounded residual updates
`tanh(atanh(reference) + correction)`. Terminal acceleration uses the radial
vector equivalent. Ratio updates add a bounded correction with step 0.5 and
remove the common logit offset. The proposal's final linear layer starts with
standard deviation 0.001; the Refiner's starts at zero.

Checkpointed scales decode the physical positions and terminal velocity;
terminal acceleration uses the configured acceleration scale. Durations are
`T = 2 * softmax(logits)`, not fixed equal times.

`MinimumJerkDecoder` solves an M=3 piecewise quintic trajectory with true start
velocity/acceleration, interior Q1/Q2 and terminal position/velocity/acceleration.
Position through snap are continuous at internal knots. The solve uses float64
internally, including CUDA, where cuSOLVER is retained.

## Limits

The joint query couples geometry, boundary derivatives and time allocation.
Global context and fresh local lookups support three corrections with shared
weights, but Hermite support can differ from the executed MINCO curve. A single
scan and expert future also limit what is observable. Imitation and the light
regularizers do not establish collision avoidance or dynamic feasibility.

The network has no scan history, previous plan, map input, candidate bank or
online optimizer. Evaluation-map truth and runtime fallbacks are assessed
separately from the raw network output.

## Public interfaces

Construct/load through `neural_minco.model.build_model` and `load_model`.
Physical prediction is `neural_minco.runtime.predict`; direct model calls
return normalized actions and optionally all four stages.

Strict loading retains the selected H10 state-dict keys, geometry buffers and
physical scales. Joint readout/attention and rebuilt queries are the supported
semantics, including H10 checkpoints that predate those optional metadata keys.
The `hermite10` support scheme must be explicit; missing support metadata or
explicit retired modes are rejected. See [training](training.md) for output-scale
fitting and exact resume.
