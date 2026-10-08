# Action-chunk segmentation results (RoboTwin 2.0)

Results of `scripts.benchmarks.segment_demos` (method in
`adaspevla/datasets/segmentation.py`, usage in the README section
"Segment demonstrations into adaptive action chunks"). Generated 2026-10-08.

## Data

- RoboTwin 2.0 demonstrations in XPolicyLab v1.0 HDF5 format, collected
  locally with the official RoboTwin collector (`demo_clean`, `aloha_agilex`),
  from `/data/lzezeng/data/codes/DynamicDAPolicy/RoboTwin/data/demo_clean/<task>/aloha_agilex/data`.
- 7 tasks with 50 episodes each (350 episodes), both arms segmented jointly.
- One sample is one saved frame, 0.06 s apart (save stride 15 x 1/250 s).
- Raw files were only read. Per-episode boundaries, summaries and plots are in
  `data/segments/robotwin/<task>/`.

## Settings

Defaults: cubic spline with 8 control points per chunk, keyframes on,
`min_samples=8`, `max_samples=64`, `tolerance=0.1`, `mass_budget_s=3.0`,
`gripper_gain=10`, `speed_gain=0`, `accel_gain=0`, `dilation_s=0.15`.
Motion scales D are calibrated per task from 16-sample windows.

## Chunk statistics (min_samples = 8, default)

| Task | Episodes | Mean episode length (samples) | Chunks per episode | Mean chunk length (samples) | Median | Range | Mean chunk duration (s) | Chunks over tolerance |
|---|---|---|---|---|---|---|---|---|
| adjust_bottle | 50 | 143.1 | 7.0 | 21.2 | 18 | 8–45 | 1.21 | 27 |
| beat_block_hammer | 50 | 113.6 | 7.0 | 17.0 | 17 | 8–33 | 0.96 | 0 |
| click_alarmclock | 50 | 85.0 | 5.2 | 17.3 | 15 | 8–39 | 0.98 | 1 |
| click_bell | 50 | 77.1 | 5.0 | 16.2 | 13 | 8–38 | 0.91 | 0 |
| dump_bin_bigbin | 50 | 242.4 | 15.0 | 17.1 | 12 | 8–46 | 0.96 | 0 |
| grab_roller | 50 | 93.8 | 5.0 | 19.6 | 16 | 8–48 | 1.11 | 20 |
| lift_pot | 50 | 111.1 | 7.6 | 15.4 | 10 | 8–43 | 0.87 | 0 |
| **All** | 350 | 123.7 | 7.4 | 17.6 | 15 | 8–48 | 0.99 | 48 |

- An episode averages 123.7 samples (7.4 s) and is split into 7.4 chunks
  (5 to 22).
- Across 2592 chunks: 10th / 25th / 75th / 90th percentile lengths are
  8 / 10 / 21 / 35 samples. A chunk of n samples has n − 1 action steps.
- Short chunks sit around grasps and releases; long ones cover free-space
  transport.
- "Over tolerance": 48 chunks (1.9%) miss the tightened tolerance near grasps,
  all at or near the 8-sample minimum. Their median residual is 0.011 in
  normalized units, at most 1.35 times the local tolerance. They are kept and
  flagged with `within_tolerance=false`.

### Comparison with min_samples = 16

With `--min_samples 16` (same data and other settings), chunks average 23.7
samples (median 20, range 16–51, 5.4 chunks per episode), but 439 of 1893
chunks (23%) miss the grasp tolerance (median residual 0.047). The longer
minimum trades local accuracy at grasps for longer, more uniform chunks.

## Spline accuracy versus chunk length

One cubic spline with 8 control points fitted to fixed-length windows
(stride 4) over 70 episodes (10 per task, all 7 tasks). Position error is the
largest end-effector position error inside a window; normalized error is the
largest absolute residual over all channels, including gripper width.

| Samples | Duration (s) | Position error median (mm) | Position error p95 (mm) | Normalized error median | Normalized error p95 | p95 with per-arm knots |
|---|---|---|---|---|---|---|
| 8 | 0.42 | 0.01 | 0.03 | 0.0002 | 0.0018 | 0.0018 |
| 12 | 0.66 | 0.04 | 0.32 | 0.0013 | 0.0240 | 0.0240 |
| 16 | 0.90 | 0.16 | 0.78 | 0.0048 | 0.0374 | 0.0376 |
| 24 | 1.38 | 0.72 | 2.08 | 0.0183 | 0.0739 | 0.0743 |
| 32 | 1.86 | 1.76 | 4.38 | 0.0444 | 0.1007 | 0.1010 |
| 48 | 2.82 | 6.18 | 12.39 | 0.1098 | 0.1631 | 0.1626 |

- 12 to 16 samples are reproduced to well under 1 mm; error grows from about
  24 samples on.
- Dual arm: fitting each arm on its own teacher knots gives the same error as
  one shared knot vector, so both arms are fitted jointly (shared boundaries
  and knots), which also keeps them synchronized.

## Observation on the paper's knot density

With the paper's implicit e_ref = 1 s^-2, the motion term e_j on RoboTwin has
a median of about 10 s^-2 and is largest during fast transport, so tanh(e_j)
is saturated almost everywhere and the teacher knot spans are close to
uniform. The segmentation therefore uses the gripper term (and optional
criticality labels) for precision and keeps e_j only for knot placement
inside a chunk.
