# AdaSpeVLA: Adaptive Knot Placement and Supervised Query Timing

AdaSpeVLA predicts one continuous spline per call from camera images and a
language instruction. Per arm it holds six base-frame offsets from the
end-effector pose observed at query time and the gripper opening width, all
generated jointly by one flow on shared learned knots. It alternates
policy calls with spline-prefix execution. Longer query intervals execute more
actions between predictions; shorter intervals refresh sooner.

This README covers installation, a smoke test, and training on your own
RoboTwin HDF5 data. Benchmark training and evaluation (RoboTwin 2.0, RoboDojo,
RoboCasa365, and the XPolicyLab adapter) are in
[docs/benchmarks.md](docs/benchmarks.md).

| I want to... | Go to |
|---|---|
| Train and evaluate on the RoboTwin / RoboCasa suite on Jean Zay (Slurm) | [docs/slurm_benchmarks.md](docs/slurm_benchmarks.md) |
| Install AdaSpeVLA and check it works | [Step 1](#step-1-install) and [Step 2](#step-2-check-the-installation) below |
| Train on a native RoboTwin HDF5 export (right arm) | [Step 3](#step-3-train-on-native-robotwin-hdf5-right-arm) and [docs/robotwin_native.md](docs/robotwin_native.md) |
| Train on HDF5 data I already converted | [Step 4](#step-4-train-on-already-converted-hdf5-data) |
| Train and evaluate on RoboTwin 2.0 through XPolicyLab | [docs/benchmarks.md, Part A](docs/benchmarks.md#part-a-robotwin-20-through-the-xpolicylab-adapter) |
| Train and evaluate on RoboDojo | [docs/benchmarks.md, Part B](docs/benchmarks.md#part-b-robodojo-through-the-same-adapter) |
| Train and evaluate on RoboCasa365 | [docs/benchmarks.md, Part C](docs/benchmarks.md#part-c-robocasa365) |
| Run the shared vs per-arm knot ablation | [docs/benchmarks.md, Part D](docs/benchmarks.md#part-d-bimanual-knot-ablation-shared-vs-per-arm-knots) |
| Split raw RoboTwin / RoboCasa demos into difficulty-adaptive spline segments | [Segment demonstrations into adaptive action chunks](#segment-demonstrations-into-adaptive-action-chunks) and [result.md](result.md) |

All commands run from the repository root unless a step says otherwise.

On Jean Zay, training and evaluation run as self-contained `.slurm` files
(`run.slurm`, `pilot_train*.slurm`, `eval.slurm`, `eval_robotwin.slurm`, ...),
submitted with a plain `sbatch <file>.slurm`. The full workflow, the hardware
and memory settings, and how to read the results are in
[docs/slurm_benchmarks.md](docs/slurm_benchmarks.md).

The short version, from the repository root:

```bash
# 1. Train: copy a training file, set its #SBATCH --array (task rows), RUN_ROOT and flags
cp pilot_train_noaug.slurm my_train.slurm && sbatch my_train.slurm
# 2. Evaluate in simulation: copy eval.slurm (RoboCasa) or eval_robotwin.slurm (RoboTwin),
#    point RUN_ROOT/METRICS_ROOT at the run, add #SBATCH --dependency=afterok:<train job>_<row>
sbatch my_eval_robotwin.slurm
# 3. Summarize success rate, queries and timing per mode
python -m scripts.benchmarks.summarize_metrics eval_metrics/<run>-robotwin/robotwin/<task>/demo_clean
```

---

## Step 1: Install

Requirements: Linux, Python 3.10 or newer, and a CUDA GPU for real training
(CPU is enough for the smoke test).

```bash
# 1. Create and activate an environment
conda create -n adaspevla python=3.10 -y
conda activate adaspevla

# 2. Install PyTorch for your CUDA version first (example: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install AdaSpeVLA (editable) with the test tools
cd /path/to/AdaSpeVLA
pip install -e ".[dev]"
```

The production backbone is `google/siglip-base-patch16-224`. It downloads from
Hugging Face on first use. For a machine without internet access, download it
once on a connected machine that shares the same `HF_HOME`, then set
`HF_HUB_OFFLINE=1` on the offline one:

```bash
python -c "from transformers import AutoModel, AutoProcessor; AutoModel.from_pretrained('google/siglip-base-patch16-224'); AutoProcessor.from_pretrained('google/siglip-base-patch16-224')"
export HF_HUB_OFFLINE=1
```

## Step 2: Check the installation

```bash
# 1. Unit and integration tests (CPU). On Jean Zay, run with the training
#    interpreter and without writing caches (WORK has a file quota):
PYTHONDONTWRITEBYTECODE=1 tmp/train-env/bin/python -m pytest tests -q -p no:cacheprovider

# 2. End-to-end smoke run: synthetic data, tiny offline backbone, CPU
python -m scripts.train_robotwin --synthetic_data --test_backbone --no-proprio \
    --epochs 1 --timing_epochs 1 --num_ode_steps 2 --batch_size 2 \
    --device cpu --output_dir /tmp/adaspevla-smoke
python -m scripts.eval_robotwin \
    --ckpt_path /tmp/adaspevla-smoke/adaspevla_robotwin_beat_block_hammer_latest.pt \
    --num_episodes 1 --duration 0.2 --device cpu
```

The smoke run uses `--no-proprio` because the smoke evaluator
`scripts.eval_robotwin` doesn't supply robot state yet. With robot state on, they stop with "This policy
conditions on robot state; pass proprio". The simulator evaluators do supply it.

Test status (Oct 15, 2026): the suite is not green. Two files,
`tests/benchmarks/test_xpolicylab_benchmarks.py` and
`tests/benchmarks/test_per_arm_knots.py`, fail to import
(`integrate_twist` no longer exists in `adaspevla/benchmarks/pose_math.py`),
and the last full run recorded 26 failures that predate the observation changes
(see `analysis_low_success.md`). The full suite takes more than 25 minutes on a
login node. Run single files with `pytest tests/<file>.py` while you work.

`scripts.eval_robotwin` and `scripts.eval_robocasa` are smoke evaluators. They
run the spline executor without a simulator, so they check that a checkpoint
loads and produces actions. They do not measure task success. For simulator
evaluation, use [docs/benchmarks.md](docs/benchmarks.md).

## Step 3: Train on native RoboTwin HDF5 (right arm)

Use this for the official RoboTwin state/action-paired HDF5 export. The native
adapter reads the observed right-arm poses and the continuous commanded gripper
width while loading; each training window re-anchors the poses at its first
sample, so there is no separate conversion step.

```bash
# 1. Locate the episodes. The loader searches <data_dir>/<task_name>/ recursively,
#    or else the HDF5 files directly inside <data_dir>.
ls /path/to/robotwin/source_domain/beat_block_hammer | head

# 2. Optional: list the dataset paths of one episode
python -m scripts.inspect_hdf5 /path/to/robotwin/source_domain/beat_block_hammer/<episode>.hdf5

# 3. Train all stages (action, SigLIP finetune, query timing)
python -m scripts.train_robotwin \
    --data_dir /path/to/robotwin/source_domain \
    --task_name beat_block_hammer \
    --robotwin_native_right \
    --epochs 10 --timing_epochs 10 --batch_size 8 \
    --output_dir ./checkpoints/robotwin_right

# 4. Check that the checkpoint loads and runs
python -m scripts.eval_robotwin \
    --ckpt_path ./checkpoints/robotwin_right/adaspevla_robotwin_beat_block_hammer_latest.pt \
    --num_episodes 1
```

The output is `./checkpoints/robotwin_right/adaspevla_robotwin_<task>_latest.pt`,
the timing-stage checkpoint that evaluation needs, plus one file per stage
(`_action.pt`, `_finetune.pt`, `_timing.pt`).

Options for this mode:

| Option | Default | Meaning |
|---|---|---|
| `--robotwin_base_wxyz qw qx qy qz` | `0.707 0 0 0.707` (Aloha) | Robot base orientation in world, for another embodiment |
| `--robotwin_simulation_dt` | `1/250` | Collector timestep; sample period = `frequency` × this |
| `--sample_dt` | from file | Override the sample period in seconds |
| `--camera_paths A B` | head + right wrist | Override the camera datasets, in view order |

Do not combine `--robotwin_native_right` with `--dual_arm`, `--actions_path` or
manual gripper coding. Conversion details and the timing caveat are in
[docs/robotwin_native.md](docs/robotwin_native.md).

## Step 4: Train on already converted HDF5 data

Use this when your HDF5 files already contain end-effector poses and gripper
widths (or base-frame twists, which are integrated into poses) that follow the
[action and dataset contract](#action-and-dataset-contract) below.

```bash
# RoboTwin layout (one arm; add --dual_arm for 14-channel two-arm data)
python -m scripts.train_robotwin \
    --data_dir /path/to/converted/data --task_name beat_block_hammer \
    --epochs 10 --timing_epochs 10 --output_dir ./checkpoints/robotwin

# RoboCasa layout
python -m scripts.train_robocasa \
    --data_dir /path/to/converted/data --task_name PnPCounterToCab \
    --epochs 10 --timing_epochs 10 --output_dir ./checkpoints/robocasa
```

If the HDF5 files carry no action metadata attributes, declare it on the command
line, but only if it matches the source:

```bash
python -m scripts.train_robotwin --data_dir /path/to/converted/data --task_name beat_block_hammer \
    --action_type ee_pose --action_frame base --sample_dt 0.02 \
    --gripper_open_value 1 --gripper_closed_value 0
```

If your camera datasets have other names, inspect one episode and pass them in
view order:

```bash
python -m scripts.inspect_hdf5 /path/to/episode.hdf5
python -m scripts.train_robotwin --data_dir /path/to/converted/data --task_name beat_block_hammer \
    --camera_paths observation/head_camera/rgb observation/wrist_camera/rgb
```

Data from RoboTwin 2.0, RoboDojo or RoboCasa365 should go through the benchmark
converters in [docs/benchmarks.md](docs/benchmarks.md) instead, which write this
contract for you.

## Segment demonstrations into adaptive action chunks

`scripts.benchmarks.segment_demos` splits raw demonstrations into
variable-length action chunks, each fitted by one cubic B-spline with a fixed
number of control points. Easy free-space motion gets long chunks and grasps
get short ones. Raw files are only read; everything is written to
`--output_dir`. The method is in
`adaspevla/datasets/segmentation.py`, and statistics on RoboTwin are in
[result.md](result.md).

How a demo is split:

1. Cut at keyframes: samples where every arm comes to rest, or where a gripper
   width starts or stops changing.
2. Give every interval a precision demand `rho` (mainly gripper activity, plus
   optional criticality labels) and a per-sample tolerance `tolerance / rho`.
3. Inside each keyframe interval, grow a chunk as long as its spline fit
   (anchor-relative offsets re-anchored at the chunk start, paper teacher
   knots) keeps every sample within tolerance, its density mass
   `sum(rho * dt)` stays below `--mass_budget_s`, and its length stays within
   `[--min_samples, --max_samples]`. Adjacent chunks share their boundary
   sample, which is the next anchor.

```bash
# RoboTwin 2.0 / XPolicyLab v1.0 HDF5 (both arms, segmented jointly)
python -m scripts.benchmarks.segment_demos --format robotwin \
    --source_dir /path/to/RoboTwin/data/demo_clean/beat_block_hammer/aloha_agilex/data \
    --output_dir ./data/segments/robotwin/beat_block_hammer --plot 3

# RoboCasa365 LeRobot v2.1 (videos are not decoded)
python -m scripts.benchmarks.segment_demos --format robocasa \
    --source_dir /path/to/lerobot --output_dir ./data/segments/robocasa/<task>
```

Output: one `<episode>.segments.json` per episode (`boundaries`, `keyframes`,
per-chunk `start`/`end`/`num_samples`/`max_residual`/`within_tolerance`/
`knot_spans`, and per-interval `density`), `segments_summary.json`, and with
`--plot N` a PNG for the first N episodes. Read a file back with
`adaspevla.datasets.segmentation.load_segments`.

| Option | Default | Meaning |
|---|---|---|
| `--num_control_points`, `--degree` | `8`, `3` | Spline per chunk |
| `--min_samples`, `--max_samples` | `8`, `64` | Chunk length limits in samples; keyframes closer than `min_samples - 1` intervals are dropped |
| `--tolerance` | `0.1` | Base fit tolerance in normalized units (offsets / D, gripper 2w-1); divided by `rho` |
| `--mass_budget_s` | `3.0` | Max `sum(rho * dt)` per chunk, so a region of demand `rho` gets chunks of at most `3/rho` s |
| `--keyframes` / `--no-keyframes` | on | Cut at stops and gripper-change edges first |
| `--stop_fraction` | `0.05` | "At rest" = speed below this fraction of the episode's 95th-percentile speed |
| `--gripper_eps` | `1e-3` | Width change per sample that counts as gripper motion |
| `--gripper_gain`, `--gripper_ref` | `10`, `1` | Gripper term of `rho` (paper beta_g, e^g_ref) |
| `--speed_gain`, `--speed_ref`, `--speed_fraction` | `0`, calibrated, `0.25` | Optional slow-motion precision term |
| `--accel_gain`, `--accel_ref` | `0`, `1` | Optional paper e_j term (saturates on RoboTwin, so off) |
| `--criticality_gain` | `10` | Weight of optional criticality labels |
| `--dilation_s` | `0.15` | Spread `rho` to neighbouring samples (seconds) |
| `--knot_density_gain` | `10` | Paper beta for the teacher knots inside a chunk |
| `--scale_horizon` | `16` | Window length used to calibrate the motion scales D |
| `--arms`, `--profile`, `--base_wxyz`, `--sample_dt` | `left right`, `robotwin` | RoboTwin reading options, as in the converter |
| `--base_motion` | `drop` | RoboCasa episodes that move the mobile base |

## Training options (all training scripts)

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 10 | `action` stage epochs (action predictor, SigLIP frozen) |
| `--finetune_epochs` | 5 | `finetune` stage epochs (action loss, last SigLIP blocks trainable) |
| `--timing_epochs` | 10 | `timing` stage epochs (query-timing head) |
| `--action_steps`, `--finetune_steps`, `--timing_steps` | off | Optimizer steps per stage; each overrides the matching epoch count |
| `--warmup_steps` | 500 | Linear warmup (at most 10% of a stage), then cosine decay |
| `--spatial_tokens` / `--no-spatial_tokens` | on | Condition the flow on all SigLIP patch tokens of every view through cross-attention; off = one pooled feature vector |
| `--proprio` / `--no-proprio` | on | Condition on robot state: base-frame end-effector position, 6-D rotation and last gripper command, per arm |
| `--augment` / `--no-augment` | on | Random 90% crop and ±20% colour jitter while training (spatial mode); evaluation uses a centre crop |
| `--vision_finetune_layers` | 4 | SigLIP vision blocks trained in the `finetune` stage; 0 skips the stage (needs spatial mode) |
| `--vision_lr_scale` | 0.1 | Learning-rate multiplier for those SigLIP blocks |
| `--num_workers` | 8 | DataLoader workers |
| `--batch_size`, `--lr` | 8, 2e-4 | Optimizer settings |
| `--num_control_points`, `--degree` | 8, 3 | B-spline size |
| `--chunk_horizon` | 16 | Window length in samples, when `--supported_horizons` is not set |
| `--supported_horizons 0.1 0.3` | off | Timestamped windows with horizons in this range (seconds) |
| `--horizon_duration` | median window | Inference horizon in seconds, inside training support |
| `--control_dt` | 0.02 | Shortest allowed window, in seconds |
| `--num_ode_steps` | 15 | Flow-matching integration steps, stored in the checkpoint |
| `--knot_loss_coeff` / `--flow_loss_coeff` | 0.5 / 1 | Loss weights alpha, gamma (Table 1); the gripper is a flow channel |
| `--gripper_density_gain` | 10 | beta_g, gripper-width variation weight in the knot density |
| `--gripper_threshold` | 0.5 | theta_g, width threshold for gripper events in the timing teacher |
| `--knot_supervision` | `kl` | `kl` (span KL) or `position` (L_pos ablation) |
| `--uniform_knots` | off | Ablation: fixed uniform knots instead of learned knots |
| `--per_arm_knots` | off | Ablation for two-arm data: one knot partition per arm |
| `--vlm_name` | `google/siglip-base-patch16-224` | Backbone |
| `--device` | `cuda` if available | Training device |
| `--test_backbone` | off | Tiny offline backbone, for tests only |

The observation options (`--spatial_tokens`, `--proprio`, `--augment`,
`--vision_finetune_layers`) default to on, so a plain training command trains
the full model. To reproduce the earlier pooled-feature model, pass
`--no-spatial_tokens --no-proprio --no-augment --vision_finetune_layers 0`.

---

## Reference

### How training works

1. Prepare overlapping timestamped windows at every observation by default.
   Each window's motion target is the offset of every sample's pose from the
   pose at the window start (Eq. anchor_delta): `R_b^T (p - p_k)` in metres and
   the rotation vector `R_b^T Log(R R_k^T)` in radians, in the arm base frame.
   The first sample is therefore zero. The gripper width in [0,1] is appended
   per arm as `2w - 1`.
   `--supported_horizons 0.1 0.3` selects nearby observed endpoints within that
   supported range; otherwise `--chunk_horizon 16` keeps fixed sample-count
   windows. Windows never cross a recorded terminal sample and must span at least
   `--control_dt`.
2. Calibrate six motion scales as the 95th percentile of absolute offsets over
   all samples of all training windows, pooling corresponding channels across
   arms (so they depend on the horizon range). A positive floor defaults to
   `1e-6`; calibration is frozen and checkpointed.
3. Knot span targets use a density with a motion term (normalized second finite
   difference of the offsets) and a separate gripper term `beta_g` on the width
   change rate, plus optional criticality.
4. The `action` stage trains the action predictor with flow matching over all
   channels (motion and width) and knot span KL; there is no separate gripper
   loss. Control-point targets are refitted on the predicted knots and detached.
   The first control point's motion is zero in targets and in generated samples.
   SigLIP is frozen and the timing head is not updated in this stage. With
   `--spatial_tokens`, every policy block cross-attends to a token memory: the
   projected patch tokens of every view plus a per-view embedding, one pooled
   instruction token and, with `--proprio`, one robot-state token. The robot
   state is normalized with statistics saved in the calibration.
5. The optional `finetune` stage keeps the action loss and also trains the last
   `--vision_finetune_layers` SigLIP vision blocks (plus its final layer norm
   and head) at `lr × --vision_lr_scale`. The text tower stays frozen. Every
   stage has its own AdamW, warmup + cosine schedule and gradient clipping at 1.0.
6. The `timing` stage freezes the whole action predictor (including the backbone), generates
   control points using the inference flow solver, and trains only
   `policy.freq_head`. Inputs are context, log horizon, detached spans, and
   probes of the planned motion and clamped width. The loss is log-MSE on query
   intervals.
7. Measured valid intervals override the fallback teacher. Fallback events at
   sample indices `j >= 1` are acceleration magnitude `e_{j-1} >= e_crit`, any
   arm's width crossing `theta_g = 0.5`, or criticality `>= 0.5`. The target is
   the first event time minus a nonnegative anticipation margin, clipped to the
   interval bounds. No event gives the maximum interval.

### Action and dataset contract

Episodes supply timestamped end-effector poses and gripper widths. The
preferred layout (written by the benchmark converters) is:

| Field | Meaning |
|---|---|
| `ee_poses` dataset | `(T, arms, 7)` poses `[x, y, z, qw, qx, qy, qz]` in the logging frame |
| `gripper` dataset | `(T, arms)` raw widths, kept continuous |
| `action_type` attribute | `ee_pose` |
| `action_frame` attribute | `base` (offsets are expressed in the arm base frame) |
| `base_wxyz` attribute | Rotation R_b from arm base to logging frame, one or one per arm (identity if omitted) |
| `gripper_open_value`, `gripper_closed_value` | Raw width values of fully open and fully closed |
| `sample_dt` | Positive sample period in seconds, if no `timestamps` dataset |

Widths map affinely to `w` in [0,1] (0 closed, 1 open) and are never
thresholded during training; datasets that log only a binary gripper map to
{0, 1}. Datasets that log motion instead use `actions` `(T, 7*arms)` with
`action_type` `twist` (m/s and axis-angle rad/s acting on `[t_j, t_{j+1})`) or
`delta_twist` (per-step increments), followed by the raw width per arm; these are
integrated into base-frame poses from the identity. Angular channels are
axis-angle, **not Euler angles**.

Controller-normalized commands and joint actions must be converted before
loading. The generic loaders do not infer units or frames, pad or truncate
actions, invent images, skip corrupt files, or substitute synthetic data after a
loading failure.

Supported HDF5 layouts:

* RoboTwin: `actions` or `action`, plus `observation/head_camera/rgb` and
  `observation/wrist_camera/rgb`.
* RoboCasa: optional `data/demo_*` groups, `actions` or `action`, plus
  `obs/agentview_image` (or `obs/frontview_image`) and `obs/robot0_eye_in_hand_image`.
* Both also accept one combined image dataset via `--images_path`. RGB must be
  uint8 or floats in [0,1], and camera/action sample counts must match. Camera
  datasets may hold decoded RGB arrays or one encoded JPEG/PNG byte string per frame.

Optional datasets are `timestamps` (seconds), `criticality`
(values in [0,1]) and measured `query_intervals` (seconds per observation).
Binary `terminated` or `dones` arrays and a `termination_time` attribute can mark
an earlier episode end. Conflicting task or action metadata is rejected.

Prediction returns physical `control_points` with shape `(B,K,7*num_arms)` (per
arm six offsets in metres/radians and the unclamped width), knots, horizon and
query interval, with `action_type="anchor_delta"`.

### Backbone and checkpoints

The production backend is SigLIP, using its checkpoint processor for image
resizing/normalization and text tokenization. Model-loading failures propagate;
the offline test backbone is selected only by `--test_backbone`.

Training saves `format_version=4`, `model_config`, `model_state`,
`training_stage`, `step` and `calibration`, once per epoch or at least every
2000 steps, as `adaspevla_<benchmark>_<task>_<stage>.pt` (`action`,
`finetune`, `timing`). `*_latest.pt` is written only when the timing stage
ends, and it is the file evaluation needs; weights load strictly. Checkpoints
trained before the observation options existed have none of their config keys
and load as the pooled-feature model. Checkpoints older than format 4 must be
retrained.

### Using the policy in your own control loop

```python
from adaspevla.benchmarks.rollout import load_policy
from adaspevla.execution import SynchronousSplineExecutor

policy, calibration, _ = load_policy(
    "checkpoints/robotwin_right/adaspevla_robotwin_beat_block_hammer_latest.pt", "cuda"
)

# observe() returns single-batch RGB images on the model device, instructions and,
# for pose commands, "anchor_pose": the (arms, 7) xyz+wxyz end-effector pose
# observed with the images. command(action) advances the robot by control_dt.
executor = SynchronousSplineExecutor(
    policy, observe, command, control_dt=0.02, command_type="pose",
    world_from_base=[[0.707, 0, 0, 0.707]],  # R_b as wxyz, one per arm (conversion metadata)
)
for _ in range(250):
    action, queried = executor.step()
executor.reset()
```

Step `n` of a prefix samples the spline at `s = (n+1) control_dt / H`.
`command_type="pose"` emits per arm the absolute target `[x,y,z,qw,qx,qy,qz,w]`
by Eq. delta_to_target, `p = p_k + R_b a_pos`, `R = Exp(R_b a_rot) R_k`.
`command_type="anchor_delta"` (default) emits the raw offsets and width; an
increment or velocity controller derives its command from consecutive targets.
The width is clamped to [0,1]; `gripper_mode="binary"` sends `1[w >= 0.5]` for
binary grippers. The number of steps executed per query is
`max(1, min(floor(H/control_dt), ceil(tau/control_dt)))`; horizons below one
control period are rejected.

Optional `boundary_matching=True` replaces the width entry of the new control
point zero with the last emitted clamped width (Eq. boundary); motion needs no
correction because each call is re-anchored. A positive `motion_limits` vector
(one entry per motion channel) checks every planned offset before the command
callback. Optional `stop_callback` and `fault_callback` halt execution with
`StopIteration`. This option is off by default and still needs robot validation.
