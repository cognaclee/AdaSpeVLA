# AdaSpeVLA: Adaptive Knot Placement and Supervised Query Timing

AdaSpeVLA predicts continuous base-frame velocity splines and separate discrete
gripper commands from camera images and language instructions. It alternates
policy calls with spline-prefix execution. Longer query intervals execute more
actions between predictions; shorter intervals refresh sooner.

This README covers installation, a smoke test, and training on your own
RoboTwin HDF5 data. Benchmark training and evaluation (RoboTwin 2.0, RoboDojo,
RoboCasa365, and the XPolicyLab adapter) are in
[docs/benchmarks.md](docs/benchmarks.md).

| I want to... | Go to |
|---|---|
| Install AdaSpeVLA and check it works | [Step 1](#step-1-install) and [Step 2](#step-2-check-the-installation) below |
| Train on a native RoboTwin HDF5 export (right arm) | [Step 3](#step-3-train-on-native-robotwin-hdf5-right-arm) and [docs/robotwin_native.md](docs/robotwin_native.md) |
| Train on HDF5 data I already converted | [Step 4](#step-4-train-on-already-converted-hdf5-data) |
| Train and evaluate on RoboTwin 2.0 through XPolicyLab | [docs/benchmarks.md, Part A](docs/benchmarks.md#part-a-robotwin-20-through-the-xpolicylab-adapter) |
| Train and evaluate on RoboDojo | [docs/benchmarks.md, Part B](docs/benchmarks.md#part-b-robodojo-through-the-same-adapter) |
| Train and evaluate on RoboCasa365 | [docs/benchmarks.md, Part C](docs/benchmarks.md#part-c-robocasa365) |
| Run the shared vs per-arm knot ablation | [docs/benchmarks.md, Part D](docs/benchmarks.md#part-d-bimanual-knot-ablation-shared-vs-per-arm-knots) |

All commands run from the repository root unless a step says otherwise.

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
cd /home/lzezeng/codes/AdaSpeVLA
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
# End-to-end smoke run: synthetic data, tiny offline backbone, CPU
python -m scripts.train_robotwin --synthetic_data --test_backbone \
    --epochs 1 --timing_epochs 1 --num_ode_steps 2 --batch_size 2 \
    --device cpu --output_dir /tmp/adaspevla-smoke
python -m scripts.eval_robotwin \
    --ckpt_path /tmp/adaspevla-smoke/adaspevla_robotwin_beat_block_hammer_latest.pt \
    --num_episodes 1 --duration 0.2 --device cpu
```

`scripts.eval_robotwin` and `scripts.eval_robocasa` are smoke evaluators. They
run the spline executor without a simulator, so they check that a checkpoint
loads and produces actions. They do not measure task success. For simulator
evaluation, use [docs/benchmarks.md](docs/benchmarks.md).

## Step 3: Train on native RoboTwin HDF5 (right arm)

Use this for the official RoboTwin state/action-paired HDF5 export. The native
adapter converts poses to base-frame twists while loading, so there is no
separate conversion step.

```bash
# 1. Locate the episodes. The loader searches <data_dir>/<task_name>/ recursively,
#    or else the HDF5 files directly inside <data_dir>.
ls /path/to/robotwin/source_domain/beat_block_hammer | head

# 2. Optional: list the dataset paths of one episode
python -m scripts.inspect_hdf5 /path/to/robotwin/source_domain/beat_block_hammer/<episode>.hdf5

# 3. Train both stages (stage 1: actions, stage 2: query timing)
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
the stage-2 checkpoint that evaluation needs, plus one file per stage.

Options for this mode:

| Option | Default | Meaning |
|---|---|---|
| `--robotwin_base_wxyz qw qx qy qz` | `0.707 0 0 0.707` (Aloha) | Robot base orientation in world, for another embodiment |
| `--robotwin_gripper_threshold` | `0.5` | Gripper counts as open when the target is at least this value |
| `--robotwin_simulation_dt` | `1/250` | Collector timestep; sample period = `frequency` × this |
| `--sample_dt` | from file | Override the sample period in seconds |
| `--camera_paths A B` | head + right wrist | Override the camera datasets, in view order |

Do not combine `--robotwin_native_right` with `--dual_arm`, `--actions_path` or
manual gripper coding. Conversion details and the timing caveat are in
[docs/robotwin_native.md](docs/robotwin_native.md).

## Step 4: Train on already converted HDF5 data

Use this when your HDF5 files already contain base-frame twists that follow the
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
    --action_type twist --action_frame base --sample_dt 0.02 \
    --gripper_open_value 1 --gripper_closed_value -1
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

## Training options (all training scripts)

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 10 | Stage-1 epochs (action predictor) |
| `--timing_epochs` | 10 | Stage-2 epochs (query-timing head) |
| `--batch_size`, `--lr` | 8, 2e-4 | Optimizer settings |
| `--num_control_points`, `--degree` | 8, 3 | B-spline size |
| `--chunk_horizon` | 16 | Window length in samples, when `--supported_horizons` is not set |
| `--supported_horizons 0.1 0.3` | off | Timestamped windows with horizons in this range (seconds) |
| `--horizon_duration` | median window | Inference horizon in seconds, inside training support |
| `--control_dt` | 0.02 | Shortest allowed window, in seconds |
| `--num_ode_steps` | 15 | Flow-matching integration steps, stored in the checkpoint |
| `--knot_loss_coeff` / `--flow_loss_coeff` / `--gripper_loss_coeff` | 0.5 / 1 / 1 | Loss weights alpha, gamma, eta (Table 1) |
| `--knot_supervision` | `kl` | `kl` (span KL) or `position` (L_pos ablation) |
| `--uniform_knots` | off | Ablation: fixed uniform knots instead of learned knots |
| `--per_arm_knots` | off | Ablation for two-arm data: one knot partition per arm |
| `--vlm_name` | `google/siglip-base-patch16-224` | Backbone |
| `--device` | `cuda` if available | Training device |
| `--test_backbone` | off | Tiny offline backbone, for tests only |

---

## Reference

### How training works

1. Calibrate six motion scales as the 95th percentile of absolute physical motion
   values over the training split, pooling corresponding channels across arms.
   A positive floor defaults to `1e-6`; calibration is frozen and checkpointed.
2. Prepare overlapping timestamped windows at every observation by default.
   `--supported_horizons 0.1 0.3` selects nearby observed endpoints within that
   supported range; otherwise `--chunk_horizon 16` keeps fixed sample-count
   windows. Windows never cross a recorded terminal sample and must span at least
   `--control_dt`.
3. Stage 1 trains the action predictor with flow matching, knot span KL and
   gripper BCE. Control-point targets are refitted on the predicted knots and
   detached. The timing head is not updated in this stage.
4. Stage 2 freezes the whole action predictor (including the backbone), generates
   motion using the inference flow solver, and trains only `policy.freq_head`.
   Inputs are context, log horizon, detached spans, physical motion probes, and
   detached predicted gripper logits. The loss is log-MSE on query intervals.
5. Measured valid intervals override the fallback teacher. Fallback events at
   sample indices `j >= 1` are acceleration magnitude `>= e_crit`, any gripper
   transition, or criticality `>= 0.5`. The target is the first event time minus a
   nonnegative anticipation margin, clipped to the interval bounds. No event gives
   the maximum interval.

### Action and dataset contract

The canonical input has exactly seven channels per arm. For already converted
datasets, declare these HDF5 attributes (on the root or episode group), or the
equivalent training arguments:

| Field | Meaning |
|---|---|
| `action_type` | `twist` or `delta_twist` |
| `action_frame` | `base` |
| `gripper_open_value`, `gripper_closed_value` | Explicit discrete source values |
| `sample_dt` | Positive sample period in seconds, if no `timestamps` dataset |

For `twist`, motion is `(vx, vy, vz, wx, wy, wz)` in m/s and rad/s. Angular
channels are base-frame axis-angle rates, **not Euler-angle rates**. For
`delta_twist`, divide translation/rotation increments by their sample period or
per-action `action_durations`.

Controller-normalized commands, joint actions, poses, and tool-frame actions must
be converted before loading. The generic loaders do not infer units or frames,
pad or truncate actions, invent images, skip corrupt files, or substitute
synthetic data after a loading failure. Continuous gripper positions need an
explicit task-specific conversion to the binary contract.

Supported HDF5 layouts:

* RoboTwin: `actions` or `action`, plus `observation/head_camera/rgb` and
  `observation/wrist_camera/rgb`.
* RoboCasa: optional `data/demo_*` groups, `actions` or `action`, plus
  `obs/agentview_image` (or `obs/frontview_image`) and `obs/robot0_eye_in_hand_image`.
* Both also accept one combined image dataset via `--images_path`. RGB must be
  uint8 or floats in [0,1], and camera/action sample counts must match. Camera
  datasets may hold decoded RGB arrays or one encoded JPEG/PNG byte string per frame.

Optional datasets are `timestamps` (seconds), `action_durations`, `criticality`
(values in [0,1]) and measured `query_intervals` (seconds per observation).
Binary `terminated` or `dones` arrays and a `termination_time` attribute can mark
an earlier episode end. Conflicting task or action metadata is rejected.

Prediction returns `control_points` with shape `(B,K,6*num_arms)`,
`gripper_commands` with shape `(B,S,num_arms)`, `gripper_times`, knots, horizon
and query interval. Emitted commands have seven channels per arm, with signed
gripper values (-1 closed, +1 open).

### Backbone and checkpoints

The production backend is SigLIP, using its checkpoint processor for image
resizing/normalization and text tokenization. Model-loading failures propagate;
the offline test backbone is selected only by `--test_backbone`.

Training saves `format_version=3`, `model_config`, `model_state`,
`training_stage` and `calibration`. Evaluation requires a stage-2 checkpoint
(`*_latest.pt`) and loads weights strictly. **Checkpoints from older versions
must be retrained.**

### Using the policy in your own control loop

```python
from adaspevla.benchmarks.rollout import load_policy
from adaspevla.execution import SynchronousSplineExecutor

policy, calibration, _ = load_policy(
    "checkpoints/robotwin_right/adaspevla_robotwin_beat_block_hammer_latest.pt", "cuda"
)

# observe() returns single-batch RGB images on the model device and instructions.
# command(action) advances the robot by control_dt or paces a controller.
executor = SynchronousSplineExecutor(
    policy, observe, command, control_dt=0.02, command_type="twist"
)
for _ in range(250):
    action, queried = executor.step()
executor.reset()
```

`command_type="delta_twist"` multiplies motion values by `control_dt`; grippers
are unchanged. Device-specific normalization and coordinate transforms belong in
the command callback. The number of steps executed per query is
`max(1, min(floor(H/control_dt), ceil(tau/control_dt)))`; horizons below one
control period are rejected.

Optional `boundary_matching=True` reuses the last emitted physical twist as the
new control point zero. Supply a positive per-channel `motion_limits` vector;
every emitted motion is checked before the command callback. Optional
`stop_callback` and `fault_callback` halt execution with `StopIteration`. This
option is off by default and still needs robot validation.
