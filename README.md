# AdaSpeVLA: Adaptive Knot Placement and Supervised Query Timing

AdaSpeVLA predicts continuous base-frame velocity splines and separate discrete
gripper commands from camera images and language instructions. It alternates
policy calls with spline-prefix execution. Longer query intervals
execute more actions between predictions; shorter intervals refresh sooner.

## Quick start: native RoboTwin, right arm only

Run from the repository root. For the official state/action-paired HDF5 export
shown in your dataset, the native adapter performs the conversion during loading:

```bash
pip install -e .
python -m scripts.train_robotwin \
  --data_dir /lustre/fswork/projects/rech/sdh/ueh86le/DynamicDAPolicy/data/robotwin/source_domain \
  --task_name beat_block_hammer \
  --robotwin_native_right \
  --epochs 10 --timing_epochs 10 \
  --output_dir ./checkpoints/robotwin_right
```

This trains the action predictor and then the query-timing head. The adapter uses:

| HDF5 dataset | Use |
|---|---|
| `vision/cam_head/colors` | Head-camera images |
| `vision/cam_right_wrist/colors` | Right-wrist images |
| `state/right_ee_poses`, `action/right_ee_poses` | Current/next world poses, converted to base-frame linear and angular velocities |
| `action/right_ee_joint_states` | Binary gripper targets, open when the value is at least `0.5` |
| `instructions` | First nonempty instruction from the JSON list |
| `additional_info/frequency` | Collector save stride used to estimate the sample period |

Only the right arm supplies action targets. Encoded images are decoded using
RoboTwin's marked/legacy color conventions. Pose quaternions use `wxyz`; angular
velocity comes from the relative rotation's axis-angle vector. The adapter checks
the one-frame state/action alignment and produces seven channels: six physical
motion values plus one gripper value.

The default base orientation is the official Aloha quaternion
`[0.707, 0, 0, 0.707]`. For a changed embodiment, set
`--robotwin_base_wxyz qw qx qy qz`. Use `--robotwin_gripper_threshold` to change
the binary threshold. Native mode supplies the action metadata; do not combine it
with `--dual_arm`, `--actions_path`, `--images_path`, or conflicting gripper coding.

**Timing is approximate for this export.** The official collector writes its
save stride into `frequency`: with stride 15 and the default simulation timestep
`1/250` s, the nominal sample period is `0.06` s, not `1/15` s. Extra frames at
segment boundaries make exact timing unrecoverable without per-frame timestamps.
The adapter warns and records the conversion settings in checkpoint calibration.
`--robotwin_simulation_dt` changes the assumed simulation timestep; `--sample_dt`
overrides the nominal sample period. Neither recovers missing irregular timing.
The dataset sample period is separate from the executor's `control_dt`.



## Two-stage training

1. Calibrate six motion scales as the 95th percentile of absolute physical motion
   values over the training split, pooling corresponding channels across arms.
   A positive floor defaults to `1e-6`; calibration is frozen and checkpointed.
2. Prepare overlapping timestamped windows at every observation by default.
   `--supported_horizons 0.1 0.3` selects nearby observed endpoints within that
   supported range; otherwise `--chunk_horizon 16` retains fixed sample-count
   windows. Windows never cross a recorded terminal sample and must span at least
   `--control_dt`. Ragged fitting samples are handled by `collate_windows`.
3. Stage 1 trains the action predictor with flow matching, knot KL/entropy and
   gripper BCE. Control-point targets are refitted on the predicted knots and
   detached. The timing head is not updated in this stage.
4. Stage 2 freezes the whole action predictor (including the backbone), generates
   motion using the inference flow solver, and trains only `policy.freq_head`.
   Inputs are context, log horizon, detached spans, physical motion probes, and
   detached predicted gripper logits. Loss remains log-MSE on query intervals.
   Neither target control points nor future demonstration motion are timing inputs.
5. Measured valid intervals override the fallback teacher. Fallback events at
   sample indices `j >= 1` are acceleration magnitude `>= e_crit`, any gripper
   transition, or criticality `>= 0.5`. The target is the first event time minus a
   nonnegative anticipation margin, clipped to the interval bounds. No event gives
   the maximum interval. Knot spans do not set timing labels.

## Action and dataset contract

The canonical input has exactly seven channels per arm. Native RoboTwin mode
above constructs this representation and its metadata. For already converted
datasets, declare these HDF5 attributes (on the root or episode group), or
equivalent loader/training arguments:

| Field | Meaning |
|---|---|
| `action_type` | `twist` or `delta_twist` |
| `action_frame` | `base` |
| `gripper_open_value`, `gripper_closed_value` | Explicit discrete source values |
| `sample_dt` | Positive sample period in seconds, if no `timestamps` dataset |

For `twist`, motion is `(vx, vy, vz, wx, wy, wz)` in m/s and rad/s. Angular
channels are base-frame axis-angle rates, **not Euler-angle rates**. For
`delta_twist`, divide spatial translation/rotation increments by their explicit
sample period or per-action `action_durations`. With irregular timestamps, every
increment still needs its own duration, including the last action.

Outside the supported native RoboTwin profile, controller-normalized commands,
joint actions, poses, and tool-frame actions must be converted explicitly before
loading. The generic adapters do not infer units or frames,
pad/truncate actions, invent images, skip corrupt files, or substitute synthetic
data after a loading failure. Continuous gripper position controls need an
explicit task-specific conversion to the binary contract.

Supported HDF5 layouts for already converted data:

* RoboTwin: `actions` or `action`, plus `observation/head_camera/rgb` and
  `observation/wrist_camera/rgb`.
* RoboCasa: optional `data/demo_*` groups, `actions` or `action`, plus
  `obs/agentview_image` (or `obs/frontview_image`) and `obs/robot0_eye_in_hand_image`.
* Both also accept an explicit combined `images` dataset. RGB must be uint8 or
  floats in [0,1], and camera/action sample counts must match.

If the camera paths differ from these defaults, inspect an episode without reading
its image/action contents:

```bash
python -m scripts.inspect_hdf5 /path/to/episode.hdf5
```

Then add `--camera_paths <first-camera-dataset> <second-camera-dataset>` to the
training command, in the desired view order. Paths are relative to the episode
group; the configured view count follows this list. A single camera is supported.
Alternatively use `--images_path <combined-image-dataset>`. An optional
`--actions_path <converted-action-dataset>` selects an already converted physical
action dataset; it does not reinterpret joint positions or end-effector poses as
twists. Camera datasets may contain decoded RGB arrays or one encoded JPEG/PNG byte
string per frame (fixed-width HDF5 strings or variable-length byte arrays).
Frames are decoded to RGB and must have matching resolution and frame counts. A missing-dataset error now includes the filename and actual
dataset paths/shapes/dtypes so schema mismatches are visible.

Optional datasets are `timestamps` (seconds), `action_durations`, `criticality`
(values in [0,1]), and measured `query_intervals` (seconds per observation). Binary `terminated`
or `dones` arrays and a `termination_time` episode attribute can mark an earlier
episode end; otherwise the last recorded timestamp bounds the episode.
Instruction text comes from metadata or the explicitly selected task name.
The loader searches the selected task subdirectory, or immediate HDF5 files in
`data_dir`; conflicting task or action metadata is rejected.

Internally, the spline has six motion channels per arm, normalized using the
training-calibrated, shared-across-arms six-channel scale vector. Grippers are trained separately and held between grid
samples. Prediction returns `control_points` with shape `(B,K,6*num_arms)`,
`gripper_commands` with shape `(B,S,num_arms)`, `gripper_times`, knots, horizon,
and query interval. Emitted commands contain seven channels per arm, with signed
gripper values (-1 closed, +1 open).

## Backbone and checkpoints

The production backend supports SigLIP and uses its cached checkpoint processor
for image resizing/normalization and fixed-length text tokenization. Vision and
text projections use their actual encoder dimensions. Frozen encoders remain in
evaluation mode. Model-loading failures propagate; the offline test backbone is
selected only by `--test_backbone`.

**Old checkpoints require retraining.** The corrected model changes the frequency
branch, motion dimensions, conditioning, and gripper head. Training saves
`format_version=3`, `model_config`, `model_state`, `training_stage`, and
`calibration`; evaluation requires a stage-2 checkpoint and loads weights strictly.
The timing MLP input size changed: version-2 checkpoints need retraining.

## Training with already converted datasets

```bash
# Real data must satisfy the contract above. Metadata may already be in HDF5.
python -m scripts.train_robotwin --data_dir /path/to/converted/data --task_name beat_block_hammer --epochs 10 --timing_epochs 10
python -m scripts.train_robocasa --data_dir /path/to/converted/data --task_name PnPCounterToCab --epochs 10 --timing_epochs 10
```

Use `--dual_arm` for already converted RoboTwin two-arm data. For explicitly declared uniform
base-frame velocity data without HDF5 action attributes, add
`--action_type twist --action_frame base --sample_dt 0.02
--gripper_open_value 1 --gripper_closed_value -1` only if these match the source.
The default inference horizon is the median chunk duration in the training data;
`--horizon_duration` can set it explicitly within training support. `--epochs`
controls stage 1; `--timing_epochs` controls stage 2. Integration steps are stored
in the checkpoint and used by default in evaluation.

A completely offline pipeline check uses separate explicit test flags:

```bash
python -m scripts.train_robotwin --synthetic_data --test_backbone --epochs 1 --timing_epochs 1 --num_ode_steps 2 --batch_size 2 --device cpu --output_dir /tmp/adaspevla-smoke
python -m scripts.eval_robotwin --ckpt_path /tmp/adaspevla-smoke/adaspevla_robotwin_beat_block_hammer_latest.pt --num_episodes 1 --duration 0.2 --device cpu
python -m pytest tests -q
```

## Adaptive prefix execution

```python
from adaspevla.execution import SynchronousSplineExecutor

# observe returns single-batch RGB images on the model device and instructions.
# command advances the environment by control_dt or explicitly paces a controller.
executor = SynchronousSplineExecutor(
    policy, observe, command, control_dt=0.02, command_type="twist"
)
for _ in range(250):
    action, queried = executor.step()
executor.reset()
```

`command_type="delta_twist"` multiplies motion values by `control_dt`; grippers
are unchanged. Device-specific normalization and coordinate transforms belong in
the command callback. Policy inference blocks its step and adds wall-clock time;
spline sample rate is distinct from achieved policy-call rate.

The executed step count is `max(1, min(floor(H/control_dt), ceil(tau/control_dt)))`,
with numerically near-integer ratios snapped before rounding. Horizons below one
control period are rejected. The complete commanded prefix remains within `H`.

Optional `boundary_matching=True` reuses the last emitted physical motion twist as
new control point zero, before computing the timing probes. Supply a positive
per-channel `motion_limits` vector; every emitted motion is checked before the
command callback. This matches the initial command value only, not derivatives
or actual robot tracking. The existing derivative-refitting utility is separate.
Optional `stop_callback` and `fault_callback` halt execution with `StopIteration`
before/after commands. The option is disabled by default and still needs robot
validation.
