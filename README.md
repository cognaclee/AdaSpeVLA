# AdaSpeVLA: Adaptive Knot Placement and Supervised Query Timing

AdaSpeVLA predicts continuous base-frame velocity splines and separate discrete
gripper commands from camera images and language instructions. It alternates
policy calls with spline-prefix execution. Longer query intervals
execute more actions between predictions; shorter intervals refresh sooner.

## Corrected training path

1. Build nonuniform knot targets by inverse cumulative complexity density. High
   local acceleration or criticality produces shorter spans, with a true minimum
   span bound. Actual demonstration timestamp widths are used.
2. Predict knots from the observation and physical horizon. **Refit flow targets on those predicted
   knots every batch**, and condition the flow model on the knots and horizon.
   Cached control points from a different basis are not used by the loss.
3. Train `freq_head` with squared error between log predicted and log target query
   intervals. This is in addition to knot KL/entropy, flow matching, and separate
   gripper binary cross-entropy losses. The KL direction is `KL(target || prediction)`.
4. Use measured `query_intervals` when supplied. Otherwise use an explicit
   heuristic teacher: near-constant velocity permits the maximum prefix; motion
   variation shifts timing toward the first physical knot span. Criticality and
   gripper-event timestamps can shorten the interval. This is **not** a measured
   manipulation-error tolerance or evidence of precision improvement.

The physical horizon is supplied from training timestamps and an explicit
inference configuration, rather than predicted by an unsupervised duration head.
The checkpoint records its default horizon. No speed-rescaling invariance is
assumed when requesting a different horizon.

## Action and dataset contract

Each input arm has exactly seven channels. Declare these HDF5 attributes (on the
root or episode group), or equivalent loader/training arguments:

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

Controller-normalized commands, joint actions, poses, and tool-frame actions must
be converted explicitly before loading. The adapters do not infer units or frames,
pad/truncate actions, invent images, skip corrupt files, or substitute synthetic
data after a loading failure. Continuous gripper position controls need an
explicit task-specific conversion to the binary contract.

Supported HDF5 layouts:

* RoboTwin: `actions` or `action`, plus `observation/head_camera/rgb` and
  `observation/wrist_camera/rgb`.
* RoboCasa: optional `data/demo_*` groups, `actions` or `action`, plus
  `obs/agentview_image` (or `obs/frontview_image`) and `obs/robot0_eye_in_hand_image`.
* Both also accept an explicit combined `images` dataset. RGB must be uint8 or
  floats in [0,1], and camera/action sample counts must match.

Optional datasets are `timestamps` (seconds), `action_durations`, `criticality`
(values in [0,1]), and measured `query_intervals` (seconds per observation).
Instruction text comes from metadata or the explicitly selected task name.
The loader searches the selected task subdirectory, or immediate HDF5 files in
`data_dir`; conflicting task or action metadata is rejected.

Internally, the spline has six motion channels per arm, scaled by 0.1 m/s and
1 rad/s for learning. Grippers are trained separately and held between grid
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
`format_version=2`, `model_config`, and `model_state`; evaluation restores that
configuration and loads weights strictly.

## Installation and training

```bash
pip install -e .
# Real data must satisfy the contract above. Metadata may already be in HDF5.
python scripts/train_robotwin.py --data_dir /path/to/converted/data --task_name beat_block_hammer --epochs 10
python scripts/train_robocasa.py --data_dir /path/to/converted/data --task_name PnPCounterToCab --epochs 10
```

Use `--dual_arm` for RoboTwin two-arm data. For explicitly declared uniform
base-frame velocity data without HDF5 action attributes, add
`--action_type twist --action_frame base --sample_dt 0.02
--gripper_open_value 1 --gripper_closed_value -1` only if these match the source.
The default inference horizon is the median chunk duration in the training data;
`--horizon_duration` can set it explicitly. `--frequency_loss_coeff` must be positive.

A completely offline pipeline check uses separate explicit test flags:

```bash
python scripts/train_robotwin.py --synthetic_data --test_backbone --epochs 1 --batch_size 2 --device cpu --output_dir /tmp/adaspevla-smoke
python scripts/eval_robotwin.py --ckpt_path /tmp/adaspevla-smoke/adaspevla_robotwin_beat_block_hammer_latest.pt --num_episodes 1 --duration 0.2 --device cpu
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

The optional boundary-refitting API matches a spline's value and physical-time
first/second derivatives. For a velocity spline these mean velocity, acceleration,
and jerk. It requires the new horizon for correct scaling and rejects a second
derivative without a first derivative. It is not applied automatically by the
executor and does not guarantee actual robot tracking or zero jerk.

## Validation status

The tests exercise numerical spline correctness, timing gradients and learning,
data validation, unit conversion, discrete grippers, backbone interfaces, and
sequential execution. Evaluation scripts are synthetic smoke checks, not robot
benchmark rollouts. They report query counts, wall time, and separate linear and
angular jerk energies. The paper's unsupported performance tables have been
replaced by a proposed evaluation protocol. No task-success, precision, GPU-cost,
or compute-saving result is claimed. See [REVIEW.md](REVIEW.md) for the resolution
of each review finding and remaining empirical validation limits.
