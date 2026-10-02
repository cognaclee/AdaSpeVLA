# Native RoboTwin HDF5 (right arm only), step by step

Use this guide for the official RoboTwin state/action-paired HDF5 export (for
example `.../DynamicDAPolicy/data/robotwin/source_domain`). The adapter reads the
head and right-wrist cameras and the right-arm poses, converts them to six
base-frame motion channels plus a binary gripper while loading, and trains on
that. The left arm is not used as an action target.

For RoboTwin 2.0 data downloaded through XPolicyLab, or for two-arm training and
simulator evaluation, use [benchmarks.md, Part A](benchmarks.md#part-a-robotwin-20-through-the-xpolicylab-adapter)
instead.

## Step 1: Install

Follow [README, Step 1](../README.md#step-1-install), then:

```bash
conda activate adaspevla
cd /home/lzezeng/codes/AdaSpeVLA
```

## Step 2: Check the data layout

The loader looks for `<data_dir>/<task_name>/**/*.hdf5` (or `*.h5`); if that
folder does not exist, it uses the HDF5 files directly inside `<data_dir>`.

```bash
DATA=/lustre/fswork/projects/rech/sdh/ueh86le/DynamicDAPolicy/data/robotwin/source_domain
ls ${DATA}/beat_block_hammer | head
python -m scripts.inspect_hdf5 $(ls ${DATA}/beat_block_hammer/*.hdf5 | head -1)
```

The episode must contain these datasets:

| HDF5 dataset | Use |
|---|---|
| `vision/cam_head/colors` | Head-camera images |
| `vision/cam_right_wrist/colors` | Right-wrist images |
| `state/right_ee_poses`, `action/right_ee_poses` | Current and next world poses, converted to base-frame twists |
| `action/right_ee_joint_states` | Gripper target, open when the value is at least `0.5` |
| `instructions` | First nonempty instruction from the JSON list |
| `additional_info/frequency` | Collector save stride, used to estimate the sample period |

## Step 3: Train

```bash
python -m scripts.train_robotwin \
    --data_dir ${DATA} \
    --task_name beat_block_hammer \
    --robotwin_native_right \
    --epochs 10 --timing_epochs 10 --batch_size 8 \
    --output_dir ./checkpoints/robotwin_right
```

Training prints a warning that timing is approximate (see "Timing limitation"
below); that is expected. Stage 1 trains the action predictor, stage 2 the
query-timing head. Output:
`./checkpoints/robotwin_right/adaspevla_robotwin_beat_block_hammer_latest.pt`.

Options for this mode:

| Option | Default | When to change it |
|---|---|---|
| `--robotwin_base_wxyz qw qx qy qz` | `0.707 0 0 0.707` | Another embodiment than the official Aloha-AgileX |
| `--robotwin_gripper_threshold` | `0.5` | Another binary threshold for the gripper |
| `--robotwin_simulation_dt` | `1/250` | The collector used another simulation timestep |
| `--sample_dt` | `frequency × simulation_dt` | You know the true sample period in seconds |
| `--camera_paths A B` | head, right wrist | Other camera datasets, in view order |

All general training options (`--supported_horizons`, `--knot_supervision`,
loss weights, ...) are listed in the
[README options table](../README.md#training-options-all-training-scripts).

Do not add `--dual_arm`, `--actions_path` or manual gripper coding
(`--gripper_open_value`, `--gripper_closed_value`) in native mode.

## Step 4: Check the checkpoint

```bash
python -m scripts.eval_robotwin \
    --ckpt_path ./checkpoints/robotwin_right/adaspevla_robotwin_beat_block_hammer_latest.pt \
    --num_episodes 1
```

This is a smoke evaluation without a simulator: it confirms the checkpoint loads
and produces actions. For simulator success rates, convert the data with
`ARMS=right` through XPolicyLab and evaluate as in
[benchmarks.md, Part A](benchmarks.md#part-a-robotwin-20-through-the-xpolicylab-adapter).

---

## Reference

### Verified upstream conventions

Inspected RoboTwin commit `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`:

- [Exporter](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/utils/pkl2hdf5.py): state and images use frames `[:-1]`; action uses frames `[1:]`. Instructions are a JSON list.
- [Robot poses and grippers](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/robot/robot.py): `_trans_endpose` returns world position and a wxyz quaternion. Normalized gripper targets range from 0 closed to 1 open and can be intermediate during motion.
- [Official Aloha configuration](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/d513175169bcdcdb6ddf3de08f10cddc55ad91f1/embodiments/aloha-agilex/config.yml): robot root quaternion is `[0.707,0,0,0.707]`.
- [Collection](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/_base_task.py): simulation timestep defaults to 1/250 s; `save_freq` is a step stride, stored as `additional_info/frequency`.
- [Image codec](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/data/decode_image_bit.py): unmarked legacy encoded images have reversed RGB channels; a JPEG COM marker `XPL-RGB1` identifies standard RGB. The native loader handles both.

### Conversion

For each row, `state/right_ee_poses` is the current pose `(p0,R0)` and
`action/right_ee_poses` the next pose `(p1,R1)`. The adapter checks the official
one-frame shift between consecutive rows, comparing orientations as rotations so
quaternion sign flips are harmless. With robot-root rotation `Rbase`:

```
v_base     = Rbase.T @ (p1-p0) / dt
omega_base = Rbase.T @ Log(R1 @ R0.T) / dt
```

`Log` returns the axis-angle rotation vector. These are interval-average
velocities reconstructed from demonstrated poses, not the original controller
commands. The gripper target is `action/right_ee_joint_states >= 0.5`, an explicit
adaptation to the binary gripper head.

### Timing limitation

The collector stores its save stride in `frequency`, so the nominal sample period
is `save_freq × simulation_dt`, e.g. `15 × (1/250) = 0.06 s`, **not** `1/15 s`.
The collector also saves extra frames at segment boundaries, so without per-frame
timestamps the exact intervals cannot be recovered. The adapter warns about this
and records its time source in the checkpoint calibration. `--sample_dt` and
`--robotwin_simulation_dt` change the nominal period but cannot recover irregular
timing; exact-time experiments need a timestamp-preserving export.

Calibration records the source profile, selected arm, base rotation, gripper
threshold and timing conversion alongside the normalization statistics.
