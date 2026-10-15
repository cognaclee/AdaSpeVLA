# Why success rates are low (benchmark-suite-20261009), 2026-10-14

## Current results

RoboCasa (target split, 50 episodes each). Each cell: success rate (mean policy queries per episode).

| Task | adaptive | fixed 0.05s | fixed 0.2s | fixed 0.4s | fixed 0.5s |
|---|---|---|---|---|---|
| CloseFridge | 40.0% (93) | 0% (900), 44 eps, still running | – | – | – |
| CoffeeSetupMug | 2% (73) | 0% (600) | 0% (150) | 2% (75) | 2% (60) |
| TurnOffStove | 0% (80) | 0% (750) | 2% (187) | 2% (93) | 4% (73) |
| TurnOnMicrowave | 2% (58) | 0% (225) | 2% (90) | 2% (50) | 2% (41) |
| TurnOnSinkFaucet | 2% (66) | 0% (600) | 0% (150) | 0% (75) | 2% (60) |

RoboTwin (demo_clean only). All 8 jobs hit the 20h limit; episodes completed in brackets.

| Task | adaptive | fixed 0.06s | fixed 0.24s |
|---|---|---|---|
| beat_block_hammer | 15.0% (100) | 1.2% (86) | – |
| press_stapler | 22.0% (100) | 33.0% (100) | 41.7% (12) |
| open_microwave | 22.4% (58) | – | – |
| hanging_mug | 1.4% (73) | – | – |
| handover_block | 0% (91) | – | – |
| place_phone_stand | 0% (100) | 0% (73) | – |
| stack_blocks_three | 0% (64) | – | – |
| stamp_seal | 0% (28), hung after episode 28 | – | – |

## Things ruled out

- RoboCasa action interface: `robocasa/robocasa/wrappers/gym_wrapper.py` maps
  `action.gripper_close` to robosuite -1/+1 correctly and flips rendered images
  upside down to match the LeRobot dataset videos.
- Low-level tracking: mean tracking error during evaluation is ~5 mm, so the
  OSC controller follows the spline targets.

## Main cause: the policy sees almost nothing of the scene

`adaspevla/models/vlm_backbone.py`: each camera image goes through a frozen
SigLIP and only `get_image_features` is kept, i.e. one pooled 768-d
image-text embedding per view. That vector, fused with the text embedding, is
the only input to the action head
(`AdaSpeVLAPolicy.forward_loss` / `predict_action_spline`).

- No spatial features: the 196 patch tokens per view are discarded, so exact
  object and handle positions are lost. This fits the pattern: CloseFridge
  (large push) works at 40%; knob, faucet and button tasks are ~0%.
- No proprioception: actions are offsets from the current end-effector pose,
  but the policy never receives that pose or the gripper width; it must infer
  them from the pooled embeddings.
- Frozen encoder and no image augmentation, so the visual features never
  adapt to the task.

## Second cause: RoboTwin is badly under-trained

RoboTwin has 50 demos per task, about 6k training windows. At batch size 32
for 20 epochs that is only ~4k gradient steps; Diffusion Policy-style RoboTwin
baselines train for hundreds of epochs. On RoboCasa, CloseFridge got ~80k
steps and its action loss was still falling at the last epoch
(0.079 at epoch 20/20).

## Proposed fixes, highest expected gain first

1. Feed spatial tokens: use SigLIP `last_hidden_state` (patch tokens) and let
   the flow transformer cross-attend to them, or compress them first with a
   small perceiver / attention pooling.
2. Add proprioception: end-effector pose (base frame) plus gripper width per
   arm as conditioning. Both benchmarks already expose it in observations.
3. Let vision adapt: unfreeze the last few SigLIP blocks at ~10x lower
   learning rate, and add random-crop and colour-jitter augmentation.
4. Train much longer on RoboTwin: ~300-500 epochs, or a fixed step budget such
   as 50k steps, instead of 20 epochs.

Fixes 1-3 change the method; fix 4 keeps the architecture as is.

Quick check before spending GPU time: predict on held-out episodes and compare
the error with the training loss (~10 min on one GPU).

Suggested pilot after the fixes: retrain TurnOffStove, TurnOnSinkFaucet,
beat_block_hammer and stamp_seal before the full suite. Hold the sharded
RoboTwin evaluation (`eval_robotwin.slurm`, 320 jobs) until the checkpoints
improve.

---

# Code changes made so far (2026-10-14)

Two sets of changes: splitting the evaluation so RoboTwin fits in 20h, then
fixes 1-4 above (the user asked for SigLIP to stay frozen in a first stage and
be fine-tuned in a second). Nothing has been submitted to SLURM yet.

## A. Evaluation splitting (no change to the method)

- `scripts/benchmarks/slurm_env.sh`: the manifest row is now
  `${TASK_ROW:-$SLURM_ARRAY_TASK_ID}`, so an array job can pick a task row
  that differs from its array index. Old scripts behave the same.
- `scripts/benchmarks/summarize_metrics.py`: `glob("*.jsonl")` became
  `rglob("*.jsonl")`, so sharded metrics in subdirectories are found.
- `eval_robotwin.slurm` (new): sharded RoboTwin evaluation. 2 configs
  (demo_clean, demo_randomized) x 5 modes (adaptive, 0.06, 0.24, 0.48, 0.5 s)
  x 8 tasks x 4 shards = 320 jobs (`--array=0-319%16`), 25 episodes per shard.
  Shard k uses RoboTwin seed k, so shards don't overlap. All demo_clean
  adaptive jobs come first. Metrics go to
  `eval_metrics/benchmark-suite-20261009-robotwin/`.
- `eval.slurm`: the array was cut to RoboCasa rows only
  (`--array=8-11,13,15,16%4`); RoboTwin rows 0-7 now go through
  `eval_robotwin.slurm`.

## B. Fixes 1-4 (spatial tokens, robot state, vision fine-tuning, longer training)

All new behaviour sits behind constructor arguments and CLI flags. The old
configuration (`--no-spatial_tokens --no-proprio`) still builds and trains,
and old checkpoints have none of the new config keys, so they load as before.
**However, the new flags default to on**, so any training run without them
now uses the new model.

### `adaspevla/benchmarks/pose_math.py`
- New `PROPRIO_PER_ARM = 10` and `proprio_features(poses, widths, world_from_base)`.
  For each arm: end-effector position in the base frame (3), a 6-D rotation
  (the first two columns of the base-frame rotation matrix, so there is no
  quaternion sign flip) (6), and the last commanded gripper width clipped to
  [0,1] (1). Training and both evaluators use this one function.

### `adaspevla/datasets/bspline_dataset.py`
- Each window also stores `proprio_features(poses[start], widths[start], ...)`,
  the robot state at the window's anchor. `widths[start]` is the last command.
- Normalisation stats: mean/std of the position part only (std floor 1e-3).
  The rotation and width parts keep mean 0 and std 1. They are saved as
  `calibration["proprio_mean"]` and `calibration["proprio_std"]`.
- `__getitem__` returns `"proprio"`.

### `adaspevla/models/vlm_backbone.py` (`MultiCameraVLMBackbone`)
- New arguments: `spatial_tokens`, `proprio_dim`, `augment`, `crop_ratio=0.9`.
- Spatial mode builds a token memory: the SigLIP patch tokens of every view
  (projected, plus a learned per-view embedding), one pooled instruction
  token, and one robot-state token if robot state is used. With 3 views that
  is 3 x 196 + 1 + 1 = 590 tokens. It is returned when
  `forward(..., return_memory=True)`. The pooled global condition is still
  computed. The GPU preprocessing constants are read only in spatial mode.
- Image preprocessing now runs on the GPU (crop, resize to the SigLIP input
  size, normalise with the processor's mean/std). When training with
  `augment`, it applies a random 90% crop and colour jitter (brightness,
  contrast, saturation ±20%). At inference it takes a centre crop.
- `proprio_proj` is an MLP. Its output is added to the global condition and,
  in spatial mode, also enters the memory.
- `set_trainable_vision_layers(n)` unfreezes the last n SigLIP vision blocks,
  plus `post_layernorm` and `head`. The text tower stays frozen. It requires
  spatial mode.

### `adaspevla/models/bspline_flow_policy.py`
- `FlowMatchingTransformerBlock` can now do cross-attention
  (`cross_attention=True`): after self-attention, pre-norm multi-head
  cross-attention from the control-point tokens to the memory, added as a
  residual.
- `BSplineFlowPolicy(memory_dim=...)` turns that on in every block and adds
  `memory_proj` (LayerNorm + Linear). `forward_velocity`, `compute_loss` and
  `sample` take an optional `memory`. It is required when `memory_dim` is set.

### `adaspevla/models/adaspe_vla_policy.py` (`AdaSpeVLAPolicy`)
- New arguments: `spatial_tokens`, `proprio_dim`, `proprio_mean`,
  `proprio_std`, `augment`, `vision_finetune_layers`. They are saved in
  `self.config`, and `proprio_mean`/`proprio_std` are also kept as buffers.
- `memory` is passed to `flow_policy.compute_loss` only when it exists, so
  legacy policies keep the old two-argument call.
- New `_encode(images, instructions, proprio)`: normalises the robot state and
  returns the condition plus the memory. Training (`forward_loss`) and
  inference (`predict_action_spline`, `_sample_control_points`) both use it.
  If a policy was trained with robot state, a query without `proprio` raises
  an error instead of silently running without it.
- `set_training_stage` has a third stage, `finetune`. It uses the action loss
  and trains the last `vision_finetune_layers` SigLIP blocks; the `action`
  stage keeps SigLIP frozen. Stage order: action -> finetune -> timing.

### `adaspevla/utils/training.py` (`train_two_stage`)
- New CLI flags (defaults in brackets): `--spatial_tokens` [on],
  `--proprio` [on], `--augment` [on], `--vision_finetune_layers` [4]
  (0 skips the finetune stage), `--finetune_epochs` [5],
  `--vision_lr_scale` [0.1], `--action_steps` / `--finetune_steps` /
  `--timing_steps` (step budgets that override the epoch counts),
  `--warmup_steps` [500], `--num_workers` [8].
- The loop is now step-based. Each stage has its own AdamW with two parameter
  groups: the head at `lr` and the SigLIP blocks at `lr * vision_lr_scale`.
  The schedule is linear warmup (at most 10% of the stage) then cosine.
  Gradients are clipped at 1.0.
- A log line and a checkpoint every epoch, or at least every 2000 steps:
  `adaspevla_<bench>_<task>_<stage>.pt`. `_latest.pt` is written at the end
  of the timing stage. Checkpoints now also store `step`.
- DataLoader: `num_workers`, `pin_memory` on CUDA, and no persistent workers
  (so each stage's dataset flags reach the workers). TF32 is enabled.
- Diagnostic targets are now built for every stage except timing (before:
  only for `action`).

### Evaluators
- `adaspevla/benchmarks/robocasa.py`: if the policy uses robot state, each
  query adds `proprio` built from the observed end-effector pose and the last
  gripper command. Episodes start open (1.0), as in the demos. The gripper
  command is now `1 - last_width`; the same thresholding as before.
- `adaspevla/benchmarks/xpolicylab_model.py` (RoboTwin): the same for both
  arms. `last_widths` is reset per episode, starts at 1, and is updated from
  the commanded `ee_joint_state` of each arm.

### `pilot_train.slurm` (new; submitted as job 1076206 on V100)
- Retrains the pilot tasks with everything on: tasks.tsv rows 0
  (beat_block_hammer), 2 (stamp_seal), 9 (TurnOnSinkFaucet) and
  10 (TurnOffStove). Batch 32.
- RoboTwin: 40k action + 20k finetune steps (~330 epochs at ~5-6k windows).
  RoboCasa: 60k + 30k (~36 epochs at ~80k windows). Then 10k timing steps.
  Finetune trains 4 SigLIP blocks at 0.1x lr.
- Output goes to `checkpoints/pilot-observation-20261014/`, logs to
  `logs/pilot-train-1076206_<row>.out`.
- Where it runs: the H100 submissions (1072763 on t3, then 1073031 on t4,
  sdh@h100) had ~1,200 higher-priority jobs ahead, so both were cancelled
  while pending. It now runs on `yfl@v100`, `qos_gpu-t4`, `-C v100-32g`,
  10 CPUs, 8 workers, 30h limit (~62 jobs ahead at submission).
- `bench_v100.slurm` (dev QoS, job 1075965) measured the pilot setup on a
  V100-32GB:

  | Task | action | finetune | timing | GPU memory | Projected total |
  |---|---|---|---|---|---|
  | beat_block_hammer | 0.52 s | 0.73 s | 0.60 s | 6.7 GB | ~11.5h |
  | TurnOffStove | 0.39 s | 0.53 s | 0.42 s | 5.3 GB | ~12.1h |

  GPU use was often 45-65% during the action stage, so data loading partly
  limits speed.
- `scripts/benchmarks/slurm_env.sh`: `ARCH_MODULE` (default `arch/h100`; empty
  means the default V100 module tree) and `TORCH_CUDA_ARCH_LIST` (default 9.0)
  can now be overridden. Existing scripts are unchanged.

## Verification

- Real SigLIP, CPU, 3 views, 2 arms: image size 224, memory (2, 590, 512).
  The action and timing stages have 0 trainable SigLIP parameters. The
  finetune stage trains blocks 8-11 of 12 (35.4M parameters); the last block
  gets gradients and the first block gets none. The text tower stays frozen.
  Eval-mode encoding is deterministic, and training-mode augmentation changes
  the output.
- Test suite (same command as the baseline): 29 failures against 26 at
  baseline. The 3 new ones:
  - `test_backbone_contract::test_pretrained_dimensions_...`: the backbone read
    `image_processor.size` in legacy mode too. **Fixed** (spatial mode only);
    the file now passes (3/3).
  - `test_review_corrections::test_loss_refits_control_points_...`: the test
    stubs `compute_loss` with two arguments. **Fixed** (memory passed only
    when present).
  - `test_robocasa_benchmark::test_train_cli_and_closed_loop_modes`: the
    subprocess died at `import numpy` with "OpenBLAS pthread_create failed:
    Resource temporarily unavailable". This is login-node thread exhaustion
    while several of my runs overlapped, not a code error. The same
    convert + train commands run to completion when started directly.
- The other 26 failures were already there before these changes (the suite
  is out of date with the code).
- Full re-run after the two fixes, with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=8`:
  **26 failed, 70 passed**, the same 26 failures as the baseline. All 3 new
  failures pass, including the CLI training test (73 s). With threads capped
  the suite takes about 3 min instead of about 50 min on the login node.
