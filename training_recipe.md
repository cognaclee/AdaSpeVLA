# Training recipe

The recipe behind the current benchmark suite (`benchmark-suite-20261018-pi05`,
`suite_train.slurm`), what each choice is based on, its known problems, and the
pilots that test changes to it. Last updated 2026-10-19.

The command-line defaults of the training scripts are **not** this recipe: the
recipe turns off the vision finetune stage, turns on padded episode ends and sets
step budgets. Use the command below, not the defaults.

## Command

One task, from the repository root:

```bash
python -m scripts.benchmarks.train_converted \
    --data_dir data/converted/robotwin --task_name beat_block_hammer \
    --vlm_name pi05 --vlm_mode joint \
    --batch_size 32 --device cuda --num_workers 6 \
    --action_steps 40000 --timing_steps 10000 \
    --vision_finetune_layers 0 --no-augment --pad_episode_end --resume \
    --output_dir checkpoints/<run>/robotwin/beat_block_hammer
```

For RoboCasa use `--data_dir data/converted/robocasa` and `--action_steps 60000`.
On Jean Zay, `sbatch suite_train.slurm` trains every row of
`checkpoints/benchmark-suite-20261018-pi05/tasks.tsv` (one array job per task, one
H100 each). Evaluation needs only `*_latest.pt`, written when the timing stage ends.

## Settings

### Data

| Setting | RoboTwin 2.0 | RoboCasa365 |
|---|---|---|
| Demos | `demo_clean`, aloha_agilex, 50 per task | target split, about 450–520 per task |
| Arms, action channels | 2 arms, 14 | 1 arm, 7 |
| Cameras | head, left wrist, right wrist; 320×240 | agent view (left), eye in hand; 256×256 |
| Sample period | 0.06 s | 0.05 s |
| Training windows | one per sample, 16 samples = 0.9 s (bbh 5.6k windows) | 16 samples = 0.75 s (CloseFridge 135k) |
| Episode ends | padded (`--pad_episode_end`): the last 15 samples are anchors too, holding the final pose | same |

Conversion from the benchmark formats: [docs/benchmarks.md](docs/benchmarks.md).

### Model

| Part | Setting |
|---|---|
| Encoder | pi0.5 (`lerobot/pi05_base`): SigLIP-So400m/14 at 224 px (256 tokens per camera) and Gemma-2B, all 18 layers. **Frozen**, Gemma in bfloat16 |
| Encoder mode | `joint`: image tokens of every camera and the prompt `Task: <instruction>;\nAction: ` go through Gemma together; the model reads Gemma's final-norm output at the image positions plus the mean over the instruction tokens |
| Image preprocessing | centre square of 90% of the short side (216×216 of a 320×240 frame), resized to 224; no augmentation. See [known problems](#known-problems) |
| Robot state | on: per arm base-frame position, 6-D rotation and last gripper command, as one extra token and added to the global condition |
| Action representation | per arm six base-frame offsets from the pose at query time plus the gripper width; one cubic B-spline with 8 control points over the window; learned knots shared by both arms |
| Action expert | flow-matching transformer, 6 blocks, hidden 384, 6 heads, AdaLN conditioning; every block cross-attends to all image tokens (projected to 512, plus a per-camera embedding), the instruction token and the robot-state token; 15 Euler steps at inference. About 28M trainable parameters |
| Timing head | MLP (128 hidden) predicting the next query interval, clamped to [0.02, 0.5] s |

### Optimization

| Stage | Steps | What trains | Loss |
|---|---|---|---|
| `action` | 40k RoboTwin, 60k RoboCasa | action expert, knot head, projections | flow-matching MSE + 0.5 × knot span KL |
| `finetune` | skipped (`--vision_finetune_layers 0`) | – | – |
| `timing` | 10k | timing head only | log-MSE on query intervals |

Batch 32, AdamW (lr 2e-4, weight decay 1e-4), 500 warmup steps then cosine
decay, gradient clipping at 1.0, each stage with its own optimizer. Runs are not
seeded.

On one H100: RoboTwin 0.41 s/step (about 5.7 h for 50k steps), RoboCasa
0.28 s/step (about 5.5 h for 70k). Final action loss: beat_block_hammer 0.0071,
press_stapler 0.0138, TurnOnSinkFaucet 0.075, CloseFridge 0.079.

### Evaluation protocol

| | RoboTwin | RoboCasa |
|---|---|---|
| Episodes | 100 per task: 4 shards of 25, shard k = RoboTwin seed k (episode seeds from 100000·(1+k)), seen instructions, 400-step limit | 50 per task, seeds 0–49, target split |
| Query modes reported | adaptive, fixed 0.24 s and 0.48 s (4 and 8 control steps) | adaptive, fixed 0.2 s and 0.4 s |
| Slurm files | `suite_eval_robotwin.slurm` | `suite_eval_robocasa.slurm` |

The suite also ran fixed 0.06 s and 0.5 s (RoboTwin) and 0.05 s and 0.5 s
(RoboCasa); they are not part of the reported comparison. Because every
evaluation uses the same seeds, two checkpoints can be compared episode by
episode (count the episodes only one of them solves).

## Why these choices

| Choice | Evidence |
|---|---|
| Padded episode ends | Without padding the last 15 samples of every demo, where the task is completed, were never training anchors. beat_block_hammer adaptive 3% → 23% on the same 100 episodes (21 vs 2 episodes solved by only one model), 2026-10-17 |
| pi0.5 encoder | Backbone comparison `backbone-cmp-20261016-pad` (frozen encoders, padded ends, adaptive, beat_block_hammer, press_stapler, CloseFridge, SlideDishwasherRack): mean success pi0.5 34%, GR00T 29%, SigLIP-B 27%, pi0 21% |
| Joint mode | Separate mode (raw SigLIP tokens) did not help pi0.5 (dev beat_block_hammer 27% vs 32%, press_stapler 22% vs 22%; not significant). It fixed pi0, whose last-layer image outputs are untrained |
| Frozen encoder | Fine-tuning the last 4 SigLIP blocks did not beat the frozen encoder (SigLIP-B, beat_block_hammer 23% fine-tuned vs 31% frozen; single runs) |
| Robot state on | Mixed: without it SigLIP scored 27% vs 23% on beat_block_hammer (not significant). Being re-tested with pi0.5 in the recipe pilot |
| No augmentation | All current comparisons use it; augmentation has not been re-tested since padded ends |
| 40k / 60k action steps | Equal budgets sized for the backbone comparison. RoboCasa action loss was still falling at 60k; twice the steps is being tested in the recipe pilot |

## Known problems

1. **The centre crop hides objects.** A 216×216 square of a 320×240 RoboTwin
   frame discards 52 px on each side, a third of the width, in training and
   evaluation, for every camera. On the first frame of the 50 demos, a block of
   stack_blocks_three is cut off in nearly every demo, the seal or the mat of
   stamp_seal in about 29, the block of beat_block_hammer in about 17. RoboCasa
   frames are square, so only 5% per side is lost there.
   Fix under test: `--image_fit pad` (letterbox the whole frame, as pi0 does).
2. **Demos contain long motionless stretches.** RoboTwin's scripted expert stops
   between motions: 25–28% of the rows of most tasks do not move, in pauses of
   about 9 samples; press_stapler is 52% idle with a 31-sample stretch in every
   demo, open_microwave 69%. With an unchanged image the policy cannot tell how
   long it has waited and can stall. Fix under test: `--max_idle_rows 4`.
3. **Gemma's last layer loses where the target is.** Linear probes on held-out
   demos decode the final end-effector position with R² 0.60 at layers 3–6 but
   0.49 at the last layer (beat_block_hammer; press_stapler 0.64 vs 0.42).
   Fix under test: `--vlm_layers 6`.
4. **Single runs are noisy.** Two trainings of this exact recipe scored 15% and 3%
   on press_stapler (same episodes, same final loss). Compare recipes with at
   least two runs per variant.

## Results

RoboTwin `demo_clean`, success over 100 episodes (fewer where shards were still
running on 2026-10-19):

| Task | Adaptive | Fixed 0.24 s | Fixed 0.48 s |
|---|---|---|---|
| beat_block_hammer | 31% | 28% | 39% |
| open_microwave | 32% (30/94) | 33% (28/86) | 35% (30/86) |
| handover_block | 23% | 5% | 22% |
| hanging_mug | 5% | 7% | 6% |
| stack_blocks_three | 4% (4/95) | 0% (0/88) | 2% (2/88) |
| place_phone_stand | 4% | 4% | 7% |
| press_stapler | 3% | 4% | 4% |
| stamp_seal | 2% | 0% | 2% |

stamp_seal is at the floor for every published method too (RoboTwin 2.0 baselines
with 50 clean demos: 1–18%, most about 2%): success needs the seal within 1 cm of
the mat centre. RoboCasa results are pending.

## Pilots in flight

| Pilot | Change | Tasks | Jobs | Baseline |
|---|---|---|---|---|
| `recipe-pilot-20261018` | `pi05-noprop`: `--no-proprio`; `pi05-long`: twice the action steps | beat_block_hammer, press_stapler, CloseFridge, SlideDishwasherRack | train 1268724 (done); eval 1268818 (RoboTwin), 1268820 (RoboCasa) | `backbone-cmp-20261016-pad/pi05` |
| `crop-pilot-20261019` | `pad`: `--image_fit pad`; `pad-trim`: pad + `--max_idle_rows 4`; `pad-mid6`: pad + `--vlm_layers 6`; two runs each | beat_block_hammer, stamp_seal, stack_blocks_three, press_stapler | train 1283709 (`crop_pilot_train.slurm`); eval 1283712 (`crop_pilot_eval_robotwin.slurm`) | this suite, same episodes |

If a variant wins, the suite is retrained with it and this file is updated.
