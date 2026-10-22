# Training recipe

The recipe behind the current benchmark suite (`benchmark-suite-20261018-pi05`,
`suite_train.slurm`), what each choice is based on, its known problems, and the
pilots that test changes to it. Last updated 2026-10-21.

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
| Robot state on | Questioned. Without it SigLIP scored 27% vs 23% on beat_block_hammer (not significant). The pi0.5 recipe pilot without it (single runs, dev episodes): press_stapler 38% vs 22% (29 vs 13 episodes solved by only one, sign test p = 0.02), beat_block_hammer 36% vs 32% (18 vs 11, not significant); RoboCasa pending |
| No augmentation | All current comparisons use it; augmentation has not been re-tested since padded ends |
| 40k / 60k action steps | Equal budgets sized for the backbone comparison. Twice the steps did not help on RoboTwin (press_stapler 14% vs 22%, beat_block_hammer 25% vs 32%, neither significant); RoboCasa, whose action loss was still falling at 60k, pending |

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
   Fix under test: `--vlm_layers 6`; it wins clearly in the crop-fix pilot
   ([results](#crop-fix-pilot-2026-10-20)).
4. **Single runs are noisy.** Two trainings of this exact recipe scored 15% and 3%
   on press_stapler (same episodes, same final loss). Compare recipes with at
   least two runs per variant.

## Results

RoboTwin `demo_clean`, success over 100 episodes (final, 2026-10-19):

| Task | Adaptive | Fixed 0.24 s | Fixed 0.48 s |
|---|---|---|---|
| beat_block_hammer | 31% | 28% | 39% |
| open_microwave | 32% | 31% | 32% |
| handover_block | 23% | 5% | 22% |
| hanging_mug | 5% | 7% | 6% |
| stack_blocks_three | 4% | 0% | 2% |
| place_phone_stand | 4% | 4% | 7% |
| press_stapler | 3% | 4% | 4% |
| stamp_seal | 2% | 0% | 2% |
| **Mean** | **13.0%** | **9.9%** | **14.3%** |

Paired over the episodes both modes ran: adaptive solves 70 that fixed 0.24 s
misses and misses 46 it solves (p = 0.03); against 0.48 s, 58 vs 72 (p = 0.25).

stamp_seal is at the floor for every published method too (RoboTwin 2.0 baselines
with 50 clean demos: 1–18%, most about 2%): success needs the seal within 1 cm of
the mat centre.

RoboCasa target split, success over 50 episodes (seeds 0–49, final 2026-10-20):

| Task | Adaptive | Fixed 0.2 s | Fixed 0.4 s |
|---|---|---|---|
| PickPlaceToasterToCounter | 56% | 38% | 68% |
| CloseFridge | 52% | 42% | 58% |
| CoffeeSetupMug | 44% | 36% | 48% |
| PickPlaceCounterToCabinet | 40% | 28% | 46% |
| TurnOnMicrowave | 26% | 20% | 20% |
| SlideDishwasherRack | 24% | 8% | 28% |
| TurnOffStove | 14% | 14% | 18% |
| TurnOnSinkFaucet | 0% | 0% | 8% |
| **Mean** | **32.0%** | **23.2%** | **36.8%** |

OpenDrawer adaptive 52%; its fixed-interval evaluations crashed and were not rerun.
Paired over episodes: adaptive vs 0.2 s 65 vs 30 (p < 0.001); vs 0.4 s 42 vs 61
(p = 0.08). Adaptive picks a median 0.34 s here (0.42 s on RoboTwin), and short
re-queries go with failure: queries of 0.2 s or less are 2% of the median
successful episode and 21% of the median failed one; 52 of 296 failures re-query
every 0.05 s for more than 30% of their queries. 76–80% of the timing labels sit at
the 0.5 s cap, and the log-MSE loss pulls predictions toward their geometric mean
(about 0.4 s). Tested by `timing-pilot-20261021`.

### Crop-fix pilot, 2026-10-20

`crop-pilot-20261019`, RoboTwin `demo_clean`, adaptive only, 100 episodes per
run; run 0 / run 1, paired on common episodes with the crop baseline above.

| Task | Crop baseline | `pad` | `pad-trim` | `pad-mid6` |
|---|---|---|---|---|
| beat_block_hammer | 31% | 24 / 13% | 26 / 24% | **57 / 63%** |
| stamp_seal | 2% | 4 / 1% | 3 / 1% | **22 / 21%** |
| press_stapler | 3% | 6 / 4% | **16 / 12%** | **21 / 18%** |
| stack_blocks_three | 4% | 0 / 0% | cancelled | pending (1332334) |

- `pad-mid6` beats the baseline on all three tasks in both runs (p ≤ 0.001;
  beat_block_hammer +29/−5 and +37/−5 episodes solved by only one). `pad` alone is
  flat or worse (beat_block_hammer run 1 p = 0.004), so the gain comes from the
  layers despite the lower resolution of the letterboxed image.
- `pad-mid6` is also cheaper: about 58 vs 76 ms per query and 3 h 25 vs 5 h 50 of
  training.
- `pad-trim` helps press_stapler only (p = 0.002 / 0.012), the task trimming
  changes most (40% of its rows).
- `pad-noprop` (train 1305479 done) and the fixed-interval modes for `pad-mid6`
  are not evaluated yet.

## Pilots in flight

| Pilot | Change | Tasks | Jobs | Baseline |
|---|---|---|---|---|
| `recipe-pilot-20261018` | `pi05-noprop`: `--no-proprio`; `pi05-long`: twice the action steps | beat_block_hammer, press_stapler, CloseFridge, SlideDishwasherRack | train 1268724 (done); eval 1268818 (RoboTwin, done), 1268820 (RoboCasa) | `backbone-cmp-20261016-pad/pi05` |
| `crop-pilot-20261019` | `pad`: `--image_fit pad`; `pad-trim`: pad + `--max_idle_rows 4`; `pad-mid6`: pad + `--vlm_layers 6`; `pad-noprop`: pad + `--no-proprio`; two runs each | beat_block_hammer, stamp_seal, stack_blocks_three, press_stapler | train 1283709 (first three, done), 1305479 (`pad-noprop`) (`crop_pilot_train.slurm`); eval 1283712, 1305485 (`crop_pilot_eval_robotwin.slurm`); stack_blocks_three evals of `pad-trim` and `pad-mid6` cancelled 2026-10-20 (pad scored 0/106 there; trim changes 8% of its data); those of `pad-mid6` resubmitted the same day as 1332334 after it won elsewhere | this suite, same episodes |
| `timing-pilot-20261021` | Timing head only, retrained 10k steps on the suite's RoboCasa action checkpoints: `floor`: `--min_query_interval 0.2`; `floor-median`: floor + `--event_quantile 0.99` + `--timing_loss log_l1` | all 9 RoboCasa tasks | train 1371517 (`timing_pilot_train.slurm`, done); eval 1376029 (`timing_pilot_eval_robocasa.slurm`, one dev-QoS job running all 18 at once; 8 aborted mid-episode and are rerun as 1396853 (A100 dev, inside a job step)): both variants, adaptive, seeds 0–49, paired with the suite's adaptive / 0.2 s / 0.4 s; the winner can be confirmed on seeds 50–99 | this suite, same episodes |
| `robocasa-mid6-20261021` | `--image_fit pad --vlm_layers 6` (the RoboTwin crop-pilot winner), timing unchanged | all 9 RoboCasa tasks | train 1371518 (`robocasa_mid6_train.slurm`); eval 1387680 (H100, 7 done) + 1396854 (A100 dev, the other 20) (`robocasa_mid6_eval.slurm`: adaptive, 0.2 s, 0.4 s, seeds 0–49) | this suite, same episodes |
| `robocasa-win24-20261021` | `win24`: pad-mid6 + `--chunk_horizon 24` (1.15 s windows) `--num_control_points 11` `--min_query_interval 0.2 --max_query_interval 0.8`, full retrain; `mid6-floor`: timing head only on the pad-mid6 action checkpoints with `--min_query_interval 0.2` | all 9 RoboCasa tasks | 1394624 (`robocasa_win24.slurm`, trains then evaluates in the same job: win24 adaptive / 0.2 s / 0.4 s, mid6-floor adaptive, seeds 0–49); crashed evals finished by 1405267 (`robocasa_win24_eval.slurm`, writes `eval_metrics/robocasa-win24-20261021/summary.txt`). **Final (2026-10-22 01:00): win24 adaptive 54.9% over 9 tasks vs pad-mid6 adaptive 44.0% (+87/−38, p<0.001), vs its own fixed 0.4 s 49.8% (+56/−33, p 0.019), vs 0.2 s 33.1%; mid6-floor 44.9% = pad-mid6.** The longer window with the higher cap is the winner, and adaptive now beats fixed 0.4 s | pad-mid6, same episodes |
| `robocasa-policy-20261021` | Policy strength on top of pad-mid6, one change each: `expert-l` (`--flow_hidden_dim 768 --flow_num_layers 12 --flow_num_heads 12`), `aug` (`--augment`: random shifts ≤5% + colour jitter, new for pad fit), `vision-ft` (`--vision_finetune_layers 4 --finetune_steps 20000`), `encoder-ft` (vision-ft + `--vlm_finetune_layers 6`: Gemma layers trained in float32, new), `expert-xl` (`--flow_hidden_dim 1024 --flow_num_layers 16 --flow_num_heads 16`, ~390M) | TurnOffStove, PickPlaceCounterToCabinet, PickPlaceToasterToCounter, SlideDishwasherRack | 1405536 (0-7), 1405807 (8-15), 1407912 (16-19) (`robocasa_policy_pilot.slurm`, trains then evaluates adaptive and fixed 0.4 s, seeds 0–49). `aug`: adaptive 41.0% vs pad-mid6 35.5% on the 4 tasks (+37/−26, p 0.21), fixed 0.4 s 42.5 vs 41.0 (tie). `expert-l` (3 tasks; SlideDishwasherRack evals aborted in read_pixels on every attempt, rerun on A100 by 1412845 (1412490 ran nothing: log-prefix sed bug) via the now generic catch-up `robocasa_win24_eval.slurm`): over all 4 tasks adaptive 41.5 vs 35.5 (+34/−22, p 0.14), fixed 0.4 s 39.0 vs 41.0 (tie). `vision-ft`: adaptive 46.5% vs 35.5% (+40/−18, p 0.005), fixed 0.4 s 44.0 vs 41.0 (p 0.53), the first significant policy-side gain. Follow-ups (baseline win24): `win24-la` = win24 + expert-l + aug, 1412499 (elements 20-23); `win24-vft` = win24 + vision-ft starting from the win24 action checkpoints (only the finetune and timing stages are trained), elements 24-32 = all 9 tasks (row = index − 16), pilot tasks submitted as 1413825 (26,30-32), the other 5 as 1414138 (24,25,27-29, `--exclude=jzxh043,jzxh045`). **All 4 tasks (06:00 on 2026-10-22, `eval_metrics/robocasa-policy-20261021/summary.txt`), adaptive / fixed 0.4 s vs pad-mid6 35.5 / 41.0%: `encoder-ft` 49.0 (p 0.001) / 51.0 (p 0.012); `vision-ft` 46.5 (p 0.005) / 44.0; `expert-l` 41.5 / 39.0; `expert-xl` 40.5 / 40.5; `aug` 41.0 / 42.5.** Fine-tuning the encoder is the policy lever. A larger expert gives about +5 at 165M and nothing more at 390M. encoder-ft vs vision-ft is +30/−25 adaptive (p 0.59), and encoder-ft is the only change significant in both modes. **`win24-vft` final (9 tasks): adaptive 55.8% vs win24 54.9% (+48/−44, p 0.76); fixed 0.4 s 51.1 vs 49.8; vs its own 0.4 s 55.8 vs 51.1 (+50/−29, p 0.024). `win24-la` (4 tasks): 48.5% vs win24 54.5% (+20/−32, p 0.13).** The mid6 policy gains do not stack on win24. Follow-up `win32` (32-sample windows = 1.55 s, 14 control points, intervals [0.2, 1.1] s, full retrain, indices 42-45 = pilot tasks): 1418819. `win24-gr00t` (user request 2026-10-22): the win24 flags with `--vlm_name gr00t` (GR00T N1.5 encoder, first 6 Qwen3 layers, pad fit; checked on GPU), indices 46-49 = pilot tasks: 1428327; baseline win24 (pi0.5). `win24-eft` (4 tasks, 09:30): adaptive 54.0% vs win24 54.5% (+20/−21), so encoder fine-tuning doesn't stack on win24 either. Follow-up `win24-eft` (win24 + encoder-ft from the win24 action checkpoints, indices 33-41 = 9 tasks): pilot tasks 1414853 (35,39-41). RoboCasa evaluations on jzxh043/jzxh045 (and jzxh059, jzxh025) abort in `read_pixels` on every attempt; expert-xl TurnOffStove and PickPlaceCounterToCabinet are rerun by 1414136 too, win24-vft TurnOffStove and SlideDishwasherRack by 1417000 | pad-mid6, same episodes |
| `robotwin-d110-20261021` | pad-mid6 (crop-pilot variant 2) on `data/converted/robotwin-d110`: the 50 official demo_clean demos + 60 DDA-VLA demos on seeds 1000+ (`convert_dda_robotwin.slurm`, job 1407605; poses by SAPIEN forward kinematics, verified on official demos: median 0 mm, max 3.6 mm). Variants `d110`, `d110-aug` (`--augment`; DDA renders are darker and clip the wrist cameras' near gripper fingers), `d110-xl` (aug + expert 768/12/12) | beat_block_hammer, stamp_seal, stack_blocks_three, press_stapler | 1407683 (`robotwin_d110.slurm`, trains then evaluates adaptive, 4 shards × 25 on demo_clean); element 7 (d110-aug press_stapler) diverged to NaN knots at step ~12.5k and was resumed as 1411361 after `train_two_stage` started skipping batches with a non-finite gradient norm. Packing 4 shards per 1-GPU job makes each ~4× slower, so stack_blocks_three will stop at the 14 h limit with ~10 of 25 episodes per shard. Interim 06:30 (paired by env seed): `d110` beats pad-mid6 with 50 demos, +42/−13 vs run0 and +44/−17 vs run1 (bbh 62/73 vs 47/73, press_stapler 18/36 vs 7/36, stamp_seal 11/32 vs 9/32); `d110-aug` and `d110-xl` are no better than `d110`. Follow-up: `d110` on rows 1, 3, 5, 6 (hanging_mug, place_phone_stand, handover_block, open_microwave; suite baseline 5/4/23/32%), training 1415215 (indices 12-15, which evaluate shard 0), single-shard evals 1415216 (afterok). `d110-win24` (user request 2026-10-22): d110 with RoboCasa's win24 window: 24-sample windows (1.38 s at 0.06 s), 11 control points, intervals [0.2, 0.8] s. Rows 0, 2, 4, 7; training 1421264 (indices 32-35, each evaluates shard 0), single-shard evals 1421266 (afterok); baseline d110, same episodes | crop-pilot pad-mid6 run0/run1 (50 demos), same episodes |

If a variant wins, the suite is retrained with it and this file is updated.
