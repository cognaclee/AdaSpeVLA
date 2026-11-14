# AdaSpeVLA expert variant (`VARIANT=expert`)

Status as of 2026-11-14 10:00: **training finished** (Action 40k + 3 timing heads per task, ALL DONE 09:54 / 09:55).
Final checkpoints (params only) are on HF: `cognaclee/Frank-Insert-Wipe`, folders `<task>_expert/`.

## Why

On the real robot, the baseline pi0.5 (`dddddsds31341/Franka-*-pi05-step-20000`, action_horizon=10, 20k steps)
was clearly more precise than AdaSpeVLA (the Nov 12 `flow_head` run). Offline checks ruled out the B-spline as the
cause: over a 9-step prefix, the fitting error at p95 is ≤ 0.05 of one typical step. The main difference was the
action generator:

- AdaSpeVLA `flow_head`: a 20.4 M flow transformer **trained from scratch**. `PluginWeightLoader` discarded
  pi0.5's pretrained action expert.
- pi0.5: a 427.9 M pretrained action expert (311.4 M Gemma-300M transformer + 116.5 M adaRMS).

Option A: put the pretrained pi0.5 expert back and have it predict the 11 control points.

## What changed

Only the part that **produces the control points** changed.

| | `flow_head` (Nov 12 run) | `pi05_expert` (this run) |
|---|---|---|
| Generator | AdaSpeVLA flow transformer, 20.4 M, random init | pi0.5 Gemma-300M action expert + adaRMS, 427.9 M, loaded from `pi05_base` |
| Observation access | cross-attention to projected last-layer prefix tokens (`token_proj`, `view_embedding`, `text_token_proj`) | attends to the PaliGemma prefix KV cache at every layer, as in pi0.5 (gradients flow through the cache) |
| Tokens | K=11 control-point tokens + global context | K=11 control-point tokens via pi0.5's `action_in_proj`; time via pi0.5's `time_mlp_in/out` (adaRMS); output via `action_out_proj` |
| Knot information | knot vector + log horizon concatenated into the context | **new** `cp_knot_proj` (zero-init): sin-cos of each control point's Greville abscissa on the predicted knots, plus the knot vector and log horizon, added to each token. Zero init means step 0 receives exactly pi0.5's inputs |
| Flow convention | x_t = t·data + (1−t)·noise, u = data − noise, t ~ U(0,1) | pi0.5: x_t = t·noise + (1−t)·data, u = noise − data, t ~ Beta(1.5,1)·0.999+0.001 |
| ODE steps (inference / timing stage) | 15 / 15 | 10 / 10 (pi0.5 default) |
| Weight loading | PaliGemma only | PaliGemma + action expert (`llm/*_1`) + `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`. **Fails** if any pretrained param is missing or has the wrong shape |

**Unchanged:** data, norm stats (`assets/baseline`), H=16, K=11, cubic B-spline, query interval 0.2–1.5 s,
timing and knot labels (gripper *command* events, `adaspe/labels.py`), knot head, KL knot loss, timing head,
spline fitting, `sample_spline` execute_steps, optimizer/LR/batch (`_COMMON`), 40k Action steps, and
timing heads trained from the 20k/30k/40k checkpoints.

### Code

- `adaspe/model.py`: `AdaSpePi05Config.generator: "flow_head" | "pi05_expert"`. Expert path in `_cp_token_bias`,
  `_expert_velocity`, and the expert branches of `sample_control_points` and `compute_loss`.
- `configs.py`: `GENERATORS`, `_EXPERT_KNOBS` (10 ODE steps), configs `<TASK>_adaspe_expert_action` and
  `<TASK>_adaspe_expert_timing_from{20,30,40}k`. Strict loader check in `PluginWeightLoader`.
- `names.sh`: `VARIANT=expert` sets `TAG=<TASK>_adaspe_expert` (configs, checkpoints, logs) and
  `JOB_BASE=adaspe-expert` (Slurm job names).
- Tests: `tests/test_expert_parity.py` (the expert pass matches OpenPI's own Pi0 suffix code),
  `tests/check_config_names.py`, and `tests/smoke_model.py` for both generators. The `flow_head` smoke losses are
  identical to job 34962, so the old path is unchanged.
- Not committed to git yet.

## Run

| | insert_tube | wipe_whiteboard |
|---|---|---|
| Account / QoS | yfl@h100, dev chain (2 h links, 4 H100, fsdp=4) | same |
| First link | 130181, started 2026-11-14 02:02, preflight passed 02:07 | 130182, started 02:10, preflight passed 02:15 |
| Links | 130181 → 170553 → 172225 → 176486 (ALL DONE 09:54) | 130182 → 170656 → 172371 → 176654 (ALL DONE 09:55) |
| Action done | 2026-11-14 08:36 | 2026-11-14 08:36 |
| Timing heads done | 09:54 (2.2 it/s, 1 GPU each) | 09:55 |
| Speed | 2.1 it/s (flow_head: 2.2 it/s) | 2.1 it/s |
| Checkpoints | `checkpoints/adaspe/insert_tube_adaspe_expert_action/formal/` 10000, 20000, 30000, 39999; `insert_tube_adaspe_expert_timing_from{20,30,40}k/formal/9999` | same under `wipe_whiteboard_adaspe_expert_*` |
| Logs | `test/franka/insert_tube_adaspe_expert/` | `test/franka/wipe_whiteboard_adaspe_expert/` |

The backup t3 chains 135256 / 135257 were cancelled automatically when the dev chains finished.

## Results: training loss (Action stage)

Mean training loss over each 2k-step window, from `test/franka/<TASK>_<variant>/action.log`.

| Steps | insert_tube flow_head | insert_tube **expert** | wipe flow_head | wipe **expert** |
|---|---|---|---|---|
| 0–2k | 0.624 | **0.334** | 0.597 | **0.295** |
| 2–4k | 0.371 | **0.180** | 0.358 | **0.170** |
| 4–6k | 0.300 | **0.152** | 0.296 | **0.146** |
| 6–8k | 0.266 | **0.134** | 0.260 | **0.131** |
| 8–10k | 0.238 | **0.121** | 0.238 | **0.121** |
| 10–12k | 0.218 | **0.105** | 0.218 | **0.114** |
| 12–14k | 0.202 | **0.098** | 0.207 | **0.105** |
| 14–16k | 0.186 | **0.087** | 0.192 | **0.098** |
| 16–18k | 0.173 | **0.076** | 0.182 | **0.087** |
| 18–20k | 0.156 | **0.067** | 0.169 | **0.083** |
| 20–22k | 0.144 | **0.058** | 0.160 | **0.073** |
| 22–24k | 0.132 | **0.054** | 0.151 | **0.066** |
| 24–26k | 0.120 | **0.047** | 0.140 | **0.058** |
| 26–28k | 0.111 | **0.044** | 0.135 | **0.053** |
| 28–30k | 0.105 | **0.038** | 0.130 | **0.049** |
| 30–32k | 0.100 | **0.040** | 0.127 | **0.046** |
| 32–34k | 0.095 | **0.032** | 0.125 | **0.045** |
| 34–36k | 0.092 | **0.030** | 0.123 | **0.043** |
| 36–38k | 0.089 | **0.031** | 0.120 | **0.042** |
| 38–40k | 0.086 | **0.030** | 0.118 | **0.040** |

Summary:
- At matched steps, the expert's loss is about **0.4×** the flow head's for both tasks (28–30k: 0.038 vs 0.105 for
  insert_tube, 0.049 vs 0.130 for wipe_whiteboard).
- The expert passed the flow head's **final** (38–40k) loss early: by 16–18k for insert_tube (0.076 vs 0.086) and by 10–12k for wipe_whiteboard (0.114 vs 0.118).
- Final (38–40k): insert_tube 0.030 vs 0.086 (0.35×), wipe_whiteboard 0.040 vs 0.118 (0.34×). insert_tube flattens
  after ~32k; wipe_whiteboard is still falling slowly.

Caveats:
1. **Training loss only.** There is no held-out eval or robot test yet. The expert has ~21× more trainable
   generator parameters on the same small dataset, so part of the gap could be memorization.
2. **Different flow conventions.** Both losses are MSE of the velocity (noise − data, up to sign) in the same normalized
   control-point space, but the time sampling differs: Beta(1.5,1) puts more weight near the noise end, which is the harder
   part. If anything, this makes the expert's loss look higher, not lower.
3. The loss includes the knot KL term (same coefficient and labels in both variants).

## Results: timing stage

Timing loss = MSE of log(query interval) against the same labels; mean over steps 9k–10k.

| Timing head trained from | insert_tube flow_head | insert_tube expert | wipe flow_head | wipe expert |
|---|---|---|---|---|
| Action 20k | 0.210 | 0.262 | 0.203 | 0.280 |
| Action 30k | 0.207 | 0.255 | 0.197 | 0.273 |
| Action 40k | **0.206** | **0.255** | **0.196** | **0.271** |

**The expert's timing head is worse** (+24 % on insert_tube, +38 % on wipe_whiteboard), even though its action loss is
much lower. The starting checkpoint barely matters for either variant (≤ 0.01).

Likely cause (from the code; not tested): the timing head reads the pooled condition `cond` (from `fusion_proj`,
`text_proj`, `cross_gate`, `final_proj`). In `flow_head`, `cond` also feeds the flow transformer, so the flow loss
trains it. In `pi05_expert`, the expert reads the KV cache instead, so `cond` is trained only by the knot KL term and
carries less information. Possible fixes if timing matters on the robot: feed the timing head expert features
(e.g. pooled suffix outputs), or train `cond` with the timing stage unfrozen.

## Not measured yet

- **Inference latency.** The expert runs a 300M transformer for 10 ODE steps, against 20M for 15 steps, so each query
  costs more. This matters because the timing head can ask for a new query every 0.2 s.
- Real-robot success and precision against pi0.5 and against `flow_head`.
- Open-loop length confound: AdaSpeVLA executes up to 1.5 s per query, while the baseline replans on its own schedule.
