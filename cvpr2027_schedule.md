# AdaSpeVLA — CVPR 2027 Plan

Deadline **Nov 16, 2026** (register by Nov 10, supplementary Nov 23). Today: Oct 22.

Four goals, in order:

1. Get good results on the existing simulation benchmarks
2. Design real-robot tasks
3. Reuse existing demos, finetune the policy, test on the real robot
4. Write the paper

---

## 1. Simulation (Oct 15 – Nov 8)

The framework and pipeline are largely fixed. Focus on simulation tuning while preparing real-task scenes and data.

**Target:** higher success than the baselines on the existing benchmarks, and adaptive timing at least as good as the best fixed interval with fewer policy calls.

### 1.1 Fix encoder and feature settings (Oct 16 – Oct 22)

Keep the current framework and use the first six feature layers instead of the last layers, following the performance gains in the current tests. A larger action expert did not help, so keep its current size. Use the 4 dev tasks (beat_block_hammer, press_stapler, CloseFridge, SlideDishwasherRack) for tuning before full evaluation.

The original encoder candidates below are retained for reference; a full backbone sweep is no longer the priority.

| Candidate   | Vision                                  | Text      | Used by                      | Code work           |
| ----------- | --------------------------------------- | --------- | ---------------------------- | ------------------- |
| A (current) | SigLIP-B/16 224                         | SigLIP    | —                            | none                |
| B           | SigLIP-B/16 256                         | SigLIP    | —                            | none (`--vlm_name`) |
| C           | SigLIP-SO400M/14                        | SigLIP    | vision tower of π0           | adapter             |
| D           | PaliGemma-3B (SigLIP-SO400M + Gemma-2B) | joint VLM | π0, π0.5                     | VLM adapter         |
| E           | Eagle 2.5 VLM                           | joint VLM | GR00T N1.5 (VLM kept frozen) | VLM adapter         |
| F           | SmolVLM (SmolVLA backbone)              | joint VLM | SmolVLA                      | VLM adapter         |

Reference notes (optional experiments only):

- D and E can start from the VLM weights inside the robot-pretrained π0 / GR00T checkpoints instead of the plain web-pretrained ones; test both if time allows.
- D and E are 2–3B parameters: keep the VLM frozen (as GR00T does) or train on H100.
- Cached locally (`~/.cache/huggingface/hub`): SigLIP, SmolVLA, π0 (`lerobot/pi0_base`, includes the PaliGemma weights; tokenizer in `google/paligemma-3b-pt-224`) and GR00T N1.5 (`nvidia/GR00T-N1.5-3B`, includes the Eagle 2.5 vision model and the first LLM layers it uses).

Done when: the encoder and feature settings are fixed for the remaining runs, with a small comparison table using available results.

### 1.2 Use the existing benchmarks and split ownership

Keep **RoboTwin 2.0**, **RoboCasa**, and the third benchmark already in use (recorded as **Robot-EVIL** in the meeting notes; confirm its name). Do not add a new benchmark.

| Owner                  | Scope                                                        |
| ---------------------- | ------------------------------------------------------------ |
| 李泽增                 | Tune and test one benchmark with fewer tasks (~20 tasks; RoboCasa or the third benchmark, to be confirmed) |
| Haichao LIU and 刘佳乐 | Full evaluation and unit tests on the remaining two benchmarks; agree on the split before launching runs |

**Before shared runs:** 李泽增 pushes the latest code; all members tune and evaluate that version. Haichao LIU updates and pushes this to-do list.

### 1.3 Get good results, better than the baselines (Oct 19 – Nov 8)

Baselines to beat:

- published methods on the same tasks, from the benchmark papers (e.g. Diffusion Policy, ACT, π0);
- our own model with fixed intervals;
- our first run (`benchmark-suite-20261009`).

Evaluation and ablation settings:

- Use consistent sample counts across benchmarks for each compared setting. The baseline is 16; 24 is still being tested.
- Query interval range is determined by the action window, not an independent parameter. 李泽增 removes it from the recipe.
- Keep the current training demo counts for the main benchmark comparisons (50 per task where currently used); do not expand them to 100. Use existing data for an internal same-task comparison of demo counts with our method only.
- Reuse existing benchmark protocols, metrics, and detailed baseline tables under matching task/data/evaluation settings; add our results without implementing new metrics or rerunning all baselines. Keep adaptive-vs-fixed and data-scaling checks separate as internal comparisons.

| Step                     | What                                                         | Done when |
| ------------------------ | ------------------------------------------------------------ | --------- |
| 1.3a Pick input features | First six feature layers as the current setting; patch tokens, robot state, vision finetune, added one at a time (with the encoder from 1.1) | Oct 25    |
| 1.3b Evaluate            | Follow existing benchmark protocols and metrics (RoboTwin 100 episodes, RoboCasa 50). Compare adaptive vs fixed intervals internally using success and policy-call/timing statistics already available | Nov 4     |
| 1.3c Compare             | Add our results to the existing detailed benchmark tables under comparable settings | Nov 6     |
| 1.3d Ablations           | Action window/sample count, learned vs uniform knots, knot loss, shared vs per-arm knots, input features; internal same-task demo-count comparison | Nov 8     |

If after evaluation (1.3b) we are below the baselines on a benchmark, the paper focuses on
efficiency at equal success (fewer calls, shorter time) on the tasks where we match them.

## 2. Design real-robot tasks (Oct 15 – Oct 21)

Each task mixes a long easy motion with a short precise phase. Adaptive timing should then call the policy rarely during the easy part and often during the precise part.

| Task | Easy part | Precise part |
| ---- | --------- | ------------ |
|      |           |              |
|      |           |              |
|      |           |              |



## 3. Demos, finetune, real-robot test (Oct 22 – Nov 8)

Reuse the gripper-task data already collected. Prepare scenes and data in parallel, but start real-robot finetuning and deployment only after simulation performance is validated and the method is fixed. The dates below remain targets subject to this gate.

| Step                   | What                                                         | Dates          |
| ---------------------- | ------------------------------------------------------------ | -------------- |
| 3.1 Prepare demos      | Reuse existing gripper-task demos; collect only task-specific gaps if needed. Check timestamps, end-effector poses, gripper width, both cameras | Oct 22–26      |
| 3.2 Convert            | Write the demos in the dataset format of the README (`ee_poses`, `gripper`, images, `sample_dt`) | Oct 27         |
| 3.3 Finetune           | Final model from step 1, finetuned on each task's demos      | Oct 28–30      |
| 3.4 Robot control loop | Connect the policy to the robot with `SynchronousSplineExecutor` (README, "Using the policy in your own control loop") | Oct 26–30      |
| 3.5 Test               | Adaptive vs best fixed interval, 20 trials each per task; record success, policy calls, time to finish, video | Oct 31 – Nov 6 |

If the robot setup is not working by **Oct 30**, the real-robot results go to the
supplementary material (Nov 23) instead of the paper.

## 4. Write the paper (Oct 23 – Nov 15)

| Part                                       | Draft by  |
| ------------------------------------------ | --------- |
| Method + method figure                     | **Done**  |
| Related work                               | **Done**  |
| Experiments: simulation tables and figures | Nov 6     |
| Experiments: real robot                    | Nov 8     |
| Introduction, abstract, teaser figure      | Nov 9     |
| Register paper on OpenReview               | Nov 10    |
| Results freeze, final edits                | Nov 12–14 |
| Submit                                     | Nov 15    |
| Supplementary: videos, extra tables, code  | Nov 22    |

---

## Timeline

| Week         | Simulation                                                | Real robot                                                   | Paper                          |
| ------------ | --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------ |
| Oct 15–18    | Benchmarks + baseline numbers (1.2), start encoders (1.1) | Choose robot, design tasks (2)                               | —                              |
| Oct 19–25    | Fix features (1.1), simulation tuning (1.3a)              | Set up scene, prepare existing demos (3.1)                   | —                              |
| Oct 26–Nov 1 | Evaluate (1.3b)                                           | Convert, finetune, control loop (3.2–3.4), subject to simulation validation | Method, related work           |
| Nov 2–8      | Compare + ablations (1.3c, 1.3d)                          | Real-robot tests (3.5)                                       | Experiments                    |
| Nov 9–15     | Freeze results Nov 12                                     | —                                                            | Register Nov 10, submit Nov 15 |
| Nov 16–22    | —                                                         | Extra videos                                                 | Supplementary                  |

