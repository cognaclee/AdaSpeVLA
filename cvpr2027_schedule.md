# AdaSpeVLA — CVPR 2027 Plan

Deadline **Nov 16, 2026** (register by Nov 10, supplementary Nov 23). Today: Oct 15.

Four goals, in order:

1. Get good results on RoboTwin and RoboCasa (simulation)
2. Design real-robot tasks
3. Collect a few demos, finetune the policy, test on the real robot
4. Write the paper

---

## 1. Good simulation results (Oct 15 – Nov 8)

Target: beat our current baseline clearly, and show adaptive timing ≥ best fixed
interval with fewer policy calls.

| Step | What | Done when |
|---|---|---|
|                         |                                                              |  |
| 1.2 Pick encoders | Compare vision/text encoders on 4 dev tasks (beat_block_hammer, press_stapler, CloseFridge, SlideDishwasherRack): SigLIP-B 224 (current), SigLIP-B 256, SigLIP-SO400M, DINOv2 + T5, CLIP-L | Oct 22 |
| 1.3 Pick input features | Patch tokens, robot state, vision finetune, added one at a time | Oct 25 |
| 1.4 Train all tasks | Final model on 8 RoboTwin + 9 RoboCasa tasks | Oct 28 |
| 1.5 Evaluate | Adaptive vs fixed intervals; RoboTwin 100 episodes (clean + randomized), RoboCasa 50 | Nov 4 |
| 1.6 Ablations | Learned vs uniform knots, knot loss, shared vs per-arm knots, input features (from 1.3) | Nov 6 |
| 1.7 Compare | Published numbers of other methods (e.g. Diffusion Policy, ACT, π0) on the same tasks | Nov 8 |

## 2. Design real-robot tasks (Oct 15 – Oct 21)

Each task mixes a long easy motion with a short precise phase. Adaptive timing should then call the policy rarely during the easy part and often during the precise part.

| Task | Easy part | Precise part |
|---|---|---|
|      |           |              |
|      |           |              |
|      |           |              |



## 3. Demos, finetune, real-robot test (Oct 22 – Nov 8)

| Step | What | Dates |
|---|---|---|
| 3.1 Collect demos | 20–30 teleoperated demos per task, with timestamps, end-effector poses, gripper width, both cameras | Oct 22–26 |
| 3.2 Convert | Write the demos in the dataset format of the README (`ee_poses`, `gripper`, images, `sample_dt`) | Oct 27 |
| 3.3 Finetune | Final model from step 1, finetuned on each task's demos | Oct 28–30 |
| 3.4 Robot control loop | Connect the policy to the robot with `SynchronousSplineExecutor` (README, "Using the policy in your own control loop") | Oct 26–30 |
| 3.5 Test | Adaptive vs best fixed interval, 20 trials each per task; record success, policy calls, time to finish, video | Oct 31 – Nov 6 |

If the robot setup is not working by **Oct 30**, the real-robot results go to the
supplementary material (Nov 23) instead of the paper.

## 4. Write the paper (Oct 23 – Nov 15)

| Part | Draft by |
|---|---|
| Method + method figure | Oct 27 |
| Related work | Oct 29 |
| Experiments: simulation tables and figures | Nov 6 |
| Experiments: real robot | Nov 8 |
| Introduction, abstract, teaser figure | Nov 9 |
| Register paper on OpenReview | Nov 10 |
| Results freeze, final edits | Nov 12–14 |
| Submit | Nov 15 |
| Supplementary: videos, extra tables, code | Nov 22 |

---

## Timeline

| Week | Simulation | Real robot | Paper |
|---|---|---|---|
| Oct 15–18 | Fix recipe (1.1) | Choose robot, design tasks (2) | — |
| Oct 19–25 | Encoders + features (1.2, 1.3) | Set up scene, start demos (3.1) | — |
| Oct 26–Nov 1 | Train + evaluate all tasks (1.4, 1.5) | Convert, finetune, control loop (3.2–3.4) | Method, related work |
| Nov 2–8 | Ablations + comparison (1.6, 1.7) | Real-robot tests (3.5) | Experiments |
| Nov 9–15 | Freeze results Nov 12 | — | Register Nov 10, submit Nov 15 |
| Nov 16–22 | — | Extra videos | Supplementary |
