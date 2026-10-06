# Benchmark training and evaluation, step by step

This guide trains and evaluates AdaSpeVLA in simulation:

* [Part A](#part-a-robotwin-20-through-the-xpolicylab-adapter): **RoboTwin 2.0** through the XPolicyLab adapter (main benchmark)
* [Part B](#part-b-robodojo-through-the-same-adapter): **RoboDojo** through the same adapter
* [Part C](#part-c-robocasa365): **RoboCasa365**, run directly without XPolicyLab
* [Part D](#part-d-bimanual-knot-ablation-shared-vs-per-arm-knots): the shared vs per-arm **knot ablation**
* [Part E](#part-e-read-the-results): reading the results
* [Reference](#reference): why these benchmarks, adapter files, caveats

Install AdaSpeVLA first ([README, Step 1](../README.md#step-1-install)).

> **About simulator installation.** The install and download commands for
> RoboTwin, XPolicyLab, RoboDojo and RoboCasa come from those projects, not from
> this repository, and were not run here. If one fails, follow that project's own
> README; every AdaSpeVLA command below was checked against the scripts in this
> repository.

Set these once per shell; every command below uses them:

```bash
export ADASPEVLA_ROOT=/home/lzezeng/codes/AdaSpeVLA   # this repository
export ROBOTWIN_ROOT=/path/to/RoboTwin                # your RoboTwin 2.0 checkout
export XPL_ROOT=${ROBOTWIN_ROOT}/XPolicyLab           # XPolicyLab checkout used by RoboTwin
export POLICY_DIR=${XPL_ROOT}/policy/AdaSpeVLA        # where the adapter gets installed
```

`XPL_ROOT` must be the folder that contains `setup_policy_server.py`; the
installer refuses any other folder.

---

## Part A: RoboTwin 2.0 through the XPolicyLab adapter

XPolicyLab runs two processes: a **policy server** (AdaSpeVLA, in the
`adaspevla` conda env) and a **benchmark client** (the simulator, in the
`RoboTwin` conda env). `eval.sh` starts both for you.

### A1. Install RoboTwin 2.0 (upstream, eval environment)

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git ${ROBOTWIN_ROOT}
cd ${ROBOTWIN_ROOT}
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh            # SAPIEN, CuRobo and dependencies (see RoboTwin README)
bash script/_download_assets.sh    # robot and object assets
```

Make sure the XPolicyLab checkout exists at `${XPL_ROOT}`:

```bash
ls ${XPL_ROOT}/setup_policy_server.py ${XPL_ROOT}/utils
```

### A2. Install the AdaSpeVLA adapter into XPolicyLab (policy environment)

```bash
conda activate adaspevla
cd ${ADASPEVLA_ROOT}

# Copies integrations/xpolicylab/AdaSpeVLA to ${POLICY_DIR} and writes this
# repository's path into ${POLICY_DIR}/.adaspevla_root. Re-running it updates the
# scripts and keeps existing data/, checkpoints/ and eval_metrics/.
bash scripts/benchmarks/install_xpolicylab_adapter.sh ${XPL_ROOT}

# Installs XPolicyLab and AdaSpeVLA (editable) into the active env (adaspevla)
cd ${POLICY_DIR}
bash install.sh
```

Check:

```bash
ls ${POLICY_DIR}          # deploy.py deploy.yml eval.sh install.sh model.py process_data.sh ... train.sh
cat ${POLICY_DIR}/.adaspevla_root    # prints ${ADASPEVLA_ROOT}
```

Run the install step again whenever you update this repository's
`integrations/xpolicylab/AdaSpeVLA`.

### A3. Download demonstrations (upstream)

RoboTwin's XPolicyLab data layout is
`${ROBOTWIN_ROOT}/data/<task_config>/<task>/<embodiment>/data/*.hdf5`, for example
`data/demo_clean/beat_block_hammer/aloha_agilex/data/`.

```bash
cd ${ROBOTWIN_ROOT}
bash scripts/download_xpolicylab_data.sh beat_block_hammer
ls data/demo_clean/beat_block_hammer/aloha_agilex/data | head
```

### A4. Convert the demonstrations

```bash
conda activate adaspevla
cd ${POLICY_DIR}
bash process_data.sh RoboTwin beat_block_hammer aloha_agilex ee
```

Arguments: `<bench_name> <ckpt_name> <env_cfg_type> <action_type>`. The action
type must be `ee`. Output:
`${POLICY_DIR}/data/RoboTwin-beat_block_hammer-aloha_agilex-ee/beat_block_hammer/episode_00000.hdf5 ...`
plus `adaspevla_conversion.json`.

Environment variables for `process_data.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `TASK_NAME` | `<ckpt_name>` | Task to convert |
| `TASK_CONFIG` | `demo_clean` | RoboTwin data folder (`demo_clean`, `demo_randomized`) |
| `SOURCE_DIR` | `${ROBOTWIN_ROOT}/data/<TASK_CONFIG>/<task>/<env_cfg>/data` | Episode directory |
| `ARMS` | `left right` | Arms that supply actions; `right` reproduces the right-arm-only setting |
| `EXTRA_ARGS` | none | Extra converter flags, e.g. `--max_episodes 50`, `--base_wxyz ...`, `--sample_dt ...` |

Example, right arm only with 50 episodes:

```bash
ARMS=right EXTRA_ARGS="--max_episodes 50" bash process_data.sh RoboTwin beat_block_hammer aloha_agilex ee
```

### A5. Train

```bash
conda activate adaspevla
cd ${POLICY_DIR}
TRAIN_ARGS="--epochs 20 --timing_epochs 10 --batch_size 32" \
    bash train.sh RoboTwin beat_block_hammer aloha_agilex ee 0 0
```

Arguments: `<bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>`.
The seed only names the run folder. Output:
`${POLICY_DIR}/checkpoints/RoboTwin-beat_block_hammer-aloha_agilex-ee-0/adaspevla_robotwin_beat_block_hammer_latest.pt`.

* `TRAIN_ARGS` takes any option from the
  [training options table](../README.md#training-options-all-training-scripts),
  for example `"--epochs 20 --knot_supervision position"`.
* `OUTPUT_TAG` adds a suffix to the checkpoint folder, for example
  `OUTPUT_TAG=-uniform TRAIN_ARGS="--uniform_knots"`.

### A6. Check the adapter without the simulator (optional)

XPolicyLab's debug environment client (`EVAL_ENV_TYPE=debug`) exercises the
policy server without SAPIEN; the adapter was tested this way. If your
XPolicyLab version does not read this variable, skip this step.

```bash
cd ${POLICY_DIR}
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin beat_block_hammer beat_block_hammer aloha_agilex ee 0 0 0 adaspevla RoboTwin
```

### A7. Evaluate in simulation

One mode at a time:

```bash
cd ${POLICY_DIR}

# Adaptive query timing (the method)
EXTRA_INFO="task_config=demo_clean,test_num=100" ADASPEVLA_QUERY_MODE=adaptive \
    bash eval.sh RoboTwin beat_block_hammer beat_block_hammer aloha_agilex ee 0 0 0 adaspevla RoboTwin

# Fixed query interval baseline (0.24 s = 4 RoboTwin samples)
EXTRA_INFO="task_config=demo_clean,test_num=100" ADASPEVLA_QUERY_MODE=fixed ADASPEVLA_FIXED_QUERY_INTERVAL=0.24 \
    bash eval.sh RoboTwin beat_block_hammer beat_block_hammer aloha_agilex ee 0 0 0 adaspevla RoboTwin
```

Arguments: `<bench_name> <task_name> <ckpt_name> <env_cfg_type> ee <seed>
<policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>`. The
checkpoint is found from `ckpt_name` and the seed (`checkpoints/<bench>-<ckpt_name>-<env_cfg>-ee-<seed>/`).
To use another checkpoint, set `ADASPEVLA_CHECKPOINT_PATH=/path/to/..._latest.pt`.

For the official protocol, evaluate on both `task_config=demo_clean` and
`task_config=demo_randomized`, 100 episodes each.

Adaptive plus several fixed intervals for several tasks in one command:

```bash
cd ${POLICY_DIR}
EXTRA_INFO="task_config=demo_randomized,test_num=100" \
FIXED_INTERVALS="0.06 0.24 0.48 0.5" \
    bash ${ADASPEVLA_ROOT}/scripts/benchmarks/run_xpolicylab_sweep.sh ${POLICY_DIR} \
        RoboTwin aloha_agilex 0 0 0 adaspevla RoboTwin \
        beat_block_hammer hanging_mug stamp_seal
```

Arguments: `<policy_dir> <bench_name> <env_cfg_type> <seed> <policy_gpu>
<env_gpu> <policy_conda_env> <eval_conda_env> <task> [task ...]`. Each task must
already be converted and trained (A4, A5) with `ckpt_name` equal to the task.
The sweep prints the comparison table at the end (Part E).

Evaluation settings (environment variable or `deploy.yml` key):

| Env variable | `deploy.yml` key | Default | Meaning |
|---|---|---|---|
| `ADASPEVLA_QUERY_MODE` | `query_mode` | `adaptive` | `adaptive` or `fixed` |
| `ADASPEVLA_FIXED_QUERY_INTERVAL` | `fixed_query_interval` | none | Seconds; required for `fixed` |
| `ADASPEVLA_CHECKPOINT_PATH` | `checkpoint_path` | from `ckpt_name` | Explicit `*_latest.pt` |
| `ADASPEVLA_NUM_ODE_STEPS` | `num_ode_steps` | from checkpoint | Flow integration steps |
| `ADASPEVLA_METRICS_DIR` | `metrics_dir` | `${POLICY_DIR}/eval_metrics` | JSONL output |
| `ADASPEVLA_CONTROL_DT` | `control_dt` | converted `sample_dt` | One action per recorded sample |
| `ADASPEVLA_BINARY_GRIPPER` | `binary_gripper` | `false` | Send `1[w >= 0.5]` instead of the continuous width |
| `EXTRA_INFO` | | none | RoboTwin options, e.g. `task_config=demo_clean,test_num=100` |
| | `eval_batch` | `false` | Keep `false` for timing studies |

### A8. Recommended task suite

`beat_block_hammer`, `hanging_mug`, `stamp_seal`, `place_phone_stand`,
`stack_blocks_three`, `handover_block`, `open_microwave`, `press_stapler`. Train
on `demo_clean`, then evaluate on `demo_clean` and `demo_randomized`. A loop:

```bash
cd ${POLICY_DIR}
for task in beat_block_hammer hanging_mug stamp_seal place_phone_stand \
            stack_blocks_three handover_block open_microwave press_stapler; do
    (cd ${ROBOTWIN_ROOT} && bash scripts/download_xpolicylab_data.sh ${task})
    bash process_data.sh RoboTwin ${task} aloha_agilex ee
    TRAIN_ARGS="--epochs 20 --timing_epochs 10 --batch_size 32" \
        bash train.sh RoboTwin ${task} aloha_agilex ee 0 0
done
```

### A9. Without XPolicyLab (training only)

The converter and trainer also run directly:

```bash
conda activate adaspevla
cd ${ADASPEVLA_ROOT}
python -m scripts.benchmarks.convert_xpolicylab \
    --source_dir ${ROBOTWIN_ROOT}/data/demo_clean/beat_block_hammer/aloha_agilex/data \
    --output_dir ./data/converted/robotwin --task_name beat_block_hammer \
    --profile robotwin --arms left right
python -m scripts.benchmarks.train_converted \
    --data_dir ./data/converted/robotwin --task_name beat_block_hammer \
    --epochs 20 --timing_epochs 10 --output_dir ./checkpoints/robotwin2
```

Converter options: `--profile robotwin|robodojo`, `--arms` (`left`, `right`,
`single`, in channel order; default `right`), `--cameras` (default: head plus the
wrist camera of each arm), `--base_wxyz` (4 values, or 4 per arm),
`--sample_dt`, `--simulation_dt`, `--max_episodes`. The converter stores the
observed poses and the continuous commanded widths; it never thresholds them.

---

## Part B: RoboDojo through the same adapter

RoboDojo (Isaac Sim 5.1) uses XPolicyLab too, so the adapter and scripts are the
same. Only the data step differs. **The RoboDojo pose frame and the `ee` mode
were not verified in a RoboDojo simulator.**

### B1. Install RoboDojo and its XPolicyLab (upstream)

Install Isaac Sim 5.1 and RoboDojo following the RoboDojo README, in its own
conda env (here called `RoboDojo`). Then point the variables at its XPolicyLab:

```bash
export XPL_ROOT=/path/to/RoboDojo/XPolicyLab
export POLICY_DIR=${XPL_ROOT}/policy/AdaSpeVLA
```

### B2. Install the adapter

```bash
conda activate adaspevla
cd ${ADASPEVLA_ROOT}
bash scripts/benchmarks/install_xpolicylab_adapter.sh ${XPL_ROOT}
cd ${POLICY_DIR} && bash install.sh
```

### B3. Convert

RoboDojo has no default data path and no default base frame, so set both. Use
`1 0 0 0` if the exported end-effector poses are already base-relative:

```bash
cd ${POLICY_DIR}
SOURCE_DIR=/path/to/robodojo/episodes/<task> \
EXTRA_ARGS="--base_wxyz 1 0 0 0" \
    bash process_data.sh RoboDojo <task> <env_cfg_type> ee
```

The `robodojo` profile reads the `frequency` field as a rate in Hz
(`collect_freq: 25`), unlike RoboTwin's save stride.

### B4. Train and evaluate

Same as A5 and A7, with `RoboDojo` as the bench name and `RoboDojo` as the eval env:

```bash
cd ${POLICY_DIR}
TRAIN_ARGS="--epochs 20 --timing_epochs 10" bash train.sh RoboDojo <task> <env_cfg_type> ee 0 0
ADASPEVLA_QUERY_MODE=adaptive \
    bash eval.sh RoboDojo <task> <task> <env_cfg_type> ee 0 0 0 adaspevla RoboDojo
```

---

## Part C: RoboCasa365

RoboCasa365 runs in-process: the evaluator imports RoboCasa directly, so
AdaSpeVLA and RoboCasa must be installed in **one** environment. No XPolicyLab.

### C1. Install RoboCasa (upstream) into the AdaSpeVLA environment

```bash
conda activate adaspevla
cd /path/to/workdir
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite
git clone https://github.com/robocasa/robocasa && pip install -e robocasa
pip install gymnasium
python robocasa/scripts/download_kitchen_assets.py   # about 10 GB
python robocasa/scripts/setup_macros.py
```

Check that the gym wrapper imports:

```bash
python -c "import gymnasium, robocasa.wrappers.gym_wrapper; print('ok')"
```

### C2. Download demonstrations (upstream)

```bash
python -m robocasa.scripts.download_datasets --tasks OpenDrawer --split target --source human
```

The data is LeRobot v2.1 at
`<robocasa>/datasets/v1.0/target/atomic/OpenDrawer/<date>/lerobot` (the folder
that contains `meta/info.json`).

### C3. Convert

```bash
cd ${ADASPEVLA_ROOT}
python -m scripts.benchmarks.convert_robocasa \
    --source_dir /path/to/robocasa/datasets/v1.0/target/atomic/OpenDrawer/<date>/lerobot \
    --output_dir ./data/converted/robocasa \
    --task_name OpenDrawer
```

Output: `./data/converted/robocasa/OpenDrawer/episode_*.hdf5` and
`adaspevla_conversion.json`. Options:

| Option | Default | Meaning |
|---|---|---|
| `--cameras` | `robot0_agentview_left robot0_eye_in_hand` | Camera names, in view order |
| `--base_motion` | `error` | Episodes that drive the mobile base: `error` stops, `drop` skips them, `ignore` keeps them |
| `--max_episodes` | all | Limit the number of episodes |

Only stationary-base tasks work. If conversion stops with a base-motion error,
the task moves the base; pick another task or use `--base_motion drop`.

### C4. Train

```bash
python -m scripts.benchmarks.train_converted \
    --data_dir ./data/converted/robocasa --task_name OpenDrawer \
    --epochs 20 --timing_epochs 10 --batch_size 32 \
    --output_dir ./checkpoints/robocasa
```

Output: `./checkpoints/robocasa/adaspevla_robocasa_OpenDrawer_latest.pt`.

### C5. Evaluate in closed loop

```bash
python -m scripts.benchmarks.eval_robocasa_closed_loop \
    --ckpt_path ./checkpoints/robocasa/adaspevla_robocasa_OpenDrawer_latest.pt \
    --task_name OpenDrawer --split target --num_episodes 50 \
    --fixed_intervals 0.05 0.2 0.4 0.5
```

This runs adaptive timing first, then each fixed interval, all on the same
episode seeds, and prints a summary. Options:

| Option | Default | Meaning |
|---|---|---|
| `--split` | `target` | `target` or `pretrain` |
| `--num_episodes` | 50 | Episodes per mode |
| `--seed` | 0 | First episode seed |
| `--max_steps` | task horizon | Step limit per episode |
| `--fixed_intervals` | none | Fixed query intervals in seconds |
| `--skip_adaptive` | off | Run only the fixed intervals |
| `--num_ode_steps` | from checkpoint | Flow integration steps |
| `--metrics_dir` | `./eval_metrics/robocasa` | JSONL output |

### C6. Recommended task suite

Atomic-Seen target split, stationary tasks: `CoffeeSetupMug`, `TurnOnSinkFaucet`,
`TurnOffStove`, `TurnOnMicrowave`, `OpenDrawer`, `CloseFridge`,
`PickPlaceCounterToCabinet`, `PickPlaceToasterToCounter`, `SlideDishwasherRack`.

```bash
cd ${ADASPEVLA_ROOT}
for task in CoffeeSetupMug TurnOnSinkFaucet TurnOffStove TurnOnMicrowave OpenDrawer \
            CloseFridge PickPlaceCounterToCabinet PickPlaceToasterToCounter SlideDishwasherRack; do
    python -m robocasa.scripts.download_datasets --tasks ${task} --split target --source human
    src=$(ls -d /path/to/robocasa/datasets/v1.0/target/atomic/${task}/*/lerobot | head -1)
    python -m scripts.benchmarks.convert_robocasa --source_dir ${src} \
        --output_dir ./data/converted/robocasa --task_name ${task}
    python -m scripts.benchmarks.train_converted --data_dir ./data/converted/robocasa \
        --task_name ${task} --epochs 20 --timing_epochs 10 --output_dir ./checkpoints/robocasa
    python -m scripts.benchmarks.eval_robocasa_closed_loop \
        --ckpt_path ./checkpoints/robocasa/adaspevla_robocasa_${task}_latest.pt \
        --task_name ${task} --num_episodes 50 --fixed_intervals 0.05 0.2 0.4 0.5
done
```

---

## Part D: Bimanual knot ablation (shared vs per-arm knots)

The paper uses one knot partition for both arms. `--per_arm_knots` gives each
arm its own learned partition and changes nothing else. It needs two-arm data,
so convert with `ARMS="left right"` (the default in A4).

### D1. All in one command

For each task this trains both variants with identical settings, compares them
offline, and evaluates both in simulation:

```bash
cd ${POLICY_DIR}
TRAIN_ARGS="--epochs 20 --timing_epochs 10 --batch_size 32" \
EXTRA_INFO="task_config=demo_clean,test_num=100" \
    bash ${ADASPEVLA_ROOT}/scripts/benchmarks/run_knot_ablation.sh ${POLICY_DIR} \
        RoboTwin aloha_agilex 0 0 0 adaspevla RoboTwin \
        handover_block hanging_mug stack_blocks_three
```

Arguments are the same as the sweep in A7. Environment variables: `TRAIN_ARGS`,
`SKIP_TRAIN=1` (reuse existing checkpoints), `FIXED_INTERVALS` (also evaluate
fixed intervals), `EXTRA_INFO`, `METRICS_DIR` (default
`${POLICY_DIR}/knot_ablation`).

### D2. Step by step

```bash
cd ${POLICY_DIR}

# 1. Train both variants
bash train.sh RoboTwin handover_block aloha_agilex ee 0 0
TRAIN_ARGS="--per_arm_knots" OUTPUT_TAG="-per_arm_knots" \
    bash train.sh RoboTwin handover_block aloha_agilex ee 0 0

SHARED=${POLICY_DIR}/checkpoints/RoboTwin-handover_block-aloha_agilex-ee-0/adaspevla_robotwin_handover_block_latest.pt
PER_ARM=${POLICY_DIR}/checkpoints/RoboTwin-handover_block-aloha_agilex-ee-0-per_arm_knots/adaspevla_robotwin_handover_block_latest.pt

# 2. Offline comparison on the converted windows
cd ${ADASPEVLA_ROOT}
python -m scripts.benchmarks.eval_knots_offline \
    --data_dir ${POLICY_DIR}/data/RoboTwin-handover_block-aloha_agilex-ee \
    --task_name handover_block --ckpt_paths ${SHARED} ${PER_ARM} \
    --output_json ./knot_offline_handover_block.json

# 3. Simulation, once per checkpoint
cd ${POLICY_DIR}
for ckpt in ${SHARED} ${PER_ARM}; do
    ADASPEVLA_CHECKPOINT_PATH=${ckpt} ADASPEVLA_QUERY_MODE=adaptive \
        bash eval.sh RoboTwin handover_block handover_block aloha_agilex ee 0 0 0 adaspevla RoboTwin
done
```

`eval_knots_offline` options: `--max_windows` (2000), `--batch_size` (16),
`--num_ode_steps`, `--seed` (0), `--output_json`, `--device`.

The offline report is in normalized motion units at the demonstration
timestamps:

* `fit_mse_<arm>`: best fit on the predicted knots (representation capacity).
* `uniform_fit_mse`: the same fit on uniform knots, as a reference.
* `pred_mse_<arm>`: the emitted spline against the demonstration.
* `knot_span_l1`: how far apart the two arms' predicted partitions are.
* `teacher_span_l1`: how asynchronous the arms are in the data. Plot the gain
  against this.

For a fair comparison use the same seeds, `TRAIN_ARGS` and
`num_control_points`. To match total spline capacity instead, also run the
shared variant with `--num_control_points 10`. Expect a gain where one arm does
contact while the other transports, and parity on synchronous tasks.

---

## Part E: Read the results

Every evaluation writes one JSONL record per episode (XPolicyLab:
`${POLICY_DIR}/eval_metrics/`, RoboCasa: `./eval_metrics/robocasa/`, knot
ablation: `${POLICY_DIR}/knot_ablation/`). Print the comparison table:

```bash
cd ${ADASPEVLA_ROOT}
python -m scripts.benchmarks.summarize_metrics ${POLICY_DIR}/eval_metrics
python -m scripts.benchmarks.summarize_metrics ./eval_metrics/robocasa --csv robocasa_summary.csv
```

Records hold success, executed steps, policy queries, queries per step, nominal
query rate, per-query inference time, the predicted-interval distribution,
prefix-length histogram, commanded-pose tracking error, linear and angular jerk of
the commanded pose targets, the velocity jump of those targets at prefix
boundaries, the mean commanded gripper width and `knot_mode`. Rows are grouped by mode (adaptive or
each fixed interval) and by `knot_mode`.

---

## Reference

### Why these benchmarks

AdaSpeVLA claims that adapting the policy-query interval gives a better
success-versus-compute trade-off than one fixed replanning rate. A benchmark
shows this only if its tasks mix long predictable motion with short phases where
fresh feedback matters, and if its controller accepts dense physical commands
between policy calls.

| Benchmark | Use | Why |
|---|---|---|
| **RoboTwin 2.0** (SAPIEN, Aloha-AgileX, 50 bimanual tasks) | **Main** | Free-space transport mixed with short contact phases. The `ee` mode takes absolute end-effector pose targets, which our anchor-relative offsets map to. Clean and randomized settings and a public leaderboard. |
| **RoboCasa365** (MuJoCo/robosuite, PandaOmron, 20 Hz OSC) | **Second** | Different simulator, embodiment and controller. Real per-frame timestamps, so timing labels are exact. Stationary-base tasks only. |
| **RoboDojo** (Isaac Sim, 42 tasks) | Optional | Shares XPolicyLab with RoboTwin. Its end-effector frame is unverified, so the base quaternion must be given explicitly. |
| ActiveArena-Bench | Not used | Targets active perception with 18-D joint actions; AdaSpeVLA has no camera control. |

LIBERO and its variants are single-arm with short horizons and near-saturated
scores, so they show little about feedback timing.

### Adapter files (`integrations/xpolicylab/AdaSpeVLA`)

| File | Role |
|---|---|
| `install.sh` | `pip install -e` XPolicyLab and AdaSpeVLA into the active env |
| `_paths.sh` | Finds `XPL_ROOT`, the benchmark root and `ADASPEVLA_ROOT` (from `.adaspevla_root` or the environment) |
| `process_data.sh` | Calls `scripts.benchmarks.convert_xpolicylab` (A4) |
| `train.sh` | Calls `scripts.benchmarks.train_converted` (A5) |
| `eval.sh` | Starts the policy server and the benchmark client (A7) |
| `setup_eval_policy_server.sh`, `setup_eval_env_client.sh` | Called by `eval.sh`; adapted from XPolicyLab's `demo_policy` |
| `deploy.yml` | Policy server configuration (A7 table) |
| `model.py` | XPolicyLab entry point; the model is `adaspevla/benchmarks/xpolicylab_model.py` |
| `deploy.py` | Episode loop. Each `get_action` returns one adaptive prefix, executed fully before the next observation |

### Caveats

* **Actions.** Each predicted offset is relative to the end-effector pose
  observed at query time and becomes an absolute `*_ee_pose` target by
  Eq. delta_to_target (`p = p_k + R_b a_pos`, `R = Exp(R_b a_rot) R_k`), using
  the per-arm base rotation recorded at conversion (Aloha default
  `[0.707, 0, 0, 0.707]`). The gripper width is clamped to [0,1] and sent as a
  continuous `*_ee_joint_state` (1 open). Arms the checkpoint does not control
  hold their pose.
* **Timing is nominal on RoboTwin.** One executed action equals one recorded
  sample (`sample_dt = save_freq × 1/250 s = 0.06 s`). The `ee` mode plans its
  own path to each target, so simulated time per action varies. Report queries
  per executed step alongside the nominal query rate.
* **Batched evaluation** (`eval_batch: true`) forces every environment to the
  shortest prefix in the batch. Use unbatched runs for timing results.
* **RoboCasa control.** Training targets are achieved motion: offsets of the
  recorded base-to-end-effector pose from the pose at each window start, with
  the `timestamp` column; the binary `gripper_close` log maps to widths {0, 1}.
  The evaluator maps offsets to pose targets from the pose observed at query
  time and commands `a = (target − current)/scale`, clipped, at every 20 Hz
  step, using fresh state; the gripper receives `1[w >= 0.5]`. The policy runs
  only at prefix boundaries.
* **Fixed vs adaptive.** Both modes run the same checkpoint on the same seeds.
  Fixed mode only replaces the predicted interval with a constant, which is the
  paper's fixed-versus-supervised ablation.

### Validation status

Run offline here: the full test suite (119 tests, including the benchmark
tests in `tests/benchmarks/`). They cover anchor-offset round trips, the converters, timing
profiles, the executor step rule, the XPolicyLab model, the convert-then-train
CLIs, the RoboCasa closed loop against a stub environment, and the per-arm knot
ablation. The adapter also ran end to end through XPolicyLab's policy server
with the debug client (`EVAL_ENV_TYPE=debug`).

**Not run here:** RoboTwin (SAPIEN with CuRobo), RoboDojo (Isaac Sim) and
RoboCasa365 simulation. Nothing here measures task success yet.
