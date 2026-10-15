# Training and evaluation on Jean Zay (Slurm)

This is how to train and evaluate AdaSpeVLA on the RoboTwin 2.0 and RoboCasa365
benchmarks on Jean Zay. Every job is a self-contained `.slurm` file in the
repository root. All `#SBATCH` options and environment variables live inside the
file, and you submit it with a plain `sbatch <file>.slurm`. To launch a new
experiment, copy the closest file, change its header, and submit the copy, so
every run can be reproduced from its file.

Run all commands from the repository root.

## 1. The pieces

| What | Where |
|---|---|
| Task list (17 tasks, one row each) | `scripts/benchmarks/slurm_tasks.tsv`; frozen copy per run in `checkpoints/<run>/tasks.tsv` |
| Job environment (modules, conda, paths, task row) | `scripts/benchmarks/slurm_env.sh`, sourced by every `.slurm` file |
| Converted demonstrations | `data/converted/<benchmark>/<task>/` |
| Checkpoints | `checkpoints/<run>/<benchmark>/<task>/adaspevla_<benchmark>_<task>_<stage>.pt` |
| Per-episode metrics (JSONL) | `eval_metrics/<run>/...` |
| Logs | `logs/<job-name>-<array id>_<task row>.out` |

Task rows (0-based array index = row of the task file):

| Row | Benchmark | Task | Row | Benchmark | Task |
|---|---|---|---|---|---|
| 0 | robotwin | beat_block_hammer | 9 | robocasa | TurnOnSinkFaucet |
| 1 | robotwin | hanging_mug | 10 | robocasa | TurnOffStove |
| 2 | robotwin | stamp_seal | 11 | robocasa | TurnOnMicrowave |
| 3 | robotwin | place_phone_stand | 12 | robocasa | OpenDrawer |
| 4 | robotwin | stack_blocks_three | 13 | robocasa | CloseFridge |
| 5 | robotwin | handover_block | 14 | robocasa | PickPlaceCounterToCabinet |
| 6 | robotwin | open_microwave | 15 | robocasa | PickPlaceToasterToCounter |
| 7 | robotwin | press_stapler | 16 | robocasa | SlideDishwasherRack |
| 8 | robocasa | CoffeeSetupMug | | | |

`slurm_env.sh` reads the row given by `TASK_ROW` (default: `SLURM_ARRAY_TASK_ID`)
from `TASK_MANIFEST` and exports `BENCHMARK`, `TASK_NAME`, `CHECKPOINT_DIR`
(`$RUN_ROOT/<benchmark>/<task>`) and `CHECKPOINT` (the `_latest.pt` file in it).
Set these variables in the `.slurm` file before sourcing it:

| Variable | Meaning | Default |
|---|---|---|
| `RUN_ROOT` | Run folder for checkpoints | `checkpoints/benchmark-suite` |
| `TASK_MANIFEST` | Task list | `scripts/benchmarks/slurm_tasks.tsv` |
| `METRICS_ROOT` | Evaluation output | `eval_metrics/benchmark-suite` |
| `ARCH_MODULE` | Module tree; `""` for V100 | `arch/h100` |
| `TORCH_CUDA_ARCH_LIST` | `7.0` for V100 | `9.0` |
| `TRAINING_PYTHON` | Training interpreter | `tmp/train-env/bin/python` |
| `ROBOCASA_PYTHON` | RoboCasa simulator interpreter | `tmp/robocasa-env/bin/python` |
| `ROBOTWIN_ROOT` | RoboTwin checkout with the demonstrations | `/lustre/fswork/projects/rech/sdh/ueh86le/GrassmannPolicy/RoboTwin` |

## 2. One-time setup

The `grassmann` conda environment provides PyTorch, SAPIEN and MuJoCo 3.3.1.
Two small virtual environments on top of it fix version conflicts:

- `tmp/train-env` pins Transformers 4.57.6, Hugging Face Hub 0.36.2 and
  Tokenizers 0.22.2. The Transformers in `grassmann` (5.16.1) doesn't work with
  its PyTorch 2.4.1.
- `tmp/robocasa-env` pins NumPy 2.2.5 and SciPy 1.15.3 for RoboCasa. It shares
  the Transformers of `tmp/train-env` through a `.pth` file, so keep both.

To recreate them on a node with internet access:

```bash
python -m venv --system-site-packages tmp/train-env
python -m venv --system-site-packages tmp/robocasa-env
tmp/train-env/bin/python -m pip install \
    transformers==4.57.6 huggingface-hub==0.36.2 tokenizers==0.22.2
tmp/robocasa-env/bin/python -m pip install \
    numpy==2.2.5 scipy==1.15.3 numba==0.61.2 llvmlite==0.44.0 \
    transformers==4.57.6 huggingface-hub==0.36.2 tokenizers==0.22.2
PYTHONPATH="$PWD/robocasa:$PWD/robosuite:$PWD" \
    tmp/robocasa-env/bin/python -c 'import robocasa.wrappers.gym_wrapper'
# Writable RoboTwin mirror in tmp/benchmark-runtime/RoboTwin, with the AdaSpeVLA adapter
python -m scripts.benchmarks.setup_benchmark_runtime
```

The SigLIP backbone (`google/siglip-base-patch16-224`) is read from the Hugging
Face cache in `$HOME/.cache/huggingface`; jobs run with `HF_HUB_OFFLINE=1`.

The WORK space has a quota of 500k files. Jobs set `PYTHONDONTWRITEBYTECODE=1`.
Don't create virtual environments or caches you don't need.

## 3. Prepare the demonstrations

Converted data for all 17 tasks goes in `data/converted/<benchmark>/<task>/`
(one HDF5 per episode plus `adaspevla_conversion.json`). Check which tasks are
there:

```bash
ls data/converted/robotwin data/converted/robocasa
```

To convert a missing task, run this on a login node; RoboCasa downloads need
internet:

```bash
python -m scripts.benchmarks.prepare_training_data --benchmark robocasa --task OpenDrawer --download
python -m scripts.benchmarks.prepare_training_data --benchmark robotwin --task beat_block_hammer
```

Episodes that move RoboCasa's mobile base are rejected. After a failed
conversion, rerun with `--restart-incomplete`. Never run two conversions of the
same task at the same time.

## 4. Train

Training runs three stages, and each one saves a checkpoint in `CHECKPOINT_DIR`:

| Stage | What trains | File |
|---|---|---|
| `action` | Flow-matching action head; SigLIP frozen | `*_action.pt` |
| `finetune` (skipped with `--vision_finetune_layers 0`) | Action head + last N SigLIP vision blocks at `lr × vision_lr_scale` | `*_finetune.pt` |
| `timing` | Query-interval head only | `*_timing.pt`, and `*_latest.pt` at the end |

Evaluation loads `*_latest.pt`, which is written only when the timing stage ends.

### Ready-made training files

| File | Hardware | What it trains |
|---|---|---|
| `run.slurm` | 1 H100, 24 CPUs | One task row per array index, epoch-based (`--epochs 20 --timing_epochs 10`). Used by `submit_suite.sh` |
| `pilot_train.slurm` | 1 V100-32GB, 10 CPUs | Rows 0, 2, 9, 10, step-based (RoboTwin 40k + 20k + 10k steps; RoboCasa 60k + 30k + 10k), all observation features on |
| `pilot_train_noaug.slurm` | 1 V100-32GB, 20 CPUs | Same as `pilot_train.slurm` with `--no-augment` |

Example: a new run on chosen task rows. Copy a training file and set the
array, the run folder and the flags:

```bash
cp pilot_train_noaug.slurm my_train.slurm
# edit: #SBATCH --array=..., RUN_ROOT=..., the train_converted flags
sbatch my_train.slurm
```

The training command inside every file:

```bash
python -m scripts.benchmarks.train_converted \
    --data_dir "$ADASPEVLA_ROOT/data/converted/$BENCHMARK" --task_name "$TASK_NAME" \
    --batch_size 32 --device cuda --num_workers 8 \
    --action_steps 40000 --finetune_steps 20000 --timing_steps 10000 \
    --vision_finetune_layers 4 --vision_lr_scale 0.1 --no-augment \
    --output_dir "$CHECKPOINT_DIR"
```

**The observation features are on by default.** That covers SigLIP patch tokens
with cross-attention, robot state, augmentation, and a 4-block vision finetune
for 5 epochs. A plain `train_converted` call trains this model, including in
`run.slurm`. To train the older pooled-feature model of
`checkpoints/benchmark-suite-20261009`, pass
`--no-spatial_tokens --no-proprio --no-augment --vision_finetune_layers 0`.
All training flags are in the
[README](../README.md#training-options-all-training-scripts).

### Hardware and memory

| | V100-32GB (`-A yfl@v100 -C v100-32g`, `qos_gpu-t4`) | H100 (`-A sdh@h100` or `yfl@h100`, `-C h100`) |
|---|---|---|
| Settings in the file | `ARCH_MODULE=""`, `TORCH_CUDA_ARCH_LIST=7.0` | defaults |
| Speed, RoboTwin (action / finetune / timing) | 0.52 / 0.73 / 0.60 s/step, 6.7 GB GPU | faster |
| Speed, RoboCasa | 0.39 / 0.53 / 0.42 s/step, 5.3 GB GPU | faster |
| Pilot run length | ~11–13 h per task | — |

The dataset decodes every image into CPU RAM. Jean Zay gives about 4 GB of RAM
per requested CPU, and TurnOnSinkFaucet ran out of memory at 40 GB with 10 CPUs.
Request **20 CPUs** (80 GB) for RoboCasa tasks.

## 5. Evaluate

Evaluation runs the closed-loop simulator with the trained `*_latest.pt`, once
with adaptive query timing and once per fixed interval, on the same seeds.

| File | Benchmark | What it runs |
|---|---|---|
| `eval.slurm` | RoboCasa | One job per task: adaptive + fixed 0.05, 0.2, 0.4, 0.5 s, 50 episodes each, target split |
| `eval_robotwin.slurm` | RoboTwin | Sharded: 2 settings (`demo_clean`, `demo_randomized`) × 5 modes (adaptive, fixed 0.06, 0.24, 0.48, 0.5 s) × tasks × 4 shards of 25 episodes. Shard k uses seed k, so shards don't overlap |
| `pilot_eval.slurm`, `pilot_eval_robotwin.slurm` | both | The same for the pilot checkpoints (rows 9, 10 / rows 0, 2) |
| `pilot_noaug_eval.slurm`, `pilot_noaug_eval_robotwin.slurm` | both | The same for the no-augmentation pilot |

To evaluate a new run, copy the matching file and change `RUN_ROOT`,
`METRICS_ROOT`, the array, and for RoboTwin the `TASK_ROWS` list. Array size for
RoboTwin is `2 settings × 5 modes × len(TASK_ROWS) × 4 shards`.

To start evaluation automatically after training, put the dependency in the
file. Use the array task IDs of the training rows it needs:

```bash
#SBATCH --dependency=afterok:<train job>_0:<train job>_2
```

If one of those rows fails, the evaluation stays pending with reason
`DependencyNeverSatisfied`. Cancel it with `scancel` and resubmit after
retraining.

RoboTwin episodes take 5–20 minutes, so a 25-episode shard takes 2–8 hours.
At 16 shards at a time, 80 shards take 1–3 days.

## 6. Read the results

Each episode is one JSONL line with `success`, `policy_queries`, `queries_per_step`,
`inference_ms_mean`, `predicted_interval_mean`, jerk and tracking error. Summarize
a folder (it searches subfolders):

```bash
python -m scripts.benchmarks.summarize_metrics eval_metrics/<run>/robocasa/TurnOffStove
python -m scripts.benchmarks.summarize_metrics \
    eval_metrics/<run>-robotwin/robotwin/beat_block_hammer/demo_clean --csv bbh_clean.csv
```

Check progress while jobs run:

```bash
squeue -u $USER
tail -3 logs/pilot-noaug-train-<job>_<row>.out          # training step and loss
sacct -j <job> --format=JobID,State,Elapsed,MaxRSS      # finished or failed tasks
```

A log ending in `Killed` together with `oom_kill` means the job ran out of CPU
RAM. Resubmit it with more CPUs.

## 7. Whole pipeline in one command (optional)

`bash scripts/benchmarks/submit_suite.sh` submits three linked arrays over the
whole task list: preparation (`scripts/benchmarks/prepare.slurm`), training
(`run.slurm`) and evaluation (`scripts/benchmarks/evaluate.slurm`). It uses
`aftercorr` dependencies, so each task evaluates only after its own training
succeeds. It writes the run to `checkpoints/benchmark-suite-<timestamp>/` and the
job IDs to `slurm-jobs.tsv` there. It runs RoboTwin evaluation unsharded (100
episodes per mode in one job), which can take longer than the time limit. For
RoboTwin, the sharded `eval_robotwin.slurm` is preferred.

## 8. Runs so far

| Run | Model | Status (Oct 15, 2026) |
|---|---|---|
| `benchmark-suite-20261009` | Pooled SigLIP features, no robot state, 20 + 10 epochs, H100 | 15 tasks trained (OpenDrawer, PickPlaceCounterToCabinet missing), evaluated; analysis in `analysis_low_success.md` |
| `pilot-observation-20261014` | All observation features + augmentation, V100 | Rows 0, 2, 10 trained; row 9 out of memory |
| `pilot-noaug-20261015` | Same without augmentation | Training rows 0, 2, 9, 10 |

The paper plan and dates are in `cvpr2027_schedule.md`.
