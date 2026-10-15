#!/usr/bin/bash
#SBATCH --job-name=adaspe-evaluate
#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=100:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --hint=nomultithread
#SBATCH --output=logs/evaluate-%A_%a.out
#SBATCH --error=logs/evaluate-%A_%a.err
source scripts/benchmarks/slurm_env.sh
[[ -f "$CHECKPOINT" ]] || { echo "Missing trained checkpoint: $CHECKPOINT" >&2; exit 1; }
if [[ "$BENCHMARK" == robocasa ]]; then
    # RoboCasa pins NumPy 2.2.5; use an isolated interpreter if provided.
    ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-$ADASPEVLA_ROOT/tmp/robocasa-env/bin/python}"
    [[ -x "$ROBOCASA_PYTHON" ]] || { echo "Create tmp/robocasa-env or set ROBOCASA_PYTHON (NumPy 2.2.5, MuJoCo 3.3.1)" >&2; exit 1; }
    "$ROBOCASA_PYTHON" -m scripts.benchmarks.eval_robocasa_closed_loop \
        --ckpt_path "$CHECKPOINT" --task_name "$TASK_NAME" --split target \
        --num_episodes "${ROBOCASA_EVAL_EPISODES:-50}" --device cuda \
        --fixed_intervals 0.05 0.2 0.4 0.5 \
        --metrics_dir "$METRICS_ROOT/robocasa/$TASK_NAME"
else
    POLICY_DIR="$ADASPEVLA_ROOT/tmp/benchmark-runtime/RoboTwin/XPolicyLab/policy/AdaSpeVLA"
    [[ -f "$POLICY_DIR/eval.sh" ]] || { echo "Run setup_benchmark_runtime.py first" >&2; exit 1; }
    export PYTHONPATH="$ADASPEVLA_ROOT/tmp/benchmark-runtime/RoboTwin:$PYTHONPATH"
    export ADASPEVLA_CHECKPOINT_PATH="$CHECKPOINT" EVAL_ENV_TYPE=sim
    for config in demo_clean demo_randomized; do
        export EXTRA_INFO="task_config=$config,test_num=${ROBOTWIN_EVAL_EPISODES:-100}"
        export ADASPEVLA_METRICS_DIR="$METRICS_ROOT/robotwin/$TASK_NAME/$config"
        ADASPEVLA_QUERY_MODE=adaptive bash "$POLICY_DIR/eval.sh" \
            RoboTwin "$TASK_NAME" "$TASK_NAME" aloha_agilex ee 0 0 0 \
            none "${ROBOTWIN_CONDA_ENV:-grassmann}"
        for interval in 0.06 0.24 0.48 0.5; do
            ADASPEVLA_QUERY_MODE=fixed ADASPEVLA_FIXED_QUERY_INTERVAL="$interval" \
                bash "$POLICY_DIR/eval.sh" RoboTwin "$TASK_NAME" "$TASK_NAME" aloha_agilex ee 0 0 0 \
                none "${ROBOTWIN_CONDA_ENV:-grassmann}"
        done
    done
fi
python -m scripts.benchmarks.summarize_metrics "$METRICS_ROOT/$BENCHMARK/$TASK_NAME"
