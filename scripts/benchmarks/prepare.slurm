#!/usr/bin/bash
#SBATCH --job-name=adaspe-prepare
#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --output=logs/prepare-%A_%a.out
#SBATCH --error=logs/prepare-%A_%a.err
source scripts/benchmarks/slurm_env.sh
args=()
[[ "${DOWNLOAD_DATA:-0}" == 1 ]] && args+=(--download)
[[ -n "${BASE_MOTION:-}" ]] && args+=(--base-motion "$BASE_MOTION")
python -m scripts.benchmarks.prepare_training_data --benchmark "$BENCHMARK" --task "$TASK_NAME" "${args[@]}"
