#!/usr/bin/bash
# Preparation -> training -> simulator evaluation with matching array indices.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
manifest="${TASK_MANIFEST:-$PWD/scripts/benchmarks/slurm_tasks.tsv}"
[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 1; }
export TASK_MANIFEST="$(realpath "$manifest")"
count=0
while read -r benchmark task extra; do
    [[ "$benchmark" == robotwin || "$benchmark" == robocasa ]] || { echo "Invalid benchmark" >&2; exit 1; }
    [[ "$task" =~ ^[A-Za-z0-9_]+$ && -z "$extra" ]] || { echo "Invalid task row" >&2; exit 1; }
    count=$((count + 1))
done < "$manifest"
((count > 0)) || { echo "Empty task manifest" >&2; exit 1; }
concurrency="${MAX_CONCURRENT_TASKS:-2}"
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_CONCURRENT_TASKS" >&2; exit 1; }
export RUN_ROOT="${RUN_ROOT:-$PWD/checkpoints/benchmark-suite-$(date +%Y%m%d-%H%M%S)}"
export METRICS_ROOT="${METRICS_ROOT:-$PWD/eval_metrics/$(basename "$RUN_ROOT")}" 
mkdir -p logs "$RUN_ROOT" "$METRICS_ROOT"
# Freeze the task mapping for queued jobs, so later manifest edits cannot change it.
cp "$TASK_MANIFEST" "$RUN_ROOT/tasks.tsv"
export TASK_MANIFEST="$RUN_ROOT/tasks.tsv"
jobs="$RUN_ROOT/slurm-jobs.tsv"
printf 'stage\tjob_id\n' > "$jobs"
array="0-$((count - 1))%$concurrency"
prepare=$(sbatch --parsable --array="$array" scripts/benchmarks/prepare.slurm)
prepare=${prepare%%;*}
printf 'prepare\t%s\n' "$prepare" >> "$jobs"
train=$(sbatch --parsable --array="$array" --dependency="aftercorr:$prepare" --kill-on-invalid-dep=yes run.slurm)
train=${train%%;*}
printf 'train\t%s\n' "$train" >> "$jobs"
evaluate=$(sbatch --parsable --array="$array" --dependency="aftercorr:$train" --kill-on-invalid-dep=yes scripts/benchmarks/evaluate.slurm)
evaluate=${evaluate%%;*}
printf 'evaluate\t%s\n' "$evaluate" >> "$jobs"
echo "Submitted $count tasks: prepare=$prepare train=$train eval=$evaluate"
echo "Submission records: $jobs"
