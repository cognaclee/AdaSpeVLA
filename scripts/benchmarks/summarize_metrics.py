"""Aggregate per-episode JSONL metrics into an adaptive-vs-fixed comparison table.

Usage: python -m scripts.benchmarks.summarize_metrics <metrics_dir_or_jsonl> [--csv out.csv]
"""

import argparse
import csv
import json
from pathlib import Path

from adaspevla.benchmarks.rollout import summarize_records

COLUMNS = [
    ("task_name", "task"),
    ("knot_mode", "knots"),
    ("query_mode", "mode"),
    ("fixed_query_interval", "fixed_s"),
    ("episodes", "n"),
    ("success_rate", "success"),
    ("queries_per_step", "queries/step"),
    ("nominal_query_hz", "query_hz"),
    ("inference_ms_total", "infer_ms/ep"),
    ("executed_steps", "steps"),
    ("predicted_interval_mean", "pred_tau_s"),
    ("tracking_translation_m_mean", "track_m"),
    ("linear_jerk", "lin_jerk"),
    ("angular_jerk", "ang_jerk"),
]


def load_records(path):
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    records = []
    for file in files:
        with file.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    rows = summarize_records(load_records(args.path))
    if not rows:
        raise SystemExit(f"No episode records under {args.path}")
    table = [[fmt(row[key]) for key, _ in COLUMNS] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in table)) for i, (_, h) in enumerate(COLUMNS)]
    print("  ".join(h.ljust(w) for (_, h), w in zip(COLUMNS, widths)))
    for r in table:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
