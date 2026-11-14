"""Compare evaluation runs on paired episode seeds: success per task and McNemar tests.

Each --run names a metrics root holding <task>/<Bench>-<task>-<mode>.jsonl files; only
complete files (--episodes records) count. A --pair compares two run:mode columns over
the tasks where both are complete, e.g. --pair win24:adaptive mid6:fixed_0.4s.

Usage: python -m scripts.benchmarks.compare_runs --run mid6 eval_metrics/robocasa-mid6-20261021/robocasa \
    --run win24 eval_metrics/robocasa-win24-20261021/win24/robocasa --pair win24:adaptive mid6:adaptive
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def load_run(root, episodes):
    """{(mode, task): {seed: record}} for the complete JSONL files under root."""
    out = {}
    for file in sorted(Path(root).glob("*/*.jsonl")):
        task = file.parent.name
        mode = file.stem.split(f"-{task}-", 1)[-1]
        records = {}
        for line in file.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["seed"]] = record
        if len(records) >= episodes:
            out[mode, task] = records
    return out


def mcnemar(wins, losses):
    """Exact two-sided McNemar p-value from the discordant pair counts."""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)


def rate(records):
    return 100 * sum(r["success"] for r in records.values()) / len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "ROOT"), required=True)
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("RUN:MODE", "RUN:MODE"))
    parser.add_argument("--episodes", type=int, default=50, help="Records a complete file holds")
    args = parser.parse_args()

    runs = {name: load_run(root, args.episodes) for name, root in args.run}
    columns = sorted({(name, mode) for name, data in runs.items() for mode, _ in data},
                     key=lambda c: ([n for n, _ in args.run].index(c[0]), c[1]))
    tasks = sorted({task for data in runs.values() for _, task in data})
    label = lambda name, mode: f"{name}:{mode.replace('fixed_', '').replace('adaptive', 'ad')}"
    width = max(12, *(len(label(*c)) + 1 for c in columns))
    print("Success (%) per task; -- = missing or incomplete")
    print(f"{'task':28s}" + "".join(f"{label(*c):>{width}s}" for c in columns))
    for task in tasks:
        cells = [runs[n].get((m, task)) for n, m in columns]
        print(f"{task:28s}" + "".join(f"{rate(c):{width}.0f}" if c else f"{'--':>{width}s}" for c in cells))

    if args.pair:
        print("\nPaired comparisons (episodes won/lost by the first column, exact McNemar p)")
    for first, second in args.pair:
        a_name, a_mode = first.split(":", 1)
        b_name, b_mode = second.split(":", 1)
        a, b = runs[a_name], runs[b_name]
        shared = [t for t in tasks if (a_mode, t) in a and (b_mode, t) in b]
        if not shared:
            print(f"{first} vs {second}: no task complete in both")
            continue
        keys = [(t, s) for t in shared for s in a[a_mode, t] if s in b[b_mode, t]]
        x = [a[a_mode, t][s]["success"] for t, s in keys]
        y = [b[b_mode, t][s]["success"] for t, s in keys]
        wins = sum(p and not q for p, q in zip(x, y))
        losses = sum(q and not p for p, q in zip(x, y))
        print(f"{first:>22s} {100 * sum(x) / len(x):5.1f}% vs {second:<22s} {100 * sum(y) / len(y):5.1f}%"
              f"  {len(shared)} tasks  +{wins}/-{losses}  p={mcnemar(wins, losses):.3f}")

    print("\nAdaptive query interval actually executed (median over episodes, s)")
    for name, data in runs.items():
        intervals = [r["executed_steps"] * r["control_dt"] / r["policy_queries"]
                     for (mode, _), records in data.items() if mode == "adaptive"
                     for r in records.values() if r.get("policy_queries")]
        if intervals:
            print(f"{name:12s} {statistics.median(intervals):.3f} ({len(intervals)} episodes)")


if __name__ == "__main__":
    main()
