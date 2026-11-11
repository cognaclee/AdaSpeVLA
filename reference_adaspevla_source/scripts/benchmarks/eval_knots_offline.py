"""Compare shared and per-arm knot checkpoints offline on demonstration windows.

Usage:
  python -m scripts.benchmarks.eval_knots_offline --data_dir <converted> --task_name <task> \
      --ckpt_paths shared/..._latest.pt per_arm/..._latest.pt [--output_json out.json]

Use held-out converted episodes for --data_dir when available; on the training
data the numbers measure fit, not generalization. See adaspevla/benchmarks/knot_analysis.py
for the definition of each column.
"""

import argparse
import json

import torch

from adaspevla.benchmarks.knot_analysis import analyze_knots, load_analysis_dataset
from adaspevla.benchmarks.rollout import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_dir", required=True, help="Converter output_dir")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--ckpt_paths", nargs="+", required=True)
    parser.add_argument("--max_windows", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_ode_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    results = []
    for path in args.ckpt_paths:
        policy, _, ckpt = load_policy(path, args.device)
        dataset, arms = load_analysis_dataset(args.data_dir, args.task_name, ckpt)
        torch.manual_seed(args.seed)
        result = analyze_knots(
            policy,
            dataset,
            arms,
            max_windows=args.max_windows,
            batch_size=args.batch_size,
            num_ode_steps=args.num_ode_steps,
            seed=args.seed,
        )
        result.update(checkpoint=str(path), task_name=args.task_name, arms=arms)
        results.append(result)
    columns = ["knot_mode", "windows", "knot_parameters"]
    columns += [f"fit_mse_{a}" for a in results[0]["arms"]] + ["uniform_fit_mse"]
    columns += [f"pred_mse_{a}" for a in results[0]["arms"]]
    columns += [c for c in ("knot_span_l1", "teacher_span_l1") if c in results[0]]
    table = [
        [f"{r.get(c):.4g}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in columns]
        for r in results
    ]
    widths = [max(len(c), *(len(row[i]) for row in table)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in table:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
