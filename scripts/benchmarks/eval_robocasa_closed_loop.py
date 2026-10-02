"""Closed-loop RoboCasa365 evaluation: adaptive query timing vs fixed intervals.

Every mode runs the same checkpoint on the same episode seeds. Per-episode JSONL
records (success, queries, inference time, tracking error, jerk) are written to
--metrics_dir and summarized at the end.

Example:
    python -m scripts.benchmarks.eval_robocasa_closed_loop \
        --ckpt_path checkpoints/robocasa/adaspevla_robocasa_CloseBlenderLid_latest.pt \
        --task_name CloseBlenderLid --split target --num_episodes 50 \
        --fixed_intervals 0.05 0.2 0.4 0.5
"""

import argparse
from pathlib import Path

import torch

from adaspevla.benchmarks.robocasa import RoboCasaClosedLoop
from adaspevla.benchmarks.rollout import append_jsonl, load_policy, summarize_records


def make_env(task_name, split, image_shape):
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<Task>)

    height, width = int(image_shape[0]), int(image_shape[1])
    return gym.make(
        f"robocasa/{task_name}", split=split, camera_widths=width, camera_heights=height
    )


def task_horizon(task_name):
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    return get_task_horizon(task_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None, help="Default: RoboCasa task horizon")
    parser.add_argument("--fixed_intervals", type=float, nargs="*", default=[],
                        help="Also evaluate these fixed query intervals (seconds)")
    parser.add_argument("--skip_adaptive", action="store_true")
    parser.add_argument("--num_ode_steps", type=int, default=None)
    parser.add_argument("--metrics_dir", default="./eval_metrics/robocasa")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    policy, calibration, _ = load_policy(args.ckpt_path, args.device)
    meta = calibration.get("source_conversion", {})
    if meta.get("source") != "robocasa365_lerobot_v21":
        raise ValueError("Checkpoint was not trained on scripts.benchmarks.convert_robocasa output")
    max_steps = args.max_steps or task_horizon(args.task_name)
    env = make_env(args.task_name, args.split, meta["image_shape"])
    modes = ([] if args.skip_adaptive else [("adaptive", None)]) + [
        ("fixed", interval) for interval in args.fixed_intervals
    ]
    records = []
    for mode, interval in modes:
        runner = RoboCasaClosedLoop(
            policy, meta, query_mode=mode, fixed_query_interval=interval,
            num_ode_steps=args.num_ode_steps, device=args.device,
        )
        tag = "adaptive" if mode == "adaptive" else f"fixed_{interval:g}s"
        path = Path(args.metrics_dir) / f"RoboCasa-{args.task_name}-{tag}.jsonl"
        for episode in range(args.num_episodes):
            record = runner.run_episode(env, seed=args.seed + episode, max_steps=max_steps)
            record.update(
                bench_name="RoboCasa", task_name=args.task_name, split=args.split,
                query_mode=mode, fixed_query_interval=interval, checkpoint=args.ckpt_path,
            )
            append_jsonl(path, record)
            records.append(record)
            print(
                f"[{tag}] episode {episode}: success={record['success']} "
                f"steps={record['executed_steps']} queries={record['policy_queries']}",
                flush=True,
            )
    env.close()
    for row in summarize_records(records):
        print(row)


if __name__ == "__main__":
    main()
