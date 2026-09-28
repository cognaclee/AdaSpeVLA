"""Synthetic execution checks; these do not measure benchmark task success."""

import argparse
import math
import time

import numpy as np
import torch

from ..execution import SynchronousSplineExecutor
from ..models.adaspe_vla_policy import AdaSpeVLAPolicy
from .metrics import compute_trajectory_jerk


def run_smoke_evaluation(
    benchmark: str, default_task: str, allow_dual_arm: bool = False
):
    parser = argparse.ArgumentParser(
        description=f"Synchronous {benchmark} spline smoke evaluation"
    )
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--task_name", default=default_task)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Sampled execution seconds per episode",
    )
    parser.add_argument(
        "--control_freq", type=float, default=50.0, help="Spline sample rate in Hz"
    )
    parser.add_argument(
        "--num_ode_steps",
        type=int,
        default=None,
        help="Defaults to checkpoint integration steps",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--test_backbone", action="store_true", help="Explicit small offline test model"
    )
    if allow_dual_arm:
        parser.add_argument("--dual_arm", action="store_true")
    args = parser.parse_args()
    if args.num_episodes < 1 or (
        args.num_ode_steps is not None and args.num_ode_steps < 1
    ):
        parser.error("num_episodes and num_ode_steps must be positive")
    if not all(math.isfinite(x) and x > 0 for x in (args.duration, args.control_freq)):
        parser.error("duration and control_freq must be finite and positive")

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=args.device, weights_only=False)
        if ckpt.get("format_version") != 3 or "model_config" not in ckpt:
            raise ValueError(
                "Checkpoint lacks the corrected action/timing contract; retrain with format_version=3"
            )
        if ckpt.get("training_stage") != "timing":
            raise ValueError("Evaluation requires a checkpoint trained through stage 2")
        config = ckpt["model_config"]
        expected_dim = 14 if getattr(args, "dual_arm", False) else 7
        if config["action_dim"] != expected_dim:
            raise ValueError(
                "Checkpoint arm count does not match the requested evaluation"
            )
        policy = AdaSpeVLAPolicy(**config).to(args.device)
        policy.load_state_dict(ckpt["model_state"], strict=True)
        print(f"Checkpoint calibration: {ckpt.get('calibration', {})}")
    else:
        policy = AdaSpeVLAPolicy(
            action_dim=14 if getattr(args, "dual_arm", False) else 7,
            test_mode=args.test_backbone,
            flow_hidden_dim=48 if args.test_backbone else 384,
            flow_num_layers=1 if args.test_backbone else 6,
        ).to(args.device)
        print(
            "No checkpoint supplied: using initialized policy weights for a smoke check."
        )
    if args.num_ode_steps is None:
        args.num_ode_steps = policy.timing_num_ode_steps
    policy.eval()

    print(
        f"{benchmark}: synthetic observations and commands; no simulator task-success measurement."
    )
    dt = 1.0 / args.control_freq
    steps = math.ceil(args.duration / dt)
    for episode in range(args.num_episodes):
        executed_actions = []

        def observe():
            return {
                "images": torch.rand(
                    1, policy.config["num_views"], 3, 224, 224, device=args.device
                ),
                "instructions": [f"perform {args.task_name}"],
            }

        executor = SynchronousSplineExecutor(
            policy,
            observe,
            lambda action: executed_actions.append(action.numpy().copy()),
            control_dt=dt,
            num_ode_steps=args.num_ode_steps,
        )
        started = time.perf_counter()
        for _ in range(steps):
            executor.step()
        wall_seconds = time.perf_counter() - started
        actions = np.array(executed_actions).reshape(steps, policy.num_arms, 7)
        linear_jerk = compute_trajectory_jerk(
            actions[..., :3].reshape(steps, -1), dt=dt, representation="velocity"
        )
        angular_jerk = compute_trajectory_jerk(
            actions[..., 3:6].reshape(steps, -1), dt=dt, representation="velocity"
        )
        print(
            f"Episode {episode + 1}: queries={executor.query_count}, "
            f"queries/sample-second={executor.query_count / (steps * dt):.2f}, "
            f"wall-seconds={wall_seconds:.3f}, linear jerk energy={linear_jerk:.4f}, angular jerk energy={angular_jerk:.4f}"
        )
