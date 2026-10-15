"""Staged training shared by benchmark dataset entry points.

Stages: ``action`` (frozen SigLIP), optional ``finetune`` (action loss with the
last SigLIP vision blocks trainable at a reduced learning rate) and ``timing``
(query-interval head). Each stage runs ``*_steps`` optimizer steps when given,
otherwise its epoch count.
"""

import math
import time
from pathlib import Path
import torch
import argparse
from torch.utils.data import DataLoader
from ..datasets.bspline_dataset import collate_windows
from ..models.adaspe_vla_policy import AdaSpeVLAPolicy

# 4: anchor-relative offsets with a continuous gripper-width spline channel.
CHECKPOINT_FORMAT = 4


def add_method_arguments(parser):
    parser.add_argument("--timing_epochs", type=int, default=10)
    parser.add_argument(
        "--spatial_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition the flow on SigLIP patch tokens via cross-attention",
    )
    parser.add_argument(
        "--proprio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition on robot state (base-frame pose, 6-D rotation, last width)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random crop and colour jitter while training (spatial mode)",
    )
    parser.add_argument(
        "--vision_finetune_layers",
        type=int,
        default=4,
        help="SigLIP vision blocks trained in the finetune stage (0 skips it)",
    )
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument(
        "--vision_lr_scale",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for the fine-tuned SigLIP blocks",
    )
    parser.add_argument(
        "--action_steps", type=int, default=None, help="Optimizer steps; overrides --epochs"
    )
    parser.add_argument(
        "--finetune_steps", type=int, default=None, help="Overrides --finetune_epochs"
    )
    parser.add_argument(
        "--timing_steps", type=int, default=None, help="Overrides --timing_epochs"
    )
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--supported_horizons", type=float, nargs="+", default=None)
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--anchor_stride", type=int, default=1)
    parser.add_argument("--e_crit", type=float, default=None)
    parser.add_argument("--event_quantile", type=float, default=0.95)
    parser.add_argument("--anticipation_margin", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-6)
    parser.add_argument("--timing_probe_count", type=int, default=16)
    parser.add_argument("--num_ode_steps", type=int, default=15)
    parser.add_argument(
        "--per_arm_knots",
        action="store_true",
        help="Ablation for two-arm data: one learned knot partition per arm "
        "(default: one shared partition, as in the manuscript)",
    )
    parser.add_argument(
        "--knot_loss_coeff", type=float, default=0.5, help="alpha in Eq. action_loss"
    )
    parser.add_argument(
        "--flow_loss_coeff", type=float, default=1.0, help="gamma (flow-loss weight)"
    )
    parser.add_argument(
        "--gripper_density_gain",
        type=float,
        default=10.0,
        help="beta_g: gripper-width variation weight in the knot density (Eq. density)",
    )
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=0.5,
        help="theta_g: width threshold for gripper events in the timing teacher",
    )
    parser.add_argument(
        "--knot_supervision",
        choices=["kl", "position"],
        default="kl",
        help="Span KL (manuscript default) or the positional loss L_pos",
    )
    parser.add_argument(
        "--uniform_knots",
        action="store_true",
        help="Ablation: fixed uniform knots instead of learned knots",
    )


def window_options(args):
    options = {
        key: getattr(args, key)
        for key in (
            "supported_horizons",
            "control_dt",
            "anchor_stride",
            "e_crit",
            "event_quantile",
            "anticipation_margin",
            "scale_floor",
            "gripper_density_gain",
            "gripper_threshold",
        )
        if hasattr(args, key)
    }
    if getattr(args, "per_arm_knots", False):
        options["per_arm_knots"] = True
    return options


def train_two_stage(args, dataset, benchmark, action_dim):
    if args.epochs < 1 or args.timing_epochs < 1 or args.batch_size < 1:
        raise ValueError(
            "Both training stages require positive epoch counts and batch size"
        )
    spatial = bool(getattr(args, "spatial_tokens", False))
    use_proprio = bool(getattr(args, "proprio", False))
    finetune_layers = int(getattr(args, "vision_finetune_layers", 0) or 0) if spatial else 0
    if getattr(args, "test_backbone", False):
        finetune_layers = min(finetune_layers, 1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    durations = torch.tensor([record[2][-1] for record in dataset.samples])
    horizon = (
        float(durations.median())
        if args.horizon_duration is None
        else args.horizon_duration
    )
    if not float(durations.min()) - 1e-7 <= horizon <= float(durations.max()) + 1e-7:
        raise ValueError("Inference horizon must lie in the training-supported range")
    per_arm_knots = bool(getattr(args, "per_arm_knots", False))
    if per_arm_knots != getattr(dataset, "per_arm_knots", False):
        raise ValueError("Dataset span targets and policy knot groups disagree")
    policy = AdaSpeVLAPolicy(
        vlm_name=args.vlm_name,
        num_views=dataset.num_views,
        action_dim=action_dim,
        num_control_points=args.num_control_points,
        degree=args.degree,
        test_mode=args.test_backbone,
        motion_scale_values=dataset.motion_scale_values,
        timing_probe_count=args.timing_probe_count,
        timing_num_ode_steps=args.num_ode_steps,
        horizon_duration=horizon,
        supported_horizon_range=[float(durations.min()), float(durations.max())],
        flow_hidden_dim=48 if args.test_backbone else 384,
        flow_num_layers=1 if args.test_backbone else 6,
        per_arm_knots=per_arm_knots,
        knot_loss_coeff=getattr(args, "knot_loss_coeff", 0.5),
        flow_loss_coeff=getattr(args, "flow_loss_coeff", 1.0),
        knot_supervision=getattr(args, "knot_supervision", "kl"),
        uniform_knots=getattr(args, "uniform_knots", False),
        spatial_tokens=spatial,
        proprio_dim=len(dataset.calibration["proprio_mean"]) if use_proprio else 0,
        proprio_mean=dataset.calibration["proprio_mean"] if use_proprio else None,
        proprio_std=dataset.calibration["proprio_std"] if use_proprio else None,
        augment=spatial and bool(getattr(args, "augment", False)),
        vision_finetune_layers=finetune_layers,
    ).to(args.device)
    workers = int(getattr(args, "num_workers", 0) or 0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        num_workers=workers,
        # Not persistent: workers re-fork per pass and see the stage's dataset flags.
        pin_memory=str(args.device).startswith("cuda"),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Training calibration: {dataset.calibration}", flush=True)
    print(
        f"Windows: {len(dataset)}; actual horizons: [{durations.min():.6f}, {durations.max():.6f}] s",
        flush=True,
    )

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(args.device)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    def budget(steps, epochs):
        return int(steps) if steps is not None else int(epochs) * len(loader)

    stages = [("action", budget(getattr(args, "action_steps", None), args.epochs))]
    if finetune_layers:
        stages.append(
            (
                "finetune",
                budget(
                    getattr(args, "finetune_steps", None),
                    getattr(args, "finetune_epochs", 5),
                ),
            )
        )
    stages.append(
        ("timing", budget(getattr(args, "timing_steps", None), args.timing_epochs))
    )
    warmup = int(getattr(args, "warmup_steps", 0) or 0)
    vision_scale = float(getattr(args, "vision_lr_scale", 0.1))
    # Log and checkpoint once per epoch, but at least every 2000 steps.
    log_every = max(1, min(len(loader), 2000))
    for stage, total_steps in stages:
        if total_steps < 1:
            raise ValueError(f"Stage {stage} needs a positive step budget")
        dataset.include_diagnostic_targets = stage != "timing"
        policy.set_training_stage(stage)
        policy.train()
        named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        vision = [p for n, p in named if n.startswith("vlm.vision_encoder.")]
        head = [p for n, p in named if not n.startswith("vlm.vision_encoder.")]
        groups = [dict(params=head, lr=args.lr)]
        if vision:
            groups.append(dict(params=vision, lr=args.lr * vision_scale))
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        stage_warmup = min(warmup, total_steps // 10)

        def schedule(step, total=total_steps, warm=stage_warmup):
            if step < warm:
                return (step + 1) / warm
            progress = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        print(
            f"{stage}: {total_steps} steps ({total_steps / len(loader):.1f} epochs), "
            f"{sum(p.numel() for p in head)} head + {sum(p.numel() for p in vision)} "
            "vision trainable parameters",
            flush=True,
        )
        step, running, started = 0, 0.0, time.time()
        while step < total_steps:
            for raw in loader:
                batch = {key: move(value) for key, value in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, logs = policy.forward_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                running += loss.item()
                step += 1
                if step % log_every == 0 or step == total_steps:
                    count = step % log_every or log_every
                    print(
                        f"{stage} step {step}/{total_steps}: loss={running / count:.6f} "
                        f"({(time.time() - started) / step:.3f} s/step)",
                        flush=True,
                    )
                    running = 0.0
                    checkpoint = dict(
                        model_state=policy.state_dict(),
                        model_config=policy.config,
                        format_version=CHECKPOINT_FORMAT,
                        training_stage=stage,
                        epoch=step / len(loader),
                        step=step,
                        calibration=dataset.calibration,
                        args=vars(args),
                    )
                    torch.save(
                        checkpoint,
                        output / f"adaspevla_{benchmark}_{args.task_name}_{stage}.pt",
                    )
                    if stage == "timing" and step == total_steps:
                        torch.save(
                            checkpoint,
                            output / f"adaspevla_{benchmark}_{args.task_name}_latest.pt",
                        )
                if step >= total_steps:
                    break
    return policy
