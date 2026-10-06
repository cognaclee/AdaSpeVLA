"""Two-stage training shared by benchmark dataset entry points."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..datasets.bspline_dataset import collate_windows
from ..models.adaspe_vla_policy import AdaSpeVLAPolicy

# 4: anchor-relative offsets with a continuous gripper-width spline channel.
CHECKPOINT_FORMAT = 4


def add_method_arguments(parser):
    parser.add_argument("--timing_epochs", type=int, default=10)
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
    ).to(args.device)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_windows
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

    for stage, epochs in [("action", args.epochs), ("timing", args.timing_epochs)]:
        dataset.include_diagnostic_targets = stage == "action"
        policy.set_training_stage(stage)
        policy.train()
        params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            total = 0.0
            for raw in loader:
                batch = {key: move(value) for key, value in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, logs = policy.forward_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            print(
                f"{stage} epoch {epoch+1}/{epochs}: loss={total/len(loader):.6f}",
                flush=True,
            )
            checkpoint = dict(
                model_state=policy.state_dict(),
                model_config=policy.config,
                format_version=CHECKPOINT_FORMAT,
                training_stage=stage,
                epoch=epoch,
                calibration=dataset.calibration,
                args=vars(args),
            )
            torch.save(
                checkpoint,
                output / f"adaspevla_{benchmark}_{args.task_name}_{stage}.pt",
            )
            if stage == "timing":
                torch.save(
                    checkpoint,
                    output / f"adaspevla_{benchmark}_{args.task_name}_latest.pt",
                )
    return policy
