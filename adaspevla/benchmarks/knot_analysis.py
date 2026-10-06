"""Offline shared-vs-per-arm knot comparison on demonstration windows.

The per-arm knot option is an ablation of the manuscript's single shared knot
partition. Every quantity here is in normalized action units (anchor-relative
offsets divided by the checkpoint's motion scales, and the gripper width mapped
to 2w-1, seven channels per arm) evaluated at the demonstration timestamps:

* ``fit_mse``: best least-squares fit of the demonstration on the knots the
  policy predicts. This isolates representation capacity of the knot layout.
* ``uniform_fit_mse``: the same fit on uniform clamped knots (reference).
* ``pred_mse``: the sampled spline the policy actually emits versus the
  demonstration (depends on the flow sample, so it is noisier).
* ``knot_span_l1``: L1 distance between the two arms' predicted span vectors
  (zero for a shared partition).
* ``teacher_span_l1``: L1 distance between the two arms' span *targets*
  computed from each arm's own motion and gripper width. It measures how asynchronous the arms'
  complex phases are in the data, independent of any checkpoint.
"""

import argparse

import torch

from ..bspline.bspline_core import compute_clamped_knot_vector, evaluate_bspline
from ..datasets.action_contract import normalize_actions
from ..datasets.bspline_dataset import collate_windows
from ..utils.training import window_options


def load_analysis_dataset(data_dir, task_name, ckpt, max_arms=2):
    """Rebuild the checkpoint's windows on ``data_dir`` with per-arm span targets."""
    from .xpolicylab_data import read_conversion
    from ..datasets.robocasa_adapter import load_robocasa_hdf5_dataset
    from ..datasets.robotwin_adapter import load_robotwin_hdf5_dataset

    args = ckpt["args"]
    calibration = ckpt.get("calibration", {})
    meta = read_conversion(data_dir, task_name)
    num_arms = len(meta["arms"])
    if num_arms > max_arms:
        raise ValueError("At most two arms are supported")
    options = window_options(argparse.Namespace(**args))
    options.pop("per_arm_knots", None)
    options.update(
        num_views=len(meta["cameras"]),
        motion_scale_values=calibration["motion_scale_values"],
        e_crit=calibration["e_crit"],
        # Per-arm targets are always built for two arms: they define the asynchrony index.
        per_arm_knots=num_arms == 2,
    )
    common = dict(
        data_dir=data_dir,
        task_name=task_name,
        chunk_horizon=args["chunk_horizon"],
        num_control_points=args["num_control_points"],
        degree=args["degree"],
        dataset_options=options,
    )
    if meta["source"] == "robocasa365_lerobot_v21":
        dataset = load_robocasa_hdf5_dataset(**common)
    else:
        dataset = load_robotwin_hdf5_dataset(dual_arm=num_arms == 2, **common)
    dataset.include_diagnostic_targets = False
    return dataset, list(meta["arms"])


def _per_arm_mse(values, target, num_arms):
    error = (values - target).square().reshape(*values.shape[:-1], num_arms, 7)
    return error.mean(dim=tuple(range(error.ndim - 2)) + (error.ndim - 1,))


@torch.no_grad()
def analyze_knots(
    policy,
    dataset,
    arm_names,
    max_windows=2000,
    batch_size=16,
    num_ode_steps=None,
    seed=0,
):
    """Return mean per-arm fit/prediction errors and span divergences."""
    device = next(policy.parameters()).device
    num_arms = policy.num_arms
    scale = policy.motion_scale
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator)[:max_windows].tolist()
    lo, hi = policy.supported_horizon_range or (0.0, float("inf"))
    uniform = compute_clamped_knot_vector(
        policy.num_control_points, policy.degree, device, scale.dtype
    )[None]
    sums = {}
    count = skipped = 0

    def add(name, value):
        sums[name] = sums.get(name, 0.0) + float(value)

    for start in range(0, len(order), batch_size):
        records = [dataset[i] for i in order[start : start + batch_size]]
        keep = [
            r for r in records if lo - 1e-7 <= r["horizon_duration"].item() <= hi + 1e-7
        ]
        skipped += len(records) - len(keep)
        if not keep:
            continue
        batch = collate_windows(keep)
        prediction = policy.predict_action_spline(
            batch["images"].to(device),
            batch["instructions"],
            horizon_duration=batch["horizon_duration"].to(device),
            num_ode_steps=num_ode_steps,
        )
        cp = normalize_actions(prediction["control_points"], scale)
        knots = prediction["knots"]
        delta = prediction["delta_simplex"]
        for i, record in enumerate(keep):
            target = normalize_actions(record["raw_actions"].to(device), scale)[None]
            times = record["query_times"].to(device)[None]
            pred = evaluate_bspline(times, knots[i : i + 1], cp[i : i + 1], policy.degree)
            fit_cp, _ = policy._fit_control_points(target, knots[i : i + 1], times)
            fit = evaluate_bspline(times, knots[i : i + 1], fit_cp, policy.degree)
            ucp, _ = policy._fit_control_points(target, uniform, times)
            ufit = evaluate_bspline(times, uniform, ucp, policy.degree)
            for name, values in (("pred_mse", pred), ("fit_mse", fit), ("uniform_fit_mse", ufit)):
                per_arm = _per_arm_mse(values, target, num_arms)
                for arm, value in zip(arm_names, per_arm):
                    add(f"{name}_{arm}", value)
                add(name, per_arm.mean())
            if num_arms == 2:
                spans = delta[i] if delta.ndim == 3 else delta[i].expand(2, -1)
                add("knot_span_l1", (spans[0] - spans[1]).abs().sum())
                teacher = record["target_delta"]
                if teacher.ndim == 2:
                    add("teacher_span_l1", (teacher[0] - teacher[1]).abs().sum())
            count += 1
    if not count:
        raise ValueError("No windows inside the checkpoint's supported horizon range")
    result = {name: value / count for name, value in sums.items()}
    result.update(
        windows=count,
        skipped_outside_horizon=skipped,
        knot_mode="per_arm" if getattr(policy, "per_arm_knots", False) else "shared",
        num_control_points=policy.num_control_points,
        knot_parameters=policy.knot_groups * (policy.num_control_points - policy.degree),
    )
    return result

