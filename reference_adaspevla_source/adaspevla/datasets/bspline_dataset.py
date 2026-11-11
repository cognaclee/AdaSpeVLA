"""Timestamped pose demonstrations as anchor-relative windows with knot and timing targets."""

import numpy as np
import torch
from torch.utils.data import Dataset
from .action_contract import (
    canonicalize_episode,
    calibrate_motion_scales,
    normalize_actions,
    validate_motion_scales,
    window_offsets,
)
from ..benchmarks.pose_math import PROPRIO_PER_ARM, proprio_features
from ..bspline.curve_fitting import fit_bspline_control_points_grouped
from ..bspline.learnable_knots import (
    compute_curvature_density_target,
    intervals_to_knots,
    motion_variation,
    query_interval_target,
)

# A row is idle when no arm moved more than these since the previous row.
IDLE_TRANSLATION_M, IDLE_ROTATION_RAD, IDLE_WIDTH = 1e-3, 1e-2, 1e-3


def idle_keep_mask(poses, widths, max_idle_rows):
    """Rows to keep when every run of idle rows is shortened to ``max_idle_rows``.

    poses (T, arms, 7) xyz+wxyz, widths (T, arms). The first row and every row
    that moves are kept; within an idle run only its first ``max_idle_rows``.
    """
    moved = np.linalg.norm(np.diff(poses[..., :3], axis=0), axis=-1)
    dot = np.abs(np.sum(poses[1:, :, 3:] * poses[:-1, :, 3:], axis=-1)).clip(0, 1)
    turned = 2 * np.arccos(dot)
    changed = np.abs(np.diff(widths, axis=0))
    idle = (
        (moved < IDLE_TRANSLATION_M) & (turned < IDLE_ROTATION_RAD) & (changed < IDLE_WIDTH)
    ).all(axis=1)
    keep = np.ones(len(poses), dtype=bool)
    run = 0
    for row, still in enumerate(idle, start=1):
        run = run + 1 if still else 0
        keep[row] = run <= max_idle_rows
    return keep


class UniversalBSplineDataset(Dataset):
    def __init__(
        self,
        episodes,
        chunk_horizon=16,
        num_control_points=8,
        degree=3,
        action_dim=7,
        num_views=2,
        min_interval_eps=0.01,
        min_query_interval=0.02,
        max_query_interval=0.5,
        supported_horizons=None,
        control_dt=0.02,
        anchor_stride=1,
        motion_scale_values=None,
        e_crit=None,
        event_quantile=0.95,
        anticipation_margin=0.0,
        scale_floor=1e-6,
        per_arm_knots=False,
        gripper_density_gain=10.0,
        gripper_threshold=0.5,
        pad_episode_end=False,
        max_idle_rows=None,
    ):
        if max_idle_rows is not None and int(max_idle_rows) < 0:
            raise ValueError("max_idle_rows must be nonnegative")
        if per_arm_knots and action_dim != 14:
            raise ValueError("Per-arm knots require two arms (14 action channels)")
        # Ablation: one span target per arm, from that arm's own motion.
        self.per_arm_knots = bool(per_arm_knots)
        if chunk_horizon < 2 or num_control_points <= degree or degree < 0:
            raise ValueError("Invalid chunk or spline dimensions")
        self.chunk_horizon, self.num_control_points, self.degree = (
            chunk_horizon,
            num_control_points,
            degree,
        )
        self.action_dim, self.num_views = action_dim, num_views
        self.min_interval_eps = min_interval_eps
        self.min_query_interval, self.max_query_interval = (
            min_query_interval,
            max_query_interval,
        )
        if not np.isfinite(control_dt) or control_dt <= 0 or anchor_stride < 1:
            raise ValueError("Require positive control_dt and anchor_stride")
        if not 0 < event_quantile <= 1 or not 0 <= anticipation_margin < float("inf"):
            raise ValueError("Invalid event calibration or anticipation margin")
        if supported_horizons is not None:
            supported_horizons = sorted(set(float(h) for h in supported_horizons))
            if not supported_horizons or any(
                not np.isfinite(h) or h < control_dt for h in supported_horizons
            ):
                raise ValueError(
                    "Supported horizons must be finite and at least control_dt"
                )
        self.supported_horizons = supported_horizons
        self.control_dt = control_dt
        self.anticipation_margin = anticipation_margin
        if not 0 <= gripper_density_gain < float("inf") or not 0 < gripper_threshold < 1:
            raise ValueError("Invalid gripper density gain or gripper threshold")
        # beta_g (Eq. density) and theta_g (event teacher; binary robots only).
        self.gripper_density_gain = float(gripper_density_gain)
        self.gripper_threshold = float(gripper_threshold)
        self.num_arms = action_dim // 7
        self.include_diagnostic_targets = True
        self.samples = []
        prepared = []
        self.action_frame = None
        idle_rows_removed = 0
        for original in episodes:
            ep = dict(original)
            poses, widths, times, world_from_base = canonicalize_episode(ep, action_dim)
            if self.action_frame is not None and ep["action_frame"] != self.action_frame:
                raise ValueError("Cannot mix base and episode_reset coordinate systems")
            self.action_frame = ep["action_frame"]
            if "observed_ee_poses" in ep or "observed_gripper" in ep:
                if "observed_ee_poses" not in ep or "observed_gripper" not in ep:
                    raise ValueError("Separate observed state requires both poses and gripper")
                observed_ep = dict(ep, ee_poses=ep["observed_ee_poses"],
                                   gripper=ep["observed_gripper"])
                observed_poses, observed_widths, observed_times, _ = canonicalize_episode(observed_ep, action_dim)
                if not np.array_equal(times, observed_times):
                    raise ValueError("Observed states and targets must share timestamps")
                ep["observed_ee_poses"] = observed_poses
                ep["observed_gripper"] = observed_widths
            # A terminal command may be the endpoint; never select later samples.
            end = len(times)
            for key in ("terminated", "dones"):
                if key in ep:
                    flags = np.asarray(ep[key])
                    if flags.shape != (len(times),) or not np.isin(flags, [0, 1]).all():
                        raise ValueError(
                            "Termination flags must be binary length-T arrays"
                        )
                    indices = np.flatnonzero(flags)
                    if len(indices):
                        end = min(end, int(indices[0]) + 1)
            if "termination_time" in ep:
                terminal = float(ep["termination_time"])
                if not np.isfinite(terminal) or terminal < times[0]:
                    raise ValueError("Invalid episode termination time")
                end = min(end, int(np.searchsorted(times, terminal, side="right")))
            if end < 2:
                continue
            for key in ("images", "criticality", "query_intervals", "observed_ee_poses", "observed_gripper"):
                if key in ep:
                    value = ep[key]
                    ep[key] = value[:end] if getattr(value, 'lazy_frames', False) else np.asarray(value)[:end]
            poses, widths, times = poses[:end], widths[:end], times[:end]
            if max_idle_rows is not None:
                keep = idle_keep_mask(poses, widths, int(max_idle_rows))
                # Close the time gaps left by dropped rows; motion timing is unchanged.
                step = np.diff(times, prepend=times[0])
                times = (times - np.cumsum(np.where(keep, 0.0, step)))[keep]
                poses, widths = poses[keep], widths[keep]
                for key in ("images", "criticality", "query_intervals", "observed_ee_poses", "observed_gripper"):
                    if key in ep:
                        ep[key] = ep[key][keep]
                idle_rows_removed += int((~keep).sum())
            prepared.append((ep, poses, widths, times, world_from_base))
        if not prepared:
            raise ValueError("No complete episodes before termination")
        conversions = [
            ep["conversion_metadata"]
            for ep, *_ in prepared
            if "conversion_metadata" in ep
        ]
        windows = []
        for ep, poses, widths, times, world_from_base in prepared:
            images = ep['images'] if getattr(ep['images'], 'lazy_frames', False) else np.asarray(ep['images'])
            if len(images) != len(poses):
                raise ValueError("Image and action sample counts must match")
            instruction = ep.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Each episode requires a nonempty instruction")
            criticality = np.asarray(
                ep.get("criticality", np.zeros(len(poses))), dtype=np.float32
            )
            if (
                criticality.shape != (len(poses),)
                or not np.isfinite(criticality).all()
                or ((criticality < 0) | (criticality > 1)).any()
            ):
                raise ValueError("criticality must be length T and in [0,1]")
            labels = ep.get("query_intervals")
            if labels is not None:
                labels = np.asarray(labels, dtype=np.float32)
                if labels.shape != (len(poses),):
                    raise ValueError(
                        "query_intervals must be length T; invalid values use fallback labels"
                    )
            num_samples = len(poses)
            if pad_episode_end and num_samples > 1:
                # Demos stop when the task succeeds. Without padding the last
                # chunk_horizon-1 samples (the motion that completes the task) are
                # never anchors; hold the final pose and width after the end instead.
                pad = chunk_horizon - 1
                dt = float(np.median(np.diff(times)))
                poses = np.concatenate((poses, np.repeat(poses[-1:], pad, axis=0)))
                widths = np.concatenate((widths, np.repeat(widths[-1:], pad, axis=0)))
                times = np.concatenate((times, times[-1] + dt * np.arange(1, pad + 1)))
                criticality = np.concatenate((criticality, np.repeat(criticality[-1:], pad)))
            for start in range(0, num_samples - 1, anchor_stride):
                if supported_horizons is None:
                    stops = (
                        [start + chunk_horizon]
                        if start + chunk_horizon <= len(poses)
                        else []
                    )
                else:
                    stops = []
                    for h in supported_horizons:
                        if times[start] + h > times[-1] + 1e-9:
                            continue
                        endpoint = (
                            start
                            + 1
                            + int(
                                np.argmin(
                                    np.abs(times[start + 1 :] - (times[start] + h))
                                )
                            )
                        )
                        duration = times[endpoint] - times[start]
                        if (
                            supported_horizons[0] - 1e-9
                            <= duration
                            <= supported_horizons[-1] + 1e-9
                        ):
                            stops.append(endpoint + 1)
                for stop in sorted(set(stops)):
                    duration = times[stop - 1] - times[start]
                    if duration < control_dt - 1e-9:
                        continue
                    label = None if labels is None else labels[start]
                    # Invalid measured intervals fall back to the event teacher.
                    if label is not None and not (
                        np.isfinite(label)
                        and label > 0
                        and min(duration, min_query_interval) - 1e-7
                        <= label
                        <= min(duration, max_query_interval) + 1e-7
                    ):
                        label = None
                    # Eq. anchor_delta: every sample is re-anchored at T_k = times[start].
                    anchor_pose = ep.get("observed_ee_poses", poses)[start]
                    anchor_width = ep.get("observed_gripper", widths)[start]
                    target_poses = poses[start:stop].copy()
                    target_widths = widths[start:stop].copy()
                    # At t=0 the curve starts at the measured state; subsequent
                    # rows are commanded targets for t+dt, not measured arrivals.
                    target_poses[0] = anchor_pose
                    target_widths[0] = anchor_width
                    offsets = window_offsets(target_poses, world_from_base)
                    raw = np.concatenate(
                        (
                            offsets.reshape(stop - start, self.num_arms, 6),
                            target_widths[:, :, None],
                        ),
                        axis=-1,
                    ).reshape(stop - start, -1)
                    windows.append(
                        (
                            images[start],
                            # float64 keeps second differences above rounding noise.
                            raw,
                            times[start:stop] - times[start],
                            criticality[start:stop],
                            instruction,
                            label,
                            # Explicit measured state when provided by a converter.
                            proprio_features(
                                anchor_pose, anchor_width, world_from_base
                            ),
                        )
                    )
        if not windows:
            raise ValueError(
                "No complete timestamped chunks; check the dataset and chunk_horizon"
            )
        motion_mask = np.ones(action_dim, dtype=bool)
        motion_mask[6::7] = False
        base_scales = (
            calibrate_motion_scales([w[1][:, motion_mask] for w in windows], scale_floor)
            if motion_scale_values is None
            else torch.as_tensor(motion_scale_values)
        )
        self.motion_scale_values = base_scales.tolist()
        self.motion_scale = validate_motion_scales(base_scales, self.num_arms)
        if e_crit is None:
            variation = np.concatenate(
                [
                    motion_variation(
                        torch.from_numpy(w[1][:, motion_mask])[None]
                        / self.motion_scale.double(),
                        torch.from_numpy(w[2])[None],
                    )[0].numpy()
                    for w in windows
                ]
            )
            e_crit = max(float(np.quantile(variation, event_quantile)), 1e-6)
        self.e_crit = float(e_crit)
        # Standardize positions only; 6-D rotations and widths are already O(1).
        proprio = np.stack([w[6] for w in windows]).reshape(-1, self.num_arms, PROPRIO_PER_ARM)
        proprio_mean = np.zeros((self.num_arms, PROPRIO_PER_ARM))
        proprio_std = np.ones((self.num_arms, PROPRIO_PER_ARM))
        proprio_mean[:, :3] = proprio[..., :3].mean(0)
        proprio_std[:, :3] = np.maximum(proprio[..., :3].std(0), 1e-3)
        if not 0 < self.e_crit < float("inf"):
            raise ValueError("e_crit must be finite and positive")
        self.calibration = dict(
            motion_scale_values=self.motion_scale_values,
            scale_percentile=95,
            scale_floor=scale_floor,
            e_crit=self.e_crit,
            event_quantile=event_quantile,
            anticipation_margin=anticipation_margin,
            control_dt=control_dt,
            supported_horizons=supported_horizons,
            action_frame=self.action_frame,
            action_type="anchor_delta",
            motion_channels=["dx", "dy", "dz", "rx", "ry", "rz"],
            motion_units=["m"] * 3 + ["rad"] * 3,
            gripper_channel="width",
            gripper_units="normalized [0,1], 0 closed, 1 open",
            channel_order="per arm: six anchor-relative offsets, then gripper width",
            gripper_density_gain=self.gripper_density_gain,
            gripper_threshold=self.gripper_threshold,
            proprio_mean=proprio_mean.reshape(-1).tolist(),
            proprio_std=proprio_std.reshape(-1).tolist(),
        )
        if self.per_arm_knots:
            self.calibration["per_arm_knots"] = True
        if pad_episode_end:
            self.calibration["pad_episode_end"] = True
        if max_idle_rows is not None:
            self.calibration.update(
                max_idle_rows=int(max_idle_rows), idle_rows_removed=idle_rows_removed
            )
        if conversions:
            if any(meta != conversions[0] for meta in conversions):
                raise ValueError("Native conversion conventions differ across episodes")
            self.calibration["source_conversion"] = conversions[0]
        self.samples = windows
        if not self.samples:
            raise ValueError(
                "No complete timestamped chunks; check the dataset and chunk_horizon"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, raw, times, criticality, instruction, label, proprio = self.samples[index]
        image = torch.as_tensor(np.array(image, copy=True))
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("Images must be (views,H,W,3) or (views,3,H,W)")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[:2] != (self.num_views, 3):
            raise ValueError("Camera count and RGB channels must match configuration")
        image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
        if not torch.isfinite(image).all() or ((image < 0) | (image > 1)).any():
            raise ValueError("Images must be uint8 RGB or floating RGB in [0,1]")
        # Targets are computed in float64 and returned in float32.
        raw = torch.from_numpy(raw)
        stamps = torch.from_numpy(np.asarray(times, dtype=np.float64))
        risk = torch.from_numpy(criticality).double()
        horizon = stamps[-1:].clone()
        # Normalized joint action: motion / D and gripper 2w-1, seven channels per arm.
        normalized = normalize_actions(raw, self.motion_scale.double())
        arms = normalized.unflatten(-1, (self.num_arms, 7))
        motion = arms[..., :6].flatten(-2)
        widths = raw.unflatten(-1, (self.num_arms, 7))[..., 6]
        if self.per_arm_knots:
            blocks = [(arms[:, a, :6], widths[:, a : a + 1]) for a in range(self.num_arms)]
        else:
            blocks = [(motion, widths)]
        delta = torch.stack(
            [
                compute_curvature_density_target(
                    block_motion[None],
                    self.num_control_points - self.degree,
                    stamps[None],
                    self.min_interval_eps,
                    criticality=risk[None],
                    gripper=block_width[None],
                    gripper_gain=self.gripper_density_gain,
                )[0]
                for block_motion, block_width in blocks
            ]
        )
        # Shared: delta (I,), knots (1,n). Per arm: delta (2,I), knots (1,2,n).
        knots = intervals_to_knots(delta, self.degree)
        if self.per_arm_knots:
            knots = knots[None]
        else:
            delta = delta[0]
        query_times = stamps / horizon
        if self.include_diagnostic_targets:
            # All channels, including the gripper width, share each arm's basis.
            cp = fit_bspline_control_points_grouped(
                normalized[None],
                self.num_control_points,
                self.degree,
                knots=knots,
                query_times=query_times[None],
            )[0]
        target_interval = query_interval_target(
            # Span lengths only enter validation; the label comes from events.
            delta.reshape(-1, delta.shape[-1])[:1],
            horizon,
            self.min_query_interval,
            self.max_query_interval,
            stamps[None],
            risk[None],
            widths[None],
            motion[None],
            e_crit=self.e_crit,
            anticipation_margin=self.anticipation_margin,
            gripper_threshold=self.gripper_threshold,
        )[0]
        if label is not None:
            lower, upper = min(horizon.item(), self.min_query_interval), min(
                horizon.item(), self.max_query_interval
            )
            if not lower - 1e-7 <= label <= upper + 1e-7:
                raise ValueError(
                    "Measured query interval lies outside configured horizon/interval bounds"
                )
            target_interval = torch.tensor([label], dtype=torch.float64)
        record = {
            "images": image.contiguous(),
            "raw_actions": raw.float(),
            "target_delta": delta.float(),
            "target_knots": knots[0].float(),
            "query_times": query_times.float(),
            "horizon_duration": horizon.float(),
            "target_query_interval": target_interval.float(),
            "instructions": instruction,
            "proprio": torch.from_numpy(proprio),
            "timing_target_source": "measured" if label is not None else "heuristic",
        }
        if self.include_diagnostic_targets:
            record["control_points"] = cp[0].float()
        return record


def collate_windows(records):
    """Keep irregular fitting samples ragged; fixed grids and images are stacked."""
    from torch.utils.data import default_collate

    ragged = {"raw_actions", "query_times"}
    return {
        key: (
            [r[key] for r in records]
            if key in ragged
            else default_collate([r[key] for r in records])
        )
        for key in records[0]
    }
