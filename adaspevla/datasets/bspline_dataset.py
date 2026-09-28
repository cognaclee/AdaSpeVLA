"""Timestamped velocity demonstrations with consistent knot and timing targets."""

import numpy as np
import torch
from torch.utils.data import Dataset
from .action_contract import (
    canonicalize_episode,
    calibrate_motion_scales,
    validate_motion_scales,
)
from ..bspline.curve_fitting import fit_bspline_control_points_least_squares
from ..bspline.learnable_knots import (
    compute_curvature_density_target,
    intervals_to_knots,
    query_interval_target,
)


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
    ):
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
        self.include_diagnostic_targets = True
        self.samples = []
        prepared = []
        for original in episodes:
            ep = dict(original)
            motion, gripper, times = canonicalize_episode(ep, action_dim)
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
            for key in ("images", "criticality", "query_intervals"):
                if key in ep:
                    ep[key] = np.asarray(ep[key])[:end]
            prepared.append((ep, motion[:end], gripper[:end], times[:end]))
        if not prepared:
            raise ValueError("No complete episodes before termination")
        base_scales = (
            calibrate_motion_scales([x[1] for x in prepared], scale_floor)
            if motion_scale_values is None
            else torch.as_tensor(motion_scale_values)
        )
        self.motion_scale_values = base_scales.tolist()
        self.motion_scale = validate_motion_scales(base_scales, action_dim // 7)
        rates = np.concatenate(
            [
                np.linalg.norm(
                    np.diff(m / self.motion_scale.numpy(), axis=0)
                    / np.diff(t)[:, None],
                    axis=-1,
                )
                for _, m, _, t in prepared
            ]
        )
        self.e_crit = float(
            max(np.quantile(rates, event_quantile), 1e-6) if e_crit is None else e_crit
        )
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
            action_frame="base",
            action_type="twist",
            motion_channels=["vx", "vy", "vz", "wx", "wy", "wz"],
            motion_units=["m/s"] * 3 + ["rad/s"] * 3,
        )
        for ep, motion, gripper, times in prepared:
            images = np.asarray(ep["images"])
            if len(images) != len(motion):
                raise ValueError("Image and action sample counts must match")
            instruction = ep.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Each episode requires a nonempty instruction")
            criticality = np.asarray(
                ep.get("criticality", np.zeros(len(motion))), dtype=np.float32
            )
            if (
                criticality.shape != (len(motion),)
                or not np.isfinite(criticality).all()
                or ((criticality < 0) | (criticality > 1)).any()
            ):
                raise ValueError("criticality must be length T and in [0,1]")
            labels = ep.get("query_intervals")
            if labels is not None:
                labels = np.asarray(labels, dtype=np.float32)
                if labels.shape != (len(motion),):
                    raise ValueError(
                        "query_intervals must be length T; invalid values use fallback labels"
                    )
            for start in range(0, len(motion) - 1, anchor_stride):
                if supported_horizons is None:
                    stops = (
                        [start + chunk_horizon]
                        if start + chunk_horizon <= len(motion)
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
                    self.samples.append(
                        (
                            images[start],
                            motion[start:stop],
                            gripper[start:stop],
                            times[start:stop] - times[start],
                            criticality[start:stop],
                            instruction,
                            label,
                        )
                    )
        if not self.samples:
            raise ValueError(
                "No complete timestamped chunks; check the dataset and chunk_horizon"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, motion, gripper, times, criticality, instruction, label = self.samples[
            index
        ]
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
        raw = torch.tensor(motion)
        grip = torch.tensor(gripper)
        stamps = torch.tensor(times, dtype=torch.float32)
        risk = torch.tensor(criticality)
        horizon = stamps[-1:].clone()
        normalized = raw / self.motion_scale
        delta = compute_curvature_density_target(
            normalized[None],
            self.num_control_points - self.degree,
            stamps[None],
            self.min_interval_eps,
            criticality=risk[None],
        )[0]
        knots = intervals_to_knots(delta[None], self.degree)
        query_times = stamps / horizon
        if self.include_diagnostic_targets:
            cp, _ = fit_bspline_control_points_least_squares(
                normalized[None],
                self.num_control_points,
                self.degree,
                knots=knots,
                query_times=query_times[None],
            )
        target_interval = query_interval_target(
            delta[None],
            horizon,
            self.min_query_interval,
            self.max_query_interval,
            stamps[None],
            risk[None],
            grip[None],
            normalized[None],
            e_crit=self.e_crit,
            anticipation_margin=self.anticipation_margin,
        )[0]
        if label is not None:
            lower, upper = min(horizon.item(), self.min_query_interval), min(
                horizon.item(), self.max_query_interval
            )
            if not lower - 1e-7 <= label <= upper + 1e-7:
                raise ValueError(
                    "Measured query interval lies outside configured horizon/interval bounds"
                )
            target_interval = torch.tensor([label])
        # Discrete gripper labels are held, never interpolated or included in the spline.
        grid = torch.linspace(0, 1, self.chunk_horizon)
        indices = (
            torch.searchsorted(query_times, grid, right=True)
            .sub(1)
            .clamp(0, len(grip) - 1)
        )
        record = {
            "images": image.contiguous(),
            "raw_actions": raw,
            "target_delta": delta,
            "target_knots": knots[0],
            "query_times": query_times,
            "horizon_duration": horizon,
            "target_query_interval": target_interval,
            "target_gripper": grip[indices],
            "instructions": instruction,
            "timing_target_source": "measured" if label is not None else "heuristic",
        }
        if self.include_diagnostic_targets:
            record["control_points"] = cp[0]
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
