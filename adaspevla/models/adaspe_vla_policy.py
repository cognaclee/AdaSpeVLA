"""Knot-conditioned anchor-relative action splines with a supervised query-interval head.

One flow generates the control points of every channel: per arm, six
anchor-relative base-frame motion offsets and the gripper opening width
(manuscript Sec. joint flow generation). There is no separate gripper head.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .vlm_backbone import MultiCameraVLMBackbone
from .bspline_flow_policy import BSplineFlowMatchingPolicy
from ..bspline.bspline_core import BSplineEvaluator
from ..bspline.learnable_knots import (
    LearnableKnotPlacementHead,
    duration_tensor,
    intervals_to_knots,
)
from ..bspline.curve_fitting import fit_bspline_control_points_least_squares
from ..bspline.continuity_refit import BoundaryRefitter
from ..datasets.action_contract import (
    denormalize_actions,
    motion_channel_mask,
    normalize_actions,
    validate_motion_scales,
)


class AdaSpeVLAPolicy(nn.Module):
    def __init__(
        self,
        vlm_name="google/siglip-base-patch16-224",
        action_dim=7,
        num_control_points=8,
        degree=3,
        cond_dim=512,
        num_views=2,
        freeze_vlm=True,
        knot_loss_coeff=0.5,
        device="cpu",
        horizon_duration=0.3,
        min_query_interval=0.02,
        max_query_interval=0.5,
        min_interval_eps=0.01,
        test_mode=False,
        flow_hidden_dim=384,
        flow_num_layers=6,
        flow_num_heads=6,
        motion_scale_values=None,
        timing_probe_count=16,
        timing_num_ode_steps=15,
        supported_horizon_range=None,
        per_arm_knots=False,
        flow_loss_coeff=1.0,
        knot_supervision="kl",
        uniform_knots=False,
    ):
        super().__init__()
        if knot_supervision not in ("kl", "position"):
            raise ValueError("knot_supervision must be 'kl' or 'position'")
        if not flow_loss_coeff > 0:
            raise ValueError("flow_loss_coeff must be positive")
        # Table 1 loss weights (alpha, gamma) = (knot, flow); the gripper width is
        # a flow channel, so there is no separate gripper loss.
        self.flow_loss_coeff = float(flow_loss_coeff)
        # Manuscript default: span KL. "position" is the L_pos alternative.
        self.knot_supervision = knot_supervision
        # Ablation: fixed uniform knots instead of the learned knot head.
        self.uniform_knots = bool(uniform_knots)
        if action_dim not in (7, 14):
            raise ValueError("Require 7 or 14 action channels (six offsets and a width per arm)")
        self.action_dim, self.num_arms = action_dim, action_dim // 7
        self.motion_dim = 6 * self.num_arms
        self.gripper_channels = list(range(6, action_dim, 7))
        if per_arm_knots and self.num_arms != 2:
            raise ValueError("Per-arm knots require two arms (14 action channels)")
        # Ablation (not the manuscript default): one knot partition per arm.
        self.per_arm_knots = bool(per_arm_knots)
        self.knot_groups = self.num_arms if self.per_arm_knots else 1
        self.num_control_points, self.degree, self.cond_dim = (
            num_control_points,
            degree,
            cond_dim,
        )
        self.horizon_duration = horizon_duration
        self.knot_loss_coeff = knot_loss_coeff
        self.config = dict(
            vlm_name=vlm_name,
            action_dim=action_dim,
            num_control_points=num_control_points,
            degree=degree,
            cond_dim=cond_dim,
            num_views=num_views,
            freeze_vlm=freeze_vlm,
            knot_loss_coeff=knot_loss_coeff,
            horizon_duration=horizon_duration,
            min_query_interval=min_query_interval,
            max_query_interval=max_query_interval,
            min_interval_eps=min_interval_eps,
            test_mode=test_mode,
            flow_hidden_dim=flow_hidden_dim,
            flow_num_layers=flow_num_layers,
            flow_num_heads=flow_num_heads,
        )
        duration_tensor(horizon_duration, torch.zeros(1, 1))
        if motion_scale_values is None:
            if not test_mode:
                raise ValueError(
                    "Supply motion_scale_values calibrated on the training split"
                )
            motion_scale_values = [
                1.0
            ] * 6  # Explicit offline-fixture normalization only.
        if timing_probe_count < 2 or timing_num_ode_steps < 1:
            raise ValueError(
                "Require at least two timing probes and one integration step"
            )
        if supported_horizon_range is not None:
            lo, hi = supported_horizon_range
            if (
                not 0 < lo <= hi < float("inf")
                or not lo - 1e-7 <= horizon_duration <= hi + 1e-7
            ):
                raise ValueError("Invalid supported horizon range or inference default")
        self.supported_horizon_range = supported_horizon_range
        self.timing_probe_count = timing_probe_count
        self.timing_num_ode_steps = timing_num_ode_steps
        self.config.update(
            motion_scale_values=list(motion_scale_values),
            timing_probe_count=timing_probe_count,
            timing_num_ode_steps=timing_num_ode_steps,
            supported_horizon_range=supported_horizon_range,
        )
        if self.per_arm_knots:
            self.config["per_arm_knots"] = True
        # Non-default options only, so existing checkpoints keep their exact config.
        if self.flow_loss_coeff != 1.0:
            self.config["flow_loss_coeff"] = self.flow_loss_coeff
        if self.knot_supervision != "kl":
            self.config["knot_supervision"] = self.knot_supervision
        if self.uniform_knots:
            self.config["uniform_knots"] = True
        self.register_buffer(
            "motion_scale", validate_motion_scales(motion_scale_values, self.num_arms)
        )
        self.register_buffer(
            "motion_mask", motion_channel_mask(action_dim), persistent=False
        )
        self.vlm = MultiCameraVLMBackbone(
            vlm_name, cond_dim, num_views, freeze_vlm, test_mode=test_mode
        )
        self.knot_head = LearnableKnotPlacementHead(
            cond_dim,
            num_control_points,
            degree,
            min_query_interval=min_query_interval,
            max_query_interval=max_query_interval,
            min_interval_eps=min_interval_eps,
            num_groups=self.knot_groups,
        )
        # The generator sees the exact knots and duration (conditioning c~).
        context_dim = cond_dim + self.knot_groups * (num_control_points + degree + 1) + 1
        self.flow_policy = BSplineFlowMatchingPolicy(
            num_control_points,
            action_dim,
            context_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
            num_heads=flow_num_heads,
        )
        self.spline_evaluator = BSplineEvaluator(degree, num_control_points)
        # Features: c, log H, spans, and R probes of every motion and width channel.
        timing_dim = (
            cond_dim
            + 1
            + self.knot_groups * (num_control_points - degree)
            + timing_probe_count * action_dim
        )
        self.freq_head = nn.Sequential(
            nn.Linear(timing_dim, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        self.training_stage = "action"
        self.set_training_stage("action")

    def set_training_stage(self, stage):
        if stage not in ("action", "timing"):
            raise ValueError("Training stage must be action or timing")
        self.zero_grad(set_to_none=True)
        self.training_stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith("freq_head.")
                if stage == "timing"
                else not name.startswith("freq_head.")
            )
        if stage == "action" and self.config["freeze_vlm"]:
            # Preserve the backbone's own freeze policy; projection remains trainable.
            for name, parameter in self.vlm.named_parameters():
                if name.startswith(("vision_encoder.", "text_encoder.")):
                    parameter.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "training_stage", None) == "timing":
            for module in (self.vlm, self.knot_head, self.flow_policy):
                module.eval()
        return self

    def _sample_control_points(self, context, num_steps):
        """Euler-integrate the flow; P_0 motion is fixed to zero, then restore units."""
        normalized = self.flow_policy.sample(context, num_steps=num_steps)
        normalized[:, 0, self.motion_mask] = 0
        return denormalize_actions(normalized, self.motion_scale)

    def _timing_interval(self, cond, delta, knots, horizon, physical_cp):
        """Eqs. timing_motion_probe--query_map: probe motion and clamped widths."""
        probes = torch.linspace(
            0,
            1,
            self.timing_probe_count,
            device=physical_cp.device,
            dtype=physical_cp.dtype,
        )
        values = self.spline_evaluator(physical_cp.detach(), knots.detach(), probes)[0]
        motion = values[..., self.motion_mask]
        width = values[..., ~self.motion_mask].clamp(0, 1)
        features = torch.cat(
            (
                cond,
                horizon.log(),
                delta.detach().flatten(1),
                motion.flatten(1).detach(),
                width.flatten(1).detach(),
            ),
            dim=-1,
        )
        logit = self.freq_head(features)
        lower = horizon.clamp(max=self.knot_head.min_query_interval)
        upper = horizon.clamp(max=self.knot_head.max_query_interval)
        return lower + (upper - lower) * logit.sigmoid()

    def forward_timing_loss(self, batch):
        # No fitting and no future demonstrated motion enters this stage.
        with torch.no_grad():
            cond = self.vlm(batch["images"], instructions=batch["instructions"])
            horizon = duration_tensor(batch["horizon_duration"], cond)
            knots, delta = self._predict_knots(cond, horizon)
            context = self._context(cond, knots, horizon)
            physical_cp = self._sample_control_points(
                context, self.timing_num_ode_steps
            )
        interval = self._timing_interval(cond, delta, knots, horizon, physical_cp)
        target = batch["target_query_interval"].reshape(-1, 1)
        lower = horizon.clamp(max=self.knot_head.min_query_interval)
        upper = horizon.clamp(max=self.knot_head.max_query_interval)
        if (
            target.shape != horizon.shape
            or not torch.isfinite(target).all()
            or (target < lower - 1e-7).any()
            or (target > upper + 1e-7).any()
            or (target <= 0).any()
        ):
            raise ValueError("Invalid target query interval")
        loss = F.mse_loss(interval.log(), target.log())
        return loss, {
            "total_loss": loss.item(),
            "frequency_loss": loss.item(),
            "avg_query_hz": interval.reciprocal().mean().item(),
        }

    def _predict_knots(self, cond, horizon):
        """Learned knots (Eq. pred_spans), or uniform spans for the ablation."""
        knots, delta = self.knot_head(cond, horizon)
        if self.uniform_knots:
            delta = torch.full_like(delta, 1.0 / delta.shape[-1])
            flat = intervals_to_knots(delta.reshape(-1, delta.shape[-1]), self.degree)
            knots = flat.reshape(*delta.shape[:-1], -1)
        return knots, delta

    def _knot_loss(self, delta, target_delta):
        """alpha-weighted span supervision: KL (default) or L_pos."""
        if self.uniform_knots:
            return delta.new_zeros(())
        if self.knot_supervision == "kl":
            # PyTorch computes KL(target || prediction); epsilon=0 may underflow spans.
            # With per-arm knots, batchmean sums the arms' KL terms per sample.
            return F.kl_div(
                delta.clamp_min(torch.finfo(delta.dtype).tiny).log(),
                target_delta,
                reduction="batchmean",
            )
        # L_pos = 1/(M-1) sum_{i=1}^{M-1} (u_i - u_i^*)^2 on interior knots.
        M = delta.shape[-1]
        if M < 2:
            return delta.new_zeros(())
        u = delta.cumsum(-1)[..., :-1]
        u_star = target_delta.cumsum(-1)[..., :-1]
        per_group = (u - u_star).square().mean(-1)
        return per_group.reshape(delta.shape[0], -1).sum(-1).mean()

    def _context(self, cond, knots, horizon):
        return torch.cat((cond, knots.flatten(1), horizon.log()), dim=-1)

    def _fit_control_points(self, motion, knots, query_times):
        """Fit on the shared partition, or each arm's seven channels on its own partition."""
        if knots.ndim == 2:
            return fit_bspline_control_points_least_squares(
                motion,
                self.num_control_points,
                self.degree,
                knots=knots,
                query_times=query_times,
            )
        fits = [
            fit_bspline_control_points_least_squares(
                block,
                self.num_control_points,
                self.degree,
                knots=knots[:, arm],
                query_times=query_times,
            )
            for arm, block in enumerate(motion.chunk(knots.shape[1], dim=-1))
        ]
        return (
            torch.cat([cp for cp, _ in fits], dim=-1),
            torch.stack([error for _, error in fits], -1).mean(-1),
        )

    def forward_loss(self, batch):
        """Fit labels on current predicted knots; train flow (motion+width) and spans."""
        if self.training_stage == "timing":
            return self.forward_timing_loss(batch)
        cond = self.vlm(batch["images"], instructions=batch["instructions"])
        horizon = duration_tensor(batch["horizon_duration"], cond)
        knots, delta = self._predict_knots(cond, horizon)
        # Fit each timestamped window at its actual length; never pad the fit.
        actions = batch["raw_actions"]
        times = batch["query_times"]
        targets, errors = [], []
        with torch.no_grad():
            for index, action in enumerate(actions):
                if action.shape[-1] != self.action_dim:
                    raise ValueError(
                        "raw_actions must contain six offsets and a width per arm"
                    )
                cp, error = self._fit_control_points(
                    normalize_actions(action, self.motion_scale).unsqueeze(0),
                    knots[index : index + 1].detach(),
                    times[index].unsqueeze(0),
                )
                targets.append(cp)
                errors.append(error)
            target_cp, recon = torch.cat(targets), torch.cat(errors)
        context = self._context(cond, knots, horizon)
        flow_loss, _ = self.flow_policy.compute_loss(target_cp, context)
        # Span targets are required for the knot term of the action loss.
        target_delta = batch.get("target_delta")
        if (
            target_delta is None
            or target_delta.shape != delta.shape
            or not torch.isfinite(target_delta).all()
            or (target_delta <= 0).any()
            or not torch.allclose(
                target_delta.sum(-1), torch.ones_like(delta[..., 0]), atol=1e-5
            )
        ):
            raise ValueError(
                "target_delta must be a positive normalized span distribution"
            )
        knot_loss = self._knot_loss(delta, target_delta)
        # Eq. action_loss: gamma L_flow + alpha KL; no separate gripper term.
        total = self.flow_loss_coeff * flow_loss + self.knot_loss_coeff * knot_loss
        logs = {
            name: value.detach().item()
            for name, value in dict(
                total_loss=total,
                flow_loss=flow_loss,
                knot_loss=knot_loss,
                reconstruction_mse=recon.mean(),
                avg_horizon_s=horizon.mean(),
            ).items()
        }
        return total, logs

    @torch.no_grad()
    def predict_action_spline(
        self,
        images,
        instructions=None,
        horizon_duration=None,
        boundary_value=None,
        num_ode_steps=None,
    ):
        """Return physical control points of anchor-relative offsets and widths.

        Per arm, channels are six base-frame offsets from the anchor pose observed
        at query time (metres, axis-angle radians) and the gripper width; widths
        are not clamped here (execution clamps to [0,1]). The optional boundary
        value (B, arms) is the last emitted clamped width (Eq. boundary); it
        replaces the width entry of control point zero only. The caller chooses a
        trained horizon.
        """
        self.eval()
        cond = self.vlm(images, instructions=instructions)
        horizon = duration_tensor(
            self.horizon_duration if horizon_duration is None else horizon_duration,
            cond,
        )
        if self.supported_horizon_range is not None:
            lo, hi = self.supported_horizon_range
            if (horizon < lo - 1e-7).any() or (horizon > hi + 1e-7).any():
                raise ValueError("Requested horizon is outside training support")
        if num_ode_steps is None:
            num_ode_steps = self.timing_num_ode_steps
        knots, delta = self._predict_knots(cond, horizon)
        context = self._context(cond, knots, horizon)
        cp = self._sample_control_points(context, num_ode_steps)
        if boundary_value is not None:
            cp = BoundaryRefitter.refit_boundary_control_points(
                cp,
                knots,
                boundary_value,
                self.degree,
                channels=self.gripper_channels,
            )
        interval = self._timing_interval(cond, delta, knots, horizon, cp)
        return {
            "control_points": cp,
            "knots": knots,
            "delta_simplex": delta,
            "horizon_duration": horizon,
            "query_interval": interval,
            "query_hz": interval.reciprocal(),
            "action_type": "anchor_delta",
            "action_frame": "base",
        }
