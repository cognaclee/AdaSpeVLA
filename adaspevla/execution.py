"""Sequential policy prediction and spline-prefix execution."""

import math
from typing import Callable, Dict, Tuple

import numpy as np
import torch

from .benchmarks.pose_math import base_rotation, delta_to_target, normalized_pose
from .bspline.bspline_core import evaluate_bspline


class SynchronousSplineExecutor:
    """Call the policy, execute a prefix, then observe and predict again.

    Each ``step`` emits one action. Policy inference blocks that step when a new
    prefix is needed. ``control_dt`` is the action sampling period in execution
    time; callers must advance their environment or pace their controller.
    Inference time is additional wall-clock time, not part of the sampled prefix.

    Step n of a prefix evaluates the spline at s=(n+1)*control_dt/H, the end of
    its control period (the offset at s=0 is zero by construction). Per arm the
    spline holds six base-frame offsets from the anchor pose observed at query
    time and the gripper width, which is clamped to [0,1] and, for
    ``gripper_mode='binary'``, thresholded at ``gripper_threshold``.

    ``command_type='anchor_delta'`` emits per arm ``[offset(6), width]``.
    ``command_type='pose'`` reads the anchor pose from ``observation['anchor_pose']``
    (arms, 7: xyz + wxyz) at each query and emits per arm the absolute target
    ``[x,y,z,qw,qx,qy,qz, width]`` by Eq. delta_to_target, using the per-arm
    base rotations ``world_from_base`` (identity by default). Controllers that
    need increments or velocities derive them from these targets.
    """

    def __init__(
        self,
        policy,
        observation_callback: Callable[[], Dict],
        command_callback: Callable[[torch.Tensor], None],
        control_dt: float = 0.02,
        num_ode_steps: int = 15,
        command_type: str = "anchor_delta",
        gripper_mode: str = "continuous",
        gripper_threshold: float = 0.5,
        world_from_base=None,
        boundary_matching: bool = False,
        motion_limits=None,
        stop_callback=None,
        fault_callback=None,
    ):
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if num_ode_steps < 1:
            raise ValueError("num_ode_steps must be positive")
        if command_type not in ("anchor_delta", "pose"):
            raise ValueError("command_type must be anchor_delta or pose")
        if gripper_mode not in ("continuous", "binary"):
            raise ValueError("gripper_mode must be continuous or binary")
        if not 0 < gripper_threshold < 1:
            raise ValueError("gripper_threshold must lie strictly between 0 and 1")
        self.boundary_matching = boundary_matching
        self.motion_limits = (
            None if motion_limits is None else torch.as_tensor(motion_limits)
        )
        if self.motion_limits is not None and (
            not torch.isfinite(self.motion_limits).all()
            or (self.motion_limits <= 0).any()
        ):
            raise ValueError("motion_limits must be positive and finite")
        self.world_from_base = (
            None
            if world_from_base is None
            else [
                q if hasattr(q, "apply") else base_rotation(q) for q in world_from_base
            ]
        )
        self.stop_callback, self.fault_callback = stop_callback, fault_callback
        self.command_type = command_type
        self.gripper_mode, self.gripper_threshold = gripper_mode, gripper_threshold
        self.policy = policy
        self.observation_callback = observation_callback
        self.command_callback = command_callback
        self.control_dt = control_dt
        self.num_ode_steps = num_ode_steps
        self.reset()

    def reset(self):
        """Discard the previous episode's prefix and counters."""
        self.control_points = None
        self.knots = None
        self.anchor = None
        self.horizon = 0.0
        self.prefix_steps = 0
        self.step_index = 0
        self.query_count = 0
        self.last_width = None

    def _query(self):
        observation = dict(self.observation_callback())
        anchor = observation.pop("anchor_pose", None)
        if self.boundary_matching and self.last_width is not None:
            reference = observation["images"]
            observation["boundary_value"] = self.last_width.to(reference).unsqueeze(0)
        prediction = self.policy.predict_action_spline(
            **observation, num_ode_steps=self.num_ode_steps
        )
        horizon = prediction["horizon_duration"].item()
        interval = prediction["query_interval"].item()
        if not all(math.isfinite(x) and x > 0 for x in (horizon, interval)):
            raise ValueError(
                "Predicted horizon and query interval must be finite and positive"
            )
        if horizon < self.control_dt and not math.isclose(
            horizon, self.control_dt, rel_tol=1e-7
        ):
            raise ValueError("Prediction horizon must be at least control_dt")
        if prediction.get("action_type", "anchor_delta") != "anchor_delta" or (
            prediction.get("action_frame", "base") != "base"
        ):
            raise ValueError("Executor requires anchor-relative base-frame offsets")
        cp = prediction["control_points"].detach().cpu()
        knots = prediction["knots"].detach().cpu()
        if (
            cp.ndim != 3
            or cp.shape[0] != 1
            or cp.shape[-1] % 7
            or knots.ndim not in (2, 3)
            or knots.shape[0] != 1
        ):
            raise ValueError(
                "Execution requires a single-observation batch with seven channels per arm"
            )
        if knots.ndim == 3 and cp.shape[-1] % knots.shape[1]:
            raise ValueError("Per-arm knot groups must split the action channels")
        if not torch.isfinite(cp).all() or not torch.isfinite(knots).all():
            raise ValueError("Spline parameters must be finite")
        arms = cp.shape[-1] // 7
        if self.command_type == "pose":
            if anchor is None:
                raise ValueError("Pose commands require observation['anchor_pose']")
            anchor = np.asarray(anchor, dtype=np.float64).reshape(arms, 7)
            self.anchor = np.stack([normalized_pose(pose) for pose in anchor])
            if self.world_from_base is None:
                self.world_from_base = [base_rotation((1.0, 0.0, 0.0, 0.0))] * arms
            if len(self.world_from_base) != arms:
                raise ValueError("Supply one base rotation per arm")
        self.control_points = cp
        self.knots = knots
        self.horizon = horizon

        # Replan on the first control tick at or after the requested interval,
        # or the horizon if it ends first. Never sample beyond the horizon.
        def snap(value):
            return (
                round(value)
                if math.isclose(value, round(value), rel_tol=1e-7, abs_tol=1e-9)
                else value
            )

        horizon_ticks = math.floor(snap(horizon / self.control_dt))
        query_ticks = math.ceil(snap(interval / self.control_dt))
        self.prefix_steps = max(1, min(horizon_ticks, query_ticks))
        self.step_index = 0
        self.query_count += 1

    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, bool]:
        """Return ``(action, queried)`` after one sequential command callback."""
        if (self.stop_callback and self.stop_callback()) or (
            self.fault_callback and self.fault_callback()
        ):
            raise StopIteration("Task terminated or controller fault")
        queried = self.control_points is None or self.step_index >= self.prefix_steps
        if queried:
            self._query()
        # Step n targets the end of its control period; prefix_steps <= floor(H/dt).
        s = min((self.step_index + 1) * self.control_dt / self.horizon, 1.0)
        u = torch.tensor([[s]], dtype=self.control_points.dtype)
        # Shared (1,n) knots or one partition per arm (1,arms,n).
        sample = evaluate_bspline(
            u, self.knots, self.control_points, self.policy.degree
        )[0, 0]
        arms = sample.reshape(-1, 7)
        offsets, width = arms[:, :6], arms[:, 6].clamp(0, 1)
        if self.motion_limits is not None:
            if (
                self.motion_limits.shape != (offsets.numel(),)
                or (offsets.flatten().abs() > self.motion_limits).any()
            ):
                raise ValueError("Planned motion exceeds controller limits")
        self.last_width = width.clone()
        if self.gripper_mode == "binary":
            width = (width >= self.gripper_threshold).to(width)
        if self.command_type == "pose":
            targets = [
                delta_to_target(anchor, offset.double().numpy(), rotation)
                for anchor, offset, rotation in zip(
                    self.anchor, offsets, self.world_from_base
                )
            ]
            motion = torch.tensor(np.stack(targets), dtype=sample.dtype)
        else:
            motion = offsets
        action = torch.cat((motion, width[:, None]), dim=-1).flatten()
        self.command_callback(action)
        self.step_index += 1
        if (self.stop_callback and self.stop_callback()) or (
            self.fault_callback and self.fault_callback()
        ):
            raise StopIteration("Task terminated or controller fault after command")
        return action, queried
