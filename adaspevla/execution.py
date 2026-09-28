"""Sequential policy prediction and spline-prefix execution."""

import math
from typing import Callable, Dict, Tuple

import torch

from .bspline.bspline_core import eval_cox_de_boor_basis


class SynchronousSplineExecutor:
    """Call the policy, execute a prefix, then observe and predict again.

    Each ``step`` emits one action. Policy inference blocks that step when a new
    prefix is needed. ``control_dt`` is the action sampling period in execution
    time; callers must advance their environment or pace their controller.
    Inference time is additional wall-clock time, not part of the sampled prefix.
    Motion values are physical base-frame twists. command_type='delta_twist'
    multiplies motion by control_dt; grippers remain discrete signed commands.
    Device-specific normalization or robot-frame conversion belongs in the callback.
    """

    def __init__(
        self,
        policy,
        observation_callback: Callable[[], Dict],
        command_callback: Callable[[torch.Tensor], None],
        control_dt: float = 0.02,
        num_ode_steps: int = 15,
        command_type: str = "twist",
        boundary_matching: bool = False,
        motion_limits=None,
        stop_callback=None,
        fault_callback=None,
    ):
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if num_ode_steps < 1:
            raise ValueError("num_ode_steps must be positive")
        if command_type not in ("twist", "delta_twist"):
            raise ValueError(
                "command_type must be twist or delta_twist in the base frame"
            )
        self.boundary_matching = boundary_matching
        self.motion_limits = (
            None if motion_limits is None else torch.as_tensor(motion_limits)
        )
        if boundary_matching and (
            self.motion_limits is None
            or not torch.isfinite(self.motion_limits).all()
            or (self.motion_limits <= 0).any()
        ):
            raise ValueError(
                "Boundary matching requires positive physical motion_limits"
            )
        self.stop_callback, self.fault_callback = stop_callback, fault_callback
        self.command_type = command_type
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
        self.gripper_commands = None
        self.gripper_times = None
        self.horizon = 0.0
        self.prefix_steps = 0
        self.step_index = 0
        self.query_count = 0
        self.last_motion = None

    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, bool]:
        """Return ``(action, queried)`` after one sequential command callback."""
        if (self.stop_callback and self.stop_callback()) or (
            self.fault_callback and self.fault_callback()
        ):
            raise StopIteration("Task terminated or controller fault")
        queried = self.control_points is None or self.step_index >= self.prefix_steps
        if queried:
            observation = self.observation_callback()
            if self.boundary_matching and self.last_motion is not None:
                reference = observation["images"]
                observation = dict(
                    observation,
                    boundary_value=self.last_motion.to(reference).unsqueeze(0),
                )
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
            cp = prediction["control_points"].detach().cpu()
            knots = prediction["knots"].detach().cpu()
            if (
                cp.ndim != 3
                or cp.shape[0] != 1
                or knots.ndim != 2
                or knots.shape[0] != 1
            ):
                raise ValueError("Execution requires a single-observation batch")
            if not torch.isfinite(cp).all() or not torch.isfinite(knots).all():
                raise ValueError("Spline parameters must be finite")
            if "gripper_commands" in prediction:
                if (
                    prediction.get("action_type") != "twist"
                    or prediction.get("action_frame") != "base"
                ):
                    raise ValueError("Executor requires physical base-frame twists")
                gripper = prediction["gripper_commands"].detach().cpu()
                grid = prediction["gripper_times"].detach().cpu()
                if (
                    gripper.ndim != 3
                    or gripper.shape[0] != 1
                    or cp.shape[-1] != 6 * gripper.shape[-1]
                ):
                    raise ValueError("Motion and gripper arm dimensions do not agree")
                if (
                    grid.ndim != 1
                    or len(grid) != gripper.shape[1]
                    or len(grid) < 2
                    or not torch.isfinite(grid).all()
                    or (grid.diff() <= 0).any()
                    or grid[0] != 0
                    or grid[-1] != 1
                ):
                    raise ValueError("Gripper timestamps must increase from 0 to 1")
                if not ((gripper == -1) | (gripper == 1)).all():
                    raise ValueError(
                        "Gripper commands must be -1 (closed) or +1 (open)"
                    )
                self.gripper_commands, self.gripper_times = gripper[0], grid
            else:
                self.gripper_commands, self.gripper_times = None, None
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

        u = torch.tensor(
            [[self.step_index * self.control_dt / self.horizon]],
            dtype=self.control_points.dtype,
        )
        basis = eval_cox_de_boor_basis(
            u,
            self.knots,
            num_control_points=self.control_points.shape[1],
            degree=self.policy.degree,
        )
        action = torch.bmm(basis, self.control_points)[0, 0]
        if self.motion_limits is not None:
            if (
                self.motion_limits.shape != action.shape
                or (action.abs() > self.motion_limits).any()
            ):
                raise ValueError("Planned motion exceeds controller limits")
        self.last_motion = action.clone()
        if self.command_type == "delta_twist":
            # Convert physical velocity to the increment for this control period.
            action = action * self.control_dt
        if self.gripper_commands is not None:
            index = (
                torch.searchsorted(self.gripper_times, u.reshape(()), right=True)
                .sub(1)
                .clamp(0, len(self.gripper_times) - 1)
            )
            gripper = self.gripper_commands[index]
            action = torch.cat(
                (action.reshape(-1, 6), gripper[:, None]), dim=-1
            ).flatten()
        self.command_callback(action)
        self.step_index += 1
        if (self.stop_callback and self.stop_callback()) or (
            self.fault_callback and self.fault_callback()
        ):
            raise StopIteration("Task terminated or controller fault after command")
        return action, queried
