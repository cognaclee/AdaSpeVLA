"""Optional endpoint constraints in physical time for any vector-valued spline."""

import torch
from .learnable_knots import duration_tensor


class BoundaryRefitter:
    @staticmethod
    def refit_boundary_control_points(
        control_points, knots, x_0, v_0=None, a_0=None, degree=3, horizon_duration=1.0
    ):
        """Match value, first and second time derivatives when provided.

        For a velocity spline, x_0 is velocity, v_0 is acceleration, and a_0 is
        jerk. They are never an absolute pose. Constraints are optional; a_0
        requires v_0. C2 means continuous second derivative, not zero jerk.
        """
        B, K, D = control_points.shape
        if degree < 0 or K <= degree:
            raise ValueError("Require K > degree >= 0")
        if a_0 is not None and v_0 is None:
            raise ValueError("a_0 requires v_0 for C2 boundary matching")
        if v_0 is not None and degree < 1 or a_0 is not None and degree < 2:
            raise ValueError("Spline degree is too low for requested derivatives")
        for value in (x_0, v_0, a_0):
            if value is not None and (
                value.shape != (B, D) or not torch.isfinite(value).all()
            ):
                raise ValueError("Boundary states must be finite (B,D) tensors")
        if knots.ndim == 1:
            knots = knots.expand(B, -1)
        if (
            knots.shape != (B, K + degree + 1)
            or not torch.isfinite(knots).all()
            or (knots.diff(dim=-1) < 0).any()
        ):
            raise ValueError("Invalid knot vector")
        if (
            not (knots[:, : degree + 1] == 0).all()
            or not (knots[:, -degree - 1 :] == 1).all()
        ):
            raise ValueError("Boundary refitting requires knots clamped to [0,1]")
        duration = duration_tensor(horizon_duration, control_points)
        result = control_points.clone()
        result[:, 0] = x_0
        if v_0 is not None:
            A = (knots[:, degree + 1] - knots[:, 1]).unsqueeze(-1)
            if (A <= 0).any():
                raise ValueError("First active span must be positive")
            result[:, 1] = x_0 + A / degree * duration * v_0
        if a_0 is not None:
            B_span = (knots[:, degree + 2] - knots[:, 2]).unsqueeze(-1)
            C = (knots[:, degree + 1] - knots[:, 2]).unsqueeze(-1)
            if (B_span <= 0).any() or (C <= 0).any():
                raise ValueError("Derivative spans must be positive")
            result[:, 2] = result[:, 1] + B_span / degree * (
                duration * v_0 + C / (degree - 1) * duration.square() * a_0
            )
        return result
