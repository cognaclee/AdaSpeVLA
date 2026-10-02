"""Optional boundary projection of the first control point (manuscript Eq. boundary)."""

import torch


class BoundaryRefitter:
    @staticmethod
    def refit_boundary_control_points(control_points, knots, x_0, degree=3):
        """Replace control point zero with the last emitted twist.

        For a clamped spline this makes the new segment's initial value equal x_0;
        all other control points are unchanged. Control points and x_0 share units
        (physical twists here, so this equals P_0 = D^{-1} a_last after scaling).
        """
        B, K, D = control_points.shape
        if degree < 0 or K <= degree:
            raise ValueError("Require K > degree >= 0")
        if x_0.shape != (B, D) or not torch.isfinite(x_0).all():
            raise ValueError("Boundary value must be a finite (B,D) tensor")
        if knots.ndim == 3:
            # One knot partition per arm: validate each, then replace P_0 for all arms.
            groups = knots.shape[1]
            if D % groups or x_0.shape != (B, D):
                raise ValueError("Boundary value and knot groups do not match the channels")
            for g, (block, value) in enumerate(
                zip(control_points.chunk(groups, -1), x_0.chunk(groups, -1))
            ):
                BoundaryRefitter.refit_boundary_control_points(
                    block, knots[:, g], value, degree
                )
            result = control_points.clone()
            result[:, 0] = x_0
            return result
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
            raise ValueError("Boundary projection requires knots clamped to [0,1]")
        result = control_points.clone()
        result[:, 0] = x_0
        return result
