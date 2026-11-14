"""Optional boundary projection of the first control point (manuscript Eq. boundary)."""

import torch


class BoundaryRefitter:
    @staticmethod
    def refit_boundary_control_points(
        control_points, knots, x_0, degree=3, channels=None
    ):
        """Replace the selected entries of control point zero with ``x_0``.

        For a clamped spline this makes the new segment's initial value equal x_0
        on those channels; every other entry is unchanged. The policy selects the
        gripper-width channels only: motion offsets start at zero by construction
        after re-anchoring. ``x_0`` shares the control points' units.
        """
        B, K, D = control_points.shape
        if degree < 0 or K <= degree:
            raise ValueError("Require K > degree >= 0")
        index = (
            torch.arange(D, device=control_points.device)
            if channels is None
            else torch.as_tensor(channels, device=control_points.device).reshape(-1)
        )
        if (
            index.numel() == 0
            or index.min() < 0
            or index.max() >= D
            or len(index.unique()) != len(index)
        ):
            raise ValueError("Boundary channels must be distinct control-point channels")
        if x_0.shape != (B, len(index)) or not torch.isfinite(x_0).all():
            raise ValueError("Boundary value must be a finite (B, channels) tensor")
        if knots.ndim == 3:
            # One knot partition per arm: validate each partition.
            groups = knots.shape[1]
            if D % groups:
                raise ValueError("Knot groups do not split the channels")
            for g in range(groups):
                BoundaryRefitter._validate(knots[:, g], B, K, degree)
        else:
            BoundaryRefitter._validate(knots, B, K, degree)
        result = control_points.clone()
        result[:, 0, index] = x_0.to(result)
        return result

    @staticmethod
    def _validate(knots, B, K, degree):
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
