"""Differentiable clamped B-spline evaluation and physical-time derivatives."""

from typing import Optional
import torch
from torch import nn


def compute_clamped_knot_vector(num_control_points, degree=3, device=None, dtype=None):
    if degree < 0 or num_control_points <= degree:
        raise ValueError("Require K > degree >= 0")
    interior = torch.linspace(
        0, 1, num_control_points - degree + 1, device=device, dtype=dtype
    )[1:-1]
    return torch.cat(
        (interior.new_zeros(degree + 1), interior, interior.new_ones(degree + 1))
    )


def eval_cox_de_boor_basis(u, knots, num_control_points, degree=3, eps=1e-8):
    """Evaluate N(u); zero denominators contribute zero, without perturbing spans.

    eps is retained for API compatibility; positive spans are divided exactly.
    Gradients through positive spans exist away from moving span boundaries.
    """
    if degree < 0 or num_control_points <= degree:
        raise ValueError("Require K > degree >= 0")
    unbatched = u.ndim == 1 and knots.ndim == 1
    if u.ndim == 1:
        u = u.unsqueeze(0)
    if knots.ndim == 1:
        knots = knots.unsqueeze(0)
    if (
        u.ndim != 2
        or knots.ndim != 2
        or knots.shape[-1] != num_control_points + degree + 1
    ):
        raise ValueError("Incompatible query or knot shape")
    x = u.unsqueeze(-1)
    k = knots.unsqueeze(1)
    basis = ((x >= k[..., :-1]) & (x < k[..., 1:])).to(u.dtype)
    for d in range(1, degree + 1):
        n = knots.shape[-1] - d - 1
        den1 = k[..., d : d + n] - k[..., :n]
        den2 = k[..., d + 1 : d + n + 1] - k[..., 1 : n + 1]
        left = (x - k[..., :n]) / torch.where(den1 > 0, den1, torch.ones_like(den1))
        right = (k[..., d + 1 : d + n + 1] - x) / torch.where(
            den2 > 0, den2, torch.ones_like(den2)
        )
        basis = torch.where(den1 > 0, left * basis[..., :n], 0) + torch.where(
            den2 > 0, right * basis[..., 1 : n + 1], 0
        )
    # The half-open degree-zero intervals require explicit right-endpoint handling.
    final = torch.zeros_like(basis)
    final[..., -1] = 1
    basis = torch.where(x == k[..., -degree - 1 : -degree or None], final, basis)
    return basis.squeeze(0) if unbatched else basis


def evaluate_bspline(query_times, knots, control_points, degree=3):
    """Batched spline values; knots (B,n) are shared, (B,G,n) give one partition per group.

    With grouped knots the control-point channels split into G equal blocks (one per
    arm, in channel order), each evaluated on its own knot vector.
    """
    if knots.ndim == 3:
        groups = knots.shape[1]
        if control_points.shape[-1] % groups:
            raise ValueError("Control-point channels must split evenly across knot groups")
        return torch.cat(
            [
                evaluate_bspline(query_times, knots[:, g], block, degree)
                for g, block in enumerate(control_points.chunk(groups, dim=-1))
            ],
            dim=-1,
        )
    basis = eval_cox_de_boor_basis(query_times, knots, control_points.shape[1], degree)
    return torch.bmm(basis, control_points)


class BSplineEvaluator(nn.Module):
    def __init__(self, degree=3, num_control_points=8):
        super().__init__()
        if degree < 0 or num_control_points <= degree:
            raise ValueError("Require K > degree >= 0")
        self.degree = degree
        self.num_control_points = num_control_points

    def forward(
        self,
        control_points,
        knots=None,
        query_times=None,
        num_samples=50,
        horizon_duration: Optional[torch.Tensor] = None,
    ):
        """Return value, first derivative, second derivative.

        Query times are normalized to [0,1]. With horizon_duration in seconds,
        derivatives are with respect to physical time; otherwise they are with
        respect to normalized time. A velocity spline's derivatives are acceleration
        and jerk, not position velocity and position acceleration.
        """
        B, K, D = control_points.shape
        if knots is not None and knots.ndim == 3:
            groups = knots.shape[1]
            if D % groups:
                raise ValueError("Control-point channels must split evenly across knot groups")
            parts = [
                self.forward(block, knots[:, g], query_times, num_samples, horizon_duration)
                for g, block in enumerate(control_points.chunk(groups, dim=-1))
            ]
            return tuple(torch.cat(values, dim=-1) for values in zip(*parts))
        if knots is None:
            knots = compute_clamped_knot_vector(
                K, self.degree, control_points.device, control_points.dtype
            )
        if knots.ndim == 1:
            knots = knots.expand(B, -1)
        if query_times is None:
            query_times = torch.linspace(
                0,
                1,
                num_samples,
                device=control_points.device,
                dtype=control_points.dtype,
            )
        if query_times.ndim == 1:
            query_times = query_times.expand(B, -1)
        value = torch.bmm(
            eval_cox_de_boor_basis(query_times, knots, K, self.degree), control_points
        )
        derivatives = []
        cp, kv, p = control_points, knots, self.degree
        for _ in range(2):
            if p > 0:
                count = cp.shape[1]
                denom = kv[:, p + 1 : count + p] - kv[:, 1:count]
                cp = torch.where(
                    denom.unsqueeze(-1) > 0,
                    p
                    * cp.diff(dim=1)
                    / torch.where(denom > 0, denom, torch.ones_like(denom)).unsqueeze(
                        -1
                    ),
                    0,
                )
                kv, p = kv[:, 1:-1], p - 1
                derivatives.append(
                    torch.bmm(eval_cox_de_boor_basis(query_times, kv, count - 1, p), cp)
                )
            else:
                derivatives.append(torch.zeros_like(value))
        if horizon_duration is not None:
            from .learnable_knots import duration_tensor

            H = duration_tensor(horizon_duration, control_points).unsqueeze(-1)
            derivatives = [derivatives[0] / H, derivatives[1] / H.square()]
        return value, *derivatives
