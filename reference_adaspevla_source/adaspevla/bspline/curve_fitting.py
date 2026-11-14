"""Endpoint-pinned ridge least squares at supplied demonstration timestamps."""

import math
import torch
from torch import nn
from .bspline_core import eval_cox_de_boor_basis, compute_clamped_knot_vector


def fit_bspline_control_points_least_squares(
    actions,
    num_control_points=8,
    degree=3,
    knots=None,
    reg_lambda=1e-4,
    pin_endpoints=True,
    query_times=None,
):
    """Minimize ||NP-A||_F^2 + lambda ||P||_F^2 for fixed knots.

    This is ridge regularization, not a curvature or jerk penalty. The optional
    endpoint constraints fix P_0=A_0 and P_last=A_last. Augmented least squares
    avoids squaring the design matrix's condition number with normal equations.
    """
    B, T, D = actions.shape
    K = num_control_points
    if (
        T < 2
        or K <= degree
        or degree < 0
        or not math.isfinite(reg_lambda)
        or reg_lambda < 0
    ):
        raise ValueError("Invalid fitting dimensions or ridge coefficient")
    if not torch.isfinite(actions).all():
        raise ValueError("Actions must be finite")
    if knots is None:
        knots = compute_clamped_knot_vector(K, degree, actions.device, actions.dtype)
    if knots.ndim == 1:
        knots = knots.expand(B, -1)
    if (
        knots.shape != (B, K + degree + 1)
        or not torch.isfinite(knots).all()
        or (knots.diff(dim=-1) < 0).any()
    ):
        raise ValueError("Invalid knot vector")
    if pin_endpoints and (
        not (knots[:, : degree + 1] == 0).all()
        or not (knots[:, -degree - 1 :] == 1).all()
    ):
        raise ValueError("Pinned fitting requires knots clamped to [0,1]")
    if query_times is None:
        query_times = torch.linspace(
            0, 1, T, device=actions.device, dtype=actions.dtype
        ).expand(B, -1)
    if query_times.ndim == 1:
        query_times = query_times.expand(B, -1)
    if (
        query_times.shape != (B, T)
        or not torch.isfinite(query_times).all()
        or (query_times.diff(dim=-1) <= 0).any()
    ):
        raise ValueError("Query times must be finite and strictly increasing")
    if pin_endpoints and (
        not torch.allclose(query_times[:, 0], actions.new_zeros(B))
        or not torch.allclose(query_times[:, -1], actions.new_ones(B))
    ):
        raise ValueError("Pinned fitting requires normalized timestamps from 0 to 1")
    N = eval_cox_de_boor_basis(query_times, knots, K, degree)
    if pin_endpoints:
        if K < 2:
            raise ValueError("Two endpoints require at least two control points")
        left, right = actions[:, :1], actions[:, -1:]
        matrix = N[:, :, 1:-1]
        rhs = actions - N[:, :, :1] @ left - N[:, :, -1:] @ right
    else:
        matrix, rhs = N, actions
    count = matrix.shape[-1]
    if count:
        ridge = math.sqrt(reg_lambda) * torch.eye(
            count, device=actions.device, dtype=actions.dtype
        ).expand(B, -1, -1)
        system = torch.cat((matrix, ridge), dim=1)
        target = torch.cat((rhs, actions.new_zeros(B, count, D)), dim=1)
        solution = torch.linalg.lstsq(system, target).solution
    else:
        solution = actions.new_empty(B, 0, D)
    cp = torch.cat((left, solution, right), dim=1) if pin_endpoints else solution
    return cp, (N @ cp - actions).square().mean(dim=(1, 2))


def fit_bspline_control_points_grouped(
    actions, num_control_points=8, degree=3, knots=None, query_times=None, **kwargs
):
    """Least-squares fit with shared (B,n) or per-group (B,G,n) knots.

    Per-group knots fit each block of D/G channels independently on its own
    partition; the error is the mean over all channels, as in the shared fit.
    """
    if knots is None or knots.ndim < 3:
        return fit_bspline_control_points_least_squares(
            actions, num_control_points, degree, knots, query_times=query_times, **kwargs
        )
    groups = knots.shape[1]
    if actions.shape[-1] % groups:
        raise ValueError("Action channels must split evenly across knot groups")
    fits = [
        fit_bspline_control_points_least_squares(
            block, num_control_points, degree, knots[:, g], query_times=query_times, **kwargs
        )
        for g, block in enumerate(actions.chunk(groups, dim=-1))
    ]
    cp = torch.cat([fit[0] for fit in fits], dim=-1)
    return cp, torch.stack([fit[1] for fit in fits], -1).mean(-1)


class BSplineCurveFitter(nn.Module):
    def __init__(self, num_control_points=8, degree=3, reg_lambda=1e-4):
        super().__init__()
        self.num_control_points, self.degree, self.reg_lambda = (
            num_control_points,
            degree,
            reg_lambda,
        )

    def forward(self, actions, knots=None, query_times=None):
        return fit_bspline_control_points_least_squares(
            actions,
            self.num_control_points,
            self.degree,
            knots,
            self.reg_lambda,
            query_times=query_times,
        )
