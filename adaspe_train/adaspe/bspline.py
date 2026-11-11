"""JAX port of AdaSpeVLA B-spline utilities.

Reference: test/franka/v1_3_bread_jax_last4_20261104/source/adaspevla/bspline/
(bspline_core.py, learnable_knots.py, curve_fitting.py). Same equations; input
validation is left to the caller because these run inside jit.
"""
import jax
import jax.numpy as jnp


def intervals_to_knots(delta, degree=3):
    """(B, I) spans summing to 1 -> clamped knot vector (B, I + 2*degree + 1)."""
    b = delta.shape[0]
    return jnp.concatenate(
        [jnp.zeros((b, degree + 1), delta.dtype), jnp.cumsum(delta, -1)[:, :-1], jnp.ones((b, degree + 1), delta.dtype)],
        axis=-1,
    )


def basis(u, knots, num_control_points, degree=3):
    """Cox-de Boor basis N(u): u (B,T), knots (B,n) -> (B,T,K). Zero spans contribute zero."""
    x = u[..., None]
    k = knots[:, None, :]
    out = ((x >= k[..., :-1]) & (x < k[..., 1:])).astype(u.dtype)
    for d in range(1, degree + 1):
        n = knots.shape[-1] - d - 1
        den1 = k[..., d : d + n] - k[..., :n]
        den2 = k[..., d + 1 : d + n + 1] - k[..., 1 : n + 1]
        left = (x - k[..., :n]) / jnp.where(den1 > 0, den1, 1.0)
        right = (k[..., d + 1 : d + n + 1] - x) / jnp.where(den2 > 0, den2, 1.0)
        out = jnp.where(den1 > 0, left * out[..., :n], 0.0) + jnp.where(den2 > 0, right * out[..., 1 : n + 1], 0.0)
    # Right endpoint of the half-open intervals.
    final = jnp.zeros_like(out).at[..., -1].set(1.0)
    hit = x == k[..., -degree - 1 : (-degree or None)]
    out = jnp.where(hit, final, out)
    assert out.shape[-1] == num_control_points
    return out


def evaluate(query_times, knots, control_points, degree=3):
    """Spline values (B,T,D) at normalized times (B,T)."""
    return jnp.einsum("btk,bkd->btd", basis(query_times, knots, control_points.shape[1], degree), control_points)


def fit_control_points(actions, knots, query_times, num_control_points, degree=3, reg_lambda=1e-4):
    """Endpoint-pinned ridge least squares (B,T,D) -> control points (B,K,D) and MSE (B,).

    Minimizes ||N P - A||^2 + lambda ||P_interior||^2 with P_0 = A_0, P_last = A_last,
    solved as an augmented least-squares problem like the PyTorch reference.
    """
    n = basis(query_times, knots, num_control_points, degree)
    left, right = actions[:, :1], actions[:, -1:]
    matrix = n[:, :, 1:-1]
    rhs = actions - n[:, :, :1] @ left - n[:, :, -1:] @ right
    count = matrix.shape[-1]
    ridge = jnp.sqrt(reg_lambda) * jnp.eye(count, dtype=actions.dtype)
    aug_m = jnp.concatenate([matrix, jnp.broadcast_to(ridge, (actions.shape[0], count, count))], axis=1)
    aug_r = jnp.concatenate([rhs, jnp.zeros((actions.shape[0], count, actions.shape[-1]), actions.dtype)], axis=1)
    interior = jax.vmap(lambda m, r: jnp.linalg.lstsq(m, r)[0])(aug_m, aug_r)
    cp = jnp.concatenate([left, interior, right], axis=1)
    recon = jnp.mean(jnp.square(n @ cp - actions), axis=(1, 2))
    return cp, recon


def motion_variation(actions, timestamps):
    """Normalized acceleration magnitude per interval (B,T-1); last copies its neighbour."""
    widths = jnp.diff(timestamps, axis=-1)
    velocity = jnp.diff(actions, axis=1) / widths[..., None]
    if actions.shape[1] < 3:
        return jnp.zeros((actions.shape[0], actions.shape[1] - 1), actions.dtype)
    accel = jnp.linalg.norm(jnp.diff(velocity, axis=1) / widths[:, :-1, None], axis=-1)
    return jnp.concatenate([accel, accel[:, -1:]], axis=-1)


def gripper_variation(gripper, timestamps):
    widths = jnp.diff(timestamps, axis=-1)
    return jnp.max(jnp.abs(jnp.diff(gripper, axis=1)) / widths[..., None], axis=-1)


def density_span_target(motion, timestamps, num_intervals, min_interval_eps=0.01, density_gain=10.0,
                        gripper=None, gripper_gain=10.0, gripper_ref=1.0):
    """Inverse-CDF knot spans (B,I) from motion acceleration and gripper variation."""
    b, t, _ = motion.shape
    widths = jnp.diff(timestamps, axis=-1)
    density = 1.0 + density_gain * jnp.tanh(motion_variation(motion, timestamps))
    if gripper is not None:
        density = density + gripper_gain * jnp.tanh(gripper_variation(gripper, timestamps) / gripper_ref)
    masses = density * widths
    cdf = jnp.concatenate([jnp.zeros((b, 1), motion.dtype), jnp.cumsum(masses, -1)], -1)
    cdf = cdf / cdf[:, -1:]
    q = jnp.broadcast_to(jnp.linspace(0.0, 1.0, num_intervals + 1, dtype=motion.dtype), (b, num_intervals + 1))
    index = jnp.clip(jax.vmap(lambda c, qq: jnp.searchsorted(c, qq, side="right"))(cdf, q) - 1, 0, t - 2)
    c0 = jnp.take_along_axis(cdf, index, 1)
    c1 = jnp.take_along_axis(cdf, index + 1, 1)
    t0 = jnp.take_along_axis(timestamps, index, 1)
    t1 = jnp.take_along_axis(timestamps, index + 1, 1)
    breaks = t0 + (q - c0) / (c1 - c0) * (t1 - t0)
    delta = jnp.diff(breaks, axis=-1) / (timestamps[:, -1:] - timestamps[:, :1])
    return min_interval_eps + (1 - num_intervals * min_interval_eps) * delta


def query_interval_target(timestamps, motion, gripper, horizon, min_interval, max_interval, e_crit,
                          anticipation_margin=0.0, gripper_threshold=0.5):
    """Event-based timing label (B,1): time to first motion/gripper event, clamped."""
    lower = jnp.minimum(horizon, min_interval)
    upper = jnp.minimum(horizon, max_interval)
    event = motion_variation(motion, timestamps) >= e_crit
    if gripper is not None:
        state = gripper >= gripper_threshold
        event = event | jnp.any(state[:, 1:] != state[:, :-1], axis=-1)
    rel = timestamps[:, 1:] - timestamps[:, :1]
    risk = jnp.min(jnp.where(event, rel, jnp.inf), axis=-1, keepdims=True)
    return jnp.clip(jnp.maximum(risk - anticipation_margin, 0.0), lower, upper)


# ---- Timing labels from measured anchor offsets (as AdaSpeVLA franka v1.x) ---------------
# AdaSpeVLA (benchmarks/pose_math.anchor_deltas, datasets/franka_lerobot.py with the bread mapping
# configs/franka/collector_verified_20261030.json): pose = state[4:7] XYZ (m) + state[7:10] Euler XYZ
# (rad, scipy "xyz") in the reset frame, output frame = reset frame (R_b = I). Offsets are
#   position p_j - p_anchor,   rotation Log(R_j R_anchor^T) as a rotation vector.

def _skew(v):
    z = jnp.zeros_like(v[..., 0])
    return jnp.stack([jnp.stack([z, -v[..., 2], v[..., 1]], -1),
                      jnp.stack([v[..., 2], z, -v[..., 0]], -1),
                      jnp.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rotvec_to_matrix(v):
    th2 = jnp.sum(v * v, -1)[..., None, None]
    th = jnp.sqrt(th2 + 1e-30)
    small = th2 < 1e-12
    a = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    b = jnp.where(small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / (th2 + 1e-30))
    k = _skew(v)
    return jnp.eye(3, dtype=v.dtype) + a * k + b * (k @ k)


def matrix_to_rotvec(r):
    w = jnp.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    sin = 0.5 * jnp.linalg.norm(w, axis=-1)
    cos = 0.5 * (jnp.trace(r, axis1=-2, axis2=-1) - 1.0)
    th = jnp.arctan2(sin, cos)  # accurate for small angles, unlike arccos
    scale = jnp.where(sin < 1e-6, 0.5 + th ** 2 / 12.0, th / (2.0 * sin + 1e-30))
    return w * scale[..., None]  # angles near pi do not occur within a 1.2 s window


def euler_xyz_to_matrix(e):
    """scipy Rotation.from_euler("xyz", e).as_matrix(): extrinsic x, then y, then z (R = Rz Ry Rx)."""
    cx, cy, cz = jnp.cos(e[..., 0]), jnp.cos(e[..., 1]), jnp.cos(e[..., 2])
    sx, sy, sz = jnp.sin(e[..., 0]), jnp.sin(e[..., 1]), jnp.sin(e[..., 2])
    one, zero = jnp.ones_like(cx), jnp.zeros_like(cx)
    rx = jnp.stack([jnp.stack([one, zero, zero], -1), jnp.stack([zero, cx, -sx], -1), jnp.stack([zero, sx, cx], -1)], -2)
    ry = jnp.stack([jnp.stack([cy, zero, sy], -1), jnp.stack([zero, one, zero], -1), jnp.stack([-sy, zero, cy], -1)], -2)
    rz = jnp.stack([jnp.stack([cz, -sz, zero], -1), jnp.stack([sz, cz, zero], -1), jnp.stack([zero, zero, one], -1)], -2)
    return rz @ ry @ rx


def measured_anchor_offsets(pose):
    """Measured poses (B,T,6) [x,y,z, euler xyz] -> anchor offsets (B,T,6) [dx,dy,dz,rx,ry,rz]."""
    rot = euler_xyz_to_matrix(pose[..., 3:])
    rel = rot @ jnp.swapaxes(rot[:, :1], -1, -2)  # R_j R_anchor^T
    return jnp.concatenate([pose[..., :3] - pose[:, :1, :3], matrix_to_rotvec(rel)], -1)
