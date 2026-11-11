"""Knot-density and timing labels, shared by the model (training), calibrate_timing_anchor.py and
tests/check_timing_labels.py, so all three compute exactly the same thing.

Label modes (cfg.timing_label_mode / cfg.knot_label_mode):
  "command_delta":        motion = normalized per-step pose commands; gripper = commanded open/closed state.
  "measured":             AdaSpeVLA franka v1.x: motion = measured anchor-relative pose offsets / motion_scale;
                          gripper = measured width state[0].
  "measured_cmd_gripper": motion as "measured"; gripper = commanded open/closed state (2026-11-10 recipe).
     Why: in the franka data the measured width only crosses 0.5 a median 19 frames (~1.9 s) after the open /
     close command (it stays flat for ~1.4 s, arm still), so width-based gripper events land ~1.9 s after the
     decision and mostly outside the chunk. The commanded state switches at the decision itself.
Knot motion normalization (cfg.knot_motion_norm):
  "none":   density = 1 + gain * tanh(motion_variation)  (original; tanh saturated in ~70 % of intervals here)
  "e_crit": density = 1 + gain * tanh(motion_variation / e_crit)
"""
import jax
import jax.numpy as jnp

from . import bspline as bs

# Padded action channels carrying the measured timing window: 6 pose + 1 gripper width (adaspe/timing_data.py).
TIMING_CHANNELS = slice(7, 14)


def gripper_open_state(cmd):
    """Gripper command (..., T) -> open(1)/closed(0) state (..., T, 1).

    Commands are -1 close / +1 open / 0 "no new command" (OpenPI-normalized ~ raw since q01=-1, q99~1).
    0 holds the previous nonzero command (or the next one at a window start); an all-zero window maps to
    0.5 and produces no event. On the franka data this in-window fill gives the same first switch as an
    episode-level forward fill in >= 99.96 % of windows.
    """
    t = cmd.shape[-1]
    nz = jnp.abs(cmd) > 0.5
    idx = jnp.arange(t)
    last = jax.lax.cummax(jnp.where(nz, idx, -1), axis=cmd.ndim - 1)
    nxt = jax.lax.cummin(jnp.where(nz, idx, t), axis=cmd.ndim - 1, reverse=True)
    src = jnp.where(last >= 0, last, jnp.minimum(nxt, t - 1))
    filled = jnp.where(jnp.any(nz, -1, keepdims=True), jnp.take_along_axis(cmd, src, -1), 0.0)
    return jnp.clip((filled[..., None] + 1) / 2, 0.0, 1.0)


def event_inputs(actions, cfg, mode):
    """(motion (B,T,6), gripper (B,T,1)) for timing events or knot density.

    actions: (B,T,>=14) model actions: [:7] normalized commands, TIMING_CHANNELS the raw measured window.
    """
    acts = actions[..., : cfg.plugin_action_dim].astype(jnp.float32)
    if mode == "command_delta":
        return acts[..., :6], gripper_open_state(acts[..., 6])
    meas = actions[..., TIMING_CHANNELS].astype(jnp.float32)
    motion = bs.measured_anchor_offsets(meas[..., :6]) / jnp.asarray(cfg.timing_motion_scale, jnp.float32)
    if mode == "measured":
        return motion, meas[..., 6:7]
    if mode == "measured_cmd_gripper":
        return motion, gripper_open_state(acts[..., 6])
    raise ValueError(f"unknown label mode {mode!r}")


def control_times(cfg, batch):
    t = jnp.arange(cfg.action_horizon, dtype=jnp.float32) * cfg.control_dt
    return jnp.broadcast_to(t, (batch, cfg.action_horizon))


def horizon(cfg, batch):
    return jnp.full((batch, 1), cfg.horizon_duration, jnp.float32)


def timing_target(actions, cfg):
    """Query-interval label (B,1): time to the first motion / gripper event, clipped to the interval range."""
    motion, gripper = event_inputs(actions, cfg, cfg.timing_label_mode)
    b = actions.shape[0]
    return bs.query_interval_target(
        control_times(cfg, b), motion, gripper, horizon(cfg, b), cfg.min_query_interval, cfg.max_query_interval,
        cfg.e_crit, gripper_threshold=cfg.gripper_threshold,
    )


def knot_span_target(actions, cfg):
    """Knot-span target (B, K - degree) from motion acceleration and gripper switches."""
    motion, gripper = event_inputs(actions, cfg, cfg.knot_label_mode)
    if cfg.knot_motion_norm == "e_crit":
        motion = motion / cfg.e_crit  # motion_variation is linear in motion: tanh(mv / e_crit)
    elif cfg.knot_motion_norm != "none":
        raise ValueError(f"unknown knot_motion_norm {cfg.knot_motion_norm!r}")
    return bs.density_span_target(
        motion, control_times(cfg, actions.shape[0]), cfg.num_control_points - cfg.degree, cfg.min_interval_eps,
        cfg.density_gain, gripper=gripper, gripper_gain=cfg.gripper_density_gain,
    )
