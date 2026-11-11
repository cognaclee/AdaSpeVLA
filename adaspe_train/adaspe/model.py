"""AdaSpeVLA plugin on the OpenPI pi0.5 backbone.

The PaliGemma prefix (images + tokenized prompt with discrete state) is exactly
pi0.5's; the pi0.5 action expert is removed and replaced with the AdaSpeVLA heads
ported from the v1.3 PyTorch reference (adaspevla/models/{vlm_backbone,
bspline_flow_policy,adaspe_vla_policy}.py, bspline/learnable_knots.py):

  prefix hidden states -> fusion (pooled cond + token memory)
  -> learnable knot head -> B-spline flow-matching transformer -> control points
  -> timing (query-interval) head.

Actions are OpenPI-normalized 7D body-frame deltas over `action_horizon` steps.
"""
import dataclasses
import math
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask
import openpi.shared.array_typing as at
import flax.nnx.bridge as nnx_bridge

from . import bspline as bs
from . import labels
from .labels import TIMING_CHANNELS  # noqa: F401  (re-exported for adaspe/timing_data.py)


@dataclasses.dataclass(frozen=True)
class AdaSpePi05Config(pi0_config.Pi0Config):
    """pi0.5 config whose action expert is replaced by the AdaSpeVLA plugin."""

    pi05: bool = True
    stage: Literal["action", "timing"] = "action"
    plugin_action_dim: int = 7  # real action channels; the rest of `action_dim` is OpenPI padding
    views: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    control_dt: float = 0.1
    cond_dim: int = 512
    num_control_points: int = 8
    degree: int = 3
    min_interval_eps: float = 0.01
    min_query_interval: float = 0.2
    max_query_interval: float = 0.8
    flow_hidden_dim: int = 384
    flow_num_layers: int = 6
    flow_num_heads: int = 6
    knot_hidden_dim: int = 256
    knot_loss_coeff: float = 0.5
    flow_loss_coeff: float = 1.0
    density_gain: float = 10.0
    gripper_density_gain: float = 10.0
    gripper_threshold: float = 0.5
    e_crit: float = 1.0  # set from training-data calibration
    # Timing labels. "command_delta": events on the normalized per-step commands (before 2026-11-09).
    # "measured": as AdaSpeVLA franka v1.x, events on measured anchor-relative pose offsets divided by
    # per-channel motion scales, and on the measured gripper width crossing gripper_threshold. The
    # measured window (state[4:10] pose, state[0] width; 13 frames) arrives in action channels
    # TIMING_CHANNELS, which are zero padding for the plugin (adaspe/timing_data.py).
    # "measured_cmd_gripper": measured motion as "measured", gripper events from the commanded open/closed
    # state instead of the measured width, which lags the command by ~1.9 s on the franka data (adaspe/labels.py).
    timing_label_mode: Literal["command_delta", "measured", "measured_cmd_gripper"] = "command_delta"
    # Knot-density supervision, same modes ("measured" = AdaSpeVLA original). The flow target
    # stays the delta commands either way; only the knot-density target changes.
    knot_label_mode: Literal["command_delta", "measured", "measured_cmd_gripper"] = "command_delta"
    # "e_crit": knot density uses tanh(motion_variation / e_crit) instead of the saturating tanh(motion_variation).
    knot_motion_norm: Literal["none", "e_crit"] = "none"
    timing_motion_scale: tuple[float, ...] | None = None  # 95th pct |offset| per channel (calibrated; knots too)
    timing_probe_count: int = 16
    timing_num_ode_steps: int = 15
    num_ode_steps: int = 15

    @override
    def create(self, rng: at.KeyArrayLike) -> "AdaSpePi05":
        return AdaSpePi05(self, rngs=nnx.Rngs(rng))

    @property
    def horizon_duration(self) -> float:
        return (self.action_horizon - 1) * self.control_dt

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if self.stage == "timing":
            return nnx.All(nnx.Param, nnx.Not(nnx_utils_path(".*freq_head.*")))
        return nnx.Nothing


def nnx_utils_path(regex):
    from openpi.shared import nnx_utils

    return nnx_utils.PathRegex(regex)


class Seq(nnx.Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Activations are module-level classes: a class defined per call would give the model a different
# graph type on every build, which breaks OpenPI's jit out_shardings check.
class _SiLU(nnx.Module):
    def __call__(self, x):
        return nnx.silu(x)


class _GELU(nnx.Module):
    def __call__(self, x):
        return jax.nn.gelu(x, approximate=False)


class _Sigmoid(nnx.Module):
    def __call__(self, x):
        return jax.nn.sigmoid(x)


def silu():
    return _SiLU()


def gelu():
    return _GELU()


def sigmoid():
    return _Sigmoid()


def ln(dim, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)


def sinusoidal(t, dim):
    half = dim // 2
    freq = jnp.exp(jnp.arange(half) * -(math.log(10000) / (half - 1)))
    arg = t[:, None] * freq[None]
    return jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], -1)


class FlowBlock(nnx.Module):
    """AdaLN transformer block with cross-attention to the observation memory."""

    def __init__(self, dim, heads, rngs, mlp_ratio=4.0):
        self.norm1 = ln(dim, rngs)
        self.attn = nnx.MultiHeadAttention(heads, dim, decode=False, rngs=rngs)
        self.norm_cross = ln(dim, rngs)
        self.cross_attn = nnx.MultiHeadAttention(heads, dim, decode=False, rngs=rngs)
        self.norm2 = ln(dim, rngs)
        hidden = int(dim * mlp_ratio)
        self.mlp = Seq(nnx.Linear(dim, hidden, rngs=rngs), gelu(), nnx.Linear(hidden, dim, rngs=rngs))
        self.ada = Seq(silu(), nnx.Linear(dim, 6 * dim, rngs=rngs))

    def __call__(self, x, c, memory, memory_mask):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = jnp.split(self.ada(c)[:, None], 6, axis=-1)
        h = self.norm1(x) * (1 + sc_msa) + s_msa
        x = x + g_msa * self.attn(h, h, h)
        x = x + self.cross_attn(self.norm_cross(x), memory, memory, mask=memory_mask[:, None, None, :])
        h = self.norm2(x) * (1 + sc_mlp) + s_mlp
        return x + g_mlp * self.mlp(h)


class FlowPolicy(nnx.Module):
    def __init__(self, cfg: AdaSpePi05Config, context_dim, rngs):
        d, k, a = cfg.flow_hidden_dim, cfg.num_control_points, cfg.plugin_action_dim
        self.dim = d
        self.input_proj = nnx.Linear(a, d, rngs=rngs)
        self.pos_emb = nnx.Param(jax.random.normal(rngs.params(), (1, k, d)) * 0.02)
        self.cond_proj = Seq(nnx.Linear(context_dim, d, rngs=rngs), silu(), nnx.Linear(d, d, rngs=rngs))
        self.time_mlp = Seq(nnx.Linear(d, d, rngs=rngs), silu(), nnx.Linear(d, d, rngs=rngs))
        self.memory_proj = Seq(ln(cfg.cond_dim, rngs), nnx.Linear(cfg.cond_dim, d, rngs=rngs))
        self.blocks = [FlowBlock(d, cfg.flow_num_heads, rngs) for _ in range(cfg.flow_num_layers)]
        self.final_norm = ln(d, rngs)
        self.output_proj = nnx.Linear(d, a, rngs=rngs)

    def __call__(self, x_t, t, context, memory, memory_mask):
        c = self.time_mlp(sinusoidal(t, self.dim)) + self.cond_proj(context)
        h = self.input_proj(x_t) + self.pos_emb.value
        mem = self.memory_proj(memory)
        for block in self.blocks:
            h = block(h, c, mem, memory_mask)
        return self.output_proj(self.final_norm(h))


class AdaSpePi05(_model.BaseModel):
    def __init__(self, config: AdaSpePi05Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.cfg = config
        pg = _gemma.get_config(config.paligemma_variant)
        # PaliGemma only: the pi0.5 action expert is replaced by the plugin.
        llm = nnx_bridge.ToNNX(_gemma.Module(configs=[pg], embed_dtype=config.dtype, adarms=True))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(num_classes=pg.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=config.dtype)
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        f, c, v = pg.width, config.cond_dim, len(config.views)
        n_knots = config.num_control_points + config.degree + 1
        n_spans = config.num_control_points - config.degree
        # Observation fusion (vlm_backbone.py, spatial-token mode).
        self.fusion_proj = Seq(nnx.Linear(f * v, c, rngs=rngs), ln(c, rngs), silu())
        self.text_proj = Seq(nnx.Linear(f, c, rngs=rngs), ln(c, rngs), silu())
        self.cross_gate = Seq(nnx.Linear(2 * c, c, rngs=rngs), sigmoid())
        self.final_proj = nnx.Linear(c, c, rngs=rngs)
        self.token_proj = nnx.Linear(f, c, rngs=rngs)
        self.view_embedding = nnx.Param(jax.random.normal(rngs.params(), (v, 1, c)) * 0.02)
        self.text_token_proj = nnx.Linear(f, c, rngs=rngs)
        # Knot head (learnable_knots.py).
        h = config.knot_hidden_dim
        self.knot_head = Seq(
            nnx.Linear(c + 1, h, rngs=rngs), silu(), nnx.Linear(h, h, rngs=rngs), silu(), nnx.Linear(h, n_spans, rngs=rngs)
        )
        self.flow_policy = FlowPolicy(config, c + n_knots + 1, rngs)
        timing_dim = c + 1 + n_spans + config.timing_probe_count * config.plugin_action_dim
        self.freq_head = Seq(nnx.Linear(timing_dim, 128, rngs=rngs), silu(), nnx.Linear(128, 1, rngs=rngs))
        self.deterministic = True

    # ---- observation encoding -------------------------------------------------
    def _prefix(self, obs: _model.Observation):
        tokens, masks, ar, view_slices = [], [], [], {}
        pos = 0
        for name in obs.images:
            t, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(t)
            masks.append(einops.repeat(obs.image_masks[name], "b -> b s", s=t.shape[1]))
            ar += [False] * t.shape[1]
            view_slices[name] = (pos, pos + t.shape[1])
            pos += t.shape[1]
        text = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(text)
        masks.append(obs.tokenized_prompt_mask)
        ar += [False] * text.shape[1]
        tokens = jnp.concatenate(tokens, 1)
        mask = jnp.concatenate(masks, 1)
        attn = make_attn_mask(mask, jnp.array(ar))
        positions = jnp.cumsum(mask, axis=1) - 1
        (out,), _ = self.PaliGemma.llm([tokens], mask=attn, positions=positions)
        return out.astype(jnp.float32), view_slices, pos

    def encode(self, obs: _model.Observation):
        """Global condition (B,C), token memory (B,M,C) and memory mask (B,M)."""
        hidden, slices, n_img = self._prefix(obs)
        b = hidden.shape[0]
        views = jnp.stack([hidden[:, slices[n][0] : slices[n][1]] for n in self.cfg.views], 1)  # (B,V,P,F)
        valid = obs.tokenized_prompt_mask[..., None].astype(jnp.float32)
        text_hidden = hidden[:, n_img:]
        text_feat = (text_hidden * valid).sum(1) / jnp.maximum(valid.sum(1), 1.0)
        vision = self.fusion_proj(views.mean(2).reshape(b, -1))
        text = self.text_proj(text_feat)
        gate = self.cross_gate(jnp.concatenate([vision, text], -1))
        cond = self.final_proj(gate * vision + (1 - gate) * text)
        mem = (self.token_proj(views) + self.view_embedding.value).reshape(b, -1, self.cfg.cond_dim)
        memory = jnp.concatenate([mem, self.text_token_proj(text_feat)[:, None]], 1)
        return cond, memory, jnp.ones(memory.shape[:2], bool)

    # ---- plugin pieces --------------------------------------------------------
    def _horizon(self, b):
        return jnp.full((b, 1), self.cfg.horizon_duration, jnp.float32)

    def predict_knots(self, cond, horizon):
        cfg = self.cfg
        n = cfg.num_control_points - cfg.degree
        raw = jax.nn.softmax(self.knot_head(jnp.concatenate([cond, jnp.log(horizon)], -1)), -1)
        delta = cfg.min_interval_eps + (1 - n * cfg.min_interval_eps) * raw
        return bs.intervals_to_knots(delta, cfg.degree), delta

    def _context(self, cond, knots, horizon):
        return jnp.concatenate([cond, knots, jnp.log(horizon)], -1)

    def _times(self, b):
        t = jnp.arange(self.action_horizon, dtype=jnp.float32) * self.cfg.control_dt
        return jnp.broadcast_to(t, (b, self.action_horizon))

    def _gripper01(self, actions):
        """Gripper command as an open(1)/closed(0) state (B,T,1); see labels.gripper_open_state."""
        return labels.gripper_open_state(actions[..., 6])

    def sample_control_points(self, rng, context, memory, mask, num_steps):
        b = context.shape[0]
        x = jax.random.normal(rng, (b, self.cfg.num_control_points, self.cfg.plugin_action_dim))
        dt = 1.0 / num_steps

        def step(i, x):
            t = jnp.full((b,), i * dt, jnp.float32)
            return x + dt * self.flow_policy(x, t, context, memory, mask)

        return jax.lax.fori_loop(0, num_steps, step, x)

    def timing_interval(self, cond, delta, knots, horizon, cp):
        cfg = self.cfg
        probes = jnp.broadcast_to(jnp.linspace(0, 1, cfg.timing_probe_count), (cond.shape[0], cfg.timing_probe_count))
        values = jax.lax.stop_gradient(bs.evaluate(probes, knots, cp, cfg.degree))
        motion, width = values[..., :6], self._gripper01(values)
        feats = jnp.concatenate(
            [cond, jnp.log(horizon), jax.lax.stop_gradient(delta), motion.reshape(cond.shape[0], -1), width.reshape(cond.shape[0], -1)], -1
        )
        lower = jnp.minimum(horizon, cfg.min_query_interval)
        upper = jnp.minimum(horizon, cfg.max_query_interval)
        return lower + (upper - lower) * jax.nn.sigmoid(self.freq_head(feats))

    # ---- OpenPI API -----------------------------------------------------------
    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        cfg = self.cfg
        pre_rng, noise_rng, time_rng, ode_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(pre_rng, observation, train=train)
        acts = actions[..., : cfg.plugin_action_dim].astype(jnp.float32)
        b = acts.shape[0]
        times = self._times(b)
        horizon = self._horizon(b)
        if cfg.stage == "timing":
            cond, memory, mask = jax.lax.stop_gradient(self.encode(observation))
            knots, delta = jax.lax.stop_gradient(self.predict_knots(cond, horizon))
            cp = jax.lax.stop_gradient(
                self.sample_control_points(ode_rng, self._context(cond, knots, horizon), memory, mask, cfg.timing_num_ode_steps)
            )
            interval = self.timing_interval(cond, delta, knots, horizon, cp)
            target = labels.timing_target(actions, cfg)
            return jnp.square(jnp.log(interval) - jnp.log(target))[:, 0]

        cond, memory, mask = self.encode(observation)
        knots, delta = self.predict_knots(cond, horizon)
        target_delta = labels.knot_span_target(actions, cfg)
        target_cp, _ = bs.fit_control_points(
            acts, jax.lax.stop_gradient(knots), times / times[:, -1:], cfg.num_control_points, cfg.degree
        )
        target_cp = jax.lax.stop_gradient(target_cp)
        t = jax.random.uniform(time_rng, (b,))
        x0 = jax.random.normal(noise_rng, target_cp.shape)
        x_t = t[:, None, None] * target_cp + (1 - t[:, None, None]) * x0
        v = self.flow_policy(x_t, t, self._context(cond, knots, horizon), memory, mask)
        flow = jnp.mean(jnp.square(v - (target_cp - x0)), axis=(1, 2))
        # KL(target || pred), batchmean -> per-sample sum.
        kl = jnp.sum(target_delta * (jnp.log(target_delta) - jnp.log(jnp.maximum(delta, jnp.finfo(delta.dtype).tiny))), -1)
        return cfg.flow_loss_coeff * flow + cfg.knot_loss_coeff * kl

    @override
    def sample_actions(self, rng, observation, *, num_steps=None, noise=None):
        """Dense actions at the control rate (B, action_horizon, action_dim), padded like OpenPI."""
        cfg = self.cfg
        observation = _model.preprocess_observation(None, observation, train=False)
        cond, memory, mask = self.encode(observation)
        b = cond.shape[0]
        horizon = self._horizon(b)
        knots, _ = self.predict_knots(cond, horizon)
        cp = self.sample_control_points(rng, self._context(cond, knots, horizon), memory, mask, num_steps or cfg.num_ode_steps)
        times = self._times(b)
        dense = bs.evaluate(times / times[:, -1:], knots, cp, cfg.degree)
        return jnp.pad(dense, ((0, 0), (0, 0), (0, self.action_dim - cfg.plugin_action_dim)))

    def sample_spline(self, rng, observation, num_steps=None):
        """Control points, knots, dense commands, the adaptive query interval and how many commands to execute.

        execute_steps = ceil(query_interval / control_dt), extended so that a gripper switch predicted anywhere
        in the chunk is executed in this prefix (a switch left after the prefix can be re-predicted after the
        prefix forever when the arm stands still and the next observation looks the same), capped at the
        chunk length. The client executes dense_actions[:, :execute_steps] and then queries again.
        """
        cfg = self.cfg
        observation = _model.preprocess_observation(None, observation, train=False)
        cond, memory, mask = self.encode(observation)
        b = cond.shape[0]
        horizon = self._horizon(b)
        knots, delta = self.predict_knots(cond, horizon)
        cp = self.sample_control_points(rng, self._context(cond, knots, horizon), memory, mask, num_steps or cfg.num_ode_steps)
        interval = self.timing_interval(cond, delta, knots, horizon, cp)
        times = self._times(b)
        dense = bs.evaluate(times / times[:, -1:], knots, cp, cfg.degree)
        steps = jnp.ceil(interval[:, 0] / cfg.control_dt - 1e-4).astype(jnp.int32)
        state = labels.gripper_open_state(dense[..., 6])[..., 0] >= cfg.gripper_threshold
        switch = state[:, 1:] != state[:, :-1]  # switch[j]: state changes at command j + 1
        first = jnp.argmax(switch, axis=-1) + 1
        steps = jnp.where(jnp.any(switch, -1), jnp.maximum(steps, first + 1), steps)
        steps = jnp.clip(steps, 1, self.action_horizon)
        return dict(control_points=cp, knots=knots, query_interval=interval, execute_steps=steps,
                    dense_actions=jnp.pad(dense, ((0, 0), (0, 0), (0, self.action_dim - cfg.plugin_action_dim))))
