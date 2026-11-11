import os, sys; os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # adaspe_train/
import jax, jax.numpy as jnp, flax.nnx as nnx
import tasks
from adaspe.model import AdaSpePi05Config
H = tasks.ACTION_HORIZON
for stage in ["action", "timing"]:
    cfg = AdaSpePi05Config(paligemma_variant="dummy", action_horizon=H, num_control_points=tasks.NUM_CONTROL_POINTS,
                           min_query_interval=tasks.MIN_QUERY_INTERVAL, max_query_interval=tasks.MAX_QUERY_INTERVAL,
                           stage=stage, flow_hidden_dim=48, flow_num_layers=1, timing_label_mode=tasks.LABEL_MODE,
                           knot_label_mode=tasks.LABEL_MODE, knot_motion_norm=tasks.KNOT_MOTION_NORM,
                           timing_motion_scale=(0.02,) * 6, e_crit=10.0)
    model = cfg.create(jax.random.key(0))
    obs, act = cfg.fake_obs(2), cfg.fake_act(2)
    grip = jnp.array(([1.] * 3 + [-1.] * 4 + [1.] * 3 + [0.] * 2 + [1.] * H)[:H])  # open/close/open + "no command" zeros
    act = act.at[..., 6].set(grip)
    loss = model.compute_loss(jax.random.key(1), obs, act, train=True)
    print(stage, "H", H, "K", cfg.num_control_points, "loss", loss.shape, loss)
    assert bool(jnp.all(jnp.isfinite(loss)))
    filt = nnx.All(nnx.Param, nnx.Not(cfg.get_freeze_filter()))
    graphdef, params, rest = nnx.split(model, filt, ...)
    def f(p):
        m = nnx.merge(graphdef, p, rest); return m.compute_loss(jax.random.key(1), obs, act, train=True).mean()
    g = jax.grad(f)(params)
    flat = {"/".join(map(str, k)): float(jnp.abs(v.value).sum()) for k, v in nnx.to_flat_state(g)} if hasattr(nnx, "to_flat_state") else {"/".join(map(str,k)): float(jnp.abs(v.value).sum()) for k, v in g.flat_state().items()}
    nz = sorted({k.split("/")[0] for k, v in flat.items() if v > 0}); print(stage, "params with grad:", len(flat), "top-level nonzero:", nz)
    if stage == "action":
        sp = model.sample_spline(jax.random.key(2), obs)
        print("sample", model.sample_actions(jax.random.key(2), obs).shape, {k: v.shape for k, v in sp.items()},
              "query_interval", sp["query_interval"][:, 0], "execute_steps", sp["execute_steps"])
        assert sp["control_points"].shape[1] == cfg.num_control_points and sp["dense_actions"].shape[1] == H
        assert bool(jnp.all((sp["execute_steps"] >= 1) & (sp["execute_steps"] <= H)))
