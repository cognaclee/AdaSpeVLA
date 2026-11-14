"""Dump the timing head's inputs (and richer candidates) for every frame, so timing heads can be fit offline.

Everything the timing stage trains on is frozen (encode / knots / control points under stop_gradient; only
freq_head trains), so its inputs are a fixed function of (frame, image augmentation, ODE noise). This runs that
function once per frame and pass and saves it; tests of new heads, learning rates, input sets then take minutes
instead of a 1 GPU-hour timing run each.

Usage (repo root, 1 GPU):
  TASK=insert_tube adaspe_train/run_py.sh adaspe_train/dump_timing_features.py --generator pi05_expert --passes 4
Params come from <TAG>_timing_from40k/formal/9999 (the Action 39999 params + the trained freq_head), so the saved
`interval_old` is exactly the current head's prediction and doubles as a parity check against its training loss.

Pass 0: no image augmentation (inference inputs). Passes 1..: train=True augmentation as in the timing stage.
Each pass draws new ODE noise. Saved per pass (npz, row i = dataset index i):
  cond (B,512) delta (B,S) knots (B,K+4) probes (B,16,7) [freq_head input, with log horizon = const]
  dense (B,H,7) predicted commands; state (B,32) normalized proprio; views_mean (B,V,2048) / text_feat (B,2048)
  mean-pooled prefix hidden states (fp16); interval_old (B,) current head; target (B,) label; episode (B,)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs  # noqa: E402
import tasks  # noqa: E402

GEN = {"flow_head": "", "pi05_expert": "_expert"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=list(GEN), required=True)
    ap.add_argument("--from_k", type=int, default=40)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", configs.ROOT), "adaspevla_train/timing_feats"))
    args = ap.parse_args()

    import einops
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import openpi.models.model as _model
    import openpi.training.data_loader as dl
    import torch

    from adaspe import bspline as bs
    from adaspe import labels

    tag = f"{tasks.TASK}_adaspe{GEN[args.generator]}"
    cfg = configs.timing_config(args.from_k * 1000, args.generator)
    mcfg = cfg.model
    ckpt = f"{configs.CKPT}/{tag}_timing_from{args.from_k}k/formal/{configs.TIMING_STEPS - 1}/params"
    out_dir = f"{args.out}/{tag}_from{args.from_k}k"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[dump] {tag} params={ckpt} -> {out_dir}", flush=True)

    dc = cfg.data.create(cfg.assets_dirs, mcfg)
    raw = dl.create_torch_dataset(dc, mcfg.action_horizon, mcfg)
    ds = dl.transform_dataset(raw, dc)
    episode = np.asarray(raw._dataset.hf_dataset["episode_index"], np.int32)
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    assert len(episode) == len(ds), (len(episode), len(ds))

    # The plugin's Seq layers have int keys; the checkpoint has them as strings (PluginCheckpointWeightLoader maps them).
    shapes = nnx.state(nnx.eval_shape(mcfg.create, jax.random.key(0))).to_pure_dict()
    params = configs.PluginCheckpointWeightLoader(ckpt).load(shapes)
    model = mcfg.load(params, remove_extra_params=False)  # intersect_trees would stringify the int layer keys
    del params
    graphdef, state = nnx.split(model)

    @jax.jit
    def feats(state, rng, obs_dict, actions, aug):
        m = nnx.merge(graphdef, state)
        pre_rng, ode_rng = jax.random.split(rng)
        obs = _model.Observation.from_dict(obs_dict)
        obs = jax.lax.cond(
            aug, lambda o: _model.preprocess_observation(pre_rng, o, train=True),
            lambda o: _model.preprocess_observation(None, o, train=False), obs)
        # == AdaSpePi05.encode, keeping the pooled prefix features it computes on the way.
        hidden, slices, n_img, kv_cache, prefix_mask = m._prefix(obs)
        b = hidden.shape[0]
        views = jnp.stack([hidden[:, slices[k][0]: slices[k][1]] for k in mcfg.views], 1)
        valid = obs.tokenized_prompt_mask[..., None].astype(jnp.float32)
        text_feat = (hidden[:, n_img:] * valid).sum(1) / jnp.maximum(valid.sum(1), 1.0)
        views_mean = views.mean(2)
        vision = m.fusion_proj(views_mean.reshape(b, -1))
        text = m.text_proj(text_feat)
        gate = m.cross_gate(jnp.concatenate([vision, text], -1))
        cond = m.final_proj(gate * vision + (1 - gate) * text)
        if m._expert:
            obs_ctx = (kv_cache, prefix_mask)
        else:
            mem = (m.token_proj(views) + m.view_embedding.value).reshape(b, -1, mcfg.cond_dim)
            memory = jnp.concatenate([mem, m.text_token_proj(text_feat)[:, None]], 1)
            obs_ctx = (memory, jnp.ones(memory.shape[:2], bool))
        horizon = m._horizon(b)
        knots, delta = m.predict_knots(cond, horizon)
        cp = m.sample_control_points(ode_rng, cond, knots, horizon, obs_ctx, mcfg.timing_num_ode_steps)
        interval = m.timing_interval(cond, delta, knots, horizon, cp)
        probes_t = jnp.broadcast_to(jnp.linspace(0, 1, mcfg.timing_probe_count), (b, mcfg.timing_probe_count))
        pv = bs.evaluate(probes_t, knots, cp, mcfg.degree)
        probes = jnp.concatenate([pv[..., :6], m._gripper01(pv)], -1)
        times = m._times(b)
        dense = bs.evaluate(times / times[:, -1:], knots, cp, mcfg.degree)
        target = labels.timing_target(actions, mcfg)
        return dict(cond=cond, delta=delta, knots=knots, probes=probes, dense=dense, state=obs.state,
                    views_mean=views_mean.astype(jnp.float16), text_feat=text_feat.astype(jnp.float16),
                    interval_old=interval[:, 0], target=target[:, 0])

    @jax.jit
    def encode_ref(state, obs_dict):
        m = nnx.merge(graphdef, state)
        return m.encode(_model.preprocess_observation(None, _model.Observation.from_dict(obs_dict), train=False))[0]

    idx = np.arange(n)
    pad = (-n) % args.batch
    order = np.concatenate([idx, np.full(pad, n - 1)])  # pad the last batch with the last frame, dropped after
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, order.tolist()), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=dl._collate_fn, persistent_workers=args.workers > 0,
        multiprocessing_context="spawn" if args.workers else None, prefetch_factor=4 if args.workers else None)
    rng = jax.random.key(1234)
    for p in range(args.passes):
        path = f"{out_dir}/pass{p}.npz"
        if os.path.exists(path):
            print(f"[dump] {path} exists, skip", flush=True)
            continue
        outs, t0 = [], time.time()
        for i, batch in enumerate(loader):
            actions = batch.pop("actions")
            rng, sub = jax.random.split(rng)
            o = jax.device_get(feats(state, sub, batch, jnp.asarray(actions), jnp.asarray(p > 0)))
            outs.append(o)
            if i == 0 and p == 0:  # parity: my inline encode == the model's encode (no augmentation)
                err = float(np.max(np.abs(np.asarray(encode_ref(state, batch)) - o["cond"])))
                print(f"[dump] parity |cond - encode()| max = {err:.2e}", flush=True)
                assert err < 1e-3, err
            if i % 50 == 0:
                print(f"[dump] pass {p} batch {i}/{len(loader)} {time.time() - t0:.0f}s", flush=True)
        res = {k: np.concatenate([o[k] for o in outs])[:n] for k in outs[0]}
        res["episode"] = episode[:n]
        loss_old = np.mean((np.log(res["interval_old"]) - np.log(res["target"])) ** 2)
        print(f"[dump] pass {p} done {time.time() - t0:.0f}s; current head loss on this pass = {loss_old:.4f}; "
              f"var(log target) = {np.var(np.log(res['target'])):.4f}", flush=True)
        np.savez(path + ".tmp.npz", **res)
        os.replace(path + ".tmp.npz", path)
    print("[dump] DONE", flush=True)


if __name__ == "__main__":
    main()
