"""Fit timing heads offline on dump_timing_features.py output and compare them on held-out episodes.

Split by episode (seed 0): 80 % train / 10 % val (model selection, early stopping) / 10 % test (reported).
Train rows: all passes (pass 0 plain + augmented passes) of train episodes. Val/test rows: pass 0 only
(no augmentation, as at inference). Loss = the timing stage's: (log interval - log label)^2 with
interval = lo + (hi - lo) * sigmoid(head(x)), lo = 0.2 s, hi = 1.5 s.

Baselines on the same rows: constant (train mean of log label) and the current freq_head (`interval_old`;
it was trained on ALL episodes, so its val/test numbers are in-sample, i.e. optimistic).

Usage: external/openpi/.venv/bin/python adaspe_train/fit_timing_heads.py <feat_dir> [<feat_dir> ...] [--quick]
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import torch

LO, HI = 0.2, 1.5


def load(feat_dir):
    passes = [dict(np.load(p)) for p in sorted(glob.glob(f"{feat_dir}/pass*.npz"))]
    assert passes, feat_dir
    return passes


def build(p, which):
    """Feature matrix for one pass. 'orig' = exactly freq_head's input (log horizon is a constant, dropped)."""
    b = p["target"].shape[0]
    blocks = {
        "orig": [p["cond"], p["delta"], p["probes"].reshape(b, -1)],
        "state": [p["state"]],
        "dense": [p["dense"].reshape(b, -1)],
        "prefix": [p["views_mean"].reshape(b, -1).astype(np.float32), p["text_feat"].astype(np.float32)],
    }
    return np.concatenate([x for k in which for x in blocks[k]], -1).astype(np.float32)


def split_episodes(ep, seed=0):
    u = np.unique(ep)
    r = np.random.default_rng(seed).permutation(u)
    n_val = max(1, round(0.1 * len(u)))
    val, test = set(r[:n_val]), set(r[n_val:2 * n_val])
    return (np.array([e not in val and e not in test for e in ep]), np.array([e in val for e in ep]),
            np.array([e in test for e in ep]))


class Head(torch.nn.Module):
    def __init__(self, d_in, hidden, depth, dropout):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(d, hidden), torch.nn.SiLU(), torch.nn.Dropout(dropout)]
            d = hidden
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return torch.log(LO + (HI - LO) * torch.sigmoid(self.net(x)[:, 0]))


def loss_np(log_pred, log_t):
    return float(np.mean((log_pred - log_t) ** 2))


def fit(xtr, ytr, xva, yva, xte, yte, hidden, depth, dropout, lr, wd, epochs, dev, bs=256):
    torch.manual_seed(0)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    t = lambda a: torch.as_tensor((a - mu) / sd, device=dev)  # noqa: E731
    Xtr, Xva, Xte = t(xtr), t(xva), t(xte)
    Ytr = torch.as_tensor(ytr, device=dev)
    m = Head(xtr.shape[1], hidden, depth, dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best = (np.inf, None, -1)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(len(Xtr) // bs):
            j = perm[i * bs:(i + 1) * bs]
            loss = torch.mean((m(Xtr[j]) - Ytr[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            va = loss_np(m(Xva).cpu().numpy(), yva)
            if va < best[0]:
                best = (va, loss_np(m(Xte).cpu().numpy(), yte), ep)
    with torch.no_grad():
        tr = loss_np(m(Xtr[:20000]).cpu().numpy(), ytr[:20000])
    return dict(val=best[0], test=best[1], best_epoch=best[2], train_end=tr)


def run(feat_dir, quick, dev):
    passes = load(feat_dir)
    ep = passes[0]["episode"]
    tr, va, te = split_episodes(ep)
    logt = [np.log(p["target"]).astype(np.float32) for p in passes]
    p0 = passes[0]
    res = dict(feat_dir=feat_dir, n_frames=len(ep), n_episodes=int(len(np.unique(ep))), n_passes=len(passes))
    c = float(np.mean(np.concatenate([lt[tr] for lt in logt])))
    res["const"] = dict(val=loss_np(np.full(va.sum(), c), logt[0][va]), test=loss_np(np.full(te.sum(), c), logt[0][te]))
    res["current_head"] = dict(
        val=loss_np(np.log(p0["interval_old"][va]), logt[0][va]), test=loss_np(np.log(p0["interval_old"][te]), logt[0][te]),
        all_passes_all_rows=float(np.mean([loss_np(np.log(p["interval_old"]), lt) for p, lt in zip(passes, logt)])))
    print(json.dumps({k: res[k] for k in ("n_frames", "n_episodes", "n_passes", "const", "current_head")}), flush=True)

    feature_sets = [("orig",), ("orig", "state"), ("orig", "state", "dense"), ("orig", "state", "prefix")]
    grid = dict(hidden=[128, 512], depth=[2, 3], dropout=[0.0, 0.2], lr=[1e-4, 1e-3], wd=[1e-4, 1e-2])
    if quick:
        grid = dict(hidden=[128, 512], depth=[2, 3], dropout=[0.1], lr=[1e-3], wd=[1e-2])
    epochs = 15 if quick else 30
    runs = []
    for fs in feature_sets:
        xtr = np.concatenate([build(p, fs)[tr] for p in passes])
        ytr = np.concatenate([lt[tr] for lt in logt])
        x0 = build(p0, fs)
        for vals in itertools.product(*grid.values()):
            hp = dict(zip(grid, vals))
            t0 = time.time()
            r = fit(xtr, ytr, x0[va], logt[0][va], x0[te], logt[0][te], epochs=epochs, dev=dev, **hp)
            r.update(features="+".join(fs), d_in=xtr.shape[1], secs=round(time.time() - t0, 1), **hp)
            runs.append(r)
            print(json.dumps(r), flush=True)
    # The original recipe's head (128 hidden, 2 layers) at the timing stage's LR, for an apples-to-apples check.
    xtr = np.concatenate([build(p, ("orig",))[tr] for p in passes])
    ytr = np.concatenate([lt[tr] for lt in logt])
    x0 = build(p0, ("orig",))
    r = fit(xtr, ytr, x0[va], logt[0][va], x0[te], logt[0][te], hidden=128, depth=2, dropout=0.0, lr=2.5e-5,
            wd=1e-10, epochs=epochs, dev=dev)
    r.update(features="orig", note="original head + OpenPI LR 2.5e-5")
    runs.append(r)
    print(json.dumps(r), flush=True)
    res["runs"] = runs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dirs", nargs="+")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in args.feat_dirs:
        res = run(d.rstrip("/"), args.quick, dev)
        out = args.out or f"{d.rstrip('/')}/fit_results{'_quick' if args.quick else ''}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        best = {}
        for r in res["runs"]:
            if r["features"] not in best or r["val"] < best[r["features"]]["val"]:
                best[r["features"]] = r
        print(f"\n== {d}: const test {res['const']['test']:.4f}, current head test {res['current_head']['test']:.4f} (in-sample)")
        for k, r in sorted(best.items(), key=lambda kv: kv[1]["val"]):
            print(f"  {k:28s} val {r['val']:.4f} test {r['test']:.4f}  "
                  f"(h={r.get('hidden')} d={r.get('depth')} do={r.get('dropout')} lr={r.get('lr')} wd={r.get('wd')})")


if __name__ == "__main__":
    main()
