"""Held-out check of the timing-stage learning rate, replaying the real timing runs offline on dumped features.

Mirrors the timing stage: freq_head = Linear(633->128) SiLU Linear(128->1) on unstandardized inputs
[cond, log horizon, delta, 16x7 spline probes], interval = lo + (hi-lo)*sigmoid, loss = (log interval - log label)^2,
AdamW (b2 0.95, eps 1e-8, wd 1e-10 as OpenPI), grad-norm clip 1.0, batch 32, 10k steps, EMA 0.99 (OpenPI saves EMA
params). Schedules: old = warmup 1k, cosine to 2.5e-6 over 30k, peak 2.5e-5; new = warmup 1k, cosine to peak/10 over 10k.

Train rows: all dumped passes of 80 % of episodes (random augmentation + ODE noise draws, as the real run sees).
Held-out: pass 0 (no augmentation) of the other 20 % of episodes. Reports final-step (= deployed) losses.
Usage: python adaspe_train/lr_check.py <feat_dir> [...]
"""
import glob
import math
import sys

import numpy as np
import torch

LO, HI = 0.2, 1.5
torch.set_num_threads(8)


def feats(p):
    b = len(p["target"])
    return np.concatenate([p["cond"], np.full((b, 1), math.log(1.5), np.float32), p["delta"], p["probes"].reshape(b, -1)], -1)


def sched(step, peak, decay_steps, end):
    if step < 1000:
        return peak * (step + 1) / 1001
    t = min(step - 1000, decay_steps - 1000) / (decay_steps - 1000)
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * t))


def run(xtr, ytr, xho, yho, peak, decay_steps, end, steps=10_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))
    ema = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.AdamW(net.parameters(), lr=peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    f = lambda m, x: torch.log(LO + (HI - LO) * torch.sigmoid(m(x)[:, 0]))  # noqa: E731
    Xtr, Ytr = torch.as_tensor(xtr), torch.as_tensor(ytr)
    curve = []
    for s in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = sched(s, peak, decay_steps, end)
        j = torch.randint(len(Xtr), (32,), generator=g)
        loss = torch.mean((f(net, Xtr[j]) - Ytr[j]) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.mul_(0.99).add_(p.detach(), alpha=0.01)
        if (s + 1) % 2000 == 0:
            curve.append(evaluate(net, ema, f, xtr, ytr, xho, yho))
    return curve


def evaluate(net, ema, f, xtr, ytr, xho, yho):
    m = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))
    with torch.no_grad():
        for d, e in zip(m.parameters(), ema):
            d.copy_(e)
        tr = float(torch.mean((f(m, torch.as_tensor(xtr[:20000])) - torch.as_tensor(ytr[:20000])) ** 2))
        ho = float(torch.mean((f(m, torch.as_tensor(xho)) - torch.as_tensor(yho)) ** 2))
    return tr, ho


def main():
    for d in sys.argv[1:]:
        passes = [dict(np.load(p)) for p in sorted(glob.glob(f"{d}/pass*.npz"))]
        ep = passes[0]["episode"]
        u = np.random.default_rng(0).permutation(np.unique(ep))
        held = np.isin(ep, u[: round(0.2 * len(u))])
        xtr = np.concatenate([feats(p)[~held] for p in passes]).astype(np.float32)
        ytr = np.concatenate([np.log(p["target"][~held]) for p in passes]).astype(np.float32)
        xho = feats(passes[0])[held].astype(np.float32)
        yho = np.log(passes[0]["target"][held]).astype(np.float32)
        print(f"== {d.rstrip('/').split('/')[-1]}: {len(passes)} passes, {len(np.unique(ep))} episodes "
              f"({held.sum()} held-out frames from {round(0.2 * len(u))} episodes); "
              f"constant predictor held-out {np.mean((yho - ytr.mean()) ** 2):.4f}", flush=True)
        for name, peak, dsteps, end in [("2.5e-5 (old)", 2.5e-5, 30_000, 2.5e-6), ("3e-4", 3e-4, 10_000, 3e-5), ("1e-3", 1e-3, 10_000, 1e-4)]:
            for seed in (0, 1):
                c = run(xtr, ytr, xho, yho, peak, dsteps, end, seed=seed)
                print(f"  lr {name:13s} seed {seed}: train/held-out at 2k,4k,..,10k: "
                      + "  ".join(f"{a:.3f}/{b:.3f}" for a, b in c), flush=True)


if __name__ == "__main__":
    main()
