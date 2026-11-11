"""Compare recomputed norm stats with the baseline model repo's norm_stats.json (README run order, step 1).
Usage: python tests/compare_norm_stats.py <recomputed norm_stats.json> <baseline norm_stats.json>"""
import json
import sys

import numpy as np

ours, base = (json.load(open(p))["norm_stats"] for p in sys.argv[1:3])
exact, worst = True, 0.0
for key in sorted(set(ours) | set(base)):
    for stat in sorted(set(ours.get(key, {})) | set(base.get(key, {}))):
        a, b = np.asarray(ours[key][stat], np.float64), np.asarray(base[key][stat], np.float64)
        if a.shape != b.shape:
            print(f"{key}/{stat}: shape {a.shape} vs {b.shape}")
            exact = False
            continue
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        same = bool(np.array_equal(a, b))
        exact &= same
        worst = max(worst, d)
        print(f"{key}/{stat:5s} shape={a.shape} identical={same} max|diff|={d:.3e}")
print(f"RESULT identical={exact} max|diff|={worst:.3e}")
sys.exit(0 if worst < 1e-5 else 1)
