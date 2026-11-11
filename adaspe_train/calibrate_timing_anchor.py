"""Calibrate knot / timing labels for the TASK env var's dataset, with the recipe in tasks.py.

As AdaSpeVLA franka v1.x (datasets/bspline_dataset.py, action_contract.calibrate_motion_scales,
pose_math.anchor_deltas):
  offsets      = measured pose relative to the window's first pose (reset frame; position difference,
                 rotation vector of R_j R_anchor^T), pose = state[4:7] + Euler XYZ state[7:10],
  motion_scale = 95th percentile of |offset| per channel over all window samples,
  e_crit       = 95th percentile of motion_variation(offset / motion_scale).
Windows: tasks.ACTION_HORIZON frames at 10 Hz; episode-end windows repeat the last frame (LeRobot clamping in
OpenPI, the same as AdaSpeVLA's pad_episode_end hold). The labels themselves come from adaspe/labels.py, the
same code the model trains with (gripper events per tasks.LABEL_MODE).

Writes tasks.calibration_path() (json) and tasks.timing_labels_path() (labels in dataset order, for
tests/check_timing_labels.py). Copy motion_scale / e_crit into tasks.py.
Usage: TASK=insert_tube python adaspe_train/calibrate_timing_anchor.py
"""
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from adaspe import bspline as bs  # noqa: E402
from adaspe import labels  # noqa: E402
import tasks  # noqa: E402  (TASK env var picks the dataset)

ROOT = tasks.ROOT
H, DT = tasks.ACTION_HORIZON, tasks.CONTROL_DT


def windows(x):
    idx = np.minimum(np.arange(len(x))[:, None] + np.arange(H)[None], len(x) - 1)
    return x[idx]


files = sorted(glob.glob(f"{ROOT}/data/lerobot/{tasks.SPEC['repo_id']}/data/chunk-*/*.parquet"))
tables = [pq.read_table(f, columns=["state", "actions"]) for f in files]
col = lambda t, c: np.stack(t.column(c).to_numpy(zero_copy_only=False)).astype(np.float64)  # noqa: E731
win = np.concatenate([windows(col(t, "state")) for t in tables])  # (N,H,19), dataset order
act = np.concatenate([windows(col(t, "actions")) for t in tables])  # (N,H,7) raw commands
pose, width = win[..., 4:10], win[..., 0:1]

# Cross-check the JAX geometry against scipy / AdaSpeVLA's anchor_deltas.
rng = np.random.default_rng(0)
e = rng.uniform(-np.pi, np.pi, (500, 3))
assert np.allclose(bs.euler_xyz_to_matrix(jnp.array(e)), Rotation.from_euler("xyz", e).as_matrix(), atol=1e-12)
v = rng.normal(size=(500, 3))
v = v / np.linalg.norm(v, axis=-1, keepdims=True) * rng.uniform(1e-8, 2.5, (500, 1))
assert np.allclose(bs.matrix_to_rotvec(bs.rotvec_to_matrix(jnp.array(v))), v, atol=1e-9)
off = np.asarray(bs.measured_anchor_offsets(jnp.array(pose)))
for n in rng.choice(len(win), 300, replace=False):
    r = Rotation.from_euler("xyz", pose[n, :, 3:])
    ref = np.concatenate([pose[n, :, :3] - pose[n, 0, :3], (r * r[0].inv()).as_rotvec()], -1)
    assert np.allclose(off[n], ref, atol=1e-9), n

motion_scale = np.maximum(np.quantile(np.abs(off.reshape(-1, 6)), 0.95, axis=0), 1e-6)
times = jnp.broadcast_to(jnp.arange(H) * DT, (len(off), H))
mv = np.asarray(bs.motion_variation(jnp.array(off / motion_scale), times))
e_crit = max(float(np.quantile(mv, 0.95)), 1e-6)

# Labels with the training code: model actions = [raw commands (~ normalized for the gripper: q01=-1,
# q99~1), measured window in TIMING_CHANNELS]; only the gripper command and the measured window are read.
import configs  # noqa: E402  (after the calibration numbers exist: builds the recipe's model config)

cfg = configs.model_config("timing", e_crit=e_crit, motion_scale=motion_scale)
model_actions = np.zeros((len(win), H, 14), np.float32)
model_actions[..., :7] = act
model_actions[..., labels.TIMING_CHANNELS] = np.concatenate([pose, width], -1)
lab = np.asarray(labels.timing_target(jnp.asarray(model_actions), cfg))[:, 0]
n_spans = cfg.num_control_points - cfg.degree
spans = np.asarray(labels.knot_span_target(jnp.asarray(model_actions), cfg)) * n_spans  # 1 = uniform
vals, counts = np.unique(np.round(lab, 3), return_counts=True)
out = {
    "task": tasks.TASK,
    "repo_id": tasks.SPEC["repo_id"],
    "action_horizon": H,
    "num_control_points": cfg.num_control_points,
    "query_interval": [cfg.min_query_interval, cfg.max_query_interval],
    "label_mode": cfg.timing_label_mode,
    "knot_motion_norm": cfg.knot_motion_norm,
    "n_windows": int(len(win)),
    "motion_scale": motion_scale.tolist(),
    "e_crit": e_crit,
    "label_distribution": {f"{v:.1f}s": round(float(c) / len(lab), 4) for v, c in zip(vals, counts)},
    "label_mean_s": round(float(lab.mean()), 4),
    "knot_spans_near_uniform": round(float(np.mean(np.all(np.abs(spans - 1) < 0.15, axis=1))), 4),
    "knot_span_ratio_p5_p50_p95": np.round(np.quantile(spans, [0.05, 0.5, 0.95]), 3).tolist(),
    "gripper_width_range": [float(width.min()), float(width.max())],
}
print(json.dumps(out, indent=1))
for path in (tasks.calibration_path(), tasks.timing_labels_path()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(tasks.calibration_path(), "w"), indent=1)
np.save(tasks.timing_labels_path(), lab.astype(np.float32))  # for the pipeline check
