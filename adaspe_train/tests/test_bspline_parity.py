"""JAX B-spline port vs PyTorch reference (v1.3 snapshot), float64 on CPU."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
REF = os.environ.get("ADASPEVLA_REF", f"{ROOT}/reference_adaspevla_source")  # original AdaSpeVLA PyTorch code
sys.path.insert(0, f"{ROOT}/adaspe_train")
sys.path.insert(0, f"{REF}")
import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
B = f"{REF}/adaspevla/bspline"
import types; pkg = types.ModuleType("refb"); pkg.__path__ = [B]; sys.modules["refb"] = pkg
core = load("refb.bspline_core", f"{B}/bspline_core.py")
lk = load("refb.learnable_knots", f"{B}/learnable_knots.py")
cf = load("refb.curve_fitting", f"{B}/curve_fitting.py")
from adaspe import bspline as jb

rng = np.random.default_rng(0)
Bn, T, D, K, deg = 6, 10, 7, 8, 3
I = K - deg
t = np.tile(np.arange(T) * 0.1, (Bn, 1))
u = t / t[:, -1:]
acts = rng.normal(size=(Bn, T, D)).cumsum(1) * 0.3
grip = (rng.random((Bn, T, 1)) > 0.5).astype(float)
tt = lambda a: torch.tensor(a, dtype=torch.float64)
def check(name, a, b, tol=1e-9):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))); print(f"{name:28s} max|diff|={err:.2e}"); assert err < tol, name

delta_ref = lk.compute_curvature_density_target(tt(acts[..., :6]), I, tt(t), 0.01, gripper=tt(grip), gripper_gain=10.0)
delta_j = jb.density_span_target(jnp.array(acts[..., :6]), jnp.array(t), I, 0.01, gripper=jnp.array(grip), gripper_gain=10.0)
check("density_span_target", delta_ref, delta_j)
knots_ref = lk.intervals_to_knots(delta_ref, deg); knots_j = jb.intervals_to_knots(delta_j, deg)
check("intervals_to_knots", knots_ref, knots_j)
check("basis", core.eval_cox_de_boor_basis(tt(u), knots_ref, K, deg), jb.basis(jnp.array(u), knots_j, K, deg))
cp_ref, rec_ref = cf.fit_bspline_control_points_least_squares(tt(acts), K, deg, knots=knots_ref, query_times=tt(u))
cp_j, rec_j = jb.fit_control_points(jnp.array(acts), knots_j, jnp.array(u), K, deg)
check("fit_control_points", cp_ref, cp_j, 1e-7); check("fit recon mse", rec_ref, rec_j, 1e-7)
check("evaluate", core.evaluate_bspline(tt(u), knots_ref, cp_ref, deg), jb.evaluate(jnp.array(u), knots_j, cp_j, deg), 1e-7)
e_crit = float(np.quantile(lk.motion_variation(tt(acts[..., :6]), tt(t)).numpy(), 0.7))
q_ref = lk.query_interval_target(delta_ref, 0.9, 0.2, 0.8, tt(t), None, tt(grip), tt(acts[..., :6]), e_crit=e_crit)
q_j = jb.query_interval_target(jnp.array(t), jnp.array(acts[..., :6]), jnp.array(grip), 0.9, 0.2, 0.8, e_crit)
check("query_interval_target", q_ref, q_j)
print("ALL PARITY CHECKS PASSED")
