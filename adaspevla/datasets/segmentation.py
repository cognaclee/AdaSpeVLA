"""Difficulty-adaptive segmentation of whole demonstrations for B-spline fitting.

An offline tool: it reads raw demonstrations (never modifies them) and returns
segment boundaries such that each segment is fitted by one clamped B-spline
with a fixed number of control points, in the paper's action space.

Precision demand. The per-interval demand reuses the manuscript's knot-density
form (Eq. density / grip_density) but answers a different question, namely how
accurately a region must be reproduced:
``rho_j = 1 + beta_g tanh(e^g_j/e^g_ref) + beta_v (1 - tanh(v_j/v_ref))
+ beta_a tanh(e_j/e_ref) + beta_r max(r_j, r_{j+1})``.
``e^g_j`` is the gripper-width rate of Eq. grip_density; ``r_j`` are optional
criticality annotations (contact, insertion). ``v_j`` is the normalized
end-effector speed (max over arms), so that slow approach and alignment phases
can demand precision; on RoboTwin planner demos every waypoint stop is slow, so
this cue mostly shortens ordinary transport and ``beta_v`` defaults to zero. ``e_j``
is the paper's normalized motion acceleration, with the middle sample of each
triple as the anchor of Eq. anchor_delta so it does not depend on where a window
starts. It measures representation complexity rather than precision, and on
RoboTwin planner demos it is large throughout fast free-space transport, so its
weight ``beta_a`` defaults to zero; it is still reported and still places the
knots inside every segment (Eq. target_spans). Optionally rho is max-filtered
over ``dilation_s`` seconds, so the samples just before and after a difficult
instant (the approach to a grasp, the settle after it) inherit its demand.

Keyframes. By default the episode is first cut at keyframes, where the arms
come to rest or a gripper starts or stops changing, so no segment straddles a
grasp or a motion stop; the adaptive split below runs inside each keyframe
interval.

Segments. The precision demand acts twice. Like the paper's inverse-density
spans, it caps each segment's density mass ``sum_j rho_j dt_j`` at
``mass_budget_s`` (seconds at unit demand), so a segment in a region of demand
rho lasts at most ``mass_budget_s / rho``. It also sets a per-sample tolerance
``tol / rho``. From each boundary a segment grows as long as the B-spline fit
(anchor-relative offsets re-anchored at the segment start, motion divided by
the scales D, gripper 2w-1, teacher knots from Eq. target_spans,
endpoint-pinned ridge fit) keeps every sample within its tolerance, the mass
stays within budget, and the length stays within
``[min_samples, max_samples]``. Easy free-space motion therefore yields long
segments, while grasps, slow alignment and shapes that K control points cannot
follow (sharp turns, stop-and-go) yield short ones. Adjacent
segments share their boundary sample, which is the next segment's anchor.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d

from ..benchmarks.pose_math import base_rotation, wxyz_to_rotation
from ..bspline.bspline_core import eval_cox_de_boor_basis
from ..bspline.curve_fitting import fit_bspline_control_points_least_squares
from ..bspline.learnable_knots import compute_curvature_density_target, intervals_to_knots
from .action_contract import calibrate_motion_scales, window_offsets

SEGMENTS_FORMAT = "adaspevla_segments_v1"


@dataclass
class SegmentConfig:
    num_control_points: int = 8
    degree: int = 3
    tolerance: float = 0.1
    min_samples: int = 8
    max_samples: int = 64
    mass_budget_s: float = 3.0
    keyframes: bool = True
    stop_fraction: float = 0.05
    gripper_eps: float = 1e-3
    gripper_gain: float = 10.0
    gripper_ref: float = 1.0
    speed_gain: float = 0.0
    speed_ref: float = None
    accel_gain: float = 0.0
    accel_ref: float = 1.0
    criticality_gain: float = 10.0
    knot_density_gain: float = 10.0
    dilation_s: float = 0.15
    min_interval_eps: float = 0.01
    reg_lambda: float = 1e-4

    def validate(self):
        if self.num_control_points <= self.degree or self.degree < 0:
            raise ValueError("Need num_control_points > degree >= 0")
        if not 0 <= self.stop_fraction < 1 or self.gripper_eps < 0:
            raise ValueError("Invalid keyframe thresholds")
        if not self.tolerance > 0 or not self.mass_budget_s > 0:
            raise ValueError("tolerance and mass_budget_s must be positive")
        if not 2 <= self.min_samples <= self.max_samples:
            raise ValueError("Need 2 <= min_samples <= max_samples")
        gains = (self.gripper_gain, self.speed_gain, self.accel_gain,
                 self.criticality_gain, self.knot_density_gain, self.dilation_s)
        if min(gains) < 0:
            raise ValueError("Gains and dilation must be nonnegative")
        if min(self.gripper_ref, self.accel_ref) <= 0 or (
            self.speed_ref is not None and not self.speed_ref > 0
        ):
            raise ValueError("Reference rates must be positive")
        num_spans = self.num_control_points - self.degree
        if not 0 <= self.min_interval_eps < 1 / num_spans:
            raise ValueError("Invalid minimum knot span")
        return self


@dataclass
class Trajectory:
    """One episode in the pose contract: poses (T,arms,7 xyz+wxyz), widths (T,arms) in [0,1]."""

    poses: np.ndarray
    widths: np.ndarray
    times: np.ndarray
    base_wxyz: np.ndarray
    source: str = ""
    arms: tuple = ("right",)

    def __post_init__(self):
        self.poses = np.asarray(self.poses, dtype=np.float64)
        self.widths = np.asarray(self.widths, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        T = len(self.times)
        if self.poses.ndim == 2:
            self.poses = self.poses[:, None]
        if self.widths.ndim == 1:
            self.widths = self.widths[:, None]
        arms = self.poses.shape[1]
        if self.poses.shape != (T, arms, 7) or self.widths.shape != (T, arms):
            raise ValueError("Expected poses (T,arms,7) and widths (T,arms)")
        if T < 2 or (np.diff(self.times) <= 0).any() or not np.isfinite(self.times).all():
            raise ValueError("Need at least two strictly increasing timestamps")
        if not np.isfinite(self.poses).all() or not np.isfinite(self.widths).all():
            raise ValueError("Poses and widths must be finite")
        if ((self.widths < 0) | (self.widths > 1)).any():
            raise ValueError("Gripper widths must lie in [0,1]")
        base = np.asarray(self.base_wxyz, dtype=np.float64)
        self.base_wxyz = np.repeat(base[None], arms, 0) if base.shape == (4,) else base
        if self.base_wxyz.shape != (arms, 4):
            raise ValueError("base_wxyz must be one quaternion or one per arm")
        self.arms = tuple(self.arms)

    @property
    def world_from_base(self):
        return [base_rotation(q) for q in self.base_wxyz]


def episode_motion_scales(trajectories, horizon=16, floor=1e-6):
    """Paper scales D: 95th percentile |offset| over all windows of ``horizon`` samples."""
    if horizon < 2:
        raise ValueError("horizon must cover at least two samples")
    offsets = []
    for traj in trajectories:
        rotations = traj.world_from_base
        T = len(traj.times)
        for start in range(T - 1):
            window = traj.poses[start : min(T, start + horizon)]
            offsets.append(window_offsets(window, rotations))
    if not offsets:
        raise ValueError("No trajectories to calibrate motion scales")
    return calibrate_motion_scales(offsets, floor).numpy().astype(np.float64)


def calibrate(trajectories, horizon=16, speed_fraction=0.25):
    """Dataset-level motion scales D and speed reference v_ref.

    ``v_ref`` is ``speed_fraction`` times the 95th percentile of the normalized
    speed over all intervals, so ``1 - tanh(v/v_ref)`` is near one only when the
    end effectors move much slower than their typical transport speed.
    """
    trajectories = list(trajectories)
    scales = episode_motion_scales(trajectories, horizon)
    speeds = np.concatenate([normalized_speed(t, scales) for t in trajectories])
    reference = speed_fraction * float(np.quantile(speeds, 0.95))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Demonstrations do not move; cannot calibrate a speed reference")
    return dict(motion_scales=scales, speed_ref=reference, horizon=horizon,
                speed_fraction=speed_fraction)


def _step_offsets(poses, world_from_base, anchor, other):
    """Offsets of samples ``other`` relative to anchors ``anchor`` (Eq. anchor_delta)."""
    to_base = world_from_base.inv()
    rotations = wxyz_to_rotation(poses[:, 3:], unit_atol=1e-3)
    position = to_base.apply(poses[other, :3] - poses[anchor, :3])
    rotation = to_base.apply((rotations[other] * rotations[anchor].inv()).as_rotvec())
    return np.concatenate((position, rotation), axis=-1)


def normalized_speed(traj, motion_scales):
    """v_j = max over arms ||offset of j+1 from anchor j / D|| / dt, (T-1,)."""
    scale = np.asarray(motion_scales, dtype=np.float64).reshape(6)
    dt = np.diff(traj.times)
    T = len(dt) + 1
    per_arm = [
        np.linalg.norm(_step_offsets(traj.poses[:, arm], rot, slice(0, T - 1), slice(1, T)) / scale, axis=-1)
        for arm, rot in enumerate(traj.world_from_base)
    ]
    return np.max(per_arm, axis=0) / dt


def motion_difficulty(traj, motion_scales):
    """Paper e_j per interval (T-1,), normalized acceleration over all arms' motion channels."""
    times = traj.times
    T = len(times)
    if T < 3:
        return np.zeros(T - 1)
    scale = np.asarray(motion_scales, dtype=np.float64).reshape(6)
    dt_prev, dt_next = np.diff(times)[:-1], np.diff(times)[1:]
    anchor = slice(1, T - 1)
    squared = np.zeros(T - 2)
    for arm, rotation in enumerate(traj.world_from_base):
        poses = traj.poses[:, arm]
        # a_j and a_{j+2} anchored at sample j+1 (a_{j+1}=0); v as in Eq. density.
        before = _step_offsets(poses, rotation, anchor, slice(0, T - 2)) / scale
        after = _step_offsets(poses, rotation, anchor, slice(2, T)) / scale
        v0 = -before / dt_prev[:, None]
        v1 = after / dt_next[:, None]
        squared += (((v1 - v0) / dt_prev[:, None]) ** 2).sum(-1)
    e = np.sqrt(squared)
    return np.append(e, e[-1])


def gripper_difficulty(traj):
    """e^g_j = max over arms |w_{j+1}-w_j| / dt (Eq. grip_density), (T-1,)."""
    return (np.abs(np.diff(traj.widths, axis=0)) / np.diff(traj.times)[:, None]).max(-1)


def difficulty_density(traj, motion_scales, config, criticality=None):
    """Per-interval precision demand rho_j (T-1,) and its raw components."""
    if config.speed_gain > 0 and config.speed_ref is None:
        raise ValueError("speed_gain > 0 needs speed_ref; calibrate() it on the dataset first")
    parts = dict(
        motion_difficulty=motion_difficulty(traj, motion_scales),
        gripper_difficulty=gripper_difficulty(traj),
        speed=normalized_speed(traj, motion_scales),
    )
    rho = (
        1.0
        + config.gripper_gain * np.tanh(parts["gripper_difficulty"] / config.gripper_ref)
        + (config.speed_gain * (1 - np.tanh(parts["speed"] / config.speed_ref))
           if config.speed_gain > 0 else 0.0)
        + config.accel_gain * np.tanh(parts["motion_difficulty"] / config.accel_ref)
    )
    if criticality is not None:
        r = np.asarray(criticality, dtype=np.float64)
        if r.shape != traj.times.shape or ((r < 0) | (r > 1)).any():
            raise ValueError("criticality must be (T,) values in [0,1]")
        rho = rho + config.criticality_gain * np.maximum(r[:-1], r[1:])
    if config.dilation_s > 0 and len(rho) > 1:
        half = int(round(config.dilation_s / np.median(np.diff(traj.times))))
        if half > 0:
            rho = maximum_filter1d(rho, size=2 * half + 1, mode="nearest")
    return rho, parts


def sample_tolerance(rho, tolerance):
    """Per-sample tolerance tol / max(rho of the adjacent intervals), (T,)."""
    padded = np.concatenate((rho[:1], rho, rho[-1:]))
    return tolerance / np.maximum(padded[:-1], padded[1:])


def normalized_segment(traj, start, end, motion_scales):
    """Normalized (L, 7*arms) rows: offsets anchored at ``start`` / D, then 2w-1."""
    poses = traj.poses[start : end + 1]
    offsets = window_offsets(poses, traj.world_from_base)
    arms = poses.shape[1]
    motion = offsets.reshape(len(poses), arms, 6) / np.asarray(motion_scales).reshape(6)
    grip = 2 * traj.widths[start : end + 1, :, None] - 1
    return np.concatenate((motion, grip), -1).reshape(len(poses), 7 * arms)


def fit_segment(traj, start, end, motion_scales, config):
    """Fit one segment; returns (per-sample max abs residual (L,), knot spans (M,))."""
    values = torch.as_tensor(
        normalized_segment(traj, start, end, motion_scales), dtype=torch.float64
    )[None]
    times = torch.as_tensor(traj.times[start : end + 1], dtype=torch.float64)[None]
    arms = traj.poses.shape[1]
    blocks = values.unflatten(-1, (arms, 7))
    num_spans = config.num_control_points - config.degree
    spans = compute_curvature_density_target(
        blocks[..., :6].flatten(-2),
        num_intervals=num_spans,
        timestamps=times,
        min_interval_eps=config.min_interval_eps,
        density_gain=config.knot_density_gain,
        gripper=(blocks[..., 6] + 1) / 2,
        gripper_gain=config.gripper_gain,
        gripper_ref=config.gripper_ref,
    )
    knots = intervals_to_knots(spans, config.degree)
    local = (times - times[:, :1]) / (times[:, -1:] - times[:, :1])
    cp, _ = fit_bspline_control_points_least_squares(
        values,
        config.num_control_points,
        config.degree,
        knots,
        config.reg_lambda,
        query_times=local,
    )
    basis = eval_cox_de_boor_basis(local, knots, config.num_control_points, config.degree)
    residual = (basis @ cp - values).abs().amax(-1)[0]
    return residual.numpy(), spans[0].numpy()


def detect_keyframes(traj, motion_scales, config):
    """Mandatory boundaries: episode ends, motion stops and gripper-change edges.

    As in keyframe-based manipulation datasets, a keyframe is a sample where every
    arm comes to rest after moving (normalized speed below ``stop_fraction`` of
    the episode's 95th-percentile speed) or where a gripper width starts or stops
    changing. Keyframes closer than ``min_samples - 1`` intervals to the
    previously kept one or to the episode end are dropped. Returns sorted indices.
    """
    T = len(traj.times)
    speed = normalized_speed(traj, motion_scales)
    moving = speed > config.stop_fraction * np.quantile(speed, 0.95)
    arrivals = np.flatnonzero(moving[:-1] & ~moving[1:]) + 1
    changing = (np.abs(np.diff(traj.widths, axis=0)) > config.gripper_eps).any(-1)
    edges = np.flatnonzero(np.diff(changing.astype(int)))
    grip = edges + 1  # first sample of each change run and the sample where it ends
    if changing[0]:
        grip = np.append(grip, 0)
    candidates = sorted(set(arrivals.tolist()) | set(grip.tolist()))
    gap = config.min_samples - 1
    keep = [0]
    for k in candidates:
        if k - keep[-1] >= gap and T - 1 - k >= gap:
            keep.append(int(k))
    keep.append(T - 1)
    return keep


def segment_trajectory(traj, motion_scales, config=None, criticality=None):
    """Greedy longest-feasible segmentation; returns a JSON-serializable dict."""
    config = (config or SegmentConfig()).validate()
    T = len(traj.times)
    rho, parts = difficulty_density(traj, motion_scales, config, criticality)
    tol = sample_tolerance(rho, config.tolerance)
    masses = rho * np.diff(traj.times)

    cache = {}

    def fits(start, end):
        if (start, end) not in cache:
            residual, spans = fit_segment(traj, start, end, motion_scales, config)
            ok = bool((residual <= tol[start : end + 1] + 1e-12).all())
            cache[start, end] = (ok, residual, spans)
        return cache[start, end]

    def split(first, last):
        """Greedy longest-feasible boundaries covering samples [first, last]."""
        bounds = [first]
        while bounds[-1] < last:
            start = bounds[-1]
            shortest = min(last, start + config.min_samples - 1)
            longest = min(last, start + config.max_samples - 1)
            mass = np.cumsum(masses[start:longest])
            longest = max(shortest, start + int(np.searchsorted(mass, config.mass_budget_s, "right")))
            # Exponential then binary search for the last feasible end; fit error
            # grows with length in practice, and the result is always feasible or
            # the minimum length.
            good, step = shortest, 1
            while good < longest:
                probe = min(longest, good + step)
                if not fits(start, probe)[0]:
                    low, high = good, probe
                    while high - low > 1:
                        mid = (low + high) // 2
                        low, high = (mid, high) if fits(start, mid)[0] else (low, mid)
                    good = low
                    break
                good, step = probe, step * 2
            bounds.append(good)
        # A tail shorter than min_samples takes samples from the previous segment
        # (shortening it keeps it feasible in practice); merge only if both are short.
        if len(bounds) > 2 and bounds[-1] - bounds[-2] + 1 < config.min_samples:
            moved = bounds[-1] - config.min_samples + 1
            if moved - bounds[-3] + 1 >= config.min_samples:
                bounds[-2] = moved
            else:
                bounds.pop(-2)
        return bounds

    keyframes = (
        detect_keyframes(traj, motion_scales, config) if config.keyframes else [0, T - 1]
    )
    bounds = [0]
    for first, last in zip(keyframes[:-1], keyframes[1:]):
        bounds += split(first, last)[1:]

    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        ok, residual, spans = fits(start, end)
        segments.append(
            dict(
                start=int(start),
                end=int(end),
                t_start=float(traj.times[start]),
                t_end=float(traj.times[end]),
                duration=float(traj.times[end] - traj.times[start]),
                num_samples=int(end - start + 1),
                within_tolerance=ok,
                max_residual=float(residual.max()),
                min_tolerance=float(tol[start : end + 1].min()),
                mean_density=float(rho[start:end].mean()),
                density_mass=float(masses[start:end].sum()),
                knot_spans=[float(x) for x in spans],
            )
        )
    return dict(
        format=SEGMENTS_FORMAT,
        source=traj.source,
        arms=list(traj.arms),
        num_samples=T,
        boundaries=[int(b) for b in bounds],
        keyframes=[int(k) for k in keyframes],
        segments=segments,
        density=[float(x) for x in rho],
        **{key: [float(x) for x in value] for key, value in parts.items()},
        times=[float(x) for x in traj.times],
        config=asdict(config),
        motion_scales=[float(x) for x in np.asarray(motion_scales).reshape(6)],
    )


# --- raw readers (read-only) ------------------------------------------------


def read_robotwin_episode(path, arms=("left", "right"), profile="robotwin", base_wxyz=None,
                          sample_dt=None):
    """Poses/widths from a raw XPolicyLab v1.0 (RoboTwin 2.0) HDF5, or a converted contract file."""
    import h5py

    from ..benchmarks.xpolicylab_data import ARM_PREFIX, PROFILES, episode_sample_dt
    from .robotwin_native import xpolicylab_arm_trajectory

    with h5py.File(path, "r") as group:
        if "ee_poses" in group:
            poses = np.asarray(group["ee_poses"], dtype=np.float64)
            raw = np.asarray(group["gripper"], dtype=np.float64).reshape(len(poses), -1)
            opened = float(group.attrs["gripper_open_value"])
            closed = float(group.attrs["gripper_closed_value"])
            widths = np.clip((raw - closed) / (opened - closed), 0, 1)
            if "timestamps" in group:
                times = np.asarray(group["timestamps"], dtype=np.float64)
            else:
                times = np.arange(len(poses)) * float(group.attrs["sample_dt"])
            meta = json.loads(group.attrs.get("adaspevla_conversion", "{}"))
            return Trajectory(poses, widths, times, np.asarray(group.attrs["base_wxyz"]),
                              str(path), meta.get("arms", [f"arm{i}" for i in range(poses.shape[1])]))
        dt, _ = episode_sample_dt(group, profile, sample_dt)
        poses, widths = zip(*(xpolicylab_arm_trajectory(group, ARM_PREFIX[a], a) for a in arms))
    base = PROFILES[profile]["base_wxyz"] if base_wxyz is None else base_wxyz
    if base is None:
        raise ValueError(f"Profile {profile!r} needs an explicit base_wxyz")
    poses = np.stack(poses, 1)
    return Trajectory(poses, np.stack(widths, 1), np.arange(len(poses)) * dt, base,
                      str(path), arms)


def read_robocasa_episodes(root, base_motion="drop", max_episodes=None):
    """Yield (episode_index, Trajectory) from a LeRobot v2.1 RoboCasa365 dataset, without videos."""
    from ..benchmarks.robocasa import LeRobotV21, episode_trajectory

    dataset = LeRobotV21(root)
    indices = dataset.episode_indices()
    for index in indices[:max_episodes] if max_episodes else indices:
        trajectory = episode_trajectory(dataset.frames(index), index, base_motion)
        if trajectory is None:
            continue
        pose, width, times, _ = trajectory
        yield index, Trajectory(pose, width, times - times[0], (1.0, 0.0, 0.0, 0.0),
                                f"{root}#episode_{index:06d}", ("right",))


def write_segments(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return path


def load_segments(path):
    result = json.loads(Path(path).read_text())
    if result.get("format") != SEGMENTS_FORMAT:
        raise ValueError(f"{path} is not an {SEGMENTS_FORMAT} file")
    return result
