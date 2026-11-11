"""Split raw RoboTwin / RoboCasa demonstrations into difficulty-adaptive B-spline segments.

Raw files are only read. Boundaries go to ``<output_dir>`` as one
``*.segments.json`` per episode plus ``segments_summary.json``.

Examples:
    python -m scripts.benchmarks.segment_demos --format robotwin \
        --source_dir RoboTwin/data/demo_clean/beat_block_hammer/aloha_agilex/data \
        --output_dir ./data/segments/beat_block_hammer --plot 2

    python -m scripts.benchmarks.segment_demos --format robocasa \
        --source_dir datasets/v1.0/target/atomic/CloseBlenderLid/20250822/lerobot \
        --output_dir ./data/segments/CloseBlenderLid
"""

import argparse
import json
import warnings
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from adaspevla.benchmarks.xpolicylab_data import find_episode_files
from adaspevla.datasets.segmentation import (
    SegmentConfig,
    calibrate,
    read_robocasa_episodes,
    read_robotwin_episode,
    segment_trajectory,
    write_segments,
)


def load_trajectories(args):
    source = Path(args.source_dir)
    if args.format == "robocasa":
        for index, traj in read_robocasa_episodes(source, args.base_motion, args.max_episodes):
            yield f"episode_{index:06d}", traj
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for path in find_episode_files(source, args.max_episodes):
            traj = read_robotwin_episode(
                path, args.arms, args.profile, args.base_wxyz, args.sample_dt
            )
            yield str(path.relative_to(source).with_suffix("")), traj


def plot_episode(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(result["times"])
    mid = (times[:-1] + times[1:]) / 2
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    traj = result["_trajectory"]
    for arm, name in enumerate(result["arms"]):
        speed = np.linalg.norm(np.diff(traj.poses[:, arm, :3], axis=0), axis=-1) / np.diff(times)
        axes[0].plot(mid, speed, label=f"{name} EE speed (m/s)")
        axes[1].plot(times, traj.widths[:, arm], label=f"{name} gripper width")
    axes[2].plot(mid, result["density"], color="k", label="precision demand rho")
    axes[2].set_yscale("log")
    for ax in axes:
        for k, seg in enumerate(result["segments"]):
            ax.axvspan(seg["t_start"], seg["t_end"], color="C0" if k % 2 else "C1", alpha=0.08)
            ax.axvline(seg["t_start"], color="gray", lw=0.6)
        ax.legend(loc="upper right", fontsize=8)
    lengths = [s["num_samples"] for s in result["segments"]]
    axes[0].set_title(
        f"{Path(result['source']).name}: {len(lengths)} segments, samples per segment {lengths}",
        fontsize=9,
    )
    axes[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--format", choices=["robotwin", "robocasa"], required=True)
    parser.add_argument("--source_dir", required=True,
                        help="RoboTwin/XPolicyLab HDF5 directory, or a LeRobot v2.1 RoboCasa root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=["left", "right"],
                        help="RoboTwin arms to segment jointly")
    parser.add_argument("--profile", choices=["robotwin", "robodojo"], default="robotwin")
    parser.add_argument("--base_wxyz", type=float, nargs=4, default=None)
    parser.add_argument("--sample_dt", type=float, default=None)
    parser.add_argument("--base_motion", choices=["error", "drop", "ignore"], default="drop",
                        help="RoboCasa episodes that drive the mobile base")
    parser.add_argument("--scale_horizon", type=int, default=16,
                        help="Window length (samples) for the motion scales D")
    parser.add_argument("--speed_fraction", type=float, default=0.25,
                        help="v_ref as a fraction of the 95th-percentile normalized speed")
    parser.add_argument("--plot", type=int, default=0, help="Save PNGs for the first N episodes")
    defaults = SegmentConfig()
    for field in fields(SegmentConfig):
        if field.name == "keyframes":
            parser.add_argument("--keyframes", action=argparse.BooleanOptionalAction,
                                default=defaults.keyframes,
                                help="Cut at motion stops and gripper-change edges first")
        elif field.name != "speed_ref":
            parser.add_argument(f"--{field.name}", type=type(getattr(defaults, field.name)),
                                default=getattr(defaults, field.name))
    parser.add_argument("--speed_ref", type=float, default=None,
                        help="Override the calibrated speed reference")
    args = parser.parse_args()

    episodes = list(load_trajectories(args))
    if not episodes:
        raise SystemExit("No episodes found")
    calibration = calibrate([t for _, t in episodes], args.scale_horizon, args.speed_fraction)
    config = SegmentConfig(**{
        f.name: getattr(args, f.name) for f in fields(SegmentConfig)
    })
    if config.speed_ref is None:
        config.speed_ref = calibration["speed_ref"]
    config.validate()

    out = Path(args.output_dir)
    durations, counts, failures = [], [], 0
    for index, (name, traj) in enumerate(episodes):
        result = segment_trajectory(traj, calibration["motion_scales"], config)
        write_segments(result, out / f"{name}.segments.json")
        durations += [s["duration"] for s in result["segments"]]
        counts.append(len(result["segments"]))
        failures += sum(not s["within_tolerance"] for s in result["segments"])
        if index < args.plot:
            result["_trajectory"] = traj
            plot_episode(result, out / f"{name}.segments.png")
    summary = dict(
        source_dir=str(args.source_dir),
        format=args.format,
        episodes=len(episodes),
        config=asdict(config),
        motion_scales=calibration["motion_scales"].tolist(),
        speed_ref=calibration["speed_ref"],
        scale_horizon=args.scale_horizon,
        segments_per_episode=dict(mean=float(np.mean(counts)), min=min(counts), max=max(counts)),
        segment_duration_s={f"p{q}": float(np.percentile(durations, q)) for q in (5, 25, 50, 75, 95)},
        segments_over_tolerance=failures,
    )
    (out / "segments_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "episodes", "segments_per_episode", "segment_duration_s", "segments_over_tolerance")}))


if __name__ == "__main__":
    main()
