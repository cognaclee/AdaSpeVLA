"""Prepare benchmark demonstrations with atomic, resumable task outputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request

from adaspevla.benchmarks.robocasa import convert_dataset
from adaspevla.benchmarks.xpolicylab_data import convert_directory


def default_robotwin_root(repo_root):
    """Prefer the workspace-local RoboTwin checkout, then the cluster default."""
    configured = os.environ.get("ROBOTWIN_ROOT")
    if configured:
        return Path(configured)
    local = repo_root.parent / "external" / "RoboTwin"
    if local.is_dir():
        return local
    return Path("/lustre/fswork/projects/rech/sdh/ueh86le/GrassmannPolicy/RoboTwin")


def download_robocasa(task, dataset_root):
    # Read the checked-out registry without importing the simulator (which has
    # strict NumPy/MuJoCo requirements irrelevant to downloading demonstrations).
    repo = Path(__file__).resolve().parents[2]
    registry = repo / "robocasa/robocasa/models/assets/box_links/box_links_ds.json"
    links = json.loads(registry.read_text())
    candidates = [key for key in links if key.startswith(f"target/atomic/{task}/")
                  and key.endswith("/lerobot.tar")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one human target archive for {task}, got {candidates}")
    key = candidates[0]
    shared = links[key]
    base, identifier = shared.rstrip("/").rsplit("/s/", 1)
    url = f"{base}/shared/static/{identifier}.tar"
    archive = dataset_root / "v1.0" / key
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".tar.part")
    print(f"Downloading {key}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(archive)
    with tarfile.open(archive) as tar:
        destination = archive.parent.resolve()
        for member in tar.getmembers():
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination) or member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination)
    archive.unlink()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["robotwin", "robocasa"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--robotwin-root", type=Path,
                        default=default_robotwin_root(repo_root))
    parser.add_argument("--robocasa-data", type=Path, default=Path("robocasa/datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("data/converted"))
    parser.add_argument("--download", action="store_true", help="Download missing human target data")
    parser.add_argument("--restart-incomplete", action="store_true",
                        help="Rebuild failed staging output; use only after its writer has stopped")
    parser.add_argument("--base-motion", choices=["error", "drop", "ignore"], default="error",
                        help="RoboCasa episodes that move the mobile base: reject the task, drop them, or keep them")
    args = parser.parse_args()
    output = args.output_root / args.benchmark
    target = output / args.task
    marker = target / "adaspevla_conversion.json"
    if marker.is_file():
        meta = json.loads(marker.read_text())
        files = list(target.glob("episode_*.hdf5"))
        if meta["task_name"] != args.task or len(files) != meta["episodes"]:
            raise ValueError(f"Incomplete conversion at {target}")
        print(f"Already prepared {args.benchmark}/{args.task}: {len(files)} episodes", flush=True)
        return
    if target.exists():
        raise FileExistsError(f"Unmarked output exists at {target}; inspect before retrying")
    if args.benchmark == "robotwin":
        source = args.robotwin_root / "data/demo_clean" / args.task / "aloha_agilex/data"
    else:
        pattern = f"v1.0/target/atomic/{args.task}/*/lerobot/meta/info.json"
        candidates = sorted(args.robocasa_data.glob(pattern))
        if not candidates and args.download:
            download_robocasa(args.task, args.robocasa_data)
            candidates = sorted(args.robocasa_data.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No RoboCasa target human dataset for {args.task}; rerun with --download on a network-enabled node")
        if len(candidates) != 1:
            raise ValueError(f"Multiple dataset versions for {args.task}: {candidates}")
        source = candidates[0].parent.parent
    staging = output / f".preparing-{args.task}"
    if staging.exists():
        if not args.restart_incomplete:
            raise FileExistsError(f"Staging exists at {staging}; wait for its writer, or retry a stopped conversion with --restart-incomplete")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"Converting {source} to {target}", flush=True)
    if args.benchmark == "robotwin":
        meta = convert_directory(source, staging, args.task, profile="robotwin", arms=["left", "right"])
    else:
        meta = convert_dataset(source, staging, args.task, base_motion=args.base_motion)
    (staging / args.task).rename(target)
    staging.rmdir()
    print(f"Prepared {args.benchmark}/{args.task}: {meta['episodes']} episodes", flush=True)


if __name__ == "__main__":
    main()
