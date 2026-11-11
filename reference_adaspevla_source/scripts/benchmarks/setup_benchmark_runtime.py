"""Make a writable RoboTwin/XPolicyLab runtime without copying assets or demos."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def default_robotwin_root(repo_root):
    """Prefer the workspace-local RoboTwin checkout, then the cluster default."""
    configured = os.environ.get("ROBOTWIN_ROOT")
    if configured:
        return Path(configured)
    local = repo_root.parent / "external" / "RoboTwin"
    if (local / "XPolicyLab/setup_policy_server.py").is_file():
        return local
    return Path("/lustre/fswork/projects/rech/sdh/ueh86le/GrassmannPolicy/RoboTwin")


def mirror(source, target, private_dirs):
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        dest = target / item.name
        if item.name in private_dirs or dest.exists() or dest.is_symlink():
            continue
        if item.is_file() and item.suffix in {".py", ".sh"}:
            shutil.copy2(item, dest)
        else:
            dest.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path,
                        default=default_robotwin_root(root))
    args = parser.parse_args()
    source = args.robotwin_root.resolve()
    if not (source / "XPolicyLab/setup_policy_server.py").is_file():
        raise FileNotFoundError(f"Missing XPolicyLab at {source}")
    target = root / "tmp/benchmark-runtime/RoboTwin"
    mirror(source, target, {"XPolicyLab", "scripts", "eval_result", "logs"})
    shutil.copytree(source / "scripts", target / "scripts", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    mirror(source / "XPolicyLab", target / "XPolicyLab", {"policy"})
    mirror(source / "XPolicyLab/policy", target / "XPolicyLab/policy", {"AdaSpeVLA"})
    subprocess.run(["bash", str(root / "scripts/benchmarks/install_xpolicylab_adapter.sh"),
                    str(target / "XPolicyLab")], check=True)
    print(f"Runtime ready: {target}")


if __name__ == "__main__":
    main()
