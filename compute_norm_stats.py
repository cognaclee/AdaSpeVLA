"""Run OpenPI's compute_norm_stats with the AdaSpeVLA configs registered.

Top-level import of `configs` installs the datasets "List" shim; it must also run in the
DataLoader's spawned workers, so only the run itself is guarded by __main__.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs  # noqa: E402

if __name__ == "__main__":
    configs.register()
    sys.argv = ["compute_norm_stats.py", *sys.argv[1:]]
    runpy.run_path(f"{configs.ROOT}/external/openpi/scripts/compute_norm_stats.py", run_name="__main__")
