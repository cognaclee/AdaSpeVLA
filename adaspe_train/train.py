"""OpenPI scripts/train.py with the AdaSpeVLA configs registered. Usage: same CLI as OpenPI, e.g.
  python adaspe_train/train.py bread_adaspe_action --exp_name formal [--resume]"""
import runpy
import sys

import configs  # noqa: F401  (python puts this script's dir on sys.path)

if __name__ == "__main__":
    configs.register()
    runpy.run_path(f"{configs.ROOT}/external/openpi/scripts/train.py", run_name="__main__")
