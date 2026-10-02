"""XPolicyLab entry point; the implementation lives in adaspevla.benchmarks."""

from pathlib import Path

from adaspevla.benchmarks.xpolicylab_model import Model as _AdaSpeVLAModel


class Model(_AdaSpeVLAModel):
    def __init__(self, model_cfg):
        model_cfg = dict(model_cfg)
        # Checkpoints and metrics resolve relative to this policy directory.
        model_cfg.setdefault("policy_dir", str(Path(__file__).parent))
        super().__init__(model_cfg)
