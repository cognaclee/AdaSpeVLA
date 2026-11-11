"""Timing pipeline check: model inputs equal the action pipeline's, and the model's timing labels (computed from
the normalized pipeline actions) equal calibrate_timing_anchor.py's from the raw parquet (same dataset indices)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # adaspe_train/
import configs  # noqa: E402

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import numpy as np
    import openpi.training.data_loader as dl

    from adaspe import labels

    configs.register()
    act_cfg, tim_cfg = configs.INPUT_REF, configs.TIMINGS[0]

    def pipeline(cfg):
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        ds = dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
        return dl.transform_dataset(ds, dc), dc

    act_ds, _ = pipeline(act_cfg)
    tim_ds, tdc = pipeline(tim_cfg)
    print("timing sequence keys:", tdc.action_sequence_keys)
    import tasks
    ref = np.load(tasks.timing_labels_path())
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(len(act_ds), 200, replace=False), [len(act_ds) - 1, len(act_ds) - 5]])
    mismatch = 0
    labs = []
    for i in idx:
        a, t = act_ds[int(i)], tim_ds[int(i)]
        for k in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
            assert np.array_equal(a[k], t[k]), (i, k)
        for k in a["image"]:
            assert np.array_equal(a["image"][k], t["image"][k]), (i, k)
        assert np.array_equal(a["actions"][:, :7], t["actions"][:, :7]), i
        assert np.all(a["actions"][:, 7:] == 0) and np.all(t["actions"][:, 14:] == 0), i
        # The exact label the model trains on (adaspe/labels.py), from the normalized pipeline actions.
        lab = float(labels.timing_target(jnp.asarray(t["actions"][None]), tim_cfg.model)[0, 0])
        labs.append(lab)
        mismatch += abs(lab - ref[i]) > 1e-4
    print(f"checked {len(idx)} samples: inputs identical to action pipeline; label mismatches vs calibration: {mismatch}")
    print("label sample:", np.round(labs[:12], 2))
