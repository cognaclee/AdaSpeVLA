import os, sys; os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # adaspe_train/
import numpy as np, configs
def main():
    import openpi.training.data_loader as dl
    cfg = configs.ACTION
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    print("norm stats keys:", list(dc.norm_stats), "quantile:", dc.use_quantile_norm, "repo:", dc.repo_id)
    loader = dl.create_data_loader(cfg, num_batches=1, skip_norm_stats=False, shuffle=True)
    obs, act = next(iter(loader))
    print("images:", {k: (v.shape, str(v.dtype), float(v.min()), float(v.max())) for k, v in obs.images.items()})
    print("masks:", {k: np.asarray(v).tolist()[:3] for k, v in obs.image_masks.items()})
    print("state", obs.state.shape, "prompt tokens", obs.tokenized_prompt.shape, "valid", np.asarray(obs.tokenized_prompt_mask).sum(1)[:4])
    a = np.asarray(act); print("actions", a.shape, "real-dim range", a[..., :7].min().round(3), a[..., :7].max().round(3), "pad all zero:", bool(np.all(a[..., 7:] == 0)))
    print("gripper values:", np.unique(np.round(a[..., 6], 2))[:10])
    from openpi.models import tokenizer
    tok = tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    ids = np.asarray(obs.tokenized_prompt[0])[np.asarray(obs.tokenized_prompt_mask[0])]
    print("decoded prompt:", tok._tokenizer.decode(ids.tolist())[:300])

if __name__ == "__main__":
    main()
