"""Observation encoders of pretrained robot policies: pi0, pi0.5 and GR00T N1.5.

Only the vision-language part of each checkpoint is loaded: the SigLIP-So400m/14
vision tower (224 px, 256 tokens per image), its projector into the language
model, and the language model itself (Gemma-2B for pi0 and pi0.5, from
``lerobot/pi0_base`` and ``lerobot/pi05_base``; the first 12 Qwen3-1.7B layers of
Eagle 2.5, which is all ``nvidia/GR00T-N1.5-3B`` keeps). Action experts and
heads are not loaded.

The two decoders are implemented here, so the same code runs under every
installed transformers version (Qwen3 needs >= 4.51, the RoboTwin evaluation
environment has 4.46); only the SigLIP vision tower comes from transformers.

Modes:

- ``joint`` (how pi0 and GR00T use them): the image tokens of every view and the
  instruction go through the language model together. pi0/pi0.5 attend
  bidirectionally over this prefix; GR00T is causal with the images first, in
  its Eagle chat template. Image tokens are the language-model outputs at the
  image positions.
- ``separate``: image tokens are the projected SigLIP tokens and the language
  model reads the instruction alone (cached per instruction: it is frozen).

In both modes the text feature is the mean language-model output over the
instruction tokens.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Eagle 2.5 tokenizer of GR00T N1.5 (Qwen3 vocabulary plus <img>, </img> and
# <IMG_CONTEXT>), built from Isaac-GR00T n1.5-release
# gr00t/model/backbone/eagle2_hg_model.
EAGLE_TOKENIZER_DIR = Path(__file__).parent / "assets" / "gr00t_n1_5_eagle_tokenizer"
EAGLE_SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
EAGLE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

SIGLIP_SO400M_14_224 = dict(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    patch_size=14,
    image_size=224,
    hidden_act="gelu_pytorch_tanh",
    layer_norm_eps=1e-6,
    vision_use_head=False,
)
GEMMA_2B = dict(
    hidden=2048, intermediate=16384, heads=8, kv_heads=1, head_dim=256, layers=18,
    vocab=257152, rope_theta=1e4, act="gelu_tanh", norm_offset=1.0, qk_norm=False,
    embed_scale=True,
)
QWEN3_1_7B_FIRST_12 = dict(
    hidden=2048, intermediate=6144, heads=16, kv_heads=8, head_dim=128, layers=12,
    vocab=151680, rope_theta=1e6, act="silu", norm_offset=0.0, qk_norm=True,
    embed_scale=False,
)

_PI_PREFIX = "paligemma_with_expert.paligemma."
_PI_RULES = (  # (checkpoint prefix, module prefix); None skips the tensor
    (_PI_PREFIX + "model.vision_tower.", "vision."),
    (_PI_PREFIX + "model.multi_modal_projector.linear.", "projector."),
    (_PI_PREFIX + "model.language_model.layers.", "lm.layers."),
    (_PI_PREFIX + "model.language_model.norm.", "lm.norm."),
    (_PI_PREFIX + "lm_head.", "lm.embed_tokens."),  # tied input embedding
)
_EAGLE_PREFIX = "backbone.eagle_model."
_EAGLE_RULES = (
    (_EAGLE_PREFIX + "vision_model.vision_model.head.", None),  # unused pooling head
    (_EAGLE_PREFIX + "vision_model.", "vision."),
    (_EAGLE_PREFIX + "mlp1.0.", "projector."),
    (_EAGLE_PREFIX + "language_model.model.layers.", "lm.layers."),
    (_EAGLE_PREFIX + "language_model.model.norm.", "lm.norm."),
    (_EAGLE_PREFIX + "language_model.model.embed_tokens.", "lm.embed_tokens."),
)

VLA_BACKBONES = {
    # pi0 (lerobot): instruction + "\n", BOS, at most 48 tokens.
    "pi0": dict(
        family="paligemma", repo="lerobot/pi0_base", rules=_PI_RULES, lm=GEMMA_2B,
        tokenizer="google/paligemma-3b-pt-224", prompt="{}\n", max_text=48,
    ),
    # pi0.5 (lerobot): "Task: ..., State: ...;\nAction: " with the state written as
    # tokens. Robot state enters our model as its own token, so it is left out.
    "pi05": dict(
        family="paligemma", repo="lerobot/pi05_base", rules=_PI_RULES, lm=GEMMA_2B,
        tokenizer="google/paligemma-3b-pt-224", prompt="Task: {};\nAction: ", max_text=200,
    ),
    "gr00t": dict(
        family="eagle", repo="nvidia/GR00T-N1.5-3B", rules=_EAGLE_RULES,
        lm=QWEN3_1_7B_FIRST_12, tokenizer=str(EAGLE_TOKENIZER_DIR),
    ),
}
_ALIASES = {"pi0.5": "pi05", "pi05_base": "pi05", "gr00t-n1.5": "gr00t", "gr00t_n1.5": "gr00t"}
VLA_MODES = ("joint", "separate")


def vla_backbone_key(name):
    """Canonical VLA backbone key for ``name``, or None for other backbones."""
    key = str(name).lower()
    key = _ALIASES.get(key, key)
    return key if key in VLA_BACKBONES else None


def resolve_dtype(name):
    """Language-model dtype. ``auto``: bfloat16 on Ampere or newer GPUs, else float32."""
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float32
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in dtypes:
        raise ValueError(f"vlm_dtype must be auto or one of {sorted(dtypes)}")
    return dtypes[name]


class _RMSNorm(nn.Module):
    """Gemma scales by (1 + weight), Qwen3 by weight; both normalize in float32."""

    def __init__(self, dim, eps, offset):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps, self.offset = eps, offset

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * (self.weight.float() + self.offset)).to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        D, Hd = spec["hidden"], spec["head_dim"]
        self.heads, self.kv_heads, self.head_dim = spec["heads"], spec["kv_heads"], Hd
        self.q_proj = nn.Linear(D, self.heads * Hd, bias=False)
        self.k_proj = nn.Linear(D, self.kv_heads * Hd, bias=False)
        self.v_proj = nn.Linear(D, self.kv_heads * Hd, bias=False)
        self.o_proj = nn.Linear(self.heads * Hd, D, bias=False)
        if spec["qk_norm"]:
            self.q_norm = _RMSNorm(Hd, 1e-6, spec["norm_offset"])
            self.k_norm = _RMSNorm(Hd, 1e-6, spec["norm_offset"])
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


class _MLP(nn.Module):
    def __init__(self, spec):
        super().__init__()
        D, I = spec["hidden"], spec["intermediate"]
        self.gate_proj = nn.Linear(D, I, bias=False)
        self.up_proj = nn.Linear(D, I, bias=False)
        self.down_proj = nn.Linear(I, D, bias=False)
        self.tanh_gelu = spec["act"] == "gelu_tanh"

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh") if self.tanh_gelu else F.silu(gate)
        return self.down_proj(gate * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.input_layernorm = _RMSNorm(spec["hidden"], 1e-6, spec["norm_offset"])
        self.self_attn = _Attention(spec)
        self.post_attention_layernorm = _RMSNorm(spec["hidden"], 1e-6, spec["norm_offset"])
        self.mlp = _MLP(spec)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class _DecoderLM(nn.Module):
    """Gemma / Qwen3 decoder stack over input embeddings; returns the final-norm output."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec["vocab"], spec["hidden"])
        self.layers = nn.ModuleList(_DecoderLayer(spec) for _ in range(spec["layers"]))
        self.norm = _RMSNorm(spec["hidden"], 1e-6, spec["norm_offset"])

    def embed(self, ids):
        x = self.embed_tokens(ids)
        if self.spec["embed_scale"]:  # Gemma multiplies token embeddings by sqrt(D)
            x = x * torch.tensor(self.spec["hidden"] ** 0.5, dtype=x.dtype)
        return x

    def forward(self, x, mask):
        Hd = self.spec["head_dim"]
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / self.spec["rope_theta"] ** (
            torch.arange(0, Hd, 2, device=x.device, dtype=torch.float32) / Hd
        )
        angles = torch.outer(positions, inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        # Only needed when vision blocks are fine-tuned through the frozen LM.
        recompute = torch.is_grad_enabled() and x.requires_grad
        for layer in self.layers:
            if recompute:
                x = checkpoint(layer, x, cos, sin, mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, mask)
        return self.norm(x)


def _checkpoint_files(repo):
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, allow_patterns=["*.safetensors", "config.json"]))
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights for {repo} in {path}")
    return files


def _map_key(key, rules):
    for source, target in rules:
        if key.startswith(source):
            return None if target is None else target + key[len(source) :]
    return None


class PretrainedVLAEncoder(nn.Module):
    """SigLIP vision tower, projector and language model of a pretrained VLA.

    The frozen language model is stored in ``dtype``. The vision tower and
    projector, which the finetune stage trains, stay float32 (bfloat16 weights
    would round away small optimizer updates) and run under autocast to ``dtype``.
    """

    def __init__(self, name, mode="joint", dtype="auto"):
        super().__init__()
        from transformers import AutoTokenizer, SiglipVisionConfig, SiglipVisionModel

        key = vla_backbone_key(name)
        if key is None:
            raise ValueError(f"Unknown VLA backbone {name!r}; choose from {sorted(VLA_BACKBONES)}")
        if mode not in VLA_MODES:
            raise ValueError(f"vlm_mode must be one of {VLA_MODES}")
        self.key, self.mode, self.spec = key, mode, VLA_BACKBONES[key]
        self.family = self.spec["family"]
        self.dtype = resolve_dtype(dtype)
        lm_spec = self.spec["lm"]
        self.hidden_size = lm_spec["hidden"]
        vision_config = SiglipVisionConfig(**SIGLIP_SO400M_14_224)
        vision_config._attn_implementation = "sdpa"
        self.image_size = (vision_config.image_size, vision_config.image_size)
        self.tokens_per_image = (vision_config.image_size // vision_config.patch_size) ** 2
        with torch.device("meta"):  # weights come from the checkpoint below
            self.vision = SiglipVisionModel(vision_config)
            self.projector = nn.Linear(vision_config.hidden_size, self.hidden_size)
            self.lm = _DecoderLM(lm_spec)
        self._load_weights()
        self.requires_grad_(False)
        self.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["tokenizer"])
        self._ids_cache, self._text_cache = {}, {}
        if self.family == "eagle":
            self.pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            self.image_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        else:
            self.pad_id = self.tokenizer.pad_token_id or 0

    def _load_weights(self):
        from safetensors import safe_open

        expected = self.state_dict(keep_vars=True)
        state = {}
        for path in _checkpoint_files(self.spec["repo"]):
            with safe_open(str(path), framework="pt") as handle:
                for source in handle.keys():
                    target = _map_key(source, self.spec["rules"])
                    if target in expected:
                        dtype = self.dtype if target.startswith("lm.") else torch.float32
                        state[target] = handle.get_tensor(source).to(dtype)
        missing = sorted(set(expected) - set(state))
        if missing:
            raise RuntimeError(f"{self.spec['repo']} lacks {len(missing)} tensors, e.g. {missing[:3]}")
        self.load_state_dict(state, strict=True, assign=True)
        for module_name, module in self.named_modules():
            for buffer_name, buffer in list(module.named_buffers(recurse=False)):
                if not buffer.is_meta:
                    continue
                if buffer_name != "position_ids":
                    raise RuntimeError(f"Uninitialized buffer {module_name}.{buffer_name}")
                # SigLIP's non-persistent patch position ids.
                module.register_buffer(
                    "position_ids", torch.arange(buffer.shape[-1]).expand(1, -1), persistent=False
                )

    def set_trainable_vision_layers(self, count):
        """Train the last ``count`` SigLIP blocks, its final norm and the projector."""
        self.requires_grad_(False)
        if count:
            vision = self.vision.vision_model
            for block in vision.encoder.layers[-count:]:
                block.requires_grad_(True)
            vision.post_layernorm.requires_grad_(True)
            self.projector.requires_grad_(True)

    def _prompt_ids(self, prompt):
        ids = self._ids_cache.get(prompt)
        if ids is None:
            text = prompt.strip()
            if self.family == "paligemma":
                text = self.spec["prompt"].format(text.replace("_", " ").replace("\n", " "))
                ids = self.tokenizer(text)["input_ids"][: self.spec["max_text"]]  # with BOS
            else:
                ids = self.tokenizer(text + EAGLE_SUFFIX, add_special_tokens=False)["input_ids"]
            self._ids_cache[prompt] = ids
        return ids

    def _eagle_prefix(self, num_views):
        """System and user header with one ``<img>`` block of 256 tokens per view."""
        key = ("eagle_prefix", num_views)
        ids = self._ids_cache.get(key)
        if ids is None:
            images = "".join(f"<image {k + 1}><img><IMG_CONTEXT></img>" for k in range(num_views))
            ids = []
            for token in self.tokenizer(EAGLE_SYSTEM + images, add_special_tokens=False)["input_ids"]:
                ids.extend([token] * (self.tokens_per_image if token == self.image_id else 1))
            self._ids_cache[key] = ids
        return ids

    def _run_lm(self, prefix_ids, prompts, images=None):
        """LM over [prefix ids | image tokens] + prompt ids, right-padded per batch.

        Returns the output at image positions (B, n_img, D) or None, and the
        mean output over each prompt's tokens (B, D).
        """
        device = self.projector.weight.device
        prompt_ids = [self._prompt_ids(p) for p in prompts]
        width = max(len(ids) for ids in prompt_ids)
        text = torch.full((len(prompts), width), self.pad_id, dtype=torch.long, device=device)
        text_valid = torch.zeros(len(prompts), width, dtype=torch.bool, device=device)
        for row, ids in enumerate(prompt_ids):
            text[row, : len(ids)] = torch.tensor(ids, device=device)
            text_valid[row, : len(ids)] = True
        B = len(prompts)
        if self.family == "paligemma":
            # pi0 prefix: images (as projected, unscaled), then text; bidirectional.
            parts = [self.lm.embed(text)]
            n_img = 0 if images is None else images.shape[1]
            if images is not None:
                parts.insert(0, images)
            x = torch.cat(parts, dim=1)
            valid = torch.cat((text_valid.new_ones(B, n_img), text_valid), dim=1)
            mask = valid[:, None, None, :]
            hidden = self.lm(x, mask)
            image_out, text_out = hidden[:, :n_img], hidden[:, n_img:]
        else:
            # Eagle chat template, causal; <IMG_CONTEXT> embeddings are replaced.
            prefix = torch.tensor(prefix_ids, device=device).expand(B, -1)
            x = self.lm.embed(torch.cat((prefix, text), dim=1))
            slots = (prefix[0] == self.image_id).nonzero().squeeze(1)
            if images is not None:
                if slots.numel() != images.shape[1]:
                    raise ValueError("Image token count does not match the Eagle prefix")
                x = x.clone()
                x[:, slots] = images.to(x.dtype)
            L, P = x.shape[1], prefix.shape[1]
            valid = torch.cat((text_valid.new_ones(B, P), text_valid), dim=1)
            causal = torch.ones(L, L, dtype=torch.bool, device=device).tril()
            hidden = self.lm(x, causal[None, None] & valid[:, None, None, :])
            image_out = hidden[:, slots] if images is not None else None
            text_out = hidden[:, P:]
        weights = text_valid.unsqueeze(-1).to(text_out.dtype)
        text_features = (text_out * weights).sum(1) / weights.sum(1).clamp_min(1)
        return (image_out if images is not None else None), text_features

    def _instruction_features(self, prompts):
        """Separate mode: frozen LM over each instruction alone, cached."""
        device = self.projector.weight.device
        if any(p.requires_grad for p in self.lm.parameters()):
            return self._run_lm(self._eagle_prefix(0) if self.family == "eagle" else None, prompts)[1]
        missing = sorted({p for p in prompts if (p, device) not in self._text_cache})
        if missing:
            prefix = self._eagle_prefix(0) if self.family == "eagle" else None
            with torch.no_grad():
                _, features = self._run_lm(prefix, missing)
            for prompt, feature in zip(missing, features):
                self._text_cache[(prompt, device)] = feature
        return torch.stack([self._text_cache[(p, device)] for p in prompts])

    def forward(self, pixels, prompts, num_views):
        """Normalized pixels (B*V, 3, 224, 224) -> image tokens (B, V, P, D), text (B, D)."""
        B = len(prompts)
        with torch.autocast(pixels.device.type, self.dtype, enabled=self.dtype != torch.float32):
            vision = self.vision(pixel_values=pixels.float()).last_hidden_state
            tokens = self.projector(vision)
        tokens = tokens.to(self.dtype).reshape(B, num_views * self.tokens_per_image, -1)
        if self.mode == "joint":
            prefix = self._eagle_prefix(num_views) if self.family == "eagle" else None
            tokens, text = self._run_lm(prefix, prompts, images=tokens)
        else:
            text = self._instruction_features(prompts)
        return tokens.reshape(B, num_views, self.tokens_per_image, -1), text
