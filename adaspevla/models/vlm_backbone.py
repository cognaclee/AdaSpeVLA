"""SigLIP conditioning with its cached processor; explicit offline test backend.

Legacy mode (``spatial_tokens=False``) conditions on one pooled image embedding
per view. Spatial mode also returns a token memory for cross-attention: every
SigLIP patch token of every view (plus a view embedding), the instruction and,
when configured, the robot state. Spatial mode preprocesses on the GPU, applies
random-crop (crop fit) or random-shift (pad fit) and colour-jitter augmentation
while training (no randomness at inference) and can fine-tune the last vision
blocks.

``model_name`` may also name the observation encoder of a pretrained VLA
(``pi0``, ``pi05``, ``gr00t``; see :mod:`.vla_backbones`). Its image tokens and
instruction feature then replace the SigLIP ones; everything after that is the
same.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .vla_backbones import PretrainedVLAEncoder, vla_backbone_key


class MultiCameraVLMBackbone(nn.Module):
    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        output_dim=512,
        num_views=2,
        freeze_backbone=True,
        use_lora=False,
        test_mode=False,
        spatial_tokens=False,
        proprio_dim=0,
        augment=False,
        crop_ratio=0.9,
        vlm_mode="joint",
        vlm_dtype="auto",
        image_fit="crop",
        vlm_layers=None,
        vlm_finetune_layers=0,
    ):
        super().__init__()
        if use_lora:
            raise ValueError(
                "LoRA is not implemented; use full or frozen backbone training"
            )
        self.num_views, self.output_dim = num_views, output_dim
        self.freeze_backbone, self.test_mode = freeze_backbone, test_mode
        self.spatial_tokens, self.proprio_dim = bool(spatial_tokens), int(proprio_dim)
        self.augment, self.crop_ratio = bool(augment), float(crop_ratio)
        if image_fit not in ("crop", "pad"):
            raise ValueError("image_fit must be 'crop' or 'pad'")
        # crop: centred square of crop_ratio * the short side (drops the sides of
        # 4:3 frames); pad: letterbox the whole frame to a square, as pi0 does.
        # Augmented crop takes a random crop_ratio square; augmented pad shifts the
        # letterboxed frame by up to (1 - crop_ratio) / 2 of its side (edges
        # replicated), so training and inference see objects at the same scale.
        self.image_fit = image_fit
        self.trainable_vision_layers = 0
        self.trainable_lm = False
        # Backbone weights trained at some point; the rest are reloaded from the
        # pretrained files instead of being saved in checkpoints.
        self._tuned_keys = set()
        self.is_vla = False
        if test_mode:
            # Explicit, small trainable test model. Never selected on load failure.
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.text_encoder = nn.Embedding(257, 32, padding_idx=0)
            self.feature_dim, self.text_feature_dim = 24, 32
            self.processor = None
        elif vla_backbone_key(model_name) is not None:
            self.is_vla, self.processor, self.text_encoder = True, None, None
            self.vision_encoder = PretrainedVLAEncoder(
                model_name, vlm_mode, vlm_dtype, lm_layers=vlm_layers,
                lm_trainable_layers=vlm_finetune_layers,
            )
            self.feature_dim = self.text_feature_dim = self.vision_encoder.hidden_size
            self.image_size = self.vision_encoder.image_size
            # SigLIP-So400m in pi0 and Eagle maps [0,1] to [-1,1].
            self.register_buffer("pixel_mean", torch.full((1, 3, 1, 1), 0.5), persistent=False)
            self.register_buffer("pixel_std", torch.full((1, 3, 1, 1), 0.5), persistent=False)
        else:
            from transformers import AutoModel, AutoProcessor

            if vlm_layers is not None or vlm_finetune_layers:
                raise ValueError(
                    "vlm_layers and vlm_finetune_layers apply to the pi0, pi05 and gr00t encoders only"
                )
            if image_fit != "crop" and not self.spatial_tokens:
                raise ValueError("image_fit='pad' needs spatial_tokens (legacy mode uses the processor)")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.vision_encoder = AutoModel.from_pretrained(model_name)
            config = self.vision_encoder.config
            if config.model_type != "siglip":
                raise ValueError(
                    "This backend supports SigLIP; other VLMs need an explicit adapter"
                )
            self.feature_dim = config.vision_config.hidden_size
            self.text_feature_dim = (
                getattr(config.text_config, "projection_size", None)
                or config.text_config.hidden_size
            )
            self.max_text_length = config.text_config.max_position_embeddings
            self.text_encoder = None
            if self.spatial_tokens:
                # GPU preprocessing constants; legacy mode uses the processor itself.
                processor = self.processor.image_processor
                self.image_size = (processor.size["height"], processor.size["width"])
                self.register_buffer(
                    "pixel_mean",
                    torch.tensor(processor.image_mean).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "pixel_std",
                    torch.tensor(processor.image_std).view(1, 3, 1, 1),
                    persistent=False,
                )
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.feature_dim * num_views, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim), nn.Sigmoid()
        )
        self.final_proj = nn.Linear(output_dim, output_dim)
        if self.spatial_tokens:
            self.token_proj = nn.Linear(self.feature_dim, output_dim)
            self.view_embedding = nn.Parameter(torch.randn(num_views, 1, output_dim) * 0.02)
            self.text_token_proj = nn.Linear(self.text_feature_dim, output_dim)
        if self.proprio_dim:
            # Robot state enters the global condition and, in spatial mode, the memory.
            self.proprio_proj = nn.Sequential(
                nn.Linear(self.proprio_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        if freeze_backbone:
            for encoder in (self.vision_encoder, self.text_encoder):
                if encoder is not None:
                    encoder.requires_grad_(False)
                    encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.vision_encoder.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
        return self

    def set_trainable_vision_layers(self, count, lm=False):
        """Train the last ``count`` vision blocks (0 freezes the encoder again) and,
        with ``lm``, a VLA encoder's float32 language-model layers.

        The text tower stays frozen. Encoders stay in eval mode: SigLIP has no
        batch statistics and zero dropout, so this only gates gradients.
        """
        if count and not self.spatial_tokens:
            raise ValueError("Vision fine-tuning requires spatial_tokens")
        self.vision_encoder.requires_grad_(False)
        if count:
            if self.test_mode:
                self.vision_encoder.requires_grad_(True)
            elif self.is_vla:
                self.vision_encoder.set_trainable_vision_layers(count)
            else:
                vision = self.vision_encoder.vision_model
                for block in vision.encoder.layers[-count:]:
                    block.requires_grad_(True)
                vision.post_layernorm.requires_grad_(True)
                if getattr(vision, "head", None) is not None:
                    vision.head.requires_grad_(True)
        if lm:
            if not self.is_vla or not self.vision_encoder.lm_trainable_layers:
                raise ValueError("Language-model fine-tuning needs a VLA encoder with vlm_finetune_layers")
            self.vision_encoder.set_trainable_lm_layers(True)
        self.trainable_vision_layers, self.trainable_lm = int(count), bool(lm)
        self.mark_tuned(
            name for name, p in self.vision_encoder.named_parameters() if p.requires_grad
        )

    def mark_tuned(self, names):
        """Record backbone weights (names inside ``vision_encoder``) as trained."""
        self._tuned_keys.update(f"vision_encoder.{name}" for name in names)

    def frozen_pretrained_keys(self):
        """State-dict keys of pretrained backbone weights that were never trained.

        They equal the pretrained files, so checkpoints leave them out and a
        rebuilt policy keeps the weights it loaded from those files.
        """
        if self.test_mode or not self.freeze_backbone:
            return set()
        keys = set()
        for encoder in ("vision_encoder", "text_encoder"):
            module = getattr(self, encoder)
            if module is not None:
                keys.update(f"{encoder}.{key}" for key in module.state_dict())
        return keys - self._tuned_keys

    def _pixels(self, flat):
        """Crop (random while training, centre otherwise), jitter, resize, normalize."""
        N, _, H, W = flat.shape
        side = max(1, int(round(min(H, W) * self.crop_ratio)))
        if self.image_fit == "pad":
            side = max(H, W)
            top, left = (side - H) // 2, (side - W) // 2
            flat = F.pad(flat, (left, side - W - left, top, side - H - top))  # black bars
            H = W = side
        if self.training and self.augment:
            if self.image_fit == "pad":
                shift = max(1, int(round(side * (1 - self.crop_ratio) / 2)))
                flat = F.pad(flat, (shift,) * 4, mode="replicate")
                H = W = side + 2 * shift
            ys = torch.randint(0, H - side + 1, (N,)).tolist()
            xs = torch.randint(0, W - side + 1, (N,)).tolist()
            crops = torch.stack(
                [flat[i, :, y : y + side, x : x + side] for i, (y, x) in enumerate(zip(ys, xs))]
            )
            jitter = lambda: 1 + (torch.rand(N, 1, 1, 1, device=flat.device) * 2 - 1) * 0.2
            crops = crops * jitter()
            mean = crops.mean(dim=(1, 2, 3), keepdim=True)
            crops = (crops - mean) * jitter() + mean
            gray = (crops * crops.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(
                1, keepdim=True
            )
            crops = ((crops - gray) * jitter() + gray).clamp(0, 1)
        elif self.image_fit == "pad":
            crops = flat
        else:
            y, x = (H - side) // 2, (W - side) // 2
            crops = flat[:, :, y : y + side, x : x + side]
        if self.test_mode:
            return crops
        crops = F.interpolate(
            crops, size=self.image_size, mode="bilinear", align_corners=False, antialias=True
        )
        return (crops - self.pixel_mean) / self.pixel_std

    def _vision_tokens(self, flat):
        """Patch tokens (N,P,F) and pooled features (N,F) for spatial mode."""
        pixels = self._pixels(flat)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable_vision_layers > 0):
            if self.test_mode:
                maps = self.vision_encoder[1](self.vision_encoder[0](pixels))
                tokens = F.adaptive_avg_pool2d(maps, 4).flatten(2).transpose(1, 2)
                return tokens, tokens.mean(1)
            pixels = pixels.to(dtype=next(self.vision_encoder.parameters()).dtype)
            out = self.vision_encoder.vision_model(pixel_values=pixels)
            return out.last_hidden_state, out.pooler_output

    def _check_images(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5 or images.shape[1:3] != (self.num_views, 3):
            raise ValueError("Expected (B, configured_views, 3, H, W) images")
        if not torch.isfinite(images).all() or ((images < 0) | (images > 1)).any():
            raise ValueError("Input RGB images must be floating values in [0,1]")
        return images

    def encode_images(self, images):
        if self.is_vla:
            raise ValueError("VLA backbones encode images and text together; call forward")
        images = self._check_images(images)
        B = images.shape[0]
        flat = images.reshape(-1, *images.shape[2:])
        if self.spatial_tokens:
            tokens, features = self._vision_tokens(flat)
            tokens = self.token_proj(tokens.to(self.token_proj.weight.dtype))
            tokens = tokens.reshape(B, self.num_views, -1, self.output_dim) + self.view_embedding
            pooled = self.fusion_proj(features.reshape(B, -1).to(self.fusion_proj[0].weight.dtype))
            return pooled, tokens.flatten(1, 2)
        if self.test_mode:
            features = self.vision_encoder(flat)
        else:
            # Use the checkpoint's resizing and normalization. Input is already [0,1].
            pixels = self.processor.image_processor(
                images=list(flat.detach().cpu()),
                return_tensors="pt",
                do_rescale=False,
                input_data_format="channels_first",
            )["pixel_values"]
            pixels = pixels.to(
                device=images.device, dtype=next(self.vision_encoder.parameters()).dtype
            )
            features = self.vision_encoder.get_image_features(pixel_values=pixels)
        if not isinstance(features, torch.Tensor) or features.shape != (
            B * self.num_views,
            self.feature_dim,
        ):
            raise ValueError("Unexpected image feature shape from configured backbone")
        return self.fusion_proj(
            features.reshape(B, -1).to(self.fusion_proj[0].weight.dtype)
        )

    def encode_text(self, text_prompts, batch_size, device):
        """Projected instruction embedding; see :meth:`_text_features`."""
        return self.text_proj(self._text_features(text_prompts, batch_size, device))

    @staticmethod
    def _prompt_list(text_prompts, batch_size):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts] * batch_size
        if (
            text_prompts is None
            or len(text_prompts) != batch_size
            or any(not isinstance(t, str) or not t.strip() for t in text_prompts)
        ):
            raise ValueError(
                "Provide one nonempty language instruction per observation"
            )
        return list(text_prompts)

    def _text_features(self, text_prompts, batch_size, device):
        if self.is_vla:
            raise ValueError("VLA backbones encode images and text together; call forward")
        text_prompts = self._prompt_list(text_prompts, batch_size)
        if self.test_mode:
            tokens = torch.zeros(batch_size, 128, dtype=torch.long, device=device)
            for row, text in enumerate(text_prompts):
                values = list(text.encode("utf-8")[:128])
                tokens[row, : len(values)] = torch.tensor(values, device=device) + 1
            mask = tokens.ne(0).unsqueeze(-1)
            features = (self.text_encoder(tokens) * mask).sum(1) / mask.sum(
                1
            ).clamp_min(1)
        else:
            inputs = self.processor.tokenizer(
                text_prompts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():  # The text tower is never trained.
                features = self.vision_encoder.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor) or features.shape != (
            batch_size,
            self.text_feature_dim,
        ):
            raise ValueError("Unexpected text feature shape from configured backbone")
        return features.to(self.text_proj[0].weight.dtype)

    def _encode_vla(self, images, instructions):
        """Global image feature, token memory (spatial mode) and text feature."""
        images = self._check_images(images)
        B, V = images.shape[:2]
        prompts = self._prompt_list(instructions, B)
        pixels = self._pixels(images.reshape(-1, *images.shape[2:]))
        trainable = self.trainable_vision_layers > 0 or self.trainable_lm
        with torch.set_grad_enabled(torch.is_grad_enabled() and trainable):
            tokens, text = self.vision_encoder(pixels, prompts, V)
        dtype = self.fusion_proj[0].weight.dtype
        tokens, text = tokens.to(dtype), text.to(dtype)
        pooled = self.fusion_proj(tokens.mean(2).reshape(B, -1))
        if not self.spatial_tokens:
            return pooled, None, text
        tokens = self.token_proj(tokens) + self.view_embedding
        return pooled, tokens.flatten(1, 2), text

    def forward(self, images, instructions=None, proprio=None, return_memory=False):
        """Global condition (B, output_dim); with ``return_memory`` also the token memory.

        ``proprio`` is the normalized robot state (B, proprio_dim), required when
        the backbone was built with ``proprio_dim``.
        """
        if self.proprio_dim and (proprio is None or proprio.shape != (images.shape[0], self.proprio_dim)):
            raise ValueError(f"Expected proprio of shape (B, {self.proprio_dim})")
        if self.is_vla:
            vision, tokens, text_features = self._encode_vla(images, instructions)
        else:
            if self.spatial_tokens:
                vision, tokens = self.encode_images(images)
            else:
                vision, tokens = self.encode_images(images), None
            text_features = self._text_features(instructions, images.shape[0], images.device)
        text = self.text_proj(text_features)
        gate = self.cross_gate(torch.cat((vision, text), dim=-1))
        cond = self.final_proj(gate * vision + (1 - gate) * text)
        state = None
        if self.proprio_dim:
            state = self.proprio_proj(proprio.to(cond.dtype))
            cond = cond + state
        if not return_memory:
            return cond
        if tokens is None:
            return cond, None
        extra = [self.text_token_proj(text_features)[:, None]]
        if state is not None:
            extra.append(state[:, None])
        return cond, torch.cat([tokens] + extra, dim=1)
