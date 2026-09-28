"""SigLIP conditioning with its cached processor; explicit offline test backend."""

import torch
from torch import nn


class MultiCameraVLMBackbone(nn.Module):
    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        output_dim=512,
        num_views=2,
        freeze_backbone=True,
        use_lora=False,
        test_mode=False,
    ):
        super().__init__()
        if use_lora:
            raise ValueError(
                "LoRA is not implemented; use full or frozen backbone training"
            )
        self.num_views, self.output_dim = num_views, output_dim
        self.freeze_backbone, self.test_mode = freeze_backbone, test_mode
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
        else:
            from transformers import AutoModel, AutoProcessor

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

    def encode_images(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5 or images.shape[1:3] != (self.num_views, 3):
            raise ValueError("Expected (B, configured_views, 3, H, W) images")
        if not torch.isfinite(images).all() or ((images < 0) | (images > 1)).any():
            raise ValueError("Input RGB images must be floating values in [0,1]")
        B = images.shape[0]
        flat = images.reshape(-1, *images.shape[2:])
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
            features = self.vision_encoder.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor) or features.shape != (
            batch_size,
            self.text_feature_dim,
        ):
            raise ValueError("Unexpected text feature shape from configured backbone")
        return self.text_proj(features.to(self.text_proj[0].weight.dtype))

    def forward(self, images, instructions=None):
        vision = self.encode_images(images)
        text = self.encode_text(instructions, images.shape[0], images.device)
        gate = self.cross_gate(torch.cat((vision, text), dim=-1))
        return self.final_proj(gate * vision + (1 - gate) * text)
