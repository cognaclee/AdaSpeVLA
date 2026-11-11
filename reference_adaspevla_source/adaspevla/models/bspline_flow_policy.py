import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal positional embedding for continuous flow time t in [0, 1].
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FlowMatchingTransformerBlock(nn.Module):
    """
    Transformer block with adaptive LayerNorm (AdaLN) modulation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        # Optional cross-attention from control-point tokens to observation tokens.
        self.cross_attention = cross_attention
        if cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, batch_first=True
            )
        # AdaLN modulation scale and shift parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, K, hidden_dim), c: (B, hidden_dim), memory: (B, M, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-attention with modulated LayerNorm
        norm_x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        if self.cross_attention:
            if memory is None:
                raise ValueError("This flow block requires an observation token memory")
            cross_out, _ = self.cross_attn(
                self.norm_cross(x), memory, memory, need_weights=False
            )
            x = x + cross_out

        # MLP with modulated LayerNorm
        norm_x2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class BSplineFlowMatchingPolicy(nn.Module):
    """
    Conditional Flow-Matching Policy Generator for Task-Space End-Effector B-Spline Control Points.
    Directly predicts control points P in R^{K x d_a}.
    """

    def __init__(
        self,
        num_control_points: int = 8,
        action_dim: int = 7,  # Per arm: six normalized offsets and the normalized width
        cond_dim: int = 512,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        memory_dim: Optional[int] = None,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_control_points = num_control_points
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input projection for (K, d_a) control points
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_control_points, hidden_dim) * 0.02
        )

        # Time & condition conditioning projection
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer Backbone
        self.blocks = nn.ModuleList(
            [
                FlowMatchingTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    cross_attention=memory_dim is not None,
                )
                for _ in range(num_layers)
            ]
        )
        if memory_dim is not None:
            self.memory_proj = nn.Sequential(
                nn.LayerNorm(memory_dim), nn.Linear(memory_dim, hidden_dim)
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluates the vector field v_theta(x_t, t, cond).
        Args:
            x_t: (B, K, d_a) Noisy control points at continuous time t in [0, 1].
            t: (B,) Generative flow time, distinct from physical trajectory time.
            cond: (B, cond_dim) Multimodal condition embedding.
            memory: (B, M, memory_dim) observation tokens, required with memory_dim.
        Returns:
            v_pred: (B, K, d_a) Predicted velocity field.
        """
        B = x_t.shape[0]

        # Combine time & condition representations
        t_embed = self.time_mlp(self.time_emb(t))
        c_embed = self.cond_proj(cond)
        global_cond = t_embed + c_embed  # (B, hidden_dim)

        # Project control points to tokens
        h = self.input_proj(x_t) + self.pos_emb  # (B, K, hidden_dim)

        if self.memory_dim is not None:
            if memory is None:
                raise ValueError("This flow policy requires an observation token memory")
            memory = self.memory_proj(memory)

        # Pass through Transformer blocks
        for block in self.blocks:
            h = block(h, global_cond, memory)

        h = self.final_norm(h)
        v_pred = self.output_proj(h)  # (B, K, d_a)
        return v_pred

    def compute_loss(
        self,
        target_control_points: torch.Tensor,
        cond: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes the Conditional Flow Matching (CFM) loss.
        Target flow: x_t = t * x_1 + (1 - t) * x_0,  v* = x_1 - x_0
        where x_1 is ground truth control points, x_0 ~ N(0, I).
        """
        B = target_control_points.shape[0]
        device = target_control_points.device

        # Sample generative flow time t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # Sample source noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(target_control_points)
        x_1 = target_control_points

        # Linear noise-to-data interpolation (independent coupling)
        t_expand = t[:, None, None]
        x_t = t_expand * x_1 + (1.0 - t_expand) * x_0
        target_velocity = x_1 - x_0

        # Predict velocity field
        pred_velocity = self.forward_velocity(x_t, t, cond, memory)

        # Flow matching MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss, {"flow_loss": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: int = 15,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Euler ODE solver; achieved inference speed must be measured.
        Integrates dx/dt = v_theta(x, t, cond) from t=0 to t=1.
        """
        B = cond.shape[0]
        device = cond.device
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        dt = 1.0 / num_steps

        # Start from Gaussian noise x_0
        x = torch.randn(B, self.num_control_points, self.action_dim, device=device)

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            v = self.forward_velocity(x, t, cond, memory)
            x = x + dt * v

        return x
