"""The noise policy: observation, instruction and proprioception -> a Gaussian over FastWAM's noise.

    images (B, V, 3, H, W) --frozen DINOv2--> patch tokens --Linear--> + view / patch embeddings ┐
    instruction            --frozen T5-----> word tokens  --Linear--> + type embedding         ├─ context
    state (B, S)           --MLP-------------------------------------> 1 token                 ┘
    K learned query tokens --[self-attn, cross-attn to context, FFN] x n_layers--> Linear -> (mu, log sigma)

K is the chunk length H with ``noise_mode=full`` (one query per time step, each emitting a
mean and log-std for the A action dimensions), or 1 with ``noise_mode=time_shared`` (one
(A,) Gaussian whose draw is replicated over all H time steps).

The output layer starts at zero, so an untrained policy is exactly N(0, I) -- FastWAM's own
noise distribution -- and training moves away from the baseline rather than starting off it.
Only the projections, embeddings, decoder and head train; the encoders are frozen and kept in
eval mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from noise_policy.config import NoisePolicyConfig


@dataclass
class NoiseDistribution:
    """A diagonal Gaussian over (H, A) noise, possibly shared across time.

    ``mean`` and ``log_std`` are ``(B, K, A)`` with K = H (full) or K = 1 (time-shared). Noise
    is always handed out as ``(..., H, A)``; with K = 1 its H rows are identical.

    Time-shared densities are per row, averaged over the H rows: for replicated noise that is
    exactly the density of its one free (A,) draw, and for any other noise (a flow-reversed
    target, say) it is the shared Gaussian fitted to all H rows.
    """

    mean: Tensor
    log_std: Tensor
    horizon: int

    @property
    def std(self) -> Tensor:
        return self.log_std.exp()

    @property
    def time_shared(self) -> bool:
        return self.mean.shape[1] == 1 and self.horizon > 1

    @property
    def n_free(self) -> int:
        """How many values one draw actually has: K x A."""
        return self.mean.shape[1] * self.mean.shape[2]

    def _to_noise(self, z: Tensor) -> Tensor:
        return z.expand(*z.shape[:-2], self.horizon, z.shape[-1]) if self.time_shared else z

    def sample(self, n_samples: int = 1, generator: torch.Generator | None = None) -> Tensor:
        """``(B, n_samples, H, A)`` noise, reparameterized (differentiable in mean and std)."""
        shape = (self.mean.shape[0], n_samples, *self.mean.shape[1:])
        eps = torch.randn(shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.transform(eps)

    def transform(self, base: Tensor) -> Tensor:
        """``(B, n, K, A)`` standard-normal draws -> ``(B, n, H, A)`` noise from this distribution.

        With the same ``base``, policy and prior noise differ only by the policy, which is
        what lets training compare them on common random numbers.
        """
        return self._to_noise(self.mean.unsqueeze(1) + self.std.unsqueeze(1) * base)

    def log_prob(self, noise: Tensor) -> Tensor:
        """Log density of ``(B, n, H, A)`` noise; returns ``(B, n)``."""
        normal = torch.distributions.Normal(self.mean.unsqueeze(1), self.std.unsqueeze(1))
        per_element = normal.log_prob(noise)                       # broadcasts K = 1 over H
        if self.time_shared:
            return per_element.mean(dim=-2).sum(dim=-1)
        return per_element.sum(dim=(-2, -1))

    def entropy(self) -> Tensor:
        """``(B,)``."""
        return torch.distributions.Normal(self.mean, self.std).entropy().sum(dim=(-2, -1))

    def kl_to_standard_normal(self) -> Tensor:
        """KL(pi || N(0, I)) over the free values, ``(B,)``: how far from FastWAM's own noise."""
        return (0.5 * (self.mean.pow(2) + self.std.pow(2) - 1.0) - self.log_std).sum(dim=(-2, -1))


class _DecoderLayer(nn.Module):
    """Pre-norm: self-attention among the queries, cross-attention to the context, FFN."""

    def __init__(self, dim: int, n_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor, context_padding: Tensor | None) -> Tensor:
        h = self.norm_self(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_cross(x)
        ctx = self.norm_context(context)
        x = x + self.cross_attn(h, ctx, ctx, key_padding_mask=context_padding, need_weights=False)[0]
        return x + self.ffn(self.norm_ffn(x))


class NoisePolicy(nn.Module):
    """pi(eps | images, instruction, state) over FastWAM's ``noise_shape = (H, A)`` noise.

    ``image_backbone`` maps ``(N, 3, s, s)`` ImageNet-normalized images to ``(N, P, D_img)``
    patch tokens; ``text_encoder.encode(texts, device)`` maps strings to ``(B, T, D_txt)``
    zero-padded token embeddings. Both are frozen here. :meth:`build` creates the real ones
    (DINOv2 and T5, as in the Q-function); tests pass small stand-ins.
    """

    def __init__(
        self,
        cfg: "NoisePolicyConfig",
        noise_shape: tuple[int, int],
        image_backbone: nn.Module,
        image_dim: int,
        text_encoder: nn.Module,
        text_dim: int,
        n_patches: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon, self.action_dim = int(noise_shape[0]), int(noise_shape[1])
        self.n_queries = self.horizon if cfg.noise_mode == "full" else 1
        d = cfg.dim_model

        self.image_backbone = image_backbone
        self.text_encoder = text_encoder
        for module in (self.image_backbone, self.text_encoder):
            module.requires_grad_(False)
            module.eval()
        self.register_buffer("_img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        n_views = len(cfg.camera_keys)
        self.image_proj = nn.Linear(image_dim, d)
        self.patch_embed = nn.Parameter(torch.zeros(n_views, n_patches, d))
        self.view_embed = nn.Parameter(torch.zeros(n_views, 1, d))
        self.text_proj = nn.Linear(text_dim, d)
        self.text_embed = nn.Parameter(torch.zeros(1, 1, d))
        self.state_proj = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.state_embed = nn.Parameter(torch.zeros(1, 1, d))
        self.queries = nn.Parameter(torch.zeros(self.n_queries, d))
        for p in (self.patch_embed, self.view_embed, self.text_embed, self.state_embed, self.queries):
            nn.init.normal_(p, std=0.02)

        self.layers = nn.ModuleList(
            _DecoderLayer(d, cfg.n_heads, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2 * self.action_dim)
        # Starts at mean 0, log sigma 0: FastWAM's own N(0, I).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ── construction ───────────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, cfg: "NoisePolicyConfig", noise_shape: tuple[int, int],
              device: "torch.device | str" = "cuda") -> "NoisePolicy":
        """With the real frozen encoders: DINOv2 and T5, as in the Q-function."""
        from lerobot.policies.q_function.modeling_q_function import (
            T5TextEncoder,
            _DINOv2FinetunableBackbone,
        )

        backbone = _DINOv2FinetunableBackbone(cfg.dino_model_name, freeze=True)
        if cfg.dino_weights_from:
            _load_dino_from_q_checkpoint(backbone, cfg.dino_weights_from)
        text = T5TextEncoder(cfg.text_encoder_model)
        n_patches = (cfg.image_size // backbone.patch_size) ** 2
        policy = cls(cfg, noise_shape, image_backbone=backbone, image_dim=backbone.hidden_size,
                     text_encoder=text, text_dim=text.hidden_size, n_patches=n_patches)
        return policy.to(device)

    def train(self, mode: bool = True) -> "NoisePolicy":
        super().train(mode)
        # The encoders are frozen: no dropout, no batch-statistics updates, ever.
        self.image_backbone.eval()
        self.text_encoder.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    _FROZEN = ("image_backbone.", "text_encoder.")

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Everything but the frozen encoders, which :meth:`build` recreates (~10M of ~424M)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith(self._FROZEN)}

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith(self._FROZEN)]
        if missing or unexpected:
            raise KeyError(f"noise policy checkpoint mismatch: missing {missing}, unexpected {unexpected}")

    # ── forward ────────────────────────────────────────────────────────────────────────
    def forward(self, images: Tensor, texts: list[str], state: Tensor) -> NoiseDistribution:
        """``images`` (B, V, 3, H, W) in [0, 1], one instruction per row, ``state`` (B, S)."""
        context, padding = self.encode_context(images, texts, state)
        return self.decode(context, padding)

    def encode_context(self, images: Tensor, texts: list[str], state: Tensor) -> tuple[Tensor, Tensor]:
        """Context tokens ``(B, T + V*P + 1, D)`` and their padding mask ``(B, T + V*P + 1)``.

        Separate from :meth:`decode` so one observation can be encoded once and decoded many
        times. Gradients flow into the projections, not into the encoders.
        """
        b, v = images.shape[:2]
        with torch.no_grad():
            flat = images.reshape(b * v, *images.shape[2:]).float()
            size = self.cfg.image_size
            if flat.shape[-2:] != (size, size):
                flat = nn.functional.interpolate(flat, size=(size, size), mode="bilinear",
                                                 align_corners=False, antialias=True)
            patches = self.image_backbone((flat - self._img_mean) / self._img_std)   # (B*V, P, D_img)
            words = self.text_encoder.encode(texts, device=images.device)             # (B, T, D_txt)
        dtype = self.image_proj.weight.dtype
        image_tokens = self.image_proj(patches.to(dtype)).view(b, v, -1, self.cfg.dim_model)
        image_tokens = (image_tokens + self.patch_embed + self.view_embed).flatten(1, 2)
        text_tokens = self.text_proj(words.to(dtype)) + self.text_embed
        state_token = self.state_proj(state.to(dtype)).unsqueeze(1) + self.state_embed

        context = torch.cat([text_tokens, image_tokens, state_token], dim=1)
        # T5TextEncoder pads shorter instructions with exact zeros.
        text_padding = words.abs().sum(dim=-1) == 0
        padding = torch.cat([
            text_padding,
            torch.zeros(b, image_tokens.shape[1] + 1, dtype=torch.bool, device=context.device),
        ], dim=1)
        return context, padding

    def decode(self, context: Tensor, padding: Tensor | None = None) -> NoiseDistribution:
        x = self.queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, context, padding)
        mean, log_std = self.head(self.out_norm(x)).chunk(2, dim=-1)               # (B, K, A) each
        log_std = log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        return NoiseDistribution(mean=mean.float(), log_std=log_std.float(), horizon=self.horizon)


def _load_dino_from_q_checkpoint(backbone: nn.Module, checkpoint: str) -> None:
    """Copy the Q-function's (fine-tuned) DINOv2 weights into ``backbone``."""
    from pathlib import Path

    from safetensors.torch import load_file

    from qplanning.config.paths import PathResolver

    path = Path(str(PathResolver().resolve(checkpoint).path or checkpoint)) / "model.safetensors"
    prefix = "q_online.image_encoder.backbone.model."
    weights = {k[len(prefix):]: v for k, v in load_file(str(path)).items() if k.startswith(prefix)}
    if not weights:
        raise ValueError(f"no DINOv2 weights under {prefix!r} in {path}")
    backbone.model.load_state_dict(weights, strict=True)
