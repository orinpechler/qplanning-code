"""Configuration of the noise policy's architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoisePolicyConfig:
    """What the noise policy looks like.

    The noise shape itself is not configured here: it is FastWAM's (``chunk_size`` x the
    action expert's ``action_dim``, 32 x 7 for the LIBERO checkpoint) and is read off the
    loaded model by :func:`noise_policy.fastwam.noise_shape`.
    """

    # "full": a mean and log-std for every element of the (H, A) noise.
    # "time_shared": one (A,) Gaussian; the single draw is replicated across all H time steps.
    noise_mode: str = "full"

    # Frozen encoders, the same models the Q-function uses.
    dino_model_name: str = "facebook/dinov2-large"
    text_encoder_model: str = "google/t5-v1_1-base"
    # Optional Q-function checkpoint to copy the (LIBERO fine-tuned) DINOv2 weights from,
    # instead of the pretrained ones. Still frozen here. Note the Q-function was trained on
    # images that went through its own preprocessor first, so those weights expect that input.
    dino_weights_from: Optional[str] = None
    camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    image_size: int = 224             # square resize before DINOv2 (patch 14 -> 16 x 16 tokens)
    state_dim: int = 8                # proprioception: FastWAM's observation.state on LIBERO

    # The trainable part: learned query tokens cross-attending to the encoded observation.
    dim_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dim_feedforward: int = 2048
    dropout: float = 0.0

    # log sigma is clamped to this range. The head starts at mean 0, log sigma 0, i.e. at
    # FastWAM's own N(0, I).
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def __post_init__(self) -> None:
        if self.noise_mode not in ("full", "time_shared"):
            raise ValueError(f"noise_mode must be 'full' or 'time_shared', got {self.noise_mode!r}")
        if self.dim_model % self.n_heads != 0:
            raise ValueError(f"dim_model ({self.dim_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.image_size % 14 != 0:
            raise ValueError(f"image_size must be a multiple of DINOv2's patch size 14, got {self.image_size}")
        if not self.log_std_min < 0.0 < self.log_std_max:
            raise ValueError("log_std_min < 0 < log_std_max is needed so the policy can start at N(0, I)")
