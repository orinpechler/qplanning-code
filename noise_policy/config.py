"""Configuration of the noise policy: its architecture, and the two ways of training it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from qoptimizing.config import QOptConfig
from qplanning.config.schema import RunConfig, WandbConfig


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


@dataclass
class DemoStatesConfig:
    """Where training states come from: single frames of the demonstrations."""

    repo_id: str = "HuggingFaceVLA/libero"
    root: Optional[str] = None        # null -> paths.dataset_root
    # Only frames of this LIBERO suite's tasks ("libero_10", ...); null uses every frame.
    suite: Optional[str] = "libero_10"


@dataclass
class NoiseTrainConfig:
    """The loop both training scripts share.

    Training alternates collection rounds and policy updates. A round draws
    ``states_per_round`` demonstration states, labels ``samples_per_state`` noises for each
    (how depends on the script) and pushes them into a FIFO replay buffer; then
    ``updates_per_round`` gradient steps are taken on batches of ``batch_size`` states drawn
    from the buffer.
    """

    steps: int = 4000                 # policy updates in total
    batch_size: int = 32              # states per update (each with all its samples)
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    lr_warmup_steps: int = 100
    grad_clip_norm: float = 1.0

    samples_per_state: int = 16       # M
    states_per_round: int = 32
    updates_per_round: int = 16
    warmup_states: int = 64           # labelled before the first update
    # Replay capacity, in states. Small on purpose: a state is dropped after
    # buffer_states / states_per_round rounds (8 by default, i.e. 128 updates), so targets
    # labelled with an old policy do not linger. How often each state is reused is set by
    # batch_size * updates_per_round / states_per_round (16 by default), not by this.
    buffer_states: int = 256
    # FastWAM flow steps, for sampling and for the reversal. null -> bc.baseline_denoise_steps.
    denoise_steps: Optional[int] = None

    log_freq: int = 20
    save_freq: int = 1000
    # Progress of pi on a fixed set of demonstration states, before training and every
    # `monitor_freq` updates: Q of FastWAM's chunks from pi's noise against N(0, I) noise, with
    # the same base draws (eps_pi = mu + sigma * eps, eps_prior = eps) so only pi changes.
    monitor_states: int = 32
    monitor_samples: int = 32         # base draws per monitored state
    monitor_freq: int = 200
    data: DemoStatesConfig = field(default_factory=DemoStatesConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "samples_per_state", "states_per_round", "updates_per_round",
                     "buffer_states", "log_freq", "save_freq", "monitor_samples", "monitor_freq"):
            if getattr(self, name) <= 0:
                raise ValueError(f"noise_train.{name} must be > 0, got {getattr(self, name)}")
        if not self.batch_size <= self.warmup_states <= self.buffer_states:
            raise ValueError("noise_train needs batch_size <= warmup_states <= buffer_states")
        if self.monitor_states < 0:
            raise ValueError("noise_train.monitor_states must be >= 0 (0 turns monitoring off)")
        if self.states_per_round > self.buffer_states:
            raise ValueError("noise_train.states_per_round must fit in the buffer")


@dataclass
class ImportanceWeightConfig:
    """``train_iw``: weight M sampled noises by softmax(Q / temperature) per state."""

    # Q differences between noises of one state are ~0.01, so tau ~0.01 gives a real preference.
    temperature: float = 0.01
    # "policy": eps ~ pi(.|s), reweighted by N(eps) / pi(eps | s) towards the tilted target
    # N(eps) exp(Q / tau); once pi is good, more of its samples land where Q is high.
    # "prior": eps ~ N(0, I), as in the notes -- the same target, sampled blindly.
    proposal: str = "policy"

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"iw.temperature must be > 0, got {self.temperature}")
        if self.proposal not in ("prior", "policy"):
            raise ValueError(f"iw.proposal must be 'prior' or 'policy', got {self.proposal!r}")


@dataclass
class ReversalConfig:
    """``train_reversal``: improve the action under Q, flow-reverse it, fit the noise."""

    # Where a_0 = F(s, eps) starts: eps ~ pi(.|s) as in the notes, or eps ~ N(0, I).
    start: str = "policy"
    # The improvement: anchored gradient ascent a_{k+1} = a_k + eta grad[Q(a_k) - lambda ||a_k - a_0||^2],
    # exactly as in the qopt rollouts. lambda is `improve.anchor_weight`; `improve.target` must be
    # "action" and `improve.n_samples` is unused (noise_train.samples_per_state applies).
    improve: QOptConfig = field(default_factory=lambda: QOptConfig(steps=30, step_size=0.01, anchor_weight=1.0))
    # Extra fixed-point passes per reversal step (see FastWAMDenoiser.invert); 0 is plain
    # backward Euler.
    refine_iters: int = 0

    def __post_init__(self) -> None:
        if self.start not in ("policy", "prior"):
            raise ValueError(f"reversal.start must be 'policy' or 'prior', got {self.start!r}")
        if self.improve.target != "action":
            raise ValueError("reversal.improve.target must be 'action'")
        if self.refine_iters < 0:
            raise ValueError(f"reversal.refine_iters must be >= 0, got {self.refine_iters}")


@dataclass
class NoisePolicyRunConfig(RunConfig):
    """A normal run config (benchmark, checkpoints, paths) plus the noise-policy blocks."""

    noise_policy: NoisePolicyConfig = field(default_factory=NoisePolicyConfig)
    noise_train: NoiseTrainConfig = field(default_factory=NoiseTrainConfig)
    iw: ImportanceWeightConfig = field(default_factory=ImportanceWeightConfig)
    reversal: ReversalConfig = field(default_factory=ReversalConfig)
