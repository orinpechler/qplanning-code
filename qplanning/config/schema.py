"""The typed configuration tree.

Every field a run needs lives here, and every run is fully described by one YAML plus
optional CLI overrides.

Paths are kept as strings here and turned into real locations by
:class:`qplanning.config.paths.PathResolver`, which is what allows a value to be a local
directory, a Hugging Face repo id, or an ``${env:VAR}`` reference without the schema caring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Paths:
    """Every location the project reads from or writes to."""

    root: str = "./artifacts"
    bc_checkpoint: Optional[str] = None
    q_checkpoint: Optional[str] = None
    wan22_weights: Optional[str] = None
    dataset_root: Optional[str] = None
    robotwin_root: Optional[str] = None
    hf_home: Optional[str] = None
    output_dir: str = "artifacts/runs/qplanning"


@dataclass
class BCConfig:
    """The frozen behaviour-cloning policy. Never updated by anything in this repository."""

    type: str = "fastwam"
    checkpoint: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    # Denoising steps for the unguided baseline. Q-Planning uses planner.denoise_steps
    # instead, which is deliberately lower to keep the candidate draws diverse.
    baseline_denoise_steps: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QFunctionConfigYAML:
    """Q-function settings.

    ``h``, ``camera_keys`` and the HL-Gauss bin geometry are properties of a trained
    checkpoint, not free parameters at evaluation time. They are still listed so a config
    is self-documenting; :func:`qplanning.configuration_qfunction.assert_matches_checkpoint`
    checks them against the loaded checkpoint and fails loudly on a mismatch.
    """

    checkpoint: Optional[str] = None
    h: int = 32                       # H, the action-chunk length; must equal BC chunk_size
    gamma: float = 0.99
    num_bins: int = 101               # B, HL-Gauss bins
    v_min: float = -0.01
    v_max: float = 1.01
    hl_gauss_sigma: float = 0.0075
    target_tau: float = 0.005         # EMA rate for the target network
    camera_keys: list[str] = field(default_factory=list)
    dino_model_name: str = "facebook/dinov2-large"
    text_encoder_model: str = "google/t5-v1_1-base"
    dim_model: int = 1024
    n_heads: int = 16
    dim_feedforward: int = 4096
    n_decoder_layers: int = 18
    reward_mode: str = "sparse"
    step_reward: float = 0.0
    terminal_bonuses: dict[str, float] = field(default_factory=lambda: {"q5": 1.0})
    bucket_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class PlannerConfig:
    """How candidate action chunks are drawn and aggregated.

    ``q_weighted`` draws ``n_samples`` chunks from a short
    flow-matching pass, score them all in one batched Q-decoder pass, and execute the
    softmax-Q-weighted mean of the best ``n_elites``. ``none`` is the unguided BC baseline.
    """

    type: str = "q_weighted"          # q_weighted | none
    n_samples: int = 64               # N
    n_elites: int = 16                # K; 0 means use all N
    temperature: float = 1.0          # lambda in the softmax weighting
    denoise_steps: int = 3            # flow-matching steps per candidate
    seed: Optional[int] = None        # None inherits the run seed

    def __post_init__(self) -> None:
        if self.type not in ("q_weighted", "none"):
            raise ValueError(
                f"planner.type must be 'q_weighted' or 'none', got {self.type!r}. "
                "The noise-based MPPI variants from the appendix are not part of this release."
            )
        if self.type == "none":
            return
        if self.n_samples <= 0:
            raise ValueError(f"planner.n_samples must be > 0, got {self.n_samples}")
        if self.n_elites < 0:
            raise ValueError(f"planner.n_elites must be >= 0, got {self.n_elites}")
        # K is clamped rather than rejected, so lowering N alone (e.g. for a smoke run)
        # stays valid. 0 keeps its meaning: aggregate over all N.
        if self.n_elites > self.n_samples:
            self.n_elites = self.n_samples
        if self.temperature <= 0:
            raise ValueError(f"planner.temperature must be > 0, got {self.temperature}")
        if self.denoise_steps <= 0:
            raise ValueError(f"planner.denoise_steps must be > 0, got {self.denoise_steps}")


@dataclass
class ExecutionConfig:
    """How much of each planned chunk is executed before replanning."""

    n_action_steps: int = 10          # 10 on LIBERO, 24 on RoboTwin


@dataclass
class BenchmarkConfig:
    """Which environment and which tasks."""

    name: str = "libero_10"
    env_type: str = "libero"          # libero | robotwin
    suite: Optional[str] = None       # LIBERO suite name
    # "all", "paper47" (RoboTwin minus the three tasks whose success check is broken),
    # or an explicit list of task names.
    tasks: Union[str, list[str]] = "all"
    # None means the benchmark's own per-suite default.
    episode_length: Optional[int] = None
    observation_height: int = 224
    observation_width: int = 224

    def __post_init__(self) -> None:
        if self.env_type not in ("libero", "robotwin"):
            raise ValueError(f"benchmark.env_type must be 'libero' or 'robotwin', got {self.env_type!r}")


@dataclass
class EvalConfig:
    """Rollout protocol. Batch size is fixed at 1 because the planner scores one state."""

    episodes_per_task: int = 20
    start_seed: int = 42
    # Eval-parity determinism: deterministic kernels, TF32 off, cudnn.benchmark off.
    # This is what the reported success rates were measured under.
    deterministic: bool = True
    max_episodes_rendered: int = 0
    shard: Optional[str] = None       # "k/n" to split tasks across jobs


@dataclass
class WandbConfig:
    project: str = ""
    entity: str = ""
    run_name: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.project)


@dataclass
class DatasetConfig:
    """The demonstration dataset used for Q training and the offline half of the SI mixture."""

    repo_ids: list[str] = field(default_factory=list)
    root: Optional[str] = None
    bucket_overrides: dict[str, str] = field(default_factory=dict)
    terminal_bonuses: dict[str, float] = field(default_factory=lambda: {"q5": 1.0})
    use_imagenet_stats: bool = False


@dataclass
class TrainQConfig:
    """Q-function training on demonstrations."""

    steps: int = 40000
    batch_size: int = 48
    num_gpus: int = 4
    optimizer_lr: float = 3e-4
    optimizer_lr_backbone: float = 9e-5
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine_decay_with_warmup"
    lr_warmup_steps: int = 2000
    lr_decay_steps: int = 40000
    lr_decay_min: float = 1e-6
    save_freq: int = 1000
    log_freq: int = 50
    num_workers: int = 6
    test_split_ratio: float = 0.1
    test_freq: int = 500
    test_n_batches: int = 1
    resume_from: Optional[str] = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class SelfImproveConfig:
    """The Q-only self-improvement loop.

    Each iteration rolls out two blocks per task in a single pass. The *held-out* block uses
    seeds that are fixed across iterations and is never written to the replay buffer -- it is
    the measurement, and its iteration-0 and iteration-N values are the two Q-Planning
    numbers for the loop. The *train* block uses advancing seeds and feeds the buffer.
    """

    iterations: int = 10
    start_iteration: int = 0
    train_episodes_per_task: int = 10
    heldout_episodes_per_task: int = 20
    heldout_seed_base: int = 42
    finetune_steps: int = 200         # S
    finetune_lr: float = 1e-5
    batch_size: int = 48
    online_fraction: float = 0.5      # half of every minibatch from online rollouts
    grad_clip_norm: float = 10.0
    collect_shards: int = 1
    min_shards_to_finetune: Optional[int] = None   # default: ceil(collect_shards / 2)
    offline: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.online_fraction <= 1.0:
            raise ValueError(f"self_improve.online_fraction must be in [0,1], got {self.online_fraction}")
        if self.collect_shards < 1:
            raise ValueError(f"self_improve.collect_shards must be >= 1, got {self.collect_shards}")

    def resolved_min_shards(self) -> int:
        if self.min_shards_to_finetune is not None:
            return self.min_shards_to_finetune
        return (self.collect_shards + 1) // 2


@dataclass
class RunConfig:
    """The root object every subcommand parses."""

    run_name: str = "qplanning"
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    bc: BCConfig = field(default_factory=BCConfig)
    qfunction: QFunctionConfigYAML = field(default_factory=QFunctionConfigYAML)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train_q: TrainQConfig = field(default_factory=TrainQConfig)
    self_improve: SelfImproveConfig = field(default_factory=SelfImproveConfig)
