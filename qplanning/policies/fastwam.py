"""FastWAM adapter -- the default behaviour-cloning policy.

FastWAM is a flow-matching vision-language-action model built on Wan2.2: a video branch
predicts what the scene will look like while an action expert denoises the action chunk
conditioned on that prediction. For Q-Planning none of that matters; all that is used is its
ability to draw N independent chunks from one observation while encoding that observation
once, which is what keeps a planning step affordable as N grows.

Nothing else in this package imports FastWAM. Swapping it out means writing another module
like this one -- see :mod:`qplanning.policies.base`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qplanning.policies.base import BCBundle
from qplanning.policies.registry import register_bc

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from lerobot.processor import PolicyProcessorPipeline


@register_bc("fastwam")
class FastWAMAdapter:
    """Wraps ``FastWAMPolicy`` behind :class:`~qplanning.policies.base.BCChunkSampler`."""

    def __init__(self, policy: Any, postprocessor: "PolicyProcessorPipeline",
                 libero_gripper: bool = False) -> None:
        self._policy = policy
        self._postprocessor = postprocessor
        self.chunk_size = int(policy.config.chunk_size)
        self.action_dim = int(policy.config.action_dim)
        #: FastWAM's raw gripper is [0, 1] (0 = close, 1 = open), LIBERO's -- and so the
        #: demonstrations' and the Q-function's -- is +1 = close / -1 = open. The environment
        #: postprocessor converts executed actions (FastWAMGripperRemapStep); actions_for_q
        #: has to do the same for the Q-function's inputs.
        self.libero_gripper = libero_gripper

    # ── construction ───────────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, cfg: Any, env_cfg: Any = None, **_: Any) -> BCBundle:
        """Load a FastWAM checkpoint and wrap it up for evaluation."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        from qplanning.config.paths import PathResolver

        checkpoint = str(PathResolver().require(cfg.checkpoint, "bc.checkpoint"))

        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cfg.pretrained_path = checkpoint
        if cfg.device:
            policy_cfg.device = cfg.device
        # Planning is driven from this package, not from the policy's own hook.
        if hasattr(policy_cfg, "use_planning"):
            policy_cfg.use_planning = False

        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()

        # Processors come from the checkpoint so that normalization statistics are the ones
        # the weights were trained with; rebuilding them from the config would silently use
        # different stats.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
        )

        # Note: no process-wide torch.set_grad_enabled(False) here. Every inference path
        # is already wrapped in @torch.no_grad(), and disabling autograd globally would
        # also disable it for the Q-function update in the self-improvement loop, which
        # runs in the same process.
        # Exactly when lerobot.envs.factory adds FastWAMGripperRemapStep to the env postprocessor.
        libero = env_cfg is not None and "libero" in str(getattr(env_cfg, "type", "")).lower()
        adapter = cls(policy=policy, postprocessor=postprocessor, libero_gripper=libero)
        return BCBundle(
            sampler=adapter,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_config=policy_cfg,
            n_action_steps=int(policy_cfg.n_action_steps),
        )

    # ── the BCChunkSampler interface ───────────────────────────────────────────────────
    def reset(self) -> None:
        self._policy.reset()

    def sample_chunks(
        self,
        batch: dict[str, "Tensor"],
        n_samples: int,
        *,
        denoise_steps: int | None = None,
        generator: "torch.Generator | None" = None,
    ) -> "Tensor":
        """Draw N chunks. FastWAM encodes the observation once and repeats only the denoiser."""
        import torch

        device = next(self._policy.parameters()).device
        with torch.no_grad():
            candidates = self._policy.predict_n_action_chunks(
                batch, n_samples, num_inference_steps=denoise_steps
            )
        candidates = candidates.to(device=device)
        # The policy squeezes the sample dimension when N == 1.
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)
        return candidates

    def unnormalize_actions(self, actions: "Tensor") -> "Tensor":
        """Apply the policy's postprocessor, flattening everything but the action dimension."""
        original_shape = actions.shape
        flat = actions.reshape(-1, original_shape[-1])
        raw = self._postprocessor(flat)
        return raw.reshape(original_shape).to(actions.device)

    def actions_for_q(self, actions: "Tensor") -> "Tensor":
        """FastWAM-normalized actions in the Q-function's raw units.

        On LIBERO the gripper goes from FastWAM's [0, 1] to LIBERO's convention with
        1 - 2x: the environment's own remap, sign(1 - 2x), without the sign, so it stays
        affine and differentiable. FastWAM's gripper sits at ~0 or ~1, i.e. at ~+1 or ~-1 here.
        """
        import torch

        raw = self.unnormalize_actions(actions)
        if not self.libero_gripper:
            return raw
        return torch.cat([raw[..., :-1], 1.0 - 2.0 * raw[..., -1:]], dim=-1)

    def to(self, device: Any) -> "FastWAMAdapter":
        self._policy.to(device)
        return self

    def parameters(self):
        return self._policy.parameters()

    # ── convenience ────────────────────────────────────────────────────────────────────
    @property
    def policy(self) -> Any:
        """The underlying LeRobot policy, for callers that need to move it between devices."""
        return self._policy

    def __repr__(self) -> str:  # pragma: no cover
        return (f"FastWAMAdapter(chunk_size={self.chunk_size}, action_dim={self.action_dim}, "
                f"libero_gripper={self.libero_gripper})")
