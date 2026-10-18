"""FastWAM's action denoising chain, differentiable in the initial noise.

FastWAM turns Gaussian noise into an action chunk with a fixed number of Euler steps of its
action expert, every step cross-attending to a video KV cache built once from the
observation. Everything that depends only on the observation (text and proprio context,
first-frame latents, the video KV cache, the timestep schedule) is computed here once,
under ``no_grad``. What remains is a pure function of the noise:

    a = denoise(eps)

which this module evaluates with autograd enabled, so dQ(a)/d eps can be taken back through
every denoising step.

The chain is a transcription of ``FastWAM.infer_action`` (the ``fastwam`` model variant).
FastWAM wraps its per-step predictor in ``@torch.no_grad()``; the undecorated function is
called instead (``__wrapped__``), which is the only change. :meth:`QGradientPlanner.plan`
checks once, on the first planning step, that this chain reproduces FastWAM's own sampler
for the same noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


class FastWAMDenoiser:
    """``denoise(eps)`` for one observation, with the observation-only work done up front."""

    def __init__(self, model: Any, prepared: dict[str, Any], action_horizon: int) -> None:
        self.model = model
        self.action_horizon = action_horizon
        self._p = prepared
        # FastWAM decorates this with @torch.no_grad(); the plain function lets autograd through.
        self._predict = type(model)._predict_action_noise_with_cache.__wrapped__

    @property
    def noise_shape(self) -> tuple[int, int]:
        return self.action_horizon, int(self.model.action_expert.action_dim)

    # ── construction ───────────────────────────────────────────────────────────────────
    @classmethod
    def prepare(cls, sampler: Any, batch: dict[str, "Tensor"], denoise_steps: int | None) -> "FastWAMDenoiser":
        """Encode one observation exactly as ``predict_n_action_chunks`` + ``infer_action`` do."""
        import torch
        from torch import Tensor

        from lerobot.policies.fastwam.modeling_fastwam import PROMPT_TEMPLATE

        policy = sampler.policy
        model = policy.model
        if str(getattr(model.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("noise optimization requires video_attention_mask_mode='first_frame_causal'")

        with torch.no_grad():
            policy.eval()
            model.eval()

            batch_i = {k: v[0:1] for k, v in batch.items() if isinstance(v, Tensor)}
            task_list = batch.get("task", None)
            if task_list is not None:
                batch_i["task"] = [task_list[0]]
            image = policy._prepare_image(batch_i)
            proprio = policy._get_proprio(batch_i)
            task = batch_i["task"][0] if batch_i.get("task") else ""
            steps = denoise_steps if denoise_steps is not None else policy.config.num_inference_steps
            action_horizon = int(policy.config.chunk_size)

            if image.ndim == 3:
                image = image.unsqueeze(0)
            if proprio is not None:
                if proprio.ndim == 1:
                    proprio = proprio.unsqueeze(0)
                proprio = proprio.to(device=model.device, dtype=model.torch_dtype)

            image = image.to(device=model.device, dtype=model.torch_dtype)
            first_frame_latents = model._encode_input_image_latents_tensor(input_image=image, tiled=False)
            fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

            context, context_mask = model.encode_prompt(PROMPT_TEMPLATE.format(task=task))
            if proprio is not None:
                context, context_mask = model._append_proprio_to_context(
                    context=context, context_mask=context_mask, proprio=proprio
                )

            timestep_video = torch.zeros(
                (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=model.device
            )
            video_pre = model.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = model._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_horizon,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = model.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
            timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=steps, device=model.device, dtype=model.torch_dtype, shift_override=None,
            )

        prepared = {
            "context": context,
            "context_mask": context_mask,
            "video_kv_cache": video_kv_cache,
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
            "timesteps": timesteps,
            "deltas": deltas,
        }
        return cls(model=model, prepared=prepared, action_horizon=action_horizon)

    # ── the chain ──────────────────────────────────────────────────────────────────────
    def draw_noise(self, n_samples: int) -> "Tensor":
        """The same draw ``infer_action`` makes: float32 on CPU from the global RNG."""
        import torch

        return torch.randn((n_samples, *self.noise_shape), device="cpu", dtype=torch.float32)

    def __call__(self, noise: "Tensor") -> "Tensor":
        """Denoise ``(N, H, A)`` noise into ``(N, H, A)`` BC-normalized float32 chunks.

        Differentiable in ``noise`` whenever autograd is enabled.
        """
        model, p = self.model, self._p
        n = noise.shape[0]
        latents = noise.to(device=model.device, dtype=model.torch_dtype)
        context = p["context"].expand(n, -1, -1)
        context_mask = p["context_mask"].expand(n, -1)
        kv_cache = [{"k": layer["k"].expand(n, -1, -1), "v": layer["v"].expand(n, -1, -1)}
                    for layer in p["video_kv_cache"]]
        for t, delta in zip(p["timesteps"], p["deltas"]):
            timestep = t.unsqueeze(0).to(dtype=latents.dtype, device=model.device)
            velocity = self._predict(
                model,
                latents_action=latents,
                timestep_action=timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=kv_cache,
                attention_mask=p["attention_mask"],
                video_seq_len=p["video_seq_len"],
            )
            latents = model.infer_action_scheduler.step(velocity, delta, latents)
        return latents.float()
