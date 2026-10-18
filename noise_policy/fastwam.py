"""Running FastWAM from a chosen noise, and handing the noise policy its inputs.

FastWAM's own sampler draws its noise internally (``infer_action`` calls ``torch.randn``), so
it cannot be given one. The denoising chain in :mod:`qoptimizing.noise` is a transcription of
that sampler that takes the noise as an argument, and it is checked against FastWAM's output
for the same noise (:func:`verify_against_sampler`; the planner in :mod:`qoptimizing` runs the
same check on its first planning step, max difference 0). FastWAM itself is left unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qoptimizing.noise import FastWAMDenoiser

if TYPE_CHECKING:
    from torch import Tensor


def noise_shape(sampler: Any) -> tuple[int, int]:
    """FastWAM's noise shape ``(H, A)``: 32 x 7 for the LIBERO checkpoint."""
    return int(sampler.chunk_size), int(sampler.policy.model.action_expert.action_dim)


def prepare(sampler: Any, batch: dict[str, "Tensor"], denoise_steps: int | None = None) -> FastWAMDenoiser:
    """Encode one observation once; the result maps ``(N, H, A)`` noise to ``(N, H, A)`` chunks.

    Call it under ``torch.no_grad()`` to sample, or with autograd on to differentiate the chunk
    with respect to the noise.
    """
    return FastWAMDenoiser.prepare(sampler, batch, denoise_steps)


def sample_chunks_from_noise(sampler: Any, batch: dict[str, "Tensor"], noise: "Tensor",
                             denoise_steps: int | None = None) -> "Tensor":
    """FastWAM's chunks for ``(N, H, A)`` noise, BC-normalized float32 -- what
    ``sampler.sample_chunks`` returns when its internal draw happens to be ``noise``."""
    import torch

    expected = noise_shape(sampler)
    if tuple(noise.shape[1:]) != expected:
        raise ValueError(f"FastWAM expects noise of shape (N, {expected[0]}, {expected[1]}), got {tuple(noise.shape)}")
    with torch.no_grad():
        return prepare(sampler, batch, denoise_steps)(noise)


def verify_against_sampler(sampler: Any, batch: dict[str, "Tensor"], n_samples: int = 4,
                           denoise_steps: int | None = None) -> float:
    """Max |difference| between FastWAM's own sampler and :func:`sample_chunks_from_noise`
    given the same noise. Leaves the global RNG state as it found it."""
    import torch

    denoiser = prepare(sampler, batch, denoise_steps)
    state = torch.get_rng_state()
    noise = denoiser.draw_noise(n_samples)       # the draw infer_action makes from this state
    torch.set_rng_state(state)
    reference = sampler.sample_chunks(batch, n_samples, denoise_steps=denoise_steps)
    torch.set_rng_state(state)
    with torch.no_grad():
        ours = denoiser(noise)
    return float((ours - reference.to(device=ours.device, dtype=ours.dtype)).abs().max())


def policy_inputs(batch: dict[str, "Tensor"], camera_keys: list[str],
                  state_key: str = "observation.state") -> tuple["Tensor", list[str], "Tensor"]:
    """The noise policy's ``(images, texts, state)`` from a FastWAM-preprocessed batch.

    Images stay in [0, 1] (the policy applies DINOv2's ImageNet normalization); the state is
    FastWAM's normalized proprioception.
    """
    import torch

    views = []
    for key in camera_keys:
        image = batch[key]
        if image.dim() == 5:          # (B, T, C, H, W): the current frame
            image = image[:, 0]
        views.append(image)
    images = torch.stack(views, dim=1)
    state = batch[state_key]
    if state.dim() == 3:
        state = state[:, 0]
    task = batch.get("task") or [""] * images.shape[0]
    texts = [task] if isinstance(task, str) else list(task)
    return images, texts, state
