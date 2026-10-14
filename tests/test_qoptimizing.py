"""Q-gradient ascent on the action chunk and on the denoiser's noise, with CPU stubs.

The real denoising chain is checked against FastWAM's own sampler on the GPU (the planner
does that on its first step; ``python -m qoptimizing.probe --qopt.target=noise``). Here a
small differentiable map stands in for it, so the ascent logic itself can be pinned down.
"""

from __future__ import annotations

import pytest
import torch

from qoptimizing.config import QOptConfig
from qoptimizing.optimizer import QGradientPlanner
from qplanning.planner import PlannerContext
from test_planner_math import StubQPolicy, identity_pre, scale_post

H, A = 4, 3


class StubQPolicyWithProj(StubQPolicy):
    """The planner reads the decoder's dtype off ``action_proj``."""

    def __init__(self):
        super().__init__(horizon=H, action_dim=A)
        self.q_online.action_proj = torch.nn.Linear(1, 1)


class StubDenoiser:
    """A fixed differentiable map standing in for FastWAM's chain; records its last input."""

    def __init__(self):
        g = torch.Generator().manual_seed(1)
        self.weight = torch.randn(A, A, generator=g)
        self.last_input = None

    def __call__(self, noise):
        self.last_input = noise.detach().clone()
        return torch.tanh(noise @ self.weight)


def _planner(**cfg):
    planner = QGradientPlanner(
        cfg=QOptConfig(**cfg),
        ctx=PlannerContext(q_policy=StubQPolicyWithProj(), q_pre=identity_pre,
                           camera_keys=("observation.images.image",), horizon=H),
        bc_unnormalize=scale_post,
        denoise_steps=None,
    )
    planner._affine = (torch.full((H, A), 2.5), torch.full((H, A), -0.75))
    return planner


def _context():
    return torch.randn(6, 1, 16, generator=torch.Generator().manual_seed(2))


def _start(n=3):
    return torch.randn(n, H, A, generator=torch.Generator().manual_seed(3))


def test_noise_ascent_regenerates_the_chunk_from_the_updated_noise():
    planner, denoiser, eps0 = _planner(target="noise", steps=1, step_size=0.1), StubDenoiser(), _start()
    initial, final, trajectory, _ = planner._ascend(eps0, _context(), denoiser)
    assert torch.allclose(initial, torch.tanh(eps0 @ denoiser.weight))
    assert torch.allclose(final, torch.tanh(denoiser.last_input @ denoiser.weight))
    assert not torch.allclose(denoiser.last_input, eps0)
    assert len(trajectory) == 2
    assert bool((trajectory[1] > trajectory[0]).all()), "a small step along dQ/d eps must raise Q"


def test_noise_keeps_its_norm():
    planner, denoiser, eps0 = _planner(target="noise", steps=2, step_size=0.5), StubDenoiser(), _start()
    planner._ascend(eps0, _context(), denoiser)
    assert torch.allclose(denoiser.last_input.flatten(1).norm(dim=1), eps0.flatten(1).norm(dim=1), atol=1e-5)


def test_noise_step_follows_dq_through_the_denoiser():
    """The step is dQ(D(eps))/d eps -- the chain rule through D -- not dQ/da."""
    planner, denoiser, eps0 = _planner(target="noise", steps=1, step_size=1e-3,
                                       normalize_grad=False, renorm_noise=False), StubDenoiser(), _start(2)
    context = _context()
    planner._ascend(eps0, context, denoiser)
    step = denoiser.last_input - eps0

    eps = eps0.clone().requires_grad_(True)
    scale, shift = planner._affine
    q = planner._q_values(denoiser(eps), context.expand(-1, 2, -1).contiguous(), scale, shift)
    (expected,) = torch.autograd.grad(q.sum(), eps)
    # The step is recovered as (eps0 + step) - eps0 in float32, hence the tolerance.
    assert torch.allclose(step, 1e-3 * expected, rtol=1e-2, atol=1e-7)


def test_zero_steps_scores_without_moving_anything():
    planner, denoiser, eps0 = _planner(target="noise", steps=0), StubDenoiser(), _start()
    initial, final, trajectory, grad_rms = planner._ascend(eps0, _context(), denoiser)
    assert torch.equal(initial, final) and len(trajectory) == 1 and grad_rms is None


def test_action_target_takes_the_plain_normalized_step():
    planner, a0 = _planner(target="action", steps=1, step_size=0.01), _start()
    initial, final, _, grad_rms = planner._ascend(a0, _context(), None)
    assert torch.equal(initial, a0)
    move = (final - a0).pow(2).mean(dim=(1, 2)).sqrt()
    assert torch.allclose(move, torch.full_like(move, 0.01), atol=1e-6)
    assert grad_rms is not None and bool((grad_rms > 0).all())


def test_unknown_target_is_refused():
    with pytest.raises(ValueError, match="target"):
        QOptConfig(target="latent")
