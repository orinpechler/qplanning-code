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


@pytest.mark.parametrize("select", ["best", "worst"])
def test_select_executes_the_highest_or_lowest_scoring_chunk(select):
    from test_planner_math import DummyBC

    planner = _planner(steps=0, n_samples=8, select=select)
    planner._affine = None   # let plan() read the BC -> Q map off the stub pipelines
    batch = {"observation.images.image": torch.randn(1, 16)}
    result = planner.plan(DummyBC(chunk_size=H, action_dim=A), batch)
    pick = torch.argmax if select == "best" else torch.argmin
    assert torch.equal(result.chunk[0], result.candidates[int(pick(result.q_values))])
    stats = planner.last_stats
    assert stats.q_final == pytest.approx(float(result.q_values.max() if select == "best" else result.q_values.min()))
    assert stats.q_cand_min <= stats.q_final <= stats.q_cand_max and stats.best_worst_rms > 0


def test_chunk_at_pads_with_the_last_action_of_the_episode():
    import numpy as np

    from qoptimizing.qrank import chunk_at

    actions = np.arange(20, dtype=np.float32).reshape(10, 2)       # frames 0..9, 2 dims
    assert chunk_at(actions, 2, 6, 3, 2).tolist() == [[6, 7], [8, 9]]
    # Past the episode end (frame 5 is its last), the last action repeats.
    assert chunk_at(actions, 2, 6, 4, 4).tolist() == [[8, 9], [10, 11], [10, 11], [10, 11]]


def test_summarize_counts_ties_as_half():
    from qoptimizing.qrank import summarize

    s = summarize([0.2, -0.1, 0.0, 0.3])
    assert s["n"] == 4 and s["rank_acc"] == pytest.approx(0.625) and s["gap_mean"] == pytest.approx(0.1)
    assert summarize([])["n"] == 0


def test_unknown_target_is_refused():
    with pytest.raises(ValueError, match="target"):
        QOptConfig(target="latent")
