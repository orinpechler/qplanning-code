"""FastWAM's actions as the Q-function reads them: LIBERO's gripper convention, nothing else changed."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from qplanning.policies.fastwam import FastWAMAdapter


def _min_max_unnormalize(x: torch.Tensor) -> torch.Tensor:
    """FastWAM's postprocessor on LIBERO: MIN_MAX, arm in [-0.9375, 0.9375], gripper in [0, 1]."""
    low = torch.tensor([-0.9375] * 6 + [0.0])
    high = torch.tensor([0.9375] * 6 + [1.0])
    return (x + 1) / 2 * (high - low) + low


def _adapter(libero: bool) -> FastWAMAdapter:
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=32, action_dim=7))
    return FastWAMAdapter(policy, _min_max_unnormalize, libero_gripper=libero)


def test_q_inputs_use_the_libero_gripper_convention():
    adapter = _adapter(libero=True)
    x = torch.zeros(2, 32, 7)
    x[0, :, -1], x[1, :, -1] = -1.0, 1.0                       # FastWAM: close (0), open (1)
    executed = adapter.unnormalize_actions(x)
    for_q = adapter.actions_for_q(x)
    assert executed[:, 0, -1].tolist() == [0.0, 1.0]          # the execution path is untouched
    assert for_q[:, 0, -1].tolist() == [1.0, -1.0]            # LIBERO: +1 close, -1 open
    assert torch.equal(for_q[..., :-1], executed[..., :-1])
    # Same direction as the environment's own remap, sign(1 - 2x).
    assert torch.equal(torch.sign(for_q[..., -1]), torch.sign(1 - 2 * executed[..., -1]))


def test_q_conversion_stays_affine_and_differentiable():
    adapter = _adapter(libero=True)
    zero, one = adapter.actions_for_q(torch.zeros(1, 7)), adapter.actions_for_q(torch.ones(1, 7))
    x = torch.randn(5, 7, requires_grad=True)
    out = adapter.actions_for_q(x)
    assert torch.allclose(out, zero + x * (one - zero), atol=1e-6)
    out.sum().backward()
    assert x.grad is not None and torch.all(x.grad != 0)


def test_without_libero_the_q_conversion_is_plain_unnormalization():
    adapter = _adapter(libero=False)
    x = torch.randn(3, 32, 7)
    assert torch.equal(adapter.actions_for_q(x), adapter.unnormalize_actions(x))
