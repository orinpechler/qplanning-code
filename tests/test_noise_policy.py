"""The noise policy's architecture, with tiny stand-ins for the frozen DINOv2 and T5 encoders."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from noise_policy.config import NoisePolicyConfig
from noise_policy.modeling import NoisePolicy

H, A, S, PATCH = 32, 7, 8, 14


class _FakeBackbone(nn.Module):
    """(N, 3, s, s) -> (N, (s/14)^2, 16) patch tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 16, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _FakeText(nn.Module):
    """Strings -> zero-padded (B, T, 12) tokens, one token per word, like T5TextEncoder."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, 12)

    def encode(self, texts: list[str], device) -> torch.Tensor:
        lengths = [len(t.split()) for t in texts]
        out = torch.zeros(len(texts), max(lengths), 12)
        for i, n in enumerate(lengths):
            out[i, :n] = self.embed(torch.arange(n) + 1)
        return out.to(device)


def _policy(**overrides) -> NoisePolicy:
    cfg = NoisePolicyConfig(image_size=28, dim_model=32, n_heads=4, dim_feedforward=64, **overrides)
    n_patches = (cfg.image_size // PATCH) ** 2
    return NoisePolicy(cfg, (H, A), image_backbone=_FakeBackbone(), image_dim=16,
                       text_encoder=_FakeText(), text_dim=12, n_patches=n_patches)


def _inputs(b: int = 3):
    images = torch.rand(b, 2, 3, 40, 40)
    texts = ["put the bowl in the drawer", "turn on the stove", "open"][:b]
    return images, texts, torch.randn(b, S)


@pytest.mark.parametrize("mode, k", [("full", H), ("time_shared", 1)])
def test_output_shapes_and_init_is_standard_normal(mode, k):
    policy = _policy(noise_mode=mode)
    dist = policy(*_inputs())
    assert dist.mean.shape == dist.log_std.shape == (3, k, A)
    # Untrained: exactly FastWAM's own N(0, I).
    assert torch.equal(dist.mean, torch.zeros_like(dist.mean))
    assert torch.equal(dist.log_std, torch.zeros_like(dist.log_std))
    noise = dist.sample(5)
    assert noise.shape == (3, 5, H, A)
    expected = -0.5 * k * A * math.log(2 * math.pi) - 0.5 * dist._free(noise).pow(2).sum(dim=(-2, -1))
    assert torch.allclose(dist.log_prob(noise), expected, atol=1e-4)


def test_time_shared_noise_is_replicated_over_time():
    dist = _policy(noise_mode="time_shared")(*_inputs())
    noise = dist.sample(4)
    assert torch.equal(noise, noise[:, :, :1].expand_as(noise))


def test_only_the_head_side_trains_and_encoders_stay_frozen():
    policy = _policy()
    policy.train()
    assert not policy.image_backbone.training and not policy.text_encoder.training
    assert all(not p.requires_grad for p in policy.image_backbone.parameters())
    assert all(not p.requires_grad for p in policy.text_encoder.parameters())

    # Move off the zero init so gradients reach every trainable layer, then check they do.
    nn.init.normal_(policy.head.weight, std=0.1)
    dist = policy(*_inputs())
    noise = dist.sample(2)
    (dist.log_prob(noise.detach()).sum() + noise.pow(2).sum()).backward()
    for name, p in policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name


def test_conditioning_changes_the_distribution():
    policy = _policy()
    nn.init.normal_(policy.head.weight, std=0.1)
    images, texts, state = _inputs(2)
    a = policy(images, texts, state)
    b = policy(images, texts, state + 1.0)
    assert not torch.allclose(a.mean, b.mean)


def test_log_std_is_clamped():
    policy = _policy(log_std_min=-1.0, log_std_max=0.5)
    nn.init.constant_(policy.head.bias[A:], 10.0)
    assert torch.all(policy(*_inputs()).log_std == 0.5)


def test_config_validation():
    with pytest.raises(ValueError):
        NoisePolicyConfig(noise_mode="per_step")
    with pytest.raises(ValueError):
        NoisePolicyConfig(dim_model=30, n_heads=4)
