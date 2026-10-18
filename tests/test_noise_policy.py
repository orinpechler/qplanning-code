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
    free = noise[:, :, :k]                     # time-shared: the one draw, replicated over H
    expected = -0.5 * k * A * math.log(2 * math.pi) - 0.5 * free.pow(2).sum(dim=(-2, -1))
    assert torch.allclose(dist.log_prob(noise), expected, atol=1e-4)
    assert dist.n_free == k * A


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


# ── losses ──────────────────────────────────────────────────────────────────────────────
from noise_policy.losses import (  # noqa: E402
    effective_sample_size,
    importance_weighted_nll,
    importance_weights,
    reversal_nll,
)
from noise_policy.modeling import NoiseDistribution  # noqa: E402


def _dist(b: int = 2, k: int = H) -> NoiseDistribution:
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(b, k, A, generator=g, requires_grad=True)
    log_std = (0.1 * torch.randn(b, k, A, generator=g)).requires_grad_(True)
    return NoiseDistribution(mean=mean, log_std=log_std, horizon=H)


def test_importance_weights_are_a_tempered_softmax():
    q = torch.tensor([[0.20, 0.21, 0.19, 0.20]])
    w = importance_weights(q, temperature=0.01)
    assert torch.allclose(w.sum(-1), torch.ones(1))
    assert w[0].argmax() == 1 and w[0].argmin() == 2
    assert torch.allclose(w, torch.softmax(q / 0.01, -1))
    # A huge temperature is uniform: every sample counts.
    assert torch.allclose(effective_sample_size(importance_weights(q, 1e6)), torch.tensor([4.0]))
    # The proposal correction enters additively in log space.
    ratio = torch.log(torch.tensor([[1.0, 1.0, 4.0, 1.0]]))
    expected = torch.softmax(q / 0.01 + ratio, -1)
    assert torch.allclose(importance_weights(q, 0.01, ratio), expected)


def test_weighted_nll_with_uniform_weights_is_the_reversal_nll():
    dist, noise = _dist(), torch.randn(2, 5, H, A)
    uniform = torch.full((2, 5), 0.2)
    assert torch.allclose(importance_weighted_nll(dist, noise, uniform), reversal_nll(dist, noise))
    assert torch.allclose(reversal_nll(dist, noise), -dist.log_prob(noise).mean() / (H * A))


def test_weighted_nll_pulls_the_mean_towards_the_heavy_sample():
    dist = _dist(b=1)
    noise = torch.stack([torch.full((H, A), 2.0), torch.full((H, A), -2.0)])[None]
    loss = importance_weighted_nll(dist, noise, torch.tensor([[0.9, 0.1]]))
    loss.backward()
    # Gradient descent moves the mean towards +2 (where 90% of the weight is) wherever it is below.
    below = dist.mean.detach() < 1.6
    assert torch.all(dist.mean.grad[below] < 0)


def test_time_shared_log_prob_fits_all_rows_of_unreplicated_noise():
    dist = _dist(b=1, k=1)
    noise = torch.randn(1, 3, H, A)
    per_row = torch.distributions.Normal(dist.mean[:, None], dist.std[:, None]).log_prob(noise)
    assert torch.allclose(dist.log_prob(noise), per_row.mean(dim=-2).sum(dim=-1))


# ── flow reversal ──────────────────────────────────────────────────────────────────────
import functools  # noqa: E402

from qoptimizing.noise import FastWAMDenoiser  # noqa: E402


class _Scheduler:
    num_train_timesteps = 1000

    @staticmethod
    def step(model_output, delta, sample):
        return sample + model_output * delta


class _FakeFastWAM:
    """A smooth, state-dependent velocity field standing in for FastWAM's action expert."""

    device = torch.device("cpu")
    torch_dtype = torch.float64
    infer_action_scheduler = _Scheduler()

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(1)
        self.w = 0.3 * torch.randn(A, A, generator=g, dtype=torch.float64)
        self.target = torch.randn(H, A, generator=g, dtype=torch.float64)

    def _velocity(self, *, latents_action, timestep_action, **_):
        sigma = timestep_action.to(torch.float64) / 1000.0
        # A rectified-flow-like field that carries noise to `target`, plus a non-linear term.
        return (latents_action - self.target) + 0.2 * torch.tanh(latents_action @ self.w) * (1 + sigma)

    @functools.wraps(_velocity)
    def _predict_action_noise_with_cache(self, **kwargs):
        return self._velocity(**kwargs)


_FakeFastWAM._predict_action_noise_with_cache.__wrapped__ = (
    lambda self, **kwargs: self._velocity(**kwargs))


def _fake_denoiser(steps: int = 10) -> FastWAMDenoiser:
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    prepared = {
        "context": torch.zeros(1, 1, 1), "context_mask": torch.ones(1, 1, dtype=torch.bool),
        "video_kv_cache": [], "attention_mask": None, "video_seq_len": 0,
        "timesteps": sigmas[:-1] * 1000.0, "deltas": sigmas[1:] - sigmas[:-1],
    }
    return FastWAMDenoiser(model=_FakeFastWAM(), prepared=prepared, action_horizon=H)


def test_flow_reversal_round_trips_and_refinement_makes_it_exact():
    denoiser = _fake_denoiser()
    noise = torch.randn(4, H, A, dtype=torch.float64)
    chunks = denoiser(noise)

    def error(refine: int) -> float:
        return float((denoiser(denoiser.invert(chunks, refine_iters=refine)) - chunks).abs().max())

    plain, refined = error(0), error(20)
    assert plain < 0.5 * float(chunks.abs().max())       # backward Euler: close, not exact
    assert refined < 1e-5                                  # fixed point: exact up to float32
    assert error(2) < plain
    # And it recovers the noise itself, not just some noise with the same chunk.
    assert torch.allclose(denoiser.invert(chunks, refine_iters=20), noise.float(), atol=1e-6)


def test_transform_is_the_reparameterization_and_kl_is_zero_at_the_prior():
    dist = _dist(b=2)
    base = torch.randn(2, 4, H, A)
    noise = dist.transform(base)
    assert torch.allclose(noise, dist.mean[:, None] + dist.std[:, None] * base)
    prior = NoiseDistribution(mean=torch.zeros(1, H, A), log_std=torch.zeros(1, H, A), horizon=H)
    assert torch.equal(prior.transform(base[:1]), base[:1])
    assert torch.allclose(prior.kl_to_standard_normal(), torch.zeros(1))
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(dist.mean, dist.std), torch.distributions.Normal(0.0, 1.0)).sum(dim=(-2, -1))
    assert torch.allclose(dist.kl_to_standard_normal(), expected, atol=1e-5)


def test_policy_stats_separate_global_and_state_dependent_shifts():
    from noise_policy.training import _policy_stats

    shared = NoiseDistribution(mean=torch.full((4, H, A), 0.3), log_std=torch.zeros(4, H, A), horizon=H)
    stats = _policy_stats(shared)
    assert abs(stats["policy/mu_global_rms"] - 0.3) < 1e-6 and stats["policy/mu_across_states_std"] == 0.0
    per_state = NoiseDistribution(mean=torch.tensor([0.3, -0.3, 0.3, -0.3]).view(4, 1, 1).expand(4, H, A).clone(),
                                  log_std=torch.zeros(4, H, A), horizon=H)
    stats = _policy_stats(per_state)
    assert stats["policy/mu_global_rms"] < 1e-6 and stats["policy/mu_across_states_std"] > 0.3


# ── tilt probe ─────────────────────────────────────────────────────────────────────────
from noise_policy.tilt_probe import consistency, tilt_direction  # noqa: E402


def _tilt_setup(signal: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(256, H, A, generator=g)
    direction = torch.randn(H * A, generator=g)
    direction /= direction.norm()
    shifted = direction + 0.3 * torch.randn(H * A, generator=g) / (H * A) ** 0.5   # a nearby state
    qs = {}
    for sid, d in (("a", direction), ("b", shifted), ("c", torch.randn(H * A, generator=g))):
        qs[sid] = signal * 0.005 * (noise.flatten(1) @ (d / d.norm())) + 0.005 * torch.randn(256, generator=g)
    pairs = {"near": [("a", "b")], "far": [("a", "c")]}
    return noise, qs, pairs


def test_a_consistent_preference_is_found_and_noise_is_not():
    noise, qs, pairs = _tilt_setup(signal=2.0)
    res = consistency(noise, qs, pairs, tau=float("inf"), generator=torch.Generator().manual_seed(1))
    # A 224-dim direction from 128 noises per half is a noisy estimate: clearly above chance, not near 1.
    assert res["within"] > 0.15 and abs(res["within_chance"]) < 0.1
    assert res["near"] > 0.5 and abs(res["far"]) < 0.3
    noise, qs, pairs = _tilt_setup(signal=0.0)
    res = consistency(noise, qs, pairs, tau=float("inf"), generator=torch.Generator().manual_seed(1))
    assert abs(res["within"]) < 0.2 and abs(res["near"]) < 0.3


def test_tilt_direction_is_the_weighted_minus_plain_mean():
    noise = torch.randn(8, H, A)
    q = torch.zeros(8)
    assert torch.allclose(tilt_direction(noise, q, 0.01), torch.zeros(H * A, dtype=torch.float64), atol=1e-12)
    q[3] = 10.0           # one sample takes all the weight at low tau
    expected = noise[3].flatten().double() - noise.flatten(1).double().mean(0)
    assert torch.allclose(tilt_direction(noise, q, 0.01), expected, atol=1e-8)
