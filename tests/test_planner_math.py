"""The planner must be numerically identical to the reference implementation.

The planner was rewritten for this package -- dropping visualisation hooks, gripper flipping
and the noise-based variants -- so the arithmetic that survives has to be shown to be
unchanged. Each test below compares against a literal transcription of the original
``plan_chunk_fastwam`` code, so a future edit that alters the result fails here rather than
silently shifting a reported number.

These run on CPU with stubs. The end-to-end check against real checkpoints is a GPU probe.
"""

from __future__ import annotations

import pytest
import torch

from qplanning.config.schema import PlannerConfig
from qplanning.planner import NoPlanner, QSpread, QWeightedPlanner, _broadcast
from qplanning.policy import QPlanningPolicy
from qplanning.policies.base import BCBundle


# ---------------------------------------------------------------------------------------
# Literal transcriptions of the original, for comparison
# ---------------------------------------------------------------------------------------
def upstream_aggregate(candidates, q_values, n_elites, temperature):
    """From plan_chunk_fastwam, the bc_diffusion_mppi branch."""
    N = candidates.shape[0]
    K = min(n_elites, N) if n_elites > 0 else N
    topk_idx = torch.topk(q_values, K).indices
    topk_q = q_values[topk_idx]
    topk_cands = candidates[topk_idx]
    weights = torch.softmax((topk_q - topk_q.max()) / temperature, dim=0)
    return (weights.view(K, 1, 1) * topk_cands).sum(dim=0, keepdim=True)


def upstream_score(candidates_norm, precomputed_context, q_policy, q_pre, bc_post, horizon, img_feats):
    """From _score_candidates_fast."""
    from lerobot.policies.q_function.modeling_q_function import _expected_value
    from lerobot.utils.constants import ACTION

    N, h, A = candidates_norm.shape
    flat_norm = candidates_norm.reshape(N * h, A)
    flat_raw = bc_post(flat_norm)
    candidates_raw = flat_raw.reshape(N, h, A).to(candidates_norm.device)
    q_batch = {ACTION: candidates_raw}
    for key, feat in img_feats.items():
        if isinstance(feat, torch.Tensor):
            q_batch[key] = feat.expand(N, *feat.shape[1:]).contiguous()
        elif isinstance(feat, (list, tuple)):
            q_batch[key] = list(feat[:1]) * N if len(feat) == 1 else list(feat) * N
        else:
            q_batch[key] = feat
    q_batch = q_pre(q_batch)
    actions = q_policy._truncate_action(q_batch[ACTION][:, :horizon, :])
    context_N = precomputed_context.expand(-1, N, -1).contiguous()
    logits = q_policy.q_online.forward_with_context(context_N, actions)
    return _expected_value(logits, q_policy.bin_centers)


# ---------------------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------------------
class StubQNet:
    """A deterministic stand-in for the Q decoder: logits from a fixed linear map."""

    CONTEXT_DIM = 16

    def __init__(self, horizon: int, action_dim: int, num_bins: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.w_action = torch.randn(horizon * action_dim, num_bins, generator=g)
        self.w_context = torch.randn(self.CONTEXT_DIM, num_bins, generator=g)

    def forward_with_context(self, context, actions):
        # context: (S, N, D) -> the planner must have expanded it across candidates.
        n = actions.shape[0]
        assert context.shape[1] == n, f"context not expanded to N: {context.shape} vs N={n}"
        flat = actions.reshape(n, -1)
        ctx_summary = context.mean(dim=0)          # (N, CONTEXT_DIM)
        return flat @ self.w_action + ctx_summary @ self.w_context


class StubQPolicy:
    def __init__(self, horizon=4, action_dim=3, num_bins=11, seed=0):
        self.q_online = StubQNet(horizon, action_dim, num_bins, seed)
        self.bin_centers = torch.linspace(-0.01, 1.01, num_bins)
        self.horizon = horizon

        class _Cfg:
            h = horizon
            camera_keys = ("observation.images.image",)

        self.config = _Cfg()

    def _truncate_action(self, actions):
        return actions

    def encode_obs_context(self, batch):
        """Return (S, B, D) observation tokens, as the real Q-function does."""
        img = batch["observation.images.image"]
        return img.reshape(1, img.shape[0], -1).expand(6, img.shape[0], img.shape[-1]).contiguous()


def identity_pre(batch):
    return batch


def scale_post(flat):
    """Stand-in for the BC unnormalizer: a fixed affine map."""
    return flat * 2.5 - 0.75


class DummyBC:
    """A minimal BCChunkSampler, to prove the protocol is implementable without FastWAM."""

    def __init__(self, chunk_size=4, action_dim=3, seed=0):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self._g = torch.Generator().manual_seed(seed)
        self.reset_calls = 0
        self.sample_calls = 0

    def reset(self):
        self.reset_calls += 1

    def sample_chunks(self, batch, n_samples, *, denoise_steps=None, generator=None):
        self.sample_calls += 1
        self.last_denoise_steps = denoise_steps
        return torch.randn(n_samples, self.chunk_size, self.action_dim, generator=self._g)

    def unnormalize_actions(self, actions):
        return scale_post(actions)

    def to(self, device):
        return self

    def parameters(self):
        return iter(())


# ---------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_samples,n_elites,temperature", [
    (64, 16, 1.0),      # LIBERO's deployed setting
    (32, 8, 1.0),       # RoboTwin's deployed setting
    (64, 0, 1.0),       # 0 means aggregate over all N
    (16, 16, 1.0),      # K == N
    (8, 4, 0.1),        # sharp weighting
    (8, 4, 10.0),       # near-uniform weighting
    (1, 1, 1.0),        # degenerate single candidate
])
def test_aggregation_matches_upstream(n_samples, n_elites, temperature):
    torch.manual_seed(0)
    candidates = torch.randn(n_samples, 4, 3)
    q_values = torch.randn(n_samples)

    cfg = PlannerConfig(n_samples=n_samples, n_elites=n_elites, temperature=temperature)
    planner = QWeightedPlanner(cfg=cfg, ctx=None, bc_unnormalize=None)

    torch.testing.assert_close(
        planner._aggregate(candidates, q_values),
        upstream_aggregate(candidates, q_values, n_elites, temperature),
        rtol=0, atol=0,
    )


def test_aggregation_is_a_convex_combination_of_the_top_k():
    """The executed chunk must lie inside the convex hull of the candidates it averages."""
    torch.manual_seed(1)
    candidates = torch.randn(32, 4, 3)
    q_values = torch.randn(32)
    cfg = PlannerConfig(n_samples=32, n_elites=8, temperature=1.0)
    out = QWeightedPlanner(cfg=cfg, ctx=None, bc_unnormalize=None)._aggregate(candidates, q_values)

    top = candidates[torch.topk(q_values, 8).indices]
    assert out.shape == (1, 4, 3)
    assert (out <= top.max(dim=0).values + 1e-6).all()
    assert (out >= top.min(dim=0).values - 1e-6).all()


def test_low_temperature_approaches_argmax():
    torch.manual_seed(2)
    candidates = torch.randn(16, 4, 3)
    q_values = torch.randn(16)
    cfg = PlannerConfig(n_samples=16, n_elites=16, temperature=1e-4)
    out = QWeightedPlanner(cfg=cfg, ctx=None, bc_unnormalize=None)._aggregate(candidates, q_values)
    torch.testing.assert_close(out[0], candidates[int(q_values.argmax())], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_samples", [1, 8, 64])
def test_scoring_matches_upstream(n_samples):
    """The full unnormalize -> renormalize -> single-decoder-pass path."""
    torch.manual_seed(3)
    horizon, action_dim = 4, 3
    q_policy = StubQPolicy(horizon=horizon, action_dim=action_dim)
    candidates = torch.randn(n_samples, horizon, action_dim)
    observation = {"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM), "task": ["pick up the mug"]}
    context = q_policy.encode_obs_context({"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)})

    from qplanning.planner import PlannerContext

    planner = QWeightedPlanner(
        cfg=PlannerConfig(n_samples=n_samples, n_elites=4),
        ctx=PlannerContext(q_policy=q_policy, q_pre=identity_pre,
                           camera_keys=("observation.images.image",), horizon=horizon),
        bc_unnormalize=scale_post,
    )
    torch.testing.assert_close(
        planner._score(candidates, context, observation),
        upstream_score(candidates, context, q_policy, identity_pre, scale_post, horizon, observation),
        rtol=0, atol=0,
    )


def test_scoring_requires_the_unnormalization_round_trip():
    """Scoring BC-normalized actions directly must give different values than the real path.

    This is the mistake the round-trip exists to prevent, so it is worth pinning: if these
    ever agree, the conversion has silently become a no-op.
    """
    torch.manual_seed(4)
    horizon, action_dim = 4, 3
    q_policy = StubQPolicy(horizon=horizon, action_dim=action_dim)
    candidates = torch.randn(8, horizon, action_dim)
    observation = {"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)}
    context = q_policy.encode_obs_context(observation)

    from qplanning.planner import PlannerContext

    ctx = PlannerContext(q_policy=q_policy, q_pre=identity_pre,
                         camera_keys=("observation.images.image",), horizon=horizon)
    with_roundtrip = QWeightedPlanner(PlannerConfig(n_samples=8), ctx, scale_post)
    without = QWeightedPlanner(PlannerConfig(n_samples=8), ctx, lambda x: x)
    assert not torch.allclose(
        with_roundtrip._score(candidates, context, observation),
        without._score(candidates, context, observation),
    )


def test_observation_is_encoded_once_per_step():
    """The encoder cost must not scale with N; that is what keeps planning affordable."""
    torch.manual_seed(5)
    q_policy = StubQPolicy()
    calls = {"n": 0}
    original = q_policy.encode_obs_context

    def counting(batch):
        calls["n"] += 1
        return original(batch)

    q_policy.encode_obs_context = counting

    from qplanning.planner import PlannerContext

    planner = QWeightedPlanner(
        cfg=PlannerConfig(n_samples=64, n_elites=16),
        ctx=PlannerContext(q_policy=q_policy, q_pre=identity_pre,
                           camera_keys=("observation.images.image",), horizon=4),
        bc_unnormalize=scale_post,
    )
    bc = DummyBC()
    planner.plan(bc, {"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)})
    assert calls["n"] == 1, f"encoders ran {calls['n']} times for 64 candidates"


def test_missing_camera_is_reported_by_name():
    from qplanning.planner import PlannerContext

    planner = QWeightedPlanner(
        cfg=PlannerConfig(),
        ctx=PlannerContext(q_policy=StubQPolicy(), q_pre=identity_pre,
                           camera_keys=("observation.images.wrist",), horizon=4),
        bc_unnormalize=scale_post,
    )
    with pytest.raises(KeyError, match="wrist"):
        planner._observation_inputs({"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)})


def test_broadcast_handles_tensors_and_task_strings():
    assert _broadcast(torch.zeros(1, 3), 4).shape == (4, 3)
    assert _broadcast(["a"], 3) == ["a", "a", "a"]
    assert _broadcast(7, 3) == 7


# ---------------------------------------------------------------------------------------
# The rollout policy
# ---------------------------------------------------------------------------------------
def _bundle(bc: DummyBC, n_action_steps: int = 2) -> BCBundle:
    return BCBundle(sampler=bc, preprocessor=identity_pre, postprocessor=identity_pre,
                    policy_config=None, n_action_steps=n_action_steps)


def test_queue_replans_at_the_configured_cadence():
    """One plan per n_action_steps environment steps -- the replanning budget."""
    bc = DummyBC(chunk_size=8)
    policy = QPlanningPolicy(_bundle(bc, n_action_steps=4), NoPlanner(), n_action_steps=4)
    policy.reset()
    for _ in range(12):
        action = policy.select_action({"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)})
        assert action.shape == (1, 3)
    assert bc.sample_calls == 3, f"expected 12/4 = 3 plans, got {bc.sample_calls}"


def test_reset_discards_unexecuted_actions():
    bc = DummyBC(chunk_size=8)
    policy = QPlanningPolicy(_bundle(bc, n_action_steps=4), NoPlanner(), n_action_steps=4)
    policy.reset()
    policy.select_action({"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)})
    assert len(policy._queue) == 3
    policy.reset()
    assert len(policy._queue) == 0 and bc.reset_calls == 2


def test_batched_observation_is_refused_with_a_usable_message():
    bc = DummyBC()
    policy = QPlanningPolicy(_bundle(bc), NoPlanner())
    policy.reset()
    with pytest.raises(NotImplementedError, match="single-environment"):
        policy.select_action({"observation.images.image": torch.randn(4, 6)})


def test_batched_observation_is_caught_behind_a_non_tensor_key():
    """The guard must scan every entry, not stop at the first one.

    A batch's first key is often ``task`` (a list of strings). A guard that inspected only
    the first entry would wave through a batched observation whenever that happened.
    """
    bc = DummyBC()
    policy = QPlanningPolicy(_bundle(bc), NoPlanner())
    policy.reset()
    with pytest.raises(NotImplementedError, match="single-environment"):
        policy.select_action({
            "task": ["pick up the mug"],
            "observation.images.image": torch.randn(4, 6),
        })


def test_n_action_steps_cannot_exceed_the_chunk():
    bc = DummyBC(chunk_size=4)
    with pytest.raises(ValueError, match="chunk"):
        QPlanningPolicy(_bundle(bc, n_action_steps=2), NoPlanner(), n_action_steps=8)


def test_baseline_uses_one_sample_and_the_full_denoising_budget():
    """Guided and unguided runs must differ only in selection, not in plumbing."""
    bc = DummyBC(chunk_size=8)
    policy = QPlanningPolicy(_bundle(bc, 4), NoPlanner(denoise_steps=10), n_action_steps=4)
    policy.reset()
    policy.select_action({"observation.images.image": torch.randn(1, StubQNet.CONTEXT_DIM)})
    assert bc.last_denoise_steps == 10


def test_dummy_bc_satisfies_the_protocol():
    from qplanning.policies.base import BCChunkSampler

    assert isinstance(DummyBC(), BCChunkSampler)


def test_q_spread_flags_a_degenerate_planner():
    assert QSpread(0.5, 0.5, 0.5, 0.0).is_degenerate
    assert not QSpread(0.1, 0.9, 0.5, 0.2).is_degenerate
