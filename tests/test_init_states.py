"""LIBERO initial-state selection: which layout each episode starts from."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qplanning.config.loader import ConfigError, load
from qplanning.config.paths import repository_root
from qplanning.envs import init_state_schedule, num_init_states, set_init_state
from qplanning.self_improve import block_init_states

CONFIGS = repository_root() / "configs"


def _vector_env(n_states: int = 50):
    """The shape LeRobot's factory hands back: a vector env holding one LiberoEnv."""
    libero = SimpleNamespace(_init_states=list(range(n_states)), _init_state_id=0)
    libero.unwrapped = libero
    return SimpleNamespace(envs=[libero]), libero


# ── schedule ───────────────────────────────────────────────────────────────────────────
def test_default_schedule_gives_each_episode_its_own_state():
    assert init_state_schedule(5, None, 50) == [0, 1, 2, 3, 4]


def test_default_schedule_covers_all_states_then_wraps():
    states = init_state_schedule(55, None, 50)
    assert states[:50] == list(range(50))
    assert states[50:] == [0, 1, 2, 3, 4]


def test_explicit_ids_are_cycled_in_order():
    assert init_state_schedule(5, [30, 31], 50) == [30, 31, 30, 31, 30]


def test_single_id_reproduces_the_original_fixed_layout():
    assert init_state_schedule(20, [0], 50) == [0] * 20


def test_offset_and_exclude_shape_the_default_pool():
    assert init_state_schedule(3, None, 10, offset=1, exclude={0, 1, 2}) == [4, 5, 6]


def test_out_of_range_ids_are_rejected():
    with pytest.raises(ValueError, match="out of range"):
        init_state_schedule(1, [50], 50)


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="no initial states left"):
        init_state_schedule(1, None, 3, exclude={0, 1, 2})


# ── setting the state on the environment ───────────────────────────────────────────────
def test_set_init_state_targets_the_wrapped_libero_env():
    env, libero = _vector_env()
    assert num_init_states(env) == 50
    set_init_state(env, 37)
    assert libero._init_state_id == 37


def test_set_init_state_rejects_an_index_the_task_does_not_have():
    env, _ = _vector_env(n_states=10)
    with pytest.raises(ValueError, match="out of range"):
        set_init_state(env, 10)


def test_an_env_without_init_states_fails_loudly():
    with pytest.raises(RuntimeError, match="_init_states"):
        set_init_state(SimpleNamespace(envs=[SimpleNamespace()]), 0)


# ── self-improvement blocks ────────────────────────────────────────────────────────────
@pytest.fixture
def libero_cfg(monkeypatch: pytest.MonkeyPatch):
    for key in ("QPLANNING_BC_CKPT_LIBERO", "QPLANNING_Q_CKPT_LIBERO", "QPLANNING_WAN22_DIR"):
        monkeypatch.setenv(key, "/tmp/placeholder")
    return load(str(CONFIGS / "libero_10" / "self_improve.yaml"))


def test_default_blocks_are_disjoint_and_train_states_advance(libero_cfg):
    heldout, train0 = block_init_states(libero_cfg, 0, 50)
    _, train1 = block_init_states(libero_cfg, 1, 50)
    assert heldout == list(range(20))
    assert train0 == list(range(20, 30))
    assert train1 == list(range(30, 40))


def test_explicit_split_trains_on_the_other_states(libero_cfg):
    libero_cfg.self_improve.heldout_init_state_ids = list(range(30, 50))
    heldout, train = block_init_states(libero_cfg, 0, 50)
    assert set(heldout) == set(range(30, 50))
    assert set(train) <= set(range(30))


def test_overlapping_blocks_are_refused(libero_cfg):
    libero_cfg.self_improve.train_init_state_ids = [5]
    with pytest.raises(ConfigError, match="both the held-out and the training block"):
        block_init_states(libero_cfg, 0, 50)
