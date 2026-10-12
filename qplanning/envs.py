"""Building evaluation environments from a run config.

A thin translation layer: it turns this package's ``BenchmarkConfig`` into the environment
config LeRobot's factory expects, and resolves ``benchmark.tasks`` into concrete tasks.

LIBERO and RoboTwin differ in an important practical way. A LIBERO suite is constructed in
one call and yields all ten of its tasks at once. RoboTwin tasks are separate SAPIEN
simulators, and holding dozens open exhausts host memory long before it exhausts the GPU, so
they are built one at a time and closed immediately -- hence the iterator interface rather
than a dictionary of live environments.

LIBERO also pins each episode's starting layout. Every task ships a fixed set of initial
states (50 per task), and LeRobot's wrapper restores ``init_states[episode_index]`` on every
reset, overwriting whatever the reset seed produced. Because one environment is built per
task and reset for every episode, the episode index is always 0 -- so without
:func:`set_init_state`, every episode of a task starts from the same layout and only the
policy's sampling noise varies between them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from qplanning.benchmarks import libero as libero_bench
from qplanning.benchmarks import robotwin as robotwin_bench

if TYPE_CHECKING:
    from qplanning.config.schema import RunConfig


def resolve_tasks(cfg: "RunConfig") -> list[str]:
    """The task names this run will evaluate, in a stable order."""
    if cfg.benchmark.env_type == "robotwin":
        return robotwin_bench.resolve_tasks(cfg.benchmark.tasks)
    suite = cfg.benchmark.suite or cfg.benchmark.name
    if suite not in libero_bench.SUITES:
        raise ValueError(f"unknown LIBERO suite {suite!r}; expected one of {libero_bench.SUITES}")
    return [suite]


def apply_shard(tasks: list[str], shard: str | None) -> list[str]:
    """Select this job's slice of the task list, as ``"k/n"``.

    Sliced by stride rather than by contiguous block so that every shard gets a similar mix
    of easy and slow tasks; a contiguous split leaves one job holding all the long ones.
    """
    if not shard:
        return tasks
    try:
        index, count = (int(part) for part in shard.split("/"))
    except ValueError:
        raise ValueError(f"shard must look like 'k/n', got {shard!r}") from None
    if not 0 <= index < count:
        raise ValueError(f"shard index must satisfy 0 <= k < n, got {shard!r}")
    selected = [task for i, task in enumerate(tasks) if i % count == index]
    if not selected:
        raise ValueError(
            f"shard {shard} selects none of the {len(tasks)} task group(s) available. "
            "A LIBERO suite is a single group whose tasks are built together, so it cannot "
            "be split this way -- shard RoboTwin, or run LIBERO unsharded."
        )
    return selected


def build_env_config(cfg: "RunConfig", task: str) -> Any:
    """Construct the LeRobot environment config for one task."""
    if cfg.benchmark.env_type == "libero":
        from lerobot.envs.configs import LiberoEnv

        return LiberoEnv(
            task=task,
            episode_length=cfg.benchmark.episode_length,
            observation_height=cfg.benchmark.observation_height,
            observation_width=cfg.benchmark.observation_width,
        )

    from lerobot.envs.configs import RoboTwinEnv

    from qplanning.config.paths import PathResolver

    robotwin_root = PathResolver().require(cfg.paths.robotwin_root, "paths.robotwin_root")
    return RoboTwinEnv(
        task=task,
        robotwin_root=str(robotwin_root),
        episode_length=cfg.benchmark.episode_length,
    )


@contextmanager
def open_task_envs(cfg: "RunConfig", task: str) -> Iterator[tuple[Any, dict[int, Any]]]:
    """Yield ``(env_config, {task_id: vector_env})`` for one task, closing them afterwards."""
    from lerobot.envs.factory import make_env

    env_cfg = build_env_config(cfg, task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    # make_env groups by suite; for a single task there is exactly one group.
    per_task = next(iter(envs.values()))
    try:
        yield env_cfg, per_task
    finally:
        for env in per_task.values():
            try:
                env.close()
            except Exception:  # noqa: BLE001 - a close failure must not mask a real error
                pass


def _libero_env(env: Any) -> Any:
    """The single LeRobot ``LiberoEnv`` inside a one-env vector env."""
    inner = getattr(env, "envs", None)
    target = inner[0] if inner else env
    target = getattr(target, "unwrapped", target)
    if not hasattr(target, "_init_states") or not hasattr(target, "_init_state_id"):
        raise RuntimeError(
            f"{type(target).__name__} exposes no `_init_states` / `_init_state_id`; the pinned "
            "LeRobot fork's LiberoEnv has changed, so the initial state cannot be selected."
        )
    if target._init_states is None:
        raise RuntimeError("LiberoEnv was built with init_states=False, so there is nothing to select")
    return target


def num_init_states(env: Any) -> int:
    """How many fixed initial states this LIBERO task ships (50 for every standard task)."""
    return len(_libero_env(env)._init_states)


def set_init_state(env: Any, index: int) -> None:
    """Make the next ``reset()`` restore LIBERO initial state ``index``."""
    target = _libero_env(env)
    available = len(target._init_states)
    if not 0 <= index < available:
        raise ValueError(f"initial state {index} out of range; this task has {available}")
    target._init_state_id = index


def init_state_schedule(
    n_episodes: int,
    ids: Sequence[int] | None,
    available: int,
    *,
    offset: int = 0,
    exclude: Collection[int] = (),
) -> list[int]:
    """The initial state for each of ``n_episodes`` episodes.

    Episode ``e`` takes entry ``offset + e`` of a pool, wrapping around once the pool is
    exhausted. The pool is ``ids`` when given, otherwise every state not in ``exclude`` --
    so with defaults, episodes 0..49 cover a task's 50 layouts once each, which is the
    standard LIBERO protocol.
    """
    if ids is not None:
        pool = list(ids)
        out_of_range = [i for i in pool if not 0 <= i < available]
        if out_of_range:
            raise ValueError(f"initial state ids {out_of_range} out of range; tasks have {available}")
    else:
        pool = [i for i in range(available) if i not in set(exclude)]
    if not pool and n_episodes > 0:
        raise ValueError(f"no initial states left to draw from ({available} available, "
                         f"{len(set(exclude))} excluded)")
    return [pool[(offset + e) % len(pool)] for e in range(n_episodes)]


def task_label(cfg: "RunConfig", task: str, task_id: int, env: Any) -> str:
    """A stable, human-readable name for a task, used in result files."""
    if cfg.benchmark.env_type == "robotwin":
        return task
    described = None
    inner = getattr(env, "envs", None)
    if inner:
        described = getattr(inner[0], "task_description", None)
    return libero_bench.task_name(task, task_id, fallback=described)
