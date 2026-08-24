"""Building evaluation environments from a run config.

A thin translation layer: it turns this package's ``BenchmarkConfig`` into the environment
config LeRobot's factory expects, and resolves ``benchmark.tasks`` into concrete tasks.

LIBERO and RoboTwin differ in an important practical way. A LIBERO suite is constructed in
one call and yields all ten of its tasks at once. RoboTwin tasks are separate SAPIEN
simulators, and holding dozens open exhausts host memory long before it exhausts the GPU, so
they are built one at a time and closed immediately -- hence the iterator interface rather
than a dictionary of live environments.
"""

from __future__ import annotations

from collections.abc import Iterator
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


def task_label(cfg: "RunConfig", task: str, task_id: int, env: Any) -> str:
    """A stable, human-readable name for a task, used in result files."""
    if cfg.benchmark.env_type == "robotwin":
        return task
    described = None
    inner = getattr(env, "envs", None)
    if inner:
        described = getattr(inner[0], "task_description", None)
    return libero_bench.task_name(task, task_id, fallback=described)
