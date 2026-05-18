"""Parallel orchestrator for running multiple TaskRunners with shared concurrency."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self

import anyio

from sieval.core.tasks.concurrency import prepare_limiters
from sieval.core.tasks.context import TaskAction
from sieval.core.tasks.task import Task

from .runner import TaskRunner, TaskRunnerConfig


class MultiTaskRunner:
    """Parallel task execution orchestrator.

    Runs multiple :class:`TaskRunner` instances concurrently with shared
    hierarchical concurrency limiters (global and per-stage).  Validates that
    no two registered tasks share the same name or output directory.
    """

    def __init__(
        self,
        result_dir: str | None = None,
        concurrency_limit: int | None = None,
        concurrency_limits: (
            dict[
                TaskAction | Literal["preprocess", "infer", "postprocess", "feedback"],
                int,
            ]
            | None
        ) = None,
        deterministic: bool = False,
    ):
        """Initialize the multi-task runner.

        When *result_dir* is set, tasks without their own ``result_dir`` will
        write to ``<result_dir>/<task.name>``.

        *deterministic* is session-wide metadata that every child
        :class:`TaskRunner` inherits (recorded in ``meta.json``). Engine-level
        determinism itself is enforced by the backend translators.
        """
        self._base_result_dir = result_dir
        self._global_limit = concurrency_limit
        self._stage_limits_cfg = concurrency_limits
        self._deterministic = deterministic
        self._runners: list[TaskRunner] = []
        # Track registered names and directories to prevent collisions
        self._task_names: set[str] = set()
        self._output_dirs: set[Path] = set()

    def add_task(self, task: Task, config: TaskRunnerConfig | None = None) -> Self:
        """Register a task for parallel execution.

        Auto-sets ``result_dir`` to ``<base_result_dir>/<task.name>`` when
        a base was provided and the task's config lacks its own.

        Raises:
            ValueError: If *task.name* or resolved output directory collides
                with an already-registered task.
        """
        if task.name in self._task_names:
            raise ValueError(
                f"Task '{task.name}' is already registered in this runner."
            )

        cfg = config or TaskRunnerConfig()
        if self._base_result_dir and cfg.result_dir is None:
            cfg = replace(cfg, result_dir=str(Path(self._base_result_dir) / task.name))
        # Monotone force-on: session True overrides per-task; session
        # False leaves per-task untouched (per-task True still wins).
        if self._deterministic:
            cfg = replace(cfg, deterministic=True)

        runner = TaskRunner(task, config=cfg)

        if runner.root_dir in self._output_dirs:
            raise ValueError(
                f"Output directory collision for task '{task.name}': {runner.root_dir}. "  # noqa: E501
                "Another task is already using this directory."
            )

        self._runners.append(runner)
        self._task_names.add(task.name)
        self._output_dirs.add(runner.root_dir)
        return self

    def run(self) -> dict[str, Any]:
        """Synchronous wrapper around :meth:`arun`."""
        return anyio.run(self.arun)

    async def arun(self) -> dict[str, Any]:
        """Run all registered tasks concurrently with shared limiters."""
        # 1. Prepare Shared Limiters
        norm_stage_limits = {}
        if self._stage_limits_cfg:
            for k, v in self._stage_limits_cfg.items():
                key = k.value if isinstance(k, TaskAction) else k
                if key in {a.value for a in TaskAction}:
                    norm_stage_limits[key] = v

        global_limiter, stage_limiters = prepare_limiters(
            self._global_limit, norm_stage_limits
        )

        # 2. Configure Runners
        for i, runner in enumerate(self._runners):
            runner.set_runtime_context(
                shared_global_limiter=global_limiter,
                shared_stage_limiters=stage_limiters,
                progress_position=i,
                shared_global_limit=self._global_limit,
                shared_stage_limits=norm_stage_limits,
            )

        # 3. Run in Parallel
        results: dict[str, Any] = {}

        async def _run_wrapper(r: TaskRunner) -> None:
            res = await r.arun()
            results[r._task.name] = res

        async with anyio.create_task_group() as tg:
            for runner in self._runners:
                tg.start_soon(_run_wrapper, runner)

        return results
