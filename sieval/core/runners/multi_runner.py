"""Parallel orchestrator for running multiple TaskRunners with shared concurrency."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self

import anyio
from loguru import logger

from sieval.core.tasks.concurrency import prepare_limiters
from sieval.core.tasks.context import TaskAction
from sieval.core.tasks.run_meta import warn_unmatched_judge_refs
from sieval.core.tasks.task import Task
from sieval.core.types import JSONValue

from .runner import TaskRunner, TaskRunnerConfig
from .stage_resume import LegacyStageMatch, RunPlan


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
        effective_config: str | None = None,
    ):
        """Initialize the multi-task runner.

        *effective_config* is recorded in each task's ``meta.json`` unless
        the task's config sets its own.

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
        self._effective_config = effective_config
        self._runners: list[TaskRunner] = []
        # Track registered names and directories to prevent collisions
        self._task_names: set[str] = set()
        self._output_dirs: set[Path] = set()

    def add_task(
        self,
        task: Task,
        config: TaskRunnerConfig | None = None,
        *,
        inference_recipe: dict[str, JSONValue] | None = None,
        context_limit: int | None = None,
        weights: str | None = None,
        stated_serving: dict[str, JSONValue] | None = None,
        resume_config_diff: list[str] | None = None,
        legacy_match: LegacyStageMatch | None = None,
        unserved_candidate: bool = False,
        serving_group: str | None = None,
    ) -> Self:
        """Register a task for parallel execution.

        *inference_recipe*, *context_limit*, *weights*, *stated_serving*,
        *serving_group*, *resume_config_diff*, *legacy_match* and
        *unserved_candidate* pass through to :class:`TaskRunner`.

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
        if self._effective_config and cfg.effective_config is None:
            cfg = replace(cfg, effective_config=self._effective_config)
        # Monotone force-on: session True overrides per-task; session
        # False leaves per-task untouched (per-task True still wins).
        if self._deterministic:
            cfg = replace(cfg, deterministic=True)

        runner = TaskRunner(
            task,
            config=cfg,
            inference_recipe=inference_recipe,
            context_limit=context_limit,
            weights=weights,
            stated_serving=stated_serving,
            resume_config_diff=resume_config_diff,
            legacy_match=legacy_match,
            unserved_candidate=unserved_candidate,
            serving_group=serving_group,
        )

        if runner.root_dir in self._output_dirs:
            raise ValueError(
                f"Output directory collision for task '{task.name}': {runner.root_dir}. "  # noqa: E501
                "Another task is already using this directory."
            )

        self._runners.append(runner)
        self._task_names.add(task.name)
        self._output_dirs.add(runner.root_dir)
        return self

    async def plan_run(self) -> dict[str, RunPlan | Exception]:
        """Each task's :meth:`TaskRunner.plan_run`, by task name; read-only.

        A task whose planning raises maps to the exception, which its run
        would raise before its first sample. Runners that planned are not run.
        """
        plans: dict[str, RunPlan | Exception] = {}
        for runner in self._runners:
            try:
                plans[runner._task.name] = await runner.plan_run()
            except Exception as exc:
                plans[runner._task.name] = exc
        return plans

    def run(self) -> dict[str, Any]:
        """Synchronous wrapper around :meth:`arun`."""
        return anyio.run(self.arun)

    async def arun(self) -> dict[str, Any]:
        """Run all registered tasks concurrently with shared limiters.

        Raises:
            ExceptionGroup: If any task raised.  A failing task does not cancel
                its siblings, which run to completion and write their own
                ``report.json``; the message names the tasks that failed.
                Nothing is returned in this case, so the successful tasks'
                results are on disk only.
        """
        # 0. A judge reference for a role no task has is never recorded.
        roles: set[str] = set()
        for runner in self._runners:
            roles |= runner.judge_roles()
        warn_unmatched_judge_refs(roles)

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
        failures: dict[str, Exception] = {}

        async def _run_wrapper(r: TaskRunner) -> None:
            name = r._task.name
            try:
                results[name] = await r.arun()
            except Exception as exc:
                # Hold, do not propagate: raising into the task group cancels every
                # sibling mid-flight, and a cancelled runner flushes its shards but
                # never writes `report.json`.
                #
                # `Exception`, deliberately not `BaseException` -- cancellation and
                # `KeyboardInterrupt` are BaseException-only, so an interrupt still
                # tears the batch down through `TaskRunner.arun`'s own save path.
                #
                # Logged with its traceback here because the re-raise below is now
                # as far away as the rest of the batch is long.
                logger.opt(exception=exc).error("Task '{}' failed: {!r}", name, exc)
                failures[name] = exc

        async with anyio.create_task_group() as tg:
            for runner in self._runners:
                tg.start_soon(_run_wrapper, runner)

        # Isolated, not swallowed: returning a partial dict would let the caller's
        # exit code read success while a task is missing from it. `ExceptionGroup`
        # is the shape an uncaught failure already had, so callers keep working.
        if failures:
            # The survivors appear in neither the group nor a return value, so
            # their only other trace is a `report.json` you would have to know to
            # look for. Guarded, because "0 task(s) completed" reads as reassurance.
            if results:
                logger.info(
                    "{} task(s) completed and saved their reports: {}",
                    len(results),
                    ", ".join(sorted(results)),
                )
            raise ExceptionGroup(
                f"{len(failures)} of {len(self._runners)} task(s) failed: "
                + ", ".join(sorted(failures)),
                [failures[name] for name in sorted(failures)],
            )

        return results
