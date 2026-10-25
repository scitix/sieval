"""Tests for serving reconciliation: context needs, windows, serving groups.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from sieval.cli.serving import (
    DEFAULT_GROUP,
    ContextNeed,
    ServedWindow,
    ServingShortfallError,
    check_context_fits,
    parse_model_context,
    plan_serving_groups,
    plan_window,
    task_context_needs,
    with_window,
)
from sieval.core.models.requirements import TaskRequirements
from sieval.core.tasks import Task
from sieval.infer.topology import (
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    RoleAssignment,
    WellKnownRole,
)

TASK = "GSM8KZeroShotGenTask"
LONG = "tests.unit.cli.test_serving._LongContextTask"


class _LongContextTask(Task):
    """Declares the window it needs, as RULER-style tasks do."""

    requires: ClassVar[TaskRequirements] = TaskRequirements(min_context=50000)


def _plan(
    window: int | None = 32768,
    *,
    backend: str = "sglang",
    checkpoint: str = "/models/m",
) -> DeploymentPlan:
    params: dict[str, Any] = {}
    if window is not None:
        params["context_length" if backend == "sglang" else "max_model_len"] = window
    return DeploymentPlan(
        checkpoint=checkpoint,
        backend=backend,
        assignments=(
            RoleAssignment(
                role=WellKnownRole.FULL,
                devices=DeviceGroup(count=1),
                topology=ParallelTopology(tp=1),
                engine_params=params,
            ),
        ),
    )


def _config(
    tasks: dict[str, dict[str, Any]],
    *,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "models": {"m": model if model is not None else {"path": "/models/m"}},
        "datasets": {"d": {"class": "GSM8KDataset"}},
        "tasks": {
            name: {"class": TASK, "dataset": "d", **task}
            for name, task in tasks.items()
        },
    }


def _gen(max_tokens: int) -> dict[str, Any]:
    return {"infer_args": {"max_tokens": max_tokens}}


class TestTaskContextNeeds:
    def test_need_is_max_tokens_from_infer_args_then_model_args(self) -> None:
        """Plus PROMPT_HEADROOM: a window of exactly max_tokens fits no prompt."""
        config = _config(
            {"short": {}, "long": _gen(38000)},
            model={"path": "/models/m", "args": {"max_tokens": 4096}},
        )

        assert task_context_needs(config) == {
            "short": ContextNeed("m", 8192, "max_tokens"),
            "long": ContextNeed("m", 42096, "max_tokens"),
        }

    def test_derived_model_inherits_max_tokens_from_its_base(self) -> None:
        config = _config({"t": {"model": "child"}})
        config["models"]["m"]["args"] = {"max_completion_tokens": 2048}
        config["models"]["child"] = {"base": "m"}

        assert task_context_needs(config) == {"t": ContextNeed("m", 6144, "max_tokens")}

    def test_declared_need_wins_when_larger_and_override_wins_always(self) -> None:
        config = _config(
            {
                "declared": {"class": LONG, **_gen(1024)},
                "max_tokens": {"class": LONG, **_gen(60000)},
                "override": {"class": LONG, "min_context": 8192},
            }
        )

        assert task_context_needs(config) == {
            "declared": ContextNeed("m", 50000, "requires"),
            "max_tokens": ContextNeed("m", 64096, "max_tokens"),
            "override": ContextNeed("m", 8192, "min_context"),
        }

    def test_task_with_no_known_need_is_left_out(self) -> None:
        assert task_context_needs(_config({"t": {}})) == {}

    @pytest.mark.parametrize("value", [0, -1, True, "4096"])
    def test_malformed_min_context_is_refused(self, value: object) -> None:
        with pytest.raises(ValueError, match="min_context must be a positive"):
            task_context_needs(_config({"t": {"min_context": value}}))


class TestParseModelContext:
    def test_absent_block_states_nothing(self) -> None:
        context = parse_model_context("m", {})
        assert (context.native, context.max, dict(context.scaling)) == (None, None, {})

    def test_full_block(self) -> None:
        context = parse_model_context(
            "m",
            {
                "context": {
                    "native": 32768,
                    "max": 131072,
                    "scaling": {"json_model_override_args": "{}"},
                }
            },
        )
        assert (context.native, context.max) == (32768, 131072)
        assert dict(context.scaling) == {"json_model_override_args": "{}"}

    @pytest.mark.parametrize(
        ("block", "message"),
        [
            ("bad", "must be a dictionary"),
            ({"window": 1}, "unknown key"),
            ({"native": 0}, "context.native must be a positive"),
            ({"native": 4096, "max": 2048}, "below context.native"),
            ({"scaling": {"a": [1]}}, "scaling must map"),
        ],
    )
    def test_malformed_block_is_refused(self, block: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_model_context("m", {"context": block})


class TestPlanWindow:
    def test_reads_the_backend_spelling_from_a_plan_or_its_mapping(self) -> None:
        assert plan_window(_plan(16384)) == 16384
        assert plan_window(_plan(8192, backend="vllm")) == 8192
        raw = {
            "backend": "vllm",
            "assignments": [{"engine_params": {"max_model_len": 4096}}],
        }
        assert plan_window(raw) == 4096

    def test_unset_or_unknown_backend_is_none(self) -> None:
        assert plan_window(_plan(None)) is None
        assert plan_window({"backend": "other", "assignments": []}) is None

    def test_with_window_sets_the_window_and_scaling(self) -> None:
        plan = with_window(_plan(32768), 131072, {"disable_cuda_graph": True})
        params = plan.assignments[0].engine_params
        assert params["context_length"] == 131072
        assert params["disable_cuda_graph"] is True


class TestPlanServingGroups:
    def test_tasks_that_fit_share_the_unchanged_recipe_plan(self) -> None:
        plan = _plan(32768)
        groups = plan_serving_groups(_config({"a": _gen(1024), "b": {}}), {"m": plan})

        (group,) = groups
        assert group.key == DEFAULT_GROUP
        assert group.plans == {"m": plan}
        assert group.tasks == ("a", "b")
        assert group.windows["a"] == ServedWindow(DEFAULT_GROUP, 32768, False)

    def test_need_within_native_raises_the_shared_plan(self) -> None:
        config = _config(
            {"short": _gen(1024), "long": _gen(36000)},
            model={"path": "/models/m", "context": {"native": 40960}},
        )

        (group,) = plan_serving_groups(config, {"m": _plan(32768)})

        assert plan_window(group.plans["m"]) == 40096
        assert group.tasks == ("short", "long")
        assert group.windows["short"].window == 40096

    def test_max_tokens_equal_to_the_recipe_window_raises_it(self) -> None:
        config = _config(
            {"t": _gen(32768)},
            model={"path": "/models/m", "context": {"native": 40960}},
        )

        (group,) = plan_serving_groups(config, {"m": _plan(32768)})

        assert plan_window(group.plans["m"]) == 36864

    def test_native_window_defaults_to_the_checkpoint_config(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"max_position_embeddings": 40960})
        )
        config = _config({"long": _gen(36000)})

        (group,) = plan_serving_groups(
            config, {"m": _plan(32768, checkpoint=str(tmp_path))}
        )

        assert plan_window(group.plans["m"]) == 40096

    def test_need_beyond_native_gets_a_scaled_group_of_its_own(self) -> None:
        scaling = {"json_model_override_args": '{"rope_scaling": {}}'}
        config = _config(
            {"short": _gen(1024), "long": _gen(100000), "longer": _gen(120000)},
            model={
                "path": "/models/m",
                "context": {"native": 32768, "max": 131072, "scaling": scaling},
            },
        )

        default, scaled = plan_serving_groups(config, {"m": _plan(32768)})

        assert default.tasks == ("short",)
        assert plan_window(default.plans["m"]) == 32768
        assert scaled.key == "m@124096"
        assert scaled.tasks == ("long", "longer")
        params = scaled.plans["m"].assignments[0].engine_params
        assert params["context_length"] == 124096
        assert params["json_model_override_args"] == scaling["json_model_override_args"]
        assert scaled.windows["long"] == ServedWindow("m@124096", 124096, True)
        assert (default.launches, scaled.launches) == ({"m"}, {"m"})

    def test_root_with_every_task_scaled_launches_only_in_its_group(self) -> None:
        config = _config(
            {"long": _gen(100000)},
            model={
                "path": "/models/m",
                "context": {"native": 32768, "max": 131072, "scaling": {"x": 1}},
            },
        )

        default, scaled = plan_serving_groups(config, {"m": _plan(32768)})

        assert default.tasks == ()
        assert default.launches == frozenset()
        assert scaled.launches == {"m"}

    def test_root_with_no_task_still_launches_in_the_default_group(self) -> None:
        (default,) = plan_serving_groups(_config({}), {"m": _plan()})

        assert default.launches == {"m"}

    def test_malformed_window_override_names_the_model(self) -> None:
        config = _config(
            {"long": _gen(100000)},
            model={
                "path": "/models/m",
                "infer": {"overrides": {"context_length": "32k"}},
            },
        )

        with pytest.raises(
            ValueError, match="Model 'm' infer.overrides.context_length"
        ):
            plan_serving_groups(config, {"m": _plan(32768)})

    @pytest.mark.parametrize(
        ("model", "message"),
        [
            ({"context": {"native": 32768}}, "at most its native 32768"),
            ({"context": {"native": 32768, "max": 65536}}, "at most 65536"),
            (
                {"context": {"native": 32768, "max": 131072}},
                "no scaling parameters",
            ),
            ({}, "no known native window"),
            (
                {
                    "context": {"native": 131072},
                    "infer": {"overrides": {"context_length": 32768}},
                },
                "pins infer.overrides.context_length: 32768",
            ),
            (
                {"context": {"native": 131072}, "context_limit": 32768},
                "pins context_limit: 32768",
            ),
        ],
    )
    def test_unservable_need_is_refused_naming_task_and_shortfall(
        self, model: dict[str, Any], message: str
    ) -> None:
        config = _config(
            {"ok": _gen(1024), "long": _gen(100000)},
            model={"path": "/models/m", **model},
        )

        with pytest.raises(ServingShortfallError) as caught:
            plan_serving_groups(config, {"m": _plan(32768)})

        text = str(caught.value)
        assert (
            "task 'long' needs 104096 tokens "
            "(its max_tokens plus 4096 for the prompt)" in text
        )
        assert message in text
        assert "'ok'" not in text

    def test_external_model_tasks_stay_in_the_default_group(self) -> None:
        config = _config({"api_task": {"model": "api"}, "local": {"model": "m"}})
        config["models"]["api"] = {"name": "x", "api_base": "http://x/v1"}

        (group,) = plan_serving_groups(config, {"m": _plan()})

        assert group.tasks == ("api_task", "local")
        assert "api_task" not in group.windows

    def test_grouping_is_stable_across_task_order(self) -> None:
        model = {
            "path": "/models/m",
            "context": {"native": 32768, "max": 131072, "scaling": {"x": 1}},
        }
        forward = _config({"a": _gen(70000), "b": _gen(90000)}, model=model)
        backward = _config({"b": _gen(90000), "a": _gen(70000)}, model=model)

        keys = [
            [g.key for g in plan_serving_groups(cfg, {"m": _plan()})]
            for cfg in (forward, backward)
        ]
        assert keys[0] == keys[1] == [DEFAULT_GROUP, "m@94096"]


class TestCheckContextFits:
    def test_served_model_is_checked_against_its_plan(self) -> None:
        config = _config({"long": _gen(38000)})
        plans = {
            "m": {
                "backend": "sglang",
                "checkpoint": "/models/m",
                "assignments": [{"engine_params": {"context_length": 32768}}],
            }
        }

        with pytest.raises(ServingShortfallError, match="serves 32768"):
            check_context_fits(config, plans)
        check_context_fits(config, plans, frozenset({"other"}))

    def test_external_model_is_checked_against_its_context_limit(self) -> None:
        config = _config(
            {"long": _gen(38000)},
            model={"name": "x", "api_base": "http://x/v1", "context_limit": 32768},
        )

        with pytest.raises(ServingShortfallError, match="declares context_limit 32768"):
            check_context_fits(config, None)

    @pytest.mark.parametrize(("max_tokens", "fits"), [(28672, True), (28673, False)])
    def test_context_limit_leaves_prompt_headroom(
        self, max_tokens: int, fits: bool
    ) -> None:
        config = _config(
            {"t": _gen(max_tokens)},
            model={"name": "x", "api_base": "http://x/v1", "context_limit": 32768},
        )

        if fits:
            check_context_fits(config, None)
        else:
            with pytest.raises(ServingShortfallError, match="needs 32769 tokens"):
                check_context_fits(config, None)

    def test_unknown_window_is_not_checked(self) -> None:
        config = _config(
            {"long": _gen(1_000_000)},
            model={"name": "x", "api_base": "http://x/v1"},
        )

        check_context_fits(config, None)
