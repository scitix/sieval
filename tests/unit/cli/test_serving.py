"""Tests for serving reconciliation: context needs, windows, serving groups.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from sieval.cli.serving import (
    DEFAULT_GROUP,
    SCORING_FP32_BYTES,
    ContextNeed,
    ScoringHeadroom,
    ScoringHeadroomError,
    ServedWindow,
    ServingShortfallError,
    check_context_fits,
    parse_model_context,
    plan_serving_groups,
    plan_window,
    reserve_scoring_headroom,
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
SCORING = "tests.unit.cli.test_serving._ScoringTask"

#: One H100 80GB, as ``nvidia-smi`` reports its memory.
H100_MIB = 81559
#: Qwen3's vocabulary.
VOCAB = 151936


class _LongContextTask(Task):
    """Declares the window it needs, as RULER-style tasks do."""

    requires: ClassVar[TaskRequirements] = TaskRequirements(min_context=50000)


class _ScoringTask(Task):
    """Scores its prompts' tokens, as perplexity tasks do."""

    requires: ClassVar[TaskRequirements] = TaskRequirements(input_scoring=True)


def _plan(
    window: int | None = 32768,
    *,
    backend: str = "sglang",
    checkpoint: str = "/models/m",
    **engine: Any,
) -> DeploymentPlan:
    params: dict[str, Any] = dict(engine)
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

        assert plan_window(group.plans["m"]) == 40960
        assert group.tasks == ("short", "long")
        assert group.windows["short"].window == 40960

    def test_max_tokens_below_native_raises_to_native(self) -> None:
        """Not to max_tokens plus headroom: the prompt gets native - max_tokens."""
        config = _config(
            {"t": _gen(32768)},
            model={"path": "/models/m", "context": {"native": 40960}},
        )

        (group,) = plan_serving_groups(config, {"m": _plan(32768)})

        assert plan_window(group.plans["m"]) == 40960

    def test_max_tokens_within_headroom_of_native_is_never_scaled(self) -> None:
        """38000 + headroom exceeds native 40960, but scaling changes outputs."""
        config = _config(
            {"t": _gen(38000)},
            model={
                "path": "/models/m",
                "context": {"native": 40960, "max": 131072, "scaling": {"x": 1}},
            },
        )

        (group,) = plan_serving_groups(config, {"m": _plan(32768)})

        assert group.tasks == ("t",)
        assert plan_window(group.plans["m"]) == 40960
        assert group.windows["t"] == ServedWindow(DEFAULT_GROUP, 40960, False)

    def test_max_tokens_at_or_above_native_gets_headroom_and_scales(self) -> None:
        config = _config(
            {"at": _gen(40960), "above": _gen(60000)},
            model={
                "path": "/models/m",
                "context": {"native": 40960, "max": 131072, "scaling": {"x": 1}},
            },
        )

        default, scaled = plan_serving_groups(config, {"m": _plan(32768)})

        assert default.tasks == ()
        assert scaled.key == "m@64096"
        assert scaled.tasks == ("at", "above")
        assert scaled.windows["at"] == ServedWindow("m@64096", 64096, True)

    def test_pinned_window_below_native_refuses_at_the_native_need(self) -> None:
        config = _config(
            {"t": _gen(38000)},
            model={
                "path": "/models/m",
                "context": {"native": 40960},
                "infer": {"overrides": {"context_length": 39000}},
            },
        )

        with pytest.raises(ServingShortfallError) as caught:
            plan_serving_groups(config, {"m": _plan(39000)})

        assert (
            "task 't' needs 40960 tokens (its max_tokens, served at the native "
            "window); model 'm' pins infer.overrides.context_length: 39000"
            in str(caught.value)
        )

    @pytest.mark.parametrize(
        ("task", "window", "scaled"),
        [
            ({"min_context": 36000, **_gen(30000)}, 36000, False),
            ({"class": LONG}, 50000, True),
        ],
        ids=["min_context-override", "declared-requires"],
    )
    def test_explicit_min_context_is_used_unchanged(
        self, task: dict[str, Any], window: int, scaled: bool
    ) -> None:
        """It already counts the prompt: no headroom, no rounding up to native."""
        config = _config(
            {"t": task},
            model={
                "path": "/models/m",
                "context": {"native": 40960, "max": 131072, "scaling": {"x": 1}},
            },
        )

        *_, group = plan_serving_groups(config, {"m": _plan(32768)})

        assert group.windows["t"].window == window
        assert group.windows["t"].scaled is scaled
        assert plan_window(group.plans["m"]) == window

    def test_unknown_native_keeps_max_tokens_plus_headroom(self) -> None:
        config = _config(
            {"t": _gen(38000)},
            model={"path": "/models/m", "context": {"max": 131072, "scaling": {}}},
        )

        with pytest.raises(ServingShortfallError) as caught:
            plan_serving_groups(config, {"m": _plan(40960)})

        assert (
            "task 't' needs 42096 tokens (its max_tokens plus 4096 for the prompt); "
            "model 'm' serves 40960 and has no known native window" in str(caught.value)
        )

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

        assert plan_window(group.plans["m"]) == 40960

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

    def test_served_native_window_fits_a_max_tokens_below_it(self) -> None:
        plan = {
            "backend": "sglang",
            "checkpoint": "/models/m",
            "assignments": [{"engine_params": {"context_length": 40960}}],
        }
        native = {"path": "/models/m", "context": {"native": 40960}}

        check_context_fits(_config({"t": _gen(38000)}, model=native), {"m": plan})
        with pytest.raises(ServingShortfallError) as caught:
            check_context_fits(
                _config({"t": _gen(38000)}, model={"path": "/models/m"}), {"m": plan}
            )
        assert "needs 42096 tokens" in str(caught.value)

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


def _checkpoint(tmp_path: Path, **config: Any) -> str:
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": VOCAB, "torch_dtype": "bfloat16", **config})
    )
    return str(tmp_path)


def _params(plan: DeploymentPlan) -> dict[str, Any]:
    return dict(plan.assignments[0].engine_params)


class TestReserveScoringHeadroom:
    """The rule, worked for Qwen3's vocabulary on one H100 80GB."""

    ROW = VOCAB * (2 + SCORING_FP32_BYTES)  # 2,127,104 B per scored token

    def _reserve(
        self, plan: DeploymentPlan, longest: int | None = None, **kwargs: Any
    ) -> tuple[DeploymentPlan, ScoringHeadroom]:
        kwargs.setdefault("gpu_memory_mib", H100_MIB)
        return reserve_scoring_headroom(
            plan, ("ppl",), longest_prompt=longest, **kwargs
        )

    def test_an_unknown_prompt_length_takes_the_largest_cap_the_share_allows(
        self, tmp_path: Path
    ) -> None:
        """10% of 81559 MiB holds 4020 rows: the cap is 3072, a whole step
        below; 3072 rows are 6231 MiB, 7.6%, so 0.87 becomes 0.79."""
        plan = _plan(checkpoint=_checkpoint(tmp_path), mem_fraction_static=0.87)

        served, headroom = self._reserve(plan)

        assert _params(served) == {
            "context_length": 32768,
            "mem_fraction_static": 0.79,
            "chunked_prefill_size": 3072,
        }
        assert headroom.prefill_tokens == 3072
        assert headroom.reserve_mib == 3072 * self.ROW // 2**20 == 6231
        assert headroom.params == {
            "chunked_prefill_size": (None, 3072),
            "mem_fraction_static": (0.87, 0.79),
        }
        assert headroom.reason == (
            "input scoring by ppl: full-vocabulary logits of 3072 prefill tokens "
            f"x {VOCAB} vocab x (2+12) B = 6231 MiB per GPU, 7.6% of 81559 MiB"
        )

    def test_a_short_scored_prompt_reserves_only_what_it_needs(
        self, tmp_path: Path
    ) -> None:
        """1500 tokens round up to 2048: 4154 MiB, 5.1%, so 0.87 becomes 0.81."""
        plan = _plan(checkpoint=_checkpoint(tmp_path), mem_fraction_static=0.87)

        served, headroom = self._reserve(plan, longest=1500)

        assert _params(served)["chunked_prefill_size"] == 2048
        assert _params(served)["mem_fraction_static"] == 0.81
        assert headroom.reserve_mib == 4154

    def test_a_larger_gpu_or_fp32_logits_scale_the_cap(self, tmp_path: Path) -> None:
        h200 = _plan(checkpoint=_checkpoint(tmp_path), mem_fraction_static=0.9)
        _, on_h200 = self._reserve(h200, gpu_memory_mib=143771)
        assert on_h200.prefill_tokens == 6144  # 7087 rows, a whole step below

        (tmp_path / "fp32").mkdir()
        fp32 = _plan(
            checkpoint=_checkpoint(tmp_path / "fp32", torch_dtype="float32"),
            mem_fraction_static=0.9,
        )
        _, wide = self._reserve(fp32)
        assert wide.prefill_tokens == 3072  # 16 B/row: 3519 rows

    def test_the_cap_never_exceeds_the_engine_default(self, tmp_path: Path) -> None:
        plan = _plan(checkpoint=_checkpoint(tmp_path), mem_fraction_static=0.9)
        _, headroom = self._reserve(plan, gpu_memory_mib=10 * H100_MIB)
        assert headroom.prefill_tokens == 8192

    def test_a_smaller_cap_already_set_is_kept_and_sizes_the_reserve(
        self, tmp_path: Path
    ) -> None:
        plan = _plan(
            checkpoint=_checkpoint(tmp_path),
            mem_fraction_static=0.87,
            chunked_prefill_size=1024,
        )

        served, headroom = self._reserve(plan)

        assert _params(served)["chunked_prefill_size"] == 1024
        assert _params(served)["mem_fraction_static"] == 0.84
        assert headroom.params == {"mem_fraction_static": (0.87, 0.84)}

    def test_overrides_win(self, tmp_path: Path) -> None:
        """A pinned cap stays, even above the budget, and sizes the reserve;
        a pinned share stays as it is."""
        plan = _plan(
            checkpoint=_checkpoint(tmp_path),
            mem_fraction_static=0.85,
            chunked_prefill_size=8192,
        )

        served, headroom = self._reserve(
            plan, pinned=frozenset({"chunked_prefill_size", "mem_fraction_static"})
        )

        assert served == plan
        assert headroom.params == {}
        assert headroom.prefill_tokens == 8192
        assert "mem_fraction_static kept: set by infer.overrides" in headroom.reason

    @pytest.mark.parametrize("cap", [16384, 32768])
    def test_a_pinned_cap_over_the_budget_is_refused(
        self, tmp_path: Path, cap: int
    ) -> None:
        """16384 rows would take 41% of the GPU, 32768 more than all of it:
        the share must not be lowered by that."""
        plan = _plan(
            checkpoint=_checkpoint(tmp_path),
            mem_fraction_static=0.87,
            chunked_prefill_size=cap,
        )

        with pytest.raises(
            ScoringHeadroomError, match=f"chunked_prefill_size {cap} pinned"
        ) as error:
            self._reserve(plan, pinned=frozenset({"chunked_prefill_size"}))
        assert "at most 4020" in str(error.value)

    def test_a_pinned_cap_over_the_budget_stands_with_a_pinned_share(
        self, tmp_path: Path
    ) -> None:
        plan = _plan(
            checkpoint=_checkpoint(tmp_path),
            mem_fraction_static=0.6,
            chunked_prefill_size=16384,
        )

        served, headroom = self._reserve(
            plan, pinned=frozenset({"chunked_prefill_size", "mem_fraction_static"})
        )

        assert served == plan
        assert headroom.prefill_tokens == 16384

    def test_each_role_reports_and_reserves_its_own_cap(self, tmp_path: Path) -> None:
        """A prefill/decode split keys *params* by role; a role that already
        holds a smaller cap reserves only for it."""
        plan = _plan(checkpoint=_checkpoint(tmp_path), mem_fraction_static=0.87)
        full = plan.assignments[0]
        prefill = dataclasses.replace(full, role=WellKnownRole.PREFILL)
        decode = dataclasses.replace(
            full,
            role=WellKnownRole.DECODE,
            engine_params={**full.engine_params, "chunked_prefill_size": 1024},
        )
        plan = dataclasses.replace(plan, assignments=(prefill, decode))

        served, headroom = self._reserve(plan)

        assert headroom.params == {
            "decode.mem_fraction_static": (0.87, 0.84),
            "prefill.chunked_prefill_size": (None, 3072),
            "prefill.mem_fraction_static": (0.87, 0.79),
        }
        assert [a.engine_params["mem_fraction_static"] for a in served.assignments] == [
            0.79,
            0.84,
        ]
        assert headroom.prefill_tokens == 3072

    def test_a_share_left_to_the_engine_is_not_lowered(self, tmp_path: Path) -> None:
        """SGLang then sizes its own share from the cap the reserve sets."""
        served, headroom = self._reserve(_plan(checkpoint=_checkpoint(tmp_path)))

        assert _params(served) == {
            "context_length": 32768,
            "chunked_prefill_size": 3072,
        }
        assert "mem_fraction_static left to the engine" in headroom.reason

    @pytest.mark.parametrize(
        ("backend", "config", "gpu", "reason"),
        [
            ("vllm", {"vocab_size": VOCAB}, H100_MIB, "no headroom rule for backend"),
            ("sglang", {}, H100_MIB, "vocab_size is unknown"),
            ("sglang", {"vocab_size": VOCAB}, None, "GPU memory is unknown"),
            ("sglang", {"vocab_size": 10**9}, H100_MIB, "exceeds the budget"),
        ],
    )
    def test_a_missing_fact_leaves_the_plan_and_says_why(
        self,
        tmp_path: Path,
        backend: str,
        config: dict[str, Any],
        gpu: int | None,
        reason: str,
    ) -> None:
        (tmp_path / "config.json").write_text(json.dumps(config))
        plan = _plan(
            backend=backend, checkpoint=str(tmp_path), mem_fraction_static=0.87
        )

        served, headroom = self._reserve(plan, gpu_memory_mib=gpu)

        assert served == plan
        assert reason in headroom.reason
        assert (headroom.params, headroom.prefill_tokens) == ({}, None)
        assert headroom.to_json()["params"] == {}


class TestScoringHeadroomInServingGroups:
    def _config(self, checkpoint: str, **model: Any) -> dict[str, Any]:
        config = _config(
            {"ppl": {"class": SCORING}, "gen": _gen(1024)},
            model={"path": checkpoint, **model},
        )
        return config

    def test_scoring_and_generation_share_one_engine_with_headroom(
        self, tmp_path: Path
    ) -> None:
        """The headroom changes no output, so it does not split the group."""
        checkpoint = _checkpoint(tmp_path)
        plan = _plan(checkpoint=checkpoint, mem_fraction_static=0.87)

        (group,) = plan_serving_groups(self._config(checkpoint), {"m": plan}, H100_MIB)

        assert group.tasks == ("ppl", "gen")
        assert _params(group.plans["m"])["mem_fraction_static"] == 0.79
        assert group.headroom["m"].tasks == ("ppl",)
        assert group.headroom["m"].to_json()["params"] == {
            "chunked_prefill_size": {"recipe": None, "served": 3072},
            "mem_fraction_static": {"recipe": 0.87, "served": 0.79},
        }

    def test_generation_alone_keeps_the_recipe(self, tmp_path: Path) -> None:
        checkpoint = _checkpoint(tmp_path)
        plan = _plan(checkpoint=checkpoint, mem_fraction_static=0.87)
        config = _config({"gen": _gen(1024)}, model={"path": checkpoint})

        (group,) = plan_serving_groups(config, {"m": plan}, H100_MIB)

        assert group.plans["m"] == plan
        assert group.headroom == {}

    def test_overrides_in_the_config_win(self, tmp_path: Path) -> None:
        checkpoint = _checkpoint(tmp_path)
        plan = _plan(checkpoint=checkpoint, mem_fraction_static=0.85)
        config = self._config(
            checkpoint, infer={"overrides": {"mem-fraction-static": 0.85}}
        )

        (group,) = plan_serving_groups(config, {"m": plan}, H100_MIB)

        assert _params(group.plans["m"])["mem_fraction_static"] == 0.85
        assert _params(group.plans["m"])["chunked_prefill_size"] == 3072

    def test_a_scaled_group_sizes_its_own_headroom(self, tmp_path: Path) -> None:
        checkpoint = _checkpoint(tmp_path)
        plan = _plan(checkpoint=checkpoint, mem_fraction_static=0.87)
        config = _config(
            {"gen": _gen(1024), "long_ppl": {"class": SCORING, "min_context": 60000}},
            model={
                "path": checkpoint,
                "context": {"native": 32768, "max": 131072, "scaling": {"x": 1}},
            },
        )

        default, scaled = plan_serving_groups(config, {"m": plan}, H100_MIB)

        assert default.plans["m"] == plan
        assert default.headroom == {}
        assert scaled.tasks == ("long_ppl",)
        assert scaled.headroom["m"].tasks == ("long_ppl",)
        assert _params(scaled.plans["m"])["chunked_prefill_size"] == 3072
        assert _params(scaled.plans["m"])["context_length"] == 60000

    def test_the_longest_scored_prompt_is_the_need_within_the_window(
        self, tmp_path: Path
    ) -> None:
        checkpoint = _checkpoint(tmp_path)
        plan = _plan(checkpoint=checkpoint, mem_fraction_static=0.87)
        config = _config(
            {"ppl": {"class": SCORING, "min_context": 1500}},
            model={"path": checkpoint},
        )

        (group,) = plan_serving_groups(config, {"m": plan}, H100_MIB)

        assert group.headroom["m"].prefill_tokens == 2048

    def test_unknown_gpu_memory_reports_the_unchanged_plan(
        self, tmp_path: Path
    ) -> None:
        checkpoint = _checkpoint(tmp_path)
        plan = _plan(checkpoint=checkpoint, mem_fraction_static=0.87)

        (group,) = plan_serving_groups(self._config(checkpoint), {"m": plan})

        assert group.plans["m"] == plan
        assert "GPU memory is unknown" in group.headroom["m"].reason
