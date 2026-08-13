"""
Unit tests for sieval/core/tasks/task.py.

Covers: name sanitisation, dialect-shape validation, requirement binding, make_context
(with and without dataset test_set).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import Dataset
from sieval.core.models import Model
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.gen_model import GenModel
from sieval.core.models.requirements import (
    InputKind,
    NamedModelBinding,
    RequirementContext,
    TaskRequirements,
)
from sieval.core.models.sglang_gen_model import SglangGenModel
from sieval.core.tasks.task import Task


# ===================================================================
# Minimal concrete implementations
# ===================================================================
class _SimpleDataset(Dataset):
    """Minimal dataset with three items."""

    def __init__(self, samples=None):
        self._samples = samples or [
            {"q": "a"},
            {"q": "b"},
            {"q": "c"},
        ]
        super().__init__("dummy")

    def load(self, name_or_path, **kwargs) -> HFDatasetDict:
        return HFDatasetDict({"test": HFDataset.from_list(self._samples)})


class _MockChatModel(ChatModel):
    """Construction-only mock: Task validation never invokes the wire."""

    def __init__(self):
        super().__init__(model="mock", api_key="fake")

    @property
    def dialect_id(self) -> str:
        return "openai_chat"

    @property
    def runtime_plan(self):
        return getattr(self, "_test_runtime_plan", None)


class _MockGenModel(GenModel):
    """Construction-only mock: Task validation never invokes the wire."""

    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")

    @property
    def dialect_id(self) -> str:
        return "openai_completions"

    @property
    def runtime_plan(self):
        return getattr(self, "_test_runtime_plan", None)


class _MockSglangGenModel(SglangGenModel):
    """Construction-only mock: Task validation never invokes the wire."""

    def __init__(self):
        super().__init__(model="mock-sglang", api_key="fake")

    @property
    def dialect_id(self) -> str:
        return "sglang_legacy"

    @property
    def runtime_plan(self):
        return getattr(self, "_test_runtime_plan", None)


class _ConcreteTask(Task):
    """Fully concrete Task with no model_type restriction."""

    model_type = None

    async def preprocess(self, raw, ctx):
        return raw

    async def infer(self, pre, ctx):
        return pre

    async def postprocess(self, inf, ctx):
        return inf

    async def feedback(self, post, ctx):
        return True, {}

    async def report(self, finals, fails):
        return {"total": len(finals)}


class _ChatOnlyTask(_ConcreteTask):
    model_type = "chat"
    requires = TaskRequirements(input=InputKind.CHAT)


class _GenOnlyTask(_ConcreteTask):
    model_type = "gen"
    requires = TaskRequirements(input=InputKind.COMPLETION)


class _ScoringTask(_ConcreteTask):
    """Declares an IR capability requirement (prompt-side scoring)."""

    requires = TaskRequirements(
        input=InputKind.COMPLETION,
        input_scoring=True,
        sampled_logprobs=True,
    )


class _ChatScoringTask(_ConcreteTask):
    """Impossible request used to prove dialect capability validation."""

    requires = TaskRequirements(
        input=InputKind.CHAT,
        input_scoring=True,
        sampled_logprobs=True,
    )


class _TopLogprobsTask(_ConcreteTask):
    """Declares the alternative-token breadth consumed by a CLP task."""

    requires = TaskRequirements(
        input=InputKind.COMPLETION,
        sampled_logprobs=True,
        min_top_logprobs=100,
    )


class _MetaNamedTask(_ConcreteTask):
    """Stands in for a ``@sieval_task``-decorated class, which sets this attr."""

    _sieval_task_meta = SimpleNamespace(name="meta_named_task")


class _MetaNamedSubTask(_MetaNamedTask):
    """Subclass with no meta of its own — must not borrow its parent's name."""


def _named_binding(name: str) -> NamedModelBinding:
    return NamedModelBinding(
        binding_id=f"binding:{name}",
        root_deployment_key=f"deployment:{name}",
        requested_model_id=f"org/{name}",
        config_name=name,
    )


# ===================================================================
# name property
# ===================================================================
class TestTaskName:
    def test_explicit_name_used(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel(), name="my_task")
        assert task.name == "my_task"

    def test_name_sanitised_removes_special_chars(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel(), name="my task/name!")
        assert "/" not in task.name
        assert " " not in task.name
        assert "!" not in task.name

    def test_name_sanitised_strips_leading_dots(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel(), name="...hidden")
        assert not task.name.startswith(".")

    def test_name_falls_back_to_class_name(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        # No explicit name; falls back to class name or "task"
        assert task.name  # not empty


# ===================================================================
# dialect-shape validation
# ===================================================================
class TestValidateModelType:
    def test_chat_task_with_chat_model_ok(self):
        # Should not raise
        _ChatOnlyTask(_SimpleDataset(), _MockChatModel())

    def test_gen_task_with_gen_model_ok(self):
        _GenOnlyTask(_SimpleDataset(), _MockGenModel())

    def test_gen_task_with_sglang_gen_model_ok(self):
        """The named SGLang legacy dialect still exposes completion input."""
        _GenOnlyTask(_SimpleDataset(), _MockSglangGenModel())

    def test_chat_task_with_gen_model_raises(self):
        with pytest.raises(TypeError, match="chat"):
            _ChatOnlyTask(_SimpleDataset(), _MockGenModel())

    def test_gen_task_with_chat_model_raises(self):
        with pytest.raises(TypeError, match="completion"):
            _GenOnlyTask(_SimpleDataset(), _MockChatModel())

    def test_no_model_type_restriction_accepts_both(self):
        _ConcreteTask(_SimpleDataset(), _MockChatModel())
        _ConcreteTask(_SimpleDataset(), _MockGenModel())

    def test_unrecognized_model_type_raises(self):
        class _CustomModel(Model):
            """Bare model with no bound dialect shape."""

        custom = object.__new__(_CustomModel)
        with pytest.raises(TypeError, match="bound dialect_id"):
            _ChatOnlyTask(_SimpleDataset(), custom)


# ===================================================================
# requires (runtime/dialect capability gate)
# ===================================================================
class TestRequiresCapabilityGate:
    def test_no_requires_accepts_any_model(self):
        """The default empty `requires` gates nothing."""
        _ConcreteTask(_SimpleDataset(), _MockChatModel())
        _ConcreteTask(_SimpleDataset(), _MockGenModel())

    def test_input_scoring_task_with_gen_model_ok(self):
        """GenModel's completions transport supplies InputScoring."""
        _ScoringTask(_SimpleDataset(), _MockGenModel())

    def test_input_scoring_task_with_sglang_model_ok(self):
        _ScoringTask(_SimpleDataset(), _MockSglangGenModel())

    def test_input_scoring_task_with_chat_model_raises(self):
        """Chat completions cannot score the prompt — fail loud at construction."""
        with pytest.raises(ValueError, match="input_scoring"):
            _ChatScoringTask(_SimpleDataset(), _MockChatModel())

    def test_top_logprobs_task_uses_semantic_capability_without_token_id_flag(self):
        _TopLogprobsTask(_SimpleDataset(), _MockGenModel())
        _TopLogprobsTask(_SimpleDataset(), _MockSglangGenModel())

    def test_runtime_plan_must_retain_required_capability(self):
        model = _MockGenModel()
        cast(Any, model)._test_runtime_plan = SimpleNamespace(
            dialect_id="openai_completions",
            available_capabilities=frozenset({"sampled_logprobs"}),
            capability_minimums={},
        )
        with pytest.raises(ValueError, match="top_logprobs"):
            _TopLogprobsTask(_SimpleDataset(), model)


class TestModelRequirementsFor:
    def test_binds_candidate_and_preserves_task_source(self):
        candidate = _named_binding("candidate")
        context = RequirementContext(
            model_bindings={
                "candidate": candidate,
                "judge": _named_binding("judge"),
            }
        )

        (result,) = _ScoringTask.model_requirements_for(context)

        assert result.role == "candidate"
        assert result.binding is candidate
        assert result.requires is _ScoringTask.requires
        assert result.source_task == "_ScoringTask"

    def test_single_legacy_model_alias_is_accepted(self):
        binding = _named_binding("legacy")
        (result,) = _ConcreteTask.model_requirements_for(
            RequirementContext(model_bindings={"model": binding})
        )

        assert result.role == "model"
        assert result.binding is binding

    def test_candidate_and_model_alias_are_ambiguous(self):
        context = RequirementContext(
            model_bindings={
                "candidate": _named_binding("candidate"),
                "model": _named_binding("legacy"),
            }
        )

        with pytest.raises(ValueError, match="ambiguous"):
            _ConcreteTask.model_requirements_for(context)


# ===================================================================
# auxiliary model-role binding — the declaration half
# ===================================================================
class TestBindRoleRequirement:
    """Reached directly by the eight grader/extractor tasks.

    ``model_requirements_for`` picks ``role`` out of the bindings it was handed,
    so it can never miss one; those tasks name a literal role instead, which is
    the only way the missing-binding branch fires.
    """

    def test_binds_the_named_role(self):
        grader = _named_binding("grader")
        context = RequirementContext(
            model_bindings={"candidate": _named_binding("candidate"), "grader": grader}
        )

        (result,) = _ConcreteTask._bind_role_requirement(
            context, "grader", _ScoringTask.requires
        )

        assert result.role == "grader"
        assert result.binding is grader
        assert result.requires is _ScoringTask.requires
        assert result.source_task == "_ConcreteTask"

    def test_source_task_prefers_the_decorated_name(self):
        context = RequirementContext(
            model_bindings={"grader": _named_binding("grader")}
        )

        (result,) = _MetaNamedTask._bind_role_requirement(
            context, "grader", _MetaNamedTask.requires
        )

        assert result.source_task == "meta_named_task"

    def test_source_task_is_not_inherited_from_the_parent(self):
        """Looked up in ``cls.__dict__``, so a subclass reports its own name."""
        context = RequirementContext(
            model_bindings={"grader": _named_binding("grader")}
        )

        (result,) = _MetaNamedSubTask._bind_role_requirement(
            context, "grader", _MetaNamedSubTask.requires
        )

        assert result.source_task == "_MetaNamedSubTask"

    def test_missing_binding_names_the_task_and_role(self):
        context = RequirementContext(
            model_bindings={"candidate": _named_binding("candidate")}
        )

        with pytest.raises(
            ValueError, match="_ConcreteTask requires a 'grader' model binding"
        ) as excinfo:
            _ConcreteTask._bind_role_requirement(
                context, "grader", _ConcreteTask.requires
            )

        assert isinstance(excinfo.value.__cause__, KeyError)

    def test_missing_binding_uses_an_before_a_vowel_role(self):
        context = RequirementContext(
            model_bindings={"candidate": _named_binding("candidate")}
        )

        with pytest.raises(ValueError, match="requires an 'extractor' model binding"):
            _ConcreteTask._bind_role_requirement(
                context, "extractor", _ConcreteTask.requires
            )

    def test_rejects_a_context_of_the_wrong_type(self):
        with pytest.raises(TypeError, match="context must be a RequirementContext"):
            _ConcreteTask._bind_role_requirement(
                cast(Any, {"grader": _named_binding("grader")}),
                "grader",
                _ConcreteTask.requires,
            )

    @pytest.mark.parametrize("role", ["", cast(Any, None)])
    def test_rejects_an_empty_or_non_string_role(self, role):
        context = RequirementContext(
            model_bindings={"grader": _named_binding("grader")}
        )

        with pytest.raises(TypeError, match="role must be a non-empty string"):
            _ConcreteTask._bind_role_requirement(context, role, _ConcreteTask.requires)

    def test_rejects_requires_of_the_wrong_type(self):
        """Pins the contract at this entry point; ``TaskModelRequirement``
        owns the check itself."""
        context = RequirementContext(
            model_bindings={"grader": _named_binding("grader")}
        )

        with pytest.raises(TypeError, match="requires must be TaskRequirements"):
            _ConcreteTask._bind_role_requirement(
                context, "grader", cast(Any, {"input": "chat"})
            )


# ===================================================================
# auxiliary model-role resolution — the resolution half
# ===================================================================
def _unreachable_build(_configured):
    raise AssertionError("build must not be called when models_by_role is supplied")


class TestResolveRoleModel:
    """The shared ``models_by_role`` contract, independent of any one task."""

    def test_returns_the_pooled_role_model(self):
        grader = _MockChatModel()

        result = Task._resolve_role_model(
            "grader", None, {"grader": grader}, build=_unreachable_build
        )

        assert result is grader

    def test_rejects_both_supply_routes(self):
        grader = _MockChatModel()

        with pytest.raises(ValueError, match="cannot both be supplied"):
            Task._resolve_role_model(
                "grader", grader, {"grader": grader}, build=_unreachable_build
            )

    def test_missing_role_names_it_and_chains_the_keyerror(self):
        with pytest.raises(ValueError, match="missing the 'grader' model") as excinfo:
            Task._resolve_role_model("grader", None, {}, build=_unreachable_build)

        assert isinstance(excinfo.value.__cause__, KeyError)

    def test_build_receives_the_configured_value(self):
        """``configured`` reaches ``build``, rather than being closed over.

        The call site threading it twice is how a mistyped ``configured`` slips
        past the both-supplied guard, so the pairing is pinned here.
        """
        built = _MockChatModel()
        seen: list[object] = []
        configured = {"model": "grader-1"}

        def build(cfg) -> Model:
            seen.append(cfg)
            return built

        result = Task._resolve_role_model("grader", configured, None, build=build)

        assert seen == [configured]
        assert result is built


# ===================================================================
# make_context
# ===================================================================
class TestMakeContext:
    def test_make_context_with_raw(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        raw = {"q": "hello"}
        ctx = task.make_context(0, raw=raw)
        assert ctx.sample_id == 0
        assert ctx.raw_sample == raw

    def test_make_context_lazy_fetch_from_dataset(self):
        """Integer sample_id with no raw should fetch from dataset.test_set."""
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        ctx = task.make_context(1)
        assert ctx.sample_id == 1
        assert ctx.raw_sample == {"q": "b"}

    def test_make_context_out_of_bounds_gives_none_raw(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        ctx = task.make_context(999)
        assert ctx.raw_sample is None

    def test_make_context_string_id_no_lazy_fetch(self):
        """String sample_id cannot index dataset, raw must remain None."""
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        ctx = task.make_context("sample-abc")
        assert ctx.sample_id == "sample-abc"
        assert ctx.raw_sample is None

    def test_make_context_no_test_set(self):
        """Dataset with no test split returns None raw."""

        class _TrainOnlyDataset(_SimpleDataset):
            def load(self, name_or_path, **kwargs) -> HFDatasetDict:
                return HFDatasetDict({"train": HFDataset.from_list([{"q": "x"}])})

        task = _ConcreteTask(_TrainOnlyDataset(), _MockChatModel())
        ctx = task.make_context(0)
        assert ctx.raw_sample is None


# ===================================================================
# setup / shutdown hooks
# ===================================================================
class TestSetupShutdown:
    @pytest.mark.anyio
    async def test_setup_is_noop_by_default(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        result = await task.setup()
        assert result is None

    @pytest.mark.anyio
    async def test_shutdown_is_noop_by_default(self):
        task = _ConcreteTask(_SimpleDataset(), _MockChatModel())
        result = await task.shutdown()
        assert result is None


def test_all_task_subclasses_bind_treport():
    """Every Task subclass that directly parameterises Task[...] binds TReport."""
    import importlib
    import pkgutil
    import types
    import typing

    import sieval.tasks
    from sieval.core.tasks.task import Task

    for _importer, name, _ispkg in pkgutil.walk_packages(
        sieval.tasks.__path__, "sieval.tasks."
    ):
        try:
            importlib.import_module(name)
        except Exception:
            continue

    for cls in _all_task_subclasses(Task):
        orig_bases = types.get_original_bases(cls)
        task_base = next(
            (b for b in orig_bases if typing.get_origin(b) is Task),
            None,
        )
        if task_base is None:
            continue

        type_args = typing.get_args(task_base)
        assert len(type_args) == 6, (
            f"{cls.__name__}: expected 6 type args on Task[...], got {len(type_args)}"
        )


def _all_task_subclasses(cls: type) -> set[type]:
    result: set[type] = set()
    for sub in cls.__subclasses__():
        result.add(sub)
        result.update(_all_task_subclasses(sub))
    return result
