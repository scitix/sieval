"""
Unit tests for sieval/core/tasks/task.py.

Covers: name sanitisation, _validate_model_type, make_context
(with and without dataset test_set).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import Dataset
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.gen_model import GenModel
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
    def __init__(self):
        super().__init__(model="mock", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        return ModelOutput(model=self.meta(), texts=["ok"])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


class _MockGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        return ModelOutput(model=self.meta(), texts=["ok"])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


class _MockSglangGenModel(SglangGenModel):
    def __init__(self):
        super().__init__(model="mock-sglang", api_key="fake")

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        return ModelOutput(model=self.meta(), texts=["ok"])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


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


class _GenOnlyTask(_ConcreteTask):
    model_type = "gen"


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
# _validate_model_type
# ===================================================================
class TestValidateModelType:
    def test_chat_task_with_chat_model_ok(self):
        # Should not raise
        _ChatOnlyTask(_SimpleDataset(), _MockChatModel())

    def test_gen_task_with_gen_model_ok(self):
        _GenOnlyTask(_SimpleDataset(), _MockGenModel())

    def test_gen_task_with_sglang_gen_model_ok(self):
        """SglangGenModel extends Model[str], not GenModel — still counts as 'gen'."""
        _GenOnlyTask(_SimpleDataset(), _MockSglangGenModel())

    def test_chat_task_with_gen_model_raises(self):
        with pytest.raises(TypeError, match="chat"):
            _ChatOnlyTask(_SimpleDataset(), _MockGenModel())

    def test_gen_task_with_chat_model_raises(self):
        with pytest.raises(TypeError, match="gen"):
            _GenOnlyTask(_SimpleDataset(), _MockChatModel())

    def test_no_model_type_restriction_accepts_both(self):
        _ConcreteTask(_SimpleDataset(), _MockChatModel())
        _ConcreteTask(_SimpleDataset(), _MockGenModel())

    def test_unrecognized_model_type_raises(self):
        """A model that is neither ChatModel nor GenModel should raise TypeError."""
        from sieval.core.models.model import Model, ModelOutput

        class _CustomModel(Model):
            async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
                return ModelOutput(model=self.meta(), texts=["ok"])

            async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
                raise NotImplementedError

        custom = _CustomModel(model="custom", api_key="fake")
        with pytest.raises(TypeError, match="requires a ChatModel or GenModel"):
            _ChatOnlyTask(_SimpleDataset(), custom)


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
