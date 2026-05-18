"""
Tests for sieval.core.utils.serialization.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import sys
import warnings
from dataclasses import dataclass
from types import ModuleType

import pytest

from sieval.core.models.model import ModelOutput
from sieval.core.tasks.context import TaskStageOutput
from sieval.core.utils.serialization import (
    dict_to_obj,
    global_type_registry,
    obj_to_dict,
    register_types,
    sieval_record,
)

# -- Test helpers --


@sieval_record
@dataclass
class SampleResult:
    score: float
    label: str
    details: dict | None = None


@sieval_record
@dataclass
class NestedResult:
    inner: SampleResult
    tags: list[str] | None = None


class SlotsOnly:
    __slots__ = ("x", "y")
    __sieval_record__ = True

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


# -- obj_to_dict tests --


class TestObjToDict:
    def test_primitives_and_basic_collections(self):
        assert obj_to_dict(None, False) is None
        assert obj_to_dict(42, False) == 42
        assert obj_to_dict(3.14, False) == 3.14
        assert obj_to_dict("hello", False) == "hello"
        assert obj_to_dict(True, False) is True

        d = {"a": 1, "b": "two", "c": None}
        result = obj_to_dict(d, False)
        # None values are stripped
        assert result == {"a": 1, "b": "two"}

        result = obj_to_dict([1, "a", None], False)
        assert result == [1, "a", None]

    def test_tuple_and_set_encoding(self):
        result = obj_to_dict((1, "a"), False)
        assert result == {"__sieval_cls__": "tuple", "items": [1, "a"]}

        result = obj_to_dict({1, 2, 3}, False)
        assert result["__sieval_cls__"] == "set"
        assert sorted(result["items"]) == [1, 2, 3]

    def test_dataclass_without_type(self):
        obj = SampleResult(score=0.95, label="correct")
        result = obj_to_dict(obj, add_type=False)
        assert result == {"score": 0.95, "label": "correct"}
        assert "__sieval_cls__" not in result

    def test_dataclass_with_type_and_nested(self):
        obj = SampleResult(score=0.95, label="correct")
        result = obj_to_dict(obj, add_type=True)
        assert result["score"] == 0.95
        assert result["label"] == "correct"
        assert result["__sieval_cls__"] == "SampleResult"
        assert "__sieval_mod__" in result

        inner = SampleResult(score=0.8, label="partial")
        outer = NestedResult(inner=inner, tags=["math", "easy"])
        result = obj_to_dict(outer, add_type=True)
        assert result["__sieval_cls__"] == "NestedResult"
        assert result["inner"]["__sieval_cls__"] == "SampleResult"
        assert result["tags"] == ["math", "easy"]

        obj = SampleResult(score=0.5, label="x", details=None)
        result = obj_to_dict(obj, add_type=False)
        assert "details" not in result

    def test_slots_object(self):
        obj = SlotsOnly(x=10, y=20)
        result = obj_to_dict(obj, add_type=True)
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["__sieval_cls__"] == "SlotsOnly"

    def test_unknown_object_falls_back_to_repr(self):
        # An object with neither __dict__ nor __slots__
        result = obj_to_dict(object(), False)
        assert isinstance(result, str)


# -- dict_to_obj tests --


class TestDictToObj:
    def test_primitives_and_basic_structures(self):
        assert dict_to_obj(42, {}) == 42
        assert dict_to_obj("hello", {}) == "hello"
        assert dict_to_obj(None, {}) is None
        result = dict_to_obj({"a": 1, "b": [2, 3]}, {})
        assert result == {"a": 1, "b": [2, 3]}
        result = dict_to_obj([1, {"x": 2}], {})
        assert result == [1, {"x": 2}]

    def test_list_items_preserve_registry_lookup(self):
        payload = [
            {
                "__sieval_mod__": SampleResult.__module__,
                "__sieval_cls__": "SampleResult",
                "score": 3.14,
                "label": "ok",
            }
        ]
        registry = {
            "SampleResult": SampleResult,
            f"{SampleResult.__module__}:SampleResult": SampleResult,
        }

        result = dict_to_obj(payload, registry)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SampleResult)
        assert result[0].score == 3.14
        assert result[0].label == "ok"

    def test_tuple_and_set_roundtrip(self):
        original = (1, "a", 3.0)
        serialized = obj_to_dict(original, False)
        restored = dict_to_obj(serialized, {})
        assert restored == original
        assert isinstance(restored, tuple)

        original = {1, 2, 3}
        serialized = obj_to_dict(original, False)
        restored = dict_to_obj(serialized, {})
        assert restored == original
        assert isinstance(restored, set)

    def test_class_name_only_lookup_is_deprecated(self, monkeypatch):
        import sieval.core.utils.serialization as _ser_mod

        monkeypatch.setattr(_ser_mod, "_warned_legacy_classname_lookup", False)
        # Intentionally register only the bare class name (no module-qualified key).
        # dict_to_obj first tries the module-qualified key (not present), then falls
        # back to the bare class name — triggering the DeprecationWarning.
        # Do NOT add the module-qualified key; that would bypass the deprecated path.
        registry = {"SampleResult": SampleResult}
        original = SampleResult(score=0.95, label="correct")
        serialized = obj_to_dict(original, add_type=True)
        with pytest.warns(DeprecationWarning, match="Class-name-only"):
            restored = dict_to_obj(serialized, registry)
        assert isinstance(restored, SampleResult)

    def test_missing_module_uses_legacy_lookup_and_warns_once(self, monkeypatch):
        import sieval.core.utils.serialization as _ser_mod

        monkeypatch.setattr(_ser_mod, "_warned_legacy_classname_lookup", False)
        payload = {"__sieval_cls__": "SampleResult", "score": 0.95, "label": "ok"}
        registry = {"SampleResult": SampleResult}

        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            restored_first = dict_to_obj(payload, registry)
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            restored_second = dict_to_obj(payload, registry)

        assert isinstance(restored_first, SampleResult)
        assert isinstance(restored_second, SampleResult)
        assert len(first) == 1
        assert "Class-name-only deserialization lookup is deprecated" in str(
            first[0].message
        )
        assert len(second) == 0

    def test_dataclass_roundtrip(self):
        registry = {
            "SampleResult": SampleResult,
            f"{SampleResult.__module__}:{SampleResult.__name__}": SampleResult,
        }
        original = SampleResult(score=0.95, label="correct", details={"key": "val"})
        serialized = obj_to_dict(original, add_type=True)
        restored = dict_to_obj(serialized, registry)
        assert isinstance(restored, SampleResult)
        assert restored.score == 0.95
        assert restored.label == "correct"
        assert restored.details == {"key": "val"}

    def test_nested_dataclass_roundtrip(self):
        registry = {
            "SampleResult": SampleResult,
            f"{SampleResult.__module__}:{SampleResult.__name__}": SampleResult,
            "NestedResult": NestedResult,
            f"{NestedResult.__module__}:{NestedResult.__name__}": NestedResult,
        }
        inner = SampleResult(score=0.8, label="partial")
        original = NestedResult(inner=inner, tags=["a", "b"])
        serialized = obj_to_dict(original, add_type=True)
        restored = dict_to_obj(serialized, registry)
        assert isinstance(restored, NestedResult)
        assert isinstance(restored.inner, SampleResult)
        assert restored.inner.score == 0.8
        assert restored.tags == ["a", "b"]

    def test_unknown_class_returns_plain_dict(self):
        data = {
            "__sieval_cls__": "UnknownClass",
            "__sieval_mod__": "fake.module",
            "x": 1,
        }
        result = dict_to_obj(data, {})
        assert isinstance(result, dict)
        assert result == {"x": 1}

    def test_registry_lookup_via_module_import(self):
        # SampleResult is registered via @sieval_record, so global_type_registry has it
        original = SampleResult(score=0.5, label="test")
        serialized = obj_to_dict(original, add_type=True)
        # Use empty registry but with valid module path — should import dynamically
        restored = dict_to_obj(serialized, {})
        # Should reconstruct via dynamic import since module is importable
        assert isinstance(restored, SampleResult)

    def test_module_qualified_lookup_prevents_name_collision(self):
        # Two same-name classes in different modules should deserialize by module+name.
        module_a = ModuleType("tests.fake_module_a")
        module_b = ModuleType("tests.fake_module_b")

        def _make_collision_type(module_name: str) -> type:
            def _init(self, value: int):
                self.value = value

            return type(
                "CollisionType",
                (),
                {
                    "__module__": module_name,
                    "__sieval_record__": True,
                    "__init__": _init,
                },
            )

        ClassA = _make_collision_type(module_a.__name__)
        ClassB = _make_collision_type(module_b.__name__)
        module_a.__dict__["CollisionType"] = ClassA
        module_b.__dict__["CollisionType"] = ClassB
        sys.modules[module_a.__name__] = module_a
        sys.modules[module_b.__name__] = module_b

        try:
            payload = {
                "__sieval_cls__": "CollisionType",
                "__sieval_mod__": module_b.__name__,
                "value": 7,
            }
            # Deliberately inject wrong class under class-name key.
            registry = {"CollisionType": ClassA}

            restored = dict_to_obj(payload, registry)

            assert isinstance(restored, ClassB)
            assert type(restored).__module__ == module_b.__name__
            assert restored.value == 7
        finally:
            sys.modules.pop(module_a.__name__, None)
            sys.modules.pop(module_b.__name__, None)

    def test_constructor_failure_returns_plain_payload(self):
        class Broken:
            __sieval_record__ = True

            def __init__(self, value: int):
                raise RuntimeError("cannot construct")

        payload = {
            "__sieval_cls__": "Broken",
            "__sieval_mod__": Broken.__module__,
            "value": 7,
        }
        registry = {"Broken": Broken, f"{Broken.__module__}:Broken": Broken}

        result = dict_to_obj(payload, registry)
        assert result == {"value": 7}


# -- register_types tests --


class TestRegisterTypes:
    def test_register_and_skip_non_sieval_types(self):
        import sieval.core.utils.serialization as _ser_mod

        _ser_mod._scanned_modules.discard("test_module_unique_1")
        _ser_mod._scanned_modules.discard("test_module_unique_2")
        registry: dict[str, type] = {}

        @sieval_record
        @dataclass
        class TempType:
            val: int

        @dataclass
        class PlainType:
            val: int

        register_types(registry, [TempType], "test_module_unique_1")
        assert "TempType" in registry

        register_types(registry, [PlainType], "test_module_unique_2")
        assert "PlainType" not in registry

    def test_deduplicates_by_module(self):
        import sieval.core.utils.serialization as _ser_mod

        _ser_mod._scanned_modules.discard("test_module_dedup")
        registry: dict[str, type] = {}

        @sieval_record
        @dataclass
        class DedupType:
            val: int

        register_types(registry, [DedupType], "test_module_dedup")
        registry.clear()
        # Second call with same module should be skipped
        register_types(registry, [DedupType], "test_module_dedup")
        assert "DedupType" not in registry


# -- ModelOutput serialization roundtrip --


class TestModelOutputSerialization:
    def test_full_and_minimal_roundtrip(self, sample_model_output, sample_model_meta):
        serialized = obj_to_dict(sample_model_output, add_type=True)
        assert serialized["__sieval_cls__"] == "ModelOutput"
        restored = dict_to_obj(serialized, global_type_registry)
        assert isinstance(restored, ModelOutput)
        assert restored.texts == ["hello world"]
        assert restored.finish_reasons == ["stop"]
        assert restored.usage["input_tokens"] == 100

        output = ModelOutput(model=sample_model_meta, texts=["hi"])
        serialized = obj_to_dict(output, add_type=True)
        restored = dict_to_obj(serialized, global_type_registry)
        assert isinstance(restored, ModelOutput)
        assert restored.texts == ["hi"]
        # Optional fields should be None
        assert restored.finish_reasons is None
        assert restored.usage is None
        assert restored.logprobs is None

    def test_with_logprobs(self, sample_model_meta):
        output = ModelOutput(
            model=sample_model_meta,
            texts=["test"],
            logprobs_tokens=["the", " cat"],
            logprobs=[-0.5, -1.2],
        )
        serialized = obj_to_dict(output, add_type=True)
        restored = dict_to_obj(serialized, global_type_registry)
        assert isinstance(restored, ModelOutput)
        assert restored.logprobs_tokens == ["the", " cat"]
        assert restored.logprobs == [-0.5, -1.2]


# -- TaskStageOutput serialization roundtrip --


class TestTaskStageOutputSerialization:
    def test_string_and_dict_value(self, sample_stage_meta):
        tso = TaskStageOutput(value="answer_A")
        serialized = obj_to_dict(tso, add_type=True)
        restored = dict_to_obj(serialized, global_type_registry)
        assert isinstance(restored, TaskStageOutput)
        assert restored.value == "answer_A"
        assert restored.meta is None

        tso = TaskStageOutput(value={"score": 0.9}, meta=sample_stage_meta)
        serialized = obj_to_dict(tso, add_type=True)
        restored = dict_to_obj(serialized, global_type_registry)
        assert isinstance(restored, TaskStageOutput)
        assert restored.value == {"score": 0.9}
        assert restored.meta["timing_s"] == 1.5

    def test_nested_model_output(self, sample_model_output):
        tso = TaskStageOutput(value=sample_model_output)
        serialized = obj_to_dict(tso, add_type=True)
        restored = dict_to_obj(serialized, global_type_registry)
        assert isinstance(restored, TaskStageOutput)
        assert isinstance(restored.value, ModelOutput)
        assert restored.value.texts == ["hello world"]
