"""Tests for sieval.cli.resolution — config-string → class, and model-type derivation.

Moved here with `sieval/cli/resolution.py` itself; `tests/unit/` mirrors
`sieval/`, and these never tested the eval session.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from sieval.cli.resolution import (
    _guess_submodule_names,
    derive_model_type,
    load_class_from_name,
    load_class_from_path,
    normalize_inline_model_binding,
    resolve_class,
    resolve_config_model_types,
    resolve_key_function,
)
from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    InlineModelBinding,
    InputKind,
    TaskModelRequirement,
    TaskRequirements,
)

# ===================================================================
# resolve_task_class: submodule search by naming convention
# ===================================================================


def _find_project_root() -> Path:
    """Find repo root: parent of the `sieval` package used in tests."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "sieval" / "tasks" / "aime_2024_0shot_gen.py").exists():
            return p  # p is parent of sieval package, i.e. repo root
        p = p.parent
    return Path(__file__).resolve().parents[4]


@contextmanager
def _project_sieval_first():
    """
    Ensure sieval.tasks is resolved from project root (not mutants).
    Under mutmut, resolve_task_class runs from mutants; import_module() uses
    sys.modules first, so if sieval/sieval.tasks are already loaded from mutants,
    we must clear them and prepend project root to sys.path so the next import
    re-resolves from the real package.
    """
    root = _find_project_root()
    inserted = str(root) not in sys.path[:1]
    if inserted:
        sys.path.insert(0, str(root))

    # Clear sieval* from sys.modules so import_module() re-resolves via sys.path.
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "sieval" or k.startswith("sieval.")
    }
    try:
        yield
    finally:
        if inserted and sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
        # Evict any sieval* modules loaded during the yield — otherwise they
        # shadow the originals (e.g. task modules that re-registered into a
        # transient TASK_REGISTRY) for subsequent tests.
        for k in [
            k
            for k in list(sys.modules)
            if (k == "sieval" or k.startswith("sieval.")) and k not in saved
        ]:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v


# ===================================================================
# load_class_from_path
# ===================================================================
class TestLoadClassFromPath:
    def test_valid_path(self):
        # `Model` is defined in `sieval.core.models.model`; resolving a class
        # that the module merely re-exports would not prove the path is walked.
        cls = load_class_from_path("sieval.core.models.model.Model")
        from sieval.core.models import Model

        assert cls is Model

    # (class_path, expected_exception, expected_error_pattern)
    @pytest.mark.parametrize(
        "class_path,error_type,error_match",
        [
            ("Model", ValueError, "Invalid class path"),
            ("nonexistent.module.Foo", ImportError, "Could not import"),
            (
                "sieval.core.models.model.NonExistentClass",
                AttributeError,
                "has no class",
            ),
            # Resolves, but not to a class. The shared loader returns any
            # attribute, so the declared `-> type` is only true because this
            # is checked rather than asserted.
            (
                "sieval.cli.resolution.load_class_from_path",
                ValueError,
                "not a class",
            ),
        ],
    )
    def test_invalid_path_raises(self, class_path, error_type, error_match):
        with pytest.raises(error_type, match=error_match):
            load_class_from_path(class_path)


# ===================================================================
# resolve_key_function
# ===================================================================
class TestResolveKeyFunction:
    """The config counterpart of handing a Python caller a function directly.

    YAML cannot hold a function body, so it names one — exactly as `class:`
    names a class rather than defining it. These tests pin that a config can
    reach any callable a Python caller could pass, and nothing else.
    """

    def test_resolves_a_dotted_path_to_the_function_itself(self):
        from sieval.cli.resolution import derive_model_type as expected

        assert resolve_key_function("sieval.cli.resolution.derive_model_type") is (
            expected
        )

    # (spec, expected_exception, expected_error_pattern)
    @pytest.mark.parametrize(
        "spec,error_type,error_match",
        [
            # A bare name has no registry to search, unlike a dataset or task
            # class, so importing it from wherever it first appeared would make
            # the resolved function depend on import order.
            ("my_key", ValueError, "Invalid function path"),
            (".relative", ValueError, "Relative import syntax"),
            ("nonexistent.module.fn", ImportError, "Could not import"),
            (
                "sieval.cli.resolution.no_such_function",
                AttributeError,
                "has no function",
            ),
            # Resolvable, importable, and useless as a key: caught here rather
            # than as a TypeError from deep inside the row loop.
            ("sieval.cli.resolution.DATASET_MODULE", ValueError, "not a callable"),
            (42, ValueError, "must be a string"),
        ],
    )
    def test_a_spec_it_cannot_use_raises(self, spec, error_type, error_match):
        with pytest.raises(error_type, match=error_match):
            resolve_key_function(spec)

    def test_a_class_is_accepted_because_a_class_is_callable(self):
        # Nothing requires a key function be a `def`; a small callable class
        # holding configuration is a legitimate way to write one.
        from sieval.core.models.model import Model

        assert resolve_key_function("sieval.core.models.model.Model") is Model


# ===================================================================
# load_class_from_name
# ===================================================================
class TestLoadClassFromName:
    # (class_name, search_modules, expected_fully_qualified_class_path)
    @pytest.mark.parametrize(
        "class_name,search_modules,expected_path",
        [
            (
                "Model",
                ["sieval.core.models.model"],
                "sieval.core.models.model.Model",
            ),
            (
                "ChatModel",
                ["sieval.core.models.model", "sieval.core.models.chat_model"],
                "sieval.core.models.chat_model.ChatModel",
            ),
            (
                "Model",
                ["nonexistent.module", "sieval.core.models.model"],
                "sieval.core.models.model.Model",
            ),
        ],
    )
    def test_find_class_in_search_modules(
        self, class_name, search_modules, expected_path
    ):
        cls = load_class_from_name(class_name, search_modules)
        assert cls is load_class_from_path(expected_path)

    def test_not_found_raises(self):
        with pytest.raises(ImportError, match="Could not find class"):
            load_class_from_name("FakeClass", ["sieval.core.models.model"])


# ===================================================================
# resolve_class
# ===================================================================
class TestResolveClass:
    # (class_spec, search_modules, expected_fully_qualified_class_path)
    @pytest.mark.parametrize(
        "class_spec,search_modules,expected_path",
        [
            (
                "sieval.core.models.model.Model",
                [],
                "sieval.core.models.model.Model",
            ),
            (
                "Model",
                ["sieval.core.models.model"],
                "sieval.core.models.model.Model",
            ),
        ],
    )
    def test_resolve_class(self, class_spec, search_modules, expected_path):
        cls = resolve_class(class_spec, search_modules)
        assert cls is load_class_from_path(expected_path)

    def test_leading_dot_raises_value_error(self):
        # Relative import syntax is explicitly rejected.
        with pytest.raises(ValueError, match="Relative import syntax"):
            resolve_class(".Model", ["sieval.core.models.model"])


# ===================================================================
# _guess_submodule_names
# ===================================================================
class TestGuessSubmoduleNames:
    # (class_name, expected_fragment_in_candidates)
    @pytest.mark.parametrize(
        "class_name,expected_fragment",
        [
            ("MathTask", "math"),
            ("AIME2024Task", "aime_2024"),
            ("GPQADiamondTask", "gpqa"),
        ],
    )
    def test_expected_candidates_present(self, class_name, expected_fragment):
        names = _guess_submodule_names(class_name)
        assert any(expected_fragment in n for n in names)

    def test_task_suffix_removed(self):
        names = _guess_submodule_names("MathTask")
        # Should not keep trailing "_task" in any candidate.
        for n in names:
            assert not n.endswith("_task")

    # (class_name, expected_fragments_in_candidates)
    @pytest.mark.parametrize(
        "class_name,expected_fragments",
        [
            ("AIME2024ZeroShotGenTask", ["0shot", "zero_shot"]),
            ("MathFewShotTask", ["kshot", "few_shot"]),
        ],
    )
    def test_shot_aliases_present(self, class_name, expected_fragments):
        names = _guess_submodule_names(class_name)
        for fragment in expected_fragments:
            assert any(fragment in n for n in names)


class TestDeriveModelType:
    """The shared resolver accepts only normalized task-side evidence."""

    @staticmethod
    def _requirements(*kinds: InputKind) -> AggregatedTaskRequirements:
        return AggregatedTaskRequirements(
            input=frozenset(kinds),
            input_sources={kind: frozenset({f"{kind.value}_task"}) for kind in kinds},
        )

    def test_explicit_type_is_fallback_only_without_evidence(self):
        empty = self._requirements()
        assert derive_model_type("m", "gen", empty) == "gen"
        assert derive_model_type("m", None, empty) == "chat"

    def test_normalized_task_evidence_is_authoritative(self):
        completion = self._requirements(InputKind.COMPLETION)
        assert derive_model_type("m", None, completion) == "gen"
        with pytest.raises(ValueError, match="checked assertion"):
            derive_model_type("m", "chat", completion)

    def test_conflicting_normalized_inputs_report_sources(self):
        requirements = self._requirements(InputKind.CHAT, InputKind.COMPLETION)
        with pytest.raises(ValueError, match="conflicting normalized input") as exc:
            derive_model_type("m", None, requirements)
        assert "chat_task" in str(exc.value)
        assert "completion_task" in str(exc.value)


class TestResolveConfigModelTypes:
    """The YAML adapter invokes requirement hooks and delegates kind choice."""

    class MisleadingCompletionTask:
        model_type = "chat"  # Legacy metadata must not be consulted.

        def __init__(self, *, grader=None, models_by_role=None):
            del grader, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.COMPLETION),
                    source_task="normalized_completion",
                ),
            )

    class CompletionWithExtractorTask:
        def __init__(self, *, extractor=None, models_by_role=None):
            del extractor, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            assert "extractor" not in context.task_args
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.COMPLETION),
                    source_task="completion_with_extractor",
                ),
                TaskModelRequirement(
                    role="extractor",
                    binding=context.model_bindings["extractor"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="completion_with_extractor",
                ),
            )

    class CompletionWithSelfExtractorTask:
        def __init__(self, *, extractor=None, models_by_role=None):
            del extractor, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            assert context.task_args["extractor"] == "self"
            assert "extractor" not in context.model_bindings
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.COMPLETION),
                    source_task="completion_with_self_extractor",
                ),
            )

    def test_hook_evidence_flows_through_derived_model_root(self):
        config = {
            "models": {
                "base": {"name": "org/model"},
                "child": {"base": "base"},
            },
            "tasks": {"task": {"model": "child", "class": "fake.CompletionTask"}},
        }
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            return_value=self.MisleadingCompletionTask,
        ):
            result = resolve_config_model_types(config)

        assert result.model_types_by_root == {"model:base": "gen"}
        assert result.model_types_by_config == {"base": "gen", "child": "gen"}

    def test_explicit_type_cannot_override_hook_evidence(self):
        config = {
            "models": {"m": {"name": "org/model", "type": "chat"}},
            "tasks": {"task": {"model": "m", "class": "fake.Task"}},
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=self.MisleadingCompletionTask,
            ),
            pytest.raises(ValueError, match="checked assertion"),
        ):
            resolve_config_model_types(config)

    def test_inline_extractor_is_available_to_requirement_hook(self):
        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {
                        "extractor": {
                            "model": "org/extractor",
                            "api_key": "secret",
                        }
                    },
                }
            },
        }
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            return_value=self.CompletionWithExtractorTask,
        ):
            result = resolve_config_model_types(config)

        assert result.model_types_by_config == {"m": "gen"}

    def test_inline_sglang_legacy_role_is_rejected_during_resolution(self):
        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {
                        "extractor": {
                            "model": "org/extractor",
                            "dialect": "sglang_legacy",
                        }
                    },
                }
            },
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=self.CompletionWithExtractorTask,
            ),
            pytest.raises(
                ValueError,
                match=r"inline extractor.*sglang_legacy.*named model",
            ),
        ):
            resolve_config_model_types(config)

    def test_configured_role_must_be_declared_by_requirement_hook(self):
        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {"grader": {"model": "org/grader"}},
                }
            },
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=self.MisleadingCompletionTask,
            ),
            pytest.raises(
                ValueError,
                match=(
                    r"MisleadingCompletionTask\.model_requirements_for\(\) "
                    r"did not declare normalized model role\(s\): 'grader'"
                ),
            ),
        ):
            resolve_config_model_types(config)

    def test_self_extractor_remains_task_arg_and_reuses_candidate(self):
        from sieval.tasks.agieval_0shot_gen import AGIEvalZeroShotGenTask

        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {"extractor": "self"},
                }
            },
        }
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            return_value=AGIEvalZeroShotGenTask,
        ):
            result = resolve_config_model_types(config)

        assert result.model_types_by_config == {"m": "chat"}

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("name", None),
            ("dataset", {}),
            ("model", None),
            ("models_by_role", {}),
        ],
    )
    def test_composition_owned_task_args_are_rejected_by_presence(self, key, value):
        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {key: value},
                }
            },
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=self.MisleadingCompletionTask,
            ),
            pytest.raises(ValueError, match=rf"composition-owned.*{key}"),
        ):
            resolve_config_model_types(config)

    @pytest.mark.parametrize(
        ("role", "source"),
        [
            ("grader", None),
            ("extractor", None),
            ("extractor", "self"),
            ("extractor", {"model": "org/extractor"}),
        ],
    )
    def test_non_owner_task_cannot_configure_model_role(self, role, source):
        from sieval.tasks.gsm8k_0shot_gen import GSM8KZeroShotGenTask

        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {role: source},
                }
            },
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=GSM8KZeroShotGenTask,
            ),
            pytest.raises(ValueError, match=rf"model role.*{role}"),
        ):
            resolve_config_model_types(config)

    def test_nested_inline_secret_is_absent_and_does_not_change_binding_id(self):
        captured: list[InlineModelBinding] = []

        class CapturingJudgeTask:
            def __init__(self, *, grader=None, models_by_role=None):
                del grader, models_by_role

            @classmethod
            def model_requirements_for(cls, context):
                candidate = TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="capturing_judge",
                )
                grader = TaskModelRequirement(
                    role="grader",
                    binding=context.model_bindings["grader"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="capturing_judge",
                )
                assert isinstance(grader.binding, InlineModelBinding)
                captured.append(grader.binding)
                return (candidate, grader)

        configs = []
        for secret in ("SECRET-A", "SECRET-B"):
            config = {
                "models": {"m": {"name": "org/candidate"}},
                "tasks": {
                    "task": {
                        "model": "m",
                        "class": "fake.Task",
                        "args": {
                            "grader": {
                                "model": "org/grader",
                                "args": {
                                    "api_key": secret,
                                    "temperature": 0,
                                },
                            }
                        },
                    }
                },
            }
            configs.append(config)
            with patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=CapturingJudgeTask,
            ):
                resolve_config_model_types(config)

        first, second = captured
        assert first.binding_id == second.binding_id
        assert first.config == second.config
        assert "SECRET-A" not in repr(first.config)
        assert "SECRET-B" not in repr(second.config)
        assert first.config["args"] == {"temperature": 0}
        assert "SECRET-A" in repr(configs[0])

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("args", [], "args must be a dictionary"),
            ("infer_args", [], "infer_args must be a dictionary"),
            ("infer_args", {"api_key": "secret"}, "binding resources.*api_key"),
        ],
    )
    def test_config_adapter_rejects_invalid_task_argument_surfaces(
        self, field, value, match
    ):
        task = {"model": "m", "class": "fake.Task", field: value}
        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {"task": task},
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=self.MisleadingCompletionTask,
            ),
            pytest.raises(ValueError, match=match),
        ):
            resolve_config_model_types(config)

    def test_non_self_extractor_string_is_rejected(self):
        config = {
            "models": {"m": {"name": "org/candidate"}},
            "tasks": {
                "task": {
                    "model": "m",
                    "class": "fake.Task",
                    "args": {"extractor": "itself"},
                }
            },
        }
        with (
            patch(
                "sieval.cli.resolution.resolve_task_class",
                return_value=self.CompletionWithSelfExtractorTask,
            ),
            pytest.raises(ValueError, match="extractor must be 'self'"),
        ):
            resolve_config_model_types(config)

    @pytest.mark.parametrize("tasks", [["arc"], {"arc": "Task"}])
    def test_rejects_malformed_tasks_mapping(self, tasks):
        with pytest.raises(ValueError, match="'tasks.*configuration must be"):
            resolve_config_model_types({"models": {"m": {}}, "tasks": tasks})

    @pytest.mark.parametrize(
        "error",
        [ImportError("module not found"), AttributeError("class not found")],
    )
    def test_unresolvable_task_class_is_skipped(self, error):
        config = {
            "models": {"m": {}},
            "tasks": {"t1": {"model": "m", "class": "missing.Task"}},
        }
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            side_effect=error,
        ):
            result = resolve_config_model_types(config)
        assert result.model_types_by_config == {"m": "chat"}

    def test_unresolvable_task_does_not_block_resolvable_task(self):
        config = {
            "models": {"m": {}},
            "tasks": {
                "bad_task": {"model": "m", "class": "bad.module.BadTask"},
                "good_task": {"model": "m", "class": "good.module.GenTask"},
            },
        }

        call_count = 0

        def mock_resolve(_spec):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ImportError("bad module")
            return self.MisleadingCompletionTask

        with patch(
            "sieval.cli.resolution.resolve_task_class",
            side_effect=mock_resolve,
        ):
            result = resolve_config_model_types(config)
        assert result.model_types_by_config == {"m": "gen"}
        assert call_count == 2


class TestResolveTaskClass:
    """
    Test resolve_task_class's submodule search path.
    """

    def test_full_path_and_short_name_resolve(self):
        """Both full-path and short-name task specs should resolve."""
        from sieval.cli.resolution import resolve_task_class

        with _project_sieval_first():
            full_path_cls = resolve_task_class(
                "sieval.tasks.aime_2024_0shot_gen.AIME2024ZeroShotGenTask"
            )
            from sieval.tasks.aime_2024_0shot_gen import AIME2024ZeroShotGenTask

            short_name_cls = resolve_task_class("AIME2024ZeroShotGenTask")
        assert full_path_cls is AIME2024ZeroShotGenTask
        assert short_name_cls is AIME2024ZeroShotGenTask

    def test_unknown_class_raises_import_error(self):
        """A class that cannot be found should raise ImportError."""
        from sieval.cli.resolution import resolve_task_class

        with pytest.raises(ImportError, match="Could not find task class"):
            resolve_task_class("NonExistentTask12345")

    def test_short_name_resolves_when_exported_in_tasks_init(self):
        """Short-name resolution should work for tasks exported in sieval.tasks."""
        from sieval.cli.resolution import resolve_task_class

        with _project_sieval_first():
            from sieval.tasks import AIME2024ZeroShotGenTask

            resolved = resolve_task_class("AIME2024ZeroShotGenTask")
        assert resolved is AIME2024ZeroShotGenTask

    def test_short_name_resolves_via_submodule_when_tasks_import_fails(self):
        """If importing `sieval.tasks` fails, submodule search should still resolve."""
        from sieval.cli.resolution import resolve_task_class

        target_cls = type("SyntheticTask", (), {})
        fake_module = types.SimpleNamespace(SyntheticTask=target_cls)

        def _fake_import_module(module_name: str):
            if module_name == "sieval.tasks":
                raise ModuleNotFoundError(
                    "No module named 'sieval.tasks'",
                    name="sieval.tasks",
                )
            if module_name == "sieval.tasks.synthetic_task":
                return fake_module
            raise ModuleNotFoundError(
                f"No module named '{module_name}'", name=module_name
            )

        with (
            patch(
                "sieval.cli.resolution._guess_submodule_names",
                return_value=["synthetic_task"],
            ),
            patch(
                "sieval.cli.resolution.importlib.import_module",
                side_effect=_fake_import_module,
            ),
        ):
            resolved = resolve_task_class("SyntheticTask")

        assert resolved is target_cls

    def test_short_name_propagates_missing_dependency_error(self):
        """If a task module exists but has a missing dependency, propagate the error."""
        from sieval.cli.resolution import resolve_task_class

        def _fake_import_module(module_name: str):
            if module_name == "sieval.tasks":
                # sieval.tasks loads, but __getattr__ triggers import of the
                # task module which fails due to a missing dependency.
                raise ModuleNotFoundError("No module named 'scipy'", name="scipy")
            raise ModuleNotFoundError(
                f"No module named '{module_name}'", name=module_name
            )

        with (
            patch(
                "sieval.cli.resolution.importlib.import_module",
                side_effect=_fake_import_module,
            ),
            pytest.raises(ModuleNotFoundError, match="scipy"),
        ):
            resolve_task_class("SomeTask")


class TestNormalizeInlineModelBinding:
    """Guards on the inline-binding normalizer shared by both entry points."""

    @staticmethod
    def _config(**overrides):
        config = {"model": "gpt-4.1", "api_base": "https://api.openai.com/v1"}
        config.update(overrides)
        return config

    def test_non_string_key_is_rejected(self):
        non_string_keys: dict = {1: "x", "model": "gpt-4.1"}
        with pytest.raises(TypeError, match="keys must be strings"):
            normalize_inline_model_binding("t", "grader", non_string_keys)

    def test_non_mapping_args_is_rejected(self):
        with pytest.raises(ValueError, match="args must be a mapping"):
            normalize_inline_model_binding("t", "grader", self._config(args=[]))

    @pytest.mark.parametrize("model", ["", None, 7])
    def test_missing_or_empty_model_is_rejected(self, model):
        with pytest.raises(ValueError, match="requires a non-empty 'model'"):
            normalize_inline_model_binding("t", "grader", self._config(model=model))

    @pytest.mark.parametrize("dialect", ["", None, 7])
    def test_empty_dialect_is_rejected(self, dialect):
        with pytest.raises(ValueError, match="dialect must be a non-empty string"):
            normalize_inline_model_binding("t", "grader", self._config(dialect=dialect))

    def test_sglang_legacy_bypass_is_rejected(self):
        with pytest.raises(ValueError, match="sglang_legacy"):
            normalize_inline_model_binding(
                "t", "grader", self._config(dialect="sglang_legacy")
            )

    def test_credentials_are_kept_out_of_the_stored_config(self):
        binding = normalize_inline_model_binding(
            "t", "grader", self._config(api_key="sk-secret", args={"api_key": "sk-n"})
        )
        assert "api_key" not in binding.config
        nested_args = binding.config["args"]
        assert isinstance(nested_args, dict)
        assert "api_key" not in nested_args
        assert "sk-secret" not in binding.binding_id
