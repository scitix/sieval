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
import yaml

from sieval.cli.resolution import (
    _guess_submodule_names,
    derive_model_type,
    load_class_from_name,
    load_class_from_path,
    resolve_class,
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
        cls = load_class_from_path("sieval.core.models.model.ModelOutput")
        from sieval.core.models.model import ModelOutput

        assert cls is ModelOutput

    # (class_path, expected_exception, expected_error_pattern)
    @pytest.mark.parametrize(
        "class_path,error_type,error_match",
        [
            ("ModelOutput", ValueError, "Invalid class path"),
            ("nonexistent.module.Foo", ImportError, "Could not import"),
            (
                "sieval.core.models.model.NonExistentClass",
                AttributeError,
                "has no class",
            ),
        ],
    )
    def test_invalid_path_raises(self, class_path, error_type, error_match):
        with pytest.raises(error_type, match=error_match):
            load_class_from_path(class_path)


# ===================================================================
# load_class_from_name
# ===================================================================
class TestLoadClassFromName:
    # (class_name, search_modules, expected_fully_qualified_class_path)
    @pytest.mark.parametrize(
        "class_name,search_modules,expected_path",
        [
            (
                "ModelOutput",
                ["sieval.core.models.model"],
                "sieval.core.models.model.ModelOutput",
            ),
            (
                "ChatModel",
                ["sieval.core.models.model", "sieval.core.models.chat_model"],
                "sieval.core.models.chat_model.ChatModel",
            ),
            (
                "ModelOutput",
                ["nonexistent.module", "sieval.core.models.model"],
                "sieval.core.models.model.ModelOutput",
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
                "sieval.core.models.model.ModelOutput",
                [],
                "sieval.core.models.model.ModelOutput",
            ),
            (
                "ModelOutput",
                ["sieval.core.models.model"],
                "sieval.core.models.model.ModelOutput",
            ),
        ],
    )
    def test_resolve_class(self, class_spec, search_modules, expected_path):
        cls = resolve_class(class_spec, search_modules)
        assert cls is load_class_from_path(expected_path)

    def test_leading_dot_raises_value_error(self):
        # Relative import syntax is explicitly rejected.
        with pytest.raises(ValueError, match="Relative import syntax"):
            resolve_class(".ModelOutput", ["sieval.core.models.model"])


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
    """`derive_model_type` is shared by the eval session and recipe resolution,
    so both reach the same answer for one model. Tested directly because
    `sieval run` calls the function, not the session method."""

    def test_explicit_type_wins(self):
        assert derive_model_type("m", "gen", {}) == "gen"

    def test_defaults_to_chat_with_no_tasks(self):
        assert derive_model_type("m", None, {}) == "chat"

    def test_rejects_non_mapping_tasks_section(self):
        """A list-shaped `tasks:` must not surface as an AttributeError.

        The infer layer reaches this before an EvalSession exists, and full
        config validation only runs under `--dry-run`, so this is the first
        code to touch the section on a normal `sieval run`. The shapes come
        from `yaml.safe_load` rather than a literal because that is how an
        untyped config actually reaches the annotated parameter.
        """
        tasks_cfg = yaml.safe_load("tasks:\n  - arc\n  - hellaswag\n")["tasks"]
        with pytest.raises(ValueError, match="'tasks' configuration must be"):
            derive_model_type("m", None, tasks_cfg)

    def test_rejects_non_mapping_task_entry(self):
        tasks_cfg = yaml.safe_load("tasks:\n  arc: ARCEasyFewShotPplTask\n")["tasks"]
        with pytest.raises(ValueError, match="'tasks.arc' configuration must be"):
            derive_model_type("m", None, tasks_cfg)

    def test_infers_gen_from_task_without_explicit_type(self):
        """The case explicit-only reading would miss: no `type:` in config."""

        class FakeTask:
            model_type = "gen"

        tasks_cfg = {"t1": {"model": "m", "class": "fake.FakeTask"}}
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            return_value=FakeTask,
        ):
            assert derive_model_type("m", None, tasks_cfg) == "gen"

    def test_ignores_tasks_pointing_at_other_models(self):
        class FakeTask:
            model_type = "gen"

        tasks_cfg = {"t1": {"model": "other", "class": "fake.FakeTask"}}
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            return_value=FakeTask,
        ):
            assert derive_model_type("m", None, tasks_cfg) == "chat"

    @pytest.mark.parametrize(
        "error",
        [ImportError("module not found"), AttributeError("class not found")],
    )
    def test_unresolvable_task_class_is_skipped(self, error):
        """Validation reports import errors; derivation must not raise here."""
        tasks_cfg = {"t1": {"model": "m", "class": "missing.Task"}}
        with patch(
            "sieval.cli.resolution.resolve_task_class",
            side_effect=error,
        ):
            assert derive_model_type("m", None, tasks_cfg) == "chat"

    def test_unresolvable_task_does_not_block_resolvable_task(self):
        """One task failing to resolve must not hide a sibling that succeeds."""

        class GenTask:
            model_type = "gen"

        tasks_cfg = {
            "bad_task": {"model": "m", "class": "bad.module.BadTask"},
            "good_task": {"model": "m", "class": "good.module.GenTask"},
        }

        call_count = 0

        def mock_resolve(_spec):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ImportError("bad module")
            return GenTask

        with patch(
            "sieval.cli.resolution.resolve_task_class",
            side_effect=mock_resolve,
        ):
            assert derive_model_type("m", None, tasks_cfg) == "gen"
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
