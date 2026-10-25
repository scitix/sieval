"""Tests for sieval.core.tasks.python_requires.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from importlib import metadata
from types import MappingProxyType

import pytest

from sieval.core.tasks import Task
from sieval.core.tasks.python_requires import (
    MissingPythonRequirementsError,
    UnmetPythonRequirement,
    unmet_python_requires,
    validate_python_requires,
)

# Installed in every environment sieval runs in: a core dependency.
_PRESENT = "packaging"
_ABSENT = "sieval_test_no_such_module_xyz"


class TestValidate:
    def test_parses_each_requirement(self):
        parsed = validate_python_requires({"sklearn": "scikit-learn>=1.3"}, "T")
        assert parsed["sklearn"].name == "scikit-learn"
        assert str(parsed["sklearn"].specifier) == ">=1.3"

    def test_a_dotted_import_name_is_accepted(self):
        assert validate_python_requires({"rdkit.Chem": "rdkit"}, "T")

    def test_a_non_mapping_is_a_type_error(self):
        with pytest.raises(TypeError, match="T python_requires must be a mapping"):
            validate_python_requires(["rdkit"], "T")

    def test_a_non_string_value_is_a_type_error(self):
        with pytest.raises(TypeError, match="must be strings"):
            validate_python_requires({"rdkit": None}, "T")

    def test_a_non_identifier_key_is_rejected(self):
        with pytest.raises(ValueError, match="'scikit-learn' is not an import name"):
            validate_python_requires({"scikit-learn": "scikit-learn"}, "T")

    def test_a_malformed_requirement_names_the_owner(self):
        with pytest.raises(ValueError, match=r"Task 'x' python_requires\['a'\]"):
            validate_python_requires({"a": "a >>= 1"}, "Task 'x'")

    def test_extras_are_rejected(self):
        with pytest.raises(ValueError, match="names extras, which are not checked"):
            validate_python_requires({_PRESENT: f"{_PRESENT}[x]>=1"}, "T")


class TestUnmet:
    def test_an_empty_declaration_is_met(self):
        assert unmet_python_requires({}, "T") == ()

    def test_an_installed_unpinned_package_is_met(self):
        assert unmet_python_requires({_PRESENT: _PRESENT}, "T") == ()

    def test_a_missing_module_is_unmet(self):
        (entry,) = unmet_python_requires({_ABSENT: "nope>=1"}, "T")
        assert entry == UnmetPythonRequirement(_ABSENT, "nope>=1", "not installed")

    def test_a_missing_dotted_module_is_unmet_not_raised(self):
        (entry,) = unmet_python_requires({f"{_ABSENT}.sub": "nope"}, "T")
        assert entry.reason == "not installed"

    def test_a_dotted_name_whose_parent_fails_to_import_is_unmet(
        self, tmp_path, monkeypatch
    ):
        # find_spec imports a dotted name's parents; one raising is not a crash.
        pkg = tmp_path / "sieval_test_broken_parent"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("raise RuntimeError('boom')\n")
        (pkg / "sub.py").write_text("")
        monkeypatch.syspath_prepend(str(tmp_path))
        (entry,) = unmet_python_requires(
            {"sieval_test_broken_parent.sub": "broken"}, "T"
        )
        assert entry.reason == "parent import failed: RuntimeError: boom"

    def test_a_parent_missing_its_own_dependency_is_not_reported_absent(
        self, tmp_path, monkeypatch
    ):
        # The parent exists; what it imports does not. Not "not installed".
        pkg = tmp_path / "sieval_test_parent_missing_dep"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("import sieval_test_no_such_dep_xyz\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        (entry,) = unmet_python_requires(
            {"sieval_test_parent_missing_dep.sub": "p"}, "T"
        )
        assert entry.reason.startswith("parent import failed: ModuleNotFoundError")

    def test_a_satisfied_pin_is_met(self):
        assert unmet_python_requires({_PRESENT: f"{_PRESENT}>=1"}, "T") == ()

    def test_an_unsatisfied_pin_names_the_installed_version(self):
        installed = metadata.version(_PRESENT)
        (entry,) = unmet_python_requires({_PRESENT: f"{_PRESENT}>=9999"}, "T")
        assert entry.reason == f"installed {installed}"

    def test_a_pin_on_an_uninstalled_distribution_is_unmet(self):
        # The module imports, but the named distribution is not what installs it.
        (entry,) = unmet_python_requires({_PRESENT: "no-such-dist-xyz>=1"}, "T")
        assert entry.reason == "no installed distribution 'no-such-dist-xyz'"

    def test_a_marker_excluding_this_interpreter_is_met(self):
        declared = {_ABSENT: 'nope; python_version < "3"'}
        assert unmet_python_requires(declared, "T") == ()

    def test_a_marker_including_this_interpreter_is_checked(self):
        declared = {_ABSENT: 'nope; python_version >= "3"'}
        assert len(unmet_python_requires(declared, "T")) == 1

    def test_str_names_requirement_import_and_reason(self):
        entry = UnmetPythonRequirement("rdkit", "rdkit>=2023.9", "not installed")
        assert str(entry) == "rdkit>=2023.9 [import rdkit]: not installed"


class TestMissingError:
    def test_message_lists_each_task_and_an_install_line(self):
        exc = MissingPythonRequirementsError(
            {
                "a": [UnmetPythonRequirement("rdkit", "rdkit", "not installed")],
                "b": [UnmetPythonRequirement("scipy", "scipy>=1", "installed 0.1")],
            }
        )
        assert str(exc).splitlines() == [
            "Missing Python packages the tasks declare in python_requires:",
            "- a: rdkit [import rdkit]: not installed",
            "- b: scipy>=1 [import scipy]: installed 0.1",
            "Install them, e.g. pip install 'rdkit' 'scipy>=1'",
        ]
        assert exc.task is None

    def test_task_is_set_for_a_single_task(self):
        exc = MissingPythonRequirementsError(
            {"a": [UnmetPythonRequirement("x", "x", "not installed")], "b": []}
        )
        assert list(exc.unmet) == ["a"]
        assert exc.task == "a"

    def test_nothing_unmet_is_a_programming_error(self):
        with pytest.raises(ValueError, match="needs an unmet entry"):
            MissingPythonRequirementsError({"a": []})


class TestTaskDefaults:
    def test_the_base_task_declares_nothing(self):
        assert Task.python_requires == {}
        assert isinstance(Task.python_requires, MappingProxyType)

    def test_the_hook_defaults_to_the_class_attribute(self):
        from sieval.core.tasks import RequirementContext

        class T(Task):
            python_requires = {"rdkit": "rdkit"}

        assert T.python_requires_for(RequirementContext()) == {"rdkit": "rdkit"}
