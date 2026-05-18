"""
Tests for scripts/check_layer_imports.py — layer boundary enforcement.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import sys
from pathlib import Path

# scripts/ is not a package — add it to sys.path so we can import directly.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from check_layer_imports import (  # noqa: E402  # type: ignore[unresolved-import]  # scripts/ added to sys.path at runtime
    _check_file,
    _check_private_access,
    _file_package,
    _get_layer,
    _is_within_subtree,
    main,
)


class TestGetLayer:
    """Extract layer name from paths containing 'sieval'."""

    def test_core_layer(self):
        assert _get_layer(Path("sieval/core/utils/foo.py")) == "core"

    def test_tasks_layer(self):
        assert _get_layer(Path("sieval/tasks/bar.py")) == "tasks"

    def test_datasets_layer(self):
        assert _get_layer(Path("sieval/datasets/some_dataset.py")) == "datasets"

    def test_infer_layer(self):
        assert _get_layer(Path("sieval/infer/runner.py")) == "infer"

    def test_unrelated_path_returns_none(self):
        assert _get_layer(Path("unrelated/foo.py")) is None

    def test_sieval_init_returns_init_filename(self):
        # sieval/__init__.py — parts[idx+1] is "__init__.py"
        result = _get_layer(Path("sieval/__init__.py"))
        assert result == "__init__.py"

    def test_nested_project_path(self):
        assert _get_layer(Path("home/user/project/sieval/core/foo.py")) == "core"

    def test_project_dir_named_sieval(self):
        # Common real case: repo directory is itself named "sieval", so
        # an absolute path contains two "sieval" segments. The package
        # root is the *last* one; picking the first would mis-identify
        # the layer as the package directory itself.
        assert _get_layer(Path("/home/user/sieval/sieval/core/foo.py")) == "core"
        assert (
            _get_layer(Path("/srv/ai-infra/user/sieval/sieval/tasks/core/_base.py"))
            == "tasks"
        )

    def test_repo_named_sieval_outer_sibling_returns_none(self):
        # Repo dir named "sieval" + file in a sibling dir (scripts/, tests/,
        # docs/, ...). Only one "sieval" segment in the path — the repo dir,
        # not the package root. Must return None, not mis-classify the
        # sibling as a sieval layer.
        assert _get_layer(Path("/home/user/sieval/scripts/foo.py")) is None
        assert _get_layer(Path("/home/user/sieval/tests/unit/foo.py")) is None
        assert _get_layer(Path("/home/user/sieval/docs/designs/foo.py")) is None

    def test_bare_sieval_dir_returns_none(self):
        # Path("sieval") has parts ("sieval",), so idx+1 is out of bounds
        assert _get_layer(Path("sieval")) is None


class TestCheckFile:
    """Core violation detection logic."""

    def test_core_importing_tasks_is_error(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.tasks\n")
        errors = _check_file(f)
        assert len(errors) == 1
        assert "core/ must not import tasks/" in errors[0]

    def test_core_importing_core_utils_is_allowed(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "ok.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.core.utils\n")
        assert _check_file(f) == []

    def test_tasks_importing_datasets_is_allowed(self, tmp_path: Path):
        f = tmp_path / "sieval" / "tasks" / "ok.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.datasets\n")
        assert _check_file(f) == []

    def test_tasks_importing_infer_is_error(self, tmp_path: Path):
        f = tmp_path / "sieval" / "tasks" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.infer\n")
        errors = _check_file(f)
        assert len(errors) == 1
        assert "tasks/ must not import infer/" in errors[0]

    def test_cli_importing_anything_is_allowed(self, tmp_path: Path):
        f = tmp_path / "sieval" / "cli" / "ok.py"
        f.parent.mkdir(parents=True)
        f.write_text(
            "import sieval.core\nimport sieval.tasks\n"
            "import sieval.datasets\nimport sieval.infer\n"
        )
        assert _check_file(f) == []

    def test_import_from_detected(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("from sieval.tasks import something\n")
        errors = _check_file(f)
        assert len(errors) == 1
        assert "core/ must not import tasks/" in errors[0]

    def test_syntax_error_returns_empty(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "broken.py"
        f.parent.mkdir(parents=True)
        f.write_text("def broken(\n")
        assert _check_file(f) == []

    def test_non_sieval_file_returns_empty(self, tmp_path: Path):
        f = tmp_path / "other" / "module.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.tasks\n")
        assert _check_file(f) == []

    def test_datasets_importing_tasks_is_error(self, tmp_path: Path):
        f = tmp_path / "sieval" / "datasets" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("from sieval.tasks import registry\n")
        errors = _check_file(f)
        assert len(errors) == 1
        assert "datasets/ must not import tasks/" in errors[0]

    def test_infer_importing_datasets_is_error(self, tmp_path: Path):
        f = tmp_path / "sieval" / "infer" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.datasets.loader\n")
        errors = _check_file(f)
        assert len(errors) == 1
        assert "infer/ must not import datasets/" in errors[0]

    def test_multiple_violations_in_one_file(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.tasks\nfrom sieval.cli import app\n")
        errors = _check_file(f)
        assert len(errors) == 2

    def test_error_includes_lineno(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1\nimport sieval.tasks\n")
        errors = _check_file(f)
        assert ":2:" in errors[0]


class TestMain:
    """Integration tests for the main() entry point."""

    def test_no_args_returns_zero(self):
        assert main([]) == 0

    def test_clean_files_return_zero(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "ok.py"
        f.parent.mkdir(parents=True)
        f.write_text("import os\n")
        assert main([str(f)]) == 0

    def test_violation_returns_one(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.tasks\n")
        assert main([str(f)]) == 1

    def test_non_py_files_ignored(self, tmp_path: Path):
        f = tmp_path / "sieval" / "core" / "data.json"
        f.parent.mkdir(parents=True)
        f.write_text('{"key": "value"}')
        assert main([str(f)]) == 0

    def test_mixed_files(self, tmp_path: Path):
        good = tmp_path / "sieval" / "tasks" / "ok.py"
        good.parent.mkdir(parents=True)
        good.write_text("import sieval.core\n")
        bad = tmp_path / "sieval" / "core" / "bad.py"
        bad.parent.mkdir(parents=True)
        bad.write_text("import sieval.tasks\n")
        assert main([str(good), str(bad)]) == 1

    def test_errors_printed_to_stderr(self, tmp_path: Path, capsys):
        f = tmp_path / "sieval" / "core" / "bad.py"
        f.parent.mkdir(parents=True)
        f.write_text("import sieval.tasks\n")
        main([str(f)])
        captured = capsys.readouterr()
        assert "core/ must not import tasks/" in captured.err

    def test_stdin_mode(self, tmp_path: Path, monkeypatch):
        bad = tmp_path / "sieval" / "core" / "bad.py"
        bad.parent.mkdir(parents=True)
        bad.write_text("import sieval.tasks\n")
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(f"{bad}\n"))
        assert main(["--stdin"]) == 1


class TestFilePackage:
    """Dotted-package derivation for the current file."""

    def test_top_level_module(self):
        assert _file_package(Path("sieval/core/foo.py")) == "sieval.core"

    def test_nested_module(self):
        assert (
            _file_package(Path("sieval/cli/leaderboard/session.py"))
            == "sieval.cli.leaderboard"
        )

    def test_init_file(self):
        assert _file_package(Path("sieval/tasks/__init__.py")) == "sieval.tasks"

    def test_outside_sieval_returns_empty(self):
        assert _file_package(Path("scripts/foo.py")) == ""

    def test_project_dir_named_sieval(self):
        # Regression: repo dir and package share the name "sieval".
        # Take the *last* occurrence as the package root.
        assert (
            _file_package(Path("/home/user/sieval/sieval/tasks/gue/foo.py"))
            == "sieval.tasks.gue"
        )

    def test_repo_named_sieval_sibling_returns_empty(self):
        # `/repo-sieval/scripts/foo.py` — the sieval segment is the repo dir,
        # not the package root. File_pkg must be "" so private-access checks
        # don't treat the file as `sieval.scripts.*` (no such package).
        assert _file_package(Path("/home/user/sieval/scripts/foo.py")) == ""
        assert _file_package(Path("/home/user/sieval/tests/unit/foo.py")) == ""


class TestIsWithinSubtree:
    """Subtree containment check (same-package or descendant)."""

    def test_same_package(self):
        assert _is_within_subtree("sieval.tasks", "sieval.tasks")

    def test_descendant(self):
        assert _is_within_subtree("sieval.tasks.beacon", "sieval.tasks")

    def test_deep_descendant(self):
        assert _is_within_subtree("sieval.tasks.beacon.sub.x", "sieval.tasks")

    def test_sibling_subpackage(self):
        # beacon is NOT a descendant of glrb — even though both live under tasks
        assert not _is_within_subtree("sieval.tasks.beacon", "sieval.tasks.glrb")

    def test_outside_subtree(self):
        assert not _is_within_subtree("sieval.cli", "sieval.tasks")

    def test_prefix_is_not_descendant(self):
        # "sieval.tasks_misc" happens to start with "sieval.tasks" as a string
        # but is NOT a descendant — must check the "." boundary.
        assert not _is_within_subtree("sieval.tasks_misc", "sieval.tasks")

    def test_empty_root_never_matches(self):
        assert not _is_within_subtree("sieval.anything", "")


class TestCheckPrivateAccess:
    """Rule 2: imports imply public + private modules are protected."""

    def _write(self, tmp_path: Path, rel: str, src: str) -> Path:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(src)
        return f

    # ── Private name across modules ──

    def test_cross_package_private_name_flagged(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "sieval/core/tasks/meta.py",
            "from sieval.core.datasets.meta import _X\n",
        )
        errors = _check_file(f)
        assert any(
            "import of private name '_X' from 'sieval.core.datasets.meta'" in e
            for e in errors
        )

    def test_public_name_from_public_module_is_clean(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "sieval/core/tasks/meta.py",
            "from sieval.core.datasets.meta import extract_sample_type\n",
        )
        assert _check_file(f) == []

    def test_relative_import_of_private_name_skipped(self, tmp_path: Path):
        # Same-package relative imports are the carve-out.
        f = self._write(
            tmp_path,
            "sieval/core/datasets/loader.py",
            "from .meta import _X\n",
        )
        assert _check_file(f) == []

    def test_dunder_name_allowed(self, tmp_path: Path):
        # __all__, __version__ etc. are Python conventions, not project-privates.
        f = self._write(
            tmp_path,
            "sieval/core/tasks/meta.py",
            "from sieval.core.datasets.meta import __all__\n",
        )
        assert _check_file(f) == []

    def test_wildcard_import_not_treated_as_private_name(self, tmp_path: Path):
        # `from x import *` — alias.name is "*", which starts with neither
        # `_` nor `__`. Should not trigger the private-name check.
        f = self._write(
            tmp_path,
            "sieval/core/tasks/meta.py",
            "from sieval.core.datasets.meta import *\n",
        )
        assert _check_file(f) == []

    def test_multiple_names_each_checked(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "sieval/core/tasks/meta.py",
            "from sieval.core.datasets.meta import _A, B, _C\n",
        )
        errors = _check_file(f)
        names_in_errors = [e for e in errors if "import of private name" in e]
        assert len(names_in_errors) == 2
        assert any("'_A'" in e for e in names_in_errors)
        assert any("'_C'" in e for e in names_in_errors)

    # ── Protected module subtree ──

    def test_descendant_can_reach_ancestor_private_module(self, tmp_path: Path):
        # `sieval/tasks/beacon/foo.py` importing `sieval.tasks._parse_utils` is
        # legal: beacon is a descendant of tasks.
        f = self._write(
            tmp_path,
            "sieval/tasks/beacon/foo.py",
            "from sieval.tasks._parse_utils import extract_boxed\n",
        )
        assert _check_file(f) == []

    def test_deep_descendant_can_reach_ancestor_private(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "sieval/tasks/beacon/sub/deep.py",
            "from sieval.tasks._parse_utils import extract_boxed\n",
        )
        assert _check_file(f) == []

    def test_out_of_subtree_access_flagged(self, tmp_path: Path):
        # cli is not a descendant of tasks.
        f = self._write(
            tmp_path,
            "sieval/cli/run.py",
            "from sieval.tasks._parse_utils import extract_boxed\n",
        )
        errors = _check_file(f)
        assert any(
            "import from private module 'sieval.tasks._parse_utils' outside its subtree"
            in e
            for e in errors
        )

    def test_sibling_subpackage_access_flagged(self, tmp_path: Path):
        # beacon and glrb are siblings under tasks; neither is a descendant of
        # the other.
        f = self._write(
            tmp_path,
            "sieval/tasks/beacon/foo.py",
            "from sieval.tasks.glrb._base import X\n",
        )
        errors = _check_file(f)
        assert any(
            "import from private module 'sieval.tasks.glrb._base' outside its subtree"
            in e
            for e in errors
        )

    def test_same_package_sibling_private_module_allowed(self, tmp_path: Path):
        # Relative import from sibling `_base` in same package — the
        # documented `_base.py` pattern.
        f = self._write(
            tmp_path,
            "sieval/tasks/beacon/foo.py",
            "from ._base import BeaconTask\n",
        )
        assert _check_file(f) == []

    def test_import_statement_private_module_flagged(self, tmp_path: Path):
        # `import sieval.x._foo` (ast.Import node, not ast.ImportFrom)
        f = self._write(
            tmp_path,
            "sieval/cli/run.py",
            "import sieval.tasks._parse_utils\n",
        )
        errors = _check_file(f)
        assert any(
            "import from private module 'sieval.tasks._parse_utils' outside its subtree"
            in e
            for e in errors
        )

    # ── Tests carve-out ──

    def test_test_file_exempt(self, tmp_path: Path):
        # Tests are the documented carve-out. Covers both layouts:
        #   (a) `<repo>/tests/...` — no `sieval` segment at all → short-
        #       circuits at the outside-sieval check.
        #   (b) `<repo-named-sieval>/tests/...` — has a `sieval` segment,
        #       but `_get_layer` treats `tests/` as an outer sibling and
        #       returns None, so the same short-circuit applies.
        # Without the carve-out, both imports in the body would be flagged
        # (private-name + protected-subtree) — confirms discriminating power.
        f = self._write(
            tmp_path,
            "sieval/tests/unit/core/test_meta.py",
            "from sieval.core.datasets.meta import _INTERNAL_STATE\n"
            "from sieval.tasks._parse_utils import extract_boxed\n",
        )
        import ast as _ast

        assert _get_layer(f) is None
        assert "tests" in f.parts
        assert _check_private_access(f, _ast.parse(f.read_text())) == []

    def test_nested_sieval_layer_tests_still_exempt(self, tmp_path: Path):
        # Edge case: a test file nested inside a sieval subpackage
        # (`sieval/tasks/tests/test_x.py`). `_get_layer` returns the real
        # layer ("tasks"), so the top-level short-circuit doesn't fire —
        # the explicit `"tests" in path.parts` branch must carry it.
        f = self._write(
            tmp_path,
            "sieval/tasks/tests/test_x.py",
            "from sieval.core.datasets.meta import _INTERNAL_STATE\n",
        )
        import ast as _ast

        assert _get_layer(f) == "tasks"
        assert "tests" in f.parts
        assert _check_private_access(f, _ast.parse(f.read_text())) == []

    # ── scripts/ enforcement ──

    def test_scripts_file_subject_to_private_name_rule(self, tmp_path: Path):
        # scripts/ is tooling, not a package, but still follows the policy —
        # otherwise a script could quietly reach into sieval internals and
        # erode the contract the rule is there to protect.
        f = self._write(
            tmp_path,
            "scripts/tool.py",
            "from sieval.core.datasets.meta import _DATASET_REGISTRY\n",
        )
        errors = _check_file(f)
        assert any("import of private name '_DATASET_REGISTRY'" in e for e in errors)

    def test_scripts_file_subject_to_protected_module_rule(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "scripts/tool.py",
            "from sieval.tasks._parse_utils import extract_boxed\n",
        )
        errors = _check_file(f)
        assert any(
            "import from private module 'sieval.tasks._parse_utils'" in e
            for e in errors
        )

    def test_scripts_file_public_imports_ok(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "scripts/tool.py",
            "from sieval.core.datasets.meta import DATASET_REGISTRY\n",
        )
        assert _check_file(f) == []

    def test_nested_scripts_subdir_flagged(self, tmp_path: Path):
        # Scripts can be organised into subdirs; the rule must still bite.
        # A naive rewrite to `path.parts[0] == "scripts"` would miss this.
        f = self._write(
            tmp_path,
            "scripts/checks/tool.py",
            "from sieval.tasks._parse_utils import extract_boxed\n",
        )
        errors = _check_file(f)
        assert any(
            "import from private module 'sieval.tasks._parse_utils'" in e
            for e in errors
        )

    def test_absolute_path_to_scripts_file_classified_as_scripts(self, tmp_path: Path):
        # When the repo directory is itself named `sieval` and a pre-commit
        # invocation hands the checker an absolute path to a scripts/ file,
        # the earlier last-sieval heuristic mis-classified it as layer
        # `sieval.scripts` (no such package) — functionally still caught the
        # violation, but the error message was inaccurate and a future
        # `sieval/scripts/` subpackage would become a loophole.
        repo = tmp_path / "sieval"  # repo dir shares the package name
        f = repo / "scripts" / "tool.py"
        f.parent.mkdir(parents=True)
        f.write_text("from sieval.tasks._parse_utils import extract_boxed\n")
        errors = _check_file(f)
        # Violation still flagged…
        assert any("sieval.tasks._parse_utils" in e for e in errors)
        # …and the importer is reported as scripts tooling (empty pkg),
        # not as a fictitious `sieval.scripts` sublayer.
        assert not any("importer at 'sieval.scripts'" in e for e in errors)
        assert any("importer at ''" in e for e in errors)

    def test_truly_outside_sieval_and_scripts_returns_empty(self, tmp_path: Path):
        # A file in neither sieval/ nor scripts/ is out of scope.
        f = self._write(tmp_path, "misc/tool.py", "from sieval.x import _y\n")
        assert _check_file(f) == []

    # ── Top-level sieval package ──

    def test_from_sieval_import_private_flagged(self, tmp_path: Path):
        # Bare `from sieval import _x` must also trigger the private-name
        # rule — logical parity with `from sieval.pkg import _x`.
        f = self._write(
            tmp_path,
            "sieval/cli/run.py",
            "from sieval import _internal\n",
        )
        errors = _check_file(f)
        assert any(
            "import of private name '_internal' from 'sieval'" in e for e in errors
        )

    def test_from_sieval_import_public_ok(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "sieval/cli/run.py",
            "from sieval import core\n",
        )
        assert _check_file(f) == []

    # ── Edge cases ──

    def test_third_party_import_ignored(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            "sieval/core/tasks/meta.py",
            "from collections import _tuplegetter\n",  # CPython internal name
        )
        # We only police sieval.* imports.
        assert _check_file(f) == []

    def test_both_rules_fire_on_same_import(self, tmp_path: Path):
        # Both: (a) importing private NAME (_X), (b) from private MODULE
        # (_hidden) that's outside the importer's subtree.
        f = self._write(
            tmp_path,
            "sieval/cli/run.py",
            "from sieval.tasks._hidden import _X\n",
        )
        errors = _check_file(f)
        assert any("import of private name '_X'" in e for e in errors)
        assert any(
            "import from private module 'sieval.tasks._hidden'" in e for e in errors
        )
