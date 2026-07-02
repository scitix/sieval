"""
Tests for scripts/check_preflight.py — CheckResult, formatting, and PreflightRunner.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# scripts/ is not a package — add it to sys.path so we can import directly.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from check_preflight import (  # noqa: E402  # type: ignore[unresolved-import]  # scripts/ added to sys.path at runtime
    CheckResult,
    PreflightRunner,
    _dataset_integrity_violations,
    format_json,
    format_text,
    main,
)


class TestCheckResult:
    """Construction and field access."""

    def test_pass_construction(self):
        r = CheckResult(status="PASS", check="check_links", message="all good")
        assert r.status == "PASS"
        assert r.check == "check_links"
        assert r.message == "all good"
        assert r.details == []

    def test_fail_construction_with_details(self):
        r = CheckResult(
            status="FAIL",
            check="check_deps",
            message="missing dep",
            details=["foo>=1.0 not found"],
        )
        assert r.status == "FAIL"
        assert r.details == ["foo>=1.0 not found"]

    def test_has_failure_detection(self):
        results = [
            CheckResult(status="PASS", check="a", message="ok"),
            CheckResult(status="FAIL", check="b", message="bad"),
            CheckResult(status="WARN", check="c", message="meh"),
        ]
        statuses = [r.status for r in results]
        assert "FAIL" in statuses
        # A list with no FAIL should not trigger failure
        ok_results = [
            CheckResult(status="PASS", check="a", message="ok"),
            CheckResult(status="SKIP", check="b", message="skipped"),
        ]
        assert "FAIL" not in [r.status for r in ok_results]


class TestFormatText:
    """Text output formatting."""

    def test_basic_format(self):
        results = [CheckResult(status="PASS", check="check_links", message="ok")]
        text = format_text(results)
        assert "[PASS] check_links — ok" in text

    def test_details_indented(self):
        results = [
            CheckResult(
                status="FAIL",
                check="check_deps",
                message="problems",
                details=["line1", "line2"],
            )
        ]
        text = format_text(results)
        assert "[FAIL] check_deps — problems" in text
        assert "  line1" in text
        assert "  line2" in text

    def test_multiple_results(self):
        results = [
            CheckResult(status="PASS", check="a", message="ok"),
            CheckResult(status="WARN", check="b", message="hmm"),
        ]
        text = format_text(results)
        assert "[PASS]" in text
        assert "[WARN]" in text


class TestFormatJson:
    """JSON output formatting."""

    def test_json_structure(self):
        results = [
            CheckResult(
                status="FAIL", check="check_deps", message="bad", details=["d1"]
            )
        ]
        raw = format_json(results)
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 1
        obj = data[0]
        assert obj["status"] == "FAIL"
        assert obj["check"] == "check_deps"
        assert obj["message"] == "bad"
        assert obj["details"] == ["d1"]

    def test_empty_details_default(self):
        results = [CheckResult(status="PASS", check="a", message="ok")]
        data = json.loads(format_json(results))
        assert data[0]["details"] == []


class TestPreflightRunner:
    """Runner orchestration."""

    def test_all_checks_listed(self):
        assert len(PreflightRunner.ALL_CHECKS) == 9
        assert "check_links" in PreflightRunner.ALL_CHECKS
        assert "check_examples" in PreflightRunner.ALL_CHECKS
        assert "check_meta_index_sync" in PreflightRunner.ALL_CHECKS
        assert "check_version" in PreflightRunner.ALL_CHECKS

    def test_run_all_returns_results(self):
        runner = PreflightRunner()
        results = runner.run()
        assert len(results) >= 9
        assert any(r.check == "check_links" for r in results)
        assert any(r.check == "check_deps" for r in results)
        assert any(r.check == "check_examples" for r in results)
        assert any(r.check == "check_meta_index_sync" for r in results)
        assert any(r.check == "check_version" for r in results)
        assert any(r.check == "check_imports" for r in results)

    def test_run_single_check(self):
        runner = PreflightRunner()
        results = runner.run(only="check_links")
        assert len(results) >= 1
        assert all(r.check == "check_links" for r in results)

    def test_run_unknown_check_raises(self):
        runner = PreflightRunner()
        with pytest.raises(ValueError, match="Unknown check"):
            runner.run(only="check_nonexistent")

    def test_project_root_default(self):
        runner = PreflightRunner()
        # project_root should be two levels up from scripts/check_preflight.py
        assert runner.project_root.is_dir()


class TestMainCLI:
    """CLI entry point."""

    def test_main_text_output(self, capsys):
        code = main(["--format", "text"])
        captured = capsys.readouterr()
        assert "[SKIP]" in captured.out or "[PASS]" in captured.out
        assert code in (0, 1)  # may FAIL due to real check findings

    def test_main_json_output(self, capsys):
        code = main(["--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert code in (0, 1)  # may FAIL due to real check findings

    def test_main_single_check(self, capsys):
        code = main(["--check", "check_links", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) >= 1
        assert all(d["check"] == "check_links" for d in data)
        assert code in (0, 1)  # may FAIL due to broken links in real repo


class TestCheckVersion:
    """Tests for check_version and its helpers."""

    def test_changelog_version_extracted(self, tmp_path: Path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [1.2.3] - 2026-04-01\n\n### Added\n- stuff\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        assert runner._parse_changelog_version() == "1.2.3"

    def test_changelog_missing(self, tmp_path: Path):
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_version()
        statuses = [r.status for r in results]
        assert "FAIL" in statuses

    def test_dockerfile_version_extracted(self, tmp_path: Path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text(
            "FROM python:3.12\nCOPY ./dist/sieval-2.0.1-py3-none-any.whl /tmp/\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        assert runner._parse_dockerfile_version() == "2.0.1"

    def test_version_mismatch_changelog_vs_dockerfile(self, tmp_path: Path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [1.0.0] - 2026-01-01\n")
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("COPY ./dist/sieval-2.0.0-py3-none-any.whl /tmp/\n")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_version()
        fail_msgs = [r.message for r in results if r.status == "FAIL"]
        assert any("Dockerfile" in m for m in fail_msgs)

    def test_changelog_compare_link(self, tmp_path: Path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n"
            "[1.0.0]: https://github.com/scitix/sieval/compare/v0.9.0...v1.0.0\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_version()
        # Should not have a FAIL or WARN about compare link
        compare_results = [r for r in results if "compare link" in r.message.lower()]
        assert all(r.status != "FAIL" for r in compare_results)

    def test_changelog_missing_compare_link(self, tmp_path: Path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [1.0.0] - 2026-01-01\n")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_version()
        warn_msgs = [r.message for r in results if r.status == "WARN"]
        assert any("compare link" in m.lower() for m in warn_msgs)


class TestCheckImports:
    """Tests for check_imports wrapping check_layer_imports.py."""

    @staticmethod
    def _make_layers_side_effect(layer_result):
        """Return a side_effect that lets git-ls-files fall through (FileNotFoundError)
        while returning *layer_result* for the check_layer_imports subprocess call."""

        def _side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if cmd and cmd[0] == "git":
                raise FileNotFoundError("git")
            return layer_result

        return _side_effect

    def test_pass_when_script_exits_zero(self, tmp_path: Path):
        # Create script path so the exists() check passes
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "check_layer_imports.py").write_text("")
        # Create sieval dir with a .py file
        sieval_dir = tmp_path / "sieval"
        sieval_dir.mkdir()
        (sieval_dir / "example.py").write_text("")

        runner = PreflightRunner(project_root=tmp_path)
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        side_effect = self._make_layers_side_effect(mock_result)
        with patch("check_preflight.subprocess.run", side_effect=side_effect):
            results = runner.check_imports()

        assert len(results) == 1
        assert results[0].status == "PASS"
        assert results[0].check == "check_imports"
        assert "no import-policy violations" in results[0].message

    def test_fail_when_script_exits_nonzero(self, tmp_path: Path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "check_layer_imports.py").write_text("")
        sieval_dir = tmp_path / "sieval"
        sieval_dir.mkdir()
        (sieval_dir / "example.py").write_text("")

        runner = PreflightRunner(project_root=tmp_path)
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="core/ must not import tasks/ (sieval.tasks)\n"
            "core/ must not import cli/ (sieval.cli)\n",
        )
        side_effect = self._make_layers_side_effect(mock_result)
        with patch("check_preflight.subprocess.run", side_effect=side_effect):
            results = runner.check_imports()

        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert results[0].check == "check_imports"
        assert "2 import-policy violation(s)" in results[0].message
        assert len(results[0].details) == 2

    def test_fail_when_script_not_found(self, tmp_path: Path):
        # tmp_path has no scripts/ directory at all
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_imports()

        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert results[0].check == "check_imports"
        assert "not found" in results[0].message

    def test_forwards_both_sieval_and_scripts_files(self, tmp_path: Path):
        # Pre-commit's `files:` filter is `^(sieval|scripts)/`. The preflight
        # wrapper must feed the same scope, otherwise the checker's
        # `in_scripts` branch is exercised by pre-commit but silently dead
        # in `sieval preflight` — two enforcement surfaces diverging.
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "check_layer_imports.py").write_text("")
        (scripts_dir / "tool.py").write_text("")
        sieval_dir = tmp_path / "sieval"
        sieval_dir.mkdir()
        (sieval_dir / "example.py").write_text("")
        # Unrelated tracked files must NOT be forwarded.
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "t.py").write_text("")

        runner = PreflightRunner(project_root=tmp_path)
        captured: dict[str, str] = {}

        def _side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if cmd and cmd[0] == "git":
                raise FileNotFoundError("git")
            # Second call: the checker subprocess — capture its stdin.
            captured["input"] = kwargs.get("input", "")
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )

        with patch("check_preflight.subprocess.run", side_effect=_side_effect):
            runner.check_imports()

        forwarded = captured["input"].splitlines()
        assert any(p.endswith("scripts/tool.py") for p in forwarded)
        assert any(p.endswith("sieval/example.py") for p in forwarded)
        # scripts/check_layer_imports.py is the checker itself — it's .py and
        # lives under scripts/, so it's legitimately in-scope and forwarded.
        # tests/t.py must NOT be forwarded (out of scope).
        assert not any(p.endswith("tests/t.py") for p in forwarded)


class TestCheckLinks:
    """Tests for check_links — URL extraction, permanent links, relative links."""

    def test_valid_urls_pass(self, tmp_path):
        (tmp_path / "README.md").write_text("Check [docs](https://example.com/page)\n")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert not any(r.status == "FAIL" for r in results)

    def test_non_permanent_github_link_warns(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "See [code](https://github.com/scitix/sieval/blob/main/README.md)\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert any(
            r.status == "WARN" and "permanent" in r.message.lower() for r in results
        )

    def test_permanent_github_link_passes(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "See [code](https://github.com/scitix/sieval/blob/abc123def/README.md)\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert not any(
            r.status == "WARN" and "permanent" in r.message.lower() for r in results
        )

    def test_relative_link_to_existing_file_passes(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide\n")
        (tmp_path / "README.md").write_text("See [guide](docs/guide.md)\n")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert not any(
            r.status == "FAIL" and "guide.md" in str(r.details) for r in results
        )

    def test_relative_link_to_missing_file_fails(self, tmp_path):
        (tmp_path / "README.md").write_text("See [guide](docs/nonexistent.md)\n")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert any(
            r.status == "FAIL" and "nonexistent" in str(r.details) for r in results
        )

    def test_no_md_files_passes(self, tmp_path):
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert any(r.status in ("PASS", "SKIP") for r in results)

    def test_docstring_urls_extracted(self, tmp_path):
        sieval_dir = tmp_path / "sieval"
        sieval_dir.mkdir()
        (sieval_dir / "example.py").write_text(
            '"""Module doc.\n\nSee https://github.com/scitix/sieval/blob/master/foo.py\n"""\n'
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert any(
            r.status == "WARN" and "permanent" in r.message.lower() for r in results
        )

    def test_reference_impl_url_included_in_scan(self, tmp_path, monkeypatch):
        """reference_impl URLs from the task registry are scanned by check_links."""
        from sieval.core.tasks.meta import EvalMode, ReferenceImpl, TaskMeta

        fake_meta = TaskMeta(
            name="fake",
            display_name="Fake",
            description="fake",
            dataset="fake_ds",
            eval_mode=EvalMode.GEN,
            reference_impl=ReferenceImpl(
                source="upstream",
                url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py",
            ),
        )

        monkeypatch.setattr("sieval.load_index", lambda: ([], [fake_meta]))

        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()

        assert not any(r.status == "FAIL" for r in results)
        assert any(r.status == "PASS" for r in results)

    def test_reference_impl_unreachable_url_warns_in_deep_mode(
        self, tmp_path, monkeypatch
    ):
        """Deep-mode preflight surfaces 404 on reference_impl URLs as WARN."""
        from sieval.core.tasks.meta import EvalMode, ReferenceImpl, TaskMeta

        fake_meta = TaskMeta(
            name="fake",
            display_name="Fake",
            description="fake",
            dataset="fake_ds",
            eval_mode=EvalMode.GEN,
            reference_impl=ReferenceImpl(
                source="upstream",
                url="https://github.com/fake/repo/blob/abc123def4567890/missing.py",
            ),
        )

        monkeypatch.setattr("sieval.load_index", lambda: ([], [fake_meta]))

        # Mock httpx.AsyncClient.head to return 404
        import httpx

        async def fake_head(_self, url, **_kwargs):
            return httpx.Response(404, request=httpx.Request("HEAD", url))

        monkeypatch.setattr(httpx.AsyncClient, "head", fake_head)

        runner = PreflightRunner(project_root=tmp_path, level="deep")
        results = runner.check_links()

        warn = next(
            (
                r
                for r in results
                if r.status == "WARN" and "unreachable" in r.message.lower()
            ),
            None,
        )
        assert warn is not None, f"expected 'unreachable' WARN, got {results!r}"
        assert any("missing.py" in d for d in warn.details), warn.details

    def test_reference_impl_non_permanent_url_warns(self, tmp_path, monkeypatch):
        """Non-permanent GitHub URL in registry still triggers _GH_NON_PERMANENT WARN.

        Guards against regression if _validate() is ever loosened. Bypasses the
        normal import-time validation by constructing ReferenceImpl directly and
        stubbing the iterator — mirrors what would happen if a mutable-ref URL
        slipped past validation.
        """
        from sieval.core.tasks.meta import EvalMode, ReferenceImpl, TaskMeta

        fake_meta = TaskMeta(
            name="fake",
            display_name="Fake",
            description="fake",
            dataset="fake_ds",
            eval_mode=EvalMode.GEN,
            reference_impl=ReferenceImpl(
                source="upstream",
                url="https://github.com/fake/repo/blob/main/foo.py",
            ),
        )

        monkeypatch.setattr("sieval.load_index", lambda: ([], [fake_meta]))

        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()

        warn = next(
            (
                r
                for r in results
                if r.status == "WARN" and "permanent" in r.message.lower()
            ),
            None,
        )
        assert warn is not None, f"expected 'permanent' WARN, got {results!r}"
        assert any("blob/main/foo.py" in d for d in warn.details), warn.details

    def test_no_reference_impl_urls_is_noop(self, tmp_path, monkeypatch):
        """Empty index adds no URLs; check_links behaves as before."""
        monkeypatch.setattr("sieval.load_index", lambda: ([], []))

        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()

        # No FAILs and no WARNs: empty index should not introduce any new
        # preflight signals beyond the baseline SKIP/PASS produced by an
        # empty tmp_path scan.
        assert not any(r.status in ("FAIL", "WARN") for r in results)

    def test_index_load_error_surfaces_as_warn(self, tmp_path, monkeypatch):
        """A broken/missing index.json produces WARN, not a silent skip.

        ``check_links`` now reads from ``sieval.load_index``; if that fails
        (e.g. missing index file, schema_version mismatch), preflight must
        still run the rest of its checks but flag the scan gap.
        """

        def _raise():
            raise RuntimeError("index.json schema_version=99 not supported")

        monkeypatch.setattr("sieval.load_index", _raise)

        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()

        warn = next(
            (
                r
                for r in results
                if r.status == "WARN" and "index load skipped" in r.message
            ),
            None,
        )
        assert warn is not None, f"expected index-load WARN, got {results!r}"

    def test_registry_url_deduped_against_docstring(self, tmp_path, monkeypatch):
        """A URL appearing in both docstring and registry is counted only once."""
        from sieval.core.tasks.meta import EvalMode, ReferenceImpl, TaskMeta

        shared_url = "https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py"

        # Create a .py file under sieval/ whose docstring contains the same URL.
        sieval_dir = tmp_path / "sieval"
        sieval_dir.mkdir()
        py_file = sieval_dir / "stub.py"
        py_file.write_text(
            f'"""Module docstring.\n\nSee {shared_url}\n"""\n',
            encoding="utf-8",
        )

        # Fake git ls-files to include this .py file.
        monkeypatch.setattr(
            PreflightRunner,
            "_git_tracked_files",
            lambda self, ext: [py_file] if ext == ".py" else [],
        )

        fake_meta = TaskMeta(
            name="fake",
            display_name="Fake",
            description="fake",
            dataset="fake_ds",
            eval_mode=EvalMode.GEN,
            reference_impl=ReferenceImpl(source="upstream", url=shared_url),
        )

        monkeypatch.setattr("sieval.load_index", lambda: ([], [fake_meta]))

        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()

        # The PASS summary reports scanned URL count; with dedup, the shared
        # URL should be counted once (from docstring), not twice.
        pass_result = next(
            (r for r in results if r.status == "PASS" and "scanned" in r.message),
            None,
        )
        assert pass_result is not None, f"expected PASS summary, got {results!r}"
        assert "1 URL(s)" in pass_result.message, pass_result.message


class TestCheckDeps:
    """Tests for check_deps — lock file consistency and optional groups."""

    def _write_pyproject(
        self,
        tmp_path: Path,
        optional_deps: dict[str, list[str]] | None = None,
    ) -> None:
        lines = ['[project]\nname = "sieval"\ndependencies = ["httpx>=0.28"]\n']
        if optional_deps:
            lines.append("[project.optional-dependencies]\n")
            for group, deps in optional_deps.items():
                deps_str = ", ".join(f'"{d}"' for d in deps)
                lines.append(f"{group} = [{deps_str}]\n")
        (tmp_path / "pyproject.toml").write_text("".join(lines))

    def _write_lockfile(
        self, tmp_path: Path, content_hash: str = "sha256:abc123"
    ) -> None:
        (tmp_path / "pdm.lock").write_text(
            f'[metadata]\ncontent_hash = "{content_hash}"\n'
        )

    def test_pass_with_valid_setup(self, tmp_path: Path):
        groups = {"dev": ["pytest>=7"], "gpu": ["torch>=2"]}
        self._write_pyproject(tmp_path, optional_deps=groups)
        self._write_lockfile(tmp_path)
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        statuses = [r.status for r in results]
        assert "FAIL" not in statuses
        assert statuses.count("PASS") >= 2  # optional-deps OK + lockfile OK

    def test_fail_missing_lockfile(self, tmp_path: Path):
        self._write_pyproject(tmp_path, optional_deps={"dev": ["pytest>=7"]})
        # no pdm.lock
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        fail_msgs = [r.message for r in results if r.status == "FAIL"]
        assert any("pdm.lock not found" in m for m in fail_msgs)

    def test_fail_empty_optional_group(self, tmp_path: Path):
        groups = {"dev": ["pytest>=7"], "empty": []}
        self._write_pyproject(tmp_path, optional_deps=groups)
        self._write_lockfile(tmp_path)
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        fail_msgs = [r.message for r in results if r.status == "FAIL"]
        assert any("empty" in m for m in fail_msgs)

    def test_warn_no_optional_deps(self, tmp_path: Path):
        self._write_pyproject(tmp_path)  # no optional-dependencies
        self._write_lockfile(tmp_path)
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        warn_msgs = [r.message for r in results if r.status == "WARN"]
        assert any("no optional-dependencies" in m for m in warn_msgs)


class TestCheckDepCoverage:
    """Tests for check_dep_coverage — AST import scanning vs declared deps."""

    def _setup_project(
        self, tmp_path: Path, task_code: str, optional_deps: dict[str, list[str]]
    ) -> None:
        # Write pyproject.toml
        lines = ['[project]\nname = "sieval"\ndependencies = ["httpx>=0.28"]\n']
        lines.append("[project.optional-dependencies]\n")
        for group, deps in optional_deps.items():
            deps_str = ", ".join(f'"{d}"' for d in deps)
            lines.append(f"{group} = [{deps_str}]\n")
        (tmp_path / "pyproject.toml").write_text("".join(lines))
        # Create sieval/tasks/ with a task file
        tasks_dir = tmp_path / "sieval" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "__init__.py").write_text("")
        (tasks_dir / "example_0shot_gen.py").write_text(task_code)
        # Create sieval/datasets/
        datasets_dir = tmp_path / "sieval" / "datasets"
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "__init__.py").write_text("")

    def test_covered_import_passes(self, tmp_path: Path):
        self._setup_project(tmp_path, "import numpy\n", {"math": ["numpy>=1.26"]})
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert all(r.status != "FAIL" for r in results)

    def test_uncovered_import_warns(self, tmp_path: Path):
        self._setup_project(tmp_path, "import pandas\n", {"math": ["numpy>=1.26"]})
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert any(r.status == "WARN" and "pandas" in str(r.details) for r in results)

    def test_stdlib_import_ignored(self, tmp_path: Path):
        self._setup_project(tmp_path, "import os\nimport json\n", {})
        # Need at least one declared dep
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sieval"\ndependencies = ["httpx>=0.28"]\n'
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert not any(
            "os" in str(r.details) or "json" in str(r.details) for r in results
        )

    def test_sieval_import_ignored(self, tmp_path: Path):
        self._setup_project(tmp_path, "from sieval.core import utils\n", {})
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sieval"\ndependencies = ["httpx>=0.28"]\n'
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert not any("sieval" in str(r.details) for r in results)

    def test_known_mapping_sklearn(self, tmp_path: Path):
        self._setup_project(
            tmp_path, "import sklearn\n", {"pring": ["scikit-learn>=1.6"]}
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert all(r.status != "FAIL" for r in results)
        assert not any("sklearn" in str(r.details) for r in results)


class TestCheckTasks:
    """Integration tests for check_tasks — registry, imports, tags, naming."""

    def test_real_task_registry_loads(self):
        """Integration test: real sieval.tasks registry should load without error."""
        runner = PreflightRunner()
        results = runner.check_tasks()
        assert len(results) >= 1
        # Registry load should not FAIL
        registry_results = [r for r in results if "registry" in r.message.lower()]
        assert all(r.status != "FAIL" for r in registry_results)

    def test_file_naming_check_runs(self):
        """File naming check should produce results."""
        runner = PreflightRunner()
        results = runner.check_tasks()
        naming_results = [r for r in results if "naming" in r.message.lower()]
        assert len(naming_results) >= 1


class TestCheckDatasets:
    """Integration tests for check_datasets — registry, imports, naming."""

    def test_real_dataset_registry_loads(self):
        """Integration test: real sieval.datasets registry should load without error."""
        runner = PreflightRunner()
        results = runner.check_datasets()
        assert len(results) >= 1
        registry_results = [r for r in results if "registry" in r.message.lower()]
        assert all(r.status != "FAIL" for r in registry_results)

    def test_naming_convention_enforced(self):
        """Naming convention check should produce results."""
        runner = PreflightRunner()
        results = runner.check_datasets()
        naming_results = [r for r in results if "naming" in r.message.lower()]
        assert len(naming_results) >= 1


class TestCheckMetaIndexSync:
    """check_meta_index_sync: live registry must match _meta/index.json."""

    def test_live_repo_index_is_in_sync(self):
        """Integration: the committed index.json matches the current registry.

        Same guarantee the check provides to CI; running it here catches
        drift locally before anyone pushes.
        """
        runner = PreflightRunner()
        results = runner.check_meta_index_sync()
        assert len(results) == 1
        assert results[0].status == "PASS", results[0].details

    def test_fails_when_script_missing(self, tmp_path: Path):
        """If someone deletes sync_meta_index.py, the check fails loudly
        instead of silently passing."""
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_meta_index_sync()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "script not found" in results[0].message

    def test_fails_when_index_stale(self, tmp_path: Path):
        """Simulate divergence by writing a script that always exits 1 and
        confirm preflight propagates FAIL + message."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "sync_meta_index.py").write_text(
            "import sys\n"
            "print('sieval/meta/index.json is out of date.', file=sys.stderr)\n"
            "sys.exit(1)\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_meta_index_sync()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "out of date" in results[0].message
        assert any("out of date" in d for d in results[0].details)


class TestCheckExamples:
    """Integration + edge-case tests for check_examples."""

    def test_real_examples_resolve(self):
        """examples/*.yaml class: references should all resolve in the live registry."""
        runner = PreflightRunner()
        results = runner.check_examples()
        failed = [r for r in results if r.status == "FAIL"]
        assert not failed, [r.message for r in failed]

    def test_skip_when_no_examples_dir(self, tmp_path: Path):
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_examples()
        assert len(results) == 1
        assert results[0].status == "SKIP"

    def test_fail_on_unresolved_class_name(self, tmp_path: Path):
        examples = tmp_path / "examples"
        examples.mkdir()
        (examples / "bad.yaml").write_text(
            "datasets:\n"
            "  foo:\n"
            "    class: NonexistentDatasetClassXyz\n"
            "tasks:\n"
            "  bar:\n"
            "    class: NonexistentTaskClassXyz\n",
            encoding="utf-8",
        )
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_examples()
        failed = [r for r in results if r.status == "FAIL"]
        assert failed
        assert any("do not resolve" in r.message for r in failed)

    def test_fail_on_malformed_yaml(self, tmp_path: Path):
        examples = tmp_path / "examples"
        examples.mkdir()
        (examples / "bad.yaml").write_text("key: [unterminated", encoding="utf-8")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_examples()
        failed = [r for r in results if r.status == "FAIL"]
        assert failed
        assert any("failed to parse" in r.message for r in failed)


class TestSyntaxErrorBranches:
    """SyntaxError graceful handling in AST parsing."""

    def test_extract_urls_from_docstrings_syntax_error(self, tmp_path: Path):
        bad_py = tmp_path / "bad.py"
        bad_py.write_text("def foo(\n")  # intentional syntax error
        runner = PreflightRunner(project_root=tmp_path)
        result = runner._extract_urls_from_docstrings(bad_py)
        assert result == []

    def test_extract_top_level_imports_syntax_error(self, tmp_path: Path):
        bad_py = tmp_path / "bad.py"
        bad_py.write_text("class Foo(\n")  # intentional syntax error
        runner = PreflightRunner(project_root=tmp_path)
        result = runner._extract_top_level_imports(bad_py)
        assert result == set()


class TestAnchorOnlyLink:
    """Anchor-only relative link [foo](#bar) should not cause a FAIL."""

    def test_anchor_only_link_no_fail(self, tmp_path: Path):
        (tmp_path / "README.md").write_text("[section](#overview)\n")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert not any(r.status == "FAIL" for r in results)

    def test_anchor_only_in_check_links_skipped(self, tmp_path: Path):
        """Cover the continue branch for anchor-only in check_links."""
        (tmp_path / "README.md").write_text("placeholder\n")
        runner = PreflightRunner(project_root=tmp_path)
        # Mock _extract_relative_links_from_md to return an anchor-only
        with patch.object(
            runner,
            "_extract_relative_links_from_md",
            return_value=[("section", "#overview", 1)],
        ):
            results = runner.check_links()
        # Should not FAIL on the anchor-only link
        assert not any(r.status == "FAIL" and "broken" in r.message for r in results)


class TestCheckLinksNoFiles:
    """check_links with no .md/.py files and empty task registry returns SKIP."""

    def test_no_md_no_py_skip(self, tmp_path: Path, monkeypatch):
        # Index must also be empty for SKIP to fire (tri-source condition).
        monkeypatch.setattr("sieval.load_index", lambda: ([], []))
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_links()
        assert any(r.status == "SKIP" and "no markdown" in r.message for r in results)


class TestCheckLinksDeepReachability:
    """Deep mode HTTP reachability via mocked httpx."""

    def _make_runner_with_url(self, tmp_path: Path) -> PreflightRunner:
        (tmp_path / "README.md").write_text("See https://example.com/page\n")
        return PreflightRunner(level="deep", project_root=tmp_path)

    def _mock_httpx(self, head_side_effect=None, status_code=200):
        mock_response = MagicMock()
        mock_response.status_code = status_code

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        if head_side_effect:
            mock_client.head = AsyncMock(side_effect=head_side_effect)
        else:
            mock_client.head = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
        mock_httpx.ConnectError = type("ConnectError", (Exception,), {})
        mock_httpx.HTTPError = type("HTTPError", (Exception,), {})
        return mock_httpx

    def test_reachability_all_ok(self, tmp_path: Path):
        runner = self._make_runner_with_url(tmp_path)
        mock_httpx = self._mock_httpx(status_code=200)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            results = runner.check_links()

        assert any(r.status == "PASS" and "reachable" in r.message for r in results)

    def test_reachability_http_error(self, tmp_path: Path):
        runner = self._make_runner_with_url(tmp_path)
        mock_httpx = self._mock_httpx(status_code=404)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            results = runner.check_links()

        assert any(r.status == "WARN" and "unreachable" in r.message for r in results)

    def test_reachability_connection_error(self, tmp_path: Path):
        runner = self._make_runner_with_url(tmp_path)
        mock_httpx = self._mock_httpx()
        exc_cls = mock_httpx.ConnectError
        mock_client = mock_httpx.AsyncClient.return_value
        mock_client.head = AsyncMock(side_effect=exc_cls("connection refused"))

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            results = runner.check_links()

        assert any(r.status == "WARN" and "unreachable" in r.message for r in results)

    def test_reachability_httpx_not_installed(self, tmp_path: Path):
        runner = self._make_runner_with_url(tmp_path)

        # Remove httpx from sys.modules and make import fail
        with patch.dict("sys.modules", {"httpx": None}):
            results = runner.check_links()

        assert any(r.status == "SKIP" and "httpx" in r.message for r in results)

    def test_reachability_real_async_lifecycle(self, tmp_path: Path):
        """Use httpx.MockTransport to exercise real async client lifecycle.

        This catches bugs like the task group running outside the client
        context manager, which fully-mocked tests cannot detect.
        """
        import httpx

        async def _handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        _RealAsyncClient = httpx.AsyncClient

        def _patched_client(**kwargs):
            kwargs.pop("transport", None)
            return _RealAsyncClient(transport=transport, **kwargs)

        runner = self._make_runner_with_url(tmp_path)
        with patch("httpx.AsyncClient", side_effect=_patched_client):
            results = runner.check_links()

        assert any(r.status == "PASS" and "reachable" in r.message for r in results)
        assert not any(
            r.status == "WARN" and "unreachable" in r.message for r in results
        )


class TestCheckDepsEdgeCases:
    """Edge cases for check_deps: missing pyproject, empty lockfile, deep mode."""

    def test_no_pyproject_fails(self, tmp_path: Path):
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "pyproject.toml not found" in results[0].message

    def test_empty_lockfile_fails(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sieval"\ndependencies = []\n'
            '[project.optional-dependencies]\ndev = ["pytest"]\n'
        )
        (tmp_path / "pdm.lock").write_text("")
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        fail_msgs = [r.message for r in results if r.status == "FAIL"]
        assert any("empty" in m for m in fail_msgs)

    def test_deep_mode_pdm_dry_run_success(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sieval"\ndependencies = []\n'
            '[project.optional-dependencies]\ndev = ["pytest"]\nmath = ["numpy"]\n'
        )
        (tmp_path / "pdm.lock").write_text("content")
        runner = PreflightRunner(level="deep", project_root=tmp_path)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("check_preflight.subprocess.run", return_value=mock_result):
            results = runner.check_deps()

        dry_run_pass = [
            r for r in results if r.status == "PASS" and "dry-run" in r.message
        ]
        assert len(dry_run_pass) == 2  # dev + math

    def test_deep_mode_pdm_dry_run_failure(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sieval"\ndependencies = []\n'
            '[project.optional-dependencies]\ndev = ["pytest"]\n'
        )
        (tmp_path / "pdm.lock").write_text("content")
        runner = PreflightRunner(level="deep", project_root=tmp_path)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="resolution failed\n"
        )
        with patch("check_preflight.subprocess.run", return_value=mock_result):
            results = runner.check_deps()

        dry_run_fail = [
            r for r in results if r.status == "FAIL" and "dry-run" in r.message
        ]
        assert len(dry_run_fail) == 1
        assert "resolution failed" in dry_run_fail[0].details[0]


@contextlib.contextmanager
def _inject_mock_registry(
    module_fqn: str,
    export_map: dict[str, str],
):
    """Inject a mock module with _EXPORT_TO_MODULE into sys.modules.

    Why the parent attribute patch: ``import sieval.tasks as tasks_mod``
    resolves via ``getattr(sieval, "tasks")``, not just ``sys.modules``.
    If we only patch sys.modules, the already-cached attribute on the parent
    package still points to the real module, so the import statement in
    preflight.py silently ignores our mock. We must overwrite the attribute
    on the parent and restore it on exit.
    """
    mock_mod = MagicMock()
    mock_mod._EXPORT_TO_MODULE = export_map

    # e.g. "sieval.tasks" -> parent="sieval", attr="tasks"
    parts = module_fqn.rsplit(".", 1)
    parent_name, attr = (parts[0], parts[1]) if len(parts) == 2 else (None, None)
    parent_mod = sys.modules.get(parent_name) if parent_name else None
    old_attr = getattr(parent_mod, attr, None) if parent_mod and attr else None

    with patch.dict("sys.modules", {module_fqn: mock_mod}):
        if parent_mod and attr:
            setattr(parent_mod, attr, mock_mod)
        try:
            yield mock_mod
        finally:
            if parent_mod and attr:
                if old_attr is not None:
                    setattr(parent_mod, attr, old_attr)
                else:
                    with contextlib.suppress(AttributeError):
                        delattr(parent_mod, attr)


class TestCheckTasksEdgeCases:
    """Edge cases for check_tasks: missing init, errors, imports."""

    def _setup(self, tmp_path: Path) -> PreflightRunner:
        tasks_dir = tmp_path / "sieval" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "__init__.py").write_text("")
        return PreflightRunner(project_root=tmp_path)

    def test_tasks_init_not_found(self, tmp_path: Path):
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_tasks()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "__init__.py not found" in results[0].message

    def test_tasks_registry_runtime_error(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        with patch(
            "builtins.__import__",
            side_effect=RuntimeError("duplicate export"),
        ):
            results = runner.check_tasks()
        assert any(
            r.status == "FAIL" and "registry error" in r.message for r in results
        )

    def test_tasks_registry_generic_exception(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        with patch(
            "builtins.__import__",
            side_effect=TypeError("weird error"),
        ):
            results = runner.check_tasks()
        assert any(
            r.status == "FAIL" and "failed to load" in r.message for r in results
        )

    def test_tasks_import_failure_non_import_error(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        exports = {"FooTask": "foo_0shot_gen"}
        with (
            _inject_mock_registry("sieval.tasks", exports),
            patch(
                "check_preflight.importlib.import_module",
                side_effect=AttributeError("boom"),
            ),
        ):
            results = runner.check_tasks()
        assert any(
            r.status == "FAIL" and "failed to import" in r.message for r in results
        )

    def test_tasks_import_warnings(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        exports = {"FooTask": "foo_0shot_gen"}
        with (
            _inject_mock_registry("sieval.tasks", exports),
            patch(
                "check_preflight.importlib.import_module",
                side_effect=ImportError("no module torch"),
            ),
        ):
            results = runner.check_tasks()
        assert any(
            r.status == "WARN" and "missing optional deps" in r.message for r in results
        )

    def test_tasks_no_tags_fail(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        exports = {"FooTask": "foo_0shot_gen"}
        mock_cls = MagicMock()
        mock_cls.tags = None
        mock_module = MagicMock()
        mock_module.FooTask = mock_cls
        with (
            _inject_mock_registry("sieval.tasks", exports),
            patch(
                "check_preflight.importlib.import_module",
                return_value=mock_module,
            ),
        ):
            results = runner.check_tasks()
        assert any(r.status == "FAIL" and "tags" in r.message for r in results)

    def test_tasks_all_tags_pass(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        tasks_dir = tmp_path / "sieval" / "tasks"
        (tasks_dir / "foo_0shot_gen.py").write_text("")
        exports = {"FooTask": "foo_0shot_gen"}
        mock_cls = MagicMock()
        mock_cls.tags = ["bio"]
        mock_module = MagicMock()
        mock_module.FooTask = mock_cls
        with (
            _inject_mock_registry("sieval.tasks", exports),
            patch(
                "check_preflight.importlib.import_module",
                return_value=mock_module,
            ),
        ):
            results = runner.check_tasks()
        assert any(r.status == "PASS" and "tags" in r.message for r in results)
        assert any(r.status == "PASS" and "naming" in r.message for r in results)


class TestCheckDatasetsEdgeCases:
    """Edge cases for check_datasets: init, errors, naming."""

    def _setup(self, tmp_path: Path) -> PreflightRunner:
        ds_dir = tmp_path / "sieval" / "datasets"
        ds_dir.mkdir(parents=True)
        (ds_dir / "__init__.py").write_text("")
        return PreflightRunner(project_root=tmp_path)

    def test_datasets_init_not_found(self, tmp_path: Path):
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_datasets()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "__init__.py not found" in results[0].message

    def test_datasets_registry_runtime_error(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        with patch(
            "builtins.__import__",
            side_effect=RuntimeError("dup"),
        ):
            results = runner.check_datasets()
        assert any(
            r.status == "FAIL" and "registry error" in r.message for r in results
        )

    def test_datasets_registry_generic_exception(self, tmp_path: Path):
        runner = self._setup(tmp_path)
        with patch(
            "builtins.__import__",
            side_effect=TypeError("weird"),
        ):
            results = runner.check_datasets()
        assert any(
            r.status == "FAIL" and "failed to load" in r.message for r in results
        )

    def test_datasets_import_failure_and_warnings(self, tmp_path: Path):
        runner = self._setup(tmp_path)

        exports = {"FooDataset": "foo", "BarDataset": "bar"}

        def side_effect(name):
            if name == "sieval.datasets.foo":
                raise ImportError("no torch")
            if name == "sieval.datasets.bar":
                raise AttributeError("boom")
            return MagicMock()

        with (
            _inject_mock_registry("sieval.datasets", exports),
            patch(
                "check_preflight.importlib.import_module",
                side_effect=side_effect,
            ),
        ):
            results = runner.check_datasets()

        assert any(
            r.status == "WARN" and "missing optional deps" in r.message for r in results
        )
        assert any(
            r.status == "FAIL" and "failed to import" in r.message for r in results
        )

    def test_datasets_bad_naming(self, tmp_path: Path):
        runner = self._setup(tmp_path)

        exports = {"BadName": "badname"}
        mock_module = MagicMock()
        mock_module.BadName = MagicMock()

        with (
            _inject_mock_registry("sieval.datasets", exports),
            patch(
                "check_preflight.importlib.import_module",
                return_value=mock_module,
            ),
        ):
            results = runner.check_datasets()
        assert any(r.status == "WARN" and "naming" in r.message for r in results)


class TestCheckDepCoverageEdgeCases:
    """Edge cases for check_dep_coverage."""

    def test_no_deps_found_warns(self, tmp_path: Path):
        # No pyproject.toml at all
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert any(
            r.status == "WARN" and "no dependencies" in r.message for r in results
        )

    def test_scan_dir_not_exist(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sieval"\ndependencies = ["httpx>=0.28"]\n'
        )
        # No sieval/tasks or sieval/datasets dirs
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_dep_coverage()
        assert any(r.status == "PASS" for r in results)


class TestCheckVersionEdgeCases:
    """Edge cases for check_version: tag mismatch."""

    def test_git_tag_mismatch(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(
            "# Changelog\n\n## [2.0.0] - 2026-04-01\n\n"
            "[2.0.0]: https://github.com/scitix/sieval/compare/v1.0.0...v2.0.0\n"
        )
        runner = PreflightRunner(project_root=tmp_path)
        with patch.object(runner, "_get_latest_git_tag", return_value="1.0.0"):
            results = runner.check_version()

        assert any(
            r.status == "FAIL" and "git tag" in r.message and "!=" in r.message
            for r in results
        )


class TestMainNameGuard:
    """Test the __name__ == '__main__' guard via main()."""

    def test_main_returns_exit_code(self, capsys):
        code = main(["--check", "check_links", "--format", "text"])
        assert code in (0, 1)
        captured = capsys.readouterr()
        assert "check_links" in captured.out


class TestDatasetIntegrity:
    def _meta(self, name, source, checksums=()):
        from sieval.core.datasets.meta import Category, DatasetMeta, Level1Category

        return DatasetMeta(
            name=name,
            display_name=name,
            description="d",
            source=tuple(source),
            categories=(Category(Level1Category.CODE, "CodeGeneration"),),
            checksums=tuple(checksums),
        )

    def test_unpinned_hf_flagged(self):
        out = _dataset_integrity_violations([self._meta("a", ["hf:org/a"])])
        assert len(out) == 1 and "a" in out[0]
        assert "hf source not pinned" in out[0]

    def test_url_without_checksum_flagged(self):
        out = _dataset_integrity_violations([self._meta("b", ["url:https://x/y.csv"])])
        assert len(out) == 1 and "b" in out[0]
        assert "url source missing checksum" in out[0]

    def test_malformed_hf_pin_flagged_not_raised(self):
        # Trailing '@' makes parse_hf_source raise; the check must report it as
        # a violation, not abort the whole preflight with a traceback.
        out = _dataset_integrity_violations([self._meta("d", ["hf:org/d@"])])
        assert len(out) == 1 and "d" in out[0]
        assert "hf source not pinned" in out[0]

    def test_local_source_exempt(self):
        out = _dataset_integrity_violations([self._meta("c", ["local:/data/c"])])
        assert out == []

    def test_pinned_and_checksummed_pass(self):
        metas = [
            self._meta("a", ["hf:org/a@" + "0" * 40]),
            self._meta(
                "b",
                ["url:https://x/y.csv"],
                checksums=[("y.csv", "sha256:" + "a" * 64)],
            ),
            self._meta("c", ["local:/data/c"]),  # local exempt
        ]
        assert _dataset_integrity_violations(metas) == []
