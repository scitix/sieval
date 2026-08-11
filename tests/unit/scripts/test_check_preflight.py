"""
Tests for scripts/check_preflight.py — CheckResult, formatting, and PreflightRunner.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import json
import re
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
    _DENOMINATOR_CONSTANTS,
    _DENOMINATOR_VALUES,
    _TASK_FILE_PATTERN,
    CheckResult,
    PreflightRunner,
    _dataset_integrity_violations,
    format_json,
    format_text,
    lock_drift,
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
        assert len(PreflightRunner.ALL_CHECKS) == 14
        assert "check_links" in PreflightRunner.ALL_CHECKS
        assert "check_examples" in PreflightRunner.ALL_CHECKS
        assert "check_meta_index_sync" in PreflightRunner.ALL_CHECKS
        assert "check_version" in PreflightRunner.ALL_CHECKS
        assert "check_task_shot_knobs" in PreflightRunner.ALL_CHECKS
        assert "check_report_declarations" in PreflightRunner.ALL_CHECKS
        assert "check_record_key_access" in PreflightRunner.ALL_CHECKS
        assert "check_mutmut_config" in PreflightRunner.ALL_CHECKS

    def test_run_all_returns_results(self):
        runner = PreflightRunner()
        results = runner.run()
        assert len(results) >= 10
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

    def test_lock_drift_skips_outside_a_repository(self, tmp_path: Path):
        self._write_pyproject(tmp_path, optional_deps={"dev": ["pytest>=7"]})
        self._write_lockfile(tmp_path)
        runner = PreflightRunner(project_root=tmp_path)
        results = runner.check_deps()
        skips = [r.message for r in results if r.status == "SKIP"]
        assert any("no baseline" in m for m in skips)


def _lock(packages: dict[str, str], edges: dict[str, list[str]] | None = None) -> str:
    """Minimal ``pdm.lock`` text: locked versions plus dependency edges."""
    edges = edges or {}
    out = ['[metadata]\ncontent_hash = "sha256:x"\n']
    for name, version in packages.items():
        out.append(f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n')
        if name in edges:
            listed = ", ".join(f'"{d}"' for d in edges[name])
            out.append(f"dependencies = [{listed}]\n")
    return "".join(out)


def _pyproject(deps: list[str], requires_python: str = "<3.15,>=3.12") -> str:
    listed = ", ".join(f'"{d}"' for d in deps)
    return (
        f'[project]\nname = "sieval"\nrequires-python = "{requires_python}"\n'
        f"dependencies = [{listed}]\n"
    )


class TestLockDrift:
    """Tests for lock_drift — which version changes a re-lock may make."""

    def test_identical_locks_have_no_drift(self):
        lock = _lock({"foo": "1.0.0"})
        proj = _pyproject(["foo>=1.0"])
        report = lock_drift(lock, lock, proj, proj)
        assert report.unjustified == []
        assert report.justified == []
        assert report.requires_python_changed is False

    def test_bump_with_no_requirement_change_is_unjustified(self):
        proj = _pyproject(["foo>=1.0"])
        report = lock_drift(
            _lock({"foo": "1.0.0"}), _lock({"foo": "2.0.0"}), proj, proj
        )
        assert report.unjustified == ["foo 1.0.0 -> 2.0.0"]
        assert report.justified == []

    def test_bump_forced_by_raised_specifier_is_justified(self):
        report = lock_drift(
            _lock({"foo": "1.0.0"}),
            _lock({"foo": "2.0.0"}),
            _pyproject(["foo>=1.0"]),
            _pyproject(["foo>=2.0"]),
        )
        assert report.justified == ["foo 1.0.0 -> 2.0.0"]
        assert report.unjustified == []

    def test_new_constraint_the_locked_version_already_satisfies_forces_nothing(self):
        """The `numpy<=2.2` case: a ceiling a locked 2.2.0 already met cannot be
        the reason numpy moved, so the move stays unjustified."""
        report = lock_drift(
            _lock({"numpy": "2.2.0"}),
            _lock({"numpy": "2.1.0"}),
            _pyproject(["httpx>=0.28"]),
            _pyproject(["httpx>=0.28", "numpy<=2.2"]),
        )
        assert report.unjustified == ["numpy 2.2.0 -> 2.1.0"]

    def test_a_new_requirement_excuses_only_what_it_actually_forces(self):
        """The real d805418a case: RULER's new `tiktoken` needed `regex`, but the
        locked regex already satisfied `regex>=2022.1.18`, so tiktoken cannot be
        the reason it moved. Being merely reachable from a new dependency is not
        justification — only an unsatisfiable pin is."""
        report = lock_drift(
            _lock({"regex": "2025.11.3", "unrelated": "1.0.0"}),
            _lock(
                {"tiktoken": "0.8.0", "regex": "2026.7.10", "unrelated": "2.0.0"},
                edges={"tiktoken": ["regex>=2022.1.18"]},
            ),
            _pyproject(["httpx>=0.28"]),
            _pyproject(["httpx>=0.28", "tiktoken>=0.8.0"]),
        )
        assert report.justified == []
        assert report.unjustified == [
            "regex 2025.11.3 -> 2026.7.10",
            "unrelated 1.0.0 -> 2.0.0",
        ]

    def test_a_new_requirement_that_outruns_the_locked_version_is_justified(self):
        """Same shape, but now the new dependency needs a regex the lock predates,
        so the move really was forced."""
        report = lock_drift(
            _lock({"regex": "2025.11.3"}),
            _lock(
                {"tiktoken": "0.8.0", "regex": "2026.7.10"},
                edges={"tiktoken": ["regex>=2026.1.1"]},
            ),
            _pyproject(["httpx>=0.28"]),
            _pyproject(["httpx>=0.28", "tiktoken>=0.8.0"]),
        )
        assert report.justified == ["regex 2025.11.3 -> 2026.7.10"]
        assert report.unjustified == []

    def test_a_parent_raised_for_its_child_reads_as_unjustified(self):
        """Documented gap. Raising `lib`'s floor forces `app` up too, because the
        old app pinned `lib<2` — but nothing imposes a floor on `app` itself, so
        it cannot be told apart from a gratuitous bump. Declaring `app`'s floor is
        the fix, and it puts the reason in the diff."""
        report = lock_drift(
            _lock({"app": "1.0.0", "lib": "1.0.0"}, edges={"app": ["lib<2"]}),
            _lock({"app": "2.0.0", "lib": "2.0.0"}, edges={"app": ["lib<3"]}),
            _pyproject(["app>=1.0", "lib>=1.0"]),
            _pyproject(["app>=1.0", "lib>=2.0"]),
        )
        assert report.justified == ["lib 1.0.0 -> 2.0.0"]
        assert report.unjustified == ["app 1.0.0 -> 2.0.0"]

    def test_a_moved_package_cannot_justify_the_packages_below_it(self):
        """Why only unmoved entries are trusted. `top` drifted for no reason, and
        the version it landed on requires a newer `dep`. If `top`'s own edge were
        trusted, its unjustified move would silently excuse `dep` as well — the
        mechanism that let 13 of d805418a's 71 moves excuse themselves."""
        proj = _pyproject(["top>=1.0"])
        report = lock_drift(
            _lock({"top": "1.0.0", "dep": "1.0.0"}, edges={"top": ["dep>=1.0"]}),
            _lock({"top": "2.0.0", "dep": "2.0.0"}, edges={"top": ["dep>=2.0"]}),
            proj,
            proj,
        )
        assert report.justified == []
        assert report.unjustified == ["dep 1.0.0 -> 2.0.0", "top 1.0.0 -> 2.0.0"]

    def test_newly_requested_extra_justifies_its_pin(self):
        """`math-verify[antlr4_11_0]` pins antlr *down*; the specifier on
        math-verify itself never stops matching, only the extra is new."""
        report = lock_drift(
            _lock({"math-verify": "0.8.0", "antlr4-python3-runtime": "4.13.2"}),
            _lock(
                {"math-verify": "0.8.0", "antlr4-python3-runtime": "4.11.0"},
                edges={"math-verify": ["antlr4-python3-runtime==4.11.0"]},
            ),
            _pyproject(["math-verify==0.8.0"]),
            _pyproject(["math-verify[antlr4_11_0]==0.8.0"]),
        )
        assert report.justified == ["antlr4-python3-runtime 4.13.2 -> 4.11.0"]
        assert report.unjustified == []

    def test_changed_requires_python_leaves_drift_unaudited(self):
        report = lock_drift(
            _lock({"foo": "1.0.0"}),
            _lock({"foo": "2.0.0"}),
            _pyproject(["foo>=1.0"], requires_python=">=3.12"),
            _pyproject(["foo>=1.0"], requires_python=">=3.13"),
        )
        assert report.requires_python_changed is True
        assert report.justified == ["foo 1.0.0 -> 2.0.0"]
        assert report.unjustified == []

    def test_a_moved_ceiling_also_leaves_drift_unaudited(self):
        """Widening the ceiling demands support for a Python no locked version
        had to cover, so it re-resolves just as a raised floor does."""
        report = lock_drift(
            _lock({"foo": "1.0.0"}),
            _lock({"foo": "2.0.0"}),
            _pyproject(["foo>=1.0"], requires_python=">=3.12,<3.15"),
            _pyproject(["foo>=1.0"], requires_python=">=3.12,<3.16"),
        )
        assert report.requires_python_changed is True
        assert report.unjustified == []

    def test_entries_disagreeing_on_version_join_instead_of_shadowing(self):
        """A multi-target lock can hold one name at two versions. Taking the
        first entry would miss a move confined to the second."""
        proj = _pyproject(["numpy>=2.0"])
        split = '\n[[package]]\nname = "numpy"\nversion = "%s"\n'
        base = _lock({"numpy": "2.2.0"}) + split % "2.3.0"
        cand = _lock({"numpy": "2.2.0"}) + split % "2.9.9"
        report = lock_drift(base, cand, proj, proj)
        assert report.unjustified == ["numpy 2.2.0/2.3.0 -> 2.2.0/2.9.9"]

    def test_a_specifier_one_entry_violates_forces_the_whole_node(self):
        """`numpy<=2.2` leaves the 2.2.0 entry alone but not the 2.3.0 one, so
        it does force a move — checking a single entry would miss that."""
        split = '\n[[package]]\nname = "numpy"\nversion = "%s"\n'
        report = lock_drift(
            _lock({"numpy": "2.2.0"}) + split % "2.3.0",
            _lock({"numpy": "2.1.0"}) + split % "2.2.0",
            _pyproject(["numpy>=2.0"]),
            _pyproject(["numpy>=2.0", "numpy<=2.2"]),
        )
        assert report.justified == ["numpy 2.2.0/2.3.0 -> 2.1.0/2.2.0"]
        assert report.unjustified == []

    def test_reordered_entries_are_not_drift(self):
        """Same versions, different entry order: sorting keeps it unchanged."""
        proj = _pyproject(["numpy>=2.0"])
        split = '\n[[package]]\nname = "numpy"\nversion = "%s"\n'
        report = lock_drift(
            _lock({"numpy": "2.2.0"}) + split % "2.3.0",
            _lock({"numpy": "2.3.0"}) + split % "2.2.0",
            proj,
            proj,
        )
        assert report.unjustified == []
        assert report.justified == []

    def test_added_and_removed_packages_are_not_drift(self):
        proj = _pyproject(["foo>=1.0"])
        report = lock_drift(
            _lock({"foo": "1.0.0", "gone": "1.0.0"}),
            _lock({"foo": "1.0.0", "fresh": "1.0.0"}),
            proj,
            proj,
        )
        assert report.unjustified == []
        assert report.justified == []

    def test_extras_variants_of_one_package_share_a_node(self):
        """``coverage`` and ``coverage[toml]`` are two lock entries at one
        version, and only the extras one carries the ``tomli`` edge. If the plain
        entry shadowed it, that constraint would go unseen and tomli's forced
        move would read as unjustified."""
        proj_before = _pyproject(["httpx>=0.28"])
        proj_after = _pyproject(["httpx>=0.28", "coverage>=7"])
        cand = (
            _lock({"coverage": "7.15.2", "tomli": "2.3.0"})
            + '\n[[package]]\nname = "coverage"\nversion = "7.15.2"\n'
            + 'extras = ["toml"]\ndependencies = ["tomli>=2.1"]\n'
        )
        report = lock_drift(_lock({"tomli": "2.0.0"}), cand, proj_before, proj_after)
        assert report.justified == ["tomli 2.0.0 -> 2.3.0"]
        assert report.unjustified == []

    def test_declaration_dropped_does_not_excuse_survivors(self):
        report = lock_drift(
            _lock({"foo": "1.0.0", "bar": "1.0.0"}),
            _lock({"foo": "2.0.0", "bar": "1.0.0"}),
            _pyproject(["foo>=1.0", "bar>=1.0"]),
            _pyproject(["foo>=1.0"]),
        )
        assert report.unjustified == ["foo 1.0.0 -> 2.0.0"]

    def test_unparsable_input_reports_nothing(self):
        report = lock_drift("not = = toml", "still ] not", "[project", "[project")
        assert report.unjustified == []
        assert report.justified == []


def _fake_git(
    blobs: dict[str, str], head: str = "aaaa1111", merge_base: str | None = None
):
    """subprocess.run stand-in answering the git plumbing lock-drift uses.

    *blobs* maps ``"<rev>:<path>"`` to contents; anything absent fails, as git
    does for a blob a shallow clone never fetched.
    """

    def run(cmd, *_args, **_kwargs):
        def ok(out: str):
            return subprocess.CompletedProcess(cmd, 0, out, "")

        def fail():
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: bad object")

        if cmd[:2] == ["git", "show"]:
            return ok(blobs[cmd[2]]) if cmd[2] in blobs else fail()
        if cmd[:2] == ["git", "rev-parse"]:
            return ok(f"{head}\n")
        if cmd[:2] == ["git", "merge-base"]:
            return ok(f"{merge_base}\n") if merge_base else fail()
        return fail()

    return run


class TestLockDriftBaseline:
    """Tests for _check_lock_drift — which revision the lock is judged against."""

    def _write_tree(self, tmp_path: Path, lock: str, pyproject: str) -> None:
        (tmp_path / "pdm.lock").write_text(lock)
        (tmp_path / "pyproject.toml").write_text(pyproject)

    def test_uncommitted_lock_change_is_judged_against_head(self, tmp_path: Path):
        proj = _pyproject(["foo>=1.0"])
        self._write_tree(tmp_path, _lock({"foo": "2.0.0"}), proj)
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git(
            {"HEAD:pdm.lock": _lock({"foo": "1.0.0"}), "HEAD:pyproject.toml": proj}
        )
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["FAIL"]
        assert "vs HEAD" in results[0].message
        assert "foo 1.0.0 -> 2.0.0" in results[0].details
        assert any("--update-reuse" in d for d in results[0].details)

    def test_committed_lock_is_judged_against_the_merge_base(self, tmp_path: Path):
        proj = _pyproject(["foo>=1.0"])
        committed = _lock({"foo": "2.0.0"})
        self._write_tree(tmp_path, committed, proj)
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git(
            {
                "HEAD:pdm.lock": committed,  # nothing uncommitted
                "bbbb2222:pdm.lock": _lock({"foo": "1.0.0"}),
                "bbbb2222:pyproject.toml": proj,
            },
            merge_base="bbbb2222",
        )
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["FAIL"]
        assert "vs bbbb2222" in results[0].message

    def test_skips_when_the_merge_base_is_head(self, tmp_path: Path):
        """On the default branch there is nothing to compare against."""
        proj = _pyproject(["foo>=1.0"])
        committed = _lock({"foo": "2.0.0"})
        self._write_tree(tmp_path, committed, proj)
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git(
            {"HEAD:pdm.lock": committed}, head="aaaa1111", merge_base="aaaa1111"
        )
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["SKIP"]

    def test_skips_when_the_baseline_blob_is_unreachable(self, tmp_path: Path):
        """Shallow clone: the merge base resolves but its blobs were never
        fetched. A baseline we cannot read is not a violation."""
        proj = _pyproject(["foo>=1.0"])
        committed = _lock({"foo": "2.0.0"})
        self._write_tree(tmp_path, committed, proj)
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git({"HEAD:pdm.lock": committed}, merge_base="bbbb2222")
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["SKIP"]

    def test_passes_when_only_requested_versions_moved(self, tmp_path: Path):
        self._write_tree(tmp_path, _lock({"foo": "2.0.0"}), _pyproject(["foo>=2.0"]))
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git(
            {
                "HEAD:pdm.lock": _lock({"foo": "1.0.0"}),
                "HEAD:pyproject.toml": _pyproject(["foo>=1.0"]),
            }
        )
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["PASS"]
        assert "1 justified" in results[0].message

    def test_long_drift_lists_are_truncated(self, tmp_path: Path):
        proj = _pyproject(["foo>=1.0"])
        before = {f"pkg{i:03d}": "1.0.0" for i in range(30)}
        after = dict.fromkeys(before, "2.0.0")
        self._write_tree(tmp_path, _lock(after), proj)
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git({"HEAD:pdm.lock": _lock(before), "HEAD:pyproject.toml": proj})
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert results[0].status == "FAIL"
        assert "30 locked package(s) drifted" in results[0].message
        assert any(d == "... and 10 more" for d in results[0].details)

    def test_requires_python_change_lists_what_it_excused(self, tmp_path: Path):
        """The one path that excuses drift wholesale still has to name the
        moves — a bare count would let any number of them through unread."""
        self._write_tree(
            tmp_path,
            _lock({"foo": "2.0.0"}),
            _pyproject(["foo>=1.0"], requires_python=">=3.13"),
        )
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git(
            {
                "HEAD:pdm.lock": _lock({"foo": "1.0.0"}),
                "HEAD:pyproject.toml": _pyproject(
                    ["foo>=1.0"], requires_python=">=3.12"
                ),
            }
        )
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["WARN"]
        assert "foo 1.0.0 -> 2.0.0" in results[0].details

    def test_requires_python_change_on_a_clean_lock_passes(self, tmp_path: Path):
        """Nothing moved, so there is nothing to hand back for review."""
        lock = _lock({"foo": "1.0.0"})
        self._write_tree(
            tmp_path, lock, _pyproject(["foo>=1.0"], requires_python=">=3.13")
        )
        runner = PreflightRunner(project_root=tmp_path)
        git = _fake_git(
            {
                "HEAD:pdm.lock": lock,  # nothing uncommitted
                "bbbb2222:pdm.lock": lock,
                "bbbb2222:pyproject.toml": _pyproject(
                    ["foo>=1.0"], requires_python=">=3.12"
                ),
            },
            merge_base="bbbb2222",
        )
        with patch("check_preflight.subprocess.run", side_effect=git):
            results = runner._check_lock_drift()
        assert [r.status for r in results] == ["PASS"]


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


class TestTaskFileNamingPattern:
    """Unit tests for the task-file naming regex (`_TASK_FILE_PATTERN`)."""

    @pytest.mark.parametrize(
        "name",
        [
            "cmmlu_kshot_clp.py",
            "foo_5shot_clp.py",
            "foo_0shot_gen.py",
            "foo_kshot_base_gen.py",
            "foo_3shot_ppl.py",
            "foo_2shot_llmjudge_gen.py",
        ],
    )
    def test_accepts_valid_suffixes(self, name):
        assert _TASK_FILE_PATTERN.match(name) is not None

    @pytest.mark.parametrize(
        "name",
        [
            "foo_0shot_gen_fixed.py",
            "foo_0shot_base_gen_fixed.py",
            "foo_kshot_ppl_fixed_v2.py",
        ],
    )
    def test_accepts_a_variant_after_the_mode(self, name):
        assert _TASK_FILE_PATTERN.match(name) is not None

    @pytest.mark.parametrize(
        "name",
        [
            "foo_clp.py",  # missing shot segment
            "foo_5shot_clpx.py",  # not a known mode
            "foo_0shot_gen_.py",  # empty variant
            "foo_0shot_gen_Fixed.py",  # variant is not lower-case
        ],
    )
    def test_rejects_malformed(self, name):
        assert _TASK_FILE_PATTERN.match(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            # The example both CLAUDE.md and rules/tasks.md name as canonical.
            "foo_0shot_clp_gen.py",
            "foo_0shot_gen_gen.py",
            "foo_5shot_clp_ppl.py",
            "foo_0shot_gen_base_gen.py",
        ],
    )
    def test_rejects_a_variant_that_spells_a_mode(self, name):
        # Two readings (mode `gen` + variant `ppl`, or mode `ppl` misplaced), so
        # the name is rejected rather than settled by alternation precedence.
        assert _TASK_FILE_PATTERN.match(name) is None


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
        if runner.project_root.name == "mutants":
            # mutmut runs the suite from a copy of the tree, where neither side
            # of this comparison is the thing it is about: the "committed" index
            # is a copy and the registry is importable only by accident. It is
            # the last test standing between `mutmut run` and a score, and
            # skipping it costs nothing — CI runs this check on the real tree.
            pytest.skip("meaningless inside a mutmut copy of the tree")
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


class TestCheckTaskShotKnobs:
    """The shot-knob naming + `self.n_shot` wiring guard."""

    def _run(self, tmp_path: Path, source: str) -> CheckResult:
        tasks_dir = tmp_path / "sieval" / "tasks"
        tasks_dir.mkdir(parents=True)
        # __init__.py is excluded from the scan; a sibling module is the subject.
        (tasks_dir / "__init__.py").write_text("")
        (tasks_dir / "demo_kshot_gen.py").write_text(source)
        results = PreflightRunner(project_root=tmp_path).check_task_shot_knobs()
        assert len(results) == 1
        return results[0]

    # --- rule 1: the shot-count parameter is spelled `n_shot` ---------------

    def test_canonical_spelling_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, n_shot: int = 5):\n"
            "        self.n_shot = n_shot\n",
        )
        assert r.status == "PASS"

    # `k` is deliberately absent: it is not a misspelling of `n_shot` but a
    # different knob, caught by rule 3 rather than rule 1 (and by two rules at
    # once when fed to `self.n_shot`). Its own cases are below.
    @pytest.mark.parametrize(
        "param", ["shots", "num_shots", "nshot", "fewshot", "shot_count"]
    )
    def test_misspelled_shot_param_flagged(self, tmp_path: Path, param: str):
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            f"    def __init__(self, dataset, model, *, {param}: int = 5):\n"
            f"        self.n_shot = {param}\n",
        )
        assert r.status == "FAIL"
        assert len(r.details) == 1
        assert "DemoTask" in r.details[0]

    @pytest.mark.parametrize(
        "param", ["fewshot_split", "fewshot_seed", "fewshot_as_multiturn"]
    )
    def test_compound_naming_another_noun_not_flagged(self, tmp_path: Path, param: str):
        """`fewshot_split` names a split, not a count — must not be renamed."""
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            f"    def __init__(self, dataset, model, *, {param}=None):\n"
            f"        self._x = {param}\n",
        )
        assert r.status == "PASS"

    # --- rule 2: accepting `n_shot` means storing it as self.n_shot --------

    def test_n_shot_not_stored_publicly_flagged(self, tmp_path: Path):
        """Storing the knob privately leaves the decorator's class value standing.

        The task itself works; only `meta.json` is silently wrong, reporting the
        declared default as the count the run used. That is the whole reason
        this rule survives the move to a public `n_shot`.
        """
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, n_shot: int = 5):\n"
            "        self._n_shot = n_shot\n",
        )
        assert r.status == "FAIL"
        assert "never assigns self.n_shot" in r.details[0]

    def test_n_shot_from_a_derived_count_passes(self, tmp_path: Path):
        """The source expression is unconstrained — only `k` is rejected."""
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, n_shot: int = 5):\n"
            "        self._examples = pick(n_shot)\n"
            "        self.n_shot = len(self._examples)\n",
        )
        assert r.status == "PASS"

    def test_task_without_shot_knob_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, seed: int = 0):\n"
            "        self._seed = seed\n",
        )
        assert r.status == "PASS"

    # --- rule 3: `k` is the k in pass@k, not the sampling budget `n` --------

    def test_k_with_pass_at_k_metric_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, k: int = 1, n: int = 10):\n"
            "        self._k = k\n"
            "    async def report(self, finals, fails):\n"
            "        return {f'pass@{self._k}': 1.0}\n",
        )
        assert r.status == "PASS"

    def test_k_without_pass_at_k_metric_flagged(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, k: int = 5):\n"
            "        self._k = k\n"
            "    async def report(self, finals, fails):\n"
            "        return {'score': 1.0}\n",
        )
        assert r.status == "FAIL"
        assert "computes no pass@k metric" in r.details[0]

    def test_pass_at_k_in_a_docstring_is_not_evidence(self, tmp_path: Path):
        """Prose must not satisfy rule 3 — else `k` slips back to meaning shots.

        A class docstring mentioning pass@1 while computing nothing of the kind
        used to flip this to PASS, which is the whole regression rule 3 guards.
        """
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            '    """Scores things. Nothing to do with pass@1 really."""\n'
            "    def __init__(self, dataset, model, *, k: int = 5):\n"
            "        self._k = k\n"
            "    async def report(self, finals, fails):\n"
            "        return {'score': 1.0}\n",
        )
        assert r.status == "FAIL"
        assert "computes no pass@k metric" in r.details[0]

    def test_pass_at_k_in_a_method_docstring_is_not_evidence(self, tmp_path: Path):
        """Method docstrings are prose too, not just the class's own."""
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, k: int = 5):\n"
            "        self._k = k\n"
            "    async def report(self, finals, fails):\n"
            '        """Would report pass@k if it did that."""\n'
            "        return {'score': 1.0}\n",
        )
        assert r.status == "FAIL"
        assert "computes no pass@k metric" in r.details[0]

    def test_pass_at_k_metric_key_still_counts_as_evidence(self, tmp_path: Path):
        """The docstring exclusion must not swallow a real metric key."""
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            '    """A docstring saying nothing about the metric."""\n'
            "    def __init__(self, dataset, model, *, k: int = 1):\n"
            "        self._k = k\n"
            "    async def report(self, finals, fails):\n"
            "        return {f'pass@{self._k}': 1.0}\n",
        )
        assert r.status == "PASS"

    def test_n_shot_fed_from_k_flagged(self, tmp_path: Path):
        """The regression this guard exists for: a shot count spelled `k`."""
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\n"
            "class DemoTask:\n"
            "    def __init__(self, dataset, model, *, k: int = 5):\n"
            "        self._k = k\n"
            "        self.n_shot = self._k\n"
            "    async def report(self, finals, fails):\n"
            "        return {f'pass@{self._k}': 1.0}\n",
        )
        assert r.status == "FAIL"
        assert any("the k in pass@k, not a shot count" in d for d in r.details)

    # --- scope -------------------------------------------------------------

    def test_k_rule_is_decorated_classes_only(self, tmp_path: Path):
        """An undecorated base's pass@k is usually computed by its subclass.

        So rule 3 cannot be judged from the base's body — applying it there
        would be a false positive. Rules 1 and 2 still bind it (below).
        """
        r = self._run(
            tmp_path,
            "class Helper:\n"
            "    def __init__(self, *, k: int = 5):\n"
            "        self._k = k\n",
        )
        assert r.status == "PASS"

    def test_subclass_without_own_init_is_not_counted(self, tmp_path: Path):
        """Nothing to check on the subclass itself — the base carries the knob.

        Coverage of the inherited constructor comes from scanning the base, not
        from the subclass; see the two `shared base` tests below.
        """
        r = self._run(
            tmp_path,
            "@sieval_task(name='demo')\nclass DemoTask(Base):\n    pass\n",
        )
        assert r.status == "PASS"
        assert "all 0 task constructor(s)" in r.message

    def test_shared_base_carrying_the_knob_is_checked(self, tmp_path: Path):
        """The hole this widening closes.

        A decorated task may inherit `__init__` from an undecorated base (the
        `arc/_base.py` layout). Keying the scan on the decorator skipped both
        ends — subclass has no `__init__`, base has no decorator — so a knob
        that never reached `self.n_shot` went unchecked, with only a silently
        lower count to show for it.
        """
        r = self._run(
            tmp_path,
            "class _SharedInit:\n"
            "    def __init__(self, dataset, model, *, n_shot: int = 25):\n"
            "        self._n_shot = n_shot\n"
            "\n"
            "@sieval_task(name='demo')\n"
            "class DemoTask(_SharedInit, Task):\n"
            "    pass\n",
        )
        assert r.status == "FAIL"
        assert "never assigns self.n_shot" in r.details[0]
        assert "_SharedInit" in r.details[0]

    def test_shared_base_wiring_the_knob_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class _SharedInit:\n"
            "    def __init__(self, dataset, model, *, n_shot: int = 25):\n"
            "        self.n_shot = n_shot\n"
            "\n"
            "@sieval_task(name='demo')\n"
            "class DemoTask(_SharedInit, Task):\n"
            "    pass\n",
        )
        assert r.status == "PASS"
        assert "all 1 task constructor(s)" in r.message

    def test_shared_base_misspelling_the_knob_is_flagged(self, tmp_path: Path):
        """Rule 1 binds the base too, not only decorated classes."""
        r = self._run(
            tmp_path,
            "class _SharedInit:\n"
            "    def __init__(self, dataset, model, *, num_shots: int = 25):\n"
            "        self.n_shot = num_shots\n",
        )
        assert r.status == "FAIL"
        assert "the repo spells it 'n_shot'" in r.details[0]

    def test_subdirectory_tasks_are_scanned(self, tmp_path: Path):
        """A benchmark with >= 5 task files lives in a subpackage — still bound.

        Non-recursive scoping here would report PASS while covering nothing in
        it, which is the silent-pass channel this check exists to close.
        """
        pkg = tmp_path / "sieval" / "tasks" / "bench"
        pkg.mkdir(parents=True)
        (tmp_path / "sieval" / "tasks" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        (pkg / "bench_kshot_gen.py").write_text(
            "@sieval_task(name='bench')\n"
            "class BenchTask:\n"
            "    def __init__(self, dataset, model, *, k: int = 5):\n"
            "        self._k = k\n"
        )
        results = PreflightRunner(project_root=tmp_path).check_task_shot_knobs()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "bench_kshot_gen.py" in results[0].details[0]

    def test_unparsable_module_reported_not_raised(self, tmp_path: Path):
        r = self._run(tmp_path, "class Broken(:\n")
        assert r.status == "FAIL"
        assert "could not parse" in r.details[0]

    def test_no_task_modules_skips(self, tmp_path: Path):
        results = PreflightRunner(project_root=tmp_path).check_task_shot_knobs()
        assert len(results) == 1
        assert results[0].status == "SKIP"

    def test_real_tasks_pass(self):
        """Integration: the shipped tasks satisfy all three rules."""
        results = PreflightRunner().check_task_shot_knobs()
        assert [r.status for r in results] == ["PASS"]


class TestCheckReportDeclarations:
    """The guard on `score_key` / `denominator_policy` in every task report."""

    #: Stand-in for `sieval/core/tasks/metrics.py`, reduced to the shape rule 4
    #: must follow: `sampling_report` reaches `pass@k` only through a call and a
    #: `|`, so reading one body sees none of its keys.
    _METRICS = (
        "def health_metrics(finals):\n"
        "    return {'n_unextracted': 0.0}\n"
        "def rollout_metrics(correct, k=1):\n"
        "    return {f'pass@{k}': 1.0, 'avg@n': 1.0}\n"
        "def budget_metrics(k=1):\n"
        "    return {'n': 1.0, 'k': 1.0}\n"
        "def sampling_report(correct, k=1):\n"
        "    return rollout_metrics(correct, k=k) | budget_metrics(k=k)\n"
    )

    def _run(
        self,
        tmp_path: Path,
        source: str,
        *,
        metrics: str | None = None,
        **extra: str,
    ) -> CheckResult:
        tasks_dir = tmp_path / "sieval" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "__init__.py").write_text("")
        (tasks_dir / "demo_0shot_gen.py").write_text(source)
        if metrics is not None:
            core_tasks = tmp_path / "sieval" / "core" / "tasks"
            core_tasks.mkdir(parents=True)
            (core_tasks / "metrics.py").write_text(metrics)
        # Kwarg name -> sibling module: `_base` becomes `_base.py` next to the
        # subject, which is the `arc/` layout the delegation rule exists for.
        for rel, text in extra.items():
            path = tasks_dir / f"{rel}.py"
            path.write_text(text)
        results = PreflightRunner(project_root=tmp_path).check_report_declarations()
        assert len(results) == 1
        return results[0]

    # --- rule 1: denominator_policy is declared ------------------------------

    def test_both_declared_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            'score': 1.0,\n"
            "            'accuracy': 1.0,\n"
            "            SCORE_KEY_FIELD: 'accuracy',\n"
            "            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,\n"
            "        }\n",
        )
        assert r.status == "PASS"
        assert "all 1 task report(s)" in r.message

    def test_missing_denominator_fails(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {'score': 1.0, 'accuracy': 1.0, "
            "SCORE_KEY_FIELD: 'accuracy'}\n",
        )
        assert r.status == "FAIL"
        assert "declares no 'denominator_policy'" in r.details[0]

    def test_string_literals_accepted(self, tmp_path: Path):
        """The constants are the idiom, but the report is JSON — literals count."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            'score': 1.0,\n"
            "            'acc': 1.0,\n"
            "            'score_key': 'acc',\n"
            "            'denominator_policy': 'judged',\n"
            "        }\n",
        )
        assert r.status == "PASS"

    # --- rule 2: score_key required only where a `score` is emitted ----------

    def test_score_without_score_key_fails(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {'score': 1.0, DENOMINATOR_FIELD: DENOMINATOR_JUDGED}\n",
        )
        assert r.status == "FAIL"
        assert "emits 'score' but no 'score_key'" in r.details[0]

    def test_no_headline_needs_no_score_key(self, tmp_path: Path):
        """The `t_eval_before_calling` shape: per-axis rates, no headline."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            'thought': 1.0,\n"
            "            'name': 1.0,\n"
            "            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,\n"
            "        }\n",
        )
        assert r.status == "PASS"

    def test_reading_a_score_key_is_not_emitting_one(self, tmp_path: Path):
        """`rollout['score']` reads a verdict; it does not publish a headline."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        rate = sum(r['score'] for r in finals)\n"
            "        return {'rate': rate, DENOMINATOR_FIELD: DENOMINATOR_JUDGED}\n",
        )
        assert r.status == "PASS"

    def test_subscript_assignment_counts_as_emitting(self, tmp_path: Path):
        """`results['score'] = ...` is the ifeval shape — a write, so rule 2 binds."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        results = {DENOMINATOR_FIELD: DENOMINATOR_JUDGED}\n"
            "        results['score'] = 1.0\n"
            "        return results\n",
        )
        assert r.status == "FAIL"
        assert "emits 'score' but no 'score_key'" in r.details[0]

    # --- rule 3: the policy is one of the two ------------------------------

    def test_freeform_policy_fails(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            'score': 1.0,\n"
            "            'acc': 1.0,\n"
            "            SCORE_KEY_FIELD: 'acc',\n"
            "            DENOMINATOR_FIELD: 'attempted',\n"
            "        }\n",
        )
        assert r.status == "FAIL"
        assert "denominator_policy is 'attempted'" in r.details[0]

    def test_freeform_policy_via_subscript_fails(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        out = {'score': 1.0, 'acc': 1.0, SCORE_KEY_FIELD: 'acc'}\n"
            "        out[DENOMINATOR_FIELD] = 'per_instruction'\n"
            "        return out\n",
        )
        assert r.status == "FAIL"
        assert "denominator_policy is 'per_instruction'" in r.details[0]

    # --- delegation: the `arc/` layout --------------------------------------

    def test_declaration_in_the_shared_helper_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "from ._base import demo_report\n"
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return demo_report(finals, fails)\n",
            _base="def demo_report(finals, fails):\n"
            "    return {\n"
            "        'score': 1.0,\n"
            "        'acc': 1.0,\n"
            "        SCORE_KEY_FIELD: 'acc',\n"
            "        DENOMINATOR_FIELD: DENOMINATOR_JUDGED,\n"
            "    }\n",
        )
        assert r.status == "PASS"

    def test_delegation_to_a_bare_helper_still_fails(self, tmp_path: Path):
        """Following the call must not become a blanket exemption."""
        r = self._run(
            tmp_path,
            "from ._base import demo_report\n"
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return demo_report(finals, fails)\n",
            _base="def demo_report(finals, fails):\n    return {'score': 1.0}\n",
        )
        assert r.status == "FAIL"
        assert len(r.details) == 2
        assert any("declares no 'denominator_policy'" in d for d in r.details)
        assert any("emits 'score' but no 'score_key'" in d for d in r.details)

    def test_absolute_import_delegation_resolves(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "from sieval.tasks.shared import demo_report\n"
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return demo_report(finals, fails)\n",
            shared="def demo_report(finals, fails):\n"
            "    return {\n"
            "        'score': 1.0,\n"
            "        'acc': 1.0,\n"
            "        SCORE_KEY_FIELD: 'acc',\n"
            "        DENOMINATOR_FIELD: DENOMINATOR_JUDGED,\n"
            "    }\n",
        )
        assert r.status == "PASS"

    # --- scope ---------------------------------------------------------------

    def test_module_without_report_is_not_checked(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True\n",
        )
        assert r.status == "PASS"
        assert "all 0 task report(s)" in r.message

    def test_unparsable_module_reported_not_raised(self, tmp_path: Path):
        r = self._run(tmp_path, "class Broken(:\n")
        assert r.status == "FAIL"
        assert "could not parse" in r.details[0]

    def test_no_task_modules_skips(self, tmp_path: Path):
        results = PreflightRunner(project_root=tmp_path).check_report_declarations()
        assert len(results) == 1
        assert results[0].status == "SKIP"

    # --- rule 4: score_key names a column the report actually writes ---------

    def test_score_key_naming_an_absent_column_fails(self, tmp_path: Path):
        """The copy-paste defect: a sibling's key name on a report without it."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            'score': 1.0,\n"
            "            'acc': 1.0,\n"
            "            SCORE_KEY_FIELD: 'exact_match',\n"
            "            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,\n"
            "        }\n",
        )
        assert r.status == "FAIL"
        assert "score_key names 'exact_match'" in r.details[0]

    def test_score_key_matching_an_fstring_key_passes(self, tmp_path: Path):
        """The ifeval shape: the headline key is built in a loop, not written flat."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        results = {DENOMINATOR_FIELD: DENOMINATOR_JUDGED}\n"
            "        for grade in ('strict', 'loose'):\n"
            "            results[f'{grade}_prompt_level_accuracy'] = 1.0\n"
            "        results['score'] = results['strict_prompt_level_accuracy']\n"
            "        results[SCORE_KEY_FIELD] = 'strict_prompt_level_accuracy'\n"
            "        return results\n",
        )
        assert r.status == "PASS"

    def test_score_key_outside_the_fstring_pattern_fails(self, tmp_path: Path):
        """The pattern constrains — it is not a blanket pass for the loop's shape."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        results = {'score': 1.0, DENOMINATOR_FIELD: DENOMINATOR_JUDGED}\n"
            "        for grade in ('strict', 'loose'):\n"
            "            results[f'{grade}_prompt_level_accuracy'] = 1.0\n"
            "        results[SCORE_KEY_FIELD] = 'strict_prompt_level_rate'\n"
            "        return results\n",
        )
        assert r.status == "FAIL"
        assert "score_key names 'strict_prompt_level_rate'" in r.details[0]

    def test_score_key_from_a_merged_helper_passes(self, tmp_path: Path):
        """`pass@1` is legitimate when the report merges the helper emitting it.

        Reachable only transitively: `sampling_report` returns
        `rollout_metrics(...) | budget_metrics(...)`, naming no key in its body.
        """
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        metrics = {\n"
            "            'score': 1.0,\n"
            "            SCORE_KEY_FIELD: 'pass@1',\n"
            "            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,\n"
            "        }\n"
            "        rolled = sampling_report([True], k=1)\n"
            "        metrics.update(rolled)\n"
            "        return metrics\n",
            metrics=self._METRICS,
        )
        assert r.status == "PASS"

    def test_score_key_absent_from_the_merged_helper_fails(self, tmp_path: Path):
        """Merging widens to THAT helper's keys, not to all of `metrics.py`.

        `health_metrics` emits only `n_unextracted`, so `pass@1` still dangles —
        the reason a merge is traced to a name, not blanket-exempted.
        """
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            'score': 1.0,\n"
            "            SCORE_KEY_FIELD: 'pass@1',\n"
            "            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,\n"
            "        } | health_metrics(finals)\n",
            metrics=self._METRICS,
        )
        assert r.status == "FAIL"
        assert "score_key names 'pass@1'" in r.details[0]

    def test_untraceable_key_source_skips_the_rule(self, tmp_path: Path):
        """A `**` spread of an expression: no module to go read, so stay silent."""
        r = self._run(
            tmp_path,
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return {\n"
            "            **self._axes(finals),\n"
            "            'score': 1.0,\n"
            "            SCORE_KEY_FIELD: 'whatever_axis',\n"
            "            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,\n"
            "        }\n",
        )
        assert r.status == "PASS"

    def test_score_key_in_the_shared_helper_is_resolved_there(self, tmp_path: Path):
        """Delegation carries rule 4 too, so one bad key reports for every leaf."""
        r = self._run(
            tmp_path,
            "from ._base import demo_report\n"
            "class DemoTask:\n"
            "    async def report(self, finals, fails):\n"
            "        return demo_report(finals, fails)\n",
            _base="def demo_report(finals, fails):\n"
            "    return {\n"
            "        'score': 1.0,\n"
            "        'acc': 1.0,\n"
            "        SCORE_KEY_FIELD: 'accuracy',\n"
            "        DENOMINATOR_FIELD: DENOMINATOR_JUDGED,\n"
            "    }\n",
        )
        assert r.status == "FAIL"
        assert "score_key names 'accuracy'" in r.details[0]

    def test_real_tasks_pass(self):
        """Integration: every shipped report declares both fields."""
        results = PreflightRunner().check_report_declarations()
        assert [r.status for r in results] == ["PASS"]

    def test_real_tasks_resolve_their_score_key(self):
        """Rule 4 has real coverage, not just fixtures.

        A rule skipping every shipped report would still pass
        `test_real_tasks_pass`, so pin the count it resolved. A DROP means a
        report grew a key source rule 4 cannot follow.
        """
        (result,) = PreflightRunner().check_report_declarations()
        match = re.search(r"\((\d+) score_key", result.message)
        assert match is not None, result.message
        assert int(match.group(1)) >= 47

    def test_vocabulary_matches_the_metrics_module(self):
        """Rule 3's vocabulary is `metrics.py`'s, not a second copy of it.

        The checker hard-codes the policy names so it stays AST-only. That makes
        it a duplicate, and a duplicate that falls behind would reject a policy
        the module legitimately added — so the two are pinned equal here rather
        than left to agree by luck.
        """
        from sieval.core.tasks import metrics

        defined = {
            name: value
            for name, value in vars(metrics).items()
            if name.startswith("DENOMINATOR_") and name != "DENOMINATOR_FIELD"
        }
        assert set(defined) == set(_DENOMINATOR_CONSTANTS)
        assert set(defined.values()) == set(_DENOMINATOR_VALUES)
        assert metrics.SCORE_KEY_FIELD == "score_key"
        assert metrics.DENOMINATOR_FIELD == "denominator_policy"


class TestCheckReferenceKind:
    """The guard on `reference_kind` vs how a task actually builds its judgement.

    The field is additive with a default, so a mis-declaration is silent in both
    directions: declared `value` on a procedural task and `missing_reference`
    fires on every sample of the benchmark; declared `procedure` on a value task
    and the rule never runs at all.
    """

    def _run(self, tmp_path: Path, source: str, **extra: str) -> list[CheckResult]:
        tasks_dir = tmp_path / "sieval" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "__init__.py").write_text("")
        (tasks_dir / "demo_0shot_gen.py").write_text(source)
        for rel, text in extra.items():
            (tasks_dir / f"{rel}.py").write_text(text)
        return PreflightRunner(project_root=tmp_path).check_reference_kind()

    def _one(self, tmp_path: Path, source: str, **extra: str) -> CheckResult:
        results = self._run(tmp_path, source, **extra)
        assert len(results) == 1
        return results[0]

    @staticmethod
    def _task(kind: str | None, reference: str) -> str:
        declaration = "" if kind is None else f"    reference_kind={kind!r},\n"
        return (
            "@sieval_task(\n"
            "    name='demo_0shot_gen',\n"
            f"{declaration}"
            ")\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            f"        return True, build_judgement_record({reference}, [])\n"
        )

    # --- the two directions a mis-declaration goes --------------------------

    def test_procedural_task_declaring_procedure_passes(self, tmp_path: Path):
        r = self._one(tmp_path, self._task("procedure", "None"))
        assert r.status == "PASS"
        assert "all 1 task(s)" in r.message

    def test_value_task_declaring_nothing_passes(self, tmp_path: Path):
        # The default is `value`, so an undeclared value task is correct — the
        # check must not demand a declaration it does not need.
        r = self._one(tmp_path, self._task(None, "raw['answer']"))
        assert r.status == "PASS"

    def test_procedural_task_left_undeclared_fails(self, tmp_path: Path):
        """The expensive direction: the rule would flag every sample."""
        r = self._one(tmp_path, self._task(None, "None"))
        assert r.status == "FAIL"
        assert "declares reference_kind='value'" in r.details[0]
        assert "passes a literal None" in r.details[0]
        assert "declare 'procedure'" in r.details[0]

    def test_value_task_declaring_procedure_fails(self, tmp_path: Path):
        """The silent direction: the rule never runs."""
        r = self._one(tmp_path, self._task("procedure", "raw['answer']"))
        assert r.status == "FAIL"
        assert "declares reference_kind='procedure'" in r.details[0]
        assert "passes a real reference" in r.details[0]
        assert "declare 'value'" in r.details[0]

    def test_a_task_with_both_call_sites_is_a_value(self, tmp_path: Path):
        """`ugmathbench_0shot_gen_fixed`'s shape, and the reason for the rule.

        It judges a missing `raw_sample` with `None` and everything else with its
        gold. That `None` branch *is* the anomaly, so reading it as a procedure
        declaration would silence the report on the one task known to need it.
        """
        source = (
            "@sieval_task(\n    name='demo_0shot_gen',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        if ctx.raw_sample is None:\n"
            "            return True, build_judgement_record(None, [])\n"
            "        return True, build_judgement_record(ctx.raw_sample['a'], [])\n"
        )
        assert self._one(tmp_path, source).status == "PASS"
        assert (
            self._one(
                tmp_path.__class__(str(tmp_path) + "2"),
                source.replace(
                    "    name='demo_0shot_gen',\n",
                    "    name='demo_0shot_gen',\n    reference_kind='procedure',\n",
                ),
            ).status
            == "FAIL"
        )

    # --- the keyword spelling and the shapes the record is built in ---------

    def test_the_keyword_form_reads_the_same_as_positional(self, tmp_path: Path):
        # `reference` is the builder's first parameter, so both spellings occur;
        # reading only the positional one would exempt the other.
        r = self._one(
            tmp_path,
            "@sieval_task(\n    name='demo_0shot_gen',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True, build_judgement_record(\n"
            "            rollouts=[], reference=None\n"
            "        )\n",
        )
        assert r.status == "FAIL"
        assert "passes a literal None" in r.details[0]

    def test_a_call_passing_no_reference_at_all_is_skipped(self, tmp_path: Path):
        # It does not type-check; treating a missing argument as a declaration
        # would report a syntax problem as a mis-declaration.
        results = self._run(
            tmp_path,
            "@sieval_task(\n    name='demo_0shot_gen',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True, build_judgement_record()\n",
        )
        assert {r.status for r in results} == {"WARN", "PASS"}

    def test_a_declaration_in_a_shared_helper_is_followed(self, tmp_path: Path):
        """`arc/`'s layout: four leaves share one `_base.arc_judgement_record`."""
        r = self._one(
            tmp_path,
            "from ._base import demo_judgement\n"
            "@sieval_task(\n    name='demo_0shot_gen',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True, demo_judgement(ctx)\n",
            _base="def demo_judgement(ctx):\n"
            "    return build_judgement_record(None, [])\n",
        )
        assert r.status == "FAIL"
        assert "found via helper" in r.details[0]

    def test_two_helpers_disagreeing_are_both_read(self, tmp_path: Path):
        """Every call site within a tier is read, not just the first.

        `ugmathbench`'s shape again, split across two helpers instead of two
        branches of one method: stopping at the first hit would read whichever
        helper sorted first as the whole story, and half the time call a value
        task procedural.
        """
        r = self._one(
            tmp_path,
            "from ._base import a_judgement, z_judgement\n"
            "@sieval_task(\n    name='demo_0shot_gen',\n"
            "    reference_kind='procedure',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        if ctx.raw_sample is None:\n"
            "            return True, a_judgement(ctx)\n"
            "        return True, z_judgement(ctx)\n",
            _base="def a_judgement(ctx):\n"
            "    return build_judgement_record(None, [])\n"
            "def z_judgement(ctx):\n"
            "    return build_judgement_record(ctx.raw_sample['a'], [])\n",
        )
        assert r.status == "FAIL"
        assert "declare 'value'" in r.details[0]

    def test_an_inline_none_branch_does_not_outvote_a_helper_gold(self, tmp_path: Path):
        """`ugmathbench`'s split, but across two tiers instead of one method.

        The dangerous direction: were the class body to win outright, that lone
        inline `None` would read as the whole story, and the FAIL would tell the
        author to declare `procedure` — the one edit that switches
        `missing_reference` off for the entire benchmark. Unioning the tiers
        keeps the verdict `value`, where a mis-declaration is loud instead.
        """
        r = self._one(
            tmp_path,
            "from ._base import z_judgement\n"
            "@sieval_task(\n    name='demo_0shot_gen',\n"
            "    reference_kind='value',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        if ctx.raw_sample is None:\n"
            "            return True, build_judgement_record(None, [])\n"
            "        return True, z_judgement(ctx)\n",
            _base="def z_judgement(ctx):\n"
            "    return build_judgement_record(ctx.raw_sample['a'], [])\n",
        )
        assert r.status == "PASS"

    def test_via_names_every_tier_that_contributed(self, tmp_path: Path):
        # Same shape mis-declared, so the detail line is visible: a task split
        # across tiers must not be reported as if only one had been read.
        r = self._one(
            tmp_path,
            "from ._base import z_judgement\n"
            "@sieval_task(\n    name='demo_0shot_gen',\n"
            "    reference_kind='procedure',\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        if ctx.raw_sample is None:\n"
            "            return True, build_judgement_record(None, [])\n"
            "        return True, z_judgement(ctx)\n",
            _base="def z_judgement(ctx):\n"
            "    return build_judgement_record(ctx.raw_sample['a'], [])\n",
        )
        assert r.status == "FAIL"
        assert "found via class+helper" in r.details[0]
        assert "declare 'value'" in r.details[0]

    def test_an_inherited_feedback_is_followed(self, tmp_path: Path):
        """`platinum_bench`'s layout: five leaves define no `feedback` at all."""
        r = self._one(
            tmp_path,
            "from ._base import DemoBase\n"
            "@sieval_task(\n    name='demo_0shot_gen',\n)\n"
            "class DemoTask(DemoBase):\n"
            "    pass\n",
            _base="class DemoBase:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True, build_judgement_record(None, [])\n",
        )
        assert r.status == "FAIL"
        assert "found via base" in r.details[0]

    # --- reach: what the check cannot read, it must not call clean ----------

    def test_an_unreachable_task_warns_rather_than_passing_silently(
        self, tmp_path: Path
    ):
        """A task it cannot read is a task the rule may be mis-routed on.

        Passing quietly would let the check's own blind spot read as a clean bill
        of health for the tasks inside it.
        """
        results = self._run(
            tmp_path,
            "@sieval_task(\n    name='demo_0shot_gen',\n)\nclass DemoTask:\n    pass\n",
        )
        statuses = {r.status for r in results}
        assert statuses == {"WARN", "PASS"}
        warn = next(r for r in results if r.status == "WARN")
        assert "reference_kind unverified" in warn.message
        assert "DemoTask" in warn.details[0]

    def test_an_undecorated_class_is_not_a_task(self, tmp_path: Path):
        # An abstract base builds records but registers nothing; demanding a
        # declaration there would fail a class that has no `reference_kind`.
        results = self._run(
            tmp_path,
            "class Helper:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True, build_judgement_record(None, [])\n",
        )
        assert [r.status for r in results] == ["PASS"]
        assert "all 0 task(s)" in results[0].message

    def test_a_non_literal_declaration_is_skipped_not_guessed(self, tmp_path: Path):
        # `reference_kind=SOME_CONSTANT` is unreadable statically; reporting it
        # as a mismatch would be inventing a verdict.
        results = self._run(
            tmp_path,
            "@sieval_task(\n    name='demo_0shot_gen',\n"
            "    reference_kind=KIND,\n)\n"
            "class DemoTask:\n"
            "    async def feedback(self, post, ctx):\n"
            "        return True, build_judgement_record(None, [])\n",
        )
        assert [r.status for r in results] == ["PASS"]
        assert "all 0 task(s)" in results[0].message

    def test_unparsable_module_reported_not_raised(self, tmp_path: Path):
        r = self._one(tmp_path, "class Broken(:\n")
        assert r.status == "FAIL"
        assert "could not parse" in r.details[0]

    def test_no_task_modules_skips(self, tmp_path: Path):
        (tmp_path / "sieval" / "tasks").mkdir(parents=True)
        results = PreflightRunner(project_root=tmp_path).check_reference_kind()
        assert [r.status for r in results] == ["SKIP"]

    def test_the_check_is_registered(self):
        assert "check_reference_kind" in PreflightRunner.ALL_CHECKS


class TestCheckRecordKeyAccess:
    """The guard on `[]` access to a rollout key that is absent on disk."""

    #: Minimal stand-in for sieval/core/tasks/records.py. The NotRequired
    #: annotations on all five record TypedDicts are read, not just the rollout
    #: ones -- a key on PromptRecord/JudgementRecord must be classified too.
    RECORDS = (
        "class PromptRecord(TypedDict):\n"
        "    prompt: JSONValue\n"
        "    reference: NotRequired[JSONValue | None]\n"
        "    extra: NotRequired[dict]\n"
        "\n"
        "class RolloutPrediction(TypedDict):\n"
        "    index: int\n"
        "    prediction: NotRequired[JSONValue | None]\n"
        "    extracted: bool\n"
        "    extra: NotRequired[dict]\n"
        "\n"
        "class PredictionRecord(TypedDict):\n"
        "    rollouts: list[RolloutPrediction]\n"
        "    extra: NotRequired[dict]\n"
        "\n"
        "class RolloutJudgement(TypedDict):\n"
        "    index: int\n"
        "    correct: bool\n"
        "    score: NotRequired[float]\n"
        "    metrics: NotRequired[dict[str, bool | float]]\n"
        "    extra: NotRequired[dict]\n"
        "\n"
        "class JudgementRecord(TypedDict):\n"
        "    reference: NotRequired[JSONValue | None]\n"
        "    rollouts: list[RolloutJudgement]\n"
        "    n_rollouts: int\n"
        "    n_correct: int\n"
        "    score: NotRequired[float]\n"
        "    metrics: NotRequired[dict[str, bool | float]]\n"
        "    extra: NotRequired[dict]\n"
    )

    def _run(
        self, tmp_path: Path, source: str, records: str | None = None
    ) -> CheckResult:
        records_dir = tmp_path / "sieval" / "core" / "tasks"
        records_dir.mkdir(parents=True)
        (records_dir / "records.py").write_text(
            self.RECORDS if records is None else records
        )
        tasks_dir = tmp_path / "sieval" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "demo_0shot_gen.py").write_text(source)
        results = PreflightRunner(project_root=tmp_path).check_record_key_access()
        assert len(results) == 1
        return results[0]

    # --- the two shapes a rollout reaches a task through -------------------

    def test_indexed_rollout_access_flagged(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    return post['rollouts'][0]['prediction']\n",
        )
        assert r.status == "FAIL"
        assert len(r.details) == 1
        assert "demo_0shot_gen.py:2" in r.details[0]

    def test_iterated_rollout_access_flagged(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    for rollout in post['rollouts']:\n"
            "        print(rollout['prediction'])\n",
        )
        assert r.status == "FAIL"
        assert len(r.details) == 1

    def test_comprehension_binding_flagged(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "async def report(self, finals, fails):\n"
            "    return [r['prediction'] for r in finals[0]['rollouts']]\n",
        )
        assert r.status == "FAIL"

    def test_get_rollouts_binding_flagged(self, tmp_path: Path):
        # `(f.feedback_result or {}).get("rollouts", [])` is the report-stage idiom.
        r = self._run(
            tmp_path,
            "async def report(self, finals, fails):\n"
            "    for r in (finals[0] or {}).get('rollouts', []):\n"
            "        print(r['prediction'])\n",
        )
        assert r.status == "FAIL"

    def test_enumerate_tuple_target_flagged(self, tmp_path: Path):
        # The natural way to write feedback for a task that needs the rollout
        # index. Needs both a tuple target and unwrapping the enumerate() call.
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    for i, rollout in enumerate(post['rollouts']):\n"
            "        print(i, rollout['prediction'])\n",
        )
        assert r.status == "FAIL"
        assert len(r.details) == 1

    def test_zip_tuple_target_flagged(self, tmp_path: Path):
        # The rollout list is the second zip argument, so every argument has to
        # be considered, not just the first.
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    for gold, rollout in zip(golds, post['rollouts']):\n"
            "        print(gold, rollout['prediction'])\n",
        )
        assert r.status == "FAIL"

    def test_passthrough_builtin_flagged(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    for rollout in reversed(post['rollouts']):\n"
            "        print(rollout['prediction'])\n",
        )
        assert r.status == "FAIL"

    def test_walrus_flagged(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    for rollout in (rs := post['rollouts']):\n"
            "        print(rollout['prediction'], len(rs))\n",
        )
        assert r.status == "FAIL"

    # --- what must NOT be flagged ------------------------------------------

    def test_get_access_passes(self, tmp_path: Path):
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    return post['rollouts'][0].get('prediction')\n",
        )
        assert r.status == "PASS"

    def test_non_record_dict_not_flagged(self, tmp_path: Path):
        # t_eval's `datum` carries a `prediction` key beside `ground_truth`; it is
        # a task-local payload, not a PredictionRecord. Name-based matching would
        # flag it and force a `.get()` that turns a real bug into a silent None.
        r = self._run(
            tmp_path,
            "def _process(self, datum: dict):\n"
            "    return datum['prediction'], datum['ground_truth']\n",
        )
        assert r.status == "PASS"

    def test_required_rollout_keys_not_flagged(self, tmp_path: Path):
        # `index` / `correct` are Required, so `[]` on them is safe.
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    for rollout in post['rollouts']:\n"
            "        print(rollout['index'], rollout['extracted'])\n",
        )
        assert r.status == "PASS"

    def test_ungated_keys_not_flagged(self, tmp_path: Path):
        # `extra` is NotRequired but deliberately ungated: every read is nested,
        # so a mechanical `.get()` would swap KeyError for TypeError.
        r = self._run(
            tmp_path,
            "async def report(self, finals, fails):\n"
            "    for r in finals[0]['rollouts']:\n"
            "        print(r['extra']['grade'], r['score'], r['metrics'])\n",
        )
        assert r.status == "PASS"

    def test_reference_not_flagged(self, tmp_path: Path):
        # `reference` is classified but ungated: it is not read off a rollout, so
        # gating it would be inert. ruler's `list(fb["reference"])` stays valid.
        r = self._run(
            tmp_path,
            "async def report(self, finals, fails):\n"
            "    fb = finals[0].feedback_result\n"
            "    return list(fb['reference'])\n",
        )
        assert r.status == "PASS"

    def test_known_limit_aliased_rollout_not_flagged(self, tmp_path: Path):
        # Documented limit: following a rollout through a local name needs
        # dataflow. Pinned so the gap is a decision, not a surprise.
        r = self._run(
            tmp_path,
            "async def feedback(self, post, ctx):\n"
            "    rollout = post['rollouts'][0]\n"
            "    return rollout['prediction']\n",
        )
        assert r.status == "PASS"

    # --- the self-maintaining half -----------------------------------------

    def test_unclassified_notrequired_key_fails(self, tmp_path: Path):
        # A NotRequired rollout key in neither _GATED nor _UNGATED must fail
        # rather than be silently skipped: that is what forces whoever adds a
        # key to records.py to decide whether `[]` on it is a latent KeyError.
        r = self._run(
            tmp_path,
            "x = 1\n",
            records=self.RECORDS.replace(
                "    extracted: bool\n",
                "    extracted: bool\n    confidence: NotRequired[float]\n",
            ),
        )
        assert r.status == "FAIL"
        assert "confidence" in r.details[0]

    def test_unclassified_key_on_non_rollout_record_fails(self, tmp_path: Path):
        # Classification spans all five record classes, so a key added to
        # JudgementRecord must be decided on too -- the hole `reference` sat in.
        r = self._run(
            tmp_path,
            "x = 1\n",
            records=self.RECORDS.replace(
                "    n_rollouts: int\n",
                "    n_rollouts: int\n    rubric: NotRequired[str]\n",
            ),
        )
        assert r.status == "FAIL"
        assert "rubric" in r.details[0]

    def test_missing_records_module_fails(self, tmp_path: Path):
        # FAIL, not SKIP: records.py is the check's subject. A rename must not
        # silently narrow the gate to nothing while preflight stays green.
        (tmp_path / "sieval" / "tasks").mkdir(parents=True)
        results = PreflightRunner(project_root=tmp_path).check_record_key_access()
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "not readable" in results[0].message

    def test_unparsable_module_reported_not_raised(self, tmp_path: Path):
        r = self._run(tmp_path, "def broken(:\n")
        assert r.status == "FAIL"
        assert "could not parse" in r.details[0]

    def test_real_modules_pass(self):
        """Integration: the shipped tree satisfies both halves of the check.

        No module indexes a gated rollout key, and the real ``records.py``
        declares no ``NotRequired`` key the check cannot place.
        """
        results = PreflightRunner().check_record_key_access()
        assert [r.status for r in results] == ["PASS"]


class TestCheckMutmutConfig:
    """`[tool.mutmut]` must still yield an importable `mutants/` copy."""

    def _write(self, tmp_path, body: str) -> PreflightRunner:
        (tmp_path / "pyproject.toml").write_text(body, encoding="utf-8")
        return PreflightRunner(project_root=tmp_path)

    def test_missing_package_root_fails(self, tmp_path):
        # The first regression: also_copy lists the subpackages but not
        # sieval/__init__.py, so mutants/sieval is a directory rather than a
        # package and every mutation run dies during stats collection.
        runner = self._write(
            tmp_path,
            '[tool.mutmut]\npaths_to_mutate = ["sieval/core"]\n'
            'also_copy = ["sieval/tasks", "scripts"]\n',
        )
        results = runner.check_mutmut_config()
        assert [r.status for r in results] == ["FAIL"]
        assert any("sieval/__init__.py" in d for d in results[0].details)

    def test_missing_scripts_fails(self, tmp_path):
        # The second, found the same way: `scripts/` is not a package, so
        # tests/unit/scripts/ puts it on sys.path by walking up from __file__.
        # In the copy that is mutants/scripts, which without this entry does not
        # exist -- and the run dies on `No module named 'check_layer_imports'`,
        # which reads as a broken test rather than a missing copy path.
        runner = self._write(
            tmp_path,
            '[tool.mutmut]\npaths_to_mutate = ["sieval/core"]\n'
            'also_copy = ["sieval/__init__.py", "sieval/tasks"]\n',
        )
        results = runner.check_mutmut_config()
        assert [r.status for r in results] == ["FAIL"]
        assert any("scripts" in d for d in results[0].details)

    def test_every_required_path_present_passes(self, tmp_path):
        runner = self._write(
            tmp_path,
            '[tool.mutmut]\npaths_to_mutate = ["sieval/core"]\n'
            'also_copy = ["sieval/__init__.py", "scripts", "sieval/tasks"]\n',
        )
        assert [r.status for r in runner.check_mutmut_config()] == ["PASS"]

    def test_a_parent_entry_carries_the_package_root(self, tmp_path):
        # Copying "sieval" wholesale brings __init__.py with it.
        runner = self._write(
            tmp_path,
            '[tool.mutmut]\npaths_to_mutate = ["sieval"]\nalso_copy = ["scripts"]\n',
        )
        assert [r.status for r in runner.check_mutmut_config()] == ["PASS"]

    def test_no_mutmut_section_is_not_a_failure(self, tmp_path):
        runner = self._write(tmp_path, "[project]\nname = 'x'\n")
        assert [r.status for r in runner.check_mutmut_config()] == ["PASS"]

    def test_missing_pyproject_fails(self, tmp_path):
        runner = PreflightRunner(project_root=tmp_path)
        assert [r.status for r in runner.check_mutmut_config()] == ["FAIL"]

    def test_the_repo_config_is_registered_and_passes(self):
        # Registration is the half that makes it run in CI; without it the
        # function is dead code that reports nothing.
        assert "check_mutmut_config" in PreflightRunner.ALL_CHECKS
        assert [r.status for r in PreflightRunner().check_mutmut_config()] == ["PASS"]
