"""Tests for leaderboard scan (list command backend + CLI).

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from pathlib import Path

from typer.testing import CliRunner

from sieval.cli.leaderboard.catalog import LeaderboardSummary, scan_leaderboards
from sieval.cli.main import app

cli_runner = CliRunner()


class TestScanLeaderboards:
    def _write_yaml(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_empty_directory(self, tmp_path: Path) -> None:
        summaries = scan_leaderboards(tmp_path)
        assert summaries == []

    def test_nonexistent_directory_returns_empty(self, tmp_path: Path) -> None:
        summaries = scan_leaderboards(tmp_path / "nope")
        assert summaries == []

    def test_valid_yaml(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "sample.yaml",
            """
alignment:
  card: alignment/sample.md
models:
  base_model: {name: m-1, args: {max_tokens: 16}}
  math_model: {base: base_model}
datasets:
  aime_2024: {class: AIME2024Dataset, path: ./data/aime}
  math_500: {class: MATH500Dataset, path: ./data/math}
tasks:
  aime_2024_0shot_gen: {class: AIME2024ZeroShotGenTask, dataset: aime_2024}
  math_500_0shot_gen: {class: MATH500ZeroShotGenTask, dataset: math_500}
""",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert isinstance(s, LeaderboardSummary)
        assert s.name == "sample"
        assert s.path == tmp_path / "sample.yaml"
        assert s.models == ["base_model", "math_model"]
        assert s.tasks == ["aime_2024_0shot_gen", "math_500_0shot_gen"]
        assert s.alignment_card == "alignment/sample.md"
        assert s.error is None

    def test_tasks_counted_independently_of_datasets(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "multi.yaml",
            """
models: {m: {name: m}}
datasets:
  shared: {class: T, path: ./data}
tasks:
  shared_em: {class: TExactMatch, dataset: shared}
  shared_f1: {class: TF1, dataset: shared}
  shared_pass_at_k: {class: TPassK, dataset: shared}
""",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.tasks == ["shared_em", "shared_f1", "shared_pass_at_k"]

    def test_no_alignment_block(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "self_built.yaml",
            """
models: {m: {name: m, args: {max_tokens: 16}}}
datasets: {t: {class: T, path: ./data}}
tasks: {t_em: {class: TExactMatch, dataset: t}}
""",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        assert summaries[0].alignment_card is None

    def test_non_string_alignment_card_populates_error(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "weird.yaml",
            "alignment: {card: [not, a, string]}\n"
            "models: {m: {name: m}}\n"
            "tasks: {t: {class: T}}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.alignment_card is None
        assert s.error is not None
        assert "alignment.card" in s.error
        # Card error is scoped — rest of the file still parses.
        assert s.models == ["m"]
        assert s.tasks == ["t"]

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        self._write_yaml(tmp_path / "bad.yaml", "models: {m: [:\n")
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.name == "bad"
        assert s.error is not None
        assert "parse" in s.error.lower() or "yaml" in s.error.lower()
        assert s.models == []
        assert s.tasks == []

    def test_binary_file_does_not_abort_scan(self, tmp_path: Path) -> None:
        # UnicodeDecodeError must surface as a row error, not kill the scan.
        (tmp_path / "binary.yaml").write_bytes(b"\xff\xfe\x00\x00not utf-8")
        self._write_yaml(
            tmp_path / "good.yaml",
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert {s.name for s in summaries} == {"binary", "good"}
        by_name = {s.name: s for s in summaries}
        assert by_name["binary"].error is not None
        assert "read error" in by_name["binary"].error
        assert by_name["good"].error is None
        assert by_name["good"].models == ["m"]

    def test_non_dict_models_populates_error(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "bad_models.yaml",
            "models: [a, b]\ntasks: {t: {class: T}}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.error is not None
        assert "models" in s.error
        assert "mapping" in s.error
        assert s.models == []
        # Other blocks still parse.
        assert s.tasks == ["t"]

    def test_non_dict_tasks_populates_error(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "bad_tasks.yaml",
            "models: {m: {name: m}}\ntasks: [x, y]\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.error is not None
        assert "tasks" in s.error
        assert "mapping" in s.error
        assert s.models == ["m"]
        assert s.tasks == []

    def test_non_dict_alignment_populates_error(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "bad_align.yaml",
            "alignment: hello\nmodels: {m: {name: m}}\ntasks: {t: {class: T}}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.error is not None
        assert "alignment" in s.error
        assert s.alignment_card is None
        assert s.models == ["m"]
        assert s.tasks == ["t"]

    def test_multiple_errors_concatenated(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "multi_err.yaml",
            "models: [a]\ntasks: x\nalignment: {card: [not_a_string]}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.error is not None
        assert "models" in s.error
        assert "tasks" in s.error
        assert "alignment.card" in s.error

    def test_yml_extension_also_scanned(self, tmp_path: Path) -> None:
        self._write_yaml(
            tmp_path / "short.yml",
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
        )
        self._write_yaml(
            tmp_path / "long.yaml",
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert {s.name for s in summaries} == {"short", "long"}

    def test_subdirectories_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "alignment").mkdir()
        self._write_yaml(
            tmp_path / "alignment" / "buried.yaml",
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
        )
        self._write_yaml(
            tmp_path / "top.yaml",
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
        )
        summaries = scan_leaderboards(tmp_path)
        assert len(summaries) == 1
        assert summaries[0].name == "top"

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        for name in ["zebra", "alpha", "middle"]:
            self._write_yaml(
                tmp_path / f"{name}.yaml",
                "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
            )
        summaries = scan_leaderboards(tmp_path)
        assert [s.name for s in summaries] == ["alpha", "middle", "zebra"]


class TestListCLI:
    def _setup_leaderboards_dir(self, tmp_path: Path) -> Path:
        lb_dir = tmp_path / "leaderboards"
        lb_dir.mkdir()
        (lb_dir / "sft_fast.yaml").write_text(
            "models: {a: {name: a}, b: {name: b}, c: {name: c}}\n"
            "datasets:\n"
            "  ds1: {class: T}\n"
            "  ds2: {class: T}\n"
            "tasks:\n"
            "  t1: {class: T, dataset: ds1}\n"
            "  t2: {class: T, dataset: ds1}\n"
            "  t3: {class: T, dataset: ds2}\n"
            "  t4: {class: T, dataset: ds2}\n",
            encoding="utf-8",
        )
        (lb_dir / "qwen3_mini.yaml").write_text(
            "alignment: {card: alignment/qwen3-x.md}\n"
            "models: {q: {name: q}}\n"
            "datasets: {d: {class: T}}\n"
            "tasks: {t: {class: T, dataset: d}}\n",
            encoding="utf-8",
        )
        (lb_dir / "broken.yaml").write_text("[unbalanced\n", encoding="utf-8")
        return lb_dir

    def test_text_mode_shows_all_rows(self, tmp_path: Path, monkeypatch) -> None:
        self._setup_leaderboards_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list"])
        assert result.exit_code == 0, result.output
        assert "sft_fast" in result.output
        assert "qwen3_mini" in result.output
        assert "broken" in result.output
        assert "NAME" in result.output
        assert "MODELS" in result.output
        assert "TASKS" in result.output
        assert "PATH" in result.output
        # sft_fast: 3 models × 4 tasks on 2 datasets — checking the numbers
        # discriminates against future len() regressions.
        sft_row = next(
            line for line in result.output.splitlines() if "sft_fast" in line
        )
        assert " 3 " in sft_row, sft_row
        assert " 4 " in sft_row, sft_row

    def test_text_mode_marks_malformed(self, tmp_path: Path, monkeypatch) -> None:
        self._setup_leaderboards_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list"])
        assert "[malformed]" in result.output

    def test_text_mode_shows_error_reason(self, tmp_path: Path, monkeypatch) -> None:
        self._setup_leaderboards_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list"])
        assert "Errors:" in result.output
        errors_idx = result.output.index("Errors:")
        errors_section = result.output[errors_idx:]
        assert "broken" in errors_section
        # YAML parse error should be surfaced in the errors section.
        assert "parse" in errors_section.lower() or "yaml" in errors_section.lower()

    def test_positional_dir_scans_custom_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        custom = tmp_path / "other_dir"
        custom.mkdir()
        (custom / "x.yaml").write_text(
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n",
            encoding="utf-8",
        )
        # cwd has no ./leaderboards — positional arg must be honored.
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list", str(custom)])
        assert result.exit_code == 0, result.output
        assert "x" in result.output
        assert "not found" not in result.output.lower()

    def test_json_mode_carries_alignment_card(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._setup_leaderboards_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list", "-o", "json"])
        assert result.exit_code == 0, result.output
        import json

        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert payload["command"] == "leaderboard.list"
        rows = payload["data"]["leaderboards"]
        by_name = {r["name"]: r for r in rows}
        assert by_name["qwen3_mini"]["alignment_card"] == "alignment/qwen3-x.md"
        assert by_name["sft_fast"]["alignment_card"] is None
        assert by_name["broken"]["error"]  # non-empty string
        assert by_name["sft_fast"]["models"] == ["a", "b", "c"]
        assert by_name["sft_fast"]["tasks"] == ["t1", "t2", "t3", "t4"]

    def test_empty_leaderboards_dir(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "leaderboards").mkdir()
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list"])
        assert result.exit_code == 0
        assert "No leaderboards found" in result.output

    def test_missing_leaderboards_dir_warns(self, tmp_path: Path, monkeypatch) -> None:
        """cwd without a leaderboards/ dir: exit 0 + warning (not silent empty)."""
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list"])
        assert result.exit_code == 0
        # CliRunner.output merges stdout+stderr, so the loguru warning surfaces here.
        assert "not found" in result.output.lower()
        assert "No leaderboards found" in result.output

    def test_file_path_warns(self, tmp_path: Path, monkeypatch) -> None:
        """Positional arg pointing at a file (not a dir) warns, not silent empty."""
        target = tmp_path / "not_a_dir.yaml"
        target.write_text(
            "models: {m: {name: m}}\ntasks: {t: {class: T}}\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["leaderboard", "list", str(target)])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()
        assert "No leaderboards found" in result.output

    def test_missing_leaderboards_dir_json_carries_warning(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Click 8.3 split stdout/stderr — parse .stdout to keep the logger
        # warning (on stderr) out of the JSON payload.
        result = cli_runner.invoke(app, ["leaderboard", "list", "-o", "json"])
        assert result.exit_code == 0
        import json

        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["warnings"]
        assert any("not found" in w.lower() for w in payload["warnings"])
