"""Tests for leaderboard scan, model resolution, and matrix assembly.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from pathlib import Path

import pytest

from sieval.cli.leaderboard import (
    LeaderboardMatrix,
    RunInfo,
    build_matrix,
    resolve_model_name,
    scan_runs,
)

# ---------------------------------------------------------------------------
# scan_runs
# ---------------------------------------------------------------------------


class TestScanRunsPatternA:
    """Pattern A: {root}/{task_name}/{14-digit-timestamp}/report.json"""

    def test_single_run(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "mmlu" / "20260412120000"
        task_dir.mkdir(parents=True)
        report = {"accuracy": 0.85}
        (task_dir / "report.json").write_text(json.dumps(report))

        runs = scan_runs([tmp_path])

        assert len(runs) == 1
        assert runs[0].task_name == "mmlu"
        assert runs[0].run_id == "20260412120000"
        assert runs[0].run_dir == task_dir
        assert runs[0].report == report


class TestScanRunsPatternB:
    """Pattern B: no timestamp parent → task_name = parent dir."""

    def test_no_timestamp(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "some_prefix" / "hellaswag"
        task_dir.mkdir(parents=True)
        report = {"accuracy": 0.70}
        (task_dir / "report.json").write_text(json.dumps(report))

        runs = scan_runs([tmp_path])

        assert len(runs) == 1
        assert runs[0].task_name == "hellaswag"
        assert runs[0].run_id == "hellaswag"
        assert runs[0].run_dir == task_dir


class TestScanRunsMultipleDirs:
    """Multiple input directories produce a union of runs."""

    def test_union(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        (dir_a / "mmlu" / "20260412120000").mkdir(parents=True)
        (dir_a / "mmlu" / "20260412120000" / "report.json").write_text(
            json.dumps({"acc": 1})
        )
        (dir_b / "gsm8k" / "20260412130000").mkdir(parents=True)
        (dir_b / "gsm8k" / "20260412130000" / "report.json").write_text(
            json.dumps({"acc": 2})
        )

        runs = scan_runs([dir_a, dir_b])

        task_names = {r.task_name for r in runs}
        assert task_names == {"mmlu", "gsm8k"}


class TestScanRunsEmpty:
    """No report.json found → empty list."""

    def test_no_reports(self, tmp_path: Path) -> None:
        (tmp_path / "empty_dir").mkdir()
        assert scan_runs([tmp_path]) == []


class TestScanRunsMalformed:
    """Malformed report.json → skipped with warning."""

    def test_malformed_json(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "mmlu" / "20260412120000"
        task_dir.mkdir(parents=True)
        (task_dir / "report.json").write_text("NOT JSON {{{")

        runs = scan_runs([tmp_path])

        assert runs == []


# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    """resolve_model_name extracts model name from JSONL or falls back."""

    def test_successful_extraction(self, tmp_path: Path) -> None:
        iteration_dir = tmp_path / "iteration_0" / "final"
        iteration_dir.mkdir(parents=True)
        record = {"infer_result": {"model": {"model": "Qwen/Qwen3-8B"}, "text": "hi"}}
        (iteration_dir / "0.jsonl").write_text(json.dumps(record))

        assert resolve_model_name(tmp_path) == "Qwen/Qwen3-8B"

    def test_fallback_no_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        assert resolve_model_name(run_dir) == "my_run"

    def test_fallback_empty_file(self, tmp_path: Path) -> None:
        iteration_dir = tmp_path / "iteration_0" / "final"
        iteration_dir.mkdir(parents=True)
        (iteration_dir / "0.jsonl").write_text("")

        assert resolve_model_name(tmp_path) == tmp_path.name

    def test_fallback_missing_field(self, tmp_path: Path) -> None:
        iteration_dir = tmp_path / "iteration_0" / "final"
        iteration_dir.mkdir(parents=True)
        (iteration_dir / "0.jsonl").write_text(json.dumps({"other": "data"}))

        assert resolve_model_name(tmp_path) == tmp_path.name

    def test_fallback_malformed_json(self, tmp_path: Path) -> None:
        iteration_dir = tmp_path / "0" / "final"
        iteration_dir.mkdir(parents=True)
        (iteration_dir / "0.jsonl").write_text("NOT VALID JSON {{{")

        assert resolve_model_name(tmp_path) == tmp_path.name


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------


def _make_run(
    task: str, run_id: str, model: str, report: dict | None = None
) -> RunInfo:
    return RunInfo(
        task_name=task,
        run_id=run_id,
        run_dir=Path(f"/fake/{task}/{run_id}"),
        report=report or {},
        model_name=model,
    )


class TestBuildMatrix:
    """build_matrix aggregates runs into a LeaderboardMatrix."""

    def test_basic(self) -> None:
        runs = [
            _make_run("mmlu", "20260412120000", "modelA", {"acc": 0.8}),
            _make_run("gsm8k", "20260412120000", "modelA", {"acc": 0.6}),
        ]
        m = build_matrix(runs)

        assert m["models"] == ["modelA"]
        assert m["tasks"] == ["gsm8k", "mmlu"]
        assert len(m["results"]) == 2

    def test_dedup_keeps_latest(self) -> None:
        runs = [
            _make_run("mmlu", "20260412100000", "modelA", {"acc": 0.7}),
            _make_run("mmlu", "20260412120000", "modelA", {"acc": 0.9}),
        ]
        m = build_matrix(runs)

        assert len(m["results"]) == 1
        assert m["results"][0]["run_id"] == "20260412120000"
        assert m["results"][0]["report"] == {"acc": 0.9}

    def test_multiple_models(self) -> None:
        runs = [
            _make_run("mmlu", "20260412120000", "modelA"),
            _make_run("mmlu", "20260412120000", "modelB"),
        ]
        m = build_matrix(runs)

        assert m["models"] == ["modelA", "modelB"]
        assert len(m["results"]) == 2

    def test_empty_input(self) -> None:
        m = build_matrix([])

        assert m["models"] == []
        assert m["tasks"] == []
        assert m["results"] == []

    def test_all_runs_keeps_everything(self) -> None:
        runs = [
            _make_run("mmlu", "20260412100000", "modelA", {"acc": 0.7}),
            _make_run("mmlu", "20260412120000", "modelA", {"acc": 0.9}),
        ]
        m = build_matrix(runs, all_runs=True)

        assert len(m["results"]) == 2


# ---------------------------------------------------------------------------
# Alignment attachment via effective_config.yaml
# ---------------------------------------------------------------------------

_CARD_BODY = """---
reference: {kind: tr, source: "arXiv:0000.00000", title: "Test Paper"}
tolerance: 3.0
reference_scores:
  model-a:
    sample_task: 50.0
---
"""


def _write_card(path: Path, body: str = _CARD_BODY) -> Path:
    card = path / "card.md"
    card.write_text(body, encoding="utf-8")
    return card


def _layout_with_alignment(
    tmp_path: Path,
    *,
    card_rel: str | None = "../card.md",
    write_card: bool = True,
    card_body: str = _CARD_BODY,
    model_name: str = "model-a",
) -> Path:
    """Build outputs/effective_config.yaml + one report.json, return outputs/."""
    if write_card:
        _write_card(tmp_path, card_body)

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    if card_rel is not None:
        (outputs / "effective_config.yaml").write_text(
            f"alignment:\n  card: {card_rel}\n"
            "models: {m: {name: m}}\n"
            "datasets: {sample_task: {class: T}}\n",
            encoding="utf-8",
        )
    task_dir = outputs / "sample_task" / "20260422000000"
    task_dir.mkdir(parents=True)
    (task_dir / "report.json").write_text('{"score": 48.1}', encoding="utf-8")

    # Create a model-name resolvable JSONL so resolve_model_name returns `model_name`.
    jsonl_dir = task_dir / "iter_0" / "final"
    jsonl_dir.mkdir(parents=True)
    (jsonl_dir / "0.jsonl").write_text(
        json.dumps({"infer_result": {"model": {"model": model_name}, "text": "x"}}),
        encoding="utf-8",
    )
    return outputs


def _matrix_with_alignment(outputs: Path) -> LeaderboardMatrix:
    """Run scan → resolve-model → build_matrix and return the matrix."""
    from dataclasses import replace

    runs = scan_runs([outputs])
    resolved = [replace(r, model_name=resolve_model_name(r.run_dir)) for r in runs]
    return build_matrix(resolved)


class TestAlignmentAnnotationPipeline:
    """build_matrix attaches annotations when effective_config cites a valid card."""

    def test_annotation_produced_for_matching_card(self, tmp_path: Path) -> None:
        outputs = _layout_with_alignment(tmp_path)
        m = _matrix_with_alignment(outputs)
        assert len(m["results"]) == 1
        ann = m["results"][0]["annotation"]
        assert ann is not None
        assert ann["reference"] == 50.0
        assert ann["tolerance"] == 3.0
        assert ann["status"] in {"pass", "fail"}

    def test_no_effective_config_means_no_annotation(self, tmp_path: Path) -> None:
        outputs = _layout_with_alignment(tmp_path, card_rel=None, write_card=False)
        m = _matrix_with_alignment(outputs)
        assert m["results"][0]["annotation"] is None

    def test_config_without_alignment_block_means_no_annotation(
        self, tmp_path: Path
    ) -> None:
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "effective_config.yaml").write_text(
            "models: {m: {name: m}}\ndatasets: {sample_task: {class: T}}\n",
            encoding="utf-8",
        )
        task_dir = outputs / "sample_task" / "20260422000000"
        task_dir.mkdir(parents=True)
        (task_dir / "report.json").write_text('{"score": 48.1}', encoding="utf-8")

        m = _matrix_with_alignment(outputs)
        assert m["results"][0]["annotation"] is None

    def test_missing_card_file_degrades_to_no_annotation(self, tmp_path: Path) -> None:
        """effective_config cites a missing card → scan survives, no annotation."""
        outputs = _layout_with_alignment(
            tmp_path, card_rel="../does_not_exist.md", write_card=False
        )
        m = _matrix_with_alignment(outputs)
        assert len(m["results"]) == 1
        assert m["results"][0]["annotation"] is None

    def test_malformed_card_degrades_to_no_annotation(self, tmp_path: Path) -> None:
        """Card with no YAML frontmatter → scan survives, no annotation."""
        outputs = _layout_with_alignment(
            tmp_path, card_body="no frontmatter here, just prose\n"
        )
        m = _matrix_with_alignment(outputs)
        assert m["results"][0]["annotation"] is None

    def test_model_not_in_card_means_no_annotation(self, tmp_path: Path) -> None:
        """Card is valid but doesn't reference this run's model."""
        outputs = _layout_with_alignment(tmp_path, model_name="unknown-model")
        m = _matrix_with_alignment(outputs)
        assert m["results"][0]["annotation"] is None

    def test_card_loaded_once_across_many_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shared effective_config → card load + yaml parse happen once, not per run."""
        from sieval.cli.leaderboard import scanner as scanner_mod

        _write_card(tmp_path)
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "effective_config.yaml").write_text(
            "alignment:\n  card: ../card.md\n", encoding="utf-8"
        )
        # Create 5 runs under the same effective_config.
        for i in range(5):
            td = outputs / f"task_{i}" / f"2026042200000{i}"
            td.mkdir(parents=True)
            (td / "report.json").write_text('{"score": 1.0}', encoding="utf-8")

        runs = scan_runs([outputs])
        assert len(runs) == 5

        call_count = {"n": 0}
        real_loader = scanner_mod._load_card_from_config

        def counting_loader(cfg: Path):
            call_count["n"] += 1
            return real_loader(cfg)

        monkeypatch.setattr(scanner_mod, "_load_card_from_config", counting_loader)
        build_matrix(runs)

        # Regression guard against per-run re-parsing: one shared cfg → one load.
        assert call_count["n"] == 1

    def test_annotations_land_across_many_runs(self, tmp_path: Path) -> None:
        """Many (model, task) cells share one card → each matching cell annotated.

        Complements :meth:`test_card_loaded_once_across_many_runs`, which only
        verifies the cache; this one verifies annotation values actually reach
        the matrix for every matching cell.
        """
        card_body = """---
reference: {kind: tr, source: "arXiv:0000.00000", title: "Test Paper"}
tolerance: 3.0
reference_scores:
  model-a:
    task_0: 10.0
    task_1: 20.0
  model-b:
    task_0: 30.0
---
"""
        _write_card(tmp_path, card_body)
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "effective_config.yaml").write_text(
            "alignment:\n  card: ../card.md\n", encoding="utf-8"
        )

        # (model, task, observed) triples: two cells in the card, one not.
        layout = [
            ("model-a", "task_0", 11.5, "20260422000001"),  # pass  (|1.5|<=3)
            ("model-a", "task_1", 25.0, "20260422000002"),  # fail  (|5|>3)
            ("model-b", "task_0", 29.0, "20260422000003"),  # pass  (|-1|<=3)
            ("model-c", "task_0", 42.0, "20260422000004"),  # no card entry
        ]
        for model_name, task_name, score, run_id in layout:
            task_dir = outputs / task_name / run_id
            task_dir.mkdir(parents=True)
            (task_dir / "report.json").write_text(
                json.dumps({"score": score}), encoding="utf-8"
            )
            jsonl_dir = task_dir / "iter_0" / "final"
            jsonl_dir.mkdir(parents=True)
            (jsonl_dir / "0.jsonl").write_text(
                json.dumps(
                    {"infer_result": {"model": {"model": model_name}, "text": "x"}}
                ),
                encoding="utf-8",
            )

        m = _matrix_with_alignment(outputs)

        by_cell = {(r["model"], r["task"]): r for r in m["results"]}
        ann_a0 = by_cell[("model-a", "task_0")]["annotation"]
        ann_a1 = by_cell[("model-a", "task_1")]["annotation"]
        ann_b0 = by_cell[("model-b", "task_0")]["annotation"]
        assert ann_a0 is not None
        assert ann_a1 is not None
        assert ann_b0 is not None
        assert ann_a0["status"] == "pass"
        assert ann_a1["status"] == "fail"
        assert ann_b0["status"] == "pass"
        # model-c is not in the card → no annotation, but cell still present.
        assert by_cell[("model-c", "task_0")]["annotation"] is None
