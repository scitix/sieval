"""Tests for leaderboard RULER effective-length aggregation + CLI command.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from sieval.cli.leaderboard.ruler import (
    collect_sweep,
    effective_length,
    len_tag,
    parse_length,
    reference_threshold,
    summarize,
)
from sieval.cli.leaderboard.scanner import RunInfo
from sieval.cli.main import app

cli_runner = CliRunner()

_TASKS_13 = [
    "ruler_niah_single_1",
    "ruler_niah_single_2",
    "ruler_niah_single_3",
    "ruler_niah_multikey_1",
    "ruler_niah_multikey_2",
    "ruler_niah_multikey_3",
    "ruler_niah_multivalue",
    "ruler_niah_multiquery",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpotqa",
]


def _runs(scores_by_len: dict[str, float], model: str = "m") -> list[RunInfo]:
    runs: list[RunInfo] = []
    for tag, score in scores_by_len.items():
        for t in _TASKS_13:
            runs.append(
                RunInfo(
                    task_name=f"{t}_{tag}",
                    run_id="20260101000000",
                    run_dir=Path("/tmp") / f"{t}_{tag}",
                    report={"score": score, "fails": 0},
                    model_name=model,
                )
            )
    return runs


def _write_sweep(root: Path, scores_by_len: dict[str, float]) -> None:
    for tag, score in scores_by_len.items():
        for t in _TASKS_13:
            d = root / f"{t}_{tag}" / "20260101000000"
            d.mkdir(parents=True, exist_ok=True)
            (d / "report.json").write_text(json.dumps({"score": score, "fails": 0}))


# ── pure logic ────────────────────────────────────────────────────────


def test_parse_length():
    assert parse_length("ruler_qa_squad_128k") == 131072
    assert parse_length("ruler_vt_4096") == 4096
    assert parse_length("ruler_niah_single_1_8k") == 8192
    assert parse_length("no_suffix_here") is None
    assert parse_length("plain_task") is None


def test_len_tag():
    assert len_tag(4096) == "4k"
    assert len_tag(131072) == "128k"
    assert len_tag(1000) == "1000"


def test_collect_sweep_groups_by_model_and_length():
    by_model = collect_sweep(_runs({"4k": 95.0, "128k": 60.0}))
    lengths = by_model["m"]
    assert sorted(lengths) == [4096, 131072]
    assert len(lengths[4096]) == 13
    assert sum(lengths[4096]) / 13 == 95.0


def test_collect_sweep_skips_unsuffixed_and_nonnumeric():
    runs = [
        RunInfo("plain_task", "r", Path("/tmp/a"), {"score": 90.0}, "m"),
        RunInfo("ruler_vt_4k", "r", Path("/tmp/b"), {"score": "bad"}, "m"),
        RunInfo("ruler_vt_4k", "r", Path("/tmp/c"), {"score": 80.0}, "m"),
    ]
    by_model = collect_sweep(runs)
    assert by_model["m"][4096] == [80.0]  # only the valid one


def test_effective_length_picks_longest_passing():
    avg = {4096: 95.0, 8192: 90.0, 16384: 80.0, 32768: 70.0}
    assert effective_length(avg, 85.6) == 8192
    assert effective_length(avg, 99.0) is None
    assert effective_length(avg, 50.0) == 32768


def test_effective_length_non_contiguous_takes_max_passing():
    avg = {4096: 95.0, 8192: 50.0, 16384: 90.0}
    assert effective_length(avg, 85.6) == 16384


def test_reference_threshold_uses_smallest_tier():
    ref = collect_sweep(_runs({"4k": 80.0, "8k": 70.0}))
    bar, base = reference_threshold(ref)
    assert base == 4096
    assert bar == 80.0


def test_reference_threshold_empty_is_none():
    assert reference_threshold({}) is None


def test_summarize_flags_incomplete_tiers():
    runs = _runs({"4k": 95.0})
    runs.append(RunInfo("ruler_vt_8k", "r", Path("/tmp"), {"score": 90.0}, "m"))
    summary = summarize(collect_sweep(runs), threshold=85.6)["m"]
    rows = {r["tag"]: r for r in summary["per_length"]}
    assert rows["4k"]["complete"] is True
    assert rows["8k"]["complete"] is False  # only 1/13
    assert summary["effective_length_tag"] == "8k"  # both pass → max


# ── CLI command ───────────────────────────────────────────────────────


def test_cli_reports_effective_length(tmp_path):
    _write_sweep(tmp_path, {"4k": 95.0, "128k": 60.0})
    res = cli_runner.invoke(
        app, ["leaderboard", "ruler-effective", str(tmp_path), "-o", "json"]
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)["data"]
    assert data["threshold"] == 85.6
    # On-disk runs carry no embedded model name; it resolves to the run-dir name.
    (summary,) = data["models"].values()
    assert summary["effective_length_tag"] == "4k"


def test_cli_threshold_from_overrides_bar(tmp_path):
    sweep = tmp_path / "sweep"
    ref = tmp_path / "ref"
    _write_sweep(sweep, {"4k": 95.0, "8k": 82.0})
    _write_sweep(ref, {"4k": 80.0})  # reference 4k avg → threshold 80
    res = cli_runner.invoke(
        app,
        [
            "leaderboard",
            "ruler-effective",
            str(sweep),
            "--threshold-from",
            str(ref),
            "-o",
            "json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)["data"]
    assert data["threshold"] == 80.0
    # 8k (82) now clears the 80 bar → effective length extends to 8k.
    (summary,) = data["models"].values()
    assert summary["effective_length_tag"] == "8k"


def test_cli_rejects_both_threshold_flags(tmp_path):
    _write_sweep(tmp_path, {"4k": 95.0})
    res = cli_runner.invoke(
        app,
        [
            "leaderboard",
            "ruler-effective",
            str(tmp_path),
            "--threshold",
            "50",
            "--threshold-from",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 1


def test_cli_no_reports_exits_nonzero(tmp_path):
    res = cli_runner.invoke(app, ["leaderboard", "ruler-effective", str(tmp_path)])
    assert res.exit_code == 1
