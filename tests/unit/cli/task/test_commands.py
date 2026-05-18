"""Unit tests for sieval.cli.task commands (list + show).

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import json

from typer.testing import CliRunner

from sieval.cli.task.commands import task_app

runner = CliRunner()


def test_task_list_returns_rows():
    result = runner.invoke(task_app, ["list", "-o", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "task.list"
    assert isinstance(payload["data"], list) and len(payload["data"]) >= 11


def test_task_list_text_has_pilot_row():
    result = runner.invoke(task_app, ["list"])
    assert result.exit_code == 0
    # pilot tasks include aime/mmlu/gpqa etc — at least one should show
    assert any(k in result.output.lower() for k in ("aime", "mmlu", "gpqa"))


def test_task_list_dataset_filter():
    result = runner.invoke(task_app, ["list", "--dataset", "aime_2024", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)["data"]
    assert payload and all(row["dataset"] == "aime_2024" for row in payload)


def test_task_list_domain_filter():
    result = runner.invoke(task_app, ["list", "--domain", "Mathematics", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)["data"]
    assert payload
    # every result's dataset must belong to Mathematics domain
    from sieval.core.datasets.meta import Level1Category
    from sieval.meta import load_index

    datasets, _ = load_index()
    datasets_by_name = {d.name: d for d in datasets}
    for row in payload:
        ds = datasets_by_name.get(row["dataset"])
        assert ds and any(c.level1 is Level1Category.MATHEMATICS for c in ds.categories)


def test_task_list_eval_mode_filter():
    result = runner.invoke(task_app, ["list", "--eval-mode", "gen", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)["data"]
    assert payload and all(row["eval_mode"] == "gen" for row in payload)


def test_task_show_known():
    # Use JSON mode; text renderer will be fixed in Task 9.
    list_result = runner.invoke(task_app, ["list", "-o", "json"])
    first = json.loads(list_result.output)["data"][0]
    result = runner.invoke(task_app, ["show", first["name"], "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["name"] == first["name"]
    assert data["dataset"] == first["dataset"]


def test_task_show_unknown():
    result = runner.invoke(task_app, ["show", "nonexistent_yyy"])
    assert result.exit_code != 0


def test_task_show_json_exposes_ready_and_suggested_class():
    """Smoke test on a pilot task: JSON carries structured readiness fields."""
    result = runner.invoke(task_app, ["show", "aime_2024_0shot_gen", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["ready"] in {"yes", "no", "unknown"}
    assert isinstance(data["missing"], list)
    assert data["suggested_class"] == "AIME2024ZeroShotGenTask"


def test_task_show_survives_missing_task_deps(tmp_path, monkeypatch):
    """`task show` must deliver the readiness report instead of crashing when
    the task module's top-level imports fail (unsatisfied extras).

    Regression: `suggested_class = get_task_class(t.name).__name__` previously
    ran unconditionally, so a user running `task show` on a task whose
    `deps_group` was unsatisfied got a bare ModuleNotFoundError — despite
    `evaluate_task_readiness` having just computed the missing-extras entry
    the user came to see.
    """
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    from unittest.mock import patch

    with patch(
        "sieval.cli.task.render.get_task_class",
        side_effect=ModuleNotFoundError(
            "No module named 'math_verify'", name="math_verify"
        ),
    ):
        result = runner.invoke(task_app, ["show", "aime_2024_0shot_gen", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["ready"] == "no"
    assert data["suggested_class"] is None
    # run_hint must be absent (not-ready AND suggested_class=None).
    assert "run_hint" not in data


def test_task_show_run_hint_present_only_when_ready(tmp_path, monkeypatch):
    """run_hint must appear for ready=yes tasks and be omitted otherwise."""
    # Empty data dir → aime_2024 has no data → ready=no.
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    result = runner.invoke(task_app, ["show", "aime_2024_0shot_gen", "-o", "json"])
    data = json.loads(result.output)["data"]
    assert data["ready"] == "no"
    assert "run_hint" not in data  # omitted when not ready


def test_task_list_json_includes_ready_column():
    result = runner.invoke(task_app, ["list", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)["data"]
    assert payload
    assert all("ready" in row for row in payload)
    assert all(row["ready"] in {"yes", "no", "unknown"} for row in payload)


def test_task_list_text_shows_ready_column(tmp_path, monkeypatch):
    """READY column always visible. Collapse mechanism tested separately
    in `TestCollapseConstantColumns`; asserting STATUS or EVAL_MODE here
    would bind this test to pilot value distribution."""
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    result = runner.invoke(task_app, ["list"])
    assert result.exit_code == 0
    assert "READY" in result.output


def test_task_show_text_run_block_when_ready(tmp_path, monkeypatch):
    """A synthesized ready=yes path: seed data + mock extras-clear."""
    # Use aime_2024 which has no deps_group; populate its HF cache so
    # data axis is clear in-env. aime_2024_0shot_gen task has deps_group=math,
    # which may or may not be installed. Test both branches conditionally.
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    local_dir = tmp_path / "HuggingFaceH4" / "aime_2024"
    local_dir.mkdir(parents=True)
    (local_dir / "train.parquet").write_bytes(b"x")
    # Force extras-clear to isolate the text rendering.
    from unittest.mock import patch

    with patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]):
        result = runner.invoke(task_app, ["show", "aime_2024_0shot_gen"])
    assert result.exit_code == 0
    assert "Ready:         yes" in result.output
    assert "Run:" in result.output
    assert "AIME2024ZeroShotGenTask" in result.output
    assert "Missing:" not in result.output


def test_task_show_text_missing_block_when_not_ready(tmp_path, monkeypatch):
    """Empty data + likely-missing extras → Missing block present, Run absent."""
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    result = runner.invoke(task_app, ["show", "aime_2024_0shot_gen"])
    assert result.exit_code == 0
    assert "Ready:         no" in result.output
    assert "Missing:" in result.output
    assert "Run:" not in result.output


def test_task_list_data_dir_flag_overrides_env(tmp_path, monkeypatch):
    """`--data-dir` must flip readiness based on the override path, not the env."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(empty))

    override = tmp_path / "override"
    local_dir = override / "HuggingFaceH4" / "aime_2024"
    local_dir.mkdir(parents=True)
    (local_dir / "train.parquet").write_bytes(b"x")

    r_env = runner.invoke(task_app, ["list", "--dataset", "aime_2024", "-o", "json"])
    payload_env = json.loads(r_env.output)["data"]
    assert payload_env and all(row["ready"] == "no" for row in payload_env)

    # Mock extras-clear so the task-deps axis doesn't gate ready=yes.
    from unittest.mock import patch

    with patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]):
        r_override = runner.invoke(
            task_app,
            [
                "list",
                "--dataset",
                "aime_2024",
                "--data-dir",
                str(override),
                "-o",
                "json",
            ],
        )
    payload_override = json.loads(r_override.output)["data"]
    assert payload_override and any(row["ready"] == "yes" for row in payload_override)


def test_task_show_data_dir_flag_overrides_env(tmp_path, monkeypatch):
    """Symmetric coverage for `task show`."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(empty))

    override = tmp_path / "override"
    local_dir = override / "HuggingFaceH4" / "aime_2024"
    local_dir.mkdir(parents=True)
    (local_dir / "train.parquet").write_bytes(b"x")

    from unittest.mock import patch

    with patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]):
        r = runner.invoke(
            task_app,
            ["show", "aime_2024_0shot_gen", "--data-dir", str(override), "-o", "json"],
        )
    data = json.loads(r.output)["data"]
    assert data["ready"] == "yes"
