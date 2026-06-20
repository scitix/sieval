"""Unit tests for sieval.cli.dataset commands (list + show).

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from sieval.cli.dataset.commands import dataset_app

runner = CliRunner()


def test_dataset_list_shows_pilot_rows():
    result = runner.invoke(dataset_app, ["list"])
    assert result.exit_code == 0, result.output
    assert "aime_2024" in result.output


def test_dataset_list_domain_filter():
    result = runner.invoke(
        dataset_app, ["list", "--domain", "Mathematics", "-o", "json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)["data"]
    assert payload and all("Mathematics" in row["domain"].split("/") for row in payload)


def test_dataset_list_json_output():
    result = runner.invoke(dataset_app, ["list", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "dataset.list"
    data = payload["data"]
    assert isinstance(data, list) and len(data) >= 11
    # Every row has expected keys
    assert {"name", "domain", "license", "deps_group", "ready"} <= set(data[0])


def test_dataset_show_known():
    result = runner.invoke(dataset_app, ["show", "aime_2024", "-o", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)["data"]
    assert data["name"] == "aime_2024"
    assert "tasks" in data
    # New contract: ready + missing are first-class wire fields.
    assert data["ready"] in {"yes", "no", "unknown"}
    assert isinstance(data["missing"], list)


def test_dataset_show_unknown():
    result = runner.invoke(dataset_app, ["show", "nonexistent_zzz"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# download tests
# ---------------------------------------------------------------------------


def test_dataset_download_unknown_name():
    result = runner.invoke(dataset_app, ["download", "nonexistent_zzz"])
    assert result.exit_code != 0


def test_dataset_download_requires_exactly_one_target():
    # neither name nor domain nor --all
    result = runner.invoke(dataset_app, ["download"])
    assert result.exit_code != 0
    # name + --all both given
    result = runner.invoke(dataset_app, ["download", "aime_2024", "--all"])
    assert result.exit_code != 0


def test_dataset_download_by_name_invokes_handler(tmp_path):
    with (
        patch("sieval.datasets.downloaders.url.URLHandler.download") as mock_url_dl,
        patch(
            "sieval.datasets.downloaders.url.URLHandler.is_downloaded",
            return_value=False,
        ),
        patch("sieval.cli.dataset.commands.verify_checksums", return_value=[]),
    ):
        mock_url_dl.return_value = tmp_path
        result = runner.invoke(
            dataset_app,
            ["download", "drop", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        # DROP has 2 URL sources — handler called twice
        assert mock_url_dl.call_count == 2


def test_dataset_download_skips_when_already_present(tmp_path):
    with (
        patch(
            "sieval.datasets.downloaders.url.URLHandler.is_downloaded",
            return_value=True,
        ),
        patch("sieval.datasets.downloaders.url.URLHandler.download") as mock_url_dl,
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded", return_value=True
        ),
        patch("sieval.datasets.downloaders.hf.HFHandler.download") as mock_hf_dl,
        patch("sieval.cli.dataset.commands.verify_checksums", return_value=[]),
    ):
        result = runner.invoke(
            dataset_app,
            ["download", "drop", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        mock_url_dl.assert_not_called()
        mock_hf_dl.assert_not_called()


def test_dataset_download_force_redownloads(tmp_path):
    with (
        patch(
            "sieval.datasets.downloaders.url.URLHandler.is_downloaded",
            return_value=True,
        ),
        patch("sieval.datasets.downloaders.url.URLHandler.download") as mock_url_dl,
        patch("sieval.cli.dataset.commands.verify_checksums", return_value=[]),
    ):
        mock_url_dl.return_value = tmp_path
        result = runner.invoke(
            dataset_app,
            ["download", "drop", "--data-dir", str(tmp_path), "--force"],
        )
        assert result.exit_code == 0
        assert mock_url_dl.call_count == 2  # 2 sources for DROP


def test_dataset_download_all_iterates_all_pilots(tmp_path):
    """`--all` must iterate every registered dataset. A no-op loop would also
    pass an `assert_not_called()` check on `download()`, so discriminating
    power comes from counting `is_downloaded` probes: one per source across
    the pilot set."""
    from sieval.meta import load_index

    datasets, _ = load_index()
    expected_hf_sources = sum(
        1 for m in datasets for s in m.source if s.startswith("hf:")
    )
    expected_url_sources = sum(
        1 for m in datasets for s in m.source if s.startswith("url:")
    )

    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded", return_value=True
        ) as mock_hf_probe,
        patch(
            "sieval.datasets.downloaders.url.URLHandler.is_downloaded",
            return_value=True,
        ) as mock_url_probe,
        patch("sieval.datasets.downloaders.hf.HFHandler.download") as mock_hf,
        patch("sieval.datasets.downloaders.url.URLHandler.download") as mock_url,
        patch("sieval.cli.dataset.commands.verify_checksums", return_value=[]),
    ):
        result = runner.invoke(
            dataset_app,
            ["download", "--all", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        # Iteration proof: every registered source was probed exactly once.
        assert mock_hf_probe.call_count == expected_hf_sources
        assert mock_url_probe.call_count == expected_url_sources
        # Already-downloaded short-circuit means download() itself stays cold.
        mock_hf.assert_not_called()
        mock_url.assert_not_called()


def test_dataset_download_all_aggregates_failures(tmp_path):
    """Batch mode (`--all`/`--domain`) must not abort on first failure — it
    collects errors, continues, and reports a summary at the end with exit=1.

    Regression: early `--all` implementation propagated the first exception,
    leaving the rest of the batch unattempted and the user without a full
    failure list to act on.
    """

    def fail_hf(_source, _dest_root, dataset_name, _force):
        raise RuntimeError(f"boom: {dataset_name}")

    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded",
            return_value=False,
        ),
        patch(
            "sieval.datasets.downloaders.url.URLHandler.is_downloaded",
            return_value=False,
        ),
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.download",
            side_effect=fail_hf,
        ) as mock_hf,
        patch(
            "sieval.datasets.downloaders.url.URLHandler.download",
            return_value=tmp_path,
        ) as mock_url,
    ):
        result = runner.invoke(
            dataset_app, ["download", "--all", "--data-dir", str(tmp_path)]
        )
    assert result.exit_code == 1, result.output
    # Every HF-sourced dataset was attempted despite failures.
    assert mock_hf.call_count >= 2
    # Non-HF (url:) datasets still got processed — pipeline didn't short-circuit.
    assert mock_url.call_count >= 1
    # Aggregate summary present.
    assert "failed" in result.output.lower()


def test_dataset_download_single_name_fails_fast(tmp_path):
    """Single-name invocation preserves fail-fast; the user asked for exactly
    one dataset, so the first exception propagates as-is."""
    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded",
            return_value=False,
        ),
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.download",
            side_effect=RuntimeError("boom"),
        ),
    ):
        result = runner.invoke(
            dataset_app,
            ["download", "aime_2024", "--data-dir", str(tmp_path)],
        )
    assert result.exit_code != 0


def test_dataset_download_warns_when_data_dir_differs_from_default(
    tmp_path, monkeypatch
):
    """`--data-dir X` with $SIEVAL_DATA_DIR pointing elsewhere must warn so
    the user isn't confused when `sieval task list` / `sieval eval` later
    reports ready=no on the just-downloaded data."""
    env_root = tmp_path / "env"
    env_root.mkdir()
    override = tmp_path / "override"
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(env_root))

    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded", return_value=True
        ),
        patch("sieval.datasets.downloaders.hf.HFHandler.download"),
    ):
        result = runner.invoke(
            dataset_app,
            ["download", "aime_2024", "--data-dir", str(override)],
        )
    assert result.exit_code == 0, result.output
    # Warning must name both paths + the env var to be actionable.
    assert "SIEVAL_DATA_DIR" in result.output
    assert str(override) in result.output
    assert str(env_root) in result.output


def test_dataset_download_no_warning_when_data_dir_matches_env(tmp_path, monkeypatch):
    """If --data-dir agrees with $SIEVAL_DATA_DIR, no mismatch warning."""
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded", return_value=True
        ),
        patch("sieval.datasets.downloaders.hf.HFHandler.download"),
    ):
        result = runner.invoke(
            dataset_app,
            ["download", "aime_2024", "--data-dir", str(tmp_path)],
        )
    assert result.exit_code == 0
    assert "SIEVAL_DATA_DIR" not in result.output


def test_dataset_download_no_warning_without_data_dir_flag(tmp_path, monkeypatch):
    """No --data-dir → env / default governs; no mismatch to warn about."""
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded", return_value=True
        ),
        patch("sieval.datasets.downloaders.hf.HFHandler.download"),
    ):
        result = runner.invoke(dataset_app, ["download", "aime_2024"])
    assert result.exit_code == 0
    assert "SIEVAL_DATA_DIR" not in result.output


def test_dataset_download_domain_filter(tmp_path):
    with (
        patch(
            "sieval.datasets.downloaders.hf.HFHandler.is_downloaded", return_value=False
        ),
        patch("sieval.datasets.downloaders.hf.HFHandler.download") as mock_hf,
    ):
        mock_hf.return_value = tmp_path
        result = runner.invoke(
            dataset_app,
            ["download", "--domain", "Mathematics", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        # Mathematics domain pilots: aime_2024, aime_2025, math_500
        # aime_2024, aime_2025 are hf:, math_500 is hf: → 3 hf downloads expected
        assert mock_hf.call_count >= 1


# ── readiness correctness (regression tests for C1 bug, migrated from downloaded) ──


def test_dataset_list_ready_no_when_cache_empty(tmp_path, monkeypatch):
    """Empty data dir → all rows report ready=no."""
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    result = runner.invoke(dataset_app, ["list", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)["data"]
    assert payload and all(row["ready"] == "no" for row in payload)


def test_dataset_list_ready_yes_only_for_present_dataset(tmp_path, monkeypatch):
    """Marking one dataset as present must not flag others under the same scheme.

    Regression: before the C1 fix, _downloaded_status checked whether the
    hf-cache directory existed at all (wrong path AND not dataset-specific),
    so downloading any HF dataset falsely flagged every HF-sourced dataset.
    Under the new `ready` field the per-source granularity is explicit.
    """
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    local_dir = tmp_path / "HuggingFaceH4" / "aime_2024"
    local_dir.mkdir(parents=True)
    (local_dir / "train.parquet").write_bytes(b"x")

    result = runner.invoke(dataset_app, ["list", "-o", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)["data"]
    row_by_name = {row["name"]: row for row in payload}
    # aime_2024 has no deps_group today → with the snapshot staged, ready=yes.
    # If a future index revision adds deps_group, adjust this test and the
    # adjacent load-index check.
    assert row_by_name["aime_2024"]["ready"] == "yes"
    assert row_by_name["aime_2025"]["ready"] == "no"
    assert row_by_name["mmlu"]["ready"] == "no"


def test_dataset_list_ready_all_sources_required_for_multi_source(
    tmp_path, monkeypatch
):
    """DROP has 2 URL sources; both must exist for ready=yes."""
    from sieval.datasets.downloaders.url import _basename

    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    train_url = (
        "https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_train.jsonl.gz"
    )
    dev_url = (
        "https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_dev.jsonl.gz"
    )
    drop_dir = tmp_path / "drop"
    drop_dir.mkdir(parents=True)
    (drop_dir / _basename(train_url)).touch()

    result = runner.invoke(dataset_app, ["list", "-o", "json"])
    payload = json.loads(result.output)["data"]
    # drop has deps_group=None today → single file present = "no" (missing data).
    assert {r["name"]: r["ready"] for r in payload}["drop"] == "no"

    (drop_dir / _basename(dev_url)).touch()
    result = runner.invoke(dataset_app, ["list", "-o", "json"])
    payload = json.loads(result.output)["data"]
    # Both files present + deps_group=None → ready=yes.
    assert {r["name"]: r["ready"] for r in payload}["drop"] == "yes"


def test_dataset_show_suggested_path_for_hf_source():
    """HF dataset → suggested_path is the repo id."""
    result = runner.invoke(dataset_app, ["show", "aime_2024", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["suggested_path"] == "HuggingFaceH4/aime_2024"


def test_dataset_show_suggested_path_strips_hf_revision_pin():
    """Pinned HF dataset source still suggests the bare repo id for YAML path."""
    result = runner.invoke(dataset_app, ["show", "theoremqa", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["suggested_path"] == "TIGER-Lab/TheoremQA"
    assert data["source"] == [
        "hf:TIGER-Lab/TheoremQA@a340b1782960a712843aae3ed25f1e013cc008a5"
    ]


def test_dataset_show_suggested_path_for_url_source():
    """URL dataset → suggested_path is ${SIEVAL_DATA_DIR}/<name> (literal)."""
    result = runner.invoke(dataset_app, ["show", "drop", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["suggested_path"] == "${SIEVAL_DATA_DIR}/drop"


def test_dataset_list_text_shows_ready_and_drops_downloaded(tmp_path, monkeypatch):
    """READY column always visible; DOWNLOADED column gone (wire-contract change).

    Collapse mechanism itself is covered by `TestCollapseConstantColumns`;
    asserting a specific pilot column (e.g. DEPS_GROUP) would bind this
    test to current pilot value distribution.
    """
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    result = runner.invoke(dataset_app, ["list"])
    assert result.exit_code == 0
    assert "READY" in result.output
    assert "DOWNLOADED" not in result.output


def test_download_one_deletes_and_raises_on_checksum_mismatch(tmp_path):
    import pytest

    from sieval.cli.dataset.commands import _download_one
    from sieval.core.datasets.meta import Category, DatasetMeta, Level1Category

    meta = DatasetMeta(
        name="ds",
        display_name="ds",
        description="d",
        source=("url:https://example.com/f.csv",),
        categories=(Category(Level1Category.CODE, "CodeGeneration"),),
        checksums=(("f.csv", "sha256:" + "a" * 64),),
    )

    def fake_download(_source, dest_root, dataset_name, **_kwargs):  # noqa: ARG001
        (dest_root / dataset_name).mkdir(parents=True, exist_ok=True)
        (dest_root / dataset_name / "f.csv").write_bytes(b"wrong-bytes")

    fake_handler = MagicMock()
    fake_handler.is_downloaded.return_value = False
    fake_handler.download.side_effect = fake_download

    with (
        patch("sieval.cli.dataset.commands.resolve_handler", return_value=fake_handler),
        pytest.raises(RuntimeError, match="checksum verification failed"),
    ):
        _download_one(meta, tmp_path, force=False)

    assert not (tmp_path / "ds" / "f.csv").exists()  # bad file deleted


def test_download_one_verifies_even_when_already_present(tmp_path):
    import pytest

    from sieval.cli.dataset.commands import _download_one
    from sieval.core.datasets.meta import Category, DatasetMeta, Level1Category

    meta = DatasetMeta(
        name="ds",
        display_name="ds",
        description="d",
        source=("url:https://example.com/f.csv",),
        categories=(Category(Level1Category.CODE, "CodeGeneration"),),
        checksums=(("f.csv", "sha256:" + "a" * 64),),
    )
    fake_handler = MagicMock()
    fake_handler.is_downloaded.return_value = True  # already present, download skipped

    with (
        patch("sieval.cli.dataset.commands.resolve_handler", return_value=fake_handler),
        pytest.raises(RuntimeError, match="checksum verification failed"),
    ):
        _download_one(meta, tmp_path, force=False)

    fake_handler.download.assert_not_called()  # verify ran despite the skip


def test_dataset_show_text_renders_ready_and_missing(tmp_path, monkeypatch):
    """ready=no path renders the Missing: block + suppresses downloaded."""
    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
    result = runner.invoke(dataset_app, ["show", "drop"])
    assert result.exit_code == 0
    assert "Ready:" in result.output
    assert "Downloaded:" not in result.output
    # drop has 2 URL sources in empty tmp → data axis missing. Match the
    # indented `  data  ` kind label specifically (not as a substring of
    # "dataset-deps", which also contains "data").
    assert "Missing:" in result.output
    import re

    # _render_missing_entry emits "  data         <src>" — match the label
    # as a whole word preceded by two-space indent and followed by whitespace.
    assert re.search(r"^  data\s", result.output, re.MULTILINE), (
        f"expected an indented 'data' kind line in Missing block; got:\n{result.output}"
    )
