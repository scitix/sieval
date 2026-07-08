"""Tests for sieval/datasets/ruler.py — RulerDataset and module-level helpers.

FWE is the only subtask that needs no external data files (source=()) so it is
used as the integration smoke test.  Integration tests require the ruler deps
group (tiktoken / wonderwords / scipy) — they are skipped when unavailable.
"""

import pytest

from sieval.datasets.ruler import tokens_to_generate

try:
    import tiktoken as _tiktoken  # noqa: F401

    _ruler_deps = True
except ImportError:
    _ruler_deps = False

_needs_ruler_deps = pytest.mark.skipif(
    not _ruler_deps, reason="ruler deps group not installed"
)

if _ruler_deps:
    from sieval.datasets.ruler import RulerDataset, RulerDatasetSample, _stamp
    from sieval.datasets.ruler.ruler import _subtask_data_path


# ---------------------------------------------------------------------------
# _subtask_data_path — single-subtask staging path (regression)
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_subtask_data_path_appends_ruler_subdir_for_staged_subtasks():
    # Regression: a direct single-subtask load must resolve the same ``/ruler``
    # staging path the "all"/list branches use. Previously only the aggregate
    # branches appended it, so a direct essay/squad subtask hit FileNotFoundError.
    for st in ("niah_single_2", "niah_multikey_1", "vt", "cwe", "qa_squad"):
        assert _subtask_data_path("/data", st) == "/data/ruler"


@_needs_ruler_deps
def test_subtask_data_path_uses_base_dir_for_synthetic_and_hf():
    # FWE is fully synthetic and qa_hotpotqa is fetched from HF → base data dir.
    assert _subtask_data_path("/data", "fwe") == "/data"
    assert _subtask_data_path("/data", "qa_hotpotqa") == "/data"


@_needs_ruler_deps
def test_single_subtask_load_applies_ruler_subdir_to_loader(monkeypatch):
    # Regression: a direct single-subtask load must pass the /ruler-suffixed path
    # to the loader. Previously only the all/list branches appended it, so a
    # direct essay/squad subtask read the base dir → FileNotFoundError. Mock the
    # loader to capture the path without needing the staged corpus.
    import sieval.datasets.ruler.ruler as ruler_mod

    captured: dict[str, str] = {}

    def fake_load_niah(path, **_kwargs):
        captured["path"] = path
        return []

    monkeypatch.setattr(ruler_mod, "load_niah", fake_load_niah)
    ds = RulerDataset(".", subtask="niah_single_2", max_seq_length=512, num_samples=1)
    assert ds.test_set is not None  # force the load
    assert captured["path"] == "./ruler"


@_needs_ruler_deps
def test_single_subtask_load_uses_base_dir_for_synthetic(monkeypatch):
    # FWE reads the base dir (no /ruler subdir) — the mirror of the case above.
    import sieval.datasets.ruler.ruler as ruler_mod

    captured: dict[str, str] = {}

    def fake_load_fwe(path, **_kwargs):
        captured["path"] = path
        return []

    monkeypatch.setattr(ruler_mod, "load_fwe", fake_load_fwe)
    ds = RulerDataset(".", subtask="fwe", max_seq_length=512, num_samples=1)
    assert ds.test_set is not None
    assert captured["path"] == "."


# ---------------------------------------------------------------------------
# tokens_to_generate helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task_name", "enable_thinking", "think_budget", "expected"),
    [
        # No thinking: base tokens only
        ("niah", False, 0, 128),
        ("qa", False, 0, 32),
        ("variable_tracking", False, 0, 30),
        ("common_words_extraction", False, 0, 120),
        ("freq_words_extraction", False, 0, 50),
        # Thinking enabled: base + think_budget
        ("niah", True, 1024, 1024 + 128),
        ("qa", True, 512, 512 + 32),
        ("variable_tracking", True, 2048, 2048 + 30),
        # think_budget=0 with enable_thinking=True still adds 0
        ("niah", True, 0, 128),
    ],
)
def test_tokens_to_generate(task_name, enable_thinking, think_budget, expected):
    assert (
        tokens_to_generate(
            task_name, enable_thinking=enable_thinking, think_budget=think_budget
        )
        == expected
    )


# ---------------------------------------------------------------------------
# _stamp helper
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_stamp_adds_subtask_and_context_length():
    rows = [
        {
            "index": 0,
            "input": "x",
            "outputs": ["y"],
            "length": 10,
            "answer_prefix": "A:",
        }
    ]
    stamped = _stamp(rows, subtask="vt", context_length=8192)
    assert stamped[0]["subtask"] == "vt"
    assert stamped[0]["context_length"] == 8192


@_needs_ruler_deps
def test_stamp_preserves_existing_fields():
    rows = [
        {
            "index": 7,
            "input": "q",
            "outputs": ["a"],
            "length": 5,
            "answer_prefix": "Answer:",
        }
    ]
    result = _stamp(rows, subtask="cwe", context_length=4096)
    assert result[0]["index"] == 7
    assert result[0]["outputs"] == ["a"]


# ---------------------------------------------------------------------------
# RulerDataset.load() — FWE subtask (no external data needed)
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_fwe_load_emits_required_fields():
    ds = RulerDataset(".", subtask="fwe", max_seq_length=512, num_samples=3)
    assert ds.test_set is not None
    rows = list(ds.test_set)
    assert len(rows) == 3
    for r in rows:
        assert "subtask" in r and r["subtask"] == "fwe"
        assert "context_length" in r and r["context_length"] == 512
        assert "input" in r and r["input"]
        assert "outputs" in r and len(r["outputs"]) == 3
        assert "answer_prefix" in r
        assert "length" in r and r["length"] > 0


@_needs_ruler_deps
def test_fwe_load_is_deterministic():
    ds1 = RulerDataset(
        ".", subtask="fwe", max_seq_length=512, num_samples=2, random_seed=42
    )
    ds2 = RulerDataset(
        ".", subtask="fwe", max_seq_length=512, num_samples=2, random_seed=42
    )
    assert ds1.test_set is not None and ds2.test_set is not None
    first = list(ds1.test_set)
    second = list(ds2.test_set)
    assert first[0]["input"] == second[0]["input"]
    assert first[0]["outputs"] == second[0]["outputs"]


@_needs_ruler_deps
def test_fwe_no_token_position_answer_field():
    ds = RulerDataset(".", subtask="fwe", max_seq_length=512, num_samples=2)
    assert ds.test_set is not None
    rows = list(ds.test_set)
    for r in rows:
        assert "token_position_answer" not in r


# ---------------------------------------------------------------------------
# Schema validation via TypedDict keys
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_fwe_sample_satisfies_required_schema():
    ds = RulerDataset(".", subtask="fwe", max_seq_length=512, num_samples=1)
    assert ds.test_set is not None
    row = list(ds.test_set)[0]
    missing = set(RulerDatasetSample.__required_keys__) - set(row.keys())
    assert not missing, f"Missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Unknown subtask
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_unknown_subtask_raises():
    with pytest.raises(ValueError, match="Unknown subtask"):
        RulerDataset(".", subtask="nonexistent_task", max_seq_length=512, num_samples=1)
