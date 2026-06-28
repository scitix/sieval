"""Tests for sieval/datasets/ruler.py — RulerDataset and module-level helpers.

FWE is the only subtask that needs no external data files (source=()) so it is
used as the integration smoke test.  Integration tests require the ruler deps
group (tiktoken / wonderwords / scipy) — they are skipped when unavailable.
"""

import pytest

from sieval.datasets.ruler import thinking_prefill

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


# ---------------------------------------------------------------------------
# thinking_prefill helper
# ---------------------------------------------------------------------------

_QWEN3_TAGS = "<think>\n\n</think>\n\n"


@pytest.mark.parametrize(
    ("model_name", "enable_thinking", "expected"),
    [
        ("Qwen/Qwen3-8B", False, _QWEN3_TAGS),
        ("qwen3-8b-instruct", False, _QWEN3_TAGS),
        ("Qwen/Qwen3-8B", True, ""),
        ("meta-llama/Llama-3-8B", False, ""),
        ("gpt-4o", False, ""),
        ("cl100k_base", True, ""),
    ],
)
def test_thinking_prefill(model_name, enable_thinking, expected):
    assert thinking_prefill(model_name, enable_thinking) == expected


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
    first = list(
        RulerDataset(
            ".", subtask="fwe", max_seq_length=512, num_samples=2, random_seed=42
        ).test_set
    )
    second = list(
        RulerDataset(
            ".", subtask="fwe", max_seq_length=512, num_samples=2, random_seed=42
        ).test_set
    )
    assert first[0]["input"] == second[0]["input"]
    assert first[0]["outputs"] == second[0]["outputs"]


@_needs_ruler_deps
def test_fwe_no_token_position_answer_field():
    rows = list(
        RulerDataset(".", subtask="fwe", max_seq_length=512, num_samples=2).test_set
    )
    for r in rows:
        assert "token_position_answer" not in r


# ---------------------------------------------------------------------------
# Schema validation via TypedDict keys
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_fwe_sample_satisfies_required_schema():
    row = list(
        RulerDataset(".", subtask="fwe", max_seq_length=512, num_samples=1).test_set
    )[0]
    missing = set(RulerDatasetSample.__required_keys__) - set(row.keys())
    assert not missing, f"Missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Unknown subtask
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_unknown_subtask_raises():
    with pytest.raises(ValueError, match="Unknown subtask"):
        RulerDataset(".", subtask="nonexistent_task", max_seq_length=512, num_samples=1)
