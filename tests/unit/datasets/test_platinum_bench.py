"""
Unit tests for the PlatinumBench dataset loader.

``load_dataset`` is monkeypatched throughout: the assertions are about the
loader's own three jobs — validate the subset, drop rejected rows, and reduce 15
heterogeneous configs to the one shared schema — none of which need the real
1042-row download.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.platinum_bench as platinum_mod
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.platinum_bench import (
    PLATINUM_BENCH_REVISION,
    PLATINUM_SUBSETS,
    PlatinumBenchDataset,
    PlatinumBenchDatasetSample,
)


def _row(
    *,
    status: str = "consensus",
    target: str = "42",
    # An "original" column that only gsm8k has — the loader must drop it, which
    # is what lets one TypedDict back all the subsets.
    question: str = "What is 40 + 2?",
    answer: str = "Reasoning.\n#### 42",
) -> dict:
    return {
        "cleaning_status": status,
        "original_target": [target],
        "platinum_parsing_strategy": "math",
        "platinum_prompt": f"{question}\nThen, provide your answer as 'Answer: <n>'.",
        "platinum_prompt_no_cot": f"{question}\nAnswer with 'Answer: <n>'.",
        "platinum_target": [target],
        "question": question,
        "answer": answer,
    }


def _fake_load_dataset(rows: list[dict], calls: list[tuple] | None = None):
    def fake(name_or_path, config=None, **kwargs):
        if calls is not None:
            calls.append((name_or_path, config, kwargs))
        return HFDatasetDict({"test": HFDataset.from_list(rows)})

    return fake


def _load(monkeypatch, rows: list[dict], *, subset: str = "gsm8k", calls=None):
    monkeypatch.setattr(
        platinum_mod, "load_dataset", _fake_load_dataset(rows, calls=calls)
    )
    return PlatinumBenchDataset("/staged/platinum-bench", subset=subset)


# ---------------------------------------------------------------------------
# Subset validation — before any download
# ---------------------------------------------------------------------------


def test_unknown_subset_raises_with_candidate_list():
    with pytest.raises(ValueError, match="Unknown PlatinumBench subset"):
        PlatinumBenchDataset("/staged/platinum-bench", subset="gsm8k_typo")


def test_vqa_subset_is_rejected_with_its_reason():
    # vqa is the 15th config and is excluded, not merely untested: its strategy
    # column is misspelled upstream so it does not share the selected schema.
    with pytest.raises(ValueError, match="platinum_parsing_stratagy"):
        PlatinumBenchDataset("/staged/platinum-bench", subset="vqa")


def test_subset_validation_precedes_loading(monkeypatch):
    # A typo must not cost a download attempt.
    def exploding_load_dataset(*_args, **_kwargs):
        raise AssertionError("load_dataset must not be reached for a bad subset")

    monkeypatch.setattr(platinum_mod, "load_dataset", exploding_load_dataset)
    with pytest.raises(ValueError, match="Unknown PlatinumBench subset"):
        PlatinumBenchDataset("/staged/platinum-bench", subset="nope")


def test_subset_set_covers_14_of_15_configs():
    assert len(PLATINUM_SUBSETS) == 14
    assert "vqa" not in PLATINUM_SUBSETS
    # The five math subsets this PR ships tasks for.
    assert {"gsm8k", "svamp", "multiarith", "singleop", "singleq"} <= PLATINUM_SUBSETS


# ---------------------------------------------------------------------------
# load() — subset routed to the HF config slot, rejected rows dropped
# ---------------------------------------------------------------------------


def test_subset_is_passed_as_the_hf_config(monkeypatch):
    calls: list[tuple] = []
    _load(monkeypatch, [_row()], subset="svamp", calls=calls)
    assert len(calls) == 1
    name_or_path, config, _kwargs = calls[0]
    assert name_or_path == "/staged/platinum-bench"
    assert config == "svamp"


def test_rejected_rows_are_dropped(monkeypatch):
    rows = [
        _row(status="consensus", target="1"),
        _row(status="rejected", target="2"),
        _row(status="verified", target="3"),
        _row(status="revised", target="4"),
        _row(status="rejected", target="5"),
    ]
    ds = _load(monkeypatch, rows)
    assert ds.test_set is not None
    kept = list(ds.test_set)
    assert len(kept) == 3
    assert [r["platinum_target"] for r in kept] == [["1"], ["3"], ["4"]]
    assert all(r["cleaning_status"] != "rejected" for r in kept)


def test_all_rejected_raises_rather_than_yielding_an_empty_split(monkeypatch):
    # An empty result means the pinned revision's cleaning_status vocabulary
    # changed; failing here beats reporting 0/0 as a score.
    with pytest.raises(ValueError, match="is empty after dropping"):
        _load(monkeypatch, [_row(status="rejected"), _row(status="rejected")])


# ---------------------------------------------------------------------------
# Schema reduction — the six shared columns plus a stamped subset
# ---------------------------------------------------------------------------


def test_original_columns_are_dropped_and_subset_is_stamped(monkeypatch):
    ds = _load(monkeypatch, [_row()], subset="multiarith")
    assert ds.test_set is not None
    row = list(ds.test_set)[0]
    assert set(row) == set(PlatinumBenchDatasetSample.__required_keys__)
    # gsm8k's own columns are gone — this is what makes the configs uniform.
    assert "question" not in row
    assert "answer" not in row
    assert row["subset"] == "multiarith"


def test_sample_satisfies_the_declared_schema(monkeypatch):
    ds = _load(monkeypatch, [_row()])
    assert ds.test_set is not None
    row = list(ds.test_set)[0]
    missing = set(PlatinumBenchDatasetSample.__required_keys__) - set(row)
    assert not missing, f"Missing required fields: {missing}"
    assert row["platinum_prompt"]
    assert row["platinum_prompt_no_cot"]
    assert row["platinum_target"] == ["42"]
    assert row["platinum_parsing_strategy"] == "math"


def test_subset_property_exposes_the_loaded_config(monkeypatch):
    # The task's setup() reads this to catch a task wired to a sibling subset's
    # dataset instance.
    ds = _load(monkeypatch, [_row()], subset="singleq")
    assert ds.subset == "singleq"


# ---------------------------------------------------------------------------
# Registered metadata
# ---------------------------------------------------------------------------


def test_dataset_meta_pins_the_revision_and_license():
    meta = get_dataset_meta(PlatinumBenchDataset)
    assert meta.name == "platinum_bench"
    # `source` is normalized to a tuple by @sieval_dataset (multi-origin support).
    assert meta.source == (f"hf:madrylab/platinum-bench@{PLATINUM_BENCH_REVISION}",)
    # Data license (CC-BY-SA-4.0) — distinct from the harness code's CC-BY-4.0.
    assert meta.license == "cc-by-sa-4.0"
