"""
Unit tests for the PlatinumBench dataset loader.

``load_dataset`` is monkeypatched throughout: the assertions are about the
loader's own three jobs — merge the 14 configs, drop rejected rows, and reduce
heterogeneous configs to the one shared schema — none of which need the real
3062-row download.

The load-order test is the load-bearing one. ``PLATINUM_SUBSETS`` is a frozenset,
so if ``load`` ever stops sorting it the merged row order follows the string hash
seed, sample ids shift between processes, and every stored run silently stops
lining up with a fresh one.

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

MATH_SUBSETS = frozenset({"gsm8k", "svamp", "multiarith", "singleop", "singleq"})


def _row(
    *,
    status: str = "consensus",
    target: str = "42",
    # An "original" column that only gsm8k has — the loader must drop it, which
    # is what lets one TypedDict back all the subsets and what lets the configs
    # concatenate at all.
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


def _fake_load_dataset(rows, calls: list[tuple] | None = None):
    """Serve *rows* for every config, or per-config rows when given a dict."""

    def fake(name_or_path, config=None, **kwargs):
        if calls is not None:
            calls.append((name_or_path, config, kwargs))
        per_config = rows.get(config, [_row()]) if isinstance(rows, dict) else rows
        return HFDatasetDict({"test": HFDataset.from_list(per_config)})

    return fake


def _load(monkeypatch, rows, *, calls=None, **kwargs):
    monkeypatch.setattr(
        platinum_mod, "load_dataset", _fake_load_dataset(rows, calls=calls)
    )
    return PlatinumBenchDataset("/staged/platinum-bench", **kwargs)


# ---------------------------------------------------------------------------
# Which configs get merged
# ---------------------------------------------------------------------------


def test_subset_set_covers_14_of_15_configs():
    assert len(PLATINUM_SUBSETS) == 14
    assert "vqa" not in PLATINUM_SUBSETS
    # The five math subsets this PR ships tasks for.
    assert MATH_SUBSETS <= PLATINUM_SUBSETS


def test_every_subset_is_loaded_as_its_own_hf_config(monkeypatch):
    calls: list[tuple] = []
    _load(monkeypatch, [_row()], calls=calls)
    assert len(calls) == 14
    requested = [config for _path, config, _kwargs in calls]
    assert requested == sorted(PLATINUM_SUBSETS)
    # vqa is never fetched: its misspelled strategy column would fail
    # select_columns and take the whole merge down with it.
    assert "vqa" not in requested
    assert {path for path, _c, _k in calls} == {"/staged/platinum-bench"}


def test_configs_are_merged_in_sorted_order_not_hash_order(monkeypatch):
    # Two rows per config makes a within-config ordering bug visible too.
    ds = _load(monkeypatch, [_row(target="1"), _row(target="2")])
    assert ds.test_set is not None
    expected = [s for s in sorted(PLATINUM_SUBSETS) for _ in range(2)]
    assert ds.test_set["subset"] == expected


def test_merged_split_holds_every_subset(monkeypatch):
    ds = _load(monkeypatch, [_row()])
    assert ds.test_set is not None
    assert len(ds.test_set) == 14
    assert sorted(ds.test_set.unique("subset")) == sorted(PLATINUM_SUBSETS)


# ---------------------------------------------------------------------------
# Narrowing back down — the property the reproduction depends on
# ---------------------------------------------------------------------------


def test_filter_recovers_exactly_one_subset_in_order(monkeypatch):
    # The merged split must be reducible to precisely what loading that config
    # alone would have produced, in the same order — that identity is what keeps
    # sample ids, and therefore scores, stable across this loader's shape.
    per_config = {
        "gsm8k": [_row(target=str(i)) for i in range(4)],
        "svamp": [_row(target="99")],
    }
    ds = _load(monkeypatch, per_config)
    gsm8k = ds.filter("subset", "gsm8k")
    assert gsm8k.test_set is not None
    assert [r["platinum_target"] for r in gsm8k.test_set] == [
        ["0"],
        ["1"],
        ["2"],
        ["3"],
    ]
    assert gsm8k.test_set.unique("subset") == ["gsm8k"]
    # The clone is a real narrowing, and the original is untouched.
    assert len(ds.test_set) == 4 + 1 + 12


def test_filter_accepts_several_subsets(monkeypatch):
    ds = _load(monkeypatch, [_row()]).filter("subset", ["gsm8k", "svamp"])
    assert ds.test_set is not None
    assert sorted(ds.test_set.unique("subset")) == ["gsm8k", "svamp"]


def test_filter_on_a_misspelled_subset_raises(monkeypatch):
    # Would otherwise be an empty split and a run that scores zero samples.
    ds = _load(monkeypatch, [_row()])
    with pytest.raises(ValueError, match="no row of split 'test' has subset="):
        ds.filter("subset", "gsm8K")


# ---------------------------------------------------------------------------
# The retired constructor argument
# ---------------------------------------------------------------------------


def test_stale_subset_argument_names_its_replacement(monkeypatch):
    # A config written against the old required kwarg must get the migration,
    # not load_dataset's opaque "unexpected keyword argument".
    with pytest.raises(ValueError, match="no longer takes a 'subset' argument"):
        _load(monkeypatch, [_row()], subset="gsm8k")


def test_stale_subset_argument_is_caught_before_any_download(monkeypatch):
    def exploding_load_dataset(*_args, **_kwargs):
        raise AssertionError("load_dataset must not be reached")

    monkeypatch.setattr(platinum_mod, "load_dataset", exploding_load_dataset)
    with pytest.raises(ValueError, match="filter: {by: subset, value: svamp}"):
        PlatinumBenchDataset("/staged/platinum-bench", subset="svamp")


# ---------------------------------------------------------------------------
# Rejected rows
# ---------------------------------------------------------------------------


def test_rejected_rows_are_dropped(monkeypatch):
    rows = [
        _row(status="consensus", target="1"),
        _row(status="rejected", target="2"),
        _row(status="verified", target="3"),
        _row(status="revised", target="4"),
        _row(status="rejected", target="5"),
    ]
    ds = _load(monkeypatch, {"gsm8k": rows})
    kept = list(ds.filter("subset", "gsm8k").test_set or [])
    assert len(kept) == 3
    assert [r["platinum_target"] for r in kept] == [["1"], ["3"], ["4"]]
    assert all(r["cleaning_status"] != "rejected" for r in kept)


def test_a_fully_rejected_subset_raises_and_names_itself(monkeypatch):
    # An empty result means the pinned revision's cleaning_status vocabulary
    # changed; failing here beats reporting 0/0 as a score. Checked per config so
    # the message names the offender rather than only the merged split.
    rejected = [_row(status="rejected"), _row(status="rejected")]
    with pytest.raises(ValueError, match="subset 'singleq'.*is empty after dropping"):
        _load(monkeypatch, {"singleq": rejected})


# ---------------------------------------------------------------------------
# Schema reduction — the six shared columns plus a stamped subset
# ---------------------------------------------------------------------------


def test_original_columns_are_dropped_and_subset_is_stamped(monkeypatch):
    ds = _load(monkeypatch, [_row()]).filter("subset", "multiarith")
    assert ds.test_set is not None
    row = list(ds.test_set)[0]
    assert set(row) == set(PlatinumBenchDatasetSample.__required_keys__)
    # gsm8k's own columns are gone — this is what makes the configs uniform, and
    # without it concatenate_datasets could not merge them at all.
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
