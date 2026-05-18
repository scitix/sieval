"""Unit tests for sieval.cli._readiness.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from sieval.cli._readiness import (
    MISSING_KINDS,
    MissingEntry,
    MissingKind,
    ReadinessReport,
    _probe_dataset,
    evaluate_dataset_readiness,
    evaluate_task_readiness,
)

# ── MissingEntry / ReadinessReport shape ──────────────────────────────


def test_missing_kinds_matches_missing_kind_literal():
    """The exported tuple must agree with the Literal annotation, otherwise the
    output-layer width calculation diverges from the dataclass's accepted
    values. Guarded here so a drift on either side fails fast.
    """
    from typing import get_args

    assert set(MISSING_KINDS) == set(get_args(MissingKind))


def test_missing_entry_is_frozen():
    """Value object — freeze prevents accidental mutation in render layer."""
    e = MissingEntry(kind="data", sources=("hf:x/y",))
    with pytest.raises(AttributeError):  # FrozenInstanceError subclasses AttributeError
        e.kind = "dataset-deps"  # type: ignore[misc]


def test_readiness_report_default_missing_is_empty_tuple():
    r = ReadinessReport(ready="yes", missing=())
    assert r.missing == ()


def test_missing_entry_to_dict_data_kind():
    from sieval.cli._readiness import missing_entry_to_dict

    e = MissingEntry(kind="data", sources=("hf:a/b", "url:https://x/y.csv"))
    assert missing_entry_to_dict(e) == {
        "kind": "data",
        "sources": ["hf:a/b", "url:https://x/y.csv"],
    }


def test_missing_entry_to_dict_deps_kind():
    from sieval.cli._readiness import missing_entry_to_dict

    e = MissingEntry(kind="task-deps", group="math", unmet=("math-verify>=0.8.0",))
    assert missing_entry_to_dict(e) == {
        "kind": "task-deps",
        "group": "math",
        "unmet": ["math-verify>=0.8.0"],
    }


def test_readiness_to_wire_round_trip():
    from sieval.cli._readiness import readiness_to_wire

    r = ReadinessReport(
        ready="no",
        missing=(MissingEntry(kind="data", sources=("hf:a/b",)),),
    )
    ready, missing = readiness_to_wire(r)
    assert ready == "no"
    assert missing == [{"kind": "data", "sources": ["hf:a/b"]}]


# ── _probe_dataset ────────────────────────────────────────────────────


def _make_dataset_meta(name="ds", sources=("hf:org/name",), deps_group=None):
    """Helper to build a DatasetMeta in tests without registry side effects."""
    from sieval.core.datasets.meta import (
        Category,
        DatasetMeta,
        Level1Category,
    )

    return DatasetMeta(
        name=name,
        display_name=name.upper(),
        description=f"test dataset {name}",
        source=sources,
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        tags=(),
        deps_group=deps_group,
        license=None,
    )


def test_probe_dataset_all_clear():
    ds = _make_dataset_meta()
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.return_value.is_downloaded.return_value = True
        entries, had_unknown = _probe_dataset(ds, Path("/tmp/data"))
    assert entries == []
    assert had_unknown is False


def test_probe_dataset_data_missing():
    ds = _make_dataset_meta(sources=("hf:a/b", "url:https://x/y.csv"))
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.return_value.is_downloaded.return_value = False
        entries, had_unknown = _probe_dataset(ds, Path("/tmp/data"))
    assert had_unknown is False
    assert len(entries) == 1
    assert entries[0].kind == "data"
    assert entries[0].sources == ("hf:a/b", "url:https://x/y.csv")


def test_probe_dataset_unknown_scheme_does_not_suppress_deps_probe():
    """Non-short-circuit: unknown scheme in data axis must not stop the
    dataset-deps axis from reporting its findings."""
    ds = _make_dataset_meta(sources=("local:/foo",), deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch(
            "sieval.cli._readiness.extras_unsatisfied",
            return_value=["math-verify>=0.8.0"],
        ),
    ):
        mock_resolve.side_effect = NotImplementedError("unknown scheme: local")
        entries, had_unknown = _probe_dataset(ds, Path("/tmp/data"))
    assert had_unknown is True
    # data entry absent (unknown), but dataset-deps entry present
    kinds = [e.kind for e in entries]
    assert "data" not in kinds
    assert "dataset-deps" in kinds


def test_probe_dataset_extras_probe_runs_when_deps_group_none():
    """deps_group=None → extras probe is skipped entirely (not a bug)."""
    ds = _make_dataset_meta(deps_group=None)
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied") as mock_extras,
    ):
        mock_resolve.return_value.is_downloaded.return_value = True
        _probe_dataset(ds, Path("/tmp/data"))
    mock_extras.assert_not_called()


# ── evaluate_dataset_readiness ────────────────────────────────────────


def test_dataset_readiness_yes_when_all_clear():
    ds = _make_dataset_meta(deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.return_value.is_downloaded.return_value = True
        r = evaluate_dataset_readiness(ds, Path("/tmp/data"))
    assert r.ready == "yes"
    assert r.missing == ()


def test_dataset_readiness_no_on_data_missing():
    ds = _make_dataset_meta()
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.return_value.is_downloaded.return_value = False
        r = evaluate_dataset_readiness(ds, Path("/tmp/data"))
    assert r.ready == "no"
    assert [e.kind for e in r.missing] == ["data"]


def test_dataset_readiness_no_on_deps_missing():
    ds = _make_dataset_meta(deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch(
            "sieval.cli._readiness.extras_unsatisfied",
            return_value=["math-verify>=0.8.0"],
        ),
    ):
        mock_resolve.return_value.is_downloaded.return_value = True
        r = evaluate_dataset_readiness(ds, Path("/tmp/data"))
    assert r.ready == "no"
    assert [e.kind for e in r.missing] == ["dataset-deps"]
    assert r.missing[0].group == "math"
    assert r.missing[0].unmet == ("math-verify>=0.8.0",)


def test_dataset_readiness_no_on_both_axes_missing_preserves_order():
    """Contract: data entry precedes dataset-deps entry."""
    ds = _make_dataset_meta(deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch(
            "sieval.cli._readiness.extras_unsatisfied",
            return_value=["math-verify>=0.8.0"],
        ),
    ):
        mock_resolve.return_value.is_downloaded.return_value = False
        r = evaluate_dataset_readiness(ds, Path("/tmp/data"))
    assert r.ready == "no"
    assert [e.kind for e in r.missing] == ["data", "dataset-deps"]


def test_dataset_readiness_unknown_scheme_with_no_other_missing():
    ds = _make_dataset_meta(sources=("local:/foo",))
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.side_effect = NotImplementedError("local scheme")
        r = evaluate_dataset_readiness(ds, Path("/tmp/data"))
    assert r.ready == "unknown"
    assert r.missing == ()


def test_dataset_readiness_unknown_with_deps_missing_keeps_deps_entry():
    """Non-short-circuit regression: unknown on data + missing deps keeps
    the deps entry so the user sees the actionable fix."""
    ds = _make_dataset_meta(sources=("local:/foo",), deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch(
            "sieval.cli._readiness.extras_unsatisfied",
            return_value=["math-verify>=0.8.0"],
        ),
    ):
        mock_resolve.side_effect = NotImplementedError("local scheme")
        r = evaluate_dataset_readiness(ds, Path("/tmp/data"))
    assert r.ready == "unknown"
    assert [e.kind for e in r.missing] == ["dataset-deps"]


# ── evaluate_task_readiness ───────────────────────────────────────────


def _make_task_meta(name="t", dataset="ds", deps_group=None):
    from sieval.core.tasks.meta import EvalMode, TaskMeta

    return TaskMeta(
        name=name,
        display_name=name.upper(),
        description=f"test task {name}",
        dataset=dataset,
        eval_mode=EvalMode.GEN,
        n_shot=0,
        deps_group=deps_group,
    )


def test_task_readiness_yes_when_all_axes_clear():
    ds = _make_dataset_meta(name="ds1")
    t = _make_task_meta(name="t1", dataset="ds1")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.return_value.is_downloaded.return_value = True
        r = evaluate_task_readiness(t, Path("/tmp/data"), ds)
    assert r.ready == "yes"
    assert r.missing == ()


def test_task_readiness_concatenates_in_contract_order():
    """Order must be: data → dataset-deps → task-deps."""
    ds = _make_dataset_meta(name="ds2", deps_group="drop")
    t = _make_task_meta(name="t2", dataset="ds2", deps_group="math")

    def fake_extras(group):
        # Different unmet per group so we can tell them apart.
        return {"drop": ["scipy>=1.16.3"], "math": ["math-verify>=0.8.0"]}[group]

    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", side_effect=fake_extras),
    ):
        mock_resolve.return_value.is_downloaded.return_value = False
        r = evaluate_task_readiness(t, Path("/tmp/data"), ds)
    assert r.ready == "no"
    assert [e.kind for e in r.missing] == ["data", "dataset-deps", "task-deps"]
    # task-deps entry carries its own group, not the dataset's
    assert r.missing[2].group == "math"
    assert r.missing[2].unmet == ("math-verify>=0.8.0",)


def test_task_readiness_unknown_propagates_from_dataset():
    ds = _make_dataset_meta(name="ds3", sources=("local:/x",), deps_group=None)
    t = _make_task_meta(name="t3", dataset="ds3", deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch("sieval.cli._readiness.extras_unsatisfied", return_value=[]),
    ):
        mock_resolve.side_effect = NotImplementedError("local scheme")
        r = evaluate_task_readiness(t, Path("/tmp/data"), ds)
    assert r.ready == "unknown"


def test_task_readiness_unknown_with_task_deps_missing_still_reports_deps():
    """Non-short-circuit: dataset unknown + task extras missing → ready is
    unknown, but the task-deps entry is preserved (actionable fix)."""
    ds = _make_dataset_meta(name="ds4", sources=("local:/x",))
    t = _make_task_meta(name="t4", dataset="ds4", deps_group="math")
    with (
        patch("sieval.cli._readiness.resolve_handler") as mock_resolve,
        patch(
            "sieval.cli._readiness.extras_unsatisfied",
            return_value=["math-verify>=0.8.0"],
        ),
    ):
        mock_resolve.side_effect = NotImplementedError("local scheme")
        r = evaluate_task_readiness(t, Path("/tmp/data"), ds)
    assert r.ready == "unknown"
    assert [e.kind for e in r.missing] == ["task-deps"]
