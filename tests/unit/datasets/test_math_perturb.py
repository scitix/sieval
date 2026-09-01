"""Unit tests for the MATH-P-Simple / MATH-P-Hard dataset loaders.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json

import pytest
from datasets import Dataset as HFDataset

from sieval.core.datasets import Dataset
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets._math_perturb import (
    MATH_PERTURB_COMMIT,
    MATH_PERTURB_ROWS,
    TVariant,
    data_file,
    data_url,
    load_math_perturb,
)
from sieval.datasets.math_perturb_hard import MATHPerturbHardDataset
from sieval.datasets.math_perturb_simple import MATHPerturbSimpleDataset

# Typed as the loader's own `TVariant`, not `str`: the literal is a real
# constraint on `data_file` / `data_url`, and a `str` here would widen it away
# at exactly the call sites meant to exercise it.
_VARIANTS: tuple[tuple[TVariant, type[Dataset]], ...] = (
    ("simple", MATHPerturbSimpleDataset),
    ("hard", MATHPerturbHardDataset),
)


def _row(problem_id: int = 1, answer: object = "42") -> dict:
    return {
        "problem_id": problem_id,
        "problem": "What is the answer?",
        "answer": answer,
        "level": "Level 5",
        "type": "Algebra",
        "original_split": "train",
    }


def _write(tmp_path, variant: TVariant, rows: list[dict]):
    directory = tmp_path / f"math_perturb_{variant}"
    directory.mkdir()
    path = directory / data_file(variant)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return directory, path


def _test_set(dataset: Dataset) -> HFDataset:
    """The ``test`` split, asserted present.

    ``Dataset.test_set`` is ``HFDataset | None``, and a ``None`` here would
    otherwise surface several lines later as an attribute error on ``None``
    rather than as "the loader published no test split".
    """
    test_set = dataset.test_set
    assert test_set is not None
    return test_set


# --- source pinning ---


@pytest.mark.parametrize(("variant", "cls"), _VARIANTS)
def test_source_is_pinned_to_the_commit(variant, cls):
    (source,) = get_dataset_meta(cls).source
    assert source == f"url:{data_url(variant)}"
    assert MATH_PERTURB_COMMIT in source
    # A branch URL would let the rows move under a checksum that no longer
    # matches — failing closed, but only after a download.
    assert "/main/" not in source


@pytest.mark.parametrize(("variant", "cls"), _VARIANTS)
def test_checksum_is_declared_for_the_staged_basename(variant, cls):
    checksums = dict(get_dataset_meta(cls).checksums)
    assert set(checksums) == {data_file(variant)}
    assert checksums[data_file(variant)].startswith("sha256:")


@pytest.mark.parametrize(("variant", "cls"), _VARIANTS)
def test_license_is_upstreams_apache(variant, cls):
    """Upstream ships a LICENSE at the pinned commit, and both files sit under it.

    Note it does NOT make the data freely usable: the README restricts it to
    academic research and asks that it never be used as training data.
    """
    meta = get_dataset_meta(cls)
    assert meta.license == "Apache-2.0"
    # One repository, so one license — asserted per variant rather than assumed.
    assert data_url(variant) in meta.source[0]


def test_the_two_datasets_do_not_share_a_staging_basename():
    """They stage under separate dataset-name directories, but not by accident."""
    assert data_file("simple") != data_file("hard")


# --- loading ---


@pytest.mark.parametrize(("variant", "cls"), _VARIANTS)
def test_load_reads_a_directory_or_the_file_itself(tmp_path, variant, cls):
    rows = [_row(i) for i in range(1, MATH_PERTURB_ROWS + 1)]
    directory, path = _write(tmp_path, variant, rows)

    from_dir = _test_set(cls(str(directory)))
    from_file = _test_set(cls(str(path)))
    assert len(from_dir) == MATH_PERTURB_ROWS
    assert from_dir[0] == from_file[0]


def test_answer_column_is_cast_to_string(tmp_path):
    """Upstream's JSON holds int, float and str in one column.

    Arrow cannot represent that at all, and the cast is upstream's own: its
    ``extract_ground_truth_answer`` ``str()``s a non-string label before wrapping
    it.
    """
    rows = [_row(1, 42), _row(2, 0.5), _row(3, "\\frac{1}{2}")]
    rows += [_row(i) for i in range(4, MATH_PERTURB_ROWS + 1)]
    directory, _ = _write(tmp_path, "simple", rows)

    test_set = _test_set(MATHPerturbSimpleDataset(str(directory)))
    assert str(test_set.features["answer"]) == "Value('string')"
    assert [test_set[i]["answer"] for i in range(3)] == ["42", "0.5", "\\frac{1}{2}"]


def test_rows_are_sorted_by_problem_id(tmp_path):
    rows = [_row(i) for i in reversed(range(1, MATH_PERTURB_ROWS + 1))]
    directory, _ = _write(tmp_path, "hard", rows)
    ids = _test_set(MATHPerturbHardDataset(str(directory)))["problem_id"]
    assert ids == sorted(ids)


def test_only_a_test_split_is_published(tmp_path):
    """Upstream asks in bold that these sets never be used as training data.

    ``original_split`` is provenance about the SEED problem, not a division of
    these rows, so there is no ``train`` for a caller to reach for by accident.
    """
    rows = [_row(i) for i in range(1, MATH_PERTURB_ROWS + 1)]
    directory, _ = _write(tmp_path, "simple", rows)
    dataset = MATHPerturbSimpleDataset(str(directory))
    assert set(dataset.dataset_dict) == {"test"}
    assert dataset.train_set is None


# --- pin integrity ---


def test_a_short_file_is_refused(tmp_path):
    directory, _ = _write(tmp_path, "simple", [_row(1), _row(2)])
    with pytest.raises(ValueError, match="expected 279"):
        MATHPerturbSimpleDataset(str(directory))


def test_an_empty_file_names_the_download_command(tmp_path):
    directory, _ = _write(tmp_path, "hard", [])
    with pytest.raises(ValueError, match="sieval dataset download math_perturb_hard"):
        MATHPerturbHardDataset(str(directory))


def test_blank_lines_are_skipped_rather_than_parsed(tmp_path):
    directory = tmp_path / "math_perturb_simple"
    directory.mkdir()
    rows = [_row(i) for i in range(1, MATH_PERTURB_ROWS + 1)]
    (directory / data_file("simple")).write_text(
        "\n\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8"
    )
    assert len(load_math_perturb(str(directory), "simple")["test"]) == MATH_PERTURB_ROWS
