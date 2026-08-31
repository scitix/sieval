"""
Shared loader for the MATH-Perturb family (MATH-P-Simple / MATH-P-Hard).

Upstream ships the two testsets as two JSONL files in one repository, with an
identical schema and an identical row count (279), differing only in how the
seed problem was perturbed. Everything except that filename lives here; the two
dataset modules differ in metadata and the variant they pass.

The two files are **row-aligned by ``problem_id``**, not by position: the same
id in each names the same seed MATH problem, which is what makes the paired
Simple-vs-Hard reading the paper leads with possible at all. Nothing in this
loader depends on that, but it is the reason the id is carried rather than
dropped as a bookkeeping column.

**``answer`` is cast to string, and the cast is load-bearing.** Upstream's JSON
holds the column as a mix of the three types its values happen to take (in
``simple``: 147 int, 125 str, 7 float; in ``hard``: 132 int, 141 str, 6 float),
which Arrow cannot represent in one column at all. The cast is ``str()``, which
is exactly what upstream's own ``extract_ground_truth_answer`` applies to an int
or float gold before wrapping it in ``\\boxed{}`` -- so this reproduces upstream's
coercion at load time rather than inventing one.

**Test-only, deliberately.** Upstream's README asks in bold that these sets never
be used as training data, and the ``original_split`` column is provenance about
the *seed* MATH problem, not a division of these rows. So only a ``test`` split
is published; there is no ``train`` for a caller to reach for by accident.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import os
from typing import Literal, TypedDict

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Features, Value

#: Commit the two data files are addressed at. A branch URL would let the rows
#: move under a checksum that no longer matches — failing closed, but only after
#: a download.
MATH_PERTURB_COMMIT = "df4840f680fce405c9449008564574961c7f4df1"

_RAW_BASE = (
    "https://raw.githubusercontent.com/Kaffaljidhmah2/MATH-Perturb/"
    f"{MATH_PERTURB_COMMIT}/math_perturb"
)

#: Rows per testset at the pinned commit. Upstream's headline count, and the
#: same for both files.
MATH_PERTURB_ROWS = 279

TVariant = Literal["simple", "hard"]


class MathPerturbRow(TypedDict):
    problem_id: int
    problem: str
    answer: str
    level: str
    type: str
    original_split: str


# Explicit rather than inferred: `answer` is the mixed-type column described in
# the module docstring, and an inferred schema would either reject the file or
# silently pick whichever type the first rows happen to carry.
_FEATURES = Features(
    {
        "problem_id": Value("int64"),
        "problem": Value("string"),
        "answer": Value("string"),
        "level": Value("string"),
        "type": Value("string"),
        "original_split": Value("string"),
    }
)


def data_file(variant: TVariant) -> str:
    """Basename the URL downloader stages this variant's file under."""
    return f"math_perturb_{variant}.jsonl"


def data_url(variant: TVariant) -> str:
    """Upstream URL for this variant, pinned to :data:`MATH_PERTURB_COMMIT`."""
    return f"{_RAW_BASE}/{data_file(variant)}"


def _row(raw: dict) -> MathPerturbRow:
    return {
        "problem_id": int(raw["problem_id"]),
        "problem": str(raw["problem"]),
        # See the module docstring: upstream's own coercion, applied at load.
        "answer": str(raw["answer"]),
        "level": str(raw["level"]),
        "type": str(raw["type"]),
        "original_split": str(raw["original_split"]),
    }


def load_math_perturb(name_or_path: str, variant: TVariant) -> HFDatasetDict:
    """Load one MATH-Perturb testset as a ``test``-only DatasetDict.

    ``name_or_path`` may be the directory ``sieval dataset download`` staged the
    file into, or the JSONL file itself, so a run works from either.
    """
    path = (
        os.path.join(name_or_path, data_file(variant))
        if os.path.isdir(name_or_path)
        else name_or_path
    )
    with open(path, encoding="utf-8") as fh:
        rows = [_row(json.loads(line)) for line in fh if line.strip()]

    if not rows:
        raise ValueError(
            f"MATH-P-{variant.capitalize()} loaded 0 samples from {path!r}; "
            f"re-run 'sieval dataset download math_perturb_{variant}'."
        )
    # Not a slice guard but a pin-integrity one: the checksum already fixes the
    # bytes, so a count that is not 279 means this loader is reading some other
    # file, and a short set would otherwise be scored as if it were the benchmark.
    if len(rows) != MATH_PERTURB_ROWS:
        raise ValueError(
            f"MATH-P-{variant.capitalize()} has {len(rows)} rows at "
            f"{path!r}, expected {MATH_PERTURB_ROWS}."
        )

    rows.sort(key=lambda row: row["problem_id"])
    return HFDatasetDict(
        {"test": HFDataset.from_list([dict(row) for row in rows], features=_FEATURES)}
    )
