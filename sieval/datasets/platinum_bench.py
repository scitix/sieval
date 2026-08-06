"""
PlatinumBench loader — all subsets in one ``test`` split, narrowed by the caller.

PlatinumBench is a *revision* of 15 existing benchmarks rather than a new one:
MadryLab re-annotated each question with frontier models plus human review, and
published the label-noise verdict per row in ``cleaning_status``. The point of
the benchmark is that the remaining questions are unambiguous, so a wrong answer
is a model error and not a bad label.

One Dataset class serves every subset because all 14 non-vqa configs carry the
same six platinum columns — and, measured at the pinned revision, carry them
with the *same* HuggingFace feature types, so they concatenate without a cast.
Only the *original* benchmark's own columns differ (gsm8k has
``question``/``answer``, singleop has ``input``/``output_answer``/``split``,
svamp has ``Body``/``Question``/``Equation``, …). This loader keeps the six
shared columns and drops the original ones, which is what lets one sample
``TypedDict`` — and therefore one dataset FK — back N per-subset tasks. Nothing
downstream needs the original columns: the prompt is carried in
``platinum_prompt`` / ``platinum_prompt_no_cot`` (upstream builds no prompt of
its own) and the gold is ``platinum_target``.

**Selecting a subset is the caller's job, not a constructor argument.** A
HuggingFace config is a load-time choice, so this loader reads all 14 and stamps
``subset`` onto every row; narrowing to one is then an ordinary
``Dataset.filter`` — the same transform any dataset gets, rather than a required
keyword argument on this one class::

    datasets:
      platinum_gsm8k:
        class: PlatinumBenchDataset
        operations:
          - filter: {by: subset, value: gsm8k}

Each config is loaded, cleaned and stamped exactly as it would be on its own and
only then concatenated, so filtering back down to one subset reproduces that
subset's rows in their original order — identical sample ids, and therefore
identical scores, to loading it alone.

Rows with ``cleaning_status == "rejected"`` are always dropped, with no knob:
their ``platinum_target`` is ``null`` (they are the questions the annotators
threw out), and the filtered set is what the published leaderboard scores.
Upstream's ``--unfiltered`` flag keeps them, but only so they can fail — a
``null`` target makes its ``float(platinum_target[0])`` raise, which its bare
``except`` turns into ``correct=False``. Reproducing that would mean shipping a
flag whose only effect is to add guaranteed-wrong rows.

Measured at the pinned revision, ``test`` split: 2725 rows kept of 3062 across
the 14 configs. Over the five math subsets that ship a task: singleop 159→150,
singleq 109→100, multiarith 174→170, svamp 300→265, gsm8k 300→268 (953 kept of
1042). Surviving statuses are ``consensus``, ``verified`` and ``revised``.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the madrylab/platinum-bench snapshot consumed by `sieval dataset download`;
# the loader reads the already-staged local dir, so the revision is not forwarded
# to `load_dataset` (it would be a no-op on local files). Pass the bare hub id
# instead of the staged path and you get whatever is current on the Hub — the row
# counts below are the pinned revision's.
PLATINUM_BENCH_REVISION = "51920a33bfb4620c789729ace14141e87a14969b"

# The configs of madrylab/platinum-bench at the pinned revision that this loader
# merges. `vqa` is the 15th and is excluded rather than merely untested: at this
# revision its strategy column is misspelled `platinum_parsing_stratagy`, so it
# does not share the schema selected below and would fail `select_columns`
# outright; its prompts also reference COCO images stored outside the dataset
# repo. A frozenset, so `load` sorts it — that sort is what makes the merged row
# order deterministic.
PLATINUM_SUBSETS: frozenset[str] = frozenset(
    {
        "bbh_logical_deduction_three_objects",
        "bbh_navigate",
        "bbh_object_counting",
        "drop",
        "gsm8k",
        "hotpotqa",
        "mmlu_math",
        "multiarith",
        "singleop",
        "singleq",
        "squad",
        "svamp",
        "tab_fact",
        "winograd_wsc",
    }
)

# Columns every non-vqa config shares. Selecting exactly these is what makes the
# 14 heterogeneous configs fit one TypedDict — and what lets them concatenate.
_PLATINUM_COLUMNS = (
    "cleaning_status",
    "original_target",
    "platinum_parsing_strategy",
    "platinum_prompt",
    "platinum_prompt_no_cot",
    "platinum_target",
)

_REJECTED_STATUS = "rejected"


class PlatinumBenchDatasetSample(TypedDict):
    """One PlatinumBench row: the six shared platinum columns plus its subset.

    ``platinum_target`` holds the cleaned gold answer(s) and ``original_target``
    the pre-cleaning label; both are sequences upstream even where a single
    answer is the only possibility (all 953 math rows have exactly one). No cast
    is applied to either — upstream already ships them as string sequences.
    """

    subset: str
    cleaning_status: str
    platinum_prompt: str
    platinum_prompt_no_cot: str
    platinum_target: list[str]
    original_target: list[str]
    platinum_parsing_strategy: str


# `categories` / `tags` describe the subsets that currently ship a task — the five
# `math` ones — not everything the merged split holds. One sample TypedDict backs
# all 14 configs, so it is also the single FK every future leaf resolves to:
# adding a `drop` / `squad` / `winograd_wsc` task means widening these here, or
# the new leaf renders as ElementaryMath.
@sieval_dataset(
    name="platinum_bench",
    display_name="PlatinumBench",
    description=(
        "Label-noise-cleaned revisions of 15 benchmarks; "
        "14 subsets in one split, keyed by `subset`."
    ),
    source=f"hf:madrylab/platinum-bench@{PLATINUM_BENCH_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("english", "math-word-problems", "open-ended"),
    license="cc-by-sa-4.0",
)
class PlatinumBenchDataset(Dataset[PlatinumBenchDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        if "subset" in kwargs:
            # This used to be a required constructor argument. Without this
            # branch a stale `args.subset` reaches `load_dataset`, which fails
            # with an opaque TypeError about an unexpected keyword. Same reason
            # `session.py` keeps a rename hint for `select` -> `slice`.
            raise ValueError(
                "PlatinumBenchDataset no longer takes a 'subset' argument: it "
                "loads every subset and you narrow it with a filter operation. "
                "Replace `args: {subset: "
                f"{kwargs['subset']}"
                "}` with `operations: [{filter: {by: subset, value: "
                f"{kwargs['subset']}"
                "}}]`."
            )
        # Each config is loaded, cleaned and stamped in full isolation and only
        # then concatenated, so the merged split contains each subset's rows in
        # exactly the order loading that subset alone would have produced. That
        # is what makes `filter(by="subset", ...)` sample-id-identical to a
        # single-config load, and therefore score-identical.
        per_split: dict[str, list[HFDataset]] = {}
        # Sorted, because PLATINUM_SUBSETS is a frozenset: iteration order would
        # otherwise vary with the string hash seed and shuffle the merged rows.
        for subset in sorted(PLATINUM_SUBSETS):
            config = ensure_dataset_dict(load_dataset(name_or_path, subset, **kwargs))
            config = config.filter(
                lambda status: status != _REJECTED_STATUS,
                input_columns="cleaning_status",
            )
            # Sorted here too, so a config contributing several splits does so in
            # a fixed order. `str(...)` because a DatasetDict key is typed
            # `str | NamedSplit`, and the merged dict is keyed by plain names.
            for raw_split, rows in sorted(config.items(), key=lambda kv: str(kv[0])):
                split = str(raw_split)
                if len(rows) == 0:
                    raise ValueError(
                        f"PlatinumBench subset '{subset}' split '{split}' is empty "
                        f"after dropping '{_REJECTED_STATUS}' rows — the pinned "
                        "revision's schema or cleaning_status vocabulary changed."
                    )
                stamped = rows.select_columns(list(_PLATINUM_COLUMNS)).map(
                    lambda row, s=subset: {**row, "subset": s}
                )
                per_split.setdefault(split, []).append(stamped)

        return HFDatasetDict(
            {split: concatenate_datasets(parts) for split, parts in per_split.items()}
        )
