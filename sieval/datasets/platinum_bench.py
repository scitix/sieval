"""
PlatinumBench loader — one instance per subset (HuggingFace config).

PlatinumBench is a *revision* of 15 existing benchmarks rather than a new one:
MadryLab re-annotated each question with frontier models plus human review, and
published the label-noise verdict per row in ``cleaning_status``. The point of
the benchmark is that the remaining questions are unambiguous, so a wrong answer
is a model error and not a bad label.

One Dataset class serves all subsets because every config carries the same six
platinum columns; only the *original* benchmark's own columns differ (gsm8k has
``question``/``answer``, singleop has ``input``/``output_answer``/``split``,
svamp has ``Body``/``Question``/``Equation``, …). This loader keeps the six
shared columns and drops the original ones, which is what lets one sample
``TypedDict`` — and therefore one dataset FK — back N per-subset tasks. Nothing
downstream needs the original columns: the prompt is carried in
``platinum_prompt`` / ``platinum_prompt_no_cot`` (upstream builds no prompt of
its own) and the gold is ``platinum_target``.

``subset`` is stamped onto every row so a task can assert it was wired to the
dataset instance it expects, and so a shard line is self-describing.

Rows with ``cleaning_status == "rejected"`` are always dropped, with no knob:
their ``platinum_target`` is ``null`` (they are the questions the annotators
threw out), and the filtered set is what the published leaderboard scores.
Upstream's ``--unfiltered`` flag keeps them, but only so they can fail — a
``null`` target makes its ``float(platinum_target[0])`` raise, which its bare
``except`` turns into ``correct=False``. Reproducing that would mean shipping a
flag whose only effect is to add guaranteed-wrong rows.

Measured at the pinned revision, ``test`` split, over the five math subsets:
singleop 159→150, singleq 109→100, multiarith 174→170, svamp 300→265,
gsm8k 300→268 (953 kept of 1042). Surviving statuses are ``consensus``,
``verified`` and ``revised``.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

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

# The configs of madrylab/platinum-bench at the pinned revision, minus `vqa`
# (see _VQA_SUBSET). Kept as a validation set so a typo'd subset fails at
# construction with the candidate list, not as an opaque HF config error.
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

# `vqa` is the 15th config and is excluded rather than merely untested: at this
# revision its strategy column is misspelled `platinum_parsing_stratagy`, so it
# does not share the schema this loader selects, and its prompts reference COCO
# images that live outside the dataset repo.
_VQA_SUBSET = "vqa"

# Columns every non-vqa config shares. Selecting exactly these is what makes the
# 15 heterogeneous configs fit one TypedDict.
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
# `math` ones. One sample TypedDict backs all 14 configs, so it is also the single
# FK every future leaf resolves to: adding a `drop` / `squad` / `winograd_wsc` task
# means widening these here, or the new leaf renders as ElementaryMath.
@sieval_dataset(
    name="platinum_bench",
    display_name="PlatinumBench",
    description=(
        "Label-noise-cleaned revisions of 15 benchmarks; one subset per instance."
    ),
    source=f"hf:madrylab/platinum-bench@{PLATINUM_BENCH_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("english", "math-word-problems", "open-ended"),
    license="cc-by-sa-4.0",
)
class PlatinumBenchDataset(Dataset[PlatinumBenchDatasetSample]):
    def __init__(
        self,
        name_or_path: str | None = None,
        *,
        subset: str,
        **kwargs,
    ):
        if subset == _VQA_SUBSET:
            raise ValueError(
                f"PlatinumBench subset '{_VQA_SUBSET}' is not supported: at "
                f"revision {PLATINUM_BENCH_REVISION} its parsing-strategy column "
                "is misspelled 'platinum_parsing_stratagy', so it does not share "
                "the schema this loader selects, and its prompts reference COCO "
                "images stored outside the dataset repo."
            )
        if subset not in PLATINUM_SUBSETS:
            raise ValueError(
                f"Unknown PlatinumBench subset '{subset}'. "
                f"Valid subsets: {sorted(PLATINUM_SUBSETS)}"
            )
        # Set before delegating: the base __init__ calls load(), which reads it.
        self._subset = subset
        super().__init__(name_or_path, **kwargs)

    @property
    def subset(self) -> str:
        """The HuggingFace config this instance was loaded from."""
        return self._subset

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # `subset` is deliberately not a parameter: the base contract is
        # load(name_or_path, **kwargs), and __init__ has already validated the
        # subset and stored it, so there is exactly one source of truth.
        subset = self._subset
        dataset = ensure_dataset_dict(load_dataset(name_or_path, subset, **kwargs))
        dataset = dataset.filter(
            lambda status: status != _REJECTED_STATUS,
            input_columns="cleaning_status",
        )
        for split, rows in dataset.items():
            if len(rows) == 0:
                raise ValueError(
                    f"PlatinumBench subset '{subset}' split '{split}' is empty "
                    f"after dropping '{_REJECTED_STATUS}' rows — the pinned "
                    "revision's schema or cleaning_status vocabulary changed."
                )
        dataset = dataset.select_columns(list(_PLATINUM_COLUMNS))
        return dataset.map(lambda row, s=subset: {**row, "subset": s})
