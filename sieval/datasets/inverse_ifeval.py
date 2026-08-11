"""Inverse IFEval dataset loader (M-A-P; Zhang et al., 2025, arXiv:2509.04292).

1,012 "counter-intuitive" instruction-following prompts — 506 Chinese and 506
English, balanced type-by-type — across eight challenge types whose instructions
conflict with the conventions models absorb during SFT. Each row carries a
``response_reference`` stating what a passing answer must do, plus **its own
judge assets**: ``judge_system_prompt`` (one of four, selected per type) and
``judge_prompt_template`` (one of three). Grading them is the task's job; this
loader only guarantees the columns are there.

The Hub repo publishes no split and two data files — ``Inverse_IFEval_Dataset.json``
and a differently-dated ``_0908.csv`` snapshot — so ``data_files`` names the JSON
explicitly. Measured, not assumed: auto-detection does *not* break, it resolves to
the JSON anyway (verified on the staged revision: 1,012 rows, correct columns).
The pin is there because that outcome rests on HF's undocumented file-extension
precedence choosing between two snapshots — if that order changed, or upstream
dropped the JSON, the eval would quietly move to the other one with nothing in the
report to show it. The single resulting split is mirrored to ``test``.

``language`` (default ``None`` = both) is a sieval addition — upstream ships one
file and reports the two languages as separate tables. It filters at load time,
before any ``operations:`` sampling, so a sampling budget covers exactly the set
that gets graded. Unset is the useful default: the task reports per-language
breakdowns anyway, so one full run reproduces both paper tables.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.community.inverse_ifeval import LANGUAGES
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

# Pin the Hub revision for reproducibility (`main` at integration time).
INVERSE_IFEVAL_REVISION = "35f1da157640526e62b7685b682d748fa55ccfd0"

# The repo also carries a `_0908.csv` snapshot; name the JSON the card documents
# rather than let extension precedence pick between the two (docstring).
INVERSE_IFEVAL_DATA_FILE = "Inverse_IFEval_Dataset.json"


class InverseIFEvalDatasetSample(TypedDict):
    instruction_types: str
    prompt: str
    response_reference: str
    language: str
    judge_prompt_template: str
    judge_system_prompt: str


@sieval_dataset(
    name="inverse_ifeval",
    display_name="Inverse IFEval",
    description=(
        "Counter-intuitive instruction following — 1,012 bilingual prompts, "
        "LLM-judge graded."
    ),
    source=f"hf:m-a-p/Inverse_IFEval@{INVERSE_IFEVAL_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("chinese", "english", "open-ended"),
    # Neither the Hub card nor the paper states a license, and the repo ships no
    # LICENSE file — recorded as unknown rather than guessed.
    license=None,
)
class InverseIFEvalDataset(Dataset[InverseIFEvalDatasetSample]):
    #: Columns the task reads; every one is load-time mandatory because the judge
    #: prompt is data here, not code. A missing one would otherwise surface as a
    #: KeyError per sample, mid-run, after the candidate has already been paid for.
    _REQUIRED_COLUMNS = (
        "instruction_types",
        "prompt",
        "response_reference",
        "language",
        "judge_prompt_template",
        "judge_system_prompt",
    )

    # Fallback for instances built from a pre-materialized dict, which bypass
    # `load`; the real value is recorded there.
    _language: str | None = None

    @property
    def language(self) -> str | None:
        """Which language subset was kept at load time (``None`` = both)."""
        return self._language

    @override
    def load(
        self, name_or_path: str, *, language: str | None = None, **kwargs
    ) -> HFDatasetDict:
        """Load the JSON file, optionally keeping one language.

        *language* is captured explicitly so it cannot leak into
        ``load_dataset``, and is validated against the two names the pinned
        revision uses — a typo would otherwise filter the split to empty and
        report a clean 0.0 over no samples.
        """
        if language is not None and language not in LANGUAGES:
            raise ValueError(
                f"Unknown Inverse IFEval language {language!r}; "
                f"expected one of {', '.join(LANGUAGES)} (or None for both)."
            )
        self._language = language

        dataset = ensure_dataset_dict(
            load_dataset(name_or_path, data_files=INVERSE_IFEVAL_DATA_FILE, **kwargs)
        )
        dataset = apply_eval_split(dataset, "train")
        self._check_columns(dataset)
        if language is None:
            return dataset

        filtered = HFDatasetDict(
            {
                split: ds.filter(
                    lambda row_language: row_language == language,
                    input_columns=["language"],
                )
                for split, ds in dataset.items()
            }
        )
        if "test" in filtered and len(filtered["test"]) == 0:
            raise ValueError(
                f"Inverse IFEval language {language!r} selected no samples; the "
                "source's `language` column may have drifted from "
                f"{LANGUAGES}."
            )
        return filtered

    def _check_columns(self, dataset: HFDatasetDict) -> None:
        for split, ds in dataset.items():
            missing = [
                column
                for column in self._REQUIRED_COLUMNS
                if column not in ds.column_names
            ]
            if missing:
                raise ValueError(
                    f"Inverse IFEval split {split!r} is missing required "
                    f"column(s): {', '.join(missing)}. The pinned revision "
                    "carries all six, so this is upstream schema drift — and the "
                    "judge prompt is per-sample DATA here, not a task constant, "
                    "so grading cannot proceed without it."
                )
