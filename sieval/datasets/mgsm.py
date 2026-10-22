"""
MGSM (Multilingual Grade School Math) dataset loader — ``juletxara/mgsm``.

One config per language: 8 ``train`` exemplars and 250 ``test`` problems.
``load`` reads one language, so few-shot exemplars never cross languages.
``test`` sets only ``question`` and ``answer_number``. ``train`` text already
carries upstream's prompt prefixes (``Question:`` / ``Step-by-Step Answer:``
and their translations); a few-shot template must not add them again.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
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

MGSM_REVISION = "b2f13d426afe3be8d69a7e739b36724db8b66bbc"

# HF config names at the pinned revision (the 11 languages of the MGSM paper).
MGSM_LANGUAGES = ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")


class MGSMDatasetSample(TypedDict):
    question: str
    answer: str | None
    # No cast needed: upstream ships `answer_number` as int32.
    answer_number: int
    equation_solution: str | None


@sieval_dataset(
    name="mgsm",
    display_name="MGSM",
    description="Multilingual Grade School Math — 250 GSM8K problems in 11 languages.",
    source=f"hf:juletxara/mgsm@{MGSM_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("multilingual", "math-word-problems", "open-ended"),
    # The mirror's card.
    license="CC-BY-SA-4.0",
)
class MGSMDataset(Dataset[MGSMDatasetSample]):
    @override
    def load(
        self, name_or_path: str, lang: str | None = None, **kwargs
    ) -> HFDatasetDict:
        # No default: picking one silently would score the wrong benchmark.
        if lang is None:
            raise ValueError(
                "MGSM needs `lang` (args: {lang: ...}), "
                f"one of {list(MGSM_LANGUAGES)}."
            )
        if lang not in MGSM_LANGUAGES:
            raise ValueError(
                f"Unknown MGSM language {lang!r}; "
                f"expected one of {list(MGSM_LANGUAGES)}."
            )
        return ensure_dataset_dict(load_dataset(name_or_path, lang, **kwargs))
