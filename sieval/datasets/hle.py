"""Humanity's Last Exam (HLE) dataset loader (Center for AI Safety).

HLE (Phan et al., 2025) is a multi-domain, closed-ended academic benchmark of
frontier-difficulty questions (mathematics, sciences, humanities), each with an
``exactMatch`` or ``multipleChoice`` gold answer suitable for automated
LLM-judge grading. The Hub repo exposes a single ``test`` split; this loader
mirrors it as-is.

Access: ``cais/hle`` is a gated Hub repo — downloading requires accepting the
gate on the dataset page and an authenticated ``HF_TOKEN`` in the environment.

Subset selection (``text_only``, default ``True``) is a sieval addition —
upstream ``hle_eval`` @ 26dca2e always grades the full set. It applies at load
time, before any ``operations:`` sampling, so a sampling budget covers exactly
the set that gets graded. Image questions are dropped by default because the
full set needs a vision-capable candidate *and* judge; reports mark text-only
numbers with ``*``, the full set being the unmarked headline.

The model-facing image is the ``image`` column — a base64 data URI string
(``""`` when absent), preserved untouched. The auxiliary ``image_preview`` /
``rationale_image`` columns are ``Image`` features nothing consumes, but they
would pull in Pillow once a row is materialized, so their decoding is disabled
(raw bytes kept). Upstream needs neither the guard nor Pillow: it reads the
split with ``.to_dict()``, a bulk export that skips the per-example decoder,
whereas sieval materializes rows one at a time. Their presence is asserted
first — ``cast_column`` fails open, silently adding an empty column instead of
raising.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import Image, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (`main` at integration time).
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"


class HLEDatasetSample(TypedDict):
    id: str
    question: str
    image: str
    answer: str
    answer_type: str
    author_name: str
    rationale: str
    raw_subject: str
    category: str


@sieval_dataset(
    name="hle",
    display_name="Humanity's Last Exam",
    description="HLE — multi-domain, closed-ended frontier academic benchmark.",
    source=f"hf:cais/hle@{HLE_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "reasoning", "academic"),
    license="MIT",
)
class HLEDataset(Dataset[HLEDatasetSample]):
    # Auxiliary Image feature columns no task consumes; decoding them on row
    # access would require Pillow, so disable it (raw bytes are kept).
    _IMAGE_FEATURE_COLUMNS = ("image_preview", "rationale_image")

    # Fallback for instances built from a pre-materialized dict, which bypass
    # `load`; the real value is recorded there.
    _text_only: bool = True

    @property
    def text_only(self) -> bool:
        """Whether image questions were dropped at load time."""
        return self._text_only

    @override
    def load(
        self, name_or_path: str, *, text_only: bool = True, **kwargs
    ) -> HFDatasetDict:
        """Load HLE, dropping image questions when *text_only*.

        *text_only* is captured explicitly so it cannot leak into
        ``load_dataset``. The filter reads only the ``image`` column so HF never
        materializes the ``Image`` features. Raises if it empties the ``test``
        split — the ``image`` column is missing, or every question is
        multi-modal.
        """
        self._text_only = text_only
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))
        for split in dataset:
            missing = [
                column
                for column in self._IMAGE_FEATURE_COLUMNS
                if column not in dataset[split].column_names
            ]
            if missing:
                raise ValueError(
                    f"HLE split {split!r} is missing auxiliary image column(s): "
                    f"{', '.join(missing)}. The pinned revision carries both, so "
                    "this is upstream schema drift; `cast_column` would silently "
                    "add them as empty and lose the Pillow guard."
                )
            for column in self._IMAGE_FEATURE_COLUMNS:
                dataset[split] = dataset[split].cast_column(column, Image(decode=False))
        if not text_only:
            return dataset

        filtered = HFDatasetDict(
            {
                split: ds.filter(lambda image: not image, input_columns=["image"])
                for split, ds in dataset.items()
            }
        )
        if "test" in filtered and len(filtered["test"]) == 0:
            raise ValueError(
                "HLE text-only selection produced an empty 'test' split; the "
                "source may lack the 'image' column or contain only multi-modal "
                "questions."
            )
        return filtered
