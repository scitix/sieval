"""
Shared loader logic for the AI2 ARC dataset family (ARC-Easy / ARC-Challenge).

Both splits share an identical schema and normalization contract — the only
difference is the HuggingFace config name — so the processing lives here and
the two dataset modules differ only in metadata and the config they pass.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from typing import Any, TypedDict

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

# Pin the AI2 ARC snapshot consumed by `sieval dataset download`; the loader
# reads the already-staged local dir, so the revision is not forwarded to
# `load_dataset` (it would be a no-op on local files).
AI2_ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"


class ARCSample(TypedDict):
    question: str
    choices: list[str]
    answer: int


def process_arc_sample(sample: dict[str, Any]) -> ARCSample:
    """Normalize one ``allenai/ai2_arc`` row to ``{question, choices, answer}``.

    The upstream schema is fixed: ``question`` is a string, ``choices`` is a
    dict with parallel ``text``/``label`` lists, and ``answerKey`` is one of
    those labels (labels are ``A``-``E`` or ``1``-``5`` depending on the row).
    The stored ``answer`` is the 0-based index of ``answerKey`` within the
    choices, so downstream tasks can relabel uniformly.
    """
    question = str(sample["question"])
    choices_field = sample["choices"]
    labels = [str(label) for label in choices_field["label"]]
    texts = [str(text) for text in choices_field["text"]]

    answer_key = str(sample["answerKey"]).strip()
    if answer_key not in labels:
        raise ValueError(
            f"ARC answerKey {answer_key!r} not found in choice labels {labels!r}; "
            "the dataset schema may have changed."
        )

    return {
        "question": question,
        "choices": texts,
        "answer": labels.index(answer_key),
    }


def load_arc(
    name_or_path: str,
    config: str,
    eval_split: str | None = None,
    **kwargs: Any,
) -> HFDatasetDict:
    """Load an ARC config (``"ARC-Easy"`` / ``"ARC-Challenge"``) as a DatasetDict."""
    dataset = load_dataset(name_or_path, config, **kwargs)
    dataset = ensure_dataset_dict(dataset)
    dataset = apply_eval_split(dataset, eval_split)
    return dataset.map(process_arc_sample)
