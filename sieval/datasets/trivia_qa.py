"""
TriviaQA dataset loader — the ``rc.nocontext`` config of ``mandarjoshi/trivia_qa``.

The source stages only ``rc.nocontext/*`` (~60 MB of a ~35 GB repo): the
closed-book questions, whose evidence columns are empty and dropped. Upstream
``test`` is unlabeled (``answer.value == "<unk>"`` on all 17,210 rows), so
``eval_split`` defaults to ``validation`` (17,944 rows), as for HellaSwag.

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
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

TRIVIA_QA_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
TRIVIA_QA_CONFIG = "rc.nocontext"

_COLUMNS = ["question", "question_id", "question_source", "answer"]


class TriviaQAAnswer(TypedDict):
    aliases: list[str]
    normalized_aliases: list[str]
    matched_wiki_entity_name: str
    normalized_matched_wiki_entity_name: str
    normalized_value: str
    type: str
    value: str


class TriviaQADatasetSample(TypedDict):
    question: str
    question_id: str
    question_source: str
    answer: TriviaQAAnswer


@sieval_dataset(
    name="trivia_qa",
    display_name="TriviaQA",
    description="TriviaQA closed-book (rc.nocontext) — 17,944 validation questions.",
    source=f"hf:mandarjoshi/trivia_qa@{TRIVIA_QA_REVISION}#{TRIVIA_QA_CONFIG}/*",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "open-ended"),
    # The mirror's card says `unknown`; this is mandarjoshi90/triviaqa's.
    license="Apache-2.0",
)
class TriviaQADataset(Dataset[TriviaQADatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        eval_split: str | None = "validation",
        **kwargs,
    ) -> HFDatasetDict:
        # By `data_dir`: a config-name lookup also resolves the card's other,
        # unstaged configs.
        dataset = ensure_dataset_dict(
            load_dataset(name_or_path, data_dir=TRIVIA_QA_CONFIG, **kwargs)
        )
        dataset = HFDatasetDict(
            {split: rows.select_columns(_COLUMNS) for split, rows in dataset.items()}
        )
        return apply_eval_split(dataset, eval_split)
