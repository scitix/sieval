"""MultiPL-E HumanEval loader — HumanEval translated to 24 languages.

Language selection is a dataset concern: pass ``languages`` (or a single
HuggingFace ``config``) to restrict it. The default is every language upstream
publishes, which is what the benchmark is; whether the deployed code evaluator
can *run* a given language is a separate question, answered by the task at
setup (see ``sieval/tasks/multipl_e/_base.py``).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import override

from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import Category, Dataset, Level1Category, sieval_dataset

from ._multipl_e import (
    HUMANEVAL_LANGUAGES,
    MULTIPL_E_REVISION,
    MultiPLESampleFields,
    load_multipl_e,
)


class MultiPLEHumanEvalDatasetSample(MultiPLESampleFields):
    """One translated HumanEval problem. Schema documented on the base."""


@sieval_dataset(
    name="multipl_e_humaneval",
    display_name="MultiPL-E (HumanEval)",
    description=(
        "HumanEval translated into 24 programming languages; graded by "
        "compiling and running the program."
    ),
    source=f"hf:nuprl/MultiPL-E@{MULTIPL_E_REVISION}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("multilingual", "code-exec"),
    license="MIT",
)
class MultiPLEHumanEvalDataset(Dataset[MultiPLEHumanEvalDatasetSample]):
    """MultiPL-E's ``humaneval-*`` configs, loaded as one dataset."""

    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = None,
        languages: list[str] | None = None,
        eval_split: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        return load_multipl_e(
            name_or_path,
            suite="humaneval",
            available=HUMANEVAL_LANGUAGES,
            languages=languages,
            config=config,
            eval_split=eval_split,
            **kwargs,
        )
