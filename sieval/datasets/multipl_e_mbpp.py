"""MultiPL-E MBPP loader — MBPP translated to 23 languages.

Twenty-three, not twenty-four: upstream translated HumanEval to Dart and MBPP
not, so this suite has no ``mbpp-dart`` config.

Language selection is a dataset concern: pass ``languages`` (or a single
HuggingFace ``config``) to restrict it. The default is every language upstream
publishes; whether the deployed code evaluator can *run* one is answered by the
task at setup (see ``sieval/tasks/multipl_e/_base.py``).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import override

from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import Category, Dataset, Level1Category, sieval_dataset

from ._multipl_e import (
    MBPP_LANGUAGES,
    MULTIPL_E_REVISION,
    MultiPLESampleFields,
    load_multipl_e,
)


class MultiPLEMbppDatasetSample(MultiPLESampleFields):
    """One translated MBPP problem. Schema documented on the base."""


@sieval_dataset(
    name="multipl_e_mbpp",
    display_name="MultiPL-E (MBPP)",
    description=(
        "MBPP translated into 23 programming languages; graded by compiling "
        "and running the program."
    ),
    source=f"hf:nuprl/MultiPL-E@{MULTIPL_E_REVISION}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("multilingual", "code-exec"),
    license="MIT",
)
class MultiPLEMbppDataset(Dataset[MultiPLEMbppDatasetSample]):
    """MultiPL-E's ``mbpp-*`` configs, loaded as one dataset."""

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
            suite="mbpp",
            available=MBPP_LANGUAGES,
            languages=languages,
            config=config,
            eval_split=eval_split,
            **kwargs,
        )
