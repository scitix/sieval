"""AdvancedIF dataset loader.

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
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

ADVANCED_IF_REVISION = "e20cba9b94b59c027dfab00b29244e8bc42e4ab4"


class AdvancedIFDatasetSample(TypedDict):
    """One AdvancedIF prompt.

    Upstream ships all three columns as strings, including the two that hold
    JSON; they are decoded in the task rather than here, so the persisted prompt
    record keeps the bytes the grader prompt was actually built from. No cast
    needed -- the pinned revision already ships these as strings.

    Attributes:
        conversation_history: JSON list of ``{"role", "content"}`` turns, ending
            on the user prompt to answer (the assistant turn under test is
            absent by construction). A system-steerability row leads with a
            ``system`` turn.
        benchmark_name: Which of the three aspects the row belongs to --
            ``complex_if_single_turn_v5`` (402), ``system_steerability_v2``
            (507) or ``carried_context_multi_turn_eval_v5`` (736).
        prompt_metadata: JSON object whose ``rubrics`` key holds the
            expert-written yes/no checks (itself sometimes JSON-encoded again).
    """

    conversation_history: str
    benchmark_name: str
    prompt_metadata: str


@sieval_dataset(
    name="advanced_if",
    display_name="AdvancedIF",
    description=(
        "Expert-written prompts with human-curated rubrics for advanced "
        "instruction following."
    ),
    source=f"hf:facebook/AdvancedIF@{ADVANCED_IF_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("english", "open-ended"),
    # Non-commercial. The judge prompts carry the same terms and are likewise
    # not redistributed -- see sieval.community.advanced_if.
    license="CC-BY-NC-4.0",
)
class AdvancedIFDataset(Dataset[AdvancedIFDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # AdvancedIF ships its 1,645 rows in a single "train" split (the card
        # calls the same rows "test"); mirror it to "test" for the runner.
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))
        return apply_eval_split(dataset, "train")
