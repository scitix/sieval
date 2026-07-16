"""C-Eval dataset loader.

C-Eval is a Chinese multi-domain exam benchmark with 52 subjects, organised on
the Hub as 52 per-subject configs (there is no combined ``all`` config), each
with ``dev`` (few-shot exemplars with explanations), ``val``, and ``test``
splits. C-Eval has since released the ``test`` answers, so all three splits
carry labels. This loader mirrors the source as-is — it loads every subject (or
a caller-supplied subset), tags each row with its ``subject`` (the English
config key, which is absent from the raw rows), and concatenates them.

The evaluation split is selected via :func:`apply_eval_split`; with the default
(``eval_split=None``) the released ``test`` split is the eval target. Pass
``eval_split="val"`` to evaluate on the validation split instead.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import Any, TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.core.datasets import Category, Dataset, Level1Category, sieval_dataset
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

# Pin the Hub revision for reproducibility (current `main` at integration time).
CEVAL_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"


class CEvalDatasetSample(TypedDict):
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str
    subject: str


@sieval_dataset(
    name="c_eval",
    display_name="C-Eval",
    description="C-Eval — Chinese multi-domain exam benchmark with 52 subjects.",
    source=f"hf:ceval/ceval-exam@{CEVAL_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("chinese", "multiple-choice"),
    license="cc-by-nc-sa-4.0",
)
class CEvalDataset(Dataset[CEvalDatasetSample]):
    """C-Eval dataset for Chinese model evaluation.

    Row fields: ``question``, ``A``/``B``/``C``/``D`` (options), ``answer`` (the
    correct option letter), and ``subject`` (the English config key, injected by
    this loader since it is not present in the raw rows).
    """

    # All C-Eval subjects (52 total), used as the Hub config names.
    SUBJECTS = [
        "accountant",
        "advanced_mathematics",
        "art_studies",
        "basic_medicine",
        "business_administration",
        "chinese_language_and_literature",
        "civil_servant",
        "clinical_medicine",
        "college_chemistry",
        "college_economics",
        "college_physics",
        "college_programming",
        "computer_architecture",
        "computer_network",
        "discrete_mathematics",
        "education_science",
        "electrical_engineer",
        "environmental_impact_assessment_engineer",
        "fire_engineer",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_chinese",
        "high_school_geography",
        "high_school_history",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "ideological_and_moral_cultivation",
        "law",
        "legal_professional",
        "logic",
        "mao_zedong_thought",
        "marxism",
        "metrology_engineer",
        "middle_school_biology",
        "middle_school_chemistry",
        "middle_school_geography",
        "middle_school_history",
        "middle_school_mathematics",
        "middle_school_physics",
        "middle_school_politics",
        "modern_chinese_history",
        "operating_system",
        "physician",
        "plant_protection",
        "probability_and_statistics",
        "professional_tour_guide",
        "sports_science",
        "tax_accountant",
        "teacher_qualification",
        "urban_and_rural_planner",
        "veterinary_medicine",
    ]

    @override
    def load(
        self,
        name_or_path: str,
        subjects: list[str] | None = None,
        eval_split: str | None = None,
        **kwargs: Any,
    ) -> HFDatasetDict:
        subjects_to_load = subjects or self.SUBJECTS

        # C-Eval has no combined config, so load each subject and tag its rows
        # with the subject (the English config key, absent from the raw rows).
        all_splits: dict[str, list[HFDataset]] = {"dev": [], "val": [], "test": []}
        for subject in subjects_to_load:
            subject_dataset = ensure_dataset_dict(
                load_dataset(name_or_path, subject, **kwargs)
            )
            for split in all_splits:
                if split in subject_dataset:
                    all_splits[split].append(
                        subject_dataset[split].map(
                            lambda row, s=subject: {**row, "subject": s}
                        )
                    )

        combined = HFDatasetDict()
        for split, parts in all_splits.items():
            if parts:
                combined[split] = concatenate_datasets(parts)

        combined = apply_eval_split(combined, eval_split)

        if combined.get("test") is None or len(combined["test"]) == 0:
            raise ValueError(
                f"C-Eval produced no eval ('test') split for subjects="
                f"{subjects_to_load!r} (eval_split={eval_split!r}); check the "
                "subject names and that the dataset has been downloaded."
            )
        return combined
