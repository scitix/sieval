"""OpenAI MMMLU full multilingual dataset loader.

This loads ``openai/MMMLU`` (human-translated MMLU test CSVs for 14 non-English
locales), not the English MMLU dataset (``MMLUDataset`` in
``sieval.datasets.mmlu``); MMMLU exposes no English locale.

Deviation from EleutherAI lm-evaluation-harness ``openai-mmmlu``, which emits
one task per locale/subject: this loader returns all selected locales in one
dataset.  ``args.locales`` (or a single Hugging Face ``config`` name) restricts
the locales; names are normalized so ``ZH_CN`` and ``ZH-CN`` resolve to
``zh_cn``.

AI-Generated Code - GPT-5-Codex (OpenAI)
"""

import os
from typing import Any, TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.core.datasets import Category, Dataset, Level1Category, sieval_dataset
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

# openai/MMMLU main HEAD (dataset last modified 2024-10-16; static since).
MMMLU_REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"

MMMLU_CATEGORY_ORDER = ("stem", "other", "social_sciences", "humanities")
MMMLU_LANGUAGES = {
    "ar_xy": {
        "dataset_name": "AR_XY",
        "display_name": "Arabic",
        "csv_name": "mmlu_AR-XY.csv",
    },
    "bn_bd": {
        "dataset_name": "BN_BD",
        "display_name": "Bengali",
        "csv_name": "mmlu_BN-BD.csv",
    },
    "de_de": {
        "dataset_name": "DE_DE",
        "display_name": "German",
        "csv_name": "mmlu_DE-DE.csv",
    },
    "es_la": {
        "dataset_name": "ES_LA",
        "display_name": "Spanish",
        "csv_name": "mmlu_ES-LA.csv",
    },
    "fr_fr": {
        "dataset_name": "FR_FR",
        "display_name": "French",
        "csv_name": "mmlu_FR-FR.csv",
    },
    "hi_in": {
        "dataset_name": "HI_IN",
        "display_name": "Hindi",
        "csv_name": "mmlu_HI-IN.csv",
    },
    "id_id": {
        "dataset_name": "ID_ID",
        "display_name": "Indonesian",
        "csv_name": "mmlu_ID-ID.csv",
    },
    "it_it": {
        "dataset_name": "IT_IT",
        "display_name": "Italian",
        "csv_name": "mmlu_IT-IT.csv",
    },
    "ja_jp": {
        "dataset_name": "JA_JP",
        "display_name": "Japanese",
        "csv_name": "mmlu_JA-JP.csv",
    },
    "ko_kr": {
        "dataset_name": "KO_KR",
        "display_name": "Korean",
        "csv_name": "mmlu_KO-KR.csv",
    },
    "pt_br": {
        "dataset_name": "PT_BR",
        "display_name": "Brazilian Portuguese",
        "csv_name": "mmlu_PT-BR.csv",
    },
    "sw_ke": {
        "dataset_name": "SW_KE",
        "display_name": "Swahili",
        "csv_name": "mmlu_SW-KE.csv",
    },
    "yo_ng": {
        "dataset_name": "YO_NG",
        "display_name": "Yoruba",
        "csv_name": "mmlu_YO-NG.csv",
    },
    "zh_cn": {
        "dataset_name": "ZH_CN",
        "display_name": "Simplified Chinese",
        "csv_name": "mmlu_ZH-CN.csv",
    },
}
MMMLU_SUBJECT_CATEGORIES = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMMLUDatasetSample(TypedDict):
    Question: str
    A: str
    B: str
    C: str
    D: str
    Answer: str
    Subject: str
    Category: str
    Locale: str
    LocaleDisplayName: str


def _normalize_answer(answer: object) -> str:
    if answer is None:
        return ""
    if isinstance(answer, int):
        return chr(65 + answer) if 0 <= answer < 4 else ""
    text = str(answer).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    if text.isdigit():
        idx = int(text)
        return chr(65 + idx) if 0 <= idx < 4 else ""
    return ""


def _normalize_subject(subject: object) -> str:
    text = str(subject or "").strip()
    for marker in ("_test.csv", "_test-"):
        idx = text.find(marker)
        if idx != -1:
            return text[:idx]
    return text


def _normalize_locale(locale: object) -> str:
    text = str(locale or "").strip().replace("-", "_").lower()
    if text in MMMLU_LANGUAGES:
        return text
    upper = text.upper()
    for slug, info in MMMLU_LANGUAGES.items():
        if upper == info["dataset_name"]:
            return slug
    raise ValueError(f"Unknown MMMLU locale: {locale!r}")


def _normalize_locales(
    locales: list[str] | None,
    config: str | None,
) -> list[str]:
    raw_locales = locales
    if raw_locales is None and config and config.lower() != "all":
        raw_locales = [config]
    if raw_locales is None:
        return list(MMMLU_LANGUAGES)
    normalized = [_normalize_locale(locale) for locale in raw_locales]
    unknown = sorted(set(normalized) - set(MMMLU_LANGUAGES))
    if unknown:
        raise ValueError(f"Unknown MMMLU locale(s): {', '.join(unknown)}")
    return normalized


def _normalize_categories(categories: list[str] | None) -> set[str] | None:
    if categories is None:
        return None
    normalized = {str(category).strip().lower() for category in categories}
    unknown = normalized - set(MMMLU_CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"Unknown MMMLU category(s): {', '.join(sorted(unknown))}")
    return normalized


def _find_local_locale_csv(name_or_path: str, locale: str) -> str | None:
    if os.path.isfile(name_or_path):
        return name_or_path if name_or_path.endswith(".csv") else None
    if not os.path.isdir(name_or_path):
        return None

    csv_name = MMMLU_LANGUAGES[locale]["csv_name"]
    candidates = [
        os.path.join(name_or_path, "test", csv_name),
        os.path.join(name_or_path, csv_name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _process_sample(
    sample: dict[str, object],
    *,
    locale: str | None = None,
) -> MMMLUDatasetSample:
    subject = _normalize_subject(sample.get("Subject", sample.get("subject")))
    raw_locale = sample.get("Locale", sample.get("locale", locale))
    normalized_locale = _normalize_locale(raw_locale)
    category = str(
        sample.get(
            "Category",
            sample.get("category", MMMLU_SUBJECT_CATEGORIES.get(subject, "other")),
        )
    ).strip()
    return {
        "Question": str(sample.get("Question", sample.get("question", ""))),
        "A": str(sample.get("A", "")),
        "B": str(sample.get("B", "")),
        "C": str(sample.get("C", "")),
        "D": str(sample.get("D", "")),
        "Answer": _normalize_answer(sample.get("Answer", sample.get("answer"))),
        "Subject": subject,
        "Category": category,
        "Locale": normalized_locale,
        "LocaleDisplayName": MMMLU_LANGUAGES[normalized_locale]["display_name"],
    }


def _load_locale_dataset(
    name_or_path: str,
    locale: str,
    *,
    eval_split: str | None,
    load_kwargs: dict[str, Any],
) -> HFDatasetDict:
    csv_path = _find_local_locale_csv(name_or_path, locale)
    if csv_path is not None:
        dataset = load_dataset("csv", data_files={"test": csv_path}, **load_kwargs)
    else:
        dataset = load_dataset(
            name_or_path,
            MMMLU_LANGUAGES[locale]["dataset_name"],
            **load_kwargs,
        )
    return apply_eval_split(ensure_dataset_dict(dataset), eval_split)


@sieval_dataset(
    name="mmmlu",
    display_name="MMMLU",
    description=(
        "OpenAI MMMLU - human-translated MMLU test split covering 57 subjects "
        "across 14 locales."
    ),
    source=f"hf:openai/MMMLU@{MMMLU_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("multilingual", "multiple-choice"),
    license="MIT",
)
class MMMLUDataset(Dataset[MMMLUDatasetSample]):
    """OpenAI MMMLU full multilingual dataset."""

    @override
    def load(
        self,
        name_or_path: str,
        config: str | None = "all",
        locales: list[str] | None = None,
        subjects: list[str] | None = None,
        categories: list[str] | None = None,
        eval_split: str | None = None,
        **kwargs: Any,
    ) -> HFDatasetDict:
        selected_locales = _normalize_locales(locales, config)
        normalized_subjects = (
            {_normalize_subject(subject) for subject in subjects}
            if subjects is not None
            else None
        )
        if normalized_subjects is not None:
            unknown_subjects = normalized_subjects - set(MMMLU_SUBJECT_CATEGORIES)
            if unknown_subjects:
                unknown = ", ".join(sorted(unknown_subjects))
                raise ValueError(f"Unknown MMMLU subject(s): {unknown}")
        normalized_categories = _normalize_categories(categories)

        processed_by_split: dict[str, list[HFDataset]] = {}
        for locale in selected_locales:
            dataset = _load_locale_dataset(
                name_or_path,
                locale,
                eval_split=eval_split,
                load_kwargs=kwargs,
            )
            for split, split_dataset in dataset.items():
                split_name = str(split)
                mapped = split_dataset.map(
                    _process_sample,
                    fn_kwargs={"locale": locale},
                    remove_columns=split_dataset.column_names,
                )
                if normalized_subjects is not None:
                    mapped = mapped.filter(
                        lambda sample: sample["Subject"] in normalized_subjects
                    )
                if normalized_categories is not None:
                    mapped = mapped.filter(
                        lambda sample: sample["Category"] in normalized_categories
                    )
                if len(mapped) > 0:
                    split_datasets = processed_by_split.get(split_name)
                    if split_datasets is None:
                        split_datasets = []
                        processed_by_split[split_name] = split_datasets
                    split_datasets.append(mapped)

        processed = HFDatasetDict()
        for split, split_datasets in processed_by_split.items():
            processed[split] = (
                split_datasets[0]
                if len(split_datasets) == 1
                else concatenate_datasets(split_datasets)
            )

        if not processed or all(
            len(split_dataset) == 0 for split_dataset in processed.values()
        ):
            raise ValueError(
                "MMMLU loader produced an empty dataset; check locales, "
                "subjects, categories, config, or schema."
            )

        return processed
