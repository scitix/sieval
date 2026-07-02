"""
CMMLU dataset loader.

Loads the official CMMLU GitHub archive (``{sha}.zip``) staged by
``sieval dataset download cmmlu``. ``load`` accepts either that staged
directory (the production path) or a direct path to the ``.zip``; CSV members
are read straight from the archive with the standard ``Question/A-D/Answer``
header.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import csv
import io
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import Category, Dataset, Level1Category, sieval_dataset
from sieval.core.utils.hf import ensure_dataset_dict

CMMLU_REVISION = "d6e7b716d8ac694f38969a6c0407437d1fded799"
CMMLU_SOURCE_URL = f"https://github.com/haonan-li/CMMLU/archive/{CMMLU_REVISION}.zip"


class CMMLUDatasetSample(TypedDict):
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str
    subject: str


@sieval_dataset(
    name="cmmlu",
    display_name="CMMLU",
    description="CMMLU Chinese multi-domain exam benchmark with 67 subjects.",
    source=f"url:{CMMLU_SOURCE_URL}",
    # Basename embeds the pinned commit SHA, so bumping CMMLU_REVISION also
    # means refreshing this digest.
    checksums={
        "d6e7b716d8ac694f38969a6c0407437d1fded799.zip": "sha256:154593336d5074d793ed990222876b83490b0aed97638a62618d1fe2da7c2cac",  # noqa: E501
    },
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("chinese", "multiple-choice"),
    license="CC-BY-NC-SA-4.0",
)
class CMMLUDataset(Dataset[CMMLUDatasetSample]):
    SUBJECTS = [
        "agronomy",
        "anatomy",
        "ancient_chinese",
        "arts",
        "astronomy",
        "business_ethics",
        "chinese_civil_service_exam",
        "chinese_driving_rule",
        "chinese_food_culture",
        "chinese_foreign_policy",
        "chinese_history",
        "chinese_literature",
        "chinese_teacher_qualification",
        "clinical_knowledge",
        "college_actuarial_science",
        "college_education",
        "college_engineering_hydrology",
        "college_law",
        "college_mathematics",
        "college_medical_statistics",
        "college_medicine",
        "computer_science",
        "computer_security",
        "conceptual_physics",
        "construction_project_management",
        "economics",
        "education",
        "electrical_engineering",
        "elementary_chinese",
        "elementary_commonsense",
        "elementary_information_and_technology",
        "elementary_mathematics",
        "ethnology",
        "food_science",
        "genetics",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_geography",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "human_sexuality",
        "international_law",
        "journalism",
        "jurisprudence",
        "legal_and_moral_basis",
        "logical",
        "machine_learning",
        "management",
        "marketing",
        "marxist_theory",
        "modern_chinese",
        "nutrition",
        "philosophy",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_study",
        "sociology",
        "sports_science",
        "traditional_chinese_medicine",
        "virology",
        "world_history",
        "world_religions",
    ]

    @override
    def load(
        self,
        name_or_path: str,
        subjects: list[str] | None = None,
        **kwargs: Any,
    ) -> HFDatasetDict:
        _ = kwargs
        subjects_to_load = subjects or self.SUBJECTS
        path = Path(name_or_path)

        if path.is_file() and path.suffix == ".zip":
            zip_path = path
        else:
            zip_path = path / f"{CMMLU_REVISION}.zip"

        if not zip_path.is_file():
            raise FileNotFoundError(
                f"CMMLU archive not found at {str(zip_path)!r}. Run "
                "'sieval dataset download cmmlu' to stage the official "
                f"{CMMLU_REVISION}.zip GitHub archive."
            )
        return self._load_zip(zip_path, subjects_to_load)

    def _load_zip(
        self,
        zip_path: Path,
        subjects_to_load: list[str],
    ) -> HFDatasetDict:
        samples: dict[str, list[CMMLUDatasetSample]] = {"dev": [], "test": []}
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            for split in ("dev", "test"):
                for subject in subjects_to_load:
                    member = _find_zip_csv_member(names, split, subject)
                    if member is None:
                        continue
                    with archive.open(member) as raw_file:
                        text_file = io.TextIOWrapper(
                            raw_file,
                            encoding="utf-8-sig",
                            newline="",
                        )
                        samples[split].extend(
                            self._process_csv_rows(text_file, subject)
                        )
        return _dataset_dict_from_samples(samples)

    def _process_csv_rows(
        self,
        csv_file: io.TextIOBase,
        subject: str,
    ) -> list[CMMLUDatasetSample]:
        reader = csv.DictReader(csv_file)
        return [self._process_sample(row, subject) for row in reader]

    def _process_sample(
        self,
        sample: Mapping[str, object],
        subject: str | None = None,
    ) -> CMMLUDatasetSample:
        if not any(key in sample for key in ("question", "Question")):
            raise ValueError(
                "Malformed CMMLU row for subject "
                f"{subject or str(sample.get('subject', ''))!r}: expected a "
                "Question/A-D/Answer CSV header, got keys "
                f"{sorted(sample.keys())}."
            )
        return {
            "question": _get_text(sample, "question", "Question"),
            "A": _get_text(sample, "A"),
            "B": _get_text(sample, "B"),
            "C": _get_text(sample, "C"),
            "D": _get_text(sample, "D"),
            "answer": _get_text(sample, "answer", "Answer").upper(),
            "subject": subject or _get_text(sample, "subject", "Subject"),
        }


def _find_zip_csv_member(
    names: list[str],
    split: str,
    subject: str,
) -> str | None:
    suffix = ("data", split, f"{subject}.csv")
    for name in names:
        if PurePosixPath(name).parts[-len(suffix) :] == suffix:
            return name
    return None


def _dataset_dict_from_samples(
    samples: dict[str, list[CMMLUDatasetSample]],
) -> HFDatasetDict:
    if not any(samples.values()):
        raise ValueError(
            "No CMMLU samples were loaded; check the data path and that the "
            "requested subjects match the available dev/test CSV files."
        )
    dataset_dict = HFDatasetDict()
    for split, split_samples in samples.items():
        rows: list[dict[str, object]] = [{**sample} for sample in split_samples]
        dataset_dict[split] = HFDataset.from_list(rows)
    return ensure_dataset_dict(dataset_dict)


def _get_text(sample: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = sample.get(key)
        if value is not None:
            return str(value).strip()
    return ""
