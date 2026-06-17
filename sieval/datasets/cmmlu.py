"""
CMMLU dataset loader.

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
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("chinese", "multiple-choice"),
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
            return self._load_zip(path, subjects_to_load)

        if path.is_dir():
            zip_path = _find_zip(path)
            if zip_path is not None:
                return self._load_zip(zip_path, subjects_to_load)

            csv_root = _find_csv_root(path)
            if csv_root is not None:
                return self._load_csv_root(csv_root, subjects_to_load)

        raise FileNotFoundError(
            "CMMLU data must be a GitHub archive zip or a directory containing "
            "data/dev/*.csv and data/test/*.csv."
        )

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

    def _load_csv_root(
        self,
        data_root: Path,
        subjects_to_load: list[str],
    ) -> HFDatasetDict:
        samples: dict[str, list[CMMLUDatasetSample]] = {"dev": [], "test": []}
        for subject in subjects_to_load:
            for split in ("dev", "test"):
                csv_path = data_root / split / f"{subject}.csv"
                if not csv_path.exists():
                    continue
                with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
                    samples[split].extend(self._process_csv_rows(csv_file, subject))
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
        if _has_structured_fields(sample):
            return _process_structured_sample(sample, subject)

        raw_string = _find_raw_csv_row(sample)
        if raw_string:
            row = next(csv.reader([raw_string]))
            if len(row) >= 7:
                return {
                    "question": row[1].strip(),
                    "A": row[2].strip(),
                    "B": row[3].strip(),
                    "C": row[4].strip(),
                    "D": row[5].strip(),
                    "answer": row[6].strip().upper(),
                    "subject": subject or str(sample.get("subject", "")),
                }

        raise ValueError(
            "Malformed CMMLU row for subject "
            f"{subject or str(sample.get('subject', ''))!r}: expected structured "
            "question/A-D/answer fields or a >=7-column CSV row, got keys "
            f"{sorted(sample.keys())}."
        )


def _has_structured_fields(sample: Mapping[str, object]) -> bool:
    return any(key in sample for key in ("question", "Question"))


def _find_zip(root: Path) -> Path | None:
    preferred = root / f"{CMMLU_REVISION}.zip"
    if preferred.exists():
        return preferred

    zip_files = sorted(root.glob("*.zip"))
    if len(zip_files) == 1:
        return zip_files[0]

    for zip_file in zip_files:
        if zip_file.name == "cmmlu_v1_0_1.zip":
            return zip_file
    return None


def _find_csv_root(root: Path) -> Path | None:
    candidates = [root / "data", root]
    candidates.extend(
        child / "data" for child in sorted(root.iterdir()) if child.is_dir()
    )
    for candidate in candidates:
        if (candidate / "dev").is_dir() and (candidate / "test").is_dir():
            return candidate
    return None


def _find_zip_csv_member(
    names: list[str],
    split: str,
    subject: str,
) -> str | None:
    suffixes = (
        ("data", split, f"{subject}.csv"),
        (split, f"{subject}.csv"),
    )
    for name in names:
        parts = PurePosixPath(name).parts
        if any(parts[-len(suffix) :] == suffix for suffix in suffixes):
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


def _process_structured_sample(
    sample: Mapping[str, object],
    subject: str | None,
) -> CMMLUDatasetSample:
    return {
        "question": _get_text(sample, "question", "Question"),
        "A": _get_text(sample, "A"),
        "B": _get_text(sample, "B"),
        "C": _get_text(sample, "C"),
        "D": _get_text(sample, "D"),
        "answer": _get_text(sample, "answer", "Answer").upper(),
        "subject": subject or _get_text(sample, "subject", "Subject"),
    }


def _get_text(sample: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = sample.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _find_raw_csv_row(sample: Mapping[str, object]) -> str:
    for key, value in sample.items():
        if key == "subject":
            continue
        if "Question" in key or key.startswith(","):
            return str(value)
    for key, value in sample.items():
        if key != "subject":
            return str(value)
    return ""
