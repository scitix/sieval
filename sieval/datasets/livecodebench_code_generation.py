import os
from datetime import datetime
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

LIVECODEBENCH_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

VERSION_FILES = {
    "release_v0": [],  # placeholder for initial version
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}


class LiveCodeBenchDatasetSample(TypedDict):
    question_content: str
    contest_date: datetime
    starter_code: str
    public_test_cases: str
    private_test_cases: str
    metadata: str


@sieval_dataset(
    name="livecodebench_code_generation",
    display_name="LiveCodeBench Code Generation",
    description="LiveCodeBench code generation lite — contamination-free benchmark.",
    source=f"hf:livecodebench/code_generation_lite@{LIVECODEBENCH_REVISION}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "python", "code-exec"),
    # Mirrors upstream HF label verbatim (unversioned 'cc'); the license
    # field accepts free-form strings, so no normalization.
    license="cc",
)
class LiveCodeBenchDataset(Dataset[LiveCodeBenchDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        # 'release_vX' or 'vX_vY'; v6 is the current superset — v1 predates the
        # 2024-08+ contest windows, so the default must not strand date filters.
        version_tag: str = "release_v6",
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        if os.path.isdir(name_or_path):
            if version_tag in VERSION_FILES:
                data_files = {
                    "test": [
                        os.path.join(name_or_path, f)
                        for f in VERSION_FILES[version_tag]
                    ]
                }
            else:
                # assume `version_tag` is in the form of 'vX_vY'
                start_v, end_v = version_tag.split("_")
                # include all files in [start_v, end_v]
                start_int, end_int = (int(start_v[1:]) - 1, int(end_v[1:]))
                start_key, end_key = f"release_v{start_int}", f"release_v{end_int}"
                files = set(VERSION_FILES[end_key]) - set(VERSION_FILES[start_key])
                data_files = {"test": [os.path.join(name_or_path, f) for f in files]}
            dataset = load_dataset("json", data_files=data_files, **kwargs)
        else:
            dataset = load_dataset(
                name_or_path, version_tag=version_tag, trust_remote_code=True, **kwargs
            )
        dataset = ensure_dataset_dict(dataset)

        # filter by date first
        if start_date is not None:
            p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            dataset = dataset.filter(lambda e: p_start_date <= e["contest_date"])
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            dataset = dataset.filter(lambda e: e["contest_date"] <= p_end_date)

        # Surface an empty window loudly instead of a silent score of 0.0.
        if any(len(split) == 0 for split in dataset.values()):
            raise ValueError(
                "LiveCodeBench dataset is empty after filtering "
                f"(version_tag={version_tag!r}, start_date={start_date!r}, "
                f"end_date={end_date!r}); check the window against the release."
            )

        return dataset
