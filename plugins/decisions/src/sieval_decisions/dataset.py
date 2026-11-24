"""Local JSONL decision questions with labels kept out of inference requests.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import DatasetDict

from sieval.core.datasets import Dataset

from ._validation import object_value
from .samples import request_from_sample


class DecisionJSONLDataset(Dataset):
    """Keep heterogeneous JSON rows in a string column to preserve native types."""

    def load(self, name_or_path: str, **kwargs) -> DatasetDict:
        if kwargs:
            raise ValueError(f"unsupported JSONL loader arguments: {sorted(kwargs)}")
        rows = []
        for line_number, line in enumerate(
            Path(name_or_path).read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                sample = object_value(json.loads(line), "sample")
                request_from_sample(sample)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"invalid decision sample at line {line_number}: {exc}"
                ) from exc
            rows.append({"record_json": line})
        if not rows:
            raise ValueError("decision JSONL dataset must contain at least one sample")
        return DatasetDict({"test": HFDataset.from_list(rows)})
