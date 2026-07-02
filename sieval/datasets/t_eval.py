import json
import os
from typing import Literal, TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from openai.types.chat import ChatCompletionMessage

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

T_EVAL_REVISION = "af355ab2b62cdbae4262ac41c7529ffeae395012"


class TEvalBeforeCallingDatasetSampleMetaData(TypedDict):
    response_format: Literal["json"]


class TEvalBeforeCallingDatasetSample(TypedDict):
    template: str | None
    meta_data: TEvalBeforeCallingDatasetSampleMetaData
    origin_prompt: list[ChatCompletionMessage]
    ground_truth: str  # stored as json string


@sieval_dataset(
    name="t_eval_before_calling",
    display_name="T-Eval Before-Calling",
    description="T-Eval tool-use benchmark — before-calling stage (plan/reason).",
    source=f"hf:lovesnowbest/T-Eval@{T_EVAL_REVISION}",
    categories=(Category(Level1Category.AGENT, "ToolUseSimple"),),
    tags=("chinese", "english", "open-ended"),
    license="Apache-2.0",
)
class TEvalBeforeCallingDataset(Dataset[TEvalBeforeCallingDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        lang: str = "cn",
        form: str = "json",
        legacy_model: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        suffix = "_zh" if lang == "cn" else ""
        path = os.path.join(
            name_or_path,
            "data",
            f"reason_retrieve_understand_{form}_v2{suffix}.json",
        )

        preprocessed_data = []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for v in data.values():
            origin_prompt = v.get("origin_prompt", [])
            if legacy_model:
                # for legacy models, replace `function` with `user`
                for msg in origin_prompt:
                    if msg["role"] == "function":
                        msg["role"] = "user"
            gt = v.get("ground_truth", {})

            preprocessed_data.append(
                {
                    "template": v.get("template", {}),
                    "meta_data": v.get("meta_data", {}),
                    "origin_prompt": origin_prompt,
                    # convert to str to avoid pyarrow type error
                    "ground_truth": json.dumps(gt, ensure_ascii=False),
                }
            )

        return HFDatasetDict({"test": HFDataset.from_list(preprocessed_data)})
