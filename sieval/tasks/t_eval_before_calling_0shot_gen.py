import json
import os
from typing import Literal, override

import anyio
import numpy as np
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.t_eval import EMB_PLACEHOLDER, ResponseDataSample, format_load
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import TEvalBeforeCallingDatasetSample


@sieval_task(
    name="t_eval_before_calling_0shot_gen",
    display_name="T-Eval Before-Calling (0-shot)",
    description="T-Eval tool-use benchmark — before-calling stage (plan/reason).",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("chinese", "english", "open-ended"),
    deps_group="t-eval",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="open-compass/T-Eval",
        url="https://github.com/open-compass/T-Eval/tree/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval",
        notes="ResponseDataSample (schema.py) + format_load (utils/format_load.py) vendored.",  # noqa: E501
    ),
)
class TEvalBeforeCallingZeroShotGenTask(
    Task[
        TEvalBeforeCallingDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        dict[str, float],
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name=None,
        bert_score_model: str = "simaas-qwen3-embedding-0-6b-v1",
        default_prompt_type: str = "json",
        eval_type: Literal[
            "reason", "retrieve", "understand"
        ] = "reason",  # not used in json mode
        eval_thought: bool = False,
    ):
        super().__init__(dataset, model, name)
        self._eval_thought = eval_thought

        if self._eval_thought:
            self._bert_api_client = AsyncOpenAI(
                base_url=os.getenv(
                    "SIEVAL_EMBED_API", "https://console.siflow.cn/model-api"
                ),
                api_key=os.getenv("SIEVAL_EMBED_API_KEY", ""),
            )
        else:
            self._bert_api_client = None

        self._bert_score_model = bert_score_model
        self._default_prompt_type = default_prompt_type
        self._eval_type = eval_type

    @override
    async def preprocess(self, raw, ctx):
        return raw["origin_prompt"]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        return inf.texts[0]  # n=1, only one choice, and pass directly

    @override
    async def feedback(self, post, ctx):
        resp_data_sample, error = self._process_response(
            {
                "template": ctx.raw_sample["template"],
                "prediction": post,
                "ground_truth": json.loads(ctx.raw_sample["ground_truth"]),
                "meta_data": ctx.raw_sample["meta_data"],
            }
        )
        metrics_result = await self._evaluate(resp_data_sample)
        return True, {**metrics_result, "parse_error": error}

    @override
    async def report(self, finals, fails):
        results_list = [ctx.feedback_result for ctx in finals]
        return {**self._post_process(results_list), "fails": len(fails)}

    def _format_load(self, data) -> dict:
        try:
            json_format = format_load(data, start_character="{", end_character="}")
        except Exception:
            return {}
        if not isinstance(json_format, dict):
            return {}
        prepared_json_format = {}
        try:
            prepared_json_format["thought"] = str(json_format["thought"])
        except Exception:
            prepared_json_format["thought"] = ""
        try:
            prepared_json_format["name"] = str(json_format["name"])
        except Exception:
            prepared_json_format["name"] = ""

        if self._default_prompt_type == "json":
            try:
                if isinstance(json_format["args"], dict):
                    prepared_json_format["args"] = json_format["args"]
                else:
                    prepared_json_format["args"] = {}
            except Exception:
                prepared_json_format["args"] = {}
        else:
            try:
                prepared_json_format["args"] = str(json_format["args"])
            except Exception:
                prepared_json_format["args"] = ""

        return prepared_json_format

    def _process_response(self, datum: dict) -> tuple[ResponseDataSample, int]:
        # Generated response, which can be a string or list
        pred_data = datum["prediction"]
        # Response of ground truth, which can be a string or list
        gt_data = datum["ground_truth"]
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta" in datum:
            prompt_type = datum["meta"].get(
                "response_format", self._default_prompt_type
            )
        else:
            prompt_type = self._default_prompt_type

        error = 0
        gt = self._format_load(gt_data)
        if prompt_type == "json":
            pred = self._format_load(pred_data)
            if pred == {} or gt == {}:
                error = 1
        elif prompt_type == "str":
            # choose the first line
            pred = {}
            if self._eval_type == "reason":
                pred["thought"] = pred_data
            if self._eval_type == "retrieve":
                pred["name"] = pred_data
            if self._eval_type == "understand":
                pred["args"] = pred_data
        else:
            raise NotImplementedError(
                f"Currently, we only support json and str format, but get {prompt_type}"
            )

        if error == 1:
            pred = {}
        return ResponseDataSample(template="", pred=pred, gt=gt), error

    async def _evaluate(self, data_sample: ResponseDataSample) -> dict[str, float]:
        """Evaluate the response data sample."""
        metrics_result = {
            "thought": 0,
            "name": 0,
            "args_precision": 0,
            "args_recall": 0,
            "args_f1_score": 0,
            "parse_rate": 0,
        }
        if (
            self._eval_thought
            and "thought" in data_sample.pred
            and "thought" in data_sample.gt
        ):
            # Lazy import: sentence_transformers pulls torch; only needed here.
            from sentence_transformers import util

            pred_thought = data_sample.pred["thought"] or EMB_PLACEHOLDER
            gt_thought = data_sample.gt["thought"] or EMB_PLACEHOLDER

            assert self._bert_api_client is not None
            resp = await self._bert_api_client.embeddings.create(
                input=[pred_thought, gt_thought], model=self._bert_score_model
            )
            await anyio.sleep(0.1)  # to avoid being rate limited
            all_embeddings = [emb.embedding for emb in resp.data]
            pred_emb, gt_emb = all_embeddings

            # ensure dtype is float64
            # keep compatible with isinstance float check in OpenCompass
            pred_emb = np.array(pred_emb, dtype=np.float64)
            gt_emb = np.array(gt_emb, dtype=np.float64)
            cosine_scores = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
            metrics_result["thought"] = cosine_scores[0, 0]

        if "name" in data_sample.pred and "name" in data_sample.gt:
            if data_sample.pred["name"] == data_sample.gt["name"]:
                metrics_result["name"] = 1
            else:
                metrics_result["name"] = 0
        if "args" in data_sample.pred and "args" in data_sample.gt:
            gt_num_keys = len(data_sample.gt["args"].keys())
            pred_num_keys = len(data_sample.pred["args"].keys())
            if pred_num_keys == 0 and gt_num_keys == 0:
                metrics_result["args_precision"] = 1
                metrics_result["args_recall"] = 1
                metrics_result["args_f1_score"] = 1
            elif pred_num_keys == 0 or gt_num_keys == 0:
                metrics_result["args_precision"] = 0
                metrics_result["args_recall"] = 0
                metrics_result["args_f1_score"] = 0
            else:
                correct_count = 0
                for key in data_sample.gt["args"]:
                    if key in data_sample.pred["args"] and str(
                        data_sample.pred["args"][key]
                    ) == str(data_sample.gt["args"][key]):
                        correct_count += 1
                metrics_result["args_precision"] = correct_count / pred_num_keys
                metrics_result["args_recall"] = correct_count / gt_num_keys
                if (
                    metrics_result["args_precision"] + metrics_result["args_recall"]
                    == 0
                ):
                    metrics_result["args_f1_score"] = 0
                else:
                    metrics_result["args_f1_score"] = (
                        2
                        * metrics_result["args_precision"]
                        * metrics_result["args_recall"]
                        / (
                            metrics_result["args_precision"]
                            + metrics_result["args_recall"]
                        )
                    )

        if len(data_sample.pred.keys()) == 0:
            metrics_result["parse_rate"] = 0
        else:
            metrics_result["parse_rate"] = 1
        return metrics_result

    def _post_process(self, results_list: list[dict]) -> dict[str, float]:
        # list of dict to dict of list
        results = {}
        if self._default_prompt_type == "json":
            metric_keys = [
                "thought",
                "name",
                "args_precision",
                "args_recall",
                "args_f1_score",
                "parse_rate",
            ]
        if self._default_prompt_type == "str":
            if self._eval_type == "reason":
                metric_keys = ["thought", "parse_rate"]
            if self._eval_type == "retrieve":
                metric_keys = ["name", "parse_rate"]
            if self._eval_type == "understand":
                metric_keys = [
                    "args_precision",
                    "args_recall",
                    "args_f1_score",
                    "parse_rate",
                ]

        # Remove 'thought' from metrics if evaluation is disabled
        if not self._eval_thought and "thought" in metric_keys:
            metric_keys.remove("thought")

        for key in metric_keys:
            results[key] = np.mean([result[key] for result in results_list]) * 100

        success_samples = [r for r in results_list if r.get("parse_rate", 0) == 1]
        results["args_precision_parsed"] = (
            np.mean([r["args_precision"] for r in success_samples]) * 100
        )
        results["args_recall_parsed"] = (
            np.mean([r["args_recall"] for r in success_samples]) * 100
        )
        results["args_f1_score_parsed"] = (
            np.mean([r["args_f1_score"] for r in success_samples]) * 100
        )
        return results
