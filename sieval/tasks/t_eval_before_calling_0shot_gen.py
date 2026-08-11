import json
import os
from typing import Literal, override

import anyio
import numpy as np
from openai import AsyncOpenAI

from sieval.community.t_eval import EMB_PLACEHOLDER, ResponseDataSample, format_load
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import DENOMINATOR_FIELD, DENOMINATOR_JUDGED
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
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="open-compass/T-Eval",
        url="https://github.com/open-compass/T-Eval/tree/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval",
        notes="ResponseDataSample (schema.py) + format_load (utils/format_load.py) vendored.",  # noqa: E501
    ),
)
class TEvalBeforeCallingZeroShotGenTask(
    Task[
        TEvalBeforeCallingDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `denominator_policy`, which names a
        # population rather than measuring one.
        dict[str, float | str],
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
            # Check the key here rather than letting the client do it. Passing an
            # explicit "" is not None, so it skips the OpenAI client's own
            # OPENAI_API_KEY fallback and raises `Missing credentials` naming
            # OPENAI_API_KEY -- a variable that would not help, since the endpoint
            # is an embedding service of ours, not OpenAI's.
            api_key = os.getenv("SIEVAL_EMBED_API_KEY")
            if not api_key:
                raise ValueError(
                    "eval_thought=True scores the thought axis by embedding "
                    "similarity, which needs a credential for the embedding "
                    "endpoint: set SIEVAL_EMBED_API_KEY (and SIEVAL_EMBED_API if "
                    "the endpoint is not the default). Leave eval_thought unset "
                    "to skip that axis -- the other axes need no embedding call."
                )
            self._bert_api_client = AsyncOpenAI(
                base_url=os.getenv(
                    "SIEVAL_EMBED_API", "https://console.siflow.cn/model-api"
                ),
                api_key=api_key,
            )
        else:
            self._bert_api_client = None

        self._bert_score_model = bert_score_model
        self._default_prompt_type = default_prompt_type
        self._eval_type = eval_type

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            raw["origin_prompt"],
            # The gold is the expected tool call, stored as a JSON string on the
            # sample; parsed at judgement time by the same path that grades it.
            reference=raw["ground_truth"],
            extra={"template": raw["template"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        # n=1. The raw response IS the answer here -- parsing into a tool call is
        # the evaluator's job, in feedback. A blank response normalizes to None so
        # `extracted` stays a real signal.
        text = inf.texts[0]
        return build_prediction_record([text if text.strip() else None])

    @override
    async def feedback(self, post, ctx):
        """Grade one response on every axis the T-Eval evaluator measures.

        T-Eval scores a tool call on several CO-EQUAL continuous axes (thought
        similarity, tool-name match, argument precision/recall/F1, parse rate) --
        there is no single published headline, and report() macro-averages each
        axis independently. So every axis THIS CONFIGURATION SCORES goes in
        `metrics`, by name, where a generic reader can enumerate them. Which axes
        those are comes from `_metric_keys()`; the axes it excludes are absent
        rather than zero, because `_evaluate` pre-seeds every axis to 0 and a 0 on
        an axis nobody measured is a hole, not a measurement.

        `correct` still has to be one bool. It is defined as the strict reading --
        every axis the evaluator scored came out at 1.0, i.e. the model produced
        exactly the right call -- and it is DERIVED from `metrics` rather than
        computed separately, so the two cannot disagree. That derivation is why
        `metrics` has to exclude the unscored axes: with `eval_thought=False` (the
        default) a retained `thought: 0.0` would pin `correct` to False on every
        sample of every run, making the one axis that is comparable across tasks
        structurally unreachable for this one. It is deliberately not
        `parse_rate`: a task whose `correct` meant "the output parsed" would look
        near-perfect next to every other task on the one axis that is supposed to
        be comparable across them.

        `parse_error` is a count of unparseable segments, not a measurement of the
        answer, so it stays in `extra`.
        """
        prediction = post["rollouts"][0].get("prediction") or ""
        resp_data_sample, error = self._process_response(
            {
                "template": ctx.raw_sample["template"],
                "prediction": prediction,
                "ground_truth": json.loads(ctx.raw_sample["ground_truth"]),
                "meta_data": ctx.raw_sample["meta_data"],
            }
        )
        metrics_result = await self._evaluate(resp_data_sample)
        metrics: dict[str, bool | float] = {
            key: float(metrics_result[key]) for key in self._metric_keys()
        }
        correct = bool(metrics) and all(value == 1.0 for value in metrics.values())
        return True, build_judgement_record(
            ctx.raw_sample["ground_truth"],
            [build_rollout_judgement(0, correct, metrics=metrics)],
            metrics=metrics,
            extra={"parse_error": error},
        )

    @override
    async def report(self, finals, fails):
        # _post_process macro-averages each named axis, so it is fed the metric
        # mapping each judgement recorded -- the same numbers, now enumerable on
        # disk instead of flattened into an untyped per-task feedback dict.
        results_list = [dict(ctx.feedback_result["metrics"]) for ctx in finals]
        # No `score_key`: this report has no `score` to name. T-Eval publishes one
        # rate per axis and no single headline, and picking one here would invent a
        # ranking upstream does not make -- so the key is omitted rather than
        # pointed at an arbitrary axis. `denominator_policy` still applies: every
        # axis is macro-averaged over the judged samples, with pipeline failures
        # reported in `fails` rather than averaged in as zeros.
        return {
            **self._post_process(results_list),
            "fails": len(fails),
            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,
        }

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

    def _metric_keys(self) -> list[str]:
        """The axes this configuration actually scores.

        Single source of truth, shared by `feedback` and `_post_process` so the
        recorded `metrics`, the derived `correct` and the macro-average cannot
        disagree about which axes are real. `_evaluate` returns all six axes
        regardless, pre-seeded to 0, to keep its mapping rectangular.
        """
        if self._default_prompt_type == "json":
            keys = [
                "thought",
                "name",
                "args_precision",
                "args_recall",
                "args_f1_score",
                "parse_rate",
            ]
        elif self._default_prompt_type == "str":
            # In str mode only the axis matching `eval_type` is populated -- the
            # response is one bare field, not a parsed call.
            keys = {
                "reason": ["thought", "parse_rate"],
                "retrieve": ["name", "parse_rate"],
                "understand": [
                    "args_precision",
                    "args_recall",
                    "args_f1_score",
                    "parse_rate",
                ],
            }[self._eval_type]
        else:
            raise NotImplementedError(
                "Currently, we only support json and str format, but get "
                f"{self._default_prompt_type}"
            )

        # Thought similarity costs an embedding call, so it is opt-in; when it is
        # off the axis was not measured at all.
        if not self._eval_thought and "thought" in keys:
            keys.remove("thought")
        return keys

    def _post_process(self, results_list: list[dict]) -> dict[str, float]:
        # list of dict to dict of list
        results = {}
        for key in self._metric_keys():
            results[key] = np.mean([result[key] for result in results_list]) * 100

        # The *_parsed variants are reported in every mode, including the str modes
        # that never score args at all -- so read defensively rather than assuming
        # these axes are among the ones recorded.
        success_samples = [r for r in results_list if r.get("parse_rate", 0) == 1]
        for key in ("args_precision", "args_recall", "args_f1_score"):
            results[f"{key}_parsed"] = (
                np.mean([r.get(key, 0.0) for r in success_samples]) * 100
            )
        return results
