"""Helpers for building model-call and stage metadata dicts."""

import time

from sieval.core.models import ModelCallMeta, ModelOutput
from sieval.core.tasks.context import TaskStageMeta


def build_model_call_meta(output: ModelOutput) -> ModelCallMeta:
    """Extract a ModelCallMeta dict from a ModelOutput."""
    model_call: ModelCallMeta = {"model": output.model}
    if output.usage:
        model_call["usage"] = output.usage
    if output.request_params is not None:
        model_call["request_params"] = dict(output.request_params)
    if output.finish_reasons:
        model_call["finish_reasons"] = output.finish_reasons
    if output.response_model is not None:
        model_call["response_model"] = output.response_model
    if output.system_fingerprint is not None:
        model_call["system_fingerprint"] = output.system_fingerprint
    return model_call


def build_stage_meta(
    *outputs: ModelOutput,
    timing_s: float | None = None,
    extra: dict | None = None,
) -> TaskStageMeta:
    """Build a TaskStageMeta dict for one pipeline stage execution."""
    meta: TaskStageMeta = {"timestamp": time.time()}
    if timing_s is not None:
        meta["timing_s"] = timing_s
    if outputs:
        meta["model_calls"] = [build_model_call_meta(output) for output in outputs]
    if extra:
        meta["extra"] = extra
    return meta
