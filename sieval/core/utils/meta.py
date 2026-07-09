"""Helpers for building model-call and stage metadata dicts."""

import time
from collections.abc import Iterable, Mapping

from packaging.version import InvalidVersion, Version

from sieval import __version__
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
    meta: TaskStageMeta = {"timestamp": time.time(), "version": __version__}
    if timing_s is not None:
        meta["timing_s"] = timing_s
    if outputs:
        meta["model_calls"] = [build_model_call_meta(output) for output in outputs]
    if extra:
        meta["extra"] = extra
    return meta


def collect_versions(stage_metas: Iterable[Mapping[str, list]]) -> list[str]:
    """Distinct sieval versions across the given per-context stage-meta maps.

    Walks each context's ``stage_meta`` history (stage name -> list of
    per-stage meta dicts) and collects every ``version`` entry present.
    Returned sorted semver-aware; unparseable tags sort last (by string).
    A context/stage that carries no ``version`` contributes nothing.
    """
    seen: set[str] = set()
    for stage_meta in stage_metas:
        for entries in stage_meta.values():
            for entry in entries:
                v = entry.get("version")
                if v:
                    seen.add(v)

    def _key(s: str) -> tuple[int, object]:
        try:
            return (0, Version(s))
        except InvalidVersion:
            return (1, s)

    return sorted(seen, key=_key)


def report_versions(
    final_stage_metas: Iterable[Mapping[str, list]],
    failed_stage_metas: Iterable[Mapping[str, list]],
) -> list[str]:
    """Distinct producing versions for a report's terminal records.

    Aggregates versions across all terminal records (finals + fails) via
    :func:`collect_versions`. If any FINAL (scored) record carries no version,
    appends the ``"unknown"`` sentinel: a completed sample always ran the full
    pipeline, so a post-provenance FINAL is always stamped — an unstamped FINAL
    predates per-record provenance, and surfacing it keeps a legacy-blended
    report from being silently reported as single-version. FAILED records are
    not sentinel-flagged: a sample that failed before any stage legitimately
    produced no versioned work.

    "Always stamped" assumes ``stage_meta`` survives to report time. On a fresh
    run it does (built in-memory per stage); on a resume it does only under
    ``record_meta=True`` — with ``record_meta=False`` the loader has no
    persisted ``stage_meta`` to hydrate, so disk-resident finals honestly
    surface as ``"unknown"`` (their provenance was never recorded).
    """
    final_metas = list(final_stage_metas)
    versions = collect_versions([*final_metas, *failed_stage_metas])
    has_unstamped_final = any(
        not any(entry.get("version") for entries in sm.values() for entry in entries)
        for sm in final_metas
    )
    if has_unstamped_final:
        versions.append("unknown")
    return versions
