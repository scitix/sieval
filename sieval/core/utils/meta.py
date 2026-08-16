"""Helpers for building model-call and stage metadata dicts."""

import time
from collections.abc import Iterable, Mapping

from packaging.version import InvalidVersion, Version

from sieval import __version__
from sieval.core.models import ModelCallMeta, ModelOutput
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskStageMeta

#: Finish reasons that mean the generation stopped before the model chose to.
#:
#: Every spelling, not one canonical name: the IR does not normalize these, so a
#: set holding only ``length`` (OpenAI-compatible) would read zero against
#: ``max_tokens`` (Anthropic). ``content_filter`` is in because the output is cut
#: short the same way; separating causes is what the raw ``finish_reasons`` are
#: for. Shared with :func:`sieval.core.tasks.anomaly.detect_truncated_output` so
#: the rule and the report key cannot drift apart.
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "content_filter"})


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


def build_model_call_meta_from_mapping(data: Mapping) -> ModelCallMeta | None:
    """Rebuild a :class:`ModelCallMeta` from an already-flattened ``ModelOutput``.

    The judge-family tasks persist a grader's whole ``ModelOutput`` as a plain
    dict inside the judgement record (``obj_to_dict``), because a record must not
    nest a typed object. That dict is the only trace of a *second* model call in
    a stage whose return value is a record rather than a ``ModelOutput``, so the
    profiler can only see grader spend by reading it back.

    Returns ``None`` when *data* carries no ``model`` -- the one field a call
    always has -- so a mapping that merely looks dict-shaped is skipped instead
    of contributing a usage-less phantom call.
    """
    model = data.get("model")
    if not isinstance(model, Mapping):
        return None
    call: ModelCallMeta = {"model": dict(model)}  # type: ignore[typeddict-item]
    for key in ("usage", "request_params"):
        value = data.get(key)
        if isinstance(value, Mapping):
            call[key] = dict(value)  # type: ignore[literal-required]
    finish_reasons = data.get("finish_reasons")
    if finish_reasons:
        call["finish_reasons"] = list(finish_reasons)
    for key in ("response_model", "system_fingerprint"):
        value = data.get(key)
        if value is not None:
            call[key] = value
    return call


def build_stage_meta(
    *outputs: ModelOutput,
    timing_s: float | None = None,
    extra: dict | None = None,
    model_calls: Iterable[ModelCallMeta] = (),
) -> TaskStageMeta:
    """Build a TaskStageMeta dict for one pipeline stage execution.

    *model_calls* appends calls that are not represented by an *outputs* entry --
    a grader invoked inside ``feedback``, whose stage value is a record.
    """
    meta: TaskStageMeta = {"timestamp": time.time(), "version": __version__}
    if timing_s is not None:
        meta["timing_s"] = timing_s
    calls = [build_model_call_meta(output) for output in outputs]
    calls.extend(model_calls)
    if calls:
        meta["model_calls"] = calls
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


def _scored_rollout_indices(
    stage_meta: Mapping[str, list],
) -> tuple[set[int], set[int]] | None:
    """``(every rollout index, the truncated ones)`` for one scored record.

    Both counts come off this one index space, so ``n_truncated <=
    n_scored_rollouts`` holds by construction rather than by two walks agreeing.

    **INFERRED only:** a judged task's FEEDBACK stage carries the GRADER's calls,
    and a grader that hit its own budget is a fact about a different model
    (already reported as ``n_grader_unparsed``).

    **Last entry only:** a retried or re-iterated sample was scored on its last
    attempt, so an earlier truncation no longer affects any number in the report.
    The opposite choice from :func:`collect_versions`, which unions the whole
    history because provenance is about everything that ran.

    ``None`` means no INFERRED history, so nothing here is measurable. A fresh run
    cannot hit it -- stage meta is built in memory whatever ``record_meta`` says
    -- but a **resume under** ``record_meta=False`` can: nothing was persisted, so
    the loader hydrates a final without it, and reducing that to ``0`` would
    report a clean run for samples whose finish reasons were never recorded.
    :func:`report_versions` says this in band with ``"unknown"``; a count has no
    such value, hence the option.
    """
    entries = stage_meta.get(TaskStage.INFERRED.value)
    if not entries:
        return None
    scored: set[int] = set()
    truncated: set[int] = set()
    for call in entries[-1].get("model_calls") or ():
        for index, reason in enumerate(call.get("finish_reasons") or ()):
            scored.add(index)
            if reason in TRUNCATION_FINISH_REASONS:
                truncated.add(index)
    return scored, truncated


def count_truncated_rollouts(
    stage_metas: Iterable[Mapping[str, list]],
) -> int | None:
    """Rollouts whose generation was cut short, over a report's scored records.

    A truncated rollout scores as wrong without the model being wrong -- it ran
    out of budget mid-answer, and the fix is ``max_tokens``. Reported so a score
    a truncation rate explains is distinguishable from one capability explains.

    Counted per rollout, not per sample -- one truncated draw in four is a
    different fact from four (RFC #74 C) -- and deduplicated by rollout index, so
    a multi-turn task whose second turn truncated counts that rollout once. That
    is the same union :func:`sieval.core.tasks.anomaly.detect_truncated_output`
    takes over one stage's calls, so the report and the anomaly file agree by
    construction.

    Scoping and the ``None`` case are :func:`_scored_rollout_indices`; ``None``
    propagates, since a count skipping an unmeasurable record reads low with
    nothing to say so.
    """
    total = 0
    for stage_meta in stage_metas:
        indices = _scored_rollout_indices(stage_meta)
        if indices is None:
            return None
        total += len(indices[1])
    return total


def count_scored_rollouts(stage_metas: Iterable[Mapping[str, list]]) -> int | None:
    """Rollouts a report's scored records actually drew -- ``n_truncated``'s base.

    Without it the numerator is unreadable: ``26`` is a different fact at 600
    rollouts than at 30, and the rule lanes publish rates plus ``fails`` and no
    sample total, so nothing in ``report.json`` was divisible into it.

    The *observed* draw, not ``n * len(finals)``: a short sample drew fewer
    rollouts than its budget asked for, and the share a reader wants is over what
    ran. Not the rate itself -- what threshold should warn, fail or annotate a
    score is a policy call left to the reader.
    """
    total = 0
    for stage_meta in stage_metas:
        indices = _scored_rollout_indices(stage_meta)
        if indices is None:
            return None
        total += len(indices[0])
    return total
