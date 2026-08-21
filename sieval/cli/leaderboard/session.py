"""
YAML-driven batch evaluation session.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import shlex
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast

import anyio
import yaml
from anyio.to_thread import run_sync
from loguru import logger
from packaging.version import InvalidVersion, Version

from sieval import __version__
from sieval.cli._filter_spec import (
    VALUES_DIGEST_KEY,
    check_arg_names,
    check_by,
    check_by_digest,
    check_values_source,
    compute_values_digest,
    key_function_spec,
    pin_filter_digests,
    relative_values_files,
    resolve_values_path,
)
from sieval.cli.leaderboard.card import AlignmentCard, load_card
from sieval.cli.resolution import (
    TASK_MODEL_ROLES,
    binding_resource_argument_paths,
    derive_model_type,
    is_task_model_role_sentinel,
    normalize_inline_model_binding,
    resolve_dataset_class,
    resolve_key_function,
    resolve_task_class,
    validate_model_type_dialect,
    validate_named_config_map,
    validate_task_config_args,
    validate_task_model_requirements,
)
from sieval.core.datasets import Dataset, TFilterKey
from sieval.core.models import ChatModel, GenModel, Model, SglangGenModel
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    CapabilityIntent,
    CapabilityKey,
    ModelCapabilityEntry,
    ModelCapabilityProfile,
    ModelCapabilityStatus,
    aggregate_capability_intents,
    legacy_capability_intents,
    validate_no_legacy_capability_ambiguity,
)
from sieval.core.models.connection_factory import (
    CONNECTION_FACTORY_REGISTRY,
    ConnectionRequest,
)
from sieval.core.models.deployment import (
    ConnectionPool,
    Deployment,
    DeploymentPlanProjection,
    Engine,
    RouteIntent,
    ServingFacts,
)
from sieval.core.models.dialect_registry import (
    RequestSeedSupport,
    dialect_is_bindable,
    get_dialect_spec,
)
from sieval.core.models.reconcile import (
    BindingReconcileInput,
    CannotVerify,
    CheckStage,
    Configured,
    ConnectionScope,
    DeferredCheck,
    DeploymentReconcileInput,
    ReconcileBatch,
    ReconcileDiagnostic,
    ReconcileResult,
    RuntimeBindingPlan,
    ServingOutcome,
    ServingReconciler,
    ServingRequirement,
    reconcile,
)
from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    ExternalModelBinding,
    InlineModelBinding,
    InputKind,
    NamedModelBinding,
    NormalizedModelBinding,
    RequirementContext,
    TaskModelRequirement,
    aggregate_task_requirements,
)
from sieval.core.runners import (
    MultiTaskRunner,
    ResumeAction,
    TaskRunnerConfig,
    read_run_version,
    resume_version_verdict,
)
from sieval.core.tasks.context import TaskAction
from sieval.core.types import JSONValue
from sieval.infer import deployment_plan_projection
from sieval.infer.params import merge_params
from sieval.infer.topology.models import DETERMINISTIC_DEFAULT_SEED

# ── Narrow scalar types for YAML configuration ──
# Mirrors sieval.infer.config.ParamValue but defined locally to keep core/
# free of infer imports.
_ParamValue = str | int | float | bool


_PR1_REQUEST_VERIFIERS = {
    "input_scoring": "validate_response_channel",
    "sampled_logprobs": "validate_response_channel",
    "top_logprobs": "validate_response_channel",
}

_DETERMINISTIC_SEED_CONTRACT_KEY = "_sieval_deterministic_seed_contract"


class _RequestSeedScope(StrEnum):
    """Where deterministic sampling is controlled for one binding."""

    PER_REQUEST = "per_request"
    ENGINE_LEVEL_ONLY = "engine_level_only"


class _RequestSeedProvenance(StrEnum):
    """Frozen source of one model default for ``sampling.seed``."""

    NONE = "none"
    AUTOMATIC = "automatic"
    BINDING_CONFIG = "binding_config"
    EXTERNAL_MODEL = "external_model"
    TASK_INFER_ARGS = "task_infer_args"


@dataclasses.dataclass(frozen=True)
class _RequestSeedDecision:
    """One frozen deterministic request-seed policy decision."""

    dialect_id: str
    support: RequestSeedSupport
    # Evidence-only: records where deterministic sampling is controlled
    # (currently derivable from dialect_id). Execution consumes the frozen
    # final seed state, not this field.
    scope: _RequestSeedScope
    seed_present: bool
    seed: int | None
    provenance: _RequestSeedProvenance

    @property
    def explicit_seed_present(self) -> bool:
        return self.provenance in {
            _RequestSeedProvenance.BINDING_CONFIG,
            _RequestSeedProvenance.EXTERNAL_MODEL,
            _RequestSeedProvenance.TASK_INFER_ARGS,
        }


def _resolve_deterministic_request_seed(
    *,
    dialect_id: str,
    explicit_seed_present: bool,
    explicit_seed: int | None = None,
    explicit_provenance: _RequestSeedProvenance = (
        _RequestSeedProvenance.BINDING_CONFIG
    ),
) -> _RequestSeedDecision:
    """Resolve one immutable decision for execution and persistence."""

    if dialect_id == "sglang_legacy":
        # TODO(PR-5): delete this compatibility policy when the legacy SGLang
        # path becomes a registered ``sglang_native`` DialectSpec/binder.
        support = RequestSeedSupport.UNSUPPORTED
        scope = _RequestSeedScope.ENGINE_LEVEL_ONLY
    else:
        support = get_dialect_spec(dialect_id).request_seed_support
        scope = _RequestSeedScope.PER_REQUEST

    if support is RequestSeedSupport.RESERVED:
        raise ValueError(
            f"dialect {dialect_id!r} has not declared whether it supports "
            "per-request deterministic seeds"
        )
    if explicit_seed_present:
        return _RequestSeedDecision(
            dialect_id=dialect_id,
            support=support,
            scope=scope,
            seed_present=True,
            seed=explicit_seed,
            provenance=explicit_provenance,
        )
    if support is RequestSeedSupport.SUPPORTED:
        return _RequestSeedDecision(
            dialect_id=dialect_id,
            support=support,
            scope=scope,
            seed_present=True,
            seed=DETERMINISTIC_DEFAULT_SEED,
            provenance=_RequestSeedProvenance.AUTOMATIC,
        )
    return _RequestSeedDecision(
        dialect_id=dialect_id,
        support=support,
        scope=scope,
        seed_present=False,
        seed=None,
        provenance=_RequestSeedProvenance.NONE,
    )


def _validated_request_seed(value: object, subject: str) -> int | None:
    """Return one typed request seed or reject an invalid explicit value.

    ``subject`` names the config site being read so a rejected value is
    locatable in a config that declares many models and tasks.
    """

    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError(f"{subject} must be an integer")
    return cast(int | None, value)


def _with_task_infer_seed(
    decision: _RequestSeedDecision,
    infer_args: Mapping[str, JSONValue],
    subject: str,
) -> _RequestSeedDecision:
    """Overlay a task candidate's frozen ``infer_args.seed`` when present."""

    if "seed" not in infer_args:
        return decision
    seed = _validated_request_seed(infer_args["seed"], subject)
    return dataclasses.replace(
        decision,
        seed_present=True,
        seed=seed,
        provenance=_RequestSeedProvenance.TASK_INFER_ARGS,
    )


def _apply_request_seed_decision_to_args(
    args: dict[str, Any], decision: _RequestSeedDecision
) -> None:
    """Make an argument mapping match a previously frozen seed decision."""

    args.pop("seed", None)
    if decision.seed_present:
        args["seed"] = decision.seed


def _apply_request_seed_decision_to_model(
    model: Model, decision: _RequestSeedDecision
) -> Model:
    """Make a bound model's defaults match a previously frozen decision."""

    if model.dialect_id != decision.dialect_id:
        raise RuntimeError(
            "frozen request-seed decision targets dialect "
            f"{decision.dialect_id!r}, but the bound model uses "
            f"{model.dialect_id!r}"
        )
    model = model.without_args("seed")
    if decision.seed_present:
        model = model.with_args(seed=decision.seed)
    return model


def _model_value_paths(value: object) -> tuple[str, ...]:
    """Return paths to Models hidden below an ordinary task-argument surface."""

    found: list[str] = []
    pending: list[tuple[str, object]] = [("", value)]
    visited: set[int] = set()
    while pending:
        path, current = pending.pop()
        if isinstance(current, Model):
            found.append(path or "<root>")
            continue
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(
                (
                    f"{path}.{key}" if path else str(key),
                    nested,
                )
                for key, nested in current.items()
            )
            continue
        if isinstance(current, list | tuple):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(
                (f"{path}[{index}]", nested) for index, nested in enumerate(current)
            )
    return tuple(sorted(found))


class _PR1CompatibilityServingReconciler:
    """Keep existing scoring paths behind their named response guards.

    PR #1 does not claim model support from a dialect descriptor.  In the
    absence of an authoritative #27/#59 profile, the three scoring capabilities
    that already have response validators remain UNKNOWN and are admitted only
    with an immutable per-request check.  #47 replaces this compatibility row
    with engine/fact-specific setup outcomes.
    """

    def __init__(
        self,
        external_runtime_plans: Callable[[str], tuple[RuntimeBindingPlan, ...]],
    ) -> None:
        self._external_runtime_plans = external_runtime_plans

    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        outcomes: dict[CapabilityKey, ServingOutcome] = {}
        for requirement in requirements:
            if requirement.verifier == "external_runtime_plan":
                baselines = self._external_runtime_plans(deployment.root_deployment_key)
                if not baselines:
                    outcomes[requirement.capability] = CannotVerify(
                        CheckStage.SETUP,
                        "external_runtime_plan",
                        "the external model's RuntimeBindingPlan is unavailable",
                    )
                    continue
                request_checks = _dedupe_deferred_checks(
                    *(
                        tuple(
                            check
                            for check in baseline.request_checks
                            if check.capability == requirement.capability
                        )
                        for baseline in baselines
                    )
                )
                verifier = _PR1_REQUEST_VERIFIERS.get(requirement.capability)
                if not request_checks and verifier is not None:
                    request_checks = (
                        DeferredCheck(
                            requirement.capability,
                            CheckStage.REQUEST,
                            verifier,
                            "external runtime support remains guarded by the "
                            "existing response contract",
                        ),
                    )
                evidence: dict[str, JSONValue] = {
                    "source": "external_runtime_plans",
                    "plan_fingerprints": cast(
                        JSONValue,
                        sorted({baseline.fingerprint for baseline in baselines}),
                    ),
                }
                outcomes[requirement.capability] = Configured(
                    evidence=evidence,
                    request_checks=request_checks,
                )
                continue
            verifier = _PR1_REQUEST_VERIFIERS.get(requirement.capability)
            if verifier is None or requirement.verifier not in (None, verifier):
                continue
            outcomes[requirement.capability] = CannotVerify(
                CheckStage.REQUEST,
                verifier,
                requirement.reason
                or "PR-1 compatibility requires response-time verification",
            )
        return outcomes


def _dedupe_deferred_checks(
    *groups: tuple[DeferredCheck, ...],
) -> tuple[DeferredCheck, ...]:
    """Return stable, exact de-duplication for serialized safety checks."""

    result: list[DeferredCheck] = []
    seen: set[DeferredCheck] = set()
    for group in groups:
        for check in group:
            if check in seen:
                continue
            seen.add(check)
            result.append(check)
    return tuple(result)


class _PR1CompositeServingReconciler:
    """Preserve external baselines while allowing a stronger injected proof."""

    def __init__(
        self,
        compatibility: _PR1CompatibilityServingReconciler,
        injected: ServingReconciler,
    ) -> None:
        self._compatibility = compatibility
        self._injected = injected

    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        baseline_outcomes = self._compatibility.reconcile(requirements, deployment)
        injected_outcomes = dict(self._injected.reconcile(requirements, deployment))
        external_keys = {
            requirement.capability
            for requirement in requirements
            if requirement.verifier == "external_runtime_plan"
        }
        for capability in external_keys:
            baseline = baseline_outcomes.get(capability)
            injected = injected_outcomes.get(capability)
            if not isinstance(baseline, Configured):
                # A custom reconciler must not hide a missing/stale external plan.
                if baseline is not None:
                    injected_outcomes[capability] = baseline
                continue
            if injected is None:
                injected_outcomes[capability] = baseline
                continue
            if isinstance(injected, Configured):
                evidence: dict[str, JSONValue] = dict(baseline.evidence)
                if injected.evidence:
                    evidence["injected_reconciler"] = dict(injected.evidence)
                injected_outcomes[capability] = Configured(
                    launch_patch=injected.launch_patch,
                    request_checks=_dedupe_deferred_checks(
                        baseline.request_checks,
                        injected.request_checks,
                    ),
                    evidence=evidence,
                )
                continue
            if (
                isinstance(injected, CannotVerify)
                and injected.stage is CheckStage.REQUEST
            ):
                evidence = dict(baseline.evidence)
                evidence["injected_reconciler"] = {
                    "outcome": "cannot_verify",
                    "verifier": injected.verifier,
                    "reason": injected.reason,
                }
                injected_outcomes[capability] = Configured(
                    request_checks=_dedupe_deferred_checks(
                        baseline.request_checks,
                        (
                            DeferredCheck(
                                capability,
                                CheckStage.REQUEST,
                                injected.verifier,
                                injected.reason,
                            ),
                        ),
                    ),
                    evidence=evidence,
                )
            # SETUP uncertainty and explicit unsupported outcomes remain stricter
            # than the baseline and are therefore allowed to stop reconciliation.
        return injected_outcomes


# Type Definitions for YAML Configuration
class _InferDict(TypedDict, total=False):
    """YAML-level infer configuration for a model (inline in core/)."""

    backend: str
    recipe: str
    checkpoint: str
    overrides: dict[str, _ParamValue]


class _InferMetaDict(TypedDict, total=False):
    """User-declared inference environment metadata for audit (inline in core/)."""

    framework: str
    dtype: str
    tp: int
    gpu: str


class ModelConfigDict(TypedDict, total=False):
    name: str  # For base models
    type: Literal["chat", "gen"]  # "chat" or "gen" (default: "chat")
    engine: str  # Optional remote/managed engine identity assertion
    dialect: str  # Canonical provider wire dialect (RFC #25)
    service_role: str  # Explicit route when a deployment has multiple endpoints
    capabilities: dict[str, object]  # Typed/normalized by cli.validation
    base: str  # For derived models
    args: dict[str, Any]
    api_key: str
    api_base: str
    max_retries: int
    concurrency_limit: int
    infer: _InferDict  # infer config for `sieval infer`
    infer_meta: _InferMetaDict  # infer metadata for audit


# Use functional syntax to support "class" key which is a Python keyword
DatasetConfigDict = TypedDict(
    "DatasetConfigDict",
    {
        "class": str,
        "path": str,
        "args": dict[str, Any],
        "operations": list[dict[str, dict[str, Any]]],
    },
    total=False,
)


TaskConfigDict = TypedDict(
    "TaskConfigDict",
    {
        "class": str,
        "dataset": str | DatasetConfigDict,
        "model": str,
        "args": dict[str, Any],
        "infer_args": dict[str, JSONValue],  # per-task overrides (scalar + structured)
        "runner_config": dict[str, Any],
    },
    total=False,
)


class AlignmentBlockDict(TypedDict):
    card: str


class RootConfigDict(TypedDict, total=False):
    _sieval_deterministic_seed_contract: dict[str, JSONValue]
    deterministic: bool
    result_dir: str
    concurrency_limit: int
    concurrency_limits: dict[
        TaskAction | Literal["preprocess", "infer", "postprocess", "feedback"], int
    ]
    runner_config: dict[str, Any]
    models: dict[str, ModelConfigDict]
    datasets: dict[str, DatasetConfigDict]
    tasks: dict[str, TaskConfigDict]
    alignment: AlignmentBlockDict


def _split_header(text: str) -> tuple[str, str]:
    """Partition ``text`` into ``(header, body)`` at the comment header block
    written by ``_format_comment_header``.

    Anchored to the ``# ---...`` border pair so only OUR header is split off,
    not arbitrary user-added top-of-file comments. A leading border with no
    closing border is treated as malformed and yields ``("", text)`` so body
    comparison detects the tampering instead of silently succeeding. When no
    header is present, returns ``("", text)``. In all cases ``header + body``
    reconstructs ``text`` exactly.
    """
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("# -"):
        return "", text
    for i in range(1, len(lines)):
        if lines[i].startswith("# -"):
            end = i + 1
            if end < len(lines) and lines[end].strip() == "":
                end += 1
            return "".join(lines[:end]), "".join(lines[end:])
    return "", text


def _strip_header(text: str) -> str:
    """Return ``text`` with the ``_format_comment_header`` block removed.

    Thin wrapper over :func:`_split_header`; see it for border/malformed
    semantics.
    """
    return _split_header(text)[1]


def _diff_lines(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return ``- <path>: <old> → <new>`` lines for every differing leaf.

    Walks two parsed-config mappings depth-first. Empty list means the two
    parse to the same structure (any textual difference was whitespace /
    formatting only).
    """
    diffs: list[str] = []

    def _walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                _walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"- {path}: list length {len(x)} → {len(y)}")
            else:
                for i, (xv, yv) in enumerate(zip(x, y, strict=True)):
                    _walk(xv, yv, f"{path}[{i}]")
        elif x != y:
            diffs.append(f"- {path}: {x!r} → {y!r}")

    _walk(a, b, "")
    return diffs


def _describe_order_change(before: list[Any], after: list[Any]) -> str:
    """Name the first key that moved. Both sides hold the same keys, so printing
    the two sequences in full makes a one-key swap in a 12-key ``engine_params``
    block a 500-character line the reader has to diff by eye.
    """
    moved = [
        (i, now)
        for i, (was, now) in enumerate(zip(before, after, strict=True))
        if was != now
    ]
    if not moved:
        return f"identical order ({len(before)} keys)"
    i, key = moved[0]
    return (
        f"{key!r} moved from position {before.index(key)} to {i} ({len(before)} keys)"
    )


def _diff_key_shape(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return ``(presence, order)`` key differences a parsed leaf diff cannot see.

    Only meaningful once :func:`_diff_lines` is empty, and two different things
    hide there. **presence**: leaves are read with ``.get()``, so a
    ``None``-valued key is indistinguishable from an absent one — dropping a
    ``foo: null`` leaves no leaf diff, and every ``infer_plans.yaml`` carries a
    ``scaling: null``. **order**: the same keys resequenced, which a
    byte-for-byte comparison rejects. Plain dicts from ``yaml.safe_load``
    preserve document order, so both are recoverable.
    """
    presence: list[str] = []
    order: list[str] = []

    def _walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            label = path or "(root)"
            removed = [k for k in x if k not in y]
            added = [k for k in y if k not in x]
            if removed or added:
                changes = []
                if removed:
                    changes.append(f"removed {removed}")
                if added:
                    changes.append(f"added {added}")
                presence.append(f"- {label}: {', '.join(changes)}")
            elif list(x) != list(y):
                order.append(f"- {label}: {_describe_order_change(list(x), list(y))}")
            for k in x:
                if k in y:
                    _walk(x[k], y[k], f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (xv, yv) in enumerate(zip(x, y, strict=True)):
                _walk(xv, yv, f"{path}[{i}]")

    _walk(a, b, "")
    return presence, order


def _diff_dicts(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Return a short human-readable hint describing which keys differ.

    Up to 10 leaf paths from :func:`_diff_lines`; failing that,
    :func:`_diff_key_shape`, since a byte comparison aborts on key set and order
    too and "nothing differs" is not actionable. Only once the key shape matches
    as well is the difference genuinely layout.
    """
    lines = _diff_lines(a, b)
    if lines:
        return "Diff:\n" + "\n".join(f"  {line}" for line in lines[:10])

    presence, order = _diff_key_shape(a, b)
    blocks: list[str] = []
    if presence:
        blocks.append(
            "keys added or removed — a null-valued key and a missing key hold "
            "the same value, so this shows up as no value difference at all:\n"
            + "\n".join(f"  {line}" for line in presence[:10])
        )
    if order:
        blocks.append(
            "same keys, different order (the comparison is byte-for-byte, so "
            "order counts):\n" + "\n".join(f"  {line}" for line in order[:10])
        )
    if blocks:
        return "Diff: " + "\nDiff: ".join(blocks)
    return "Diff: (whitespace / formatting only)"


def _sort_versions(versions: set[str]) -> list[str]:
    """Sort by version, not text (``0.10.0`` after ``0.7.0``). Unparseable strings
    sort last — ``resume_version_verdict`` rejects those too, so they reach here.
    """

    def key(v: str) -> tuple[int, Version, str]:
        try:
            return (0, Version(v), v)
        except InvalidVersion:
            return (1, Version("0"), v)

    return sorted(versions, key=key)


async def _cross_version_resume_hint(result_path: anyio.Path) -> str:
    """Return a note when this result_dir was produced by an incompatible version.

    ``EvalSession`` compares its artifacts before any ``TaskRunner`` exists, so
    ``gate_resume_version`` (in ``TaskRunner.__init__``) cannot pre-empt those
    aborts and a cross-version resume arrives as a bare artifact diff. This
    annotates the diff rather than moving the gate, which reads a *per-task*
    ``meta.json`` and is fail-closed on a missing one — a not-yet-started task
    legitimately has none, so hoisting it changes resume semantics.

    Scoped to the dirs the gate would read: ``meta.json`` lands at run start, but
    a dir only counts as resumable once ``manifest.json`` exists.

    States the incompatibility and stops — it does **not** claim to explain the
    diff above it. Both can be true at once, and on a dev/local install every
    non-identical version pair is incompatible, so a claim about cause would
    often be the wrong one.

    ``""`` when nothing incompatible is found, and on any read problem: a hint
    must never mask the caller's own abort, nor invent a version problem.
    """

    def _scan() -> set[str]:
        found: set[str] = set()
        root = Path(result_path)
        if not root.is_dir():
            return found
        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not (task_dir / "manifest.json").exists():
                continue
            version = read_run_version(task_dir)
            if version is None:
                continue
            verdict = resume_version_verdict(version, __version__)
            if verdict.action is ResumeAction.REJECT:
                found.add(version)
        return found

    try:
        persisted = await run_sync(_scan)
    except Exception as e:
        # Never mask the caller's own abort — this is only an annotation on it.
        logger.debug("Skipping cross-version resume hint: {}", e)
        return ""

    if not persisted:
        return ""
    versions = _sort_versions(persisted)
    subject = (
        f"{versions[0]}, which is not"
        if len(versions) == 1
        else f"{', '.join(versions)}, none of which is"
    )
    return (
        f"Note: this result_dir holds task directories produced by sieval "
        f"{subject} resume-compatible with {__version__} — any task resuming "
        f"from those is refused on the version alone, independently of the "
        f"difference above.\n"
    )


def _brief_diff(existing: str, current: str) -> str:
    """Return a short human-readable hint describing which YAML keys differ.

    Falls back to a generic message if either body fails to parse — this
    can happen when the persisted file has been hand-edited into invalid
    YAML, and we don't want the parse error to mask the caller's Resume
    aborted RuntimeError.
    """
    try:
        e = yaml.safe_load(existing) or {}
        c = yaml.safe_load(current) or {}
    except yaml.YAMLError:
        return "Diff: (existing file is not valid YAML — cannot compute key-level diff)"
    return _diff_dicts(e, c)


# ── Resume strict-match field policy (must partition TaskRunnerConfig) ──
# Adjustable across --resume only if a field touches neither the sample data
# nor any persisted artifact: pure scheduling + console-only progress.
_THROUGHPUT_RUNNER_KEYS: frozenset[str] = frozenset(
    {
        "concurrency_limit",
        "concurrency_limits",
        "shard_read_concurrency",
        "shard_write_concurrency",
        "write_buffer_size",
        "write_buffer_flush_interval",
        # Console-only (tqdm bar + log cadence); not the progress.json dump.
        "show_progress",
        "progress_log_interval",
        "progress_log_pct_interval",
    }
)

# Must match: affect sample data, an on-disk artifact, or a recorded outcome's
# meaning — e.g. max_retries is the failure signal written into FAILED records;
# profile_*/detect_anomalies*/dump_progress write profiler/anomaly/progress files.
_STRICT_RUNNER_KEYS: frozenset[str] = frozenset(
    {
        "shard_samples",
        "record_each_stage",
        "record_type_metadata",
        "record_meta",
        "max_iterations",
        "deterministic",
        "max_retries",
        "profile_io",
        "profile_stages",
        "profile_usage",
        "detect_anomalies",
        "detect_anomalies_on_resume",
        "dump_progress",
        "progress_dump_interval",
    }
)

# Neither adjustable nor strict — listed only so the three buckets partition
# TaskRunnerConfig exactly (see test_every_field_classified_exactly_once). The
# strip removes result_dir at top level (reification injects it there); the rest
# are never reached because they don't survive into a persisted runner_config
# block: auto_resume is set by the orchestration layer at runtime, stage_meta
# hooks are non-serializable callables. A hand-authored runner_config field from
# this set that changed across a resume would still be compared strictly.
_NONMATCH_RUNNER_KEYS: frozenset[str] = frozenset(
    {
        "result_dir",
        "auto_resume",
        "stage_meta_hook",
        "stage_meta_hooks",
    }
)


def _strip_noncomparable_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``cfg`` with resume-mutable fields removed, for comparison.

    Strips (input never mutated) top-level ``concurrency_limit`` /
    ``concurrency_limits`` / ``result_dir``, ``models.*.args.concurrency_limit``,
    and ``_THROUGHPUT_RUNNER_KEYS`` from every ``runner_config`` block.
    """
    out = copy.deepcopy(cfg)

    for key in ("concurrency_limit", "concurrency_limits", "result_dir"):
        out.pop(key, None)

    models = out.get("models")
    if isinstance(models, dict):
        for mcfg in models.values():
            if isinstance(mcfg, dict):
                args = mcfg.get("args")
                if isinstance(args, dict):
                    args.pop("concurrency_limit", None)

    # runner_config carries throughput knobs in two equivalent places: the
    # top-level defaults block (merged into every task) and per-task overrides.
    # Strip both identically.
    runner_config_blocks = [out.get("runner_config")]
    tasks = out.get("tasks")
    if isinstance(tasks, dict):
        runner_config_blocks.extend(
            tcfg.get("runner_config")
            for tcfg in tasks.values()
            if isinstance(tcfg, dict)
        )
    for rc in runner_config_blocks:
        if isinstance(rc, dict):
            for key in _THROUGHPUT_RUNNER_KEYS:
                rc.pop(key, None)

    return out


def resolve_deterministic(cli_override: bool | None, config: Mapping[str, Any]) -> bool:
    """Effective deterministic flag: monotone OR of YAML and CLI.

    The CLI flag is one-way — can force on, cannot downgrade YAML.
    """
    return bool(config.get("deterministic", False)) or bool(cli_override)


def unwrap_proxies(obj: Any) -> Any:
    """Recursively convert dataclasses / MappingProxyType to YAML-safe dicts/lists.

    Why we can't use ``dataclasses.asdict`` upstream: on Python 3.13 it
    invokes ``copy.deepcopy`` on "other" values, which raises
    ``TypeError: cannot pickle 'mappingproxy' object`` for any frozen
    ``MappingProxyType`` (used by ``RoleAssignment.engine_params`` via
    ``_freeze_dict``). This walker sidesteps both the pickle dependency
    and the leftover ``MappingProxyType`` nodes that a successful
    ``asdict`` would leave behind.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: unwrap_proxies(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, MappingProxyType):
        return {k: unwrap_proxies(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: unwrap_proxies(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [unwrap_proxies(v) for v in obj]
    return obj


def _reify_cli_overrides(
    cfg: dict[str, Any],
    *,
    deterministic: bool | None = None,
    model: str | None = None,
    result_dir: str | None = None,
) -> dict[str, Any]:
    """Apply CLI overrides onto a config dict in place, return the same dict.

    Mirrors EvalSession's runtime override behavior so that `sieval eval
    <persisted_effective_config>` with NO CLI args reproduces the session:
        --deterministic → root `deterministic: true`; automatic request seeds
                          are resolved per binding after dialect reconciliation
        --model X       → overwrite `name: X` on every base model
        --result-dir D  → root `result_dir: D`
    Per-op seeds (dataset `shuffle.seed`, task `args.seed`) are not
    CLI-overridable; users edit YAML directly for those.
    """
    if deterministic:
        cfg["deterministic"] = True

    if model is not None:
        models = cfg.get("models") or {}
        if isinstance(models, dict):
            for mcfg in models.values():
                if isinstance(mcfg, dict) and "base" not in mcfg:
                    mcfg["name"] = model

    if result_dir is not None:
        cfg["result_dir"] = result_dir

    return cfg


def _apply_endpoint_injection(
    cfg: dict[str, Any], endpoint_map: Mapping[str, str]
) -> dict[str, Any]:
    """Adapt legacy endpoint-only callers into the YAML configuration shape.

    This compatibility adapter is for external callers that do not have a
    typed :class:`Deployment`. Internal auto-serve orchestration must hand the
    complete realized deployment to :class:`EvalSession` instead.

    For each model in ``endpoint_map``:
        - Set ``api_base`` to the given endpoint (always overrides).
        - If no ``api_key`` present at either top level or inside ``args``,
          set ``api_key: "local"`` placeholder (OpenAI client needs one).
        - If ``name`` is absent, derive from checkpoint basename (from
          ``infer.checkpoint`` or top-level ``path``).
    """
    models = cfg.get("models")
    if not isinstance(models, dict):
        return cfg

    for model_key, endpoint in endpoint_map.items():
        mcfg = models.get(model_key)
        if not isinstance(mcfg, dict):
            continue

        mcfg["api_base"] = endpoint

        if not mcfg.get("api_key"):
            args = mcfg.get("args") or {}
            if not (isinstance(args, dict) and args.get("api_key")):
                mcfg["api_key"] = "local"

        if "name" not in mcfg:
            checkpoint = ""
            infer_dict = mcfg.get("infer") or {}
            if isinstance(infer_dict, dict):
                checkpoint = infer_dict.get("checkpoint", "")
            if not checkpoint:
                checkpoint = mcfg.get("path", "")
            if checkpoint:
                mcfg["name"] = Path(checkpoint).name

    return cfg


def _format_comment_header(
    *,
    title: str,
    source_config: str,
    invocation: str,
    extra_lines: list[str] | None = None,
) -> str:
    """Return a YAML comment block capturing provenance for an audit file.

    Standard lines (always present):
        title @ sieval <version> at <ISO-8601 UTC>
        Invocation: <argv joined with spaces>
        Original source: <abs path of the user's YAML>

    ``extra_lines`` — caller-supplied free-form lines (no leading ``#``)
    inserted between ``Original source`` and the closing border. Callers
    use this for artifact-specific hints.

    ``yaml.safe_dump`` cannot preserve comments, so this header is
    prepended to the dumped body via string concatenation.
    """
    from sieval import __version__

    now = datetime.now(UTC).isoformat()
    border = "# " + "-" * 70
    lines = [
        border,
        f"# {title} sieval {__version__} at {now}",
        f"# Invocation: {invocation}",
        f"# Original source: {source_config}",
    ]
    if extra_lines:
        lines.extend(f"# {line}" for line in extra_lines)
    lines.extend([border, ""])
    return "\n".join(lines) + "\n"


def _append_resume_note(header: str, diff_lines: list[str]) -> str:
    """Insert a ``Resumed by …`` audit block into ``header``, before its border.

    Called when ``--resume`` rewrites a file because only resume-mutable fields
    changed. The original provenance survives and ``diff_lines`` is recorded with
    a timestamp, so the header accumulates the full lineage across resumes. The
    note sits inside the ``# ---`` border pair so :func:`_split_header` keeps
    treating the whole block as the header next time.

    Assumes a well-formed ``header`` (two borders) — the only kind the caller
    passes (from :func:`_format_comment_header` or :func:`_split_header`).
    """
    from sieval import __version__

    now = datetime.now(UTC).isoformat()
    note = [f"# Resumed by sieval {__version__} at {now}:\n"]
    note.extend(f"#   {line}\n" for line in diff_lines)

    lines = header.splitlines(keepends=True)
    borders = [i for i, line in enumerate(lines) if line.startswith("# -")]
    close = borders[-1]
    return "".join(lines[:close] + note + lines[close:])


def _warn_best_effort_deterministic(
    config: Mapping[str, Any],
    effective_deterministic: bool,
    self_managed_endpoints: frozenset[str] | set[str],
) -> None:
    """Warn when deterministic mode talks to engines we don't manage.

    For models reaching an externally-hosted ``api_base`` we apply a request
    seed when the selected protocol supports one, but cannot verify the remote
    process seed or batch-invariant kernels. Reproducibility is best-effort on
    those models.

    Only base models with their own ``api_base`` are listed; derived
    models that inherit ``api_base`` from a flagged base are covered
    transitively by the base's warning.
    """
    if not effective_deterministic:
        return
    external = sorted(
        name
        for name, cfg in (config.get("models") or {}).items()
        if isinstance(cfg, dict)
        and cfg.get("api_base")
        and name not in self_managed_endpoints
    )
    if external:
        _emit_best_effort_deterministic_warning(external)


def _emit_best_effort_deterministic_warning(labels: list[str]) -> None:
    """Emit the shared warning for externally managed model bindings."""

    logger.warning(
        "Deterministic mode is best-effort for model binding(s) {} — "
        "sieval applies request-level seeds where the protocol supports them, "
        "but cannot verify the remote process seed or batch-invariant kernels. "
        "For guaranteed reproducibility, self-host via `sieval run` / "
        "`sieval infer start` with a local checkpoint.",
        labels,
    )


class EvalSession:
    """
    YAML-based evaluation session.

    Example YAML structure:
    ```yaml
    result_dir: "./outputs/my-run"

    models:
      base_model:
        name: "gpt-4o"
        args:
          temperature: 0.0
        # infer:                  # optional, used by `sieval infer`
        #     backend: vllm
        #     checkpoint: /path/to/weights
        # infer_meta:             # optional, for result auditing
        #     framework: vllm==0.6.0
      math_model:
        base: base_model
        args:
          temperature: 0.7

    datasets:
      aime_2024:
        class: AIME2024Dataset  # or full path
        path: "./data/aime_2024"
        operations:
          - shuffle: {seed: 42}
          - slice: {num: 100}

    tasks:
      aime_2024_eval:
        class: AIME2024ZeroShotGenTask  # or full path
        dataset: aime_2024
        model: math_model
        args:
          k: 1
          n: 64
        infer_args:               # optional, per-task inference overrides
          max_tokens: 512         # overrides model's default
        runner_config:
          concurrency_limits:
            infer: 4
    ```
    """

    def __init__(
        self,
        config_path: str | Path,
        model_override: str | None = None,
        resume: bool = False,
        result_dir_override: str | None = None,
        deterministic_override: bool | None = None,
        endpoint_map: Mapping[str, str] | None = None,
        infer_plans: Mapping[str, dict[str, Any]] | None = None,
        invocation: str | None = None,
        self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
        realized_deployments: Mapping[str, Deployment] | None = None,
        model_capability_profiles: Mapping[str, ModelCapabilityProfile] | None = None,
        serving_reconciler: ServingReconciler | None = None,
    ):
        self.config_path = Path(config_path)
        self.model_override = model_override
        self.resume_override = resume
        self.result_dir_override = result_dir_override
        self.deterministic_override = deterministic_override
        legacy_endpoint_map = dict(endpoint_map or {})
        realized_deployment_map = dict(realized_deployments or {})
        if legacy_endpoint_map and realized_deployment_map:
            raise ValueError(
                "endpoint_map is a legacy endpoint-only adapter and cannot be "
                "combined with typed realized_deployments"
            )
        self._legacy_endpoint_map: Mapping[str, str] = legacy_endpoint_map
        self._infer_plans: Mapping[str, dict[str, Any]] | None = infer_plans
        self._realized_deployments = realized_deployment_map
        for root_name, deployment in self._realized_deployments.items():
            if not isinstance(root_name, str) or not root_name:
                raise TypeError("realized deployment keys must be non-empty strings")
            if not isinstance(deployment, Deployment):
                raise TypeError(
                    f"realized deployment {root_name!r} must be a Deployment"
                )
        self._model_capability_profiles = dict(model_capability_profiles or {})
        for profile_key, profile in self._model_capability_profiles.items():
            if not isinstance(profile_key, str) or not profile_key:
                raise TypeError(
                    "model capability profile keys must be non-empty strings"
                )
            if not isinstance(profile, ModelCapabilityProfile):
                raise TypeError(
                    f"model capability profile {profile_key!r} must be a "
                    "ModelCapabilityProfile"
                )
        compatibility_reconciler = _PR1CompatibilityServingReconciler(
            self._external_runtime_plans_for_root
        )
        self._serving_reconciler: ServingReconciler = (
            compatibility_reconciler
            if serving_reconciler is None
            else _PR1CompositeServingReconciler(
                compatibility_reconciler,
                serving_reconciler,
            )
        )
        # Snapshot at init time so every audit file this session writes
        # carries the same string. Library/test callers pass explicit; CLI
        # falls back to sys.argv.
        self.invocation: str = (
            invocation if invocation is not None else shlex.join(sys.argv)
        )

        with open(self.config_path, encoding="utf-8") as f:
            loaded_config: RootConfigDict | None = yaml.safe_load(f)
            if loaded_config is None:
                loaded_config = {}
            if not isinstance(loaded_config, dict):
                raise ValueError("Top-level YAML config must be a dictionary")

        # Pristine YAML — source of truth for deterministic / result_dir
        # resolution. Deep-copied so downstream in-place mutation on
        # ``loaded_config`` cannot leak in.
        self._raw_config: RootConfigDict = copy.deepcopy(loaded_config)

        # Optional alignment block. Card path is stored verbatim (relative
        # to ``config_path.parent``) in both raw and reified views so
        # ``effective_config.yaml`` stays portable across machines.
        alignment_card: AlignmentCard | None = None
        alignment_block = loaded_config.get("alignment")
        if alignment_block is not None:
            if not isinstance(alignment_block, dict):
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} `alignment` must be a mapping"
                )
            if "card" not in alignment_block:
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} has `alignment` block "
                    f"without required sub-key `alignment.card`"
                )
            unknown = set(alignment_block) - {"card"}
            if unknown:
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} `alignment` has unknown "
                    f"keys: {sorted(unknown)} (only `card` is supported)"
                )
            card_rel = alignment_block["card"]
            if not isinstance(card_rel, str) or not card_rel:
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} `alignment.card` must "
                    f"be a non-empty string"
                )
            card_path = (self.config_path.parent / card_rel).resolve()
            alignment_card = load_card(card_path)
        self.alignment_card: AlignmentCard | None = alignment_card

        # Raw + CLI reification, BEFORE legacy endpoint injection — this is
        # what gets persisted to effective_config.yaml, so rerun via
        # `sieval run` re-launches services instead of connecting to a stale
        # endpoint.
        reified = _reify_cli_overrides(
            # cast: ty rejects TypedDict → dict[str, Any] and its natural shims.
            cast(dict[str, Any], copy.deepcopy(loaded_config)),
            deterministic=deterministic_override,
            model=model_override,
            result_dir=result_dir_override,
        )
        # Before the copy, so the digests reach both the persisted view (and
        # therefore --resume) and the runtime view below. Not where the
        # operation runs: `arun` persists before `_prepare_execution`, so by
        # then the comparison the digests exist for has already been made.
        pin_filter_digests(reified, self.config_path.parent)
        self._reified_config: dict[str, Any] = copy.deepcopy(reified)

        # Runtime view = reified + the legacy external adapter (mutates
        # ``reified``). Typed deployments do not rewrite YAML state.
        # cast: helper is typed dict[str, Any] for mutation; narrow at the boundary.
        self.config: RootConfigDict = cast(
            RootConfigDict,
            _apply_endpoint_injection(reified, self._legacy_endpoint_map),
        )

        self.deterministic: bool = resolve_deterministic(
            deterministic_override, self._raw_config
        )

        _warn_best_effort_deterministic(
            self.config, self.deterministic, self_managed_endpoints
        )

        self.models: dict[str, Model] = {}
        self.datasets: dict[str, Dataset] = {}
        self.prelaunch_reconcile_result: ReconcileResult | None = None
        self.postlaunch_reconcile_result: ReconcileResult | None = None
        self._normalized_model_bindings: dict[str, NormalizedModelBinding] = {}
        self._task_requirement_contexts: dict[str, RequirementContext] = {}
        self._task_model_requirements: tuple[TaskModelRequirement, ...] = ()
        self._aggregated_requirements: dict[str, AggregatedTaskRequirements] = {}
        self._model_types_by_root: dict[str, Literal["chat", "gen"]] = {}
        self._legacy_bypass_bindings: frozenset[str] = frozenset()
        self._prelaunch_binding_inputs: tuple[BindingReconcileInput, ...] = ()
        self._prelaunch_deployment_inputs: dict[str, DeploymentReconcileInput] = {}
        self._realized_deployments_by_root: dict[str, Deployment] = {}
        self._owned_pools: dict[str, ConnectionPool[Any]] = {}
        self._root_shared_limiters: dict[str, anyio.CapacityLimiter | None] = {}
        self._owned_legacy_models: dict[str, SglangGenModel] = {}
        self._warned_best_effort_role_bindings: set[str] = set()
        # Runtime-only sources for the post-launch ``models_by_role`` binding
        # seam.  They are never serialized or fingerprinted by reconciliation.
        self._task_role_model_sources: dict[str, dict[str, object]] = {}
        self._bound_task_role_models: dict[str, dict[str, Model]] = {}
        # Runtime-only deterministic request-default decisions.  Prelaunch
        # freezes these from normalized bindings, exact external role sources,
        # and RequirementContext.infer_args.  Only their JSON projection enters
        # effective_config; the caches themselves are never fingerprinted.
        self._request_seed_decisions_by_binding: dict[str, _RequestSeedDecision] = {}
        self._request_seed_decisions_by_external_role: dict[
            str, _RequestSeedDecision
        ] = {}
        self._request_seed_decisions_by_candidate: dict[str, _RequestSeedDecision] = {}
        self._request_seed_decisions_frozen = False

        # Resolved lazily in `_init_runner` at the start of `_prepare_execution`.
        self.result_dir: str | None = None
        self.runner: MultiTaskRunner | None = None

    def _init_runner(self) -> None:
        self.result_dir = self.result_dir_override or self.config.get("result_dir")

        self.runner = MultiTaskRunner(
            result_dir=self.result_dir,
            concurrency_limit=self.config.get("concurrency_limit"),
            concurrency_limits=self.config.get("concurrency_limits"),
            deterministic=self.deterministic,
        )

    def _get_named_config_map(self, section_name: str) -> dict[str, dict[str, Any]]:
        """Get a config section and validate it is a name -> dict mapping."""
        return validate_named_config_map(
            section_name,
            self.config.get(section_name, {}),
        )

    @staticmethod
    def _normalize_dict(value: Any, field_name: str) -> dict[str, Any]:
        """Normalize optional dict fields, rejecting invalid shapes."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a dictionary")
        return value.copy()

    @staticmethod
    def _normalize_list(value: Any, field_name: str) -> list[Any]:
        """Normalize optional list fields, rejecting invalid shapes."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list")
        return value

    def _model_config_chain(
        self,
        model_name: str,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return a root-to-leaf model config chain without constructing models."""

        chain: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()

        def visit(current: str) -> None:
            if current in seen:
                names = [name for name, _ in chain]
                names.append(current)
                cycle = " -> ".join(names)
                raise ValueError(f"Circular model inheritance detected: {cycle}")
            seen.add(current)
            try:
                config = models_cfg[current]
            except KeyError as exc:
                raise ValueError(
                    f"Model '{model_name}' references unknown base model '{current}'"
                ) from exc
            chain.append((current, config))
            base = config.get("base")
            if base is None:
                return
            if not isinstance(base, str) or not base:
                raise ValueError(
                    f"Model '{current}' has invalid 'base' value: {base!r}"
                )
            visit(base)

        visit(model_name)
        chain.reverse()
        return tuple(chain)

    @staticmethod
    def _merged_model_value(
        chain: tuple[tuple[str, dict[str, Any]], ...], field: str
    ) -> object | None:
        value: object | None = None
        for _, config in chain:
            if field in config:
                value = config[field]
        return value

    @staticmethod
    def _merged_capability_declarations(
        chain: tuple[tuple[str, dict[str, Any]], ...],
    ) -> dict[str, JSONValue]:
        declarations: dict[str, JSONValue] = {}
        for config_name, config in chain:
            raw = config.get("capabilities")
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Model '{config_name}' capabilities must be a dictionary"
                )
            for key, value in raw.items():
                if not isinstance(key, str):
                    raise TypeError("capability declaration keys must be strings")
                declarations[key] = cast(JSONValue, copy.deepcopy(value))
        return declarations

    def _task_model_config_name(
        self,
        task_name: str,
        task_cfg: Mapping[str, Any],
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> str:
        model_ref = task_cfg.get("model")
        if model_ref is not None and not isinstance(model_ref, str):
            raise ValueError(f"Task '{task_name}': 'model' must be a string reference")
        if model_ref:
            if model_ref not in models_cfg:
                raise ValueError(
                    f"Task '{task_name}' references unknown model '{model_ref}'"
                )
            return model_ref
        if len(models_cfg) == 1:
            return next(iter(models_cfg))
        if not models_cfg:
            raise ValueError(f"Task '{task_name}': no models defined in config")
        raise ValueError(
            f"Task '{task_name}': 'model' required when multiple models are defined"
        )

    def _provisional_named_binding(
        self,
        model_name: str,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> NamedModelBinding:
        """Build the stable named identity used by task requirement hooks.

        Dialect selection depends on normalized task input evidence, so it is
        deliberately absent until every hook has run and the deployment root's
        legacy kind has been derived exactly once.
        """

        chain = self._model_config_chain(model_name, models_cfg)
        self._validate_named_resource_config(model_name, chain[-1][1])
        root_name, root_config = chain[0]
        requested_model_id = self.model_override or root_config.get("name")
        if not requested_model_id:
            infer_config = root_config.get("infer")
            checkpoint = (
                infer_config.get("checkpoint")
                if isinstance(infer_config, Mapping)
                else None
            )
            if not checkpoint:
                checkpoint = root_config.get("path")
            requested_model_id = Path(checkpoint).name if checkpoint else root_name
        if not isinstance(requested_model_id, str) or not requested_model_id:
            raise ValueError(f"Model '{root_name}' has no usable requested model id")

        return NamedModelBinding(
            binding_id=f"model:{model_name}",
            root_deployment_key=f"model:{root_name}",
            requested_model_id=requested_model_id,
            config_name=model_name,
        )

    @staticmethod
    def _explicit_model_type_for_root(
        root_name: str,
        bindings: tuple[NamedModelBinding, ...],
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> Literal["chat", "gen"] | None:
        """Merge YAML ``type:`` assertions across one inheritance root."""

        declarations: dict[str, object] = {
            binding.config_name: models_cfg[binding.config_name]["type"]
            for binding in bindings
            if "type" in models_cfg[binding.config_name]
        }
        invalid = {
            name: value
            for name, value in declarations.items()
            if value not in ("chat", "gen")
        }
        if invalid:
            details = ", ".join(
                f"{name}={value!r}" for name, value in sorted(invalid.items())
            )
            raise ValueError(
                f"Model deployment root '{root_name}' has invalid type "
                f"assertion(s): {details}; expected 'chat' or 'gen'"
            )
        values = set(declarations.values())
        if len(values) > 1:
            details = ", ".join(
                f"{name}={value!r}" for name, value in sorted(declarations.items())
            )
            raise ValueError(
                f"Models sharing deployment root '{root_name}' declare "
                f"conflicting type assertions: {details}"
            )
        if not values:
            return None
        return cast(Literal["chat", "gen"], next(iter(values)))

    def _finalize_named_binding(
        self,
        binding: NamedModelBinding,
        model_type: Literal["chat", "gen"],
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> NamedModelBinding:
        """Select a dialect after root model-kind derivation."""

        chain = self._model_config_chain(binding.config_name, models_cfg)
        _, root_config = chain[0]
        dialect_declared = any("dialect" in config for _, config in chain)
        explicit_dialect = self._merged_model_value(chain, "dialect")
        if dialect_declared and (
            not isinstance(explicit_dialect, str) or not explicit_dialect
        ):
            raise ValueError(
                f"Model '{binding.config_name}' dialect must be a non-empty string"
            )

        # Only the explicit model-level assertion activates the one-cycle
        # SGLang compatibility shim.  A managed DeploymentPlan's engine is a
        # serving fact, not a dialect selector.
        root_engine = root_config.get("engine")
        if dialect_declared:
            assert isinstance(explicit_dialect, str)
            dialect_id = explicit_dialect
        elif root_engine == "sglang" and model_type == "gen":
            dialect_id = (
                "sglang_native"
                if dialect_is_bindable("sglang_native")
                else "sglang_legacy"
            )
        else:
            dialect_id = "openai_chat" if model_type == "chat" else "openai_completions"
        return dataclasses.replace(binding, dialect_id=dialect_id)

    def _validate_named_resource_config(
        self,
        model_name: str,
        config: Mapping[str, object],
    ) -> None:
        """Reject resource overrides before reconciliation, allocation, or I/O."""

        is_root = config.get("base") is None
        allowed_top_level = (
            frozenset(
                {
                    "api_base",
                    "api_key",
                    "capabilities",
                    "dialect",
                    "engine",
                    "max_retries",
                    "service_role",
                }
            )
            if is_root
            else frozenset({"capabilities", "dialect", "service_role"})
        )
        misplaced = binding_resource_argument_paths(
            config,
            allowed=allowed_top_level,
        )
        raw_args = config.get("args", {})
        if not isinstance(raw_args, Mapping):
            raise ValueError(f"Model '{model_name}' args must be a dictionary")
        allowed_args = (
            frozenset({"api_base", "api_key", "max_retries"})
            if is_root
            else frozenset()
        )
        nested = binding_resource_argument_paths(
            cast(Mapping[str, object], raw_args),
            allowed=allowed_args,
        )
        paths = (*misplaced, *(f"args.{path}" for path in nested))
        if paths:
            kind = "Root" if is_root else "Derived"
            raise ValueError(
                f"{kind} model '{model_name}' places binding resource(s) on an "
                f"unsupported surface: {', '.join(paths)}"
            )

    def _normalized_external_binding(
        self,
        task_name: str,
        role: str,
        model: Model,
    ) -> ExternalModelBinding:
        runtime_plan = getattr(model, "runtime_plan", None)
        if runtime_plan is None:
            raise ValueError(
                f"Task '{task_name}' external {role} model has no RuntimeBindingPlan"
            )
        return ExternalModelBinding(
            binding_id=runtime_plan.binding_id,
            root_deployment_key=runtime_plan.root_deployment_key,
            requested_model_id=runtime_plan.requested_model_id,
            runtime_plan_fingerprint=runtime_plan.fingerprint,
            dialect_id=runtime_plan.dialect_id,
        )

    def _requirement_dataset_config(
        self, task_name: str, task_cfg: Mapping[str, Any]
    ) -> Mapping[str, JSONValue]:
        dataset_ref = task_cfg.get("dataset")
        if isinstance(dataset_ref, str):
            datasets_cfg = self._get_named_config_map("datasets")
            if dataset_ref not in datasets_cfg:
                raise ValueError(
                    f"Task '{task_name}' references unknown dataset '{dataset_ref}'"
                )
            return cast(
                Mapping[str, JSONValue], copy.deepcopy(datasets_cfg[dataset_ref])
            )
        if isinstance(dataset_ref, Mapping):
            return cast(Mapping[str, JSONValue], copy.deepcopy(dataset_ref))
        raise ValueError(
            f"Task '{task_name}': 'dataset' must be a string reference or "
            "inline definition"
        )

    def _requirement_context_for_task(
        self,
        task_name: str,
        task_cfg: Mapping[str, Any],
        task_class: type[Any],
        candidate: NamedModelBinding,
    ) -> RequirementContext:
        raw_task_args = self._normalize_dict(
            task_cfg.get("args"), f"Task '{task_name}' args"
        )
        validate_task_config_args(
            task_name,
            raw_task_args,
            task_class=task_class,
        )
        json_task_args = dict(raw_task_args)
        bindings: dict[str, NormalizedModelBinding] = {"candidate": candidate}
        sources: dict[str, object] = {"candidate": candidate.config_name}

        for role in TASK_MODEL_ROLES:
            role_source = json_task_args.get(role)
            if is_task_model_role_sentinel(role, role_source):
                # Keep the sentinel in task_args. Task construction resolves it
                # against the candidate only after task infer_args are applied.
                continue
            json_task_args.pop(role, None)
            if role_source is None:
                continue
            if isinstance(role_source, Model):
                role_binding = self._normalized_external_binding(
                    task_name, role, role_source
                )
            elif isinstance(role_source, Mapping):
                role_binding = normalize_inline_model_binding(
                    task_name, role, role_source
                )
            else:
                raise ValueError(
                    f"Task '{task_name}' {role} must be "
                    + (
                        "'self', an inline mapping, or Model"
                        if role == "extractor"
                        else "an inline mapping or Model"
                    )
                )
            bindings[role] = role_binding
            sources[role] = role_source
            self._normalized_model_bindings.setdefault(
                role_binding.binding_id, role_binding
            )

        hidden_model_paths = _model_value_paths(json_task_args)
        if hidden_model_paths:
            raise ValueError(
                f"Task '{task_name}' args contain Model value(s) outside registered "
                f"model roles: {', '.join(hidden_model_paths)}. Use one of: "
                + ", ".join(TASK_MODEL_ROLES)
            )

        infer_args = self._normalize_dict(
            task_cfg.get("infer_args"), f"Task '{task_name}' infer_args"
        )
        misplaced_resources = binding_resource_argument_paths(infer_args)
        if misplaced_resources:
            raise ValueError(
                f"Task '{task_name}' infer_args cannot change binding resources: "
                + ", ".join(misplaced_resources)
            )
        context = RequirementContext(
            model_bindings=bindings,
            task_args=cast(Mapping[str, JSONValue], json_task_args),
            dataset_config=self._requirement_dataset_config(task_name, task_cfg),
            infer_args=cast(Mapping[str, JSONValue], infer_args),
        )
        self._task_role_model_sources[task_name] = sources
        return context

    @staticmethod
    def _format_reconcile_diagnostic(diagnostic: ReconcileDiagnostic) -> str:
        location = diagnostic.binding_id or diagnostic.root_deployment_key or "batch"
        capability = (
            f" [{diagnostic.capability}]" if diagnostic.capability is not None else ""
        )
        sources = (
            f" (sources: {', '.join(diagnostic.sources)})" if diagnostic.sources else ""
        )
        return (
            f"- {diagnostic.code} at {location}{capability}: "
            f"{diagnostic.message}{sources}"
        )

    def _external_model_for_binding(self, binding_id: str) -> Model | None:
        for sources in self._task_role_model_sources.values():
            for source in sources.values():
                if not isinstance(source, Model):
                    continue
                plan = getattr(source, "runtime_plan", None)
                if plan is not None and plan.binding_id == binding_id:
                    return source
        return None

    def _external_model_for_task_role(
        self,
        task_name: str,
        role: str,
        binding_id: str,
    ) -> Model:
        """Return the exact caller-owned source attached to one task role."""

        source = self._task_role_model_sources.get(task_name, {}).get(role)
        if not isinstance(source, Model) or source.runtime_plan is None:
            raise ValueError(
                f"External binding '{binding_id}' lost the model source for "
                f"task {task_name!r} role {role!r}"
            )
        if source.runtime_plan.binding_id != binding_id:
            raise ValueError(
                f"External model source for task {task_name!r} role {role!r} "
                f"changed binding identity from '{binding_id}' to "
                f"'{source.runtime_plan.binding_id}'"
            )
        return source

    def _external_runtime_plans_for_root(
        self, root_deployment_key: str
    ) -> tuple[RuntimeBindingPlan, ...]:
        """Return every distinct external binding plan for one deployment root."""

        found: dict[str, RuntimeBindingPlan] = {}
        for sources in self._task_role_model_sources.values():
            for source in sources.values():
                if not isinstance(source, Model) or source.runtime_plan is None:
                    continue
                plan = source.runtime_plan
                if plan.root_deployment_key != root_deployment_key:
                    continue
                previous = found.get(plan.binding_id)
                if previous is not None and previous != plan:
                    raise ValueError(
                        f"external binding {plan.binding_id!r} resolves to different "
                        "runtime plans within deployment root "
                        f"{root_deployment_key!r}"
                    )
                found[plan.binding_id] = plan
        return tuple(found[key] for key in sorted(found))

    def _validate_external_runtime_obligations(self, result: ReconcileResult) -> None:
        """Reject a rebound external plan that weakens its existing guarantees."""

        for binding in self._normalized_model_bindings.values():
            if not isinstance(binding, ExternalModelBinding):
                continue
            model = self._external_model_for_binding(binding.binding_id)
            if model is None or model.runtime_plan is None:
                raise ValueError(
                    f"External binding '{binding.binding_id}' lost its runtime plan"
                )
            baseline = model.runtime_plan
            if baseline.fingerprint != binding.runtime_plan_fingerprint:
                raise ValueError(
                    f"External binding '{binding.binding_id}' runtime plan changed "
                    "after requirement normalization"
                )
            rebound = result.runtime_plans.get(binding.binding_id)
            if rebound is None:
                continue
            identity_fields = (
                "binding_id",
                "root_deployment_key",
                "requested_model_id",
                "dialect_id",
                "deployment_fingerprint",
                "resolved_route",
                "connection_identity",
            )
            changed = [
                field
                for field in identity_fields
                if getattr(rebound, field) != getattr(baseline, field)
            ]
            if changed:
                raise ValueError(
                    f"External binding '{binding.binding_id}' changed immutable "
                    "runtime identity fields: " + ", ".join(changed)
                )

            newly_claimed = (
                rebound.available_capabilities - baseline.available_capabilities
            )
            if newly_claimed:
                raise ValueError(
                    f"External binding '{binding.binding_id}' cannot add runtime "
                    "capabilities absent from its existing plan: "
                    + ", ".join(sorted(newly_claimed))
                )

            rebound_checks = set(rebound.request_checks)
            missing_checks = [
                check
                for check in baseline.request_checks
                if check.capability in rebound.available_capabilities
                and check not in rebound_checks
            ]
            if missing_checks:
                raise ValueError(
                    f"External binding '{binding.binding_id}' dropped existing "
                    "request-time safety checks: "
                    + ", ".join(
                        f"{check.capability}:{check.verifier}"
                        for check in missing_checks
                    )
                )

            for capability, minimums in rebound.capability_minimums.items():
                requested = minimums.get("minimum")
                previous = baseline.capability_minimums.get(capability, {}).get(
                    "minimum"
                )
                stronger = (
                    isinstance(requested, int)
                    and not isinstance(requested, bool)
                    and (
                        not isinstance(previous, int)
                        or isinstance(previous, bool)
                        or requested > previous
                    )
                )
                required_verifier = _PR1_REQUEST_VERIFIERS.get(
                    cast(CapabilityKey, capability)
                )
                if stronger and not any(
                    check.capability == capability
                    and required_verifier is not None
                    and check.verifier == required_verifier
                    for check in rebound.request_checks
                ):
                    raise ValueError(
                        f"External binding '{binding.binding_id}' requires a "
                        f"stronger {capability!r} minimum without a preserved "
                        "registered request-time check"
                    )

    def _model_profile_for(
        self, binding: NormalizedModelBinding
    ) -> ModelCapabilityProfile:
        lookup_keys = [binding.binding_id, binding.requested_model_id]
        if isinstance(binding, NamedModelBinding):
            lookup_keys.insert(0, binding.config_name)
        for key in lookup_keys:
            profile = self._model_capability_profiles.get(key)
            if profile is not None:
                return profile

        if isinstance(binding, ExternalModelBinding):
            model = self._external_model_for_binding(binding.binding_id)
            if model is None:
                raise ValueError(
                    f"External binding '{binding.binding_id}' lost its live "
                    "model source"
                )
            runtime_plan = model.runtime_plan
            if runtime_plan is None:
                raise ValueError(
                    f"External binding '{binding.binding_id}' has no runtime plan"
                )
            entries = {
                key: ModelCapabilityEntry(
                    (
                        ModelCapabilityStatus.SUPPORTED
                        if key in runtime_plan.available_capabilities
                        else ModelCapabilityStatus.UNSUPPORTED
                    ),
                    source="external-runtime-plan",
                    reason=(
                        None
                        if key in runtime_plan.available_capabilities
                        else (
                            "external runtime plan does not make this capability "
                            "available"
                        )
                    ),
                    verifier=(
                        "external_runtime_plan"
                        if key in runtime_plan.available_capabilities
                        else None
                    ),
                )
                for key in CAPABILITY_KEYS
            }
            return ModelCapabilityProfile(entries, authoritative=True)

        entries = {
            cast(CapabilityKey, capability): ModelCapabilityEntry(
                ModelCapabilityStatus.UNKNOWN,
                source="pr1-existing-response-contract",
                reason=(
                    "no authoritative model profile is available; the existing "
                    "response guard must verify this capability on every call"
                ),
                verifier=verifier,
            )
            for capability, verifier in _PR1_REQUEST_VERIFIERS.items()
        }
        return ModelCapabilityProfile(entries, authoritative=False)

    def _declarations_for_binding(
        self,
        binding: NormalizedModelBinding,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> Mapping[str, JSONValue]:
        if isinstance(binding, NamedModelBinding):
            chain = self._model_config_chain(binding.config_name, models_cfg)
            return self._merged_capability_declarations(chain)
        if isinstance(binding, InlineModelBinding):
            declarations = binding.config.get("capabilities", {})
            if not isinstance(declarations, Mapping):
                raise ValueError(
                    f"Inline binding '{binding.binding_id}' capabilities must "
                    "be a mapping"
                )
            return cast(Mapping[str, JSONValue], declarations)
        model = self._external_model_for_binding(binding.binding_id)
        assert model is not None
        runtime_plan = model.runtime_plan
        if runtime_plan is None:
            raise ValueError(
                f"External binding '{binding.binding_id}' has no runtime plan"
            )
        return runtime_plan.declared_capabilities

    def _validate_legacy_capability_surfaces(
        self,
        binding: NormalizedModelBinding,
        declarations: Mapping[str, JSONValue],
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> None:
        """Reject two config owners for one canonical capability semantic."""

        if not declarations or isinstance(binding, ExternalModelBinding):
            return

        canonical_source: str
        if isinstance(binding, NamedModelBinding):
            canonical_source = f"models.{binding.config_name}.capabilities"
            for config_name, config in self._model_config_chain(
                binding.config_name, models_cfg
            ):
                raw_args = config.get("args")
                if raw_args is None:
                    continue
                if not isinstance(raw_args, Mapping):
                    raise ValueError(f"Model '{config_name}' args must be a dictionary")
                validate_no_legacy_capability_ambiguity(
                    declarations,
                    raw_args,
                    canonical_source=canonical_source,
                    legacy_source=f"models.{config_name}.args",
                )
        else:
            canonical_source = f"{binding.binding_id}.capabilities"
            # Inline model configs accept legacy request defaults both at the
            # top level and in an optional nested args mapping.
            validate_no_legacy_capability_ambiguity(
                declarations,
                binding.config,
                canonical_source=canonical_source,
                legacy_source=f"{binding.binding_id} inline config",
            )
            raw_args = binding.config.get("args")
            if raw_args is not None:
                if not isinstance(raw_args, Mapping):
                    raise ValueError(
                        f"Inline binding '{binding.binding_id}' args must be a mapping"
                    )
                validate_no_legacy_capability_ambiguity(
                    declarations,
                    cast(Mapping[str, object], raw_args),
                    canonical_source=canonical_source,
                    legacy_source=f"{binding.binding_id}.args",
                )

        for task_name, context in sorted(self._task_requirement_contexts.items()):
            candidate = context.model_bindings.get("candidate")
            if candidate is None or candidate.binding_id != binding.binding_id:
                continue
            validate_no_legacy_capability_ambiguity(
                declarations,
                context.infer_args,
                canonical_source=canonical_source,
                legacy_source=f"tasks.{task_name}.infer_args",
            )

    def _legacy_request_intents_for(
        self,
        binding: NormalizedModelBinding,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> Mapping[CapabilityKey, CapabilityIntent]:
        """Project active migration-era request defaults into reconciliation."""

        intents: list[CapabilityIntent] = []

        def extend(arguments: Mapping[str, object], source: str) -> None:
            intents.extend(legacy_capability_intents(arguments, source=source).values())

        if isinstance(binding, NamedModelBinding):
            # Match ``with_args`` inheritance: each child replaces a same-name
            # builder default from its parent.  Preserve the winning config as
            # diagnostic evidence rather than OR-ing overridden values.
            effective: dict[str, tuple[object, str]] = {}
            for config_name, config in self._model_config_chain(
                binding.config_name, models_cfg
            ):
                raw_args = config.get("args")
                if raw_args is None:
                    continue
                if not isinstance(raw_args, Mapping):
                    raise ValueError(f"Model '{config_name}' args must be a dictionary")
                for name, value in raw_args.items():
                    if isinstance(name, str):
                        effective[name] = (value, f"models.{config_name}.args")
            by_source: dict[str, dict[str, object]] = {}
            for name, (value, source) in effective.items():
                by_source.setdefault(source, {})[name] = value
            for source, arguments in sorted(by_source.items()):
                extend(arguments, source)
        elif isinstance(binding, InlineModelBinding):
            # Inline task models apply nested ``args`` after direct fields.
            effective = {
                name: (value, f"{binding.binding_id} inline config")
                for name, value in binding.config.items()
                if name != "args"
            }
            raw_args = binding.config.get("args")
            if raw_args is not None:
                if not isinstance(raw_args, Mapping):
                    raise ValueError(
                        f"Inline binding '{binding.binding_id}' args must be a mapping"
                    )
                for name, value in raw_args.items():
                    if isinstance(name, str):
                        effective[name] = (value, f"{binding.binding_id}.args")
            by_source = {}
            for name, (value, source) in effective.items():
                by_source.setdefault(source, {})[name] = value
            for source, arguments in sorted(by_source.items()):
                extend(arguments, source)

        for task_name, context in sorted(self._task_requirement_contexts.items()):
            candidate = context.model_bindings.get("candidate")
            if candidate is None or candidate.binding_id != binding.binding_id:
                continue
            extend(context.infer_args, f"tasks.{task_name}.infer_args")

        return aggregate_capability_intents(intents)

    def _connection_scope_for(
        self,
        binding: NormalizedModelBinding,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> ConnectionScope:
        if binding.dialect_id is None:
            raise ValueError(f"Model binding '{binding.binding_id}' has no dialect")
        connection_family = get_dialect_spec(binding.dialect_id).connection_family
        factory = CONNECTION_FACTORY_REGISTRY.get(connection_family)

        if isinstance(binding, ExternalModelBinding):
            model = self._external_model_for_binding(binding.binding_id)
            assert model is not None
            runtime_plan = model.runtime_plan
            if runtime_plan is None:
                raise ValueError(
                    f"External binding '{binding.binding_id}' has no runtime plan"
                )
            identity = runtime_plan.connection_identity
            if identity.connection_family != connection_family:
                raise ValueError(
                    f"External binding '{binding.binding_id}' uses connection family "
                    f"{identity.connection_family!r}, but dialect "
                    f"{binding.dialect_id!r} requires {connection_family!r}"
                )
            factory.parse_retry_policy(identity.retry_policy)
            return ConnectionScope(
                identity.credential_scope,
                identity.retry_policy,
                identity.quota_scope,
            )
        if isinstance(binding, InlineModelBinding):
            source = self._source_for_binding(binding.binding_id)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"Inline binding '{binding.binding_id}' lost its config source"
                )
            typed_source = cast(Mapping[str, object], source)
            _, max_retries, _ = self._connection_options(
                typed_source,
                nested_args=True,
            )
            raw_args = typed_source.get("args")
            nested: Mapping[str, object] = (
                cast(Mapping[str, object], raw_args)
                if isinstance(raw_args, Mapping)
                else {}
            )
            credential_kind = (
                "explicit"
                if "api_key" in typed_source or "api_key" in nested
                else "environment"
            )
            return ConnectionScope(
                f"inline:{binding.binding_id}:{credential_kind}-credential",
                factory.retry_policy(max_retries),
                binding.root_deployment_key,
            )

        chain = self._model_config_chain(binding.config_name, models_cfg)
        root_name, root_config = chain[0]
        args = root_config.get("args")
        args = args if isinstance(args, Mapping) else {}
        has_explicit_credential = "api_key" in root_config or "api_key" in args
        _, max_retries, _ = self._connection_options(
            root_config,
            nested_args=True,
        )
        return ConnectionScope(
            (
                f"model:{root_name}:managed-local-credential"
                if root_name in (self._infer_plans or {})
                else (
                    f"model:{root_name}:explicit-credential"
                    if has_explicit_credential
                    else f"model:{root_name}:environment-credential"
                )
            ),
            factory.retry_policy(max_retries),
            binding.root_deployment_key,
        )

    def _route_intent_for(
        self,
        binding: NormalizedModelBinding,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> RouteIntent:
        """Normalize a service-role route before launch/reconciliation."""

        if isinstance(binding, ExternalModelBinding):
            model = self._external_model_for_binding(binding.binding_id)
            if model is None or model.runtime_plan is None:
                raise ValueError(
                    f"External binding '{binding.binding_id}' has no runtime plan"
                )
            service_role: object | None = model.runtime_plan.resolved_route.service_role
            service_role_declared = service_role is not None
        elif isinstance(binding, InlineModelBinding):
            service_role_declared = "service_role" in binding.config
            service_role = binding.config.get("service_role")
        else:
            chain = self._model_config_chain(binding.config_name, models_cfg)
            service_role_declared = any("service_role" in config for _, config in chain)
            service_role = self._merged_model_value(chain, "service_role")
        if not service_role_declared:
            return RouteIntent()
        if not isinstance(service_role, str) or not service_role:
            raise ValueError("model service_role must be a non-empty string")
        return RouteIntent(service_role)

    @staticmethod
    def _plan_projection(raw_plan: Mapping[str, object]) -> DeploymentPlanProjection:
        return deployment_plan_projection(raw_plan)

    @staticmethod
    def _explicit_engine_parameters(
        root_name: str,
        root_config: Mapping[str, object],
    ) -> Mapping[str, JSONValue]:
        """Project only engine parameters with a provable explicit user source.

        Effective ``DeploymentPlan`` engine params already merge recipe defaults
        with user overrides, so reverse-engineering recipe ownership from that
        value would be guesswork.  The explicit ``infer.overrides`` subtree is a
        stable source boundary and uses the infer layer's canonical key spelling.
        """

        raw_infer = root_config.get("infer")
        if raw_infer is None:
            return {}
        if not isinstance(raw_infer, Mapping):
            raise ValueError(f"Model '{root_name}' infer must be a dictionary")
        typed_infer = cast(Mapping[str, object], raw_infer)
        raw_overrides = typed_infer.get("overrides")
        if raw_overrides is None:
            return {}
        if not isinstance(raw_overrides, Mapping):
            raise ValueError(
                f"Model '{root_name}' infer.overrides must be a dictionary"
            )
        validated: dict[str, _ParamValue] = {}
        for key, value in raw_overrides.items():
            if not isinstance(key, str) or not key:
                raise TypeError(
                    f"Model '{root_name}' infer.overrides keys must be strings"
                )
            if not isinstance(value, str | int | float | bool):
                raise TypeError(
                    f"Model '{root_name}' infer.overrides.{key} must be a scalar"
                )
            validated[key] = value
        return cast(Mapping[str, JSONValue], merge_params(validated))

    def _deployment_input_for(
        self,
        binding: NormalizedModelBinding,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> DeploymentReconcileInput:
        if isinstance(binding, ExternalModelBinding):
            model = self._external_model_for_binding(binding.binding_id)
            assert model is not None
            deployment = model.deployment
            return DeploymentReconcileInput(
                root_deployment_key=binding.root_deployment_key,
                engine_id=deployment.engine.engine_id,
                deployment=deployment,
            )

        if isinstance(binding, InlineModelBinding):
            engine_declared = "engine" in binding.config
            raw_engine = binding.config.get("engine")
            if engine_declared and (not isinstance(raw_engine, str) or not raw_engine):
                raise TypeError(
                    f"Inline binding '{binding.binding_id}' engine must be a "
                    "non-empty string"
                )
            if binding.dialect_id is None:
                raise ValueError(f"Model binding '{binding.binding_id}' has no dialect")
            self._external_endpoint(binding.config, binding.dialect_id)
            engine_id = raw_engine if engine_declared else "unknown"
            assert isinstance(engine_id, str)
            return DeploymentReconcileInput(
                root_deployment_key=binding.root_deployment_key,
                engine_id=engine_id,
            )

        chain = self._model_config_chain(binding.config_name, models_cfg)
        root_name, root_config = chain[0]
        raw_plan = (self._infer_plans or {}).get(root_name)
        projection = self._plan_projection(raw_plan) if raw_plan is not None else None
        engine_declared = "engine" in root_config
        raw_engine = root_config.get("engine")
        if engine_declared and (not isinstance(raw_engine, str) or not raw_engine):
            raise TypeError(f"Model '{root_name}' engine must be a non-empty string")
        if projection is not None:
            if engine_declared and raw_engine != projection.engine_id:
                raise ValueError(
                    f"Model '{root_name}' engine assertion {raw_engine!r} does not "
                    f"match normalized deployment engine {projection.engine_id!r}"
                )
            engine_id = projection.engine_id
        else:
            engine_id = raw_engine if engine_declared else "unknown"
            assert isinstance(engine_id, str)
            if root_name not in self._realized_deployments:
                if binding.dialect_id is None:
                    raise ValueError(
                        f"Model binding '{binding.binding_id}' has no dialect"
                    )
                self._external_endpoint(root_config, binding.dialect_id)
        return DeploymentReconcileInput(
            root_deployment_key=binding.root_deployment_key,
            engine_id=engine_id,
            plan=projection,
            # Recipe/profile ownership is not recoverable from the already
            # merged DeploymentPlan.  Keep that column empty until #27/#59
            # supplies a typed projection; never mislabel effective values.
            recipe_parameters={},
            explicit_parameters=self._explicit_engine_parameters(
                root_name, root_config
            ),
        )

    def _setup_prelaunch_reconciliation(self) -> None:
        """Resolve and reconcile every task/model binding before client creation."""

        self._request_seed_decisions_frozen = False
        self._request_seed_decisions_by_binding = {}
        self._request_seed_decisions_by_external_role = {}
        self._request_seed_decisions_by_candidate = {}
        self.prelaunch_reconcile_result = None
        self.postlaunch_reconcile_result = None
        models_cfg = self._get_named_config_map("models")
        tasks_cfg = self._get_named_config_map("tasks")

        self._normalized_model_bindings = {}
        self._task_requirement_contexts = {}
        self._task_role_model_sources = {}
        named_by_config: dict[str, NamedModelBinding] = {}
        for model_name in models_cfg:
            binding = self._provisional_named_binding(model_name, models_cfg)
            named_by_config[model_name] = binding
            self._normalized_model_bindings[binding.binding_id] = binding

        records: list[TaskModelRequirement] = []
        for task_name, task_cfg in tasks_cfg.items():
            task_spec = task_cfg.get("class")
            if not isinstance(task_spec, str) or not task_spec:
                raise ValueError(f"Task '{task_name}' requires 'class' field")
            task_class = resolve_task_class(task_spec)
            model_name = self._task_model_config_name(task_name, task_cfg, models_cfg)
            context = self._requirement_context_for_task(
                task_name,
                task_cfg,
                task_class,
                named_by_config[model_name],
            )
            self._task_requirement_contexts[task_name] = context
            requirement_hook = getattr(task_class, "model_requirements_for", None)
            if not callable(requirement_hook):
                raise TypeError(
                    f"{task_class.__name__} has no model_requirements_for() hook"
                )
            task_records = validate_task_model_requirements(
                task_class,
                context,
                requirement_hook(context),
            )
            for record in task_records:
                existing = self._normalized_model_bindings.get(
                    record.binding.binding_id
                )
                if existing is not None and existing != record.binding:
                    raise ValueError(
                        f"binding id '{record.binding.binding_id}' resolves to "
                        "different normalized bindings"
                    )
                self._normalized_model_bindings[record.binding.binding_id] = (
                    record.binding
                )
                records.append(record)

        # Task hooks intentionally receive dialect-free named bindings.  Their
        # normalized input evidence is then aggregated across the complete
        # inheritance root, and the public resolver is called exactly once for
        # that root.  A derived model cannot independently choose a kind that
        # would disagree with its shared deployment/recipe selection.
        named_by_root: dict[str, list[NamedModelBinding]] = {}
        for binding in named_by_config.values():
            named_by_root.setdefault(binding.root_deployment_key, []).append(binding)

        finalized_named: dict[str, NamedModelBinding] = {}
        model_types_by_root: dict[str, Literal["chat", "gen"]] = {}
        for root_key, root_bindings_list in named_by_root.items():
            root_bindings = tuple(root_bindings_list)
            root_name = self._model_config_chain(
                root_bindings[0].config_name, models_cfg
            )[0][0]
            root_records = (
                record
                for record in records
                if isinstance(record.binding, NamedModelBinding)
                and record.binding.root_deployment_key == root_key
            )
            root_requirements = aggregate_task_requirements(root_records)
            explicit_type = self._explicit_model_type_for_root(
                root_name, root_bindings, models_cfg
            )
            model_type = derive_model_type(root_name, explicit_type, root_requirements)
            model_types_by_root[root_key] = model_type
            for binding in root_bindings:
                finalized = self._finalize_named_binding(
                    binding, model_type, models_cfg
                )
                assert finalized.dialect_id is not None
                validate_model_type_dialect(
                    binding.config_name,
                    model_type,
                    finalized.dialect_id,
                )
                finalized_named[binding.binding_id] = finalized

        def finalized_binding(
            binding: NormalizedModelBinding,
        ) -> NormalizedModelBinding:
            return finalized_named.get(binding.binding_id, binding)

        records = [
            dataclasses.replace(record, binding=finalized_binding(record.binding))
            for record in records
        ]
        self._task_requirement_contexts = {
            task_name: dataclasses.replace(
                context,
                model_bindings={
                    role: finalized_binding(binding)
                    for role, binding in context.model_bindings.items()
                },
            )
            for task_name, context in self._task_requirement_contexts.items()
        }
        self._normalized_model_bindings = {
            binding_id: finalized_binding(binding)
            for binding_id, binding in self._normalized_model_bindings.items()
        }
        self._model_types_by_root = model_types_by_root

        grouped: dict[str, list[TaskModelRequirement]] = {
            binding_id: [] for binding_id in self._normalized_model_bindings
        }
        for record in records:
            grouped.setdefault(record.binding.binding_id, []).append(record)
        aggregated = {
            binding_id: aggregate_task_requirements(group)
            for binding_id, group in grouped.items()
        }

        legacy_bypass: set[str] = set()
        binding_inputs: list[BindingReconcileInput] = []
        deployment_inputs: dict[str, DeploymentReconcileInput] = {}
        for binding_id, binding in self._normalized_model_bindings.items():
            requirements = aggregated[binding_id]
            if binding.dialect_id == "sglang_legacy":
                declarations = self._declarations_for_binding(binding, models_cfg)
                if declarations:
                    raise ValueError(
                        f"Model binding '{binding_id}' uses the temporary "
                        "sglang_legacy bypass and cannot declare canonical "
                        "capabilities before the sglang_native PR-5 binder"
                    )
                legacy_bypass.add(binding_id)
                continue
            if binding.dialect_id is None:
                raise ValueError(f"Model binding '{binding_id}' has no dialect")

            declarations = self._declarations_for_binding(binding, models_cfg)
            self._validate_legacy_capability_surfaces(binding, declarations, models_cfg)

            binding_inputs.append(
                BindingReconcileInput(
                    binding_id=binding.binding_id,
                    root_deployment_key=binding.root_deployment_key,
                    requested_model_id=binding.requested_model_id,
                    dialect_id=binding.dialect_id,
                    requirements=requirements,
                    model_profile=self._model_profile_for(binding),
                    connection_scope=self._connection_scope_for(binding, models_cfg),
                    declarations=declarations,
                    request_intents=self._legacy_request_intents_for(
                        binding, models_cfg
                    ),
                    route_intent=self._route_intent_for(binding, models_cfg),
                )
            )
            deployment = self._deployment_input_for(binding, models_cfg)
            previous = deployment_inputs.get(deployment.root_deployment_key)
            if previous is not None and previous != deployment:
                raise ValueError(
                    f"bindings sharing root '{deployment.root_deployment_key}' "
                    "produced different deployment inputs"
                )
            deployment_inputs[deployment.root_deployment_key] = deployment

        result = reconcile(
            ReconcileBatch(tuple(binding_inputs), deployment_inputs),
            self._serving_reconciler,
        )
        if result.errors:
            details = "\n".join(
                self._format_reconcile_diagnostic(item) for item in result.errors
            )
            raise ValueError(f"Model capability reconciliation failed:\n{details}")
        self._validate_external_runtime_obligations(result)
        for diagnostic in result.diagnostics:
            logger.warning(self._format_reconcile_diagnostic(diagnostic))

        self._task_model_requirements = tuple(records)
        self._aggregated_requirements = aggregated
        self._legacy_bypass_bindings = frozenset(legacy_bypass)
        self._prelaunch_binding_inputs = tuple(binding_inputs)
        self._prelaunch_deployment_inputs = dict(deployment_inputs)
        self._freeze_deterministic_request_seed_decisions()
        self.prelaunch_reconcile_result = result
        self._warn_best_effort_deterministic_roles()

    def _warn_best_effort_deterministic_roles(self) -> None:
        """Warn for inline or borrowed role models outside managed topology."""

        if not self.deterministic:
            return
        warned = self._warned_best_effort_role_bindings
        labels_by_binding: dict[str, list[str]] = {}
        for task_name, context in self._task_requirement_contexts.items():
            for role, binding in context.model_bindings.items():
                if not isinstance(binding, InlineModelBinding | ExternalModelBinding):
                    continue
                if binding.binding_id in warned:
                    continue
                labels_by_binding.setdefault(binding.binding_id, []).append(
                    f"{task_name}.{role}"
                )
        if not labels_by_binding:
            return
        warned.update(labels_by_binding.keys())
        labels = sorted(
            label for values in labels_by_binding.values() for label in values
        )
        _emit_best_effort_deterministic_warning(labels)

    def prepare_prelaunch(self) -> ReconcileResult:
        """Public pure setup seam for launch-before-I/O capability validation."""

        self._setup_prelaunch_reconciliation()
        result = self.prelaunch_reconcile_result
        if result is None:  # defensive: the setup method assigns on success
            raise RuntimeError("pre-launch reconciliation produced no result")
        return result

    @staticmethod
    def _seed_from_mapping(
        values: Mapping[str, object],
        subject: str,
    ) -> tuple[bool, int | None]:
        if "seed" not in values:
            return False, None
        return True, _validated_request_seed(values["seed"], subject)

    def _freeze_deterministic_request_seed_decisions(self) -> None:
        """Freeze every model request-default seed decision after prelaunch."""

        if not self.deterministic:
            self._request_seed_decisions_by_binding = {}
            self._request_seed_decisions_by_external_role = {}
            self._request_seed_decisions_by_candidate = {}
            self._request_seed_decisions_frozen = True
            return

        models_cfg = self._get_named_config_map("models")
        by_binding: dict[str, _RequestSeedDecision] = {}
        for binding in self._normalized_model_bindings.values():
            if isinstance(binding, ExternalModelBinding):
                continue
            dialect_id = binding.dialect_id
            if dialect_id is None:
                raise ValueError(f"Model binding '{binding.binding_id}' has no dialect")
            if isinstance(binding, NamedModelBinding):
                explicit_seed_present = False
                explicit_seed: int | None = None
                for _, config in self._model_config_chain(
                    binding.config_name, models_cfg
                ):
                    raw_args = config.get("args")
                    if raw_args is None:
                        continue
                    if not isinstance(raw_args, Mapping):
                        raise ValueError(
                            f"Model '{binding.config_name}' args must be a dictionary"
                        )
                    present, value = self._seed_from_mapping(
                        raw_args, f"Model '{binding.config_name}' args.seed"
                    )
                    if present:
                        explicit_seed_present = True
                        explicit_seed = value
            else:
                explicit_seed_present, explicit_seed = self._seed_from_mapping(
                    binding.config, f"Inline binding '{binding.binding_id}' seed"
                )
                raw_args = binding.config.get("args")
                if raw_args is not None:
                    if not isinstance(raw_args, Mapping):
                        raise ValueError(
                            f"Inline binding '{binding.binding_id}' args must be "
                            "a mapping"
                        )
                    nested_args = cast(Mapping[str, object], raw_args)
                    nested_present, nested_seed = self._seed_from_mapping(
                        nested_args,
                        f"Inline binding '{binding.binding_id}' args.seed",
                    )
                    if nested_present:
                        explicit_seed_present = True
                        explicit_seed = nested_seed
            by_binding[binding.binding_id] = _resolve_deterministic_request_seed(
                dialect_id=dialect_id,
                explicit_seed_present=explicit_seed_present,
                explicit_seed=explicit_seed,
            )

        by_external_role: dict[str, _RequestSeedDecision] = {}
        by_candidate: dict[str, _RequestSeedDecision] = {}
        for task_name, context in sorted(self._task_requirement_contexts.items()):
            sources = self._task_role_model_sources.get(task_name, {})
            for role, binding in sorted(context.model_bindings.items()):
                if isinstance(binding, ExternalModelBinding):
                    source = sources.get(role)
                    if not isinstance(source, Model):
                        raise ValueError(
                            f"External binding '{binding.binding_id}' lost its "
                            "Model source"
                        )
                    dialect_id = binding.dialect_id
                    if dialect_id is None:
                        raise ValueError(
                            f"External binding '{binding.binding_id}' has no dialect"
                        )
                    present, value = self._seed_from_mapping(
                        source.meta()["default_params"],
                        f"Task '{task_name}' {role} model seed",
                    )
                    decision = _resolve_deterministic_request_seed(
                        dialect_id=dialect_id,
                        explicit_seed_present=present,
                        explicit_seed=value,
                        explicit_provenance=_RequestSeedProvenance.EXTERNAL_MODEL,
                    )
                    by_external_role[f"{task_name}.{role}"] = decision

            candidate = context.model_bindings.get("candidate")
            if candidate is None:
                raise ValueError(f"Task '{task_name}' has no candidate binding")
            if isinstance(candidate, ExternalModelBinding):
                raise ValueError(
                    f"Task '{task_name}' candidate must be a named or inline binding"
                )
            by_candidate[task_name] = _with_task_infer_seed(
                by_binding[candidate.binding_id],
                context.infer_args,
                f"Task '{task_name}' infer_args.seed",
            )

        self._request_seed_decisions_by_binding = by_binding
        self._request_seed_decisions_by_external_role = by_external_role
        self._request_seed_decisions_by_candidate = by_candidate
        self._request_seed_decisions_frozen = True

    @staticmethod
    def _request_seed_contract_entry(
        decision: _RequestSeedDecision,
        binding: NormalizedModelBinding,
    ) -> dict[str, JSONValue]:
        """Project one frozen decision into strict, human-auditable evidence.

        This entry's key set IS the resume-invalidation surface: ``--resume``
        compares the persisted config strictly, so adding or removing a field
        here invalidates every in-flight deterministic result dir. Two fields
        are deliberately kept despite being derivable from their siblings:

        * ``seed_scope`` — implied by ``dialect_id`` only while
          ``sglang_legacy`` is the one engine-scoped dialect; a registered
          ``sglang_native`` binder decouples the two, and rotating the
          contract then would be more expensive than carrying it now.
        * ``explicit_seed_present`` — implied by ``seed_provenance``, and kept
          so an audit can answer "did a human choose this seed?" without
          knowing which provenance values count as explicit.

        ``binding_id`` is omitted for external bindings: it carries a
        per-process identity that would rotate the contract between runs that
        differ only by endpoint. Their stable identity is the outer
        ``task.role`` key plus ``requested_model_id`` and ``dialect_id``.
        """

        entry: dict[str, JSONValue] = {
            "requested_model_id": binding.requested_model_id,
            "dialect_id": decision.dialect_id,
            "request_seed_support": decision.support.value,
            "seed_scope": decision.scope.value,
            "seed_present": decision.seed_present,
            "seed": decision.seed,
            "seed_provenance": decision.provenance.value,
            "explicit_seed_present": decision.explicit_seed_present,
        }
        if not isinstance(binding, ExternalModelBinding):
            entry["binding_id"] = binding.binding_id
        return entry

    def _deterministic_seed_contract(self) -> dict[str, JSONValue]:
        """Return strict evidence for frozen model request-default seeds."""

        if not self._request_seed_decisions_frozen:
            raise RuntimeError(
                "deterministic request-seed decisions must be frozen by "
                "pre-launch reconciliation before stamping the contract"
            )
        bindings = self._normalized_model_bindings
        return {
            "bindings": {
                key: self._request_seed_contract_entry(decision, bindings[key])
                for key, decision in sorted(
                    self._request_seed_decisions_by_binding.items()
                )
            },
            "external_roles": {
                key: self._request_seed_contract_entry(
                    decision,
                    self._task_requirement_contexts[
                        key.rsplit(".", 1)[0]
                    ].model_bindings[key.rsplit(".", 1)[1]],
                )
                for key, decision in sorted(
                    self._request_seed_decisions_by_external_role.items()
                )
            },
            "candidates": {
                task_name: self._request_seed_contract_entry(
                    decision,
                    self._task_requirement_contexts[task_name].model_bindings[
                        "candidate"
                    ],
                )
                for task_name, decision in sorted(
                    self._request_seed_decisions_by_candidate.items()
                )
            },
        }

    def _stamp_deterministic_seed_contract(self) -> None:
        """Put the current resolved seed contract in the strict persisted config."""

        if not self.deterministic:
            if _DETERMINISTIC_SEED_CONTRACT_KEY in self._raw_config:
                raise RuntimeError(
                    "persisted deterministic-seed contract cannot be replayed "
                    "with deterministic mode disabled"
                )
            self._reified_config.pop(_DETERMINISTIC_SEED_CONTRACT_KEY, None)
            return
        resolved = self._deterministic_seed_contract()
        raw_config = cast(Mapping[str, object], self._raw_config)
        if (
            _DETERMINISTIC_SEED_CONTRACT_KEY in raw_config
            and raw_config[_DETERMINISTIC_SEED_CONTRACT_KEY] != resolved
        ):
            raise RuntimeError(
                "persisted deterministic-seed contract no longer matches the "
                "resolved dialect seed policy; use the same sieval version and "
                "dialect declarations, or start from the original source config"
            )
        self._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY] = resolved

    def _source_for_binding(self, binding_id: str) -> object | None:
        """Return the runtime-only config/model source for a role binding."""

        for task_name, context in self._task_requirement_contexts.items():
            sources = self._task_role_model_sources.get(task_name, {})
            for role, binding in context.model_bindings.items():
                if binding.binding_id == binding_id:
                    return sources.get(role)
        return None

    @staticmethod
    def _external_endpoint(config: Mapping[str, object], dialect_id: str) -> str:
        """Resolve one external endpoint without crossing protocol families."""

        endpoint = config.get("api_base")
        args = config.get("args")
        if endpoint is None and isinstance(args, Mapping):
            endpoint = cast(Mapping[str, object], args).get("api_base")
        if endpoint is None:
            connection_family = get_dialect_spec(dialect_id).connection_family
            if connection_family != "openai_sdk":
                raise ValueError(
                    f"Dialect {dialect_id!r} uses connection family "
                    f"{connection_family!r} and requires an explicit api_base "
                    "for an external deployment"
                )
            endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(
                f"api_base for dialect {dialect_id!r} must be a non-empty string"
            )
        normalized = endpoint.rstrip("/")
        if not normalized:
            raise ValueError(
                f"api_base for dialect {dialect_id!r} must be a non-empty string"
            )
        return normalized

    @staticmethod
    def _connection_options(
        config: Mapping[str, object],
        *,
        nested_args: bool,
    ) -> tuple[str | None, int, int | None]:
        """Extract client/pool options without treating them as request defaults."""

        args = config.get("args")
        nested: Mapping[str, object] = (
            cast(Mapping[str, object], args)
            if nested_args and isinstance(args, Mapping)
            else {}
        )
        api_key = config.get("api_key", nested.get("api_key"))
        max_retries = config.get("max_retries", nested.get("max_retries", 3))
        concurrency_limit = config.get(
            "concurrency_limit", nested.get("concurrency_limit")
        )
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if isinstance(concurrency_limit, bool) or (
            concurrency_limit is not None
            and (not isinstance(concurrency_limit, int) or concurrency_limit < 1)
        ):
            raise ValueError("concurrency_limit must be a positive integer")
        return api_key, max_retries, concurrency_limit

    def _configured_deployment_for(
        self,
        binding: NormalizedModelBinding,
        template: DeploymentReconcileInput,
        models_cfg: Mapping[str, dict[str, Any]],
    ) -> Deployment:
        """Build an immutable external deployment when no realized one was supplied."""

        if isinstance(binding, ExternalModelBinding):
            model = self._external_model_for_binding(binding.binding_id)
            if model is None:
                raise ValueError(
                    f"External binding '{binding.binding_id}' lost its live model"
                )
            return model.deployment

        if isinstance(binding, NamedModelBinding):
            chain = self._model_config_chain(binding.config_name, models_cfg)
            root_name, root_config = chain[0]
            realized = self._realized_deployments.get(root_name)
            if realized is not None:
                return realized
            if template.plan is not None:
                raise ValueError(
                    f"Managed model '{root_name}' has a desired deployment plan "
                    "but no realized Deployment handoff"
                )
            config: Mapping[str, object] = root_config
        else:
            source = self._source_for_binding(binding.binding_id)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"Inline binding '{binding.binding_id}' lost its config source"
                )
            config = cast(Mapping[str, object], source)

        if binding.dialect_id is None:
            raise ValueError(f"Model binding '{binding.binding_id}' has no dialect")
        endpoint = self._external_endpoint(config, binding.dialect_id)
        engine_source = "config" if template.engine_id != "unknown" else "unknown"
        return Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine(template.engine_id),
            engine_source=engine_source,
            api_base=endpoint,
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )

    def _setup_postlaunch_reconciliation(self) -> None:
        """Re-run the same batch against realized/configured deployments."""

        prelaunch = self.prelaunch_reconcile_result
        if prelaunch is None:
            raise RuntimeError("post-launch reconciliation requires a pre-launch plan")

        models_cfg = self._get_named_config_map("models")
        configured_roots = {
            self._model_config_chain(name, models_cfg)[0][0] for name in models_cfg
        }
        unknown_realized = set(self._realized_deployments) - configured_roots
        if unknown_realized:
            raise ValueError(
                "realized deployments reference unknown root model configs: "
                + ", ".join(sorted(unknown_realized))
            )

        realized_by_root: dict[str, Deployment] = {}
        postlaunch_inputs: dict[str, DeploymentReconcileInput] = {}
        for binding in self._normalized_model_bindings.values():
            if binding.binding_id in self._legacy_bypass_bindings:
                continue
            template = self._prelaunch_deployment_inputs[binding.root_deployment_key]
            deployment = self._configured_deployment_for(binding, template, models_cfg)
            previous = realized_by_root.get(binding.root_deployment_key)
            if previous is not None and previous != deployment:
                raise ValueError(
                    f"bindings sharing root '{binding.root_deployment_key}' "
                    "resolved to different deployments"
                )
            realized_by_root[binding.root_deployment_key] = deployment

        for root_key, deployment in realized_by_root.items():
            template = self._prelaunch_deployment_inputs[root_key]
            prelaunch_plan = prelaunch.deployment_plans.get(root_key)
            if prelaunch_plan is None:
                raise RuntimeError(
                    f"pre-launch plan missing deployment root {root_key!r}"
                )
            desired_plan = template.effective_plan
            if desired_plan is not None and deployment.plan != desired_plan:
                raise ValueError(
                    f"Realized deployment for root '{root_key}' does not match "
                    "the pre-launch desired plan"
                )
            postlaunch_inputs[root_key] = DeploymentReconcileInput(
                root_deployment_key=root_key,
                # A typed realized deployment may refine an engine that was
                # deliberately unknown before launch.  Known pre-launch
                # identities remain strict in DeploymentReconcileInput.
                engine_id=deployment.engine.engine_id,
                deployment=deployment,
                plan=desired_plan,
                recipe_parameters=template.recipe_parameters,
                explicit_parameters=template.explicit_parameters,
                prelaunch_plan=prelaunch_plan,
            )

        result = reconcile(
            ReconcileBatch(self._prelaunch_binding_inputs, postlaunch_inputs),
            self._serving_reconciler,
        )
        if result.errors:
            details = "\n".join(
                self._format_reconcile_diagnostic(item) for item in result.errors
            )
            raise ValueError(
                f"Post-launch model capability reconciliation failed:\n{details}"
            )
        drifted_bindings = sorted(
            binding_id
            for binding_id, prelaunch_binding in prelaunch.binding_plans.items()
            if (
                (postlaunch_binding := result.binding_plans.get(binding_id)) is None
                or postlaunch_binding.fingerprint != prelaunch_binding.fingerprint
            )
        )
        if drifted_bindings:
            raise RuntimeError(
                "post-launch verification changed pre-launch binding plan(s): "
                + ", ".join(drifted_bindings)
            )
        expected = {
            binding.binding_id
            for binding in self._normalized_model_bindings.values()
            if binding.binding_id not in self._legacy_bypass_bindings
        }
        missing = expected - set(result.runtime_plans)
        if missing:
            raise RuntimeError(
                "post-launch reconciliation produced no RuntimeBindingPlan for: "
                + ", ".join(sorted(missing))
            )

        external_bindings = tuple(
            binding
            for binding in self._normalized_model_bindings.values()
            if isinstance(binding, ExternalModelBinding)
        )
        external_roots = {binding.root_deployment_key for binding in external_bindings}
        drifted_external_roots = sorted(
            root_key
            for root_key in external_roots
            if (
                (before := prelaunch.deployment_plans.get(root_key)) is None
                or (after := result.deployment_plans.get(root_key)) is None
                or before.fingerprint != after.fingerprint
            )
        )
        if drifted_external_roots:
            raise RuntimeError(
                "post-launch reconciliation changed serving evidence or checks "
                "for unchanged external deployment root(s): "
                + ", ".join(drifted_external_roots)
            )
        drifted_external_bindings = sorted(
            binding.binding_id
            for binding in external_bindings
            if (
                (before := prelaunch.runtime_plans.get(binding.binding_id)) is None
                or (after := result.runtime_plans.get(binding.binding_id)) is None
                or before.fingerprint != after.fingerprint
            )
        )
        if drifted_external_bindings:
            raise RuntimeError(
                "post-launch reconciliation changed runtime verification for "
                "unchanged external binding(s): " + ", ".join(drifted_external_bindings)
            )
        self._validate_external_runtime_obligations(result)
        for diagnostic in result.diagnostics:
            logger.warning(self._format_reconcile_diagnostic(diagnostic))

        self._realized_deployments_by_root = realized_by_root
        self.postlaunch_reconcile_result = result

    @staticmethod
    def _request_builder_args(
        config: Mapping[str, object],
    ) -> tuple[dict[str, Any], Any]:
        raw = config.get("args", {})
        if not isinstance(raw, Mapping):
            raise ValueError("model args must be a dictionary")
        args = dict(raw)
        for key in ("api_base", "api_key", "max_retries", "concurrency_limit"):
            args.pop(key, None)
        extra = args.pop("extra", None)
        return args, extra

    def _create_owned_pool(
        self,
        root_key: str,
        deployment: Deployment,
        runtime_plan: RuntimeBindingPlan,
        config: Mapping[str, object],
        *,
        nested_args: bool,
    ) -> ConnectionPool[Any]:
        """Create or reuse one owned pool for one complete connection identity."""

        plan = runtime_plan
        identity = plan.connection_identity
        connection_family = get_dialect_spec(plan.dialect_id).connection_family
        factory = CONNECTION_FACTORY_REGISTRY.get(connection_family)
        if plan.resolved_route.connection_family != connection_family:
            raise ValueError(
                f"runtime route for root '{root_key}' uses connection family "
                f"{plan.resolved_route.connection_family!r}, but dialect "
                f"{plan.dialect_id!r} requires {connection_family!r}"
            )
        if identity.connection_family != connection_family:
            raise ValueError(
                f"runtime identity for root '{root_key}' uses connection family "
                f"{identity.connection_family!r}, but dialect {plan.dialect_id!r} "
                f"requires {connection_family!r}"
            )
        for pool in self._owned_pools.values():
            if pool.identity == identity:
                return pool

        api_key, max_retries, concurrency_limit = self._connection_options(
            config, nested_args=nested_args
        )
        retry_policy = factory.retry_policy(max_retries)
        if identity.retry_policy != retry_policy:
            raise ValueError(
                f"runtime connection identity for root '{root_key}' records "
                f"{identity.retry_policy!r}, but client configuration resolves "
                f"to {retry_policy!r}"
            )
        root_limiters = getattr(self, "_root_shared_limiters", None)
        if root_limiters is None:
            root_limiters = {}
            self._root_shared_limiters = root_limiters
        if root_key not in root_limiters:
            root_limiters[root_key] = (
                anyio.CapacityLimiter(concurrency_limit)
                if concurrency_limit is not None
                else None
            )
        connection = CONNECTION_FACTORY_REGISTRY.create(
            connection_family,
            ConnectionRequest(
                endpoint=identity.endpoint,
                credential=api_key,
                max_retries=max_retries,
            ),
        )
        pool = ConnectionPool(connection, identity, root_limiters[root_key])
        identity_payload = {
            "endpoint": identity.endpoint,
            "connection_family": identity.connection_family,
            "credential_scope": identity.credential_scope,
            "retry_policy": identity.retry_policy,
            "quota_scope": identity.quota_scope,
        }
        encoded = json.dumps(
            identity_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        pool_key = f"{root_key}:{hashlib.sha256(encoded).hexdigest()[:16]}"
        self._owned_pools[pool_key] = pool
        self._realized_deployments_by_root[root_key] = deployment
        return pool

    def _as_legacy_wrapper(
        self,
        model: Model,
        requirements: AggregatedTaskRequirements,
        model_type: Literal["chat", "gen"] | None = None,
    ) -> Model:
        """Use normalized evidence or the deployment root's single type."""

        if model_type is not None:
            input_kind = (
                InputKind.CHAT if model_type == "chat" else InputKind.COMPLETION
            )
        elif len(requirements.input) == 1:
            input_kind = next(iter(requirements.input))
        else:
            runtime_plan = model.runtime_plan
            binding_id = runtime_plan.binding_id if runtime_plan is not None else "?"
            raise ValueError(f"binding '{binding_id}' has no legacy input kind")
        wrapper_type: type[Model] = (
            ChatModel if input_kind is InputKind.CHAT else GenModel
        )
        compat_factory = getattr(model, "as_compat_type", None)
        if not callable(compat_factory):
            raise RuntimeError(
                "core Model.as_compat_type() is required for truthful legacy "
                "agenerate/alogprobs input coercion"
            )
        return cast(Model, compat_factory(wrapper_type))

    def _setup_bound_models(self) -> None:
        """Bind canonical runtime plans while preserving compatibility wrappers."""

        result = self.postlaunch_reconcile_result
        if result is None:
            raise RuntimeError("bound model setup requires post-launch reconciliation")
        if self.deterministic and not self._request_seed_decisions_frozen:
            raise RuntimeError(
                "deterministic request-seed decisions were not frozen before "
                "model binding"
            )
        models_cfg = self._get_named_config_map("models")

        by_root: dict[str, list[NormalizedModelBinding]] = {}
        for binding in self._normalized_model_bindings.values():
            by_root.setdefault(binding.root_deployment_key, []).append(binding)
        for root_key, bindings in by_root.items():
            has_legacy = any(
                binding.binding_id in self._legacy_bypass_bindings
                for binding in bindings
            )
            has_canonical = any(
                binding.binding_id not in self._legacy_bypass_bindings
                for binding in bindings
            )
            if has_legacy and has_canonical:
                raise ValueError(
                    f"deployment root '{root_key}' mixes sglang_legacy and canonical "
                    "dialects; split them into separate root model configs"
                )

        bound_by_binding: dict[str, Model] = {}
        pending_named = {
            binding.config_name: binding
            for binding in self._normalized_model_bindings.values()
            if isinstance(binding, NamedModelBinding)
        }

        # Legacy SGLang stays on its explicit bypass until the PR-5 binder.
        for name, binding in tuple(pending_named.items()):
            if binding.binding_id not in self._legacy_bypass_bindings:
                continue
            chain = self._model_config_chain(name, models_cfg)
            if len(chain) != 1:
                continue
            cfg = chain[0][1]
            args = self._normalize_dict(cfg.get("args"), f"Model '{name}' args")
            if self.deterministic:
                _apply_request_seed_decision_to_args(
                    args,
                    self._request_seed_decisions_by_binding[binding.binding_id],
                )
            if "api_key" in cfg:
                args["api_key"] = cfg["api_key"]
            if "api_base" in cfg:
                args["api_base"] = cfg["api_base"]
            legacy_model = SglangGenModel(model=binding.requested_model_id, **args)
            self.models[name] = legacy_model
            bound_by_binding[binding.binding_id] = legacy_model
            self._owned_legacy_models[binding.root_deployment_key] = legacy_model
            del pending_named[name]

        while pending_named:
            resolved_any = False
            for name, binding in tuple(pending_named.items()):
                cfg = models_cfg[name]
                base_name = cfg.get("base")
                is_root = base_name is None
                if not is_root and base_name not in self.models:
                    continue

                if binding.binding_id in self._legacy_bypass_bindings:
                    assert isinstance(base_name, str)
                    model = self.models[base_name]
                    args = self._normalize_dict(cfg.get("args"), f"Model '{name}' args")
                    concurrency_limit = args.pop("concurrency_limit", None)
                    model = model.with_args(concurrency_limit=concurrency_limit, **args)
                else:
                    runtime_plan = result.runtime_plans[binding.binding_id]
                    deployment = self._realized_deployments_by_root[
                        binding.root_deployment_key
                    ]
                    chain = self._model_config_chain(name, models_cfg)
                    root_config = chain[0][1]
                    pool = self._create_owned_pool(
                        binding.root_deployment_key,
                        deployment,
                        runtime_plan,
                        root_config,
                        nested_args=True,
                    )
                    if is_root:
                        model = Model.bind(deployment, pool, runtime_plan)
                    else:
                        assert isinstance(base_name, str)
                        base_model = self.models[base_name]
                        if base_model.pool.identity == runtime_plan.connection_identity:
                            model = base_model.with_dialect(
                                runtime_plan.dialect_id, runtime_plan
                            )
                        else:
                            inherited_local_limiter = (
                                base_model._limiter
                                if base_model._parent_limiter is not None
                                else None
                            )
                            model = Model.bind(
                                deployment,
                                pool,
                                runtime_plan,
                                local_limiter=inherited_local_limiter,
                            )
                            if base_model._kwargs or base_model.extra:
                                inherited_defaults = cast(
                                    dict[str, Any], base_model._kwargs
                                )
                                model = model.with_args(
                                    extra=base_model.extra or None,
                                    **inherited_defaults,
                                )

                    args, extra = self._request_builder_args(cfg)
                    concurrency_limit = None
                    if not is_root:
                        raw_args = cfg.get("args", {})
                        assert isinstance(raw_args, Mapping)
                        concurrency_limit = raw_args.get("concurrency_limit")
                    if args or extra is not None or concurrency_limit is not None:
                        model = model.with_args(
                            concurrency_limit=cast(int | None, concurrency_limit),
                            extra=cast(dict[str, JSONValue] | None, extra),
                            **args,
                        )
                    model = self._as_legacy_wrapper(
                        model,
                        self._aggregated_requirements[binding.binding_id],
                        self._model_types_by_root[binding.root_deployment_key],
                    )

                if self.deterministic:
                    model = _apply_request_seed_decision_to_model(
                        model,
                        self._request_seed_decisions_by_binding[binding.binding_id],
                    )

                self.models[name] = model
                bound_by_binding[binding.binding_id] = model
                del pending_named[name]
                resolved_any = True

            if not resolved_any:
                unresolved = ", ".join(sorted(pending_named))
                raise ValueError(f"Unable to bind derived models: {unresolved}")

        for binding in self._normalized_model_bindings.values():
            if isinstance(binding, NamedModelBinding | ExternalModelBinding):
                continue
            runtime_plan = result.runtime_plans[binding.binding_id]
            deployment = self._realized_deployments_by_root[binding.root_deployment_key]
            source = self._source_for_binding(binding.binding_id)
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"Inline binding '{binding.binding_id}' lost its config source"
                )
            typed_source = cast(Mapping[str, object], source)
            pool = self._create_owned_pool(
                binding.root_deployment_key,
                deployment,
                runtime_plan,
                typed_source,
                nested_args=True,
            )
            model = Model.bind(deployment, pool, runtime_plan)
            direct_config: dict[str, Any] = dict(typed_source)
            nested = direct_config.pop("args", None)
            if isinstance(nested, Mapping):
                direct_config.update(cast(Mapping[str, Any], nested))
            for key in (
                "model",
                "api_base",
                "api_key",
                "max_retries",
                "concurrency_limit",
                "dialect",
                "service_role",
                "capabilities",
                "engine",
            ):
                direct_config.pop(key, None)
            extra = direct_config.pop("extra", None)
            if direct_config or extra is not None:
                model = model.with_args(
                    extra=cast(dict[str, JSONValue] | None, extra), **direct_config
                )
            if self.deterministic:
                model = _apply_request_seed_decision_to_model(
                    model,
                    self._request_seed_decisions_by_binding[binding.binding_id],
                )
            bound_by_binding[binding.binding_id] = self._as_legacy_wrapper(
                model, self._aggregated_requirements[binding.binding_id]
            )

        role_models: dict[str, dict[str, Model]] = {}
        for task_name, context in self._task_requirement_contexts.items():
            for role, binding in context.model_bindings.items():
                if role in {"candidate", "model"}:
                    continue
                if isinstance(binding, ExternalModelBinding):
                    live_model = self._external_model_for_task_role(
                        task_name,
                        role,
                        binding.binding_id,
                    )
                    runtime_plan = result.runtime_plans[binding.binding_id]
                    rebound = live_model.with_dialect(
                        runtime_plan.dialect_id, runtime_plan
                    )
                    if self.deterministic:
                        rebound = _apply_request_seed_decision_to_model(
                            rebound,
                            self._request_seed_decisions_by_external_role[
                                f"{task_name}.{role}"
                            ],
                        )
                    role_model = self._as_legacy_wrapper(
                        rebound, self._aggregated_requirements[binding.binding_id]
                    )
                else:
                    role_model = bound_by_binding[binding.binding_id]
                role_models.setdefault(task_name, {})[role] = role_model
        self._bound_task_role_models = role_models

    def _setup_models(self) -> None:
        """Bind models from the already-reconciled post-launch plan."""

        if getattr(self, "postlaunch_reconcile_result", None) is None:
            raise RuntimeError(
                "model setup requires post-launch capability reconciliation"
            )
        self._setup_bound_models()

    def _check_over_subscription(self) -> None:
        """Check for over-subscription and warn if detected."""
        # Find all base models (those without parent_limiter)
        base_models = {
            name: model
            for name, model in self.models.items()
            if model._limiter is not None and model._parent_limiter is None
        }

        for base_name, base_model in base_models.items():
            # Find all derived models from this base
            children = [
                (name, m)
                for name, m in self.models.items()
                if getattr(m, "_parent_limiter", None) is base_model._limiter
            ]

            if not children:
                continue

            base_limiter = base_model._limiter
            if base_limiter is None:
                continue
            base_quota = base_limiter.total_tokens
            child_quotas = [
                (name, m._limiter.total_tokens)
                for name, m in children
                if m._limiter is not None
            ]

            if child_quotas:
                total_reserved = sum(quota for _, quota in child_quotas)
                if total_reserved > base_quota:
                    child_info = ", ".join(
                        f"{name}={quota}" for name, quota in child_quotas
                    )
                    logger.warning(
                        "Over-subscription detected for model '{}': "
                        "total quota={}, but derived models reserve "
                        "{} ({}). "
                        "Actual concurrency will be capped at {}.",
                        base_name,
                        base_quota,
                        total_reserved,
                        child_info,
                        base_quota,
                    )

    def _setup_datasets(self) -> None:
        """Initialize all datasets from config."""
        datasets_cfg = self._get_named_config_map("datasets")

        for name, cfg in datasets_cfg.items():
            # Resolve class
            class_spec = cfg.get("class")
            if not class_spec:
                raise ValueError(f"Dataset '{name}' requires 'class' field")

            ds_class = resolve_dataset_class(class_spec)

            # Instantiate dataset
            path = cfg.get("path")
            # Expand ${VAR} so ${SIEVAL_DATA_DIR}/drop resolves; scoped to
            # `datasets.*.path` only to avoid surprising users whose model
            # names or args contain `$`.
            if path is not None:
                path = os.path.expandvars(path)
            init_args = self._normalize_dict(cfg.get("args"), f"Dataset '{name}' args")

            try:
                dataset = ds_class(path, **init_args) if path else ds_class(**init_args)
            except FileNotFoundError as exc:
                # Also catches `datasets.exceptions.DataFilesNotFoundError`.
                from sieval.core.datasets.meta import get_dataset_meta

                try:
                    meta = get_dataset_meta(ds_class)
                except AttributeError:
                    # Undecorated user-custom class: no registered name →
                    # skip the hint (would point at an invalid command).
                    meta = None
                hint = (
                    f"\n\nHint: run `sieval dataset download {meta.name}` first, "
                    f"then retry."
                    if meta is not None
                    else ""
                )
                # Wrap in RuntimeError + chain via `from exc` so callers
                # still see the OSError attrs (.filename, .errno) through
                # __cause__. Reconstructing via `type(exc)(msg)` would
                # discard them via the 1-arg constructor path.
                raise RuntimeError(f"{type(exc).__name__}: {exc}{hint}") from exc

            # Apply operations
            operations = self._normalize_list(
                cfg.get("operations"), f"Dataset '{name}' operations"
            )
            dataset = self._apply_dataset_operations(dataset, operations, name)

            self.datasets[name] = dataset
            logger.info("Created dataset '{}' with class '{}'", name, class_spec)

    def _apply_dataset_operations(
        self,
        dataset: Dataset,
        operations: list[dict],
        dataset_name: str,
    ) -> Dataset:
        """Apply a sequence of operations to a dataset."""
        for op in operations:
            if not isinstance(op, dict) or len(op) != 1:
                raise ValueError(
                    f"Dataset '{dataset_name}': Invalid operation format. "
                    f"Expected dict with single key, got: {op}"
                )

            op_name, op_args_raw = next(iter(op.items()))
            if op_args_raw is None:
                op_args: dict[str, Any] = {}
            elif not isinstance(op_args_raw, dict):
                raise ValueError(
                    f"Dataset '{dataset_name}': Operation '{op_name}' args "
                    "must be a dictionary"
                )
            else:
                op_args = op_args_raw.copy()

            match op_name:
                case "select":
                    raise ValueError(
                        f"Dataset '{dataset_name}': operation 'select' was renamed "
                        f"to 'slice'; update your config."
                    )

                case "slice":
                    num = op_args.get("num", op_args.get("n"))
                    split = op_args.get("split", "test")
                    if num is None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'slice' requires 'num'"
                        )
                    dataset = dataset.slice(num, split=split)
                    logger.debug(
                        "Dataset '{}': sliced to first {} samples", dataset_name, num
                    )

                case "shuffle":
                    seed = op_args.get("seed", 0)
                    split = op_args.get("split", "test")
                    dataset = dataset.shuffle(seed=seed, split=split)
                    logger.debug(
                        "Dataset '{}': shuffled with seed={}",
                        dataset_name,
                        seed,
                    )

                case "repeat":
                    times = op_args.get("times", op_args.get("n"))
                    split = op_args.get("split", "test")
                    if times is None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'repeat' requires 'times'"
                        )
                    dataset = dataset.repeat(times, split=split)
                    logger.debug("Dataset '{}': repeated {} times", dataset_name, times)

                case "filter":
                    by_spec = op_args.get("by")
                    split = op_args.get("split", "test")
                    by = self._resolve_filter_by(by_spec, dataset_name)
                    problems = (
                        check_arg_names(op_args)
                        + check_values_source(op_args)
                        + check_by_digest(op_args)
                    )
                    if problems:
                        raise ValueError(f"Dataset '{dataset_name}': {problems[0]}")
                    values_file = op_args.get("values_file")
                    if values_file is not None:
                        value = self._read_filter_values(
                            values_file, dataset_name, op_args.get(VALUES_DIGEST_KEY)
                        )
                    else:
                        value = op_args["value"]
                    require_all = op_args.get("require_all", False)
                    dataset = dataset.filter(
                        by, value, require_all=require_all, split=split
                    )
                    logger.debug(
                        "Dataset '{}': filtered to {}={}",
                        dataset_name,
                        by_spec,
                        f"<{len(value)} keys from {values_file}>"
                        if values_file is not None
                        else value,
                    )

                case "stratified_sample":
                    by = op_args.get("by")
                    num = op_args.get("num", op_args.get("n"))
                    per_group = op_args.get("per_group")
                    fraction = op_args.get("fraction")
                    min_per_group = op_args.get("min_per_group")
                    if by is None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"requires 'by'"
                        )
                    budgets = [num, per_group, fraction]
                    if sum(budget is not None for budget in budgets) != 1:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"requires exactly one of 'num', 'per_group' or "
                            f"'fraction'"
                        )
                    if per_group is not None and min_per_group is not None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"'min_per_group' cannot be combined with 'per_group'"
                        )
                    # Checked here as well as in the transform so the message
                    # names the offending dataset, like the guards above. `bool`
                    # is an `int` subclass, so `fraction: true` would otherwise
                    # pass the range test and silently keep every row.
                    if fraction is not None and (
                        isinstance(fraction, bool)
                        or not isinstance(fraction, int | float)
                        or not 0 < fraction <= 1
                    ):
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"'fraction' must be a number in the interval (0, 1]; "
                            f"got {fraction!r}"
                        )
                    seed = op_args.get("seed", 0)
                    split = op_args.get("split", "test")
                    dataset = dataset.stratified_sample(
                        by,
                        num=num,
                        per_group=per_group,
                        fraction=fraction,
                        min_per_group=min_per_group,
                        seed=seed,
                        split=split,
                    )
                    if per_group is not None:
                        budget_desc = f"per_group={per_group}"
                    elif fraction is not None:
                        budget_desc = f"fraction={fraction}"
                    else:
                        budget_desc = f"num={num}"
                    if min_per_group is not None:
                        budget_desc += f", min_per_group={min_per_group}"
                    logger.debug(
                        "Dataset '{}': stratified-sampled by '{}' ({}, seed={})",
                        dataset_name,
                        by,
                        budget_desc,
                        seed,
                    )

                case _:
                    # Kept in step with `validation._VALID_OPERATIONS` by
                    # `test_the_unknown_operation_message_lists_every_valid_operation`.
                    raise ValueError(
                        f"Dataset '{dataset_name}': Unknown operation '{op_name}'. "
                        f"Valid operations: filter, repeat, shuffle, slice, "
                        f"stratified_sample"
                    )

        return dataset

    def _resolve_filter_by(self, by_spec: object, dataset_name: str) -> TFilterKey:
        """A ``filter`` operation's ``by``, in any of its three config forms.

        ``by: tag`` is a column, ``by: [tag, lang]`` a composite key, and
        ``by: {callable: 'pkg.module.fn'}`` a derived one — the config spelling
        of the callable a Python caller passes directly.

        The shape check is :func:`~sieval.cli._filter_spec.check_by`, shared with
        ``cli.validation``; only the resolution below is this surface's own.
        """
        problem = check_by(by_spec)
        if problem is not None:
            raise ValueError(f"Dataset '{dataset_name}': {problem}")
        if isinstance(by_spec, str):
            return by_spec
        if isinstance(by_spec, list):
            # cast, not a rebuild: `check_by` above has already established
            # every element is a `str`, which the checker cannot see.
            return cast(list[str], by_spec)
        # `check_by` passed and the two column forms are out, so this is the
        # callable form; `key_function_spec` is what read it there too.
        spec = key_function_spec(by_spec)
        assert spec is not None
        try:
            return resolve_key_function(spec)
        except (ValueError, ImportError, AttributeError) as exc:
            raise ValueError(
                f"Dataset '{dataset_name}': 'filter' 'by' callable "
                f"{spec!r} could not be resolved: {exc}"
            ) from exc

    def _read_filter_values(
        self, values_file: object, dataset_name: str, expected_digest: object = None
    ) -> list:
        """The accepted values a ``filter`` operation's ``values_file`` names.

        Reading happens here rather than in :meth:`Dataset.filter` so the
        transform never learns where a config lives: the same selection stays
        expressible from Python by passing the list directly.

        A ``.json`` file is a list of values, or an object whose *keys* are the
        values (so a map from id to whatever metadata produced the selection can
        be used as-is). Anything else is one value per line, blanks and ``#``
        comments skipped.

        Only the JSON *list* form carries a type: lines are strings, and so are
        an object's keys, so a numeric id column needs the list. A mismatch
        raises rather than passing silently, but reads ``id=['0', '1']`` against
        ``present values: [0, 1]`` — the difference is one pair of quotes.

        *expected_digest* is the ``values_digest`` pinned at load time.
        Re-checking it here closes the window between that read and this one, so
        the digest stored beside the results describes the bytes the run
        selected on, not the bytes that were there when the config was parsed.
        """
        if not isinstance(values_file, str):
            raise ValueError(
                f"Dataset '{dataset_name}': 'filter' 'values_file' must be a "
                f"path; got {values_file!r}"
            )
        path = resolve_values_path(values_file, self.config_path.parent)
        if not path.is_file():
            raise ValueError(
                f"Dataset '{dataset_name}': 'filter' 'values_file' not found: {path}"
            )
        data = path.read_bytes()
        if expected_digest is not None:
            digest = compute_values_digest(data)
            if digest != expected_digest:
                raise ValueError(
                    f"Dataset '{dataset_name}': 'filter' 'values_file' {path} "
                    f"changed while the run was starting ({VALUES_DIGEST_KEY} pinned "
                    f"{expected_digest}, the file is now {digest})"
                )
        text = data.decode("utf-8")
        if path.suffix == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Dataset '{dataset_name}': 'filter' 'values_file' {path} is "
                    f"not valid JSON: {exc}"
                ) from exc
            if isinstance(payload, dict):
                return list(payload)
            if not isinstance(payload, list):
                raise ValueError(
                    f"Dataset '{dataset_name}': 'filter' 'values_file' {path} "
                    f"must hold a JSON list of values or an object keyed by "
                    f"them; got {type(payload).__name__}"
                )
            return payload
        return [
            stripped
            for line in text.splitlines()
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]

    def _setup_tasks(self) -> None:
        """Initialize all tasks from config."""
        if self.deterministic and not self._request_seed_decisions_frozen:
            raise RuntimeError(
                "deterministic request-seed decisions were not frozen before task setup"
            )
        tasks_cfg = self._get_named_config_map("tasks")
        runner_defaults_raw = self.config.get("runner_config", {})
        if not isinstance(runner_defaults_raw, dict):
            raise ValueError("'runner_config' configuration must be a dictionary")
        runner_defaults = runner_defaults_raw

        for task_name, raw_task_cfg in tasks_cfg.items():
            task_cfg = cast(TaskConfigDict, raw_task_cfg)
            # Resolve task class
            task_spec = task_cfg.get("class")
            if not task_spec:
                raise ValueError(f"Task '{task_name}' requires 'class' field")

            task_class = resolve_task_class(task_spec)

            task_args = self._normalize_dict(
                task_cfg.get("args", {}),
                f"Task '{task_name}' args",
            )
            validate_task_config_args(
                task_name,
                task_args,
                task_class=task_class,
            )

            # Resolve dataset
            dataset = self._resolve_task_dataset(task_cfg, task_name)

            # RequirementContext is the prelaunch-frozen source for both the
            # candidate identity and every task-local inference default. Never
            # resolve ``task_cfg["model"]`` from mutable config a second time:
            # capability reconciliation and the deterministic seed contract
            # were both derived from this exact binding.
            try:
                context = self._task_requirement_contexts[task_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Task '{task_name}' has no frozen requirement context"
                ) from exc
            candidate = context.model_bindings.get("candidate")
            if not isinstance(candidate, NamedModelBinding):
                raise RuntimeError(
                    f"Task '{task_name}' has no frozen named candidate binding"
                )
            try:
                model = self.models[candidate.config_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Task '{task_name}' frozen candidate model "
                    f"'{candidate.config_name}' was not bound"
                ) from exc

            infer_args = dict(context.infer_args)
            if infer_args:
                model = model.with_args(**cast(dict[str, Any], infer_args))
            if self.deterministic:
                # The frozen candidate decision already folded in this task's
                # ``infer_args.seed``, so it is the single authority for the
                # final seed and overrides whatever landed above.
                model = _apply_request_seed_decision_to_model(
                    model,
                    self._request_seed_decisions_by_candidate[task_name],
                )
            if infer_args:
                logger.info(
                    "Task '{}': applied infer_args override {}",
                    task_name,
                    infer_args,
                )

            # Create task instance
            models_by_role = getattr(self, "_bound_task_role_models", {}).get(task_name)
            if models_by_role is not None:
                # Role-aware tasks reject two competing sources. Post-launch
                # composition supplies reconciled models here, so raw inline
                # configs may no longer construct hidden clients in Task.__init__.
                for role in models_by_role:
                    task_args.pop(role, None)
                task_args["models_by_role"] = models_by_role

            task = task_class(
                name=task_name,
                dataset=dataset,
                model=model,
                **task_args,
            )

            # Build runner config
            runner_config = self._build_runner_config(task_cfg, runner_defaults)

            assert self.runner is not None, "Runner not initialized"
            self.runner.add_task(task, config=runner_config)
            logger.info("Added task '{}' with class '{}'", task_name, task_spec)

    def _resolve_task_dataset(
        self, task_cfg: TaskConfigDict, task_name: str
    ) -> Dataset:
        """Resolve dataset for a task - either by reference or inline definition."""
        # Option 1: Reference to pre-defined dataset
        dataset_ref = task_cfg.get("dataset")
        if isinstance(dataset_ref, str):
            if dataset_ref not in self.datasets:
                raise ValueError(
                    f"Task '{task_name}' references unknown dataset '{dataset_ref}'"
                )
            return self.datasets[dataset_ref]

        # Option 2: Inline dataset definition
        if isinstance(dataset_ref, dict):
            class_spec = dataset_ref.get("class")
            if not class_spec:
                raise ValueError(
                    f"Task '{task_name}': inline dataset requires 'class' field"
                )

            ds_class = resolve_dataset_class(class_spec)
            path = dataset_ref.get("path")
            # Mirror the top-level expansion so inline `tasks.*.dataset.path`
            # resolves `${SIEVAL_DATA_DIR}` identically.
            if path is not None:
                path = os.path.expandvars(path)
            init_args = self._normalize_dict(
                dataset_ref.get("args"), f"Task '{task_name}' inline dataset args"
            )
            dataset = ds_class(path, **init_args) if path else ds_class(**init_args)

            # Apply operations if any
            operations = self._normalize_list(
                dataset_ref.get("operations"),
                f"Task '{task_name}' inline dataset operations",
            )
            dataset = self._apply_dataset_operations(
                dataset, operations, f"{task_name}.dataset"
            )

            return dataset

        raise ValueError(
            f"Task '{task_name}': 'dataset' must be a string reference or inline definition"  # noqa: E501
        )

    def _build_runner_config(
        self, task_cfg: TaskConfigDict, defaults: dict[str, Any]
    ) -> TaskRunnerConfig:
        """Build TaskRunnerConfig from task config and defaults."""
        # Start with defaults
        cfg_dict = dict(defaults)

        # Override with task-specific config
        task_runner_cfg = self._normalize_dict(
            task_cfg.get("runner_config"), "Task 'runner_config'"
        )
        cfg_dict.update(task_runner_cfg)

        # Handle resume override
        if self.resume_override:
            cfg_dict["auto_resume"] = True

        # Filter to valid TaskRunnerConfig fields
        valid_fields = set(TaskRunnerConfig.__dataclass_fields__.keys())
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

        return TaskRunnerConfig(**cfg_dict)

    async def _prepare_execution(self) -> None:
        """Asynchronous preparation pipeline."""
        logger.info("Loading config from: {}", self.config_path)

        self._init_runner()

        # Resolve task-side requirements and fail before any model/client is
        # constructed.  This is pure setup work; managed runs re-run the same
        # reconciliation after launch with realized Deployment/ServingFacts.
        if self.prelaunch_reconcile_result is None:
            await run_sync(self.prepare_prelaunch)
        await run_sync(self._setup_postlaunch_reconciliation)

        # Wrap in to_thread: dataset download / heavy model init can block the
        # event loop otherwise.
        await run_sync(self._setup_models)
        await run_sync(self._check_over_subscription)
        await run_sync(self._setup_datasets)
        await run_sync(self._setup_tasks)

        assert self.runner is not None, "Runner not initialized"
        logger.info("Starting {} tasks", len(self.runner._runners))

    def _resolve_result_dir(self) -> str | None:
        """Resolve target result_dir before ``_init_runner`` has run.

        Persistence + resume checks fire before the runner is constructed,
        so they can't read ``self.result_dir``.
        """
        return self.result_dir_override or self._raw_config.get("result_dir")

    async def _persist_yaml_with_strict_resume(
        self,
        *,
        target_name: str,
        body: str,
        header: str,
        audit_label: str,
        mutable_strip: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Atomically write ``header + body`` to ``result_dir/target_name``.

        Under ``--resume`` with an existing file: an identical body skips the
        rewrite (timestamps survive). With ``mutable_strip=None`` (e.g. infer
        plans) any other diff raises. Otherwise both bodies are parsed and
        compared with ``mutable_strip`` applied; a diff that vanishes
        (resume-mutable or formatting) is tolerated — the file is rewritten
        with the new body, and the original header gains an appended
        ``Resumed by …`` record of what changed — and any residual diff
        raises. ``RuntimeError`` is the only failure the caller observes; all
        else is best-effort and logged.
        """
        effective_result_dir = self._resolve_result_dir()
        if effective_result_dir is None:
            logger.warning(
                "No result_dir configured; skipping {} persistence", target_name
            )
            return

        result_path = anyio.Path(effective_result_dir)
        target = result_path / target_name

        write_header = header

        if self.resume_override and await target.exists():
            try:
                existing = await target.read_text(encoding="utf-8")
            except OSError as e:
                raise RuntimeError(
                    f"Resume aborted: cannot read existing {target}: {e}\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Ensure {target} is readable"
                ) from e

            existing_header, existing_body = _split_header(existing)

            if existing_body == body:
                logger.info("Resume: {} matches — skipping rewrite", target_name)
                return

            if mutable_strip is None:
                # Byte-for-byte strict (e.g. infer plans): any diff aborts.
                raise RuntimeError(
                    f"Resume aborted: {target} does not match current invocation.\n"
                    f"{_brief_diff(existing_body, body)}\n"
                    f"{await _cross_version_resume_hint(result_path)}"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Match the invocation to the persisted {audit_label}"
                )

            # Parse both sides so YAML type coercion (tuple→list) and key
            # ordering can't cause a spurious mismatch.
            try:
                existing_cfg = yaml.safe_load(existing_body) or {}
                current_cfg = yaml.safe_load(body) or {}
            except yaml.YAMLError as e:
                raise RuntimeError(
                    f"Resume aborted: cannot parse existing {target} to verify "
                    f"match: {e}\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Restore {target} to valid YAML matching the persisted "
                    f"{audit_label}"
                ) from e

            # current_cfg is always a dict (we dump a mapping); a tampered file
            # may parse to a scalar/list. Refuse cleanly so mutable_strip can't
            # raise an opaque AttributeError instead of the documented RuntimeError.
            if not isinstance(existing_cfg, dict) or not isinstance(current_cfg, dict):
                raise RuntimeError(
                    f"Resume aborted: existing {target} is not a YAML mapping — "
                    "cannot verify match.\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Restore {target} to valid YAML matching the persisted "
                    f"{audit_label}"
                )

            stripped_existing = mutable_strip(existing_cfg)
            stripped_current = mutable_strip(current_cfg)
            if stripped_existing != stripped_current:
                raise RuntimeError(
                    f"Resume aborted: {target} does not match current invocation.\n"
                    f"{_diff_dicts(stripped_existing, stripped_current)}\n"
                    f"{await _cross_version_resume_hint(result_path)}"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Match the invocation to the persisted {audit_label}"
                )

            # Only resume-mutable (or formatting) fields differ — rewrite with
            # the new body. When real fields changed (not just formatting) and
            # the file had a header, append a timestamped record of the change
            # so the header keeps the full resume lineage. Otherwise no note is
            # added: a formatting-only diff keeps the original header, and a
            # header-less file gets a fresh one (it had no lineage to extend).
            logger.info(
                "Resume: {} resume-mutable fields changed — updating file",
                target_name,
            )
            # The note records genuine resume-mutable changes only; result_dir is
            # a never-compared location field (reification injects it), so drop it
            # to keep it out of the audit trail.
            note_before = {k: v for k, v in existing_cfg.items() if k != "result_dir"}
            note_after = {k: v for k, v in current_cfg.items() if k != "result_dir"}
            change_lines = _diff_lines(note_before, note_after)
            if existing_header and change_lines:
                write_header = _append_resume_note(existing_header, change_lines)
            else:
                write_header = existing_header or header

        tmp_path = target.with_name(target.name + ".tmp")
        content = write_header + body

        try:
            await result_path.mkdir(parents=True, exist_ok=True)
            async with await anyio.open_file(tmp_path, "w", encoding="utf-8") as f:
                await f.write(content)
            await tmp_path.replace(target)
            logger.info("Saved {} to: {}", audit_label, target)
        except Exception as e:
            with contextlib.suppress(OSError):
                await tmp_path.unlink(missing_ok=True)
            logger.error("Failed to save {}: {}", target_name, e)

    async def _persist_effective_config(self) -> None:
        """Write effective_config.yaml to result_dir at session start.

        Dumps ``self._reified_config`` (raw YAML + CLI reification + the
        resolved deterministic-seed contract, minus legacy endpoint-adapter
        injection). User-supplied ``api_base`` / ``api_key`` in the source
        YAML ARE preserved. Runtime endpoints are excluded, so ``sieval run
        <this file>`` can re-launch services from the preserved ``path`` /
        ``infer`` fields.
        """
        body = yaml.safe_dump(
            self._reified_config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        extra_lines = [
            "Reproduce:",
            "  sieval run <this file>",
            "    — universal; re-launches auto-served models",
            "  sieval eval <this file>",
            "    — only when every model already has api_base",
        ]
        if relative := relative_values_files(self._reified_config):
            extra_lines += [
                "",
                "Relative filter values_file: " + ", ".join(relative),
                "  resolves against the config being run, so reproduce from",
                "  beside the source config above, or make the path absolute.",
            ]
        header = _format_comment_header(
            title="Persisted by",
            source_config=str(self.config_path.resolve()),
            invocation=self.invocation,
            extra_lines=extra_lines,
        )
        await self._persist_yaml_with_strict_resume(
            target_name="effective_config.yaml",
            body=body,
            header=header,
            audit_label="effective config",
            mutable_strip=_strip_noncomparable_fields,
        )

    async def _persist_infer_plans(self) -> None:
        """Write infer_plans.yaml to result_dir when the caller supplied plans.

        Rerun re-resolves plans from the ``infer:`` section of
        effective_config.yaml plus the installed sieval version, so this file
        is audit-only — not load-bearing for rerun. Under ``--resume`` it IS
        still part of the strict-match contract: a re-resolved plan that
        differs from the persisted one (different GPU fleet, different sieval
        version emitting a different recipe translation) means resuming would
        merge results from incompatible deployments.
        """
        if not self._infer_plans:
            return

        payload = {"models": dict(self._infer_plans)}
        body = yaml.safe_dump(
            payload,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        header = _format_comment_header(
            title="Persisted by",
            source_config=str(self.config_path.resolve()),
            invocation=self.invocation,
            extra_lines=[
                "Reference only: audit log of the DeploymentPlan used",
                "  for each served model. Re-resolved at runtime from",
                "  effective_config.yaml + installed sieval version.",
            ],
        )
        await self._persist_yaml_with_strict_resume(
            target_name="infer_plans.yaml",
            body=body,
            header=header,
            audit_label="infer plans",
        )

    async def arun(self) -> dict[str, Any]:
        """Run all configured tasks asynchronously."""
        try:
            # Pure prelaunch resolution stamps the binding-local seed contract
            # into effective_config before the strict resume comparison. No
            # model client, dataset, or serving I/O has happened at this point;
            # pure config/requirement errors may therefore surface first.
            await run_sync(self.prepare_prelaunch)
            self._stamp_deterministic_seed_contract()
            await self._persist_effective_config()
            await self._persist_infer_plans()
            await self._prepare_execution()
            if self.runner is None:
                raise RuntimeError("Runner not initialized")
            return await self.runner.arun()
        finally:
            await self._close_owned_model_resources()

    async def _close_owned_model_resources(self) -> None:
        """Drain and close every session-owned root once; never close externals."""

        owned_pools = getattr(self, "_owned_pools", {})
        owned_legacy_models = getattr(self, "_owned_legacy_models", {})
        root_shared_limiters = getattr(self, "_root_shared_limiters", {})
        pools = tuple({id(pool): pool for pool in owned_pools.values()}.values())
        legacy_models = tuple(
            {id(model): model for model in owned_legacy_models.values()}.values()
        )
        owned_pools.clear()
        owned_legacy_models.clear()
        root_shared_limiters.clear()
        first_error: BaseException | None = None
        for pool in pools:
            try:
                await pool.aclose()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        for model in legacy_models:
            try:
                await model.aclose()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def run(self) -> dict[str, Any]:
        """Run all configured tasks synchronously (blocking)."""
        return anyio.run(self.arun)


async def arun_session(
    config: str | Path,
    model: str | None = None,
    resume: bool = False,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    endpoint_map: Mapping[str, str] | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
    realized_deployments: Mapping[str, Deployment] | None = None,
) -> dict[str, Any]:
    """Run tasks defined in a YAML configuration file asynchronously.

    Args:
        config: Path to the YAML configuration file.
        model: Override model name for all base models.
        resume: Enable auto-resume for all tasks.
        result_dir: Override result directory.
        deterministic: Monotone override. ``None`` defers to YAML, ``True``
            forces on, ``False`` is a no-op (cannot downgrade YAML).
        endpoint_map: Legacy external adapter for callers that only have a
            ``{model_name: endpoint_url}`` mapping. It is injected at runtime,
            is not persisted to effective_config.yaml, and cannot be combined
            with ``realized_deployments``. Internal orchestration must pass a
            typed deployment instead.
        infer_plans: ``{model_name: DeploymentPlan-dict}`` for audit-level
            persistence to infer_plans.yaml.
        invocation: Provenance string for audit headers. ``None`` falls back
            to ``sys.argv`` at ``EvalSession.__init__`` time.
        self_managed_endpoints: Names of models whose ``api_base`` points at
            a sieval-launched engine — scopes the best-effort deterministic
            warning to genuinely external endpoints.
        realized_deployments: Typed realized deployments keyed by root model
            config name.

    Returns:
        A dictionary mapping task names to their reports.
    """
    runner = EvalSession(
        config_path=config,
        model_override=model,
        resume=resume,
        result_dir_override=result_dir,
        deterministic_override=deterministic,
        endpoint_map=endpoint_map,
        infer_plans=infer_plans,
        invocation=invocation,
        self_managed_endpoints=self_managed_endpoints,
        realized_deployments=realized_deployments,
    )

    return await runner.arun()


def run_session(
    config: str | Path,
    model: str | None = None,
    resume: bool = False,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    endpoint_map: Mapping[str, str] | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
    realized_deployments: Mapping[str, Deployment] | None = None,
) -> dict[str, Any]:
    return anyio.run(
        arun_session,
        config,
        model,
        resume,
        result_dir,
        deterministic,
        endpoint_map,
        infer_plans,
        invocation,
        self_managed_endpoints,
        realized_deployments,
    )
