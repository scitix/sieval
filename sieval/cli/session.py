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
import re
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast

import anyio
import httpx
import yaml
from anyio.to_thread import run_sync
from loguru import logger

from sieval import __version__
from sieval.cli._cli_model import CliModel, apply_cli_model
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
from sieval.cli.resolution import (
    TASK_MODEL_ROLES,
    binding_resource_argument_paths,
    derive_model_type,
    finalize_inline_model_binding,
    is_task_model_role_sentinel,
    model_config_extra,
    normalize_inline_model_binding,
    resolve_dataset_class,
    resolve_key_function,
    resolve_task_class,
    validate_configured_engine_id,
    validate_model_type_dialect,
    validate_named_config_map,
    validate_omitted_dialect_migration,
    validate_task_config_args,
    validate_task_model_requirements,
)
from sieval.cli.services import RESERVED_LOG_DIR, vendored_services_for
from sieval.core.datasets import Dataset, TFilterKey
from sieval.core.datasets.meta import get_dataset_meta
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
    DEFAULT_REQUEST_TIMEOUT,
    MANAGED_ENGINE_REQUEST_TIMEOUT,
    ConnectionRequest,
    resolve_client_timeout,
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
    get_dialect_spec,
)
from sieval.core.models.reconcile import (
    BindingCapabilityPlan,
    BindingReconcileInput,
    CannotVerify,
    CheckStage,
    Configured,
    ConnectionScope,
    DeferredCheck,
    DeploymentCapabilityPlan,
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
    LegacyStageMatch,
    MultiTaskRunner,
    ResumeValue,
    RunPlan,
    TaskRunnerConfig,
    as_resume,
    resume_source,
    same_dir,
)
from sieval.core.tasks.context import TaskAction
from sieval.core.tasks.services import check_service_names
from sieval.core.types import JSONValue
from sieval.core.utils.hf import maybe_resolve_hf_path
from sieval.core.utils.paths import resolve_data_dir
from sieval.datasets.downloaders.staged import probe_source, staged_dir
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

#: Verifier for ``DialectSpec.provider_enforced_capabilities``.
_PROVIDER_ENFORCED_VERIFIER = "provider_enforced"

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


def _record_provenance_replacement(
    replacements: dict[str, str],
    ambiguous: set[str],
    runtime_value: str,
    stable_value: str,
) -> None:
    """Record a replacement, retaining collisions for fail-loud evidence checks."""

    if runtime_value == stable_value:
        return
    if runtime_value in ambiguous:
        return
    previous = replacements.get(runtime_value)
    if previous is not None and previous != stable_value:
        del replacements[runtime_value]
        ambiguous.add(runtime_value)
        return
    replacements[runtime_value] = stable_value


def _record_plan_provenance_replacements(
    replacements: dict[str, str],
    ambiguous: set[str],
    runtime_plan: RuntimeBindingPlan,
    provenance_plan: RuntimeBindingPlan,
) -> None:
    """Map every runtime-only plan identity that can appear in evidence."""

    pairs = (
        (runtime_plan.binding_id, provenance_plan.binding_id),
        (runtime_plan.root_deployment_key, provenance_plan.root_deployment_key),
        (runtime_plan.fingerprint, provenance_plan.fingerprint),
        (
            runtime_plan.verification_fingerprint,
            provenance_plan.verification_fingerprint,
        ),
        (
            runtime_plan.binding_plan_fingerprint,
            provenance_plan.binding_plan_fingerprint,
        ),
        (
            runtime_plan.deployment_plan_fingerprint,
            provenance_plan.deployment_plan_fingerprint,
        ),
        (
            runtime_plan.connection_identity.credential_scope,
            provenance_plan.connection_identity.credential_scope,
        ),
        (
            runtime_plan.connection_identity.quota_scope,
            provenance_plan.connection_identity.quota_scope,
        ),
    )
    for runtime_value, stable_value in pairs:
        _record_provenance_replacement(
            replacements,
            ambiguous,
            runtime_value,
            stable_value,
        )


def _replace_provenance_reference_text(
    value: str,
    replacements: Mapping[str, str],
) -> str:
    """Replace identity references in a field whose schema declares them.

    This helper is deliberately limited to provenance ``sources`` fields.  In
    those fields, every occurrence of a known runtime token is an identity
    reference by contract.  Arbitrary JSON, diagnostic text, mapping keys, and
    user parameters must instead remain byte-for-byte unchanged and fail loud
    if they contain a volatile token.
    """

    if not replacements:
        return value
    pattern = re.compile(
        "|".join(
            re.escape(runtime_value)
            for runtime_value in sorted(
                replacements,
                key=lambda runtime_value: len(runtime_value),
                reverse=True,
            )
        )
    )
    return pattern.sub(lambda match: replacements[match.group(0)], value)


def _provenance_tokens_in(
    value: JSONValue,
    runtime_tokens: frozenset[str],
) -> frozenset[str]:
    """Return volatile identity tokens present in unprojected JSON data."""

    found: set[str] = set()

    def visit(current: JSONValue) -> None:
        if isinstance(current, str):
            found.update(token for token in runtime_tokens if token in current)
            return
        if isinstance(current, list):
            for item in current:
                visit(item)
            return
        if isinstance(current, Mapping):
            # isinstance erases the type arguments; widening visit() to object
            # instead would need a non-str key branch that can never run.
            typed_mapping = cast(Mapping[str, JSONValue], current)
            for key, item in typed_mapping.items():
                found.update(token for token in runtime_tokens if token in key)
                visit(item)

    visit(value)
    return frozenset(found)


def _reject_unprojected_provenance_tokens(
    value: JSONValue,
    runtime_tokens: frozenset[str],
    *,
    context: str,
) -> None:
    """Reject volatile identities in fields that provenance must not rewrite."""

    leaked = sorted(_provenance_tokens_in(value, runtime_tokens))
    if leaked:
        raise ValueError(
            f"{context} contains runtime identity token(s) in semantic data: "
            + ", ".join(leaked)
        )


# Deliberately empty: no DeferredCheck field is projected, so a runtime token
# in one fails loud instead of being rewritten.
_EXTERNAL_PROVENANCE_PROJECTED_CHECK_FIELDS: frozenset[str] = frozenset()
_EXTERNAL_PROVENANCE_SEMANTIC_CHECK_FIELDS = frozenset(
    {"capability", "stage", "verifier", "reason"}
)
_EXTERNAL_PROVENANCE_PROJECTED_REQUIREMENT_FIELDS = frozenset({"sources"})
_EXTERNAL_PROVENANCE_SEMANTIC_REQUIREMENT_FIELDS = frozenset(
    {"capability", "minimums", "verifier", "reason"}
)
_EXTERNAL_PROVENANCE_PROJECTED_INTENT_FIELDS = frozenset({"sources"})
_EXTERNAL_PROVENANCE_SEMANTIC_INTENT_FIELDS = frozenset(
    {"key", "required", "minimums", "request_defaults"}
)
_EXTERNAL_PROVENANCE_PROJECTED_CONNECTION_SCOPE_FIELDS = frozenset(
    {"credential_scope", "quota_scope"}
)
_EXTERNAL_PROVENANCE_SEMANTIC_CONNECTION_SCOPE_FIELDS = frozenset({"retry_policy"})
_EXTERNAL_PROVENANCE_PROJECTED_BINDING_PLAN_FIELDS = frozenset(
    {"binding_id", "root_deployment_key"}
)
_EXTERNAL_PROVENANCE_SEMANTIC_BINDING_PLAN_FIELDS = frozenset(
    {
        "requested_model_id",
        "dialect_id",
        "declared_capabilities",
        "required_capabilities",
        "available_capabilities",
        "pending_capabilities",
        "capability_minimums",
        "request_defaults",
        "output_channels",
        "required_output_channels",
        "route_intent",
        "unavailable_capabilities",
    }
)
_EXTERNAL_PROVENANCE_SPECIAL_BINDING_PLAN_FIELDS = frozenset(
    {"intents", "serving_requirements", "connection_scope"}
)
_EXTERNAL_PROVENANCE_COMPUTED_BINDING_PLAN_FIELDS = frozenset({"fingerprint"})
_EXTERNAL_PROVENANCE_PROJECTED_DEPLOYMENT_PLAN_FIELDS = frozenset(
    {"root_deployment_key"}
)
_EXTERNAL_PROVENANCE_SEMANTIC_DEPLOYMENT_PLAN_FIELDS = frozenset(
    {
        "engine_id",
        "desired_plan_fingerprint",
        "recipe_parameters",
        "explicit_parameters",
        "launch_patch",
        "setup_checks",
        "request_checks",
        "outcome_kinds",
    }
)
_EXTERNAL_PROVENANCE_SPECIAL_DEPLOYMENT_PLAN_FIELDS = frozenset(
    {"serving_requirements", "outcome_evidence"}
)
_EXTERNAL_PROVENANCE_COMPUTED_DEPLOYMENT_PLAN_FIELDS = frozenset({"fingerprint"})
_EXTERNAL_PROVENANCE_PROJECTED_RECONCILE_INPUT_FIELDS = frozenset(
    {"root_deployment_key"}
)
_EXTERNAL_PROVENANCE_SEMANTIC_RECONCILE_INPUT_FIELDS = frozenset(
    {"engine_id", "deployment", "plan", "recipe_parameters", "explicit_parameters"}
)
_EXTERNAL_PROVENANCE_SPECIAL_RECONCILE_INPUT_FIELDS = frozenset({"prelaunch_plan"})


_EXTERNAL_PROVENANCE_POLICIES = (
    (
        DeferredCheck,
        _EXTERNAL_PROVENANCE_PROJECTED_CHECK_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_CHECK_FIELDS,
    ),
    (
        ServingRequirement,
        _EXTERNAL_PROVENANCE_PROJECTED_REQUIREMENT_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_REQUIREMENT_FIELDS,
    ),
    (
        CapabilityIntent,
        _EXTERNAL_PROVENANCE_PROJECTED_INTENT_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_INTENT_FIELDS,
    ),
    (
        ConnectionScope,
        _EXTERNAL_PROVENANCE_PROJECTED_CONNECTION_SCOPE_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_CONNECTION_SCOPE_FIELDS,
    ),
    (
        BindingCapabilityPlan,
        _EXTERNAL_PROVENANCE_PROJECTED_BINDING_PLAN_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_BINDING_PLAN_FIELDS
        | _EXTERNAL_PROVENANCE_SPECIAL_BINDING_PLAN_FIELDS
        | _EXTERNAL_PROVENANCE_COMPUTED_BINDING_PLAN_FIELDS,
    ),
    (
        DeploymentCapabilityPlan,
        _EXTERNAL_PROVENANCE_PROJECTED_DEPLOYMENT_PLAN_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_DEPLOYMENT_PLAN_FIELDS
        | _EXTERNAL_PROVENANCE_SPECIAL_DEPLOYMENT_PLAN_FIELDS
        | _EXTERNAL_PROVENANCE_COMPUTED_DEPLOYMENT_PLAN_FIELDS,
    ),
    (
        DeploymentReconcileInput,
        _EXTERNAL_PROVENANCE_PROJECTED_RECONCILE_INPUT_FIELDS
        | _EXTERNAL_PROVENANCE_SEMANTIC_RECONCILE_INPUT_FIELDS
        | _EXTERNAL_PROVENANCE_SPECIAL_RECONCILE_INPUT_FIELDS,
    ),
)


def _reject_provenance_tokens_in_checks(
    checks: Iterable[DeferredCheck],
    runtime_tokens: frozenset[str],
) -> None:
    """Reject leaked identity in check text, which is never rewritten.

    Fails loud only, so callers pass their original tuple through unchanged.
    """

    for check in checks:
        _reject_unprojected_provenance_tokens(
            check.reason,
            runtime_tokens,
            context="external provenance check reason",
        )


def _project_provenance_requirement(
    requirement: ServingRequirement,
    replacements: Mapping[str, str],
    runtime_tokens: frozenset[str],
) -> ServingRequirement:
    _reject_unprojected_provenance_tokens(
        requirement.minimums,
        runtime_tokens,
        context="external provenance serving minimums",
    )
    if requirement.reason is not None:
        _reject_unprojected_provenance_tokens(
            requirement.reason,
            runtime_tokens,
            context="external provenance serving reason",
        )
    return dataclasses.replace(
        requirement,
        sources=tuple(
            _replace_provenance_reference_text(source, replacements)
            for source in requirement.sources
        ),
    )


def _project_provenance_intent(
    intent: CapabilityIntent,
    replacements: Mapping[str, str],
    runtime_tokens: frozenset[str],
) -> CapabilityIntent:
    """Project diagnostic sources without changing the requested capability."""

    _reject_unprojected_provenance_tokens(
        intent.minimums,
        runtime_tokens,
        context="external provenance intent minimums",
    )
    _reject_unprojected_provenance_tokens(
        intent.request_defaults.to_json_value(),
        runtime_tokens,
        context="external provenance intent request defaults",
    )
    return dataclasses.replace(
        intent,
        sources=tuple(
            _replace_provenance_reference_text(source, replacements)
            for source in intent.sources
        ),
    )


def _project_external_binding_plan(
    plan: BindingCapabilityPlan,
    *,
    binding_id: str,
    root_deployment_key: str,
    connection_scope: ConnectionScope,
    replacements: Mapping[str, str],
    runtime_tokens: frozenset[str],
) -> BindingCapabilityPlan:
    """Build a stable view of every persisted binding-plan field."""

    projected = dataclasses.replace(
        plan,
        binding_id=binding_id,
        root_deployment_key=root_deployment_key,
        intents={
            capability: _project_provenance_intent(
                intent,
                replacements,
                runtime_tokens,
            )
            for capability, intent in plan.intents.items()
        },
        serving_requirements=tuple(
            _project_provenance_requirement(
                requirement,
                replacements,
                runtime_tokens,
            )
            for requirement in plan.serving_requirements
        ),
        connection_scope=connection_scope,
    )
    _reject_unprojected_provenance_tokens(
        projected.to_json_value(),
        runtime_tokens,
        context="external provenance binding plan",
    )
    return projected


def _project_external_deployment_plan(
    plan: DeploymentCapabilityPlan,
    *,
    root_deployment_key: str,
    replacements: Mapping[str, str],
    runtime_to_provenance: Mapping[str, str],
    runtime_tokens: frozenset[str],
) -> DeploymentCapabilityPlan:
    """Build a stable view of every persisted deployment-plan field."""

    outcome_evidence: dict[CapabilityKey, Mapping[str, JSONValue]] = {}
    for capability, evidence in plan.outcome_evidence.items():
        plan_fingerprints = evidence.get("plan_fingerprints")
        projected_evidence: dict[str, JSONValue] = dict(evidence)
        if evidence.get("source") == "external_runtime_plans":
            if not isinstance(plan_fingerprints, list) or any(
                not isinstance(fingerprint, str) for fingerprint in plan_fingerprints
            ):
                raise TypeError(
                    "external runtime plan evidence must contain string "
                    "plan_fingerprints"
                )
            # Lossless after the guard above; carries the element type that
            # isinstance cannot.
            typed_fingerprints = [
                fingerprint
                for fingerprint in plan_fingerprints
                if isinstance(fingerprint, str)
            ]
            missing = sorted(set(typed_fingerprints) - runtime_to_provenance.keys())
            if missing:
                raise ValueError(
                    "external provenance projection cannot resolve runtime plan "
                    "fingerprint(s): " + ", ".join(missing)
                )
            projected_evidence["plan_fingerprints"] = sorted(
                {runtime_to_provenance[item] for item in typed_fingerprints}
            )
        outcome_evidence[capability] = projected_evidence

    _reject_provenance_tokens_in_checks(plan.setup_checks, runtime_tokens)
    _reject_provenance_tokens_in_checks(plan.request_checks, runtime_tokens)
    projected_plan = dataclasses.replace(
        plan,
        root_deployment_key=root_deployment_key,
        serving_requirements=tuple(
            _project_provenance_requirement(
                requirement,
                replacements,
                runtime_tokens,
            )
            for requirement in plan.serving_requirements
        ),
        outcome_evidence=outcome_evidence,
    )
    _reject_unprojected_provenance_tokens(
        projected_plan.to_json_value(),
        runtime_tokens,
        context="external provenance plan",
    )
    return projected_plan


@dataclasses.dataclass(frozen=True)
class _ExternalProvenanceContext:
    root_deployment_key: str
    replacements: Mapping[str, str]
    ambiguous_replacements: frozenset[str]
    runtime_to_provenance: Mapping[str, str]

    @property
    def runtime_tokens(self) -> frozenset[str]:
        return frozenset(self.replacements) | self.ambiguous_replacements


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
                if not request_checks:
                    request_checks = _response_contract_checks(
                        requirement.capability, "external runtime"
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
            if requirement.verifier == _PROVIDER_ENFORCED_VERIFIER:
                outcomes[requirement.capability] = Configured(
                    evidence={"source": _PROVIDER_ENFORCED_VERIFIER},
                    request_checks=_response_contract_checks(
                        requirement.capability, "provider-enforced"
                    ),
                )
                continue
            verifier = _PR1_REQUEST_VERIFIERS.get(requirement.capability)
            if verifier is None or requirement.verifier not in (None, verifier):
                continue
            outcomes[requirement.capability] = CannotVerify(
                CheckStage.REQUEST,
                verifier,
                requirement.reason
                or "legacy compatibility requires response-time verification",
            )
        return outcomes


def _response_contract_checks(
    capability: CapabilityKey, source: str
) -> tuple[DeferredCheck, ...]:
    """Keep a capability with a response validator behind it on every request."""

    verifier = _PR1_REQUEST_VERIFIERS.get(capability)
    if verifier is None:
        return ()
    return (
        DeferredCheck(
            capability,
            CheckStage.REQUEST,
            verifier,
            f"{source} support remains guarded by the existing response contract",
        ),
    )


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
        injected_inputs: Callable[
            [tuple[ServingRequirement, ...], DeploymentReconcileInput],
            tuple[tuple[ServingRequirement, ...], DeploymentReconcileInput],
        ],
    ) -> None:
        self._compatibility = compatibility
        self._injected = injected
        self._injected_inputs = injected_inputs

    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> Mapping[CapabilityKey, ServingOutcome]:
        baseline_outcomes = self._compatibility.reconcile(requirements, deployment)
        injected_requirements, injected_deployment = self._injected_inputs(
            requirements, deployment
        )
        injected_outcomes = dict(
            self._injected.reconcile(injected_requirements, injected_deployment)
        )
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
    api_key_env: str  # Environment variable holding the API key
    api_base: str
    max_retries: int
    client_timeout: dict[str, float]  # connect/read/write/pool seconds
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


#: Keys whose value is a secret. ``api_key_env`` only names a variable.
_SECRET_CONFIG_KEYS = frozenset({"api_key", "authorization"})
_REDACTED = "<redacted>"


def redact_secrets(value: Any) -> Any:
    """A copy of *value* with every secret config value, at any depth, redacted.

    Matched by key name anywhere, so a task arg named ``api_key`` is redacted
    too. Credentials are in no stage identity, so a resume compares the same.
    """
    if isinstance(value, Mapping):
        return {
            k: _REDACTED
            if k in _SECRET_CONFIG_KEYS and v is not None
            else redact_secrets(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def holds_secret(value: Any) -> bool:
    """Whether :func:`redact_secrets` redacts anything in *value*.

    A record written through :func:`redact_secrets` holds the marker in its
    place, so such a record cannot stand in for the value it was written from.
    """
    if isinstance(value, Mapping):
        return any(
            (k in _SECRET_CONFIG_KEYS and v is not None) or holds_secret(v)
            for k, v in value.items()
        )
    if isinstance(value, list):
        return any(holds_secret(item) for item in value)
    return False


def _diff_lines(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return ``- <path>: <old> → <new>`` lines for every differing leaf.

    Walks two parsed-config mappings depth-first. Empty list means the two
    parse to the same structure (any textual difference was whitespace /
    formatting only). Secret values are shown redacted: these lines reach
    ``meta.json``.
    """
    diffs: list[str] = []

    def _shown(key: str, value: Any) -> str:
        if key in _SECRET_CONFIG_KEYS and value is not None:
            return _REDACTED
        return repr(value)

    def _walk(x: Any, y: Any, path: str, key: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                _walk(x.get(k), y.get(k), f"{path}.{k}" if path else k, k)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"- {path}: list length {len(x)} → {len(y)}")
            else:
                for i, (xv, yv) in enumerate(zip(x, y, strict=True)):
                    _walk(xv, yv, f"{path}[{i}]", key)
        elif x != y:
            diffs.append(f"- {path}: {_shown(key, x)} → {_shown(key, y)}")

    _walk(a, b, "", "")
    return diffs


#: Task ``args`` that reach the request, as :meth:`Task.inference_args` reads them.
_INFERENCE_TASK_ARGS = ("n", "stop", "logprobs")
#: Runner keys in the inference identity (``INFERENCE_RUNNER_KEYS``).
_INFERENCE_RUNNER_CONFIG_KEYS = ("max_iterations", "max_retries")


#: A model's credential, by value or by variable name. Neither is an inference
#: input, and both are one credential scope.
_CREDENTIAL_KEYS = ("api_key", "api_key_env")
_ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A launched engine is unauthenticated; the OpenAI SDK still demands a key.
_LAUNCHED_ENGINE_CREDENTIAL = "local"


def _names_credential(config: Mapping[str, object]) -> bool:
    return any(key in config for key in _CREDENTIAL_KEYS)


def _configured_credential(
    config: Mapping[str, object],
    nested: Mapping[str, object],
    subject: str,
    environ: Mapping[str, str],
) -> str | None:
    """The credential ``config`` names, read from ``environ`` for ``api_key_env``.

    A top-level key wins over ``args``. Both forms at once, an unset or empty
    variable, and a redacted key are refused, so a missing key fails before
    any engine starts rather than as a 401.
    """
    api_key = config.get("api_key", nested.get("api_key"))
    env_name = config.get("api_key_env", nested.get("api_key_env"))
    if api_key is not None and env_name is not None:
        raise ValueError(
            f"{subject} sets both 'api_key' and 'api_key_env'; keep one: "
            "'api_key' is the key itself, 'api_key_env' names the environment "
            "variable that holds it"
        )
    if env_name is not None:
        if not isinstance(env_name, str) or not _ENV_VAR_NAME.fullmatch(env_name):
            raise ValueError(
                f"{subject} 'api_key_env' must name an environment variable, "
                f"got {env_name!r}"
            )
        value = environ.get(env_name)
        if not value:
            state = "empty" if value is not None else "unset"
            raise ValueError(
                f"{subject} reads its API key from environment variable "
                f"{env_name}, which is {state}; export {env_name} before the run"
            )
        return value
    if api_key is not None and not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if api_key == _REDACTED:
        raise ValueError(
            f"{subject} has api_key {_REDACTED!r}: this config was persisted "
            "with its key removed. Set 'api_key_env' to the variable holding "
            "the key, or write the key back, before rerunning it"
        )
    return api_key


def _nested_args(config: Mapping[str, object]) -> Mapping[str, object]:
    args = config.get("args")
    return cast(Mapping[str, object], args) if isinstance(args, Mapping) else {}


def _default_client_timeout(deployment: Deployment | None) -> httpx.Timeout:
    """Managed (``sieval run`` launched, ``plan`` set) or external default."""
    return (
        MANAGED_ENGINE_REQUEST_TIMEOUT
        if deployment is not None and deployment.plan is not None
        else DEFAULT_REQUEST_TIMEOUT
    )


def _client_credential(
    credential: str | None, deployment: Deployment | None
) -> str | None:
    """*credential*, or the placeholder for a launched engine that names none.

    External endpoints keep ``None``: the provider SDK reads its env var.
    """
    if credential is None and deployment is not None and deployment.plan is not None:
        return _LAUNCHED_ENGINE_CREDENTIAL
    return credential


def _configured_client_timeout(
    config: Mapping[str, object],
    nested: Mapping[str, object],
    default: httpx.Timeout,
    subject: str,
) -> httpx.Timeout:
    """``config``'s ``client_timeout`` over *default*; a top-level key wins."""
    return resolve_client_timeout(
        config.get("client_timeout", nested.get("client_timeout")),
        default=default,
        subject=subject,
    )


#: Resume keys removed in 0.12, refused rather than dropped: a dropped
#: ``resume_from`` would turn a resume into a fresh run without a word.
_REMOVED_RESUME_KEYS = {"auto_resume": "resume: true", "resume_from": "resume: <dir>"}


def _runner_config_blocks(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The top-level ``runner_config`` and each task's, by where they sit."""
    blocks: dict[str, Any] = {"runner_config": cfg.get("runner_config")}
    tasks = cfg.get("tasks")
    if isinstance(tasks, dict):
        for name, task in tasks.items():
            if isinstance(task, dict):
                blocks[f"tasks.{name}.runner_config"] = task.get("runner_config")
    return {k: v for k, v in blocks.items() if isinstance(v, dict)}


def config_resumes(cfg: Mapping[str, Any]) -> bool:
    """Whether some ``runner_config`` of *cfg* sets ``resume``."""
    return any(block.get("resume") for block in _runner_config_blocks(cfg).values())


def _drop_resume_keys(cfg: dict[str, Any]) -> None:
    """Remove ``resume`` in place, and a ``runner_config`` left empty."""
    for block in _runner_config_blocks(cfg).values():
        block.pop("resume", None)
    if cfg.get("runner_config") == {}:
        del cfg["runner_config"]
    for task in (cfg.get("tasks") or {}).values():
        if isinstance(task, dict) and task.get("runner_config") == {}:
            del task["runner_config"]


def _refuse_removed_resume_keys(cfg: Mapping[str, Any]) -> None:
    """Raise on ``auto_resume`` / ``resume_from``, naming the ``resume`` form."""
    for where, block in _runner_config_blocks(cfg).items():
        for key, use in _REMOVED_RESUME_KEYS.items():
            if key in block:
                raise ValueError(
                    f"{where}.{key} was removed in 0.12; use {where}.{use}"
                )


def read_persisted_config(path: Path) -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _stage_config(cfg: Mapping[str, Any], task_name: str, stage: str) -> object:
    """The part of a session config that decides *stage* for *task_name*.

    For a recorded run without stage identities, compared between its
    persisted config and this run's. ``data``: the task's dataset block,
    resolved by name. ``inference``: the candidate model's config chain without
    endpoint, key or concurrency, the task's ``infer_args`` and inference
    ``args``, the result-changing runner keys and ``deterministic``.
    """
    tasks = cfg.get("tasks") or {}
    task = tasks.get(task_name) or {}
    models = cfg.get("models") or {}
    datasets = cfg.get("datasets") or {}
    if stage == "data":
        ref = task.get("dataset")
        return datasets.get(ref) if isinstance(ref, str) else ref
    model_name = task.get("model")
    if model_name is None and len(models) == 1:
        model_name = next(iter(models))
    chain = []
    seen: set[str] = set()
    while (
        isinstance(model_name, str) and model_name in models and model_name not in seen
    ):
        seen.add(model_name)
        mcfg = dict(models[model_name] or {})
        for key in (
            "api_base",
            "api_key",
            "api_key_env",
            "client_timeout",
            "concurrency_limit",
            "max_retries",
        ):
            mcfg.pop(key, None)
        args = dict(mcfg.get("args") or {})
        for key in ("api_key", "api_key_env", "client_timeout", "concurrency_limit"):
            args.pop(key, None)
        mcfg["args"] = args
        chain.append(mcfg)
        model_name = mcfg.get("base")
    runner = {**(cfg.get("runner_config") or {}), **(task.get("runner_config") or {})}
    task_args = task.get("args") or {}
    return {
        "models": chain,
        "infer_args": task.get("infer_args"),
        "args": {k: task_args.get(k) for k in _INFERENCE_TASK_ARGS},
        "runner": {k: runner.get(k) for k in _INFERENCE_RUNNER_CONFIG_KEYS},
        "deterministic": bool(cfg.get("deterministic", False)),
    }


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


def _refuse_endpoint_with_realized_deployment(
    config: Mapping[str, Any], realized_deployments: Mapping[str, Deployment]
) -> None:
    """Refuse a configured ``api_base`` that differs from the realized one.

    The realized deployment would silently replace it.
    """

    for name, model_cfg in (config.get("models") or {}).items():
        deployment = realized_deployments.get(name)
        if deployment is None or not isinstance(model_cfg, dict):
            continue
        args = model_cfg.get("args")
        configured = {
            "api_base": model_cfg.get("api_base"),
            "args.api_base": args.get("api_base") if isinstance(args, dict) else None,
        }
        keys = [k for k, v in configured.items() if v and v != deployment.api_base]
        if keys:
            raise ValueError(
                f"Model '{name}' sets {' and '.join(repr(k) for k in keys)} "
                f"but its realized deployment serves {deployment.api_base!r}; "
                "remove the api_base to use the deployment, or drop the "
                "deployment to use the endpoint"
            )


def _apply_endpoint_injection(
    cfg: dict[str, Any], endpoint_map: Mapping[str, str]
) -> dict[str, Any]:
    """Adapt legacy endpoint-only callers into the YAML configuration shape.

    This compatibility adapter is for external callers that do not have a
    typed :class:`Deployment`. Internal auto-serve orchestration must hand the
    complete realized deployment to :class:`EvalSession` instead.

    For each model in ``endpoint_map``:
        - Set ``api_base`` to the given endpoint (always overrides).
        - If neither ``api_key`` nor ``api_key_env`` is present at either top
          level or inside ``args``, set ``api_key: "local"`` placeholder
          (OpenAI client needs one).
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

        args = mcfg.get("args") or {}
        nested = args if isinstance(args, dict) else {}
        if not any(mcfg.get(k) or nested.get(k) for k in _CREDENTIAL_KEYS):
            mcfg["api_key"] = _LAUNCHED_ENGINE_CREDENTIAL

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


def _refuse_unpinned_staging(name: str, ds_class: type, path: str) -> None:
    """Refuse *path* when it is a registered source's staging dir holding other bytes.

    Only the staging dir is checked: there the registry's pin says what the
    bytes must be. A user's own copy elsewhere loads as given.
    """
    try:
        meta = get_dataset_meta(ds_class)
    except AttributeError:
        return  # undecorated user-custom class: no registered sources
    data_dir = resolve_data_dir(None)
    target = Path(maybe_resolve_hf_path(path)).resolve()
    for src in meta.source:
        if staged_dir(meta, src, data_dir).resolve() != target:
            continue
        try:
            probe = probe_source(meta, src, data_dir)
        except NotImplementedError:
            continue
        if probe.status == "stale":
            raise RuntimeError(
                f"Dataset '{name}' loads {target}, which is not the data {src} "
                f"pins: {probe.reason}. Run `sieval dataset download "
                f"{meta.name} --force` to replace it."
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
        resume: bool | str | Path | None = None,
        result_dir_override: str | None = None,
        deterministic_override: bool | None = None,
        endpoint_map: Mapping[str, str] | None = None,
        infer_plans: Mapping[str, dict[str, Any]] | None = None,
        invocation: str | None = None,
        self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
        realized_deployments: Mapping[str, Deployment] | None = None,
        model_capability_profiles: Mapping[str, ModelCapabilityProfile] | None = None,
        serving_reconciler: ServingReconciler | None = None,
        cli_model: CliModel | None = None,
        infer_gpu: str | None = None,
        unlaunched_deployments: Mapping[str, Deployment] | None = None,
    ):
        self.config_path = Path(config_path)
        self.model_override = model_override
        # `True`, or a recorded session dir each task starts from; see
        # `_cli_resume_for`.
        self.resume_override: ResumeValue = as_resume(resume)
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
        self._infer_gpu = infer_gpu
        unlaunched = dict(unlaunched_deployments or {})
        if set(unlaunched) & set(realized_deployment_map):
            raise ValueError(
                "a root model cannot have both a realized and an unlaunched "
                "deployment: "
                + ", ".join(sorted(set(unlaunched) & set(realized_deployment_map)))
            )
        # Roots served by no engine: bound like launched ones, so every
        # identity is the launched one, but a task never infers through them.
        self._unlaunched_roots = frozenset(unlaunched)
        self._realized_deployments = {**realized_deployment_map, **unlaunched}
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
                self._external_reconciler_inputs,
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
        if "alignment" in loaded_config:
            logger.warning(
                "{}: the `alignment:` block was removed in 0.11 and is ignored; "
                "delete it",
                self.config_path,
            )
        # Before every view below, so the entry is persisted as config.
        apply_cli_model(cast(dict[str, Any], loaded_config), cli_model, model_override)

        # Pristine YAML — source of truth for deterministic / result_dir
        # resolution. Deep-copied so downstream in-place mutation on
        # ``loaded_config`` cannot leak in.
        self._raw_config: RootConfigDict = copy.deepcopy(loaded_config)

        # A directory that is this session's own result_dir resumes in place.
        own = self._resolve_result_dir()
        if (
            isinstance(self.resume_override, Path)
            and own is not None
            and same_dir(self.resume_override, Path(own))
        ):
            self.resume_override = True

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
        _refuse_removed_resume_keys(reified)
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
        _refuse_endpoint_with_realized_deployment(
            self.config, self._realized_deployments
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
        # Vendored services the configured tasks declare (`Task.services`).
        self._required_services: set[str] = set()
        self._task_model_requirements: tuple[TaskModelRequirement, ...] = ()
        self._aggregated_requirements: dict[str, AggregatedTaskRequirements] = {}
        self._model_types_by_root: dict[str, Literal["chat", "gen"]] = {}
        self._model_types_by_binding: dict[str, Literal["chat", "gen"]] = {}
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

        # What a resume changed in the config, one line per leaf, for each
        # task's `meta.json` `runs[].config`.
        self._resume_config_diff: list[str] = []
        # The `effective_config.yaml` a resume found, read before this run
        # rewrites it: what a run recorded without identities is compared with.
        self._recorded_config: dict | None = None

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
            # Written by `_persist_effective_config`.
            effective_config=(
                str(Path(self.result_dir) / "effective_config.yaml")
                if self.result_dir
                else None
            ),
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
        if "engine" in root_config:
            validate_configured_engine_id(
                root_config.get("engine"),
                context=f"Model '{root_name}'",
            )
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
        """Select a dialect from explicit protocol or the derived model kind."""

        chain = self._model_config_chain(binding.config_name, models_cfg)
        dialect_declared = any("dialect" in config for _, config in chain)
        explicit_dialect = self._merged_model_value(chain, "dialect")
        validate_omitted_dialect_migration(
            binding.config_name,
            model_type,
            self._merged_model_value(chain, "engine"),
            dialect_declared=dialect_declared,
        )
        if dialect_declared and (
            not isinstance(explicit_dialect, str) or not explicit_dialect
        ):
            raise ValueError(
                f"Model '{binding.config_name}' dialect must be a non-empty string"
            )

        if dialect_declared:
            assert isinstance(explicit_dialect, str)
            dialect_id = explicit_dialect
        else:
            dialect_id = "openai_chat" if model_type == "chat" else "openai_completions"
        return dataclasses.replace(binding, dialect_id=dialect_id)

    def _validate_named_resource_config(
        self,
        model_name: str,
        config: Mapping[str, object],
    ) -> None:
        """Reject resource overrides before reconciliation, allocation, or I/O."""

        # Refuse both ``extra`` placements before any engine starts.
        model_config_extra(config, f"Model '{model_name}'")
        is_root = config.get("base") is None
        allowed_top_level = (
            frozenset(
                {
                    "api_base",
                    "api_key",
                    "api_key_env",
                    "capabilities",
                    "client_timeout",
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
            frozenset(
                {"api_base", "api_key", "api_key_env", "client_timeout", "max_retries"}
            )
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
        if is_root:
            # Fail an unset ``api_key_env`` or bad ``client_timeout`` on every
            # dialect.
            _configured_credential(
                config,
                cast(Mapping[str, object], raw_args),
                f"Model '{model_name}'",
                os.environ,
            )
            _configured_client_timeout(
                config,
                cast(Mapping[str, object], raw_args),
                DEFAULT_REQUEST_TIMEOUT,
                f"Model '{model_name}'",
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

    def _external_provenance_context(
        self,
        root_deployment_key: str,
    ) -> _ExternalProvenanceContext | None:
        """Resolve the stable identity view for one external runtime root."""

        found: dict[str, tuple[RuntimeBindingPlan, RuntimeBindingPlan]] = {}
        has_projection = False
        for sources in self._task_role_model_sources.values():
            for source in sources.values():
                if not isinstance(source, Model) or source.runtime_plan is None:
                    continue
                runtime_plan = source.runtime_plan
                if runtime_plan.root_deployment_key != root_deployment_key:
                    continue
                provenance_plan = source.provenance_plan
                if provenance_plan is None:
                    raise ValueError(
                        "external legacy source has incomplete provenance: "
                        f"{runtime_plan.binding_id}"
                    )
                if provenance_plan != runtime_plan:
                    has_projection = True

                previous = found.get(runtime_plan.binding_id)
                current = (runtime_plan, provenance_plan)
                if previous is not None and previous != current:
                    raise ValueError(
                        f"external binding {runtime_plan.binding_id!r} resolves to "
                        "different runtime or provenance plans within deployment "
                        f"root {root_deployment_key!r}"
                    )
                found[runtime_plan.binding_id] = current

        if not has_projection:
            return None

        stable_roots = {
            provenance.root_deployment_key for _, provenance in found.values()
        }
        if len(stable_roots) != 1:
            raise ValueError(
                "external deployment root mixes incompatible provenance policies: "
                + ", ".join(sorted(stable_roots))
            )
        stable_root = next(iter(stable_roots))
        replacements: dict[str, str] = {}
        ambiguous_replacements: set[str] = set()
        runtime_to_provenance: dict[str, str] = {}
        for runtime_plan, provenance_plan in found.values():
            previous_fingerprint = runtime_to_provenance.get(runtime_plan.fingerprint)
            if (
                previous_fingerprint is not None
                and previous_fingerprint != provenance_plan.fingerprint
            ):
                raise ValueError(
                    "external runtime plan has conflicting stable provenance "
                    f"projections: {runtime_plan.fingerprint}"
                )
            runtime_to_provenance[runtime_plan.fingerprint] = (
                provenance_plan.fingerprint
            )
            _record_plan_provenance_replacements(
                replacements,
                ambiguous_replacements,
                runtime_plan,
                provenance_plan,
            )

        return _ExternalProvenanceContext(
            root_deployment_key=stable_root,
            replacements=MappingProxyType(dict(replacements)),
            ambiguous_replacements=frozenset(ambiguous_replacements),
            runtime_to_provenance=MappingProxyType(dict(runtime_to_provenance)),
        )

    def _external_reconciler_inputs(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: DeploymentReconcileInput,
    ) -> tuple[tuple[ServingRequirement, ...], DeploymentReconcileInput]:
        """Hide wrapper-private ownership identity from injected reconcilers."""

        context = self._external_provenance_context(deployment.root_deployment_key)
        if context is None:
            return requirements, deployment

        projected_requirements = tuple(
            _project_provenance_requirement(
                requirement,
                context.replacements,
                context.runtime_tokens,
            )
            for requirement in requirements
        )

        prelaunch_plan = deployment.prelaunch_plan
        if prelaunch_plan is not None:
            prelaunch_plan = _project_external_deployment_plan(
                prelaunch_plan,
                root_deployment_key=context.root_deployment_key,
                replacements=context.replacements,
                runtime_to_provenance=context.runtime_to_provenance,
                runtime_tokens=context.runtime_tokens,
            )
        projected_deployment = dataclasses.replace(
            deployment,
            root_deployment_key=context.root_deployment_key,
            prelaunch_plan=prelaunch_plan,
        )
        projected_input: JSONValue = {
            "requirements": [
                requirement.to_json_value() for requirement in projected_requirements
            ],
            "deployment": projected_deployment.to_json_value(),
        }
        _reject_unprojected_provenance_tokens(
            projected_input,
            context.runtime_tokens,
            context="external reconciler inputs",
        )
        return projected_requirements, projected_deployment

    def _external_provenance_plan(
        self,
        model: Model,
        result: ReconcileResult,
        binding_id: str,
    ) -> RuntimeBindingPlan:
        """Project a reconciled external plan without its runtime-only identity."""

        runtime_plan = result.runtime_plans[binding_id]
        baseline = model.provenance_plan
        if baseline is None:
            raise ValueError(
                f"external binding {binding_id!r} has incomplete baseline provenance"
            )
        source_baseline = model.runtime_plan
        if source_baseline is None:
            raise ValueError(f"external binding {binding_id!r} has no runtime plan")
        if baseline == source_baseline:
            return runtime_plan
        context = self._external_provenance_context(runtime_plan.root_deployment_key)
        if context is None:
            raise ValueError(
                f"external binding {binding_id!r} has no stable provenance context"
            )
        if baseline.root_deployment_key != context.root_deployment_key:
            raise ValueError(
                f"external binding {binding_id!r} has a conflicting stable "
                "deployment root"
            )
        binding_plan = result.binding_plans[binding_id]
        stable_scope = ConnectionScope(
            credential_scope=baseline.connection_identity.credential_scope,
            retry_policy=baseline.connection_identity.retry_policy,
            quota_scope=baseline.connection_identity.quota_scope,
        )
        provenance_binding_plan = _project_external_binding_plan(
            binding_plan,
            binding_id=baseline.binding_id,
            root_deployment_key=baseline.root_deployment_key,
            connection_scope=stable_scope,
            replacements=context.replacements,
            runtime_tokens=context.runtime_tokens,
        )

        replacements = dict(context.replacements)
        ambiguous_replacements = set(context.ambiguous_replacements)
        runtime_to_provenance = dict(context.runtime_to_provenance)

        _record_provenance_replacement(
            replacements,
            ambiguous_replacements,
            runtime_plan.binding_plan_fingerprint,
            provenance_binding_plan.fingerprint,
        )
        _record_provenance_replacement(
            replacements,
            ambiguous_replacements,
            runtime_plan.binding_id,
            provenance_binding_plan.binding_id,
        )
        _record_provenance_replacement(
            replacements,
            ambiguous_replacements,
            runtime_plan.root_deployment_key,
            provenance_binding_plan.root_deployment_key,
        )
        _record_provenance_replacement(
            replacements,
            ambiguous_replacements,
            runtime_plan.connection_identity.credential_scope,
            stable_scope.credential_scope,
        )
        _record_provenance_replacement(
            replacements,
            ambiguous_replacements,
            runtime_plan.connection_identity.quota_scope,
            stable_scope.quota_scope,
        )

        runtime_tokens = context.runtime_tokens | {
            runtime_plan.fingerprint,
            runtime_plan.verification_fingerprint,
            runtime_plan.deployment_plan_fingerprint,
        }

        deployment_plan = result.deployment_plans[runtime_plan.root_deployment_key]
        provenance_deployment_plan = _project_external_deployment_plan(
            deployment_plan,
            root_deployment_key=baseline.root_deployment_key,
            replacements=replacements,
            runtime_to_provenance=runtime_to_provenance,
            runtime_tokens=frozenset(runtime_tokens),
        )
        _reject_provenance_tokens_in_checks(
            runtime_plan.request_checks,
            frozenset(runtime_tokens),
        )
        provenance_plan = dataclasses.replace(
            runtime_plan,
            binding_id=baseline.binding_id,
            root_deployment_key=baseline.root_deployment_key,
            connection_identity=baseline.connection_identity,
            binding_plan_fingerprint=provenance_binding_plan.fingerprint,
            deployment_plan_fingerprint=provenance_deployment_plan.fingerprint,
        )
        _reject_unprojected_provenance_tokens(
            provenance_plan.to_json_value(),
            frozenset(runtime_tokens),
            context="external provenance plan",
        )
        return provenance_plan

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

        if binding.dialect_id is not None:
            spec = get_dialect_spec(binding.dialect_id)
            if spec.provider_enforced_capabilities:
                # Keyed by dialect, never by model name.
                return ModelCapabilityProfile(
                    {
                        key: ModelCapabilityEntry(
                            ModelCapabilityStatus.SUPPORTED,
                            source="dialect-provider-enforced",
                            reason=(
                                f"{binding.dialect_id} rejects this with an "
                                "error on a model that lacks it"
                            ),
                            verifier=_PROVIDER_ENFORCED_VERIFIER,
                        )
                        for key in CAPABILITY_KEYS
                        if key in spec.provider_enforced_capabilities
                    },
                    authoritative=False,
                )

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
                subject=self._binding_subject(binding.binding_id),
            )
            raw_args = typed_source.get("args")
            nested: Mapping[str, object] = (
                cast(Mapping[str, object], raw_args)
                if isinstance(raw_args, Mapping)
                else {}
            )
            credential_kind = (
                "explicit"
                if _names_credential(typed_source) or _names_credential(nested)
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
        has_explicit_credential = _names_credential(root_config) or _names_credential(
            args
        )
        _, max_retries, _ = self._connection_options(
            root_config,
            nested_args=True,
            subject=f"Model '{root_name}'",
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
            if engine_declared:
                raw_engine = validate_configured_engine_id(
                    raw_engine,
                    context=f"Inline binding '{binding.binding_id}'",
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
        if engine_declared:
            raw_engine = validate_configured_engine_id(
                raw_engine,
                context=f"Model '{root_name}'",
            )
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

    @staticmethod
    def _validate_engine_dialect_requirements(
        binding: NormalizedModelBinding,
        requirements: AggregatedTaskRequirements,
        declarations: Mapping[str, JSONValue],
        request_intents: Mapping[CapabilityKey, CapabilityIntent],
        engine_id: str,
    ) -> None:
        """Reject known engine/dialect incompatibilities before model I/O."""

        input_scoring_intent = request_intents.get("input_scoring")
        declares_input_scoring = (
            "input_scoring" in declarations
            and declarations["input_scoring"] is not False
        )
        requires_input_scoring = (
            requirements.input_scoring
            or (input_scoring_intent is not None and input_scoring_intent.required)
            or declares_input_scoring
        )
        # TODO(PR-5): replace this deployment-specific guard with the
        # registered sglang_native binder's capability contract.
        if (
            engine_id != "sglang"
            or binding.dialect_id != "openai_completions"
            or not requires_input_scoring
        ):
            return

        sources_set = set(requirements.input_scoring_sources)
        if input_scoring_intent is not None:
            sources_set.update(input_scoring_intent.sources)
        if declares_input_scoring:
            sources_set.add("model.capabilities.input_scoring")
        sources = ", ".join(sorted(sources_set))
        source_detail = f" (required by {sources})" if sources else ""
        raise ValueError(
            f"Model binding '{binding.binding_id}' requires input_scoring"
            f"{source_detail}, but deployment engine 'sglang' with dialect "
            "'openai_completions' cannot serve input_scoring: SGLang's "
            "/v1/completions endpoint rejects echo=True with logprobs. Use a "
            "named model binding with dialect: sglang_legacy until the "
            "sglang_native binder replaces the compatibility bypass."
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
        self._required_services = set()
        for task_name, task_cfg in tasks_cfg.items():
            if task_name == RESERVED_LOG_DIR:
                raise ValueError(
                    f"Task name {RESERVED_LOG_DIR!r} is reserved for "
                    f"<result_dir>/{RESERVED_LOG_DIR}/; rename the task."
                )
            task_spec = task_cfg.get("class")
            if not isinstance(task_spec, str) or not task_spec:
                raise ValueError(f"Task '{task_name}' requires 'class' field")
            task_class = resolve_task_class(task_spec)
            declared = frozenset(getattr(task_class, "services", frozenset()))
            check_service_names(declared, f"Task '{task_name}'")
            self._required_services |= declared
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

        grouped: dict[str, list[TaskModelRequirement]] = {
            binding_id: [] for binding_id in self._normalized_model_bindings
        }
        for record in records:
            grouped.setdefault(record.binding.binding_id, []).append(record)
        aggregated = {
            binding_id: aggregate_task_requirements(group)
            for binding_id, group in grouped.items()
        }

        # Freeze one compatibility-wrapper kind for every binding form. Named
        # bindings inherit their deployment root's decision; inline bindings
        # use the shared resolver's historical zero-evidence Chat fallback;
        # borrowed legacy wrappers contribute their explicit ChatModel/GenModel
        # type without letting the dialect select it.
        model_types_by_binding: dict[str, Literal["chat", "gen"]] = {}
        finalized_inline: dict[str, InlineModelBinding] = {}
        for binding_id, binding in self._normalized_model_bindings.items():
            if isinstance(binding, NamedModelBinding):
                model_type = model_types_by_root[binding.root_deployment_key]
            else:
                explicit_type: Literal["chat", "gen"] | None = None
                if isinstance(binding, ExternalModelBinding):
                    external_model = self._external_model_for_binding(binding_id)
                    if isinstance(external_model, ChatModel):
                        explicit_type = "chat"
                    elif isinstance(external_model, GenModel):
                        explicit_type = "gen"
                model_type = derive_model_type(
                    binding_id,
                    explicit_type,
                    aggregated[binding_id],
                )
                if isinstance(binding, InlineModelBinding):
                    finalized_inline[binding_id] = finalize_inline_model_binding(
                        binding,
                        model_type,
                    )
                elif binding.dialect_id is None:
                    raise ValueError(f"Model binding '{binding_id}' has no dialect")
                else:
                    validate_model_type_dialect(
                        binding_id,
                        model_type,
                        binding.dialect_id,
                    )
            model_types_by_binding[binding_id] = model_type

        if finalized_inline:

            def replace_inline(
                binding: NormalizedModelBinding,
            ) -> NormalizedModelBinding:
                return finalized_inline.get(binding.binding_id, binding)

            records = [
                dataclasses.replace(record, binding=replace_inline(record.binding))
                for record in records
            ]
            self._task_requirement_contexts = {
                task_name: dataclasses.replace(
                    context,
                    model_bindings={
                        role: replace_inline(binding)
                        for role, binding in context.model_bindings.items()
                    },
                )
                for task_name, context in self._task_requirement_contexts.items()
            }
            self._normalized_model_bindings = {
                binding_id: replace_inline(binding)
                for binding_id, binding in self._normalized_model_bindings.items()
            }

        self._model_types_by_root = model_types_by_root
        self._model_types_by_binding = model_types_by_binding

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
                        "capabilities while sglang_native has no executable binder"
                    )
                legacy_bypass.add(binding_id)
                continue
            if binding.dialect_id is None:
                raise ValueError(f"Model binding '{binding_id}' has no dialect")

            declarations = self._declarations_for_binding(binding, models_cfg)
            self._validate_legacy_capability_surfaces(binding, declarations, models_cfg)
            request_intents = self._legacy_request_intents_for(binding, models_cfg)
            deployment = self._deployment_input_for(binding, models_cfg)
            self._validate_engine_dialect_requirements(
                binding,
                requirements,
                declarations,
                request_intents,
                deployment.engine_id,
            )

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
                    request_intents=request_intents,
                    route_intent=self._route_intent_for(binding, models_cfg),
                )
            )
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

    @property
    def required_services(self) -> frozenset[str]:
        """The vendored services the configured tasks declare, once prelaunch ran."""
        return frozenset(self._required_services)

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

    def _binding_subject(self, binding_id: str) -> str:
        """Name a role binding the way its config does, for error messages."""

        for task_name, context in self._task_requirement_contexts.items():
            for role, binding in context.model_bindings.items():
                if binding.binding_id == binding_id:
                    return f"Task '{task_name}' {role}"
        return f"Model binding '{binding_id}'"

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
        subject: str = "Model",
        environ: Mapping[str, str] | None = None,
    ) -> tuple[str | None, int, int | None]:
        """Extract client/pool options without treating them as request defaults."""

        args = config.get("args")
        nested: Mapping[str, object] = (
            cast(Mapping[str, object], args)
            if nested_args and isinstance(args, Mapping)
            else {}
        )
        api_key = _configured_credential(
            config, nested, subject, os.environ if environ is None else environ
        )
        max_retries = config.get("max_retries", nested.get("max_retries", 3))
        concurrency_limit = config.get(
            "concurrency_limit", nested.get("concurrency_limit")
        )
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if isinstance(concurrency_limit, bool) or (
            concurrency_limit is not None
            and (not isinstance(concurrency_limit, int) or concurrency_limit < 1)
        ):
            raise ValueError("concurrency_limit must be a positive integer")
        # Validate at prelaunch; the real default needs the realized deployment.
        _configured_client_timeout(config, nested, DEFAULT_REQUEST_TIMEOUT, subject)
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
        prelaunch_bindings = {
            binding.binding_id: binding for binding in self._prelaunch_binding_inputs
        }
        for binding in self._normalized_model_bindings.values():
            if binding.binding_id in self._legacy_bypass_bindings:
                continue
            template = self._prelaunch_deployment_inputs[binding.root_deployment_key]
            deployment = self._configured_deployment_for(binding, template, models_cfg)
            prelaunch_binding = prelaunch_bindings[binding.binding_id]
            self._validate_engine_dialect_requirements(
                binding,
                self._aggregated_requirements[binding.binding_id],
                prelaunch_binding.declarations,
                prelaunch_binding.request_intents,
                deployment.engine.engine_id,
            )
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
        subject: str,
    ) -> tuple[dict[str, Any], Any]:
        raw = config.get("args", {})
        if not isinstance(raw, Mapping):
            raise ValueError("model args must be a dictionary")
        extra = model_config_extra(config, subject)
        args = dict(raw)
        for key in (
            "api_base",
            "api_key",
            "api_key_env",
            "client_timeout",
            "max_retries",
            "concurrency_limit",
        ):
            args.pop(key, None)
        args.pop("extra", None)
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
            config, nested_args=nested_args, subject=f"Model root '{root_key}'"
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
                credential=_client_credential(api_key, deployment),
                max_retries=max_retries,
                timeout=_configured_client_timeout(
                    config,
                    _nested_args(config) if nested_args else {},
                    _default_client_timeout(deployment),
                    f"Model root '{root_key}'",
                ),
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
        model_type: Literal["chat", "gen"],
    ) -> Model:
        """Apply the one model-kind decision frozen during prelaunch."""

        input_kind = InputKind.CHAT if model_type == "chat" else InputKind.COMPLETION
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
            args.pop("extra", None)
            extra = model_config_extra(cfg, f"Model '{name}'")
            if extra is not None:
                args["extra"] = extra
            if self.deterministic:
                _apply_request_seed_decision_to_args(
                    args,
                    self._request_seed_decisions_by_binding[binding.binding_id],
                )
            realized = self._realized_deployments.get(name)
            credential = _client_credential(
                _configured_credential(cfg, args, f"Model '{name}'", os.environ),
                realized,
            )
            for key in _CREDENTIAL_KEYS:
                args.pop(key, None)
            if credential is not None:
                args["api_key"] = credential
            realized = self._realized_deployments.get(name)
            # A conflicting configured api_base was refused at construction.
            if realized is not None:
                args["api_base"] = realized.api_base
            elif "api_base" in cfg:
                args["api_base"] = cfg["api_base"]
            timeout = _configured_client_timeout(
                cfg,
                args,
                _default_client_timeout(realized),
                f"Model '{name}'",
            )
            args.pop("client_timeout", None)
            legacy_model = SglangGenModel(
                model=binding.requested_model_id, client_timeout=timeout, **args
            )
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
                    args.pop("extra", None)
                    model = model.with_args(
                        concurrency_limit=concurrency_limit,
                        extra=cast(
                            dict[str, JSONValue] | None,
                            model_config_extra(cfg, f"Model '{name}'"),
                        ),
                        **args,
                    )
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

                    args, extra = self._request_builder_args(cfg, f"Model '{name}'")
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
                        self._model_types_by_binding[binding.binding_id],
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
            extra = model_config_extra(
                typed_source, f"Inline binding '{binding.binding_id}'"
            )
            direct_config: dict[str, Any] = dict(typed_source)
            nested = direct_config.pop("args", None)
            if isinstance(nested, Mapping):
                direct_config.update(cast(Mapping[str, Any], nested))
            for key in (
                "model",
                "api_base",
                "api_key",
                "api_key_env",
                "client_timeout",
                "max_retries",
                "concurrency_limit",
                "dialect",
                "service_role",
                "capabilities",
                "engine",
                "extra",
            ):
                direct_config.pop(key, None)
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
                model, self._model_types_by_binding[binding.binding_id]
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
                    provenance_plan = self._external_provenance_plan(
                        live_model,
                        result,
                        binding.binding_id,
                    )
                    rebound = live_model.with_reconciled_plan(
                        runtime_plan,
                        provenance_plan,
                    )
                    if self.deterministic:
                        rebound = _apply_request_seed_decision_to_model(
                            rebound,
                            self._request_seed_decisions_by_external_role[
                                f"{task_name}.{role}"
                            ],
                        )
                    role_model = self._as_legacy_wrapper(
                        rebound, self._model_types_by_binding[binding.binding_id]
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
            if path is not None:
                _refuse_unpinned_staging(name, ds_class, path)

            try:
                dataset = ds_class(path, **init_args) if path else ds_class(**init_args)
            except FileNotFoundError as exc:
                # Also catches `datasets.exceptions.DataFilesNotFoundError`.
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
            runner_config = self._build_runner_config(
                task_cfg, runner_defaults, task_name
            )

            assert self.runner is not None, "Runner not initialized"
            self.runner.add_task(
                task,
                config=runner_config,
                inference_recipe=self._inference_recipe_for(candidate.config_name),
                resume_config_diff=list(self._resume_config_diff),
                legacy_match=self._legacy_match_for(task_name, runner_config),
                unserved_candidate=(
                    self._root_model(candidate.config_name) in self._unlaunched_roots
                ),
            )
            logger.info("Added task '{}' with class '{}'", task_name, task_spec)

    def _cli_resume_for(self, task_name: str) -> ResumeValue:
        """What the session's ``resume`` (``--resume [DIR]``) means for *task_name*.

        ``True`` resumes the task in place. ``DIR`` names a recorded session
        ``result_dir``: the task starts from ``DIR/<task_name>``. When ``DIR``
        is itself a task directory (it holds ``meta.json``) and the config has
        one task, that task starts from ``DIR``. A task with no directory
        under ``DIR`` runs fresh.
        """
        resume = self.resume_override
        if not isinstance(resume, Path):
            return resume
        candidate = resume / task_name
        if candidate.is_dir():
            return candidate
        tasks = self._get_named_config_map("tasks")
        if len(tasks) == 1 and (resume / "meta.json").is_file():
            return resume
        logger.info(
            "Task '{}': no recorded directory under {}; runs fresh",
            task_name,
            resume,
        )
        return None

    def _legacy_match_for(
        self, task_name: str, runner_config: TaskRunnerConfig
    ) -> LegacyStageMatch | None:
        """Compare this run's config with the one a recorded run persisted.

        Read only for a recorded directory without stage identities (see
        :func:`_stage_config`). The recorded config is the
        ``effective_config.yaml`` of the session ``result_dir`` the task
        directory sits in: the parent of the ``resume`` directory copied, or
        for an in-place resume the one this run found before rewriting it.
        ``None`` when there is none: both stages stay unknown.
        """
        if isinstance(runner_config.resume, Path):
            persisted = read_persisted_config(
                runner_config.resume.parent / "effective_config.yaml"
            )
        elif runner_config.resume:
            persisted = self._recorded_config
        else:
            return None
        if persisted is None:
            return None
        recorded_tasks = persisted.get("tasks") or {}
        recorded_name = task_name if task_name in recorded_tasks else None
        if recorded_name is None and len(recorded_tasks) == 1:
            recorded_name = next(iter(recorded_tasks))
        if recorded_name is None:
            return None
        current = self._reified_config
        return {
            "data": _stage_config(current, task_name, "data")
            == _stage_config(persisted, recorded_name, "data"),
            "inference": _stage_config(current, task_name, "inference")
            == _stage_config(persisted, recorded_name, "inference"),
        }

    def _root_model(self, model_name: str) -> str:
        """The root of *model_name*'s ``base`` chain: the model that is served."""
        models_cfg = self._get_named_config_map("models")
        if model_name not in models_cfg:
            return model_name
        return self._model_config_chain(model_name, models_cfg)[0][0]

    def task_models(self) -> dict[str, str]:
        """Each task's model under test, as the root model config that serves it.

        Read after :meth:`prepare_prelaunch`.
        """
        out: dict[str, str] = {}
        for task_name, context in self._task_requirement_contexts.items():
            candidate = context.model_bindings.get("candidate")
            if isinstance(candidate, NamedModelBinding):
                out[task_name] = self._root_model(candidate.config_name)
        return out

    def _inference_recipe_for(self, model_name: str) -> dict[str, JSONValue] | None:
        """The serving plan of *model_name*'s root, for the inference identity.

        ``None`` for a model SiEval does not serve. The checkpoint is the model
        under test, so it is left out, and so is every field that says where
        the plan ran rather than how: each device group's ``host`` and each
        attachment's ``address``. The rest of a ``DeploymentPlan`` is
        launch-invariant -- the serving port is the translator's, not the
        plan's.
        """
        infer_plans = self._infer_plans
        if not infer_plans:
            return None
        plan = infer_plans.get(self._root_model(model_name))
        if plan is None:
            return None
        recipe = cast(dict[str, Any], copy.deepcopy(unwrap_proxies(plan)))
        recipe.pop("checkpoint", None)
        for assignment in recipe.get("assignments") or ():
            if isinstance(assignment, dict) and isinstance(
                assignment.get("devices"), dict
            ):
                assignment["devices"].pop("host", None)
        for attachment in recipe.get("attachments") or ():
            if isinstance(attachment, dict):
                attachment.pop("address", None)
        return cast(dict[str, JSONValue], recipe)

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
        self,
        task_cfg: TaskConfigDict,
        defaults: dict[str, Any],
        task_name: str | None = None,
    ) -> TaskRunnerConfig:
        """Build TaskRunnerConfig from task config and defaults.

        ``resume`` precedence: a YAML directory, then the CLI, then the YAML's
        ``true`` / ``false``.
        """
        # Start with defaults
        cfg_dict = dict(defaults)

        # Override with task-specific config
        task_runner_cfg = self._normalize_dict(
            task_cfg.get("runner_config"), "Task 'runner_config'"
        )
        cfg_dict.update(task_runner_cfg)

        configured = as_resume(cfg_dict.get("resume"), "runner_config.resume")
        if not isinstance(configured, Path):
            cli = (
                self._cli_resume_for(task_name)
                if task_name is not None
                else as_resume(self.resume_override)
            )
            # A CLI resume wins over the YAML's true or false.
            if cli or configured is None:
                configured = cli
        if isinstance(configured, Path):
            own = cfg_dict.get("result_dir") or (
                Path(session_dir) / task_name
                if task_name is not None
                and (session_dir := self._resolve_result_dir()) is not None
                else None
            )
            if own is not None and resume_source(configured, Path(own)) is None:
                configured = True
        cfg_dict["resume"] = configured

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

    async def _persist_audit_yaml(
        self,
        *,
        target_name: str,
        body: str,
        header: str,
        audit_label: str,
        note_diff: bool = False,
    ) -> None:
        """Atomically write ``header + body`` to ``result_dir/target_name``.

        Under ``--resume`` with an existing file, an identical body skips the
        rewrite (timestamps survive); a different one is rewritten under the
        original header, which keeps the first run's provenance. Nothing here
        refuses a resume: each task compares its stage identities with the
        recorded ``meta.json`` instead. With *note_diff*, what changed goes to
        each task's ``meta.json`` ``runs[].config``. Best effort: a failure is
        logged.
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

        try:
            if self.resume_override is True and await target.exists():
                existing = await target.read_text(encoding="utf-8")
                existing_header, existing_body = _split_header(existing)
                if existing_body == body:
                    logger.info("Resume: {} matches — skipping rewrite", target_name)
                    return
                write_header = existing_header or header
                if note_diff:
                    self._note_config_diff(existing_body, body)
        except OSError as e:
            logger.warning("Could not read existing {}: {}", target, e)

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

    def _note_config_diff(self, existing_body: str, body: str) -> None:
        """Keep what a resume changed in the config, for ``runs[].config``."""
        try:
            before = yaml.safe_load(existing_body) or {}
            after = yaml.safe_load(body) or {}
        except yaml.YAMLError as e:
            logger.warning("Cannot diff the persisted config: {}", e)
            return
        if not isinstance(before, dict) or not isinstance(after, dict):
            return
        # Where the file lives and how the run started, not what ran.
        for cfg in (before, after):
            cfg.pop("result_dir", None)
            _drop_resume_keys(cfg)
        self._resume_config_diff.extend(
            line.removeprefix("- ") for line in _diff_lines(before, after)
        )

    async def _persist_effective_config(self) -> None:
        """Write effective_config.yaml to result_dir at session start.

        Dumps ``self._reified_config`` (raw YAML + CLI reification + the
        resolved deterministic-seed contract, minus legacy endpoint-adapter
        injection). User-supplied ``api_base`` and ``api_key_env`` are
        preserved; ``api_key`` / ``authorization`` values are redacted, and a
        rerun of this file refuses them. Runtime endpoints are excluded, so
        ``sieval run <this file>`` can re-launch services from the preserved
        ``path`` / ``infer`` fields.
        """
        body = yaml.safe_dump(
            redact_secrets(self._reified_config),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        result_dir = self._resolve_result_dir()
        if self.resume_override is True and result_dir:
            self._recorded_config = await run_sync(
                read_persisted_config, Path(result_dir) / "effective_config.yaml"
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
        await self._persist_audit_yaml(
            target_name="effective_config.yaml",
            body=body,
            header=header,
            audit_label="effective config",
            note_diff=True,
        )

    async def _persist_infer_plans(self) -> None:
        """Write infer_plans.yaml to result_dir when the caller supplied plans.

        Rerun re-resolves plans from the ``infer:`` section of
        effective_config.yaml plus the installed sieval version. A resume
        under ``SIEVAL_TARGET_GPU`` equal to the recorded ``gpu`` reuses a plan
        whose model config is unchanged and that holds no redacted secret
        (``sieval run``). It never blocks a resume: the serving recipe is part
        of each task's inference identity, which a resume compares instead.
        """
        if not self._infer_plans:
            return

        payload: dict[str, Any] = {"models": dict(self._infer_plans)}
        if self._infer_gpu is not None:
            # Outside `models`, so it stays out of every inference identity.
            payload = {"gpu": self._infer_gpu, **payload}
        body = yaml.safe_dump(
            redact_secrets(payload),
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
                "  effective_config.yaml + installed sieval version, except",
                "  that a resume reuses a model's plan when SIEVAL_TARGET_GPU",
                "  equals `gpu`, the model's config is unchanged and its",
                "  plan holds no <redacted> value.",
            ],
        )
        await self._persist_audit_yaml(
            target_name="infer_plans.yaml",
            body=body,
            header=header,
            audit_label="infer plans",
        )

    async def arun(self) -> dict[str, Any]:
        """Run all configured tasks asynchronously."""
        try:
            # Pure prelaunch resolution stamps the binding-local seed contract
            # into effective_config before it is persisted. No
            # model client, dataset, or serving I/O has happened at this point;
            # pure config/requirement errors may therefore surface first.
            await run_sync(self.prepare_prelaunch)
            self._stamp_deterministic_seed_contract()
            await self._persist_effective_config()
            await self._persist_infer_plans()
            # Tasks read their endpoints when built, so services start first.
            async with vendored_services_for(
                self._required_services,
                result_dir=self._resolve_result_dir,
            ):
                await self._prepare_execution()
                if self.runner is None:
                    raise RuntimeError("Runner not initialized")
                return await self.runner.arun()
        finally:
            await self._close_owned_model_resources()

    async def aplan(self) -> dict[str, RunPlan | Exception]:
        """What :meth:`arun` would do with each task, by task name; writes nothing.

        Prepares the session as :meth:`arun` does, vendored services included,
        and asks each task's runner for its :meth:`TaskRunner.plan_run`. A
        task whose planning raises maps to the exception its run would raise.
        Raises what :meth:`arun` would raise before any task starts. Service
        logs go to ``~/.sieval/logs``, never into a result directory.
        """
        try:
            await run_sync(self.prepare_prelaunch)
            self._stamp_deterministic_seed_contract()
            result_dir = self._resolve_result_dir()
            if self.resume_override is True and result_dir:
                # What `_persist_effective_config` reads before it rewrites.
                self._recorded_config = await run_sync(
                    read_persisted_config, Path(result_dir) / "effective_config.yaml"
                )
            async with vendored_services_for(
                self._required_services, result_dir=lambda: None
            ):
                await self._prepare_execution()
                if self.runner is None:
                    raise RuntimeError("Runner not initialized")
                return await self.runner.plan_run()
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
    resume: bool | str | Path | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    endpoint_map: Mapping[str, str] | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
    realized_deployments: Mapping[str, Deployment] | None = None,
    cli_model: CliModel | None = None,
    infer_gpu: str | None = None,
    unlaunched_deployments: Mapping[str, Deployment] | None = None,
) -> dict[str, Any]:
    """Run tasks defined in a YAML configuration file asynchronously.

    Args:
        config: Path to the YAML configuration file.
        model: Override model name for all base models.
        resume: ``True`` resumes each task in place; a directory is a
            recorded session ``result_dir`` each task starts from (see
            :meth:`EvalSession._cli_resume_for`).
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
        cli_model: ``--checkpoint``/``--api-base``, added as the ``models:``
            entry of a config that has none.
        infer_gpu: The GPUs *infer_plans* were resolved for, as
            ``SIEVAL_TARGET_GPU`` spells them; recorded in ``infer_plans.yaml``
            as ``gpu``, so a resume can tell whether it may reuse the plans.
        unlaunched_deployments: For a root model ``sieval run`` serves but did
            not launch, since no task had anything for it to infer: the
            deployment launching it would realize, nothing listening. Its
            tasks bind and resume as if it were launched, and refuse to infer.

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
        cli_model=cli_model,
        infer_gpu=infer_gpu,
        unlaunched_deployments=unlaunched_deployments,
    )

    return await runner.arun()


async def aplan_session(
    config: str | Path,
    *,
    model: str | None = None,
    resume: bool | str | Path | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
    unlaunched_deployments: Mapping[str, Deployment] | None = None,
    cli_model: CliModel | None = None,
) -> tuple[dict[str, str], dict[str, RunPlan | Exception]]:
    """Plan the run :func:`arun_session` would make with the same arguments.

    Returns each task's model under test (its root model config) and its
    :meth:`EvalSession.aplan` entry. Every served model passes as one of
    *unlaunched_deployments*. Reads ``--resume`` directories, writes nothing.
    """
    session = EvalSession(
        config_path=config,
        model_override=model,
        resume=resume,
        result_dir_override=result_dir,
        deterministic_override=deterministic,
        infer_plans=infer_plans,
        invocation=invocation,
        self_managed_endpoints=self_managed_endpoints,
        unlaunched_deployments=unlaunched_deployments,
        cli_model=cli_model,
    )
    plans = await session.aplan()
    return session.task_models(), plans


def run_session(
    config: str | Path,
    model: str | None = None,
    resume: bool | str | Path | None = None,
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
