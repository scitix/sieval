"""Tests for typed Model-IR capability declarations.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import dataclasses
from enum import Flag
from typing import Any, cast

import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    CAPABILITY_SPECS,
    Capability,
    CapabilityConfigError,
    CapabilityIntent,
    DecisionsOptions,
    DialectCapabilityBinding,
    DialectCapabilityStatus,
    FunctionToolsOptions,
    HostedToolsOptions,
    InputScoringOptions,
    ModelCapabilityEntry,
    ModelCapabilityProfile,
    ModelCapabilityStatus,
    ReasoningOptions,
    RequestDefaults,
    StructuredOutputOptions,
    Supported,
    TopLogprobsOptions,
    Unsupported,
    aggregate_capability_intents,
    canonical_capability_json,
    capability_declarations_to_json,
    legacy_capability_ambiguities,
    legacy_capability_intents,
    normalize_capability_declarations,
    normalize_dialect_capability_outcomes,
    validate_no_legacy_capability_ambiguity,
)

EXPECTED_KEYS = (
    "input_scoring",
    "sampled_logprobs",
    "top_logprobs",
    "reasoning",
    "function_tools",
    "hosted_tools",
    "structured_output",
    "stateful_session",
    "opaque_continuation",
    "multimodal_input",
    "prefill",
    "fim",
    "decisions",
)


def _outcomes(
    **overrides: DialectCapabilityStatus | str,
) -> dict[str, DialectCapabilityStatus | str]:
    outcomes: dict[str, DialectCapabilityStatus | str] = dict.fromkeys(
        CAPABILITY_KEYS, DialectCapabilityStatus.SUPPORTED
    )
    outcomes.update(overrides)
    return outcomes


class TestCapabilitySpecs:
    def test_stable_symbol_space_and_typed_specs(self) -> None:
        assert CAPABILITY_KEYS == EXPECTED_KEYS
        assert tuple(CAPABILITY_SPECS) == EXPECTED_KEYS
        assert len({spec.options_type for spec in CAPABILITY_SPECS.values()}) == 13

    def test_option_records_are_frozen(self) -> None:
        options = TopLogprobsOptions(minimum=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, options).minimum = 4

    def test_legacy_namespace_is_not_a_flag_or_composable(self) -> None:
        assert not issubclass(Capability, Flag)
        with pytest.raises(TypeError):
            cast(Any, Capability.Chat) | Capability.FunctionCalling

    def test_legacy_namespace_contains_only_projected_compatibility_symbols(
        self,
    ) -> None:
        assert tuple(member.name for member in Capability) == (
            "Completion",
            "Chat",
            "FunctionCalling",
            "ServerTools",
            "Reasoning",
            "ReasoningEffort",
            "TopKLogprobs",
            "InputScoring",
            "SampledLogprobs",
            "SampledLogprobsWithTokenIds",
            "StructuredOutput",
            "Prefill",
            "FIM",
        )


class TestCapabilityNormalization:
    def test_decision_forms_and_requirements_normalize_without_coercion(self):
        normalized = normalize_capability_declarations(
            {
                "decisions": {
                    "question_types": ["predicate", "choice"],
                    "evidence_kinds": ["json"],
                    "guidance_kinds": ["text", "null"],
                    "require_probabilities": True,
                }
            },
            dialect_id="test_decisions",
            outcomes=_outcomes(),
        )
        assert normalized["decisions"] == DecisionsOptions(
            question_types=("predicate", "choice"),
            evidence_kinds=("json",),
            guidance_kinds=("text", "null"),
            require_probabilities=True,
        )

    @pytest.mark.parametrize(
        "options",
        [
            {"question_types": ["generate"]},
            {"evidence_kinds": ["image"]},
            {"guidance_kinds": ["boolean"]},
            {"require_confidence": 1},
        ],
    )
    def test_invalid_decision_options_are_rejected(self, options):
        with pytest.raises(CapabilityConfigError):
            normalize_capability_declarations(
                {"decisions": options},
                dialect_id="test_decisions",
                outcomes=_outcomes(),
            )

    def test_missing_key_has_no_user_opinion(self) -> None:
        normalized = normalize_capability_declarations(
            {}, dialect_id="openai_chat", outcomes=_outcomes()
        )
        assert normalized == {}

    def test_false_is_retained_and_true_matches_empty_mapping(self) -> None:
        false_value = normalize_capability_declarations(
            {"input_scoring": False},
            dialect_id="openai_completions",
            outcomes=_outcomes(),
        )
        true_value = normalize_capability_declarations(
            {"input_scoring": True},
            dialect_id="openai_completions",
            outcomes=_outcomes(),
        )
        empty_value = normalize_capability_declarations(
            {"input_scoring": {}},
            dialect_id="openai_completions",
            outcomes=_outcomes(),
        )
        assert false_value == {"input_scoring": False}
        assert true_value == empty_value

    def test_mapping_builds_typed_options(self) -> None:
        normalized = normalize_capability_declarations(
            {
                "top_logprobs": {"minimum": 100},
                "reasoning": {"effort": "high", "summary": "auto"},
                "structured_output": {"formats": ["json_schema"]},
            },
            dialect_id="openai_chat",
            outcomes=_outcomes(),
        )
        assert normalized["top_logprobs"] == TopLogprobsOptions(minimum=100)
        assert normalized["reasoning"] == ReasoningOptions(
            effort="high", summary="auto"
        )
        assert normalized["structured_output"] == StructuredOutputOptions(
            formats=("json_schema",)
        )

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"reasoning": None}, "cannot be null"),
            ({"not_a_capability": True}, "unknown capability key"),
            ({"reasoning": {"unknown": True}}, "unknown option"),
            ({"top_logprobs": {"minimum": 0}}, "minimum must be >= 1"),
            ({"top_logprobs": {"minimum": True}}, "minimum must be an integer"),
            (
                {"reasoning": {"effort": "high", "budget_tokens": 10}},
                "mutually exclusive",
            ),
            ({"reasoning": {"summary": "banana"}}, "summary must be one of"),
            ({"function_tools": {"parallel": 1}}, "parallel must be a boolean"),
            (
                {"structured_output": {"formats": ["xml"]}},
                "is not one of",
            ),
            (
                {"multimodal_input": {"modalities": ["audio"]}},
                "is not one of",
            ),
        ],
    )
    def test_invalid_declarations_are_rejected(self, raw: object, message: str) -> None:
        with pytest.raises(CapabilityConfigError, match=message):
            normalize_capability_declarations(
                raw, dialect_id="openai_chat", outcomes=_outcomes()
            )

    def test_capability_subtree_must_be_a_mapping(self) -> None:
        with pytest.raises(CapabilityConfigError, match="must be a mapping"):
            normalize_capability_declarations(
                None, dialect_id="openai_chat", outcomes=_outcomes()
            )

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({1: True}, "capability names must be strings"),
            ({"reasoning": {1: True}}, "option names must be strings"),
            (
                {"structured_output": {"formats": "json_schema"}},
                "must be a sequence",
            ),
            ({"reasoning": "high"}, "false, true, or a mapping"),
        ],
    )
    def test_declaration_mapping_shapes_are_checked(
        self, raw: object, message: str
    ) -> None:
        with pytest.raises(CapabilityConfigError, match=message):
            normalize_capability_declarations(
                raw, dialect_id="openai_chat", outcomes=_outcomes()
            )

    @pytest.mark.parametrize("budget", [True, 0])
    def test_reasoning_budget_must_be_a_positive_integer(self, budget: object) -> None:
        with pytest.raises(CapabilityConfigError, match="integer|>= 1"):
            normalize_capability_declarations(
                {"reasoning": {"budget_tokens": budget}},
                dialect_id="openai_chat",
                outcomes=_outcomes(),
            )

    def test_option_tuples_reject_wrong_container_and_duplicates(self) -> None:
        with pytest.raises(TypeError, match="tuple of strings"):
            HostedToolsOptions(cast(Any, ["web_search"]))
        with pytest.raises(ValueError, match="duplicate"):
            HostedToolsOptions(("web_search", "web_search"))

    def test_dialect_outcome_row_must_be_complete_and_known(self) -> None:
        with pytest.raises(CapabilityConfigError, match="missing capability outcome"):
            normalize_dialect_capability_outcomes({})
        with pytest.raises(CapabilityConfigError, match="unknown capability outcome"):
            normalize_dialect_capability_outcomes(
                {**_outcomes(), "other": DialectCapabilityStatus.RESERVED}
            )
        bad = _outcomes()
        bad["reasoning"] = "maybe"
        with pytest.raises(CapabilityConfigError, match="invalid outcome"):
            normalize_dialect_capability_outcomes(bad)

    def test_unsupported_allows_explicit_false_but_not_enablement(self) -> None:
        outcomes = _outcomes(reasoning=DialectCapabilityStatus.UNSUPPORTED)
        assert normalize_capability_declarations(
            {"reasoning": False},
            dialect_id="anthropic_messages",
            outcomes=outcomes,
        ) == {"reasoning": False}
        with pytest.raises(CapabilityConfigError, match="does not support"):
            normalize_capability_declarations(
                {"reasoning": True},
                dialect_id="anthropic_messages",
                outcomes=outcomes,
            )

    @pytest.mark.parametrize("value", [True, False])
    def test_reserved_key_is_rejected_even_when_disabled(self, value: bool) -> None:
        with pytest.raises(CapabilityConfigError, match="is reserved"):
            normalize_capability_declarations(
                {"fim": value},
                dialect_id="openai_chat",
                outcomes=_outcomes(fim=DialectCapabilityStatus.RESERVED),
            )

    def test_canonical_serialization_expands_true_and_retains_false(self) -> None:
        normalized = normalize_capability_declarations(
            {
                "reasoning": {"summary": "auto", "effort": "high"},
                "input_scoring": True,
                "hosted_tools": False,
                "top_logprobs": {"minimum": 1},
            },
            dialect_id="openai_chat",
            outcomes=_outcomes(),
        )
        assert capability_declarations_to_json(normalized) == {
            "hosted_tools": False,
            "input_scoring": {},
            "reasoning": {"effort": "high", "summary": "auto"},
            "top_logprobs": {},
        }
        assert canonical_capability_json(normalized) == (
            '{"hosted_tools":false,"input_scoring":{},'
            '"reasoning":{"effort":"high","summary":"auto"},'
            '"top_logprobs":{}}'
        )

    def test_serialization_rejects_unknown_or_mismatched_normalized_values(
        self,
    ) -> None:
        with pytest.raises(CapabilityConfigError, match="unknown capability key"):
            capability_declarations_to_json(cast(Any, {"typo": False}))
        with pytest.raises(CapabilityConfigError, match="wrong capability"):
            capability_declarations_to_json(
                cast(Any, {"reasoning": TopLogprobsOptions()})
            )


class TestLegacyCapabilityAmbiguity:
    @pytest.mark.parametrize(
        ("capability", "legacy_arguments", "paths"),
        [
            ("reasoning", {"reasoning_effort": "high"}, ("reasoning_effort",)),
            ("function_tools", {"tools": []}, ("tools",)),
            (
                "structured_output",
                {"extra_body": {"response_format": {"type": "json_object"}}},
                ("extra_body.response_format",),
            ),
            ("input_scoring", {"echo": False}, ("echo",)),
            ("stateful_session", {"session_id": "prior"}, ("session_id",)),
            ("fim", {"suffix": "tail"}, ("suffix",)),
        ],
    )
    def test_maps_legacy_semantics_to_declared_capability(
        self,
        capability: str,
        legacy_arguments: dict[str, object],
        paths: tuple[str, ...],
    ) -> None:
        assert legacy_capability_ambiguities({capability: False}, legacy_arguments) == {
            capability: paths
        }

    def test_numeric_logprobs_owns_switch_and_breadth(self) -> None:
        declarations = {"sampled_logprobs": True, "top_logprobs": {"minimum": 5}}
        assert legacy_capability_ambiguities(declarations, {"logprobs": 10}) == {
            "sampled_logprobs": ("logprobs",),
            "top_logprobs": ("logprobs",),
        }
        assert legacy_capability_ambiguities(declarations, {"logprobs": True}) == {
            "sampled_logprobs": ("logprobs",)
        }
        assert legacy_capability_ambiguities(declarations, {"logprobs": 0}) == {
            "sampled_logprobs": ("logprobs",)
        }
        assert legacy_capability_ambiguities(declarations, {"top_logprobs": 5}) == {
            "sampled_logprobs": ("top_logprobs",),
            "top_logprobs": ("top_logprobs",),
        }

    def test_ordinary_sampling_and_unrelated_capabilities_do_not_conflict(
        self,
    ) -> None:
        assert (
            legacy_capability_ambiguities(
                {"reasoning": True, "top_logprobs": True},
                {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 128,
                    "stop": ["done"],
                    "seed": 7,
                },
            )
            == {}
        )

    def test_active_legacy_values_project_intents_without_sampling_noise(self) -> None:
        intents = legacy_capability_intents(
            {
                "temperature": 0.2,
                "echo": True,
                "logprobs": 8,
                "reasoning_effort": "high",
                "extra_body": {"response_format": {"type": "json_object"}},
            },
            source="tasks.eval.infer_args",
        )

        assert set(intents) == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
            "reasoning",
            "structured_output",
        }
        assert intents["top_logprobs"].minimums == {"minimum": 8}
        assert intents["top_logprobs"].sources == ("tasks.eval.infer_args.logprobs",)
        assert all(
            "temperature" not in source
            for intent in intents.values()
            for source in intent.sources
        )

    def test_inactive_legacy_values_do_not_require_capabilities(self) -> None:
        assert (
            legacy_capability_intents(
                {
                    "echo": False,
                    "return_logprobs": False,
                    "top_logprobs": 0,
                    "tools": [],
                    "server_tools": (),
                    "reasoning_effort": None,
                },
                source="models.m.args",
            )
            == {}
        )

    def test_validator_reports_both_sources_and_remedy(self) -> None:
        with pytest.raises(
            CapabilityConfigError,
            match=(
                r"models\.m\.capabilities and tasks\.score\.infer_args.*"
                r"reasoning via reasoning_effort.*remove the legacy argument"
            ),
        ):
            validate_no_legacy_capability_ambiguity(
                {"reasoning": {"effort": "high"}},
                {"reasoning_effort": "high"},
                canonical_source="models.m.capabilities",
                legacy_source="tasks.score.infer_args",
            )

    def test_validator_is_a_noop_when_surfaces_do_not_overlap(self) -> None:
        validate_no_legacy_capability_ambiguity(
            {"reasoning": True},
            {"temperature": 0.2},
            canonical_source="models.m.capabilities",
            legacy_source="tasks.eval.infer_args",
        )

    def test_legacy_projection_accounts_for_invalid_and_alternate_spellings(
        self,
    ) -> None:
        intents = legacy_capability_intents(
            cast(
                Any,
                {
                    1: "ignored non-string key",
                    "logprobs": "invalid breadth",
                    "top_logprobs": 5,
                },
            ),
            source="tasks.eval.infer_args",
        )

        assert intents["sampled_logprobs"].required
        assert intents["top_logprobs"].minimums == {"minimum": 5}
        assert (
            legacy_capability_intents(
                {"logprobs": False}, source="tasks.eval.infer_args"
            )
            == {}
        )


class TestIntentAndModelProfile:
    def test_request_defaults_and_intent_are_serializable(self) -> None:
        source = {"reasoning.effort": "high"}
        defaults = RequestDefaults(source)
        source["reasoning.effort"] = "low"
        intent = CapabilityIntent(
            key="top_logprobs",
            required=True,
            minimums={"minimum": 100},
            request_defaults=defaults,
            sources=("clp",),
        )
        assert intent.to_json_value() == {
            "key": "top_logprobs",
            "required": True,
            "minimums": {"minimum": 100},
            "request_defaults": {"reasoning.effort": "high"},
            "sources": ["clp"],
        }
        with pytest.raises(TypeError):
            cast(dict[str, object], defaults.values)["new"] = 1

    def test_json_values_are_copied_and_invalid_values_are_rejected(self) -> None:
        defaults = RequestDefaults(cast(Any, {"values": (1, 2.5)}))
        assert defaults.to_json_value() == {"values": [1, 2.5]}

        with pytest.raises(ValueError, match="non-finite"):
            RequestDefaults({"bad": float("nan")})
        with pytest.raises(TypeError, match="keys must be strings"):
            RequestDefaults(cast(Any, {1: "bad"}))
        with pytest.raises(TypeError, match="non-JSON"):
            RequestDefaults({"bad": cast(Any, object())})

    @pytest.mark.parametrize(
        ("key", "required", "message"),
        [
            ("not-a-capability", True, "unknown capability key"),
            ("reasoning", cast(Any, 1), "required must be a boolean"),
        ],
    )
    def test_intent_identity_is_validated(
        self, key: object, required: object, message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            CapabilityIntent(cast(Any, key), cast(Any, required))

    def test_intent_aggregation_preserves_required_or_and_canonical_sources(
        self,
    ) -> None:
        optional = CapabilityIntent(
            "reasoning",
            False,
            sources=("task-z", "task-a"),
        )
        required = CapabilityIntent(
            "reasoning",
            True,
            sources=("task-m",),
        )

        optional_only = aggregate_capability_intents([optional])["reasoning"]
        combined = aggregate_capability_intents([optional, required])["reasoning"]

        assert optional.sources == ("task-a", "task-z")
        assert optional_only.required is False
        assert optional_only.sources == ("task-a", "task-z")
        assert combined.required is True
        assert combined.sources == ("task-a", "task-m", "task-z")

    def test_intent_aggregation_does_not_collapse_boolean_and_integer(self) -> None:
        intents = [
            CapabilityIntent("top_logprobs", True, minimums={"minimum": True}),
            CapabilityIntent("top_logprobs", True, minimums={"minimum": 1}),
        ]

        with pytest.raises(CapabilityConfigError, match="incompatible minimum"):
            aggregate_capability_intents(intents)

    def test_model_profile_preserves_sourced_outcomes(self) -> None:
        profile = ModelCapabilityProfile(
            entries={
                "input_scoring": ModelCapabilityEntry(
                    status=ModelCapabilityStatus.SUPPORTED,
                    source="recipe:qwen",
                ),
                "reasoning": ModelCapabilityEntry(
                    status=ModelCapabilityStatus.UNKNOWN,
                    source="hosted-config",
                    verifier="probe_reasoning",
                ),
            },
            authoritative=False,
        )
        assert profile.to_json_value() == {
            "authoritative": False,
            "entries": {
                "input_scoring": {
                    "status": "supported",
                    "source": "recipe:qwen",
                },
                "reasoning": {
                    "status": "unknown",
                    "source": "hosted-config",
                    "verifier": "probe_reasoning",
                },
            },
        }
        with pytest.raises(TypeError):
            cast(dict[str, object], profile.entries)["fim"] = ModelCapabilityEntry(
                status=ModelCapabilityStatus.SUPPORTED,
                source="test",
            )

    def test_unsupported_and_unknown_model_entries_need_evidence(self) -> None:
        with pytest.raises(ValueError, match="requires a reason"):
            ModelCapabilityEntry(
                status=ModelCapabilityStatus.UNSUPPORTED,
                source="catalog",
            )
        with pytest.raises(ValueError, match="requires a verifier"):
            ModelCapabilityEntry(
                status=ModelCapabilityStatus.UNKNOWN,
                source="catalog",
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"status": "supported", "source": "catalog"}, "ModelCapabilityStatus"),
            (
                {
                    "status": ModelCapabilityStatus.SUPPORTED,
                    "source": "catalog",
                    "reason": "",
                },
                "reason",
            ),
            (
                {
                    "status": ModelCapabilityStatus.SUPPORTED,
                    "source": "catalog",
                    "verifier": "",
                },
                "verifier",
            ),
        ],
    )
    def test_model_entry_types_and_evidence_strings_are_validated(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(TypeError, match=message):
            ModelCapabilityEntry(**cast(Any, kwargs))

    @pytest.mark.parametrize(
        ("entries", "authoritative", "message"),
        [
            ({}, cast(Any, 1), "authoritative"),
            ({"typo": cast(Any, object())}, False, "unknown capability key"),
            ({"reasoning": cast(Any, object())}, False, "ModelCapabilityEntry"),
        ],
    )
    def test_model_profile_rejects_invalid_entries(
        self, entries: dict[str, object], authoritative: object, message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            ModelCapabilityProfile(cast(Any, entries), cast(Any, authoritative))

    def test_profile_serializes_optional_reason_evidence(self) -> None:
        profile = ModelCapabilityProfile(
            {
                "reasoning": ModelCapabilityEntry(
                    ModelCapabilityStatus.UNSUPPORTED,
                    "catalog",
                    reason="checkpoint architecture has no reasoning mode",
                )
            }
        )

        entries = profile.to_json_value()["entries"]
        assert isinstance(entries, dict)
        reasoning = entries["reasoning"]
        assert isinstance(reasoning, dict)
        reason = reasoning["reason"]
        assert isinstance(reason, str)
        assert reason.startswith("checkpoint")


class TestDialectCapabilityDecisions:
    def test_supported_binding_runs_ordinary_validator(self) -> None:
        def require_large_breadth(options: object) -> None:
            assert isinstance(options, TopLogprobsOptions)
            if options.minimum < 10:
                raise ValueError("breadth too small")

        binding = DialectCapabilityBinding(
            key="top_logprobs",
            request_leaves=("scoring.top_logprobs",),
            response_channels=("output_scoring",),
            _config_validator=require_large_breadth,
        )
        decision = Supported(binding)
        decision.binding.validate_config(TopLogprobsOptions(minimum=10))
        with pytest.raises(ValueError, match="breadth too small"):
            decision.binding.validate_config(TopLogprobsOptions(minimum=1))
        with pytest.raises(CapabilityConfigError, match="another capability"):
            decision.binding.validate_config(FunctionToolsOptions())

    def test_binding_rejects_wildcards_and_unsupported_reason_is_required(self) -> None:
        with pytest.raises(ValueError, match="wildcards"):
            DialectCapabilityBinding(
                key="reasoning",
                request_leaves=("reasoning.*",),
            )
        with pytest.raises(TypeError, match="non-empty string"):
            Unsupported("")
        assert Unsupported("native API has no logprobs").reason.startswith("native")

    def test_binding_identity_and_default_validator_are_checked(self) -> None:
        binding = DialectCapabilityBinding("input_scoring")
        binding.validate_config(InputScoringOptions())

        with pytest.raises(ValueError, match="unknown capability key"):
            DialectCapabilityBinding(cast(Any, "typo"))
        with pytest.raises(TypeError, match="validator must be callable"):
            DialectCapabilityBinding("input_scoring", _config_validator=cast(Any, None))
