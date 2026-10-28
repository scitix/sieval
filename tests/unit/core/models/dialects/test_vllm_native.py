"""Tests for the vLLM native completions dialect.

Requests go through ``VllmNativeDialect.arun`` (the full audit, prepare,
execute and output-contract path) over an ``httpx.MockTransport``.  The reply
shapes are vLLM 0.26.0's, recorded from a live server: a ``prompt_logprobs``
position is keyed by token id and lists the scored token beside its top-k.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import math
from typing import Any

import httpx
import pytest

from sieval.core.models.capabilities import Supported
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import (
    Dialect,
    OutputContractError,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialect_registry import DIALECT_SPECS, RequestSeedSupport
from sieval.core.models.dialects.vllm_native import (
    CAPABILITY_DECISIONS,
    VllmNativeDialect,
    _LegacyPlan,
    _ScoringVerifier,
    _StructuredOutputsVerifier,
    completions_url,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    DialectOptions,
    Request,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    StructuredOutput,
    StructuredOutputParams,
    TextPart,
    TokenLogprob,
    TopKEntry,
)

_PROMPT = "The capital"


class _Server:
    """Answer each POST with the next canned reply and keep the requests."""

    def __init__(self, *replies: object, status: int = 200) -> None:
        self.replies = list(replies)
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # Python's encoder, as vLLM's: ``-Infinity`` and ``NaN`` stay literal.
        return httpx.Response(self.status, text=json.dumps(self.replies.pop(0)))

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def _dialect(
    server: _Server,
    *,
    base_url: str = "http://host:8000/v1",
    credential: str | None = "local",
) -> VllmNativeDialect:
    connection = AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(server)),
        credential,
    )
    return VllmNativeDialect(connection, "Qwen/Qwen2.5-0.5B")


def _choice(text: str = " Paris", index: int = 0, **fields: Any) -> dict:
    choice = {"index": index, "text": text, "finish_reason": "length"}
    choice.update(fields)
    return choice


def _reply(*choices: dict, prompt_tokens: int = 2, completion_tokens: int = 1) -> dict:
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "model": "Qwen/Qwen2.5-0.5B",
        "system_fingerprint": "vllm-0.26.0-0c752b4e",
        "choices": list(choices) or [_choice()],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": None,
        },
    }


def _entry(logprob: float, rank: int, text: str | None) -> dict:
    return {"logprob": logprob, "rank": rank, "decoded_token": text}


#: "The capital" scored with prompt_logprobs=2: " capital" ranks 522, so it is
#: listed before the two entries that rank 1 and 2.
_PROMPT_LOGPROBS = [
    None,
    {
        "6722": _entry(-9.45, 522, " capital"),
        "2701": _entry(-0.32, 1, " following"),
        "40389": _entry(-4.95, 2, " Purple"),
    },
]
_PROMPT_IDS = [785, 6722]

#: The same prompt at breadth 0: the scored token alone.
_SCORED_ONLY = [None, {"6722": _entry(-9.45, 522, " capital")}]


#: The sampled " Paris" at rank 1 of a top-2.
_SAMPLED = {
    "tokens": [" Paris"],
    "token_logprobs": [-0.1],
    "top_logprobs": [{" Paris": -0.1, " Lyon": -2.0}],
}


def _scored_choice(**fields: Any) -> dict:
    return _choice(
        prompt_logprobs=_PROMPT_LOGPROBS,
        prompt_token_ids=_PROMPT_IDS,
        token_ids=[12095],
        logprobs=_SAMPLED,
        **fields,
    )


def _top_k_scoring(**kwargs: Any) -> Request:
    """Input scoring with a top-2, which the IR pairs with sampled logprobs."""

    return _request(input_scoring=True, sampled_logprobs=True, top_logprobs=2, **kwargs)


def _request(
    text: str = _PROMPT,
    *,
    sampling: SamplingParams | None = None,
    input_scoring: bool = False,
    sampled_logprobs: bool = False,
    top_logprobs: int = 0,
    structured_output: StructuredOutputParams | None = None,
    options: dict | None = None,
    stream: bool = False,
) -> Request:
    return Request(
        input=CompletionInput(text),
        sampling=sampling or SamplingParams(),
        scoring=ScoringParams(
            input_scoring=input_scoring,
            sampled_logprobs=sampled_logprobs,
            top_logprobs=top_logprobs,
        ),
        structured_output=structured_output or StructuredOutputParams(),
        scheduling=SchedulingParams(stream=stream),
        dialect_options=(
            DialectOptions("vllm_native", options) if options is not None else None
        ),
    )


async def _run(reply: object, req: Request, **kwargs: Any) -> tuple[Any, _Server]:
    server = _Server(reply)
    response = await _dialect(server, **kwargs).arun(req)
    return response, server


class TestDescriptor:
    def test_is_an_active_canonical_dialect_with_a_request_seed(self):
        spec = DIALECT_SPECS["vllm_native"]

        assert isinstance(_dialect(_Server()), Dialect)
        assert spec.connection_family == "async_http_json"
        assert spec.request_seed_support is RequestSeedSupport.SUPPORTED
        assert spec.engine_level_seed is False
        assert spec.input_kinds == ("completion",)

    def test_supports_scoring_and_structured_output_only(self):
        supported = {
            key
            for key, decision in CAPABILITY_DECISIONS.items()
            if isinstance(decision, Supported)
        }

        assert supported == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
            "structured_output",
        }

    def test_requires_an_http_connection_and_model_id(self):
        with pytest.raises(TypeError, match="AsyncHTTPJSONConnection"):
            VllmNativeDialect(object(), "m")
        connection = AsyncHTTPJSONConnection(httpx.AsyncClient(), None)
        with pytest.raises(ValueError, match="requested_model_id"):
            VllmNativeDialect(connection, "")

    def test_legacy_plan_offers_the_supported_row(self):
        assert _LegacyPlan().available_capabilities == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
            "structured_output",
        }


class TestCompletionsUrl:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("http://host:8000/v1", "http://host:8000/v1/completions"),
            ("http://host:8000/v1/", "http://host:8000/v1/completions"),
            ("http://gw/vllm/v1?x=1", "http://gw/vllm/v1/completions"),
        ],
    )
    def test_completions_sits_under_the_openai_base(self, base, expected):
        assert str(completions_url(httpx.URL(base))) == expected


class TestLowering:
    @pytest.mark.anyio
    async def test_posts_the_prompt_with_the_model_and_bearer_auth(self):
        response, server = await _run(_reply(), _request("hi"))

        request = server.requests[0]
        assert str(request.url) == "http://host:8000/v1/completions"
        assert request.headers["Authorization"] == "Bearer local"
        assert server.body == {
            "model": "Qwen/Qwen2.5-0.5B",
            "prompt": "hi",
            "stream": False,
        }
        assert response.texts == (" Paris",)

    @pytest.mark.anyio
    async def test_no_credential_sends_no_authorization(self):
        _, server = await _run(_reply(), _request(), credential=None)

        assert "Authorization" not in server.requests[0].headers

    @pytest.mark.anyio
    async def test_every_sampling_leaf_is_a_top_level_field(self):
        sampling = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            top_k=20,
            max_tokens=64,
            stop=("\n",),
            seed=7,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )

        _, server = await _run(
            _reply(),
            _request(sampling=sampling, options={"min_p": 0.05}),
        )

        assert server.body == {
            "model": "Qwen/Qwen2.5-0.5B",
            "prompt": _PROMPT,
            "stream": False,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 20,
            "max_tokens": 64,
            "stop": ["\n"],
            "seed": 7,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "min_p": 0.05,
        }

    @pytest.mark.anyio
    async def test_n_is_sent_only_above_one(self):
        reply = _reply(_choice(index=0), _choice(index=1))

        _, server = await _run(reply, _request(sampling=SamplingParams(n=2)))

        assert server.body["n"] == 2

    @pytest.mark.anyio
    async def test_request_params_are_the_body_without_model_and_prompt(self):
        response, _ = await _run(
            _reply(), _request("secret", sampling=SamplingParams(seed=1))
        )

        assert response.request_params == {"stream": False, "seed": 1}

    @pytest.mark.anyio
    async def test_stream_is_a_noop_sent_as_one_post(self):
        _, server = await _run(_reply(), _request(stream=True))

        assert len(server.requests) == 1
        assert server.body["stream"] is False
        assert "stream_options" not in server.body

    @pytest.mark.anyio
    async def test_input_scoring_sends_prompt_logprobs_without_echo(self):
        positions = _SCORED_ONLY
        choice = _choice(prompt_logprobs=positions, prompt_token_ids=_PROMPT_IDS)

        _, server = await _run(_reply(choice), _request(input_scoring=True))

        assert server.body["prompt_logprobs"] == 0
        assert server.body["return_token_ids"] is True
        assert "echo" not in server.body
        assert "logprobs" not in server.body

    @pytest.mark.anyio
    async def test_input_top_k_sends_both_breadths(self):
        _, server = await _run(_reply(_scored_choice()), _top_k_scoring())

        assert server.body["prompt_logprobs"] == 2
        assert server.body["logprobs"] == 2

    @pytest.mark.anyio
    async def test_sampled_logprobs_send_logprobs_with_token_ids(self):
        choice = _choice(
            logprobs={
                "tokens": [" Paris"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{" Paris": -0.1, " Lyon": -2.0}],
            },
            token_ids=[12095],
        )

        _, server = await _run(
            _reply(choice), _request(sampled_logprobs=True, top_logprobs=2)
        )

        assert server.body["logprobs"] == 2
        assert server.body["return_token_ids"] is True
        assert "prompt_logprobs" not in server.body

    @pytest.mark.anyio
    async def test_json_schema_lowers_to_structured_outputs_json(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        params = StructuredOutputParams(
            format="json_schema", schema=schema, name="person", strict=True
        )

        response, server = await _run(
            _reply(_choice('{"name":"Ann"}')), _request(structured_output=params)
        )

        assert server.body["structured_outputs"] == {"json": schema}
        assert "response_format" not in server.body
        assert response.structured_output == StructuredOutput({"name": "Ann"})

    @pytest.mark.anyio
    async def test_json_object_lowers_to_structured_outputs_json_object(self):
        _, server = await _run(
            _reply(_choice("{}")),
            _request(structured_output=StructuredOutputParams(format="json_object")),
        )

        assert server.body["structured_outputs"] == {"json_object": True}


class TestRejections:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "match"),
        [
            (_request(sampling=SamplingParams(max_tokens=0)), "max_tokens >= 1"),
            (_request(options={"prefill": "x"}), "prefill"),
            (_request(options={"prefix": "x"}), "prefill"),
            (_request(options={"guided_json": {}}), "unknown vLLM option"),
            (_request(options={"logprobs": 5}), "scoring.sampled_logprobs"),
            (_request(options={"prompt_logprobs": 0}), "scoring.input_scoring"),
            (_request(options={"seed": 1}), "sampling.seed"),
            (_request(options={"response_format": {}}), "structured_output"),
            (_request(options={"min_p": None}), "null option"),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={}, strict=False
                    )
                ),
                "cannot relax",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_object", name="x"
                    )
                ),
                "require format='json_schema'",
            ),
        ],
    )
    async def test_refused_before_io(self, req, match):
        server = _Server(_reply())

        with pytest.raises(RequestAuditError, match=match):
            await _dialect(server).arun(req)

        assert server.requests == []

    @pytest.mark.anyio
    async def test_chat_input_is_refused(self):
        server = _Server(_reply())
        req = Request(input=ChatInput((ChatMessage("user", (TextPart("x"),)),)))

        with pytest.raises(RequestAuditError, match="does not support"):
            await _dialect(server).arun(req)

        assert server.requests == []

    @pytest.mark.anyio
    async def test_suffix_is_refused(self):
        from sieval.core.models.dialect import DialectError

        server = _Server(_reply())

        # The plan refuses it first: fim is unsupported on this row.
        with pytest.raises(DialectError, match="unavailable capability 'fim'"):
            await _dialect(server).arun(Request(CompletionInput("x", suffix="y")))

        assert server.requests == []

    @pytest.mark.anyio
    async def test_every_forwarded_option_reaches_the_body(self):
        _, server = await _run(
            _reply(),
            _request(
                options={
                    "min_p": 0.1,
                    "repetition_penalty": 1.1,
                    "stop_token_ids": [13],
                }
            ),
        )

        assert server.body["min_p"] == 0.1
        assert server.body["repetition_penalty"] == 1.1
        assert server.body["stop_token_ids"] == [13]

    @pytest.mark.anyio
    async def test_empty_options_are_a_noop(self):
        _, server = await _run(_reply(), _request(options={}))

        assert set(server.body) == {"model", "prompt", "stream"}


class TestWireEvidence:
    def test_scoring_verifier_checks_breadth_and_token_ids(self):
        verifier = _ScoringVerifier("prompt_logprobs", 2)

        assert verifier.verify({"prompt_logprobs": 2, "return_token_ids": True}) is None
        assert "breadth" in str(
            verifier.verify({"prompt_logprobs": 3, "return_token_ids": True})
        )
        assert "breadth" in str(
            verifier.verify({"prompt_logprobs": True, "return_token_ids": True})
        )
        assert "return_token_ids" in str(verifier.verify({"prompt_logprobs": 2}))

    def test_structured_outputs_verifier_compares_the_whole_field(self):
        verifier = _StructuredOutputsVerifier({"json_object": True})

        assert verifier.verify({"structured_outputs": {"json_object": True}}) is None
        assert verifier.verify({"structured_outputs": {"json": {}}}) is not None
        assert verifier.verify({}) is not None

    def test_finish_rejects_a_tampered_body(self):
        dialect = _dialect(_Server())
        req = _request(input_scoring=True)
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, _LegacyPlan())
        prepared = dialect.prepare(req, audit)
        tampered = PreparedRequest(
            operation=prepared.operation,
            body={**prepared.thaw_body(), "return_token_ids": False},
            context=prepared.context,
        )

        with pytest.raises(RequestAuditError, match="return_token_ids"):
            audit.finish(tampered)


class TestInputScoring:
    @pytest.mark.anyio
    async def test_scored_token_and_rank_ordered_top_k(self):
        response, _ = await _run(_reply(_scored_choice()), _top_k_scoring())

        scoring = response.input_scoring
        assert scoring.token_logprobs == (
            TokenLogprob(token="The", logprob=None, token_id=785),
            TokenLogprob(token=" capital", logprob=-9.45, token_id=6722),
        )
        # The scored token ranks 522, below the top 2, so it is not in them.
        assert scoring.top_logprobs == (
            (),
            (
                TopKEntry(token=" following", logprob=-0.32, token_id=2701),
                TopKEntry(token=" Purple", logprob=-4.95, token_id=40389),
            ),
        )

    @pytest.mark.anyio
    async def test_a_scored_token_inside_the_top_k_is_listed_once(self):
        positions = [
            None,
            {"6722": _entry(-0.5, 1, " capital"), "11": _entry(-1.0, 2, ",")},
        ]
        choice = _scored_choice()
        choice["prompt_logprobs"] = positions

        response, _ = await _run(_reply(choice), _top_k_scoring())

        assert [entry.token for entry in response.input_scoring.top_logprobs[1]] == [
            " capital",
            ",",
        ]

    @pytest.mark.anyio
    async def test_a_rank_tie_beside_the_top_k_stays_out_of_it(self):
        # " capital" ties the rank-2 entry but was not selected, so it is
        # listed beside the top 2 with rank 2.
        positions = [
            None,
            {
                "6722": _entry(-4.95, 2, " capital"),
                "2701": _entry(-0.32, 1, " following"),
                "40389": _entry(-4.95, 2, " Purple"),
            },
        ]
        choice = _scored_choice()
        choice["prompt_logprobs"] = positions

        response, _ = await _run(_reply(choice), _top_k_scoring())

        assert [entry.token for entry in response.input_scoring.top_logprobs[1]] == [
            " following",
            " Purple",
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("count", [1, 4])
    async def test_a_position_without_k_or_k_plus_one_entries_is_refused(self, count):
        filler = {str(100 + i): _entry(-2.0 - i, 2 + i, f"t{i}") for i in range(3)}
        position = {"6722": _entry(-0.5, 1, " capital"), **filler}
        choice = _scored_choice()
        choice["prompt_logprobs"] = [None, dict(list(position.items())[:count])]

        with pytest.raises(OutputContractError, match=f"lists {count} entries"):
            await _run(_reply(choice), _top_k_scoring())

    @pytest.mark.anyio
    async def test_breadth_zero_carries_no_input_top_k(self):
        positions = _SCORED_ONLY
        choice = _choice(prompt_logprobs=positions, prompt_token_ids=_PROMPT_IDS)

        response, _ = await _run(_reply(choice), _request(input_scoring=True))

        assert response.input_scoring.top_logprobs is None
        # Generated-token scores are not requested, and none are invented.
        assert response.logprobs is None

    @pytest.mark.anyio
    async def test_reaches_model_output_as_echo_did(self):
        """The legacy view puts input scores first, as the echo split did."""

        from sieval.core.models._legacy_bridge import response_to_model_output

        positions = _SCORED_ONLY
        choice = _choice(prompt_logprobs=positions, prompt_token_ids=_PROMPT_IDS)
        response, _ = await _run(_reply(choice), _request(input_scoring=True))

        output = response_to_model_output(
            {"model": "m", "api_base": None, "default_params": {}}, response
        )

        assert output.logprobs_tokens == ["The", " capital"]
        assert output.logprobs == [None, -9.45]
        assert output.usage is not None
        assert output.usage["input_tokens"] == 2

    @pytest.mark.anyio
    async def test_first_token_text_is_empty_when_the_rest_does_not_spell_it(self):
        positions = [None, {"6722": _entry(-1.0, 1, "<unk>")}]
        choice = _choice(prompt_logprobs=positions, prompt_token_ids=_PROMPT_IDS)

        response, _ = await _run(_reply(choice), _request(input_scoring=True))

        assert response.input_scoring.token_logprobs[0].token == ""

    @pytest.mark.anyio
    async def test_negative_infinity_is_kept(self):
        positions = [None, {"6722": _entry(-math.inf, 9, " capital")}]
        choice = _choice(prompt_logprobs=positions, prompt_token_ids=_PROMPT_IDS)

        response, _ = await _run(_reply(choice), _request(input_scoring=True))

        assert response.input_scoring.token_logprobs[1].logprob == -math.inf

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("choice", "prompt_tokens", "match"),
        [
            (_choice(prompt_token_ids=_PROMPT_IDS), 2, "no prompt_logprobs"),
            (
                _choice(prompt_logprobs=_PROMPT_LOGPROBS),
                2,
                "no prompt_token_ids",
            ),
            (
                _choice(prompt_logprobs=[], prompt_token_ids=[]),
                0,
                "scored no prompt tokens",
            ),
            # Truncated: fewer positions than the usage count.
            (
                _choice(prompt_logprobs=_PROMPT_LOGPROBS, prompt_token_ids=_PROMPT_IDS),
                3,
                "partial prompt",
            ),
            (
                _choice(prompt_logprobs=_PROMPT_LOGPROBS, prompt_token_ids=[785]),
                2,
                "partial prompt",
            ),
            (
                _choice(
                    prompt_logprobs=[_PROMPT_LOGPROBS[1], _PROMPT_LOGPROBS[1]],
                    prompt_token_ids=_PROMPT_IDS,
                ),
                2,
                "first prompt position",
            ),
            (
                _choice(prompt_logprobs=_SCORED_ONLY, prompt_token_ids=[785, 9]),
                2,
                "prompt's own token id 9",
            ),
            (
                _choice(
                    prompt_logprobs=[None, {"x": _entry(-1.0, 1, "a")}],
                    prompt_token_ids=_PROMPT_IDS,
                ),
                2,
                "key must be a token id",
            ),
            (
                _choice(
                    prompt_logprobs=[None, {"6722": _entry(-1.0, 0, " capital")}],
                    prompt_token_ids=_PROMPT_IDS,
                ),
                2,
                "rank must be a positive integer",
            ),
            (
                _choice(
                    prompt_logprobs=[None, {"6722": _entry(-1.0, 1, None)}],
                    prompt_token_ids=_PROMPT_IDS,
                ),
                2,
                "skip-tokenizer-init",
            ),
            (
                _choice(
                    prompt_logprobs=[None, {"6722": _entry(math.nan, 1, " c")}],
                    prompt_token_ids=_PROMPT_IDS,
                ),
                2,
                "NaN",
            ),
            (
                _choice(prompt_logprobs=_PROMPT_LOGPROBS, prompt_token_ids=[785, -1]),
                2,
                "non-negative integer",
            ),
            (
                _choice(prompt_logprobs="bad", prompt_token_ids=_PROMPT_IDS),
                2,
                "must be a list",
            ),
        ],
    )
    async def test_incomplete_or_malformed_scores_are_refused(
        self, choice, prompt_tokens, match
    ):
        with pytest.raises(OutputContractError, match=match):
            await _run(
                _reply(choice, prompt_tokens=prompt_tokens),
                _request(input_scoring=True),
            )

    @pytest.mark.anyio
    async def test_without_usage_the_token_ids_still_bound_the_prompt(self):
        choice = _choice(prompt_logprobs=_SCORED_ONLY, prompt_token_ids=_PROMPT_IDS)
        reply = _reply(choice)
        reply["usage"] = None

        response, _ = await _run(reply, _request(input_scoring=True))

        assert len(response.input_scoring.token_logprobs) == 2
        assert response.usage is None


class TestSampledLogprobs:
    @pytest.mark.anyio
    async def test_tokens_keep_ids_and_top_k_drops_the_extra_sampled_entry(self):
        # " B" was sampled at rank 3, so vLLM lists it beside the top 2.
        choice = _choice(
            text=" B",
            logprobs={
                "tokens": [" B"],
                "token_logprobs": [-3.0],
                "top_logprobs": [{" A": -0.1, " B": -3.0, " C": -1.0}],
            },
            token_ids=[425],
        )

        response, _ = await _run(
            _reply(choice), _request(sampled_logprobs=True, top_logprobs=2)
        )

        assert response.logprobs == (TokenLogprob(" B", -3.0, token_id=425),)
        assert response.top_logprobs == (
            (TopKEntry(" A", -0.1), TopKEntry(" C", -1.0)),
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("sampled", "top"),
        [
            # " B" sampled inside the top 3; the other two decode to "", so
            # the map is one short.
            (-0.5, {" B": -0.5, "": -0.7}),
            # " B" sampled inside the top 3; another decodes alike and
            # overwrites its value.
            (-0.5, {" B": -0.9, " A": -0.1, " C": -1.0}),
        ],
    )
    async def test_a_decoded_text_collision_is_refused(self, sampled, top):
        choice = _choice(
            text=" B",
            logprobs={
                "tokens": [" B"],
                "token_logprobs": [sampled],
                "top_logprobs": [top],
            },
            token_ids=[425],
        )

        with pytest.raises(OutputContractError, match="decode to the same text"):
            await _run(_reply(choice), _request(sampled_logprobs=True, top_logprobs=3))

    @pytest.mark.anyio
    async def test_breadth_zero_carries_no_top_k(self):
        choice = _choice(
            logprobs={
                "tokens": [" Paris"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{" Paris": -0.1}],
            },
            token_ids=[12095],
        )

        response, _ = await _run(_reply(choice), _request(sampled_logprobs=True))

        assert response.top_logprobs is None
        assert response.logprobs[0].token_id == 12095

    @pytest.mark.anyio
    async def test_n_samples_lift_per_choice(self):
        choices = [
            _choice(
                text=text,
                index=index,
                logprobs={
                    "tokens": [text],
                    "token_logprobs": [-0.5],
                    "top_logprobs": [{text: -0.5}],
                },
                token_ids=[index + 1],
            )
            # Out of index order: the lift orders by index, not position.
            for index, text in ((1, " b"), (0, " a"))
        ]

        response, _ = await _run(
            _reply(*choices),
            _request(
                sampling=SamplingParams(n=2), sampled_logprobs=True, top_logprobs=1
            ),
        )

        assert response.texts == (" a", " b")
        assert [c.tokens[0].token_id for c in response.choice_logprobs] == [1, 2]
        assert response.logprobs is None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("choice", "match"),
        [
            (_choice(token_ids=[1]), "no logprobs"),
            (
                _choice(
                    logprobs={"tokens": [" a"], "token_logprobs": [-1.0]},
                    token_ids=None,
                ),
                "no token_ids",
            ),
            (
                _choice(
                    logprobs={"tokens": [" a", " b"], "token_logprobs": [-1.0]},
                    token_ids=[1],
                ),
                "2 sampled tokens, 1 logprobs",
            ),
            (
                _choice(
                    logprobs={
                        "tokens": [" a"],
                        "token_logprobs": [-1.0],
                        "top_logprobs": [],
                    },
                    token_ids=[1],
                ),
                "top logprobs for 0 of 1",
            ),
            (
                _choice(
                    logprobs={"tokens": [" a"], "token_logprobs": [None]},
                    token_ids=[1],
                ),
                "must be numeric",
            ),
        ],
    )
    async def test_missing_or_misaligned_scores_are_refused(self, choice, match):
        with pytest.raises(OutputContractError, match=match):
            await _run(_reply(choice), _request(sampled_logprobs=True, top_logprobs=1))


class TestReply:
    @pytest.mark.anyio
    async def test_model_fingerprint_usage_and_finish_reason(self):
        response, _ = await _run(_reply(), _request())

        assert response.response_model == "Qwen/Qwen2.5-0.5B"
        assert response.system_fingerprint == "vllm-0.26.0-0c752b4e"
        assert response.finish_reasons == ("length",)
        assert response.usage.input_tokens == 2
        assert response.usage.total_tokens == 3
        assert response.usage.cached_tokens is None

    @pytest.mark.anyio
    async def test_cached_tokens_are_read_when_reported(self):
        reply = _reply()
        reply["usage"]["prompt_tokens_details"] = {"cached_tokens": 1}

        response, _ = await _run(reply, _request())

        assert response.usage.cached_tokens == 1

    @pytest.mark.anyio
    async def test_a_null_finish_reason_reads_as_empty(self):
        response, _ = await _run(_reply(_choice(finish_reason=None)), _request())

        assert response.finish_reasons == ("",)

    @pytest.mark.anyio
    async def test_truncated_structured_output_is_refused(self):
        params = StructuredOutputParams(format="json_object")

        with pytest.raises(OutputContractError, match="not valid JSON"):
            await _run(_reply(_choice('{"name":"')), _request(structured_output=params))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("reply", "match"),
        [
            ([], "must be an object"),
            ({"choices": "x"}, "must be a list"),
            ({"choices": [_choice(index=1)]}, "outside the requested range"),
            ({"choices": [_choice(), _choice()]}, "duplicated choice index 0"),
            ({"choices": []}, r"omitted choice indexes \[0\]"),
            ({"choices": [{"index": 0}]}, "has no text"),
            ({"choices": [_choice()], "model": 3}, "model must be a string"),
            (
                {"choices": [_choice()], "usage": {"prompt_tokens": -1}},
                "usage.prompt_tokens",
            ),
        ],
    )
    async def test_malformed_replies_are_refused(self, reply, match):
        with pytest.raises(OutputContractError, match=match):
            await _run(reply, _request())

    @pytest.mark.anyio
    async def test_invalid_json_is_refused(self):
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(200, text="not json")

        connection = AsyncHTTPJSONConnection(
            httpx.AsyncClient(
                base_url="http://host:8000/v1", transport=httpx.MockTransport(handler)
            ),
            None,
        )

        with pytest.raises(OutputContractError, match="invalid JSON"):
            await VllmNativeDialect(connection, "m").arun(_request())

    @pytest.mark.anyio
    async def test_http_error_carries_the_server_message(self):
        server = _Server(
            {"error": {"message": "max_tokens must be at least 1, got 0."}},
            status=400,
        )

        with pytest.raises(httpx.HTTPStatusError, match="vLLM: max_tokens"):
            await _dialect(server).arun(_request())


class TestPreparedRequestIntegrity:
    @pytest.mark.anyio
    async def test_execute_rejects_a_foreign_operation_or_context(self):
        from sieval.core.models.dialect import DialectError

        dialect = _dialect(_Server())

        with pytest.raises(DialectError, match="unexpected vLLM operation"):
            await dialect.execute(PreparedRequest("generate", {}, None))
        with pytest.raises(DialectError, match="invalid context"):
            await dialect.execute(PreparedRequest("completions", {}, None))

    def test_prepare_refuses_chat_input(self):
        from sieval.core.models.dialect import DialectError

        req = Request(input=ChatInput((ChatMessage("user", (TextPart("x"),)),)))
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(DialectError, match="requires CompletionInput"):
            _dialect(_Server()).prepare(req, audit)

    def test_capabilities_enum_view(self):
        assert _dialect(_Server()).capabilities == VllmNativeDialect.CAPABILITIES
