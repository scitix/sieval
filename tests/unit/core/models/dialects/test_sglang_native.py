"""Tests for the SGLang native ``/generate`` dialect.

Ported from the ``sglang_legacy`` transport tests: URL derivation, token-text
normalization, triple parsing, the radix-cache guard and the ``n > 1`` list
reply.  Requests go through ``SglangNativeDialect.arun`` (the full audit,
prepare, execute and output-contract path) over an ``httpx.MockTransport``,
so no traffic leaves the process.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from typing import Any

import httpx
import pytest

from sieval.core.models.capabilities import Supported
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import (
    Dialect,
    OutputContractError,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialect_registry import DIALECT_SPECS, RequestSeedSupport
from sieval.core.models.dialects.sglang_native import (
    CAPABILITY_DECISIONS,
    SglangNativeDialect,
    _LegacyPlan,
    _LogprobRequestVerifier,
    _MaxNewTokensVerifier,
    _normalize_token_text,
    generate_url,
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
    TextPart,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)


class _Server:
    """Answer each POST with the next canned reply and keep the requests."""

    def __init__(self, *replies: object, status: int = 200) -> None:
        self.replies = list(replies)
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.replies.pop(0))

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def _dialect(
    server: _Server,
    *,
    base_url: str = "http://host:30000/v1",
    credential: str | None = "local",
) -> SglangNativeDialect:
    connection = AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(server)),
        credential,
    )
    return SglangNativeDialect(connection, "m")


def _meta(**overrides: Any) -> dict:
    meta = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "cached_tokens": 0,
        "finish_reason": {"type": "stop"},
    }
    meta.update(overrides)
    return meta


def _request(
    text: str = "x",
    *,
    sampling: SamplingParams | None = None,
    input_scoring: bool = False,
    sampled_logprobs: bool = False,
    top_logprobs: int = 0,
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
        scheduling=SchedulingParams(stream=stream),
        dialect_options=(
            DialectOptions("sglang_native", options) if options is not None else None
        ),
    )


async def _run(reply: object, req: Request, **kwargs: Any) -> tuple[Any, _Server]:
    server = _Server(reply)
    response = await _dialect(server, **kwargs).arun(req)
    return response, server


class TestDescriptor:
    def test_is_an_active_canonical_dialect(self):
        spec = DIALECT_SPECS["sglang_native"]
        server = _Server()

        assert isinstance(_dialect(server), Dialect)
        assert spec.connection_family == "async_http_json"
        assert spec.request_seed_support is RequestSeedSupport.UNSUPPORTED
        assert spec.input_kinds == ("completion",)

    def test_supports_exactly_the_scoring_capabilities(self):
        supported = {
            key
            for key, decision in CAPABILITY_DECISIONS.items()
            if isinstance(decision, Supported)
        }

        assert supported == {"input_scoring", "sampled_logprobs", "top_logprobs"}

    def test_requires_an_http_connection_and_model_id(self):
        with pytest.raises(TypeError, match="AsyncHTTPJSONConnection"):
            SglangNativeDialect(object(), "m")
        connection = AsyncHTTPJSONConnection(httpx.AsyncClient(), None)
        with pytest.raises(ValueError, match="requested_model_id"):
            SglangNativeDialect(connection, "")


class TestGenerateUrl:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("http://host:8000/v1", "http://host:8000/generate"),
            ("http://host:8000/v1/", "http://host:8000/generate"),
            ("http://host:8000", "http://host:8000/generate"),
            ("http://gw/sglang/v1", "http://gw/sglang/generate"),
        ],
    )
    def test_generate_sits_beside_the_openai_base(self, base, expected):
        assert str(generate_url(httpx.URL(base))) == expected


class TestNormalizeTokenText:
    def test_byte_level_markers_become_whitespace(self):
        assert _normalize_token_text("ĠA") == " A"
        assert _normalize_token_text("Ċ") == "\n"
        assert _normalize_token_text("plain") == "plain"

    def test_missing_text_fails_loud(self):
        with pytest.raises(OutputContractError, match="skip-tokenizer-init"):
            _normalize_token_text(None)

    def test_non_string_text_fails(self):
        with pytest.raises(OutputContractError, match="must be a string"):
            _normalize_token_text(7)


class TestLowering:
    @pytest.mark.anyio
    async def test_posts_the_prompt_to_generate_with_bearer_auth(self):
        response, server = await _run(
            {"text": "out", "meta_info": _meta()}, _request("hi")
        )

        request = server.requests[0]
        assert str(request.url) == "http://host:30000/generate"
        assert request.headers["Authorization"] == "Bearer local"
        assert server.body == {"text": "hi", "sampling_params": {}}
        assert response.texts == ("out",)

    @pytest.mark.anyio
    async def test_no_credential_sends_no_authorization(self):
        _, server = await _run(
            {"text": "", "meta_info": _meta()}, _request(), credential=None
        )

        assert "Authorization" not in server.requests[0].headers

    @pytest.mark.anyio
    async def test_every_sampling_leaf_maps_into_sampling_params(self):
        sampling = SamplingParams(
            temperature=0.5,
            top_p=0.9,
            top_k=20,
            max_tokens=64,
            stop=("\n",),
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )

        _, server = await _run(
            {"text": "", "meta_info": _meta()},
            _request(sampling=sampling, options={"min_p": 0.05}),
        )

        # A list, not a dict compare: the order is the retired transport's,
        # so request_params records stay byte-identical across the migration.
        assert list(server.body["sampling_params"].items()) == [
            ("max_new_tokens", 64),
            ("temperature", 0.5),
            ("top_p", 0.9),
            ("top_k", 20),
            ("stop", ["\n"]),
            ("frequency_penalty", 0.1),
            ("presence_penalty", 0.2),
            ("min_p", 0.05),
        ]

    @pytest.mark.anyio
    async def test_scoring_clamp_keeps_max_new_tokens_first(self):
        response, server = await _run(
            {
                "text": "",
                "meta_info": _meta(
                    prompt_tokens=1, input_token_logprobs=[[None, 1, "x"]]
                ),
            },
            _request(sampling=SamplingParams(temperature=0.0), input_scoring=True),
        )

        assert json.dumps(response.request_params["sampling_params"]) == (
            '{"max_new_tokens": 1, "temperature": 0.0}'
        )
        assert list(server.body["sampling_params"]) == [
            "max_new_tokens",
            "temperature",
        ]

    @pytest.mark.anyio
    async def test_request_params_are_the_body_without_the_prompt(self):
        response, _ = await _run(
            {"text": "", "meta_info": _meta()},
            _request("secret", sampling=SamplingParams(temperature=0.0)),
        )

        assert response.request_params == {"sampling_params": {"temperature": 0.0}}

    @pytest.mark.anyio
    async def test_stream_never_reaches_the_wire(self):
        _, server = await _run(
            {"text": "", "meta_info": _meta()}, _request(stream=True)
        )

        assert len(server.requests) == 1
        assert server.body == {"text": "x", "sampling_params": {}}

    @pytest.mark.anyio
    async def test_a_request_seed_is_refused_before_io(self):
        server = _Server({"text": "", "meta_info": _meta()})

        with pytest.raises(RequestAuditError, match="no per-request seed"):
            await _dialect(server).arun(_request(sampling=SamplingParams(seed=7)))

        assert server.requests == []

    @pytest.mark.anyio
    async def test_input_scoring_scores_from_position_zero(self):
        meta = _meta(prompt_tokens=1, input_token_logprobs=[[None, 1, "x"]])

        _, server = await _run(
            {"text": "", "meta_info": meta}, _request(input_scoring=True)
        )

        assert server.body == {
            "text": "x",
            "sampling_params": {"max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": True,
        }

    @pytest.mark.anyio
    async def test_sampled_logprobs_score_output_only(self):
        meta = _meta(
            output_token_logprobs=[[-0.1, 1, "a"]],
            output_top_logprobs=[[[-0.1, 1, "a"]]],
        )

        _, server = await _run(
            {"text": "a", "meta_info": meta},
            _request(sampled_logprobs=True, top_logprobs=3),
        )

        assert server.body["logprob_start_len"] == -1
        assert server.body["top_logprobs_num"] == 3
        # No length set and no input scoring: the server's default stands.
        assert "max_new_tokens" not in server.body["sampling_params"]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("input_scoring", "sampled_logprobs"), [(True, False), (False, True)]
    )
    async def test_a_scoring_request_decodes_at_least_one_token(
        self, input_scoring, sampled_logprobs
    ):
        meta = _meta(
            input_token_logprobs=[[None, 1, "x"]],
            output_token_logprobs=[[-0.1, 2, "y"]],
        )

        _, server = await _run(
            {"text": "y", "meta_info": meta},
            _request(
                sampling=SamplingParams(max_tokens=0),
                input_scoring=input_scoring,
                sampled_logprobs=sampled_logprobs,
            ),
        )

        assert server.body["sampling_params"]["max_new_tokens"] == 1

    @pytest.mark.anyio
    async def test_plain_generation_keeps_an_explicit_zero(self):
        _, server = await _run(
            {"text": "", "meta_info": _meta()},
            _request(sampling=SamplingParams(max_tokens=0)),
        )

        assert server.body["sampling_params"]["max_new_tokens"] == 0


class TestRejections:
    @pytest.mark.anyio
    @pytest.mark.parametrize("key", ["prefill", "prefix"])
    async def test_prefill_is_refused_before_io(self, key):
        server = _Server()

        with pytest.raises(RequestAuditError, match="no prefill"):
            await _dialect(server).arun(_request(options={key: "A:"}))

        assert server.requests == []

    @pytest.mark.anyio
    async def test_unknown_option_is_refused_not_dropped(self):
        server = _Server()

        with pytest.raises(RequestAuditError, match="unknown SGLang option"):
            await _dialect(server).arun(_request(options={"echo": True}))

        assert server.requests == []

    @pytest.mark.anyio
    async def test_option_spelling_a_request_leaf_is_refused(self):
        server = _Server()

        with pytest.raises(RequestAuditError, match=r"sampling\.max_tokens"):
            await _dialect(server).arun(_request(options={"max_new_tokens": 5}))

    @pytest.mark.anyio
    async def test_null_option_is_refused(self):
        server = _Server()

        with pytest.raises(RequestAuditError, match="null option"):
            await _dialect(server).arun(_request(options={"min_p": None}))

    @pytest.mark.anyio
    async def test_chat_input_is_refused(self):
        req = Request(input=ChatInput((ChatMessage("user", (TextPart("hi"),)),)))
        server = _Server()

        with pytest.raises(RequestAuditError, match="input.chat"):
            await _dialect(server).arun(req)

    @pytest.mark.anyio
    async def test_suffix_is_refused(self):
        req = Request(input=CompletionInput("a", suffix="b"))

        with pytest.raises(Exception, match="fim"):
            await _dialect(_Server()).arun(req)

    @pytest.mark.anyio
    async def test_empty_options_are_a_noop(self):
        _, server = await _run({"text": "", "meta_info": _meta()}, _request(options={}))

        assert server.body == {"text": "x", "sampling_params": {}}


class TestWireEvidence:
    def test_max_new_tokens_verifier_rejects_a_changed_value(self):
        verifier = _MaxNewTokensVerifier(4)

        assert verifier.verify({"sampling_params": {"max_new_tokens": 4}}) is None
        assert verifier.verify({"sampling_params": {"max_new_tokens": 5}})
        assert verifier.verify({"sampling_params": {"max_new_tokens": True}})
        assert verifier.verify({}) == "body.sampling_params is missing"

    def test_logprob_verifier_checks_every_switch(self):
        verifier = _LogprobRequestVerifier(input_scoring=True, top_logprobs=2)
        body = {
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
            "return_text_in_logprobs": True,
        }

        assert verifier.verify(body) is None
        for key, wrong in [
            ("return_logprob", 1),
            ("logprob_start_len", -1),
            ("top_logprobs_num", 3),
            ("return_text_in_logprobs", False),
        ]:
            assert verifier.verify({**body, key: wrong}), key

    def test_finish_rejects_a_tampered_body(self):
        dialect = _dialect(_Server())
        req = _request(sampling=SamplingParams(max_tokens=8))
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, _LegacyPlan())
        prepared = dialect.prepare(req, audit)
        body = prepared.thaw_body()
        body["sampling_params"] = {"max_new_tokens": 9}

        with pytest.raises(RequestAuditError, match="max_tokens"):
            audit.finish(type(prepared)(prepared.operation, body, prepared.context))


class TestLifting:
    @pytest.mark.anyio
    async def test_triples_keep_token_ids_and_normalize_text(self):
        meta = _meta(
            output_token_logprobs=[[-0.5, 42, "ĠA"], [-0.2, 9, "Ċ"]],
            output_top_logprobs=[
                [[-0.5, 42, "ĠA"], [-1.0, 43, "ĠB"]],
                [[-0.2, 9, "Ċ"]],
            ],
        )

        response, _ = await _run(
            {"text": " A\n", "meta_info": meta},
            _request(sampled_logprobs=True, top_logprobs=2),
        )

        assert response.logprobs == (
            TokenLogprob(token=" A", logprob=-0.5, token_id=42),
            TokenLogprob(token="\n", logprob=-0.2, token_id=9),
        )
        assert response.top_logprobs == (
            (
                TopKEntry(token=" A", logprob=-0.5, token_id=42),
                TopKEntry(token=" B", logprob=-1.0, token_id=43),
            ),
            (TopKEntry(token="\n", logprob=-0.2, token_id=9),),
        )
        assert response.response_model is None

    @pytest.mark.anyio
    async def test_input_and_output_scores_are_never_merged(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "a"], [-1.2, 2, "b"]],
            input_top_logprobs=[None, [[-1.2, 2, "b"], [-3.0, 7, "z"]]],
            output_token_logprobs=[[-0.3, 3, "c"]],
            output_top_logprobs=[[[-0.3, 3, "c"]]],
        )

        response, _ = await _run(
            {"text": "c", "meta_info": meta},
            _request("ab", input_scoring=True, sampled_logprobs=True, top_logprobs=2),
        )

        assert response.input_scoring is not None
        assert response.input_scoring.token_logprobs == (
            TokenLogprob(token="a", logprob=None, token_id=1),
            TokenLogprob(token="b", logprob=-1.2, token_id=2),
        )
        assert response.input_scoring.top_logprobs == (
            (),
            (
                TopKEntry(token="b", logprob=-1.2, token_id=2),
                TopKEntry(token="z", logprob=-3.0, token_id=7),
            ),
        )
        assert response.logprobs == (TokenLogprob(token="c", logprob=-0.3, token_id=3),)

    @pytest.mark.anyio
    async def test_input_scoring_alone_carries_no_input_top_k(self):
        meta = _meta(
            prompt_tokens=1,
            input_token_logprobs=[[None, 1, "Q"]],
            output_token_logprobs=[[-0.4, 5, "R"]],
        )

        response, _ = await _run(
            {"text": "R", "meta_info": meta}, _request("Q", input_scoring=True)
        )

        assert response.input_scoring is not None
        assert response.input_scoring.top_logprobs is None
        # Generated-token scores the server sent are kept (the legacy shape).
        assert response.logprobs == (TokenLogprob(token="R", logprob=-0.4, token_id=5),)

    @pytest.mark.anyio
    async def test_usage_and_finish_reason(self):
        meta = _meta(
            prompt_tokens=4, completion_tokens=6, finish_reason={"type": "length"}
        )

        response, _ = await _run({"text": "out", "meta_info": meta}, _request())

        assert response.usage == UsageStats(
            input_tokens=4, output_tokens=6, total_tokens=10, cached_tokens=0
        )
        assert response.finish_reasons == ("length",)

    @pytest.mark.anyio
    async def test_string_finish_reason_and_absent_counts(self):
        meta = {"finish_reason": "stop"}

        response, _ = await _run({"text": "", "meta_info": meta}, _request())

        assert response.finish_reasons == ("stop",)
        assert response.usage is None

    @pytest.mark.anyio
    async def test_absent_cached_tokens_reads_as_unreported(self):
        meta = _meta(prompt_tokens=4)
        del meta["cached_tokens"]

        response, _ = await _run({"text": "", "meta_info": meta}, _request())

        assert response.usage is not None
        assert response.usage.cached_tokens is None

    @pytest.mark.anyio
    async def test_negative_count_is_refused(self):
        with pytest.raises(OutputContractError, match="prompt_tokens"):
            await _run({"text": "", "meta_info": _meta(prompt_tokens=-1)}, _request())

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "key", ["prompt_tokens", "completion_tokens", "cached_tokens"]
    )
    async def test_non_int_count_is_refused(self, key):
        # The retired transport passed these through (cached_tokens became
        # None); a float count now fails the sample instead.
        with pytest.raises(OutputContractError, match=f"meta_info.{key}"):
            await _run({"text": "", "meta_info": _meta(**{key: 1.0})}, _request())


class TestNSamples:
    @pytest.mark.anyio
    async def test_list_reply_yields_per_sample_texts_and_usage(self):
        replies = [
            {"text": "a", "meta_info": _meta(prompt_tokens=4, cached_tokens=3)},
            {
                "text": "b",
                "meta_info": _meta(
                    prompt_tokens=4,
                    completion_tokens=2,
                    cached_tokens=3,
                    finish_reason={"type": "length"},
                ),
            },
        ]

        response, server = await _run(replies, _request(sampling=SamplingParams(n=2)))

        assert server.body["sampling_params"] == {"n": 2}
        assert response.texts == ("a", "b")
        assert response.finish_reasons == ("stop", "length")
        # The prompt and its cached prefix are counted once; completions sum.
        assert response.usage == UsageStats(
            input_tokens=4, output_tokens=3, total_tokens=7, cached_tokens=3
        )

    @pytest.mark.anyio
    async def test_logprobs_are_lifted_per_sample(self):
        replies = [
            {
                "text": "a",
                "meta_info": _meta(
                    output_token_logprobs=[[-0.1, 1, "Ġa"]],
                    output_top_logprobs=[[[-0.1, 1, "Ġa"]]],
                ),
            },
            {
                "text": "bc",
                "meta_info": _meta(
                    output_token_logprobs=[[-0.2, 2, "b"], [-0.3, 3, "c"]],
                    output_top_logprobs=[[[-0.2, 2, "b"]], [[-0.3, 3, "c"]]],
                ),
            },
        ]

        response, _ = await _run(
            replies,
            _request(
                sampling=SamplingParams(n=2), sampled_logprobs=True, top_logprobs=1
            ),
        )

        assert response.logprobs is None
        assert response.choice_logprobs is not None
        first, second = response.choice_logprobs
        assert first.tokens == (TokenLogprob(token=" a", logprob=-0.1, token_id=1),)
        assert [item.token for item in second.tokens] == ["b", "c"]

    @pytest.mark.anyio
    async def test_an_empty_sample_is_an_empty_sequence(self):
        replies = [
            {"text": "a", "meta_info": _meta(output_token_logprobs=[[-0.1, 1, "a"]])},
            {"text": "", "meta_info": _meta(output_token_logprobs=[])},
        ]

        response, _ = await _run(
            replies, _request(sampling=SamplingParams(n=2), sampled_logprobs=True)
        )

        assert response.choice_logprobs is not None
        assert response.choice_logprobs[1].tokens == ()
        assert [c.top_logprobs for c in response.choice_logprobs] == [None, None]

    @pytest.mark.anyio
    async def test_a_sample_missing_its_logprobs_is_refused(self):
        replies = [
            {"text": "a", "meta_info": _meta(output_token_logprobs=[[-0.1, 1, "a"]])},
            {"text": "b", "meta_info": _meta()},
        ]

        with pytest.raises(OutputContractError, match="no output_token_logprobs"):
            await _run(
                replies, _request(sampling=SamplingParams(n=2), sampled_logprobs=True)
            )

    @pytest.mark.anyio
    async def test_a_sample_short_of_its_top_k_is_refused(self):
        replies = [
            {
                "text": "ab",
                "meta_info": _meta(
                    output_token_logprobs=[[-0.1, 1, "a"], [-0.2, 2, "b"]],
                    output_top_logprobs=[[[-0.1, 1, "a"]]],
                ),
            },
            {
                "text": "c",
                "meta_info": _meta(
                    output_token_logprobs=[[-0.3, 3, "c"]],
                    output_top_logprobs=[[[-0.3, 3, "c"]]],
                ),
            },
        ]

        with pytest.raises(OutputContractError, match="top logprobs for 1 of 2"):
            await _run(
                replies,
                _request(
                    sampling=SamplingParams(n=2), sampled_logprobs=True, top_logprobs=1
                ),
            )

    @pytest.mark.anyio
    async def test_sample_count_other_than_n_is_refused(self):
        with pytest.raises(OutputContractError, match="1 samples for n=2"):
            await _run(
                {"text": "a", "meta_info": _meta()},
                _request(sampling=SamplingParams(n=2)),
            )


class TestGuards:
    @pytest.mark.anyio
    async def test_radix_cache_hit_is_refused(self):
        meta = _meta(
            prompt_tokens=3, cached_tokens=2, input_token_logprobs=[[None, 1, "a"]]
        )

        with pytest.raises(OutputContractError, match="--disable-radix-cache"):
            await _run({"text": "", "meta_info": meta}, _request(input_scoring=True))

    @pytest.mark.anyio
    async def test_truncated_input_scores_are_refused(self):
        meta = _meta(prompt_tokens=5, input_token_logprobs=[[-0.1, 1, "a"]])

        with pytest.raises(OutputContractError, match="1 of 5 prompt tokens"):
            await _run({"text": "", "meta_info": meta}, _request(input_scoring=True))

    @pytest.mark.anyio
    async def test_unverifiable_input_scores_are_refused(self):
        meta = {"finish_reason": "stop", "input_token_logprobs": [[None, 1, "a"]]}

        with pytest.raises(OutputContractError, match="omitted prompt_tokens"):
            await _run({"text": "", "meta_info": meta}, _request(input_scoring=True))

    @pytest.mark.anyio
    @pytest.mark.parametrize("raw", [None, []])
    async def test_absent_or_empty_input_scores_are_refused(self, raw):
        meta = _meta(prompt_tokens=0, input_token_logprobs=raw)

        with pytest.raises(OutputContractError, match="input"):
            await _run({"text": "", "meta_info": meta}, _request(input_scoring=True))

    @pytest.mark.anyio
    @pytest.mark.parametrize("input_top", [None, [None]])
    async def test_misaligned_input_top_k_is_refused(self, input_top):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "Q"], [-0.2, 2, "B"]],
            input_top_logprobs=input_top,
            output_token_logprobs=[[-0.1, 3, "C"]],
            output_top_logprobs=[[[-0.1, 3, "C"]]],
        )

        with pytest.raises(OutputContractError, match="input top-k logprobs"):
            await _run(
                {"text": "", "meta_info": meta},
                _request(input_scoring=True, sampled_logprobs=True, top_logprobs=2),
            )

    @pytest.mark.anyio
    async def test_output_only_scoring_skips_the_radix_guard(self):
        meta = _meta(
            prompt_tokens=5, cached_tokens=4, output_token_logprobs=[[-0.1, 1, " A"]]
        )

        response, _ = await _run(
            {"text": " A", "meta_info": meta}, _request(sampled_logprobs=True)
        )

        assert response.logprobs == (
            TokenLogprob(token=" A", logprob=-0.1, token_id=1),
        )

    @pytest.mark.anyio
    async def test_missing_output_scores_are_refused(self):
        with pytest.raises(OutputContractError, match="no output_token_logprobs"):
            await _run(
                {"text": "", "meta_info": _meta()}, _request(sampled_logprobs=True)
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            ([-0.1, 1], "triple"),
            ("abc", "triple"),
            (["x", 1, "a"], "numeric"),
            ([-0.1, True, "a"], "token id"),
            ([-0.1, 1, None], "no token text"),
        ],
    )
    async def test_malformed_triples_are_refused(self, entry, match):
        meta = _meta(output_token_logprobs=[entry])

        with pytest.raises(OutputContractError, match=match):
            await _run({"text": "", "meta_info": meta}, _request(sampled_logprobs=True))

    @pytest.mark.anyio
    async def test_a_top_k_entry_without_a_logprob_is_refused(self):
        meta = _meta(
            output_token_logprobs=[[-0.1, 1, "a"]],
            output_top_logprobs=[[[None, 1, "a"]]],
        )

        with pytest.raises(OutputContractError, match="top-k logprob"):
            await _run(
                {"text": "", "meta_info": meta},
                _request(sampled_logprobs=True, top_logprobs=1),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("reply", "match"),
        [
            ({"text": "hi"}, "no meta_info"),
            ({"meta_info": {}}, "no text"),
            (["nope"], "must be an object"),
            ([], "0 samples"),
        ],
    )
    async def test_malformed_replies_are_refused(self, reply, match):
        with pytest.raises(OutputContractError, match=match):
            await _run(reply, _request())

    @pytest.mark.anyio
    async def test_invalid_json_is_refused(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>")

        connection = AsyncHTTPJSONConnection(
            httpx.AsyncClient(
                base_url="http://h/v1", transport=httpx.MockTransport(handler)
            ),
            None,
        )

        with pytest.raises(OutputContractError, match="invalid JSON"):
            await SglangNativeDialect(connection, "m").arun(_request())

    @pytest.mark.anyio
    async def test_http_error_carries_the_server_message(self):
        server = _Server({"error": {"message": "context too long"}}, status=400)

        with pytest.raises(httpx.HTTPStatusError, match="SGLang: context too long"):
            await _dialect(server).arun(_request())


class TestPreparedRequestIntegrity:
    @pytest.mark.anyio
    async def test_execute_rejects_a_foreign_operation_or_context(self):
        dialect = _dialect(_Server())
        req = _request()
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, _LegacyPlan())
        prepared = dialect.prepare(req, audit)

        with pytest.raises(Exception, match="unexpected SGLang operation"):
            await dialect.execute(
                type(prepared)("completions.create", prepared.body, prepared.context)
            )
        with pytest.raises(Exception, match="invalid context"):
            await dialect.execute(type(prepared)("generate", prepared.body, None))

    def test_prepare_refuses_chat_input(self):
        dialect = _dialect(_Server())
        req = Request(input=ChatInput((ChatMessage("user", (TextPart("hi"),)),)))
        audit = RequestAudit(active_request_leaves(req))

        with pytest.raises(Exception, match="CompletionInput"):
            dialect.prepare(req, audit)
