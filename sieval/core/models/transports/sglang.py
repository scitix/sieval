"""Legacy native SGLang ``/generate`` executor retained until PR 5.

This module deliberately does not implement the canonical split ``Dialect``
contract and is not the reserved ``sglang_native`` dialect.  Only the explicit
``SglangGenModel``/``sglang_legacy`` compatibility bypass calls it in PR 1.

sglang's OpenAI ``/v1/completions`` endpoint rejects ``echo=True`` together
with ``logprobs``, so PPL-style scoring cannot go through it. This transport
speaks sglang's native ``/generate`` protocol for BOTH generation and logprob
extraction, lowering a :class:`Request` and lifting a :class:`Response`.

The token-text normalization (:func:`_normalize_token_text`), finish-reason and
usage extraction, and the radix prefix-cache guard were moved here verbatim from
the legacy ``SglangGenModel`` implementation — they have been validated against
real sglang responses. One improvement over the legacy path: the native
``[logprob, token_id, token_text]`` triples are parsed directly so
``TokenLogprob.token_id`` is populated (``SampledLogprobsWithTokenIds``).

The IR split is cleaner than the legacy ``echo`` concatenation:
``input_token_logprobs`` → ``Response.input_scoring`` and
``output_token_logprobs`` → ``Response.logprobs``, never merged. sglang's
``input_top_logprobs`` has no IR field and is not lifted (no consumer reads
prompt-side top-k).

``Request.stream`` is ignored: the native ``/generate`` path has always been a
single POST (pure scheduling, no content impact). Unrecognized
``extra_wire_params`` are mapped through the OpenAI→sglang sampling-param table
when known and otherwise dropped, matching the legacy stance of never risking
sglang rejecting an unknown param.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from typing import Any, cast

from sieval.core.models.capabilities import Capability
from sieval.core.models.ir import (
    CompletionInput,
    InputScoringResult,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue

# OpenAI-style generation kwarg -> sglang sampling_params key. Only these are
# forwarded to /generate; unrecognized kwargs (e.g. seed, stream, echo) are
# dropped rather than risk sglang rejecting an unknown sampling param.
_SAMPLING_PARAM_MAP: dict[str, str] = {
    "max_tokens": "max_new_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "stop": "stop",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
}

SGLANG_LEGACY_DIALECT_OPTION_KEYS: frozenset[str] = frozenset(
    {"prefill", "prefix", *_SAMPLING_PARAM_MAP}
)


def _request_params(body: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Return the persisted request params: the /generate body minus the prompt.

    ``body["text"]`` is the full prompt, already recorded as the sample input —
    copying it verbatim into every per-call record would duplicate it. This
    shape is sglang-native (``sampling_params`` etc.) and intentionally differs
    from the OpenAI-flavoured transports' request_params.
    """
    return {k: v for k, v in body.items() if k != "text"}


def _normalize_token_text(text: str | None) -> str:
    """Map GPT-2 byte-level BPE markers back to literal whitespace.

    sglang detokenizes when ``return_text_in_logprobs=True``, but some
    tokenizers (e.g. Qwen) surface the raw byte-level markers ``Ġ`` (space)
    and ``Ċ`` (newline). ``extract_option_logprob`` matches ``" A"`` /
    ``A`` and CMMLU keys its top-k on the token text, so an un-normalized
    ``"ĠA"`` would silently never match and the prediction would degrade.
    Normalize here so downstream scoring is fed the same token text the
    OpenAI path would produce.

    ``text`` is ``None`` when the server did not detokenize the logprobs
    (a server launched with ``--skip-tokenizer-init`` ignores
    ``return_text_in_logprobs``). Letter/option scoring cannot work without
    token text, so fail loud with an actionable message rather than crash on
    ``None.replace`` or silently degrade every token to ``""``.

    Limitation: only GPT-2 byte-level markers are handled. SentencePiece
    (``▁``, U+2581) and other tokenizer conventions pass through unchanged —
    add them here if a tokenizer that uses them needs the same contract.
    """
    if text is None:
        raise RuntimeError(
            "sglang returned a logprob entry with no token text; option/letter "
            "scoring needs detokenized text. Do not launch sglang with "
            "--skip-tokenizer-init (it ignores return_text_in_logprobs)."
        )
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def _finish_reason(meta: dict[str, Any]) -> str:
    """Extract a flat finish-reason string from sglang ``meta_info``."""
    fr = meta.get("finish_reason")
    if isinstance(fr, dict):
        return str(fr.get("type", ""))
    return str(fr) if fr else ""


class SglangTransport:
    """Legacy executor for SGLang's native ``/generate`` endpoint.

    ``token_id`` is always populated (sglang returns ``[logprob, token_id,
    token_text]`` triples).
    """

    CAPABILITIES: frozenset[Capability] = frozenset(
        {
            Capability.Completion,
            Capability.InputScoring,
            Capability.SampledLogprobs,
            Capability.SampledLogprobsWithTokenIds,
            Capability.TopKLogprobs,
            Capability.Prefill,
        }
    )

    def __init__(self, client: Any, model: str, api_base: str | None = None):
        self._client = client
        self._model = model
        self._api_base = api_base

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    # ── wire helpers ──────────────────────────────────────────────────────────

    def _generate_url(self) -> str:
        """Derive the native ``/generate`` URL from the OpenAI ``/v1`` base."""
        base = (self._api_base or "").rstrip("/").removesuffix("/v1").rstrip("/")
        return f"{base}/generate"

    async def _post(
        self, body: dict[str, JSONValue]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """POST ``body`` to ``/generate`` via the OpenAI client.

        Reuses the OpenAI SDK's low-level ``post`` to speak the native
        ``/generate`` protocol: this keeps the configured auth and
        ``max_retries``, and an absolute URL is required because the client
        would otherwise append the path to the ``/v1`` base. Returns the parsed
        JSON (a dict, or a list when ``sampling_params.n > 1``).
        """
        return cast(
            "dict[str, Any] | list[dict[str, Any]]",
            await self._client.post(self._generate_url(), cast_to=object, body=body),
        )

    # ── lower ─────────────────────────────────────────────────────────────────

    def _lower(self, req: Request) -> dict[str, JSONValue]:
        if not isinstance(req.input, CompletionInput):
            raise TypeError("SglangTransport requires CompletionInput.")

        sampling: dict[str, JSONValue] = {}
        sp = req.sampling
        n = sp.n
        if sp.max_tokens is not None:
            sampling["max_new_tokens"] = sp.max_tokens
        if sp.temperature is not None:
            sampling["temperature"] = sp.temperature
        if sp.top_p is not None:
            sampling["top_p"] = sp.top_p
        if sp.top_k is not None:
            sampling["top_k"] = sp.top_k
        if sp.stop is not None:
            sampling["stop"] = list(sp.stop)
        if sp.frequency_penalty is not None:
            sampling["frequency_penalty"] = sp.frequency_penalty
        if sp.presence_penalty is not None:
            sampling["presence_penalty"] = sp.presence_penalty
        if n > 1:
            sampling["n"] = n

        # sglang rejects max_new_tokens=0.  Input scoring also needs a token
        # when the caller left the length unset, while sampled-output scoring
        # keeps the server default unless the caller explicitly supplied zero.
        if req.scoring.input_scoring and "max_new_tokens" not in sampling:
            sampling["max_new_tokens"] = 1
        if (req.scoring.input_scoring or req.scoring.sampled_logprobs) and sampling.get(
            "max_new_tokens"
        ) == 0:
            sampling["max_new_tokens"] = 1

        # Prefill capability: sglang accepts a forced prefill on sampling_params.
        options = req.dialect_options
        if options is not None:
            prefill = options.values.get("prefill", options.values.get("prefix"))
            if prefill is not None:
                sampling["prefill"] = prefill

        # Legacy-parity passthrough: only kwargs with a known sglang
        # sampling-param equivalent are forwarded; the rest are dropped.
        if options is not None:
            for k, v in options.values.items():
                if k in {"prefill", "prefix"}:
                    continue
                dst = _SAMPLING_PARAM_MAP.get(k)
                if dst is not None and v is not None:
                    sampling.setdefault(dst, v)

        body: dict[str, JSONValue] = {
            "text": req.input.text,
            "sampling_params": sampling,
        }

        if req.scoring.sampled_logprobs or req.scoring.input_scoring:
            body["return_logprob"] = True
            # 0 → all echoed input token logprobs; -1 → output only.
            body["logprob_start_len"] = 0 if req.scoring.input_scoring else -1
            body["top_logprobs_num"] = req.scoring.top_logprobs
            body["return_text_in_logprobs"] = True

        return body

    # ── lift ────────────────────────────────────────────────────────────────

    @staticmethod
    def _triples_to_tokens(entries: list[Any]) -> tuple[TokenLogprob, ...]:
        """Map sglang ``[logprob, token_id, token_text]`` triples to TokenLogprobs.

        Unlike the legacy parser, ``token_id`` is preserved.
        """
        return tuple(
            TokenLogprob(
                token=_normalize_token_text(token_text),
                logprob=logprob,
                token_id=token_id,
            )
            for logprob, token_id, token_text in entries
        )

    @staticmethod
    def _triples_to_topk(
        entries: list[Any],
    ) -> tuple[tuple[TopKEntry, ...], ...] | None:
        """Map per-token sglang top-k triple lists to tuples of TopKEntry.

        Returns ``None`` when the server sent no top-k at all, matching the
        legacy optional shape. A ``None``/empty per-token entry becomes ``()``.
        """
        if not entries:
            return None
        result: list[tuple[TopKEntry, ...]] = []
        for per_token in entries:
            if not per_token:
                result.append(())
                continue
            result.append(
                tuple(
                    TopKEntry(
                        token=_normalize_token_text(token_text),
                        logprob=logprob,
                        token_id=token_id,
                    )
                    for logprob, token_id, token_text in per_token
                )
            )
        return tuple(result)

    @staticmethod
    def _parse_usage(metas: list[dict[str, Any]]) -> UsageStats | None:
        """Build usage from sglang ``meta_info`` token counts.

        Prompt tokens are shared across n samples; completions sum. So is
        ``cached_tokens``, which describes the shared prefix -- it is read off
        ``metas[0]`` for the same reason ``prompt_tokens`` is, and summing it
        would multiply one cache hit by n.

        sglang reports no reasoning or speculative-decoding breakdown, so those
        stay ``None``: absent, not zero.
        """
        input_tokens = metas[0].get("prompt_tokens")
        if input_tokens is None:
            return None
        output_tokens = sum(m.get("completion_tokens") or 0 for m in metas)
        cached = metas[0].get("cached_tokens")
        if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
            cached = None
        return UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached,
        )

    def _guard_radix_cache(self, meta: dict[str, Any]) -> None:
        """Reject partial echoed-input logprobs from the radix prefix cache.

        sglang's radix prefix cache does not recompute logprobs for cached
        positions: on a cache hit it truncates ``input_token_logprobs`` to
        ``prompt_tokens - cached_tokens``. Echo-based scoring reads the full
        echoed input sequence, so a truncated set would score silently wrong
        (vLLM errors in this case; sglang stays silent). Deliberate stance: ANY
        cache touch — or a response we can't verify against because it omitted
        ``prompt_tokens`` — is untrusted, so fail loud. Input scoring requires
        launching sglang with ``--disable-radix-cache``.
        """
        input_lps = meta.get("input_token_logprobs") or []
        prompt_tokens = meta.get("prompt_tokens")
        cached_tokens = meta.get("cached_tokens") or 0
        if prompt_tokens is None:
            raise RuntimeError(
                "sglang response omitted prompt_tokens, so echoed-input "
                "completeness cannot be verified; refusing to score silently. "
                "Launch sglang with --disable-radix-cache."
            )
        if cached_tokens or len(input_lps) != prompt_tokens:
            raise RuntimeError(
                "sglang returned partial echoed-input logprobs "
                f"({len(input_lps)} of {prompt_tokens} prompt tokens, "
                f"cached_tokens={cached_tokens}): its radix prefix cache does "
                "not recompute logprobs for cached positions, so echo-based "
                "scoring would be silently wrong. Launch sglang with "
                "--disable-radix-cache."
            )

    def _lift(
        self,
        results: list[dict[str, Any]],
        body: dict[str, JSONValue],
        *,
        score_input: bool,
        want_logprobs: bool,
    ) -> Response:
        metas = [r["meta_info"] for r in results]
        texts = tuple(r.get("text", "") for r in results)
        finish_reasons = tuple(_finish_reason(m) for m in metas)
        usage = self._parse_usage(metas)

        input_scoring: InputScoringResult | None = None
        logprobs: tuple[TokenLogprob, ...] | None = None
        top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None

        if want_logprobs or score_input:
            # Logprobs are read from the first sample only (alogprobs enforces
            # n=1; direct IR callers sampling n>1 with logprobs get sample 0).
            meta = metas[0]
            if score_input:
                self._guard_radix_cache(meta)
                input_scoring = InputScoringResult(
                    token_logprobs=self._triples_to_tokens(
                        meta.get("input_token_logprobs") or []
                    )
                )
            logprobs = self._triples_to_tokens(meta.get("output_token_logprobs") or [])
            top_logprobs = self._triples_to_topk(meta.get("output_top_logprobs") or [])
            if (
                not logprobs
                and not top_logprobs
                and not (input_scoring and input_scoring.token_logprobs)
            ):
                raise RuntimeError("sglang /generate returned no logprobs.")

        return Response(
            texts=texts,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            input_scoring=input_scoring,
            usage=usage,
            finish_reasons=finish_reasons,
            request_params=_request_params(body),
            response_model=None,
        )

    # ── arun ──────────────────────────────────────────────────────────────────

    async def arun(self, req: Request) -> Response:
        body = self._lower(req)
        raw = await self._post(body)
        # n>1 yields a list of per-sample dicts; n==1 a single dict.
        results = raw if isinstance(raw, list) else [raw]
        if not results or not all(
            isinstance(r, dict) and "meta_info" in r for r in results
        ):
            raise RuntimeError(
                "sglang /generate returned an unexpected response shape "
                "(missing meta_info)."
            )
        return self._lift(
            results,
            body,
            score_input=req.scoring.input_scoring,
            want_logprobs=req.scoring.sampled_logprobs,
        )
