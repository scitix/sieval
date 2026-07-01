"""SglangGenModel: native sglang ``/generate`` backend for text + logprobs.

sglang's OpenAI ``/v1/completions`` endpoint rejects ``echo=True`` together
with ``logprobs``, so PPL-style scoring (ARC/HellaSwag read the logprob of an
answer token appended to the prompt; CMMLU/MMLU-Base read the first output
token's top-k) cannot go through it. This model speaks sglang's native
``/generate`` protocol for BOTH generation and logprob extraction, so a single
object talks one wire protocol end-to-end.

It extends ``Model[str]`` rather than ``GenModel`` deliberately: the only thing
``GenModel`` would contribute is its OpenAI-completions ``_agenerate_impl``,
which is a different protocol than the ``/generate`` logprob path — incidental
reuse, not coupling. The genuinely shared infrastructure (OpenAI async client,
limiters, ``with_args``/``meta``, the public ``agenerate``/``alogprobs``
wrappers) lives in ``Model`` and is inherited directly.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import Any, override

from sieval.core.types import JSONValue

from .model import Model, ModelOutput, ModelUsage

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


def _normalize_token_text(text: str) -> str:
    """Map GPT-2 byte-level BPE markers back to literal whitespace.

    sglang detokenizes when ``return_text_in_logprobs=True``, but some
    tokenizers (e.g. Qwen) surface the raw byte-level markers ``Ġ`` (space)
    and ``Ċ`` (newline). ``extract_option_logprob`` matches ``" A"`` /
    ``A`` and CMMLU keys its top-k on the token text, so an un-normalized
    ``"ĠA"`` would silently never match and the prediction would degrade.
    Normalize here so downstream scoring is fed the same token text the
    OpenAI path would produce.
    """
    return text.replace("Ġ", " ").replace("Ċ", "\n")


class SglangGenModel(Model[str]):
    """Model backend reading text and logprobs from sglang native ``/generate``.

    AI-Generated Code - Claude Opus 4.8 (Anthropic)
    """

    def _generate_url(self) -> str:
        """Derive the native ``/generate`` URL from the OpenAI ``/v1`` base."""
        base = (self._api_base or "").rstrip("/").removesuffix("/v1").rstrip("/")
        return f"{base}/generate"

    async def _post(self, body: dict[str, JSONValue]) -> Any:
        """POST ``body`` to ``/generate`` via the OpenAI client.

        Using the public ``self._client.post`` (not the private httpx handle)
        keeps the configured auth and ``max_retries`` behaviour; an absolute
        URL is required because the client would otherwise append the path to
        the ``/v1`` base. Returns the parsed JSON (a dict, or a list when
        ``sampling_params.n > 1``).
        """
        return await self._client.post(self._generate_url(), cast_to=object, body=body)

    @staticmethod
    def _validate_n(final_kwargs: dict[str, Any]) -> int:
        """Validate and return ``n`` (mirrors GenModel's guard)."""
        n = final_kwargs.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}: {n!r}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return n

    @classmethod
    def _sampling_params(
        cls, final_kwargs: dict[str, Any], *, temperature: float | None = None
    ) -> dict[str, JSONValue]:
        """Translate recognized OpenAI-style kwargs into sglang sampling_params."""
        params: dict[str, JSONValue] = {}
        for src, dst in _SAMPLING_PARAM_MAP.items():
            if src in final_kwargs and final_kwargs[src] is not None:
                params[dst] = final_kwargs[src]
        if temperature is not None:
            params["temperature"] = temperature
        return params

    @staticmethod
    def _finish_reason(meta: dict) -> str:
        """Extract a flat finish-reason string from sglang ``meta_info``."""
        fr = meta.get("finish_reason")
        if isinstance(fr, dict):
            return str(fr.get("type", ""))
        return str(fr) if fr else ""

    @staticmethod
    def _parse_logprobs(meta: dict, echo: bool) -> tuple[list[str], list[float | None]]:
        """Flatten sglang ``*_token_logprobs`` into token-text + logprob lists.

        Each entry is ``[logprob, token_id, token_text]`` (first input
        logprob is ``None``). With ``echo`` the input segment precedes the
        output segment so echoed candidate tokens land at the sequence end.
        """
        entries: list[list] = []
        if echo:
            entries.extend(meta.get("input_token_logprobs") or [])
        entries.extend(meta.get("output_token_logprobs") or [])

        tokens: list[str] = []
        token_logprobs: list[float | None] = []
        for logprob, _token_id, token_text in entries:
            tokens.append(_normalize_token_text(token_text))
            token_logprobs.append(logprob)
        return tokens, token_logprobs

    @staticmethod
    def _parse_top_logprobs(meta: dict, echo: bool) -> list[dict[str, float]] | None:
        """Flatten sglang ``*_top_logprobs`` into ``[{token: logprob}, ...]``.

        Aligns index-for-index with the token list from ``_parse_logprobs``
        (input segment first when ``echo``). A ``None``/empty per-token entry
        (e.g. the first input token) becomes ``{}``. Returns ``None`` when the
        server sent no top-k at all, matching ``ModelOutput.top_logprobs``'s
        optional shape. CMMLU keys A/B/C/D off ``top_logprobs[0]``.
        """
        entries: list[Any] = []
        if echo:
            entries.extend(meta.get("input_top_logprobs") or [])
        entries.extend(meta.get("output_top_logprobs") or [])
        if not entries:
            return None

        result: list[dict[str, float]] = []
        for per_token in entries:
            if not per_token:
                result.append({})
                continue
            result.append(
                {
                    _normalize_token_text(token_text): logprob
                    for logprob, _token_id, token_text in per_token
                }
            )
        return result

    @staticmethod
    def _parse_usage(meta: dict) -> ModelUsage | None:
        """Build ``ModelUsage`` from sglang ``meta_info`` token counts."""
        input_tokens = meta.get("prompt_tokens")
        output_tokens = meta.get("completion_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @override
    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        if not isinstance(prompt, str):
            raise TypeError("SglangGenModel requires a string prompt.")

        final_kwargs = {**self._kwargs, **kwargs}
        num_choices = self._validate_n(final_kwargs)

        sampling = self._sampling_params(final_kwargs)
        if num_choices > 1:
            sampling["n"] = num_choices

        body: dict[str, JSONValue] = {"text": prompt, "sampling_params": sampling}
        raw = await self._post(body)

        # n>1 yields a list of per-sample dicts; n==1 a single dict.
        results = raw if isinstance(raw, list) else [raw]
        texts = [r.get("text", "") for r in results]
        metas = [r["meta_info"] for r in results]
        finish_reasons = [self._finish_reason(m) for m in metas]

        # Prompt tokens are shared across samples; completions sum.
        input_tokens = metas[0].get("prompt_tokens")
        output_tokens = sum(m.get("completion_tokens") or 0 for m in metas)
        usage: ModelUsage | None = (
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if input_tokens is not None
            else None
        )

        return ModelOutput(
            model=self.meta(),
            texts=texts,
            finish_reasons=finish_reasons,
            usage=usage,
            request_params=body,
            response_model=self._model,
        )

    @override
    async def _alogprobs_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        final_kwargs = {**self._kwargs, **kwargs}
        num_choices = self._validate_n(final_kwargs)
        if num_choices > 1:
            raise ValueError(f"alogprobs only supports n=1; received n={num_choices}")

        sampling = self._sampling_params(final_kwargs, temperature=temperature)
        # sglang rejects max_new_tokens=0; the generated token is ignored for
        # scoring but at least one is required.
        sampling["max_new_tokens"] = max(max_tokens, 1)

        body: dict[str, JSONValue] = {
            "text": prompt,
            "sampling_params": sampling,
            "return_logprob": True,
            # 0 → all echoed input token logprobs; -1 → output only.
            "logprob_start_len": 0 if echo else -1,
            "top_logprobs_num": logprobs,
            "return_text_in_logprobs": True,
        }

        data = await self._post(body)
        meta = data["meta_info"]

        tokens, token_logprobs = self._parse_logprobs(meta, echo)
        top_logprobs = self._parse_top_logprobs(meta, echo)
        if not token_logprobs and not top_logprobs:
            raise RuntimeError("sglang /generate returned no logprobs.")

        return ModelOutput(
            model=self.meta(),
            texts=[data.get("text", "")],
            finish_reasons=[self._finish_reason(meta)],
            logprobs_tokens=tokens,
            logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            usage=self._parse_usage(meta),
            request_params=body,
            response_model=self._model,
        )
