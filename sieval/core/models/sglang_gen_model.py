"""SglangGenModel: echoed-input logprobs via sglang native /generate.

sglang's OpenAI ``/v1/completions`` endpoint rejects ``echo=True`` together
with ``logprobs``, so PPL-style tasks (ARC, MMLU-Base, HellaSwag) that read
the logprob of an answer token appended to the prompt cannot use it. This
variant overrides only logprob extraction to call the native ``/generate``
endpoint (``return_logprob=True`` + ``logprob_start_len=0``), which returns
per-token logprobs over the full echoed input sequence.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import override

from sieval.core.types import JSONValue

from .gen_model import GenModel
from .model import ModelOutput, ModelUsage


def _normalize_token_text(text: str) -> str:
    """Map GPT-2 byte-level BPE markers back to literal whitespace.

    sglang detokenizes when ``return_text_in_logprobs=True``, but some
    tokenizers (e.g. Qwen) surface the raw byte-level markers ``Ġ`` (space)
    and ``Ċ`` (newline). ``extract_option_logprob`` matches ``" A"`` /
    ``A``, so an un-normalized ``"ĠA"`` would silently never match and the
    prediction would degrade. Normalize here so downstream scoring is fed
    the same token text the OpenAI path would produce.
    """
    return text.replace("Ġ", " ").replace("Ċ", "\n")


class SglangGenModel(GenModel):
    """GenModel variant reading echoed-input logprobs via sglang /generate.

    Only ``_alogprobs_impl`` is overridden; ``_agenerate_impl`` keeps using
    the OpenAI ``/v1/completions`` endpoint (plain generation works there).

    AI-Generated Code - Claude Opus 4.8 (Anthropic)
    """

    def _generate_url(self) -> str:
        """Derive the native ``/generate`` URL from the OpenAI ``/v1`` base."""
        base = (self._api_base or "").rstrip("/").removesuffix("/v1").rstrip("/")
        return f"{base}/generate"

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
        num_choices_raw = {**self._kwargs, **kwargs}.get("n", 1)
        if isinstance(num_choices_raw, bool) or not isinstance(num_choices_raw, int):
            raise TypeError(
                "n must be an int, got "
                f"{type(num_choices_raw).__name__}: {num_choices_raw!r}"
            )
        if num_choices_raw != 1:
            raise ValueError(
                f"alogprobs only supports n=1; received n={num_choices_raw}"
            )

        body: dict[str, JSONValue] = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                # sglang rejects max_new_tokens=0; the generated tokens are
                # ignored for scoring.
                "max_new_tokens": max(max_tokens, 1),
            },
            "return_logprob": True,
            # 0 → all echoed input token logprobs; -1 → output only.
            "logprob_start_len": 0 if echo else -1,
            "top_logprobs_num": logprobs,
            "return_text_in_logprobs": True,
        }

        resp = await self._client._client.post(self._generate_url(), json=body)
        resp.raise_for_status()
        data = resp.json()
        meta = data["meta_info"]

        tokens, token_logprobs = self._parse_logprobs(meta, echo)
        if not token_logprobs:
            raise RuntimeError("sglang /generate returned no logprobs.")

        return ModelOutput(
            model=self.meta(),
            texts=[data.get("text", "")],
            logprobs_tokens=tokens,
            logprobs=token_logprobs,
            usage=self._parse_usage(meta),
            request_params=body,
            response_model=self._model,
        )
