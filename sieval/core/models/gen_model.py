"""GenModel: text completions API backend."""

from collections.abc import Mapping, Sequence
from typing import override

from .model import Model, ModelOutput, ModelUsage


def _completion_top_logprobs(raw: object) -> list[dict[str, float]]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return []

    top_logprobs = []
    for item in raw:
        if item is None:
            top_logprobs.append({})
            continue
        if not isinstance(item, Mapping):
            continue
        top_logprobs.append(
            {
                token: float(logprob)
                for token, logprob in item.items()
                if isinstance(token, str) and isinstance(logprob, int | float)
            }
        )
    return top_logprobs


class GenModel(Model[str]):
    """Model subclass for the completions API (streaming + non-streaming)."""

    @override
    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        if not isinstance(prompt, str):
            raise TypeError("GenModel requires a string prompt.")

        final_kwargs = {**self._kwargs, **kwargs}

        num_choices_raw = final_kwargs.get("n", 1)
        if isinstance(num_choices_raw, bool) or not isinstance(num_choices_raw, int):
            raise TypeError(
                "n must be an int, got "
                f"{type(num_choices_raw).__name__}: {num_choices_raw!r}"
            )
        if num_choices_raw < 1:
            raise ValueError(f"n must be >= 1, got {num_choices_raw}")
        num_choices = num_choices_raw
        texts = [""] * num_choices
        finish_reasons = [""] * num_choices
        usage: ModelUsage | None = None

        # Snapshot params for stable meta serialization
        request_params = dict(final_kwargs)
        if "stream" not in request_params:
            request_params["stream"] = True
        stream_mode = request_params["stream"]
        if not isinstance(stream_mode, bool):
            raise TypeError(
                "stream must be a bool, got "
                f"{type(stream_mode).__name__}: {stream_mode!r}"
            )
        if stream_mode and "stream_options" not in request_params:
            request_params["stream_options"] = {"include_usage": True}

        resp = await self._client.completions.create(
            model=self._model,
            prompt=prompt,
            **request_params,
        )
        response_model: str | None = None
        system_fingerprint: str | None = None

        if stream_mode:
            async for chunk in resp:
                if response_model is None:
                    response_model = getattr(chunk, "model", None)
                if system_fingerprint is None:
                    system_fingerprint = getattr(chunk, "system_fingerprint", None)
                if chunk.choices:
                    for choice in chunk.choices:
                        idx = choice.index
                        if not 0 <= idx < num_choices:
                            continue
                        texts[idx] += choice.text or ""
                        finish_reasons[idx] = choice.finish_reason or ""
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = {
                        "input_tokens": chunk_usage.prompt_tokens,
                        "output_tokens": chunk_usage.completion_tokens,
                        "total_tokens": chunk_usage.total_tokens,
                    }
        else:
            response_model = getattr(resp, "model", None)
            system_fingerprint = getattr(resp, "system_fingerprint", None)
            for choice in resp.choices:
                idx = choice.index
                if not 0 <= idx < num_choices:
                    continue
                texts[idx] += choice.text or ""
                finish_reasons[idx] = choice.finish_reason or ""
            raw_usage = resp.usage
            if raw_usage is not None:
                usage = {
                    "input_tokens": raw_usage.prompt_tokens,
                    "output_tokens": raw_usage.completion_tokens,
                    "total_tokens": raw_usage.total_tokens,
                }

        # No reasoning_texts for GenModel now
        return ModelOutput(
            model=self.meta(),  # Auto-attach model info
            texts=texts,
            finish_reasons=finish_reasons,
            usage=usage,
            request_params=request_params,
            response_model=response_model,
            system_fingerprint=system_fingerprint,
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
        final_kwargs = {
            **self._kwargs,
            **kwargs,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "echo": echo,
            "temperature": temperature,
        }

        num_choices_raw = final_kwargs.get("n", 1)
        if isinstance(num_choices_raw, bool) or not isinstance(num_choices_raw, int):
            raise TypeError(
                "n must be an int, got "
                f"{type(num_choices_raw).__name__}: {num_choices_raw!r}"
            )
        if num_choices_raw < 1:
            raise ValueError(f"n must be >= 1, got {num_choices_raw}")
        num_choices = num_choices_raw
        if num_choices > 1:
            raise ValueError(f"alogprobs only supports n=1; received n={num_choices}")
        texts = [""] * num_choices
        finish_reasons = [""] * num_choices
        usage: ModelUsage | None = None
        tokens: list[str] = []
        token_logprobs: list[float | None] = []
        top_token_logprobs: list[dict[str, float]] = []
        saw_logprobs = False

        # Snapshot params for stable meta serialization
        request_params = dict(final_kwargs)
        if "stream" not in request_params:
            request_params["stream"] = True
        stream_mode = request_params["stream"]
        if not isinstance(stream_mode, bool):
            raise TypeError(
                "stream must be a bool, got "
                f"{type(stream_mode).__name__}: {stream_mode!r}"
            )
        if stream_mode and "stream_options" not in request_params:
            request_params["stream_options"] = {"include_usage": True}

        resp = await self._client.completions.create(  # ty: ignore[no-matching-overload]
            model=self._model,
            prompt=prompt,
            **request_params,
        )
        response_model: str | None = None
        system_fingerprint: str | None = None

        if stream_mode:
            async for chunk in resp:
                if response_model is None:
                    response_model = getattr(chunk, "model", None)
                if system_fingerprint is None:
                    system_fingerprint = getattr(chunk, "system_fingerprint", None)
                if chunk.choices:
                    for choice in chunk.choices:
                        idx = choice.index
                        if not 0 <= idx < num_choices:
                            continue

                        texts[idx] += choice.text or ""
                        finish_reasons[idx] = choice.finish_reason or ""
                        if idx == 0:
                            logprobs_obj = getattr(choice, "logprobs", None)
                            if logprobs_obj is not None:
                                saw_logprobs = True
                                if logprobs_obj.tokens:
                                    tokens.extend(logprobs_obj.tokens)
                                if logprobs_obj.token_logprobs:
                                    token_logprobs.extend(logprobs_obj.token_logprobs)
                                top_token_logprobs.extend(
                                    _completion_top_logprobs(
                                        getattr(logprobs_obj, "top_logprobs", None)
                                    )
                                )

                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = {
                        "input_tokens": chunk_usage.prompt_tokens,
                        "output_tokens": chunk_usage.completion_tokens,
                        "total_tokens": chunk_usage.total_tokens,
                    }
        else:
            response_model = getattr(resp, "model", None)
            system_fingerprint = getattr(resp, "system_fingerprint", None)
            for choice in resp.choices:
                idx = choice.index
                if not 0 <= idx < num_choices:
                    continue

                texts[idx] += choice.text or ""
                finish_reasons[idx] = choice.finish_reason or ""
                if idx == 0:
                    logprobs_obj = choice.logprobs
                    if logprobs_obj is not None:
                        saw_logprobs = True
                        if logprobs_obj.tokens:
                            tokens.extend(logprobs_obj.tokens)
                        if logprobs_obj.token_logprobs:
                            token_logprobs.extend(logprobs_obj.token_logprobs)
                        top_token_logprobs.extend(
                            _completion_top_logprobs(
                                getattr(logprobs_obj, "top_logprobs", None)
                            )
                        )

            raw_usage = resp.usage
            if raw_usage is not None:
                usage = {
                    "input_tokens": raw_usage.prompt_tokens,
                    "output_tokens": raw_usage.completion_tokens,
                    "total_tokens": raw_usage.total_tokens,
                }

        if not saw_logprobs:
            raise RuntimeError(
                "Streaming logprobs not supported by server for completions."
            )

        # No reasoning_texts for GenModel now
        return ModelOutput(
            model=self.meta(),  # Auto-attach model info
            texts=texts,
            finish_reasons=finish_reasons,
            logprobs_tokens=tokens,
            logprobs=token_logprobs,
            top_logprobs=top_token_logprobs or None,
            usage=usage,
            request_params=request_params,
            response_model=response_model,
            system_fingerprint=system_fingerprint,
        )
