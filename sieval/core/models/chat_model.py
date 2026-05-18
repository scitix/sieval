"""ChatModel: chat completions API backend with reasoning content support."""

from collections.abc import Iterable
from typing import override

from openai.types.chat import ChatCompletionMessageParam

from .model import Model, ModelOutput, ModelUsage


class ChatModel(Model[str | Iterable[ChatCompletionMessageParam]]):
    """Model subclass for the chat completions API (streaming + non-streaming)."""

    @override
    async def _agenerate_impl(
        self, prompt: str | Iterable[ChatCompletionMessageParam], **kwargs
    ) -> ModelOutput:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Iterable):
            messages = prompt
        else:
            raise TypeError("ChatModel requires a string or iterable of messages.")

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
        reasoning_texts = [""] * num_choices
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

        resp = await self._client.chat.completions.create(  # type: ignore[no-matching-overload]
            model=self._model,
            messages=messages,
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
                        finish_reasons[idx] = choice.finish_reason or ""
                        if choice.delta:
                            if choice.delta.content:
                                texts[idx] += choice.delta.content
                            rc = getattr(choice.delta, "reasoning", None)
                            if rc:
                                reasoning_texts[idx] += rc
                            else:
                                rc = getattr(choice.delta, "reasoning_content", None)
                                if rc:
                                    reasoning_texts[idx] += rc
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

                message = choice.message
                content = message.content
                if content is not None:
                    texts[idx] += content
                finish_reasons[idx] = choice.finish_reason or ""
                rc = getattr(message, "reasoning", None)
                if rc:
                    reasoning_texts[idx] += rc
                else:
                    rc = getattr(message, "reasoning_content", None)
                    if rc:
                        reasoning_texts[idx] += rc

            raw_usage = resp.usage
            if raw_usage is not None:
                usage = {
                    "input_tokens": raw_usage.prompt_tokens,
                    "output_tokens": raw_usage.completion_tokens,
                    "total_tokens": raw_usage.total_tokens,
                }

        return ModelOutput(
            model=self.meta(),  # Auto-attach model info
            texts=texts,
            finish_reasons=finish_reasons,
            reasoning_texts=reasoning_texts,
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
        messages = [{"role": "user", "content": prompt}]

        final_kwargs = {
            **self._kwargs,
            **kwargs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": logprobs,
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
        reasoning_texts = [""] * num_choices
        finish_reasons: list[str] = [""] * num_choices
        usage: ModelUsage | None = None
        tokens: list[str] = []
        token_logprobs: list[float | None] = []
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

        resp = await self._client.chat.completions.create(  # type: ignore[no-matching-overload]
            model=self._model,
            messages=messages,
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
                        finish_reasons[idx] = choice.finish_reason or ""

                        if choice.delta:
                            if choice.delta.content:
                                texts[idx] += choice.delta.content
                            rc = getattr(choice.delta, "reasoning", None)
                            if rc:
                                reasoning_texts[idx] += rc
                            else:
                                rc = getattr(choice.delta, "reasoning_content", None)
                                if rc:
                                    reasoning_texts[idx] += rc
                        if idx == 0:
                            logprobs_obj = getattr(choice, "logprobs", None)
                            if logprobs_obj is not None:
                                saw_logprobs = True
                                content = getattr(logprobs_obj, "content", None) or []
                                if content:
                                    tokens.extend(item.token for item in content)
                                    token_logprobs.extend(
                                        item.logprob for item in content
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

                message = choice.message
                content = message.content
                if content is not None:
                    texts[idx] += content
                finish_reasons[idx] = choice.finish_reason or ""
                rc = getattr(message, "reasoning", None)
                if rc:
                    reasoning_texts[idx] += rc
                else:
                    rc = getattr(message, "reasoning_content", None)
                    if rc:
                        reasoning_texts[idx] += rc
                if idx == 0:
                    logprobs_obj = choice.logprobs
                    if logprobs_obj is not None:
                        saw_logprobs = True
                        if logprobs_obj.content:
                            tokens.extend(item.token for item in logprobs_obj.content)
                            token_logprobs.extend(
                                item.logprob for item in logprobs_obj.content
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
                "Streaming logprobs not supported by server for chat completions."
            )

        return ModelOutput(
            model=self.meta(),
            texts=texts,
            finish_reasons=finish_reasons,
            reasoning_texts=reasoning_texts,
            logprobs_tokens=tokens,
            logprobs=token_logprobs,
            usage=usage,
            request_params=request_params,
            response_model=response_model,
            system_fingerprint=system_fingerprint,
        )
