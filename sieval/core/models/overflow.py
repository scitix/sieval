"""Context overflow: a prompt the model's context window cannot hold.

The refusal is deterministic for one model and one prompt, so no retry or
resume can change it. Most benchmarks score an unanswerable prompt as wrong;
counting it as a failed sample leaves a gap that looks retryable and never
closes.

* :func:`context_overflow_message` recognizes a server's refusal: an HTTP 400 or
  413, or an error with no status (a stream error), whose message says the input
  does not fit. Timeouts, connection errors and 5xx never match.
* :class:`OverflowScope` marks the model under test inside one ``infer``. The
  runner first runs ``infer`` *observing*: an overflow raises as it always did,
  so a task that handles one itself (a multi-turn walk that stops at the turn
  that no longer fits) keeps doing so. Only an overflow that escapes ``infer``,
  the case that used to fail the sample, makes the runner run it again
  *converting*: a refusal becomes :func:`overflow_output`, an empty output
  whose finish reason is :data:`CONTEXT_OVERFLOW`, which the task grades like
  any blank answer. The calls the observing run already made are answered
  from it, so nothing is asked twice.
* Outside a scope nothing changes: a model used directly, or a judge, raises
  as before.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, fields

import httpx
import openai

from sieval.core.types import JSONValue

from ._legacy_bridge import ModelMeta, ModelOutput
from .ir import Request

#: Finish reason of a rollout the model never generated because its prompt did
#: not fit, and the name of the anomaly rule and failure reason. Persisted in
#: records and ``model_calls``, so the spelling is frozen.
CONTEXT_OVERFLOW = "context_overflow"

#: Statuses a server answers an over-long prompt with. 413 is what some gateways
#: send for a body they refuse by size.
_OVERFLOW_STATUSES = frozenset({400, 413})

#: Lowercase fragments of refusals that mean "the prompt does not fit", matched
#: as substrings so a gateway that wraps the message still matches.
_OVERFLOW_MARKERS = (
    # SGLang: "The input (N tokens) is longer than the model's context length
    # (M tokens)." and "Requested token count exceeds the model's maximum
    # context length of M tokens. ..."; vLLM and OpenAI: "This model's maximum
    # context length is M tokens. ...".
    "longer than the model's context length",
    "maximum context length",
    # vLLM: "The decoder prompt (length N) is longer than the maximum model
    # length of M."
    "longer than the maximum model length",
    # OpenAI's error code, and the Responses API's wording.
    "context_length_exceeded",
    "exceeds the context window",
    # Anthropic: "prompt is too long: N tokens > M maximum".
    "prompt is too long",
    # Google: "The input token count (N) exceeds the maximum number of tokens
    # allowed (M)."
    "exceeds the maximum number of tokens allowed",
)

#: Exception names that never carry a refusal, as a failed record keeps them
#: in ``exception::<name>``: a request that did not reach a verdict.
_TRANSPORT_FAILURES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "InternalServerError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
)


class ContextOverflowError(Exception):
    """A prompt did not fit the model's context window, and was not scored.

    ``alogprobs`` raises it, since its callers need logprobs an overflow cannot
    give; so does :meth:`sieval.core.models.Model.context_overflow_output`
    outside a converting scope, as the refusal would have. A task may raise it
    from its own length check. The runner records it as ``context_overflow``,
    a failure no resume retries.
    """


def is_context_overflow_text(text: str) -> bool:
    """Whether *text* reads as a context-overflow refusal.

    Text only: the caller must already know it is a refusal rather than a
    transport failure. :func:`context_overflow_message` checks both.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


#: The status a status error writes into its message, which is all a FAILED
#: record keeps of it: openai's ``Error code: 400 - {...}`` and httpx's
#: ``Client error '400 Bad Request' for url ...`` (``Server error``, ...).
_RECORDED_STATUS = re.compile(
    r"Error code: (\d{3})\b|"
    r"(?:Informational response|Success|Redirect response|Client error|Server error)"
    r" '(\d{3}) "
)


def recorded_status(message: str) -> int | None:
    """The HTTP status a recorded failure message names first, if any."""
    match = _RECORDED_STATUS.search(message)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def is_recorded_context_overflow(reason: str | None, message: str | None) -> bool:
    """Whether a recorded ``exception::<name>`` failure was a context overflow.

    Read off a FAILED record of an earlier run, which kept the exception's name
    and message but not the status as a field. The rule is
    :func:`context_overflow_message`'s: the status the message names must be
    400 or 413, and a message naming none (a stream error) must not come from a
    transport failure, ruled out by name. A 429 or a 5xx quoting the words is
    never an overflow, and stays retriable.
    """
    if not reason or not message or not reason.startswith("exception::"):
        return False
    if reason.removeprefix("exception::") in _TRANSPORT_FAILURES:
        return False
    status = recorded_status(message)
    if status is not None and status not in _OVERFLOW_STATUSES:
        return False
    return is_context_overflow_text(message)


def _status_of(exc: BaseException) -> int | None:
    for holder in (exc, getattr(exc, "response", None)):
        status = getattr(holder, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def _texts_of(exc: BaseException) -> list[str]:
    texts = [str(exc)]
    message = getattr(exc, "message", None)
    if isinstance(message, str):
        texts.append(message)
    # An httpx status error carries the server's message only in its body.
    response = getattr(exc, "response", None)
    if isinstance(response, httpx.Response):
        with suppress(httpx.ResponseNotRead):
            texts.append(response.text)
    return texts


def context_overflow_message(exc: BaseException) -> str | None:
    """The server's message when *exc* refused a prompt as too long, else ``None``.

    Needs a status of 400 or 413, or none at all (an error inside a stream), so
    a timeout, a dropped connection or a 5xx is never read as an overflow.
    """
    if isinstance(exc, ContextOverflowError):
        return str(exc)
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            openai.APIConnectionError,
            httpx.TransportError,
        ),
    ):
        return None
    status = _status_of(exc)
    if status is not None and status not in _OVERFLOW_STATUSES:
        return None
    for text in _texts_of(exc):
        if is_context_overflow_text(text):
            return text
    return None


#: What one model call came back with: its output, or a refusal's message.
type CallResult = ModelOutput | str


class ReplayAmbiguous(Exception):
    """A replayed ``infer`` made a second call to the model under test."""


@dataclass
class OverflowScope:
    """One ``infer`` run, the model under test in it, and what its calls returned.

    *model* is the model under test. Observing (``convert=False``) its overflow
    raises as it always did and is kept in *escaped*; converting, it returns
    :func:`overflow_output`. Another model's errors, a judge's included, always
    raise.

    *known* answers calls, any model's, by position from the observing run of
    the same ``infer``: a call to the same model with a request equal to the one
    at its position gets that result without a request, so neither the model
    under test nor a judge is asked twice. The first mismatch ends the replay.

    *replay* is a refusal an earlier run recorded for this sample, without
    saying which call it answered. The first call to the model under test takes
    it; a second raises :class:`ReplayAmbiguous` and marks the scope
    *ambiguous*, and the runner runs ``infer`` again live.

    A call takes its position, and the replay, when it *asks* (:meth:`ask`),
    not when it is answered: two calls an ``infer`` makes concurrently each
    hold their own position in *calls*, in the order they were made, and the
    second of them meets the replay as ambiguous rather than both taking it.
    A position whose result is ``None`` was asked but never answered (a call
    cancelled when a sibling's refusal escaped), and is sent again.
    """

    model: object
    convert: bool = False
    known: list[tuple[object, Request, CallResult | None]] = field(default_factory=list)
    replay: str | None = None
    calls: list[tuple[object, Request, CallResult | None]] = field(default_factory=list)
    escaped: list[tuple[BaseException, str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    ambiguous: bool = False
    _subject_asked: int = 0
    _replaying: bool = True

    def ask(self, model: object, req: Request) -> tuple[int, CallResult | None]:
        """Take this call's position; with the result it already has, if any.

        Every call to a model inside the scope asks first, once, and passes the
        position to :meth:`answered`. A ``None`` result means send it.
        """
        slot = len(self.calls)
        self.calls.append((model, req, None))
        if self.replay is not None and model is self.model:
            self._subject_asked += 1
            if self._subject_asked == 1:
                return slot, self.replay
            self.ambiguous = True
            raise ReplayAmbiguous(
                "a recorded context overflow met a second call to the model under test"
            )
        if self._replaying and slot < len(self.known):
            known_model, known_req, result = self.known[slot]
            if known_model is model and known_req == req and result is not None:
                return slot, result
        self._replaying = False
        return slot, None

    def answered(self, slot: int, result: CallResult) -> None:
        model, req, _ = self.calls[slot]
        self.calls[slot] = (model, req, result)

    def escaping(self, exc: BaseException) -> str | None:
        """The refusal's message when *exc* is, or wraps, an overflow seen here."""
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            for recorded, message in self.escaped:
                if recorded is current:
                    return message
            current = current.__cause__ or current.__context__
        return None


_SCOPE: ContextVar[OverflowScope | None] = ContextVar(
    "sieval_context_overflow_scope", default=None
)


def current_overflow_scope() -> OverflowScope | None:
    """The scope the runner opened around the running ``infer``, if any."""
    return _SCOPE.get()


@contextmanager
def overflow_scope(scope: OverflowScope) -> Iterator[OverflowScope]:
    """Apply *scope* to the model calls made inside the block."""
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def overflow_output(
    model_meta: ModelMeta, req: Request, message: str, scope: OverflowScope
) -> ModelOutput:
    """The output of a call whose prompt did not fit: one empty text per rollout.

    Each rollout finishes with :data:`CONTEXT_OVERFLOW`, so a task grades it as
    the blank answer it is and the anomaly rule finds it. ``request_params``
    holds the sampling asked for, marked as never answered.
    """
    scope.messages.append(message)
    n = req.sampling.n
    params: dict[str, JSONValue] = {CONTEXT_OVERFLOW: True}
    for item in fields(req.sampling):
        value = getattr(req.sampling, item.name)
        if value is not None:
            params[item.name] = list(value) if isinstance(value, tuple) else value
    return ModelOutput(
        model=model_meta.copy(),
        texts=[""] * n,
        finish_reasons=[CONTEXT_OVERFLOW] * n,
        request_params=params,
    )
