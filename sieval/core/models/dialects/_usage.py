"""Usage lifting shared by the OpenAI-shaped dialects.

Chat and completions receive the same ``CompletionUsage`` wire object, so what
counts as a readable token breakdown has one owner here rather than two copies
that drift the next time OpenAI adds a detail field.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import Any

from sieval.core.models.ir import UsageStats

_COMPLETION_DETAILS = "completion_tokens_details"
_PROMPT_DETAILS = "prompt_tokens_details"


def _detail(raw: Any, container: str, name: str) -> int | None:
    """Read ``raw.<container>.<name>`` as a non-negative int, else ``None``.

    Deliberately never raises. These counts are observational, the detail
    objects are absent on most OpenAI-compatible servers, and the reply they
    describe has already been billed -- so a malformed value reads as "not
    reported" rather than costing the caller a completed response.
    """
    details = getattr(raw, container, None)
    if details is None:
        return None
    value = getattr(details, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _reported_total(raw: Any, computed: int) -> int | None:
    """The server's own total, kept only where it disagrees with *computed*.

    Agreement is the overwhelmingly common case and carries no information, so
    storing it on every record would bury the disagreements. Present means the
    server's accounting differs from ours -- usually explained by the reasoning
    count beside it.
    """
    value = getattr(raw, "total_tokens", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return None if value == computed else value


def usage_stats(raw: Any, input_tokens: int, output_tokens: int) -> UsageStats:
    """Assemble usage from the two validated counts and the optional details.

    ``total_tokens`` is computed, not read: a reported total that does not
    decompose is recorded as ``reported_total_tokens`` rather than rejected,
    because the tokens it describes were already generated and billed.
    """
    total = input_tokens + output_tokens
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        reasoning_tokens=_detail(raw, _COMPLETION_DETAILS, "reasoning_tokens"),
        cached_tokens=_detail(raw, _PROMPT_DETAILS, "cached_tokens"),
        accepted_prediction_tokens=_detail(
            raw, _COMPLETION_DETAILS, "accepted_prediction_tokens"
        ),
        rejected_prediction_tokens=_detail(
            raw, _COMPLETION_DETAILS, "rejected_prediction_tokens"
        ),
        reported_total_tokens=_reported_total(raw, total),
    )
