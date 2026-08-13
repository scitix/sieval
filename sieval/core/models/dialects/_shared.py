"""Response-shape checks shared by dialects that speak the same wire dialect.

Only genuinely dialect-independent checks belong here. A check whose behaviour
differs per dialect — token accounting, for instance — stays in that dialect
under a name that says which one it is, so the difference is visible at the
definition rather than hidden behind a shared spelling.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.core.models.dialect import OutputContractError
from sieval.core.models.ir import TopKEntry


def validate_top_logprobs(value: object) -> None:
    """Reject a ``top_logprobs`` channel that is not a tuple of entry tuples."""

    if not isinstance(value, tuple) or not all(
        isinstance(position, tuple)
        and all(isinstance(item, TopKEntry) for item in position)
        for position in value
    ):
        raise OutputContractError("top_logprobs channel has invalid shape")


def resolve_choice_index(choice: object, n: int, kind: str) -> int:
    """Return a choice's index, rejecting a non-integer or out-of-range one.

    ``kind`` names the dialect in the error only; the check itself is identical
    across dialects, which is why this is shared rather than duplicated.
    """

    index = getattr(choice, "index", None)
    if isinstance(index, bool) or not isinstance(index, int):
        raise OutputContractError(f"{kind} choice index must be an integer")
    if not 0 <= index < n:
        raise OutputContractError(
            f"{kind} choice index {index} is outside the requested range [0, {n})"
        )
    return index
