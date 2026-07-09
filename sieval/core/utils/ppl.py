"""Perplexity and log-probability extraction utilities."""


def extract_option_logprob(
    tokens: list[str],
    token_logprobs: list[float | None],
    option_label: str,
    *,
    max_search: int = 15,
) -> float | None:
    """Return the logprob for a multiple-choice option label (A/B/C/D...)."""
    max_len = min(len(tokens), len(token_logprobs))
    if max_len == 0:
        return None

    start_idx = max_len - 1
    end_idx = max(0, max_len - max_search)
    for i in range(start_idx, end_idx - 1, -1):
        token = tokens[i]
        logprob = token_logprobs[i]
        if logprob is None:
            continue
        token_stripped = token.strip()
        if token_stripped == option_label:
            return logprob
        if token == f" {option_label}":
            return logprob
        if option_label in token_stripped and len(token_stripped) <= 2:
            return logprob
    return None


def choice_scores_from_top_logprobs(
    top_logprobs: list[dict[str, float]] | None,
    choices: tuple[str, ...],
) -> tuple[dict[str, float], bool]:
    """Map the first generated token's top-k onto per-choice logprobs.

    Reads ``top_logprobs[0]`` (the next-token distribution) and records, for
    each label in *choices*, the logprob of the matching token — matched by the
    stripped token string, so ``" A"`` maps to ``"A"``. Returns
    ``(scores, all_present)``; *all_present* is ``True`` only when every choice
    label appears in the top-k, so a caller can fail loudly instead of
    argmax-ing a subset. Missing labels keep a ``-inf`` score.
    """
    scores = {label: float("-inf") for label in choices}
    if not top_logprobs:
        return scores, False

    seen: set[str] = set()
    for token, logprob in top_logprobs[0].items():
        label = token.strip()
        if label in scores:
            scores[label] = max(scores[label], logprob)
            seen.add(label)
    return scores, len(seen) == len(choices)


def total_logprob(
    tokens: list[str],
    token_logprobs: list[float | None],
    *,
    skip_first: bool = True,
) -> tuple[float, int]:
    """Sum all log-probabilities, skipping ``None`` entries.

    *skip_first* drops the BOS/prompt-boundary token whose logprob is
    uninformative.  Returns ``(total_logprob, count)``.
    """
    max_len = min(len(tokens), len(token_logprobs))
    if max_len == 0:
        return 0.0, 0
    start = 1 if skip_first else 0
    total = 0.0
    count = 0
    for i in range(start, max_len):
        logprob = token_logprobs[i]
        if logprob is None:
            continue
        total += logprob
        count += 1
    return total, count
