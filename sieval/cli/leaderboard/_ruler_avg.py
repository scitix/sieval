"""RULER headline aggregation: the mean of the per-subtask scores.

RULER is a suite of 13 subtasks (niah ×8, vt, cwe, fwe, qa ×2) run as
independent sieval tasks, each producing its own ``report.json`` with a single
``score``. RULER's headline number is the unweighted mean of those subtask
scores; when a sweep spans context lengths (task names carry a ``_<len>``
suffix, e.g. ``ruler_cwe_64k``) the mean is taken per length.

This module holds the pure aggregation; the CLI command and renderer wire it to
scanned runs. It is intentionally minimal — no thresholds, no effective-length
logic.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import re

_LEN_SUFFIX = re.compile(r"_(\d+)(k?)$", re.IGNORECASE)


# A bare numeric suffix below this is treated as a variant index (e.g. the `_1`
# in `ruler_niah_single_1`), not a context length. Real RULER lengths are always
# >= 1024 tokens; a `k` suffix is always a length regardless of magnitude.
_MIN_BARE_LENGTH = 1024


def parse_length(task_name: str) -> int | None:
    """Return the context length encoded in a task name suffix, or ``None``.

    ``ruler_cwe_64k`` → 65536, ``ruler_vt_4096`` → 4096, ``ruler_cwe`` → None.
    A small bare number is a variant index, not a length:
    ``ruler_niah_single_1`` → None.
    """
    m = _LEN_SUFFIX.search(task_name)
    if m is None:
        return None
    n = int(m.group(1))
    if m.group(2):  # explicit `k` suffix is unambiguously a length
        return n * 1024
    return n if n >= _MIN_BARE_LENGTH else None


def length_tag(length: int) -> str:
    """Render a length back to a tag: 65536 → ``64k``, 4096 → ``4k``."""
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


def ruler_average(
    runs: list[tuple[str, str, dict]],
) -> dict[str, dict]:
    """Average RULER subtask scores per model, split by context length.

    *runs* is a list of ``(model_name, task_name, report)`` triples. Only tasks
    whose name starts with ``ruler_`` and whose report carries a numeric
    ``score`` are counted; everything else is ignored.

    Returns ``{model: {"per_length": {tag: {"avg": float, "n": int}},
    "overall": {"avg": float, "n": int}}}``. ``overall`` averages every counted
    subtask across all lengths (the full-sweep headline). A length of ``None``
    (no suffix — a single-length run) is bucketed under the tag ``"all"``.
    """
    # model -> length(int|None) -> list[score]
    by_model: dict[str, dict[int | None, list[float]]] = {}

    for model_name, task_name, report in runs:
        if not task_name.startswith("ruler_"):
            continue
        score = report.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            continue
        length = parse_length(task_name)
        by_model.setdefault(model_name, {}).setdefault(length, []).append(
            float(score)
        )

    out: dict[str, dict] = {}
    for model_name, by_length in by_model.items():
        per_length: dict[str, dict] = {}
        all_scores: list[float] = []
        for length, scores in by_length.items():
            tag = "all" if length is None else length_tag(length)
            per_length[tag] = {
                "avg": round(sum(scores) / len(scores), 1),
                "n": len(scores),
            }
            all_scores.extend(scores)
        overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
        out[model_name] = {
            "per_length": per_length,
            "overall": {"avg": overall, "n": len(all_scores)},
        }
    return out
