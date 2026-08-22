# adapted from https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/generate_rankings.py
# (normalize_name, NON_TASK_COLUMNS, identify_task_columns, calculate_percentile,
#  and the per-contest branch of compute_human_metrics)
"""Ranking a model against the contest's human contestants.

Divergence from upstream:

* Upstream operates on a pandas DataFrame built from ``contestants_ranking``;
  here the rows arrive as the list of dicts that column actually holds, so
  ``df.columns`` becomes the union of row keys and the vectorized column sum
  becomes a per-row sum. The numbers are the same.
* Upstream prefers a ``Recalculated_Total`` column when one exists and, on that
  branch only, re-derives the Canadian Computing Olympiad percentile from
  ``Rank``. No published contest carries that column, so neither branch is
  ported; ``score_contest`` implements the column-matching branch that the
  released data takes.
* Codeforces Elo is not ported — it is a separate upstream fit against
  contestant CF ratings.

Human totals are re-summed over **only** the tasks the model was scored on, so a
contest whose problems were partly filtered out (interactive ones are) still
compares like with like. Medal cutoffs are the contest's published ones and are
not rescaled, which is upstream's behaviour.
"""

import re
from typing import Any, Mapping, Sequence

NON_TASK_COLUMNS = {
    "rank",
    "contestant",
    "country",
    "total",
    "recalculated_total",
    "medal",
    "cf_rating",
    "day1",
    "day2",
    "day 1",
    "day 2",
    "score rel.",
    "division",
    "team",
    "nationality",
}

_FALLBACK_TOTAL_COLUMNS = ("Total", "total", "Total Score", "score", "Score")


def normalize_name(value: str) -> str:
    """Normalize task column names for consistent matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def identify_task_columns(columns: Sequence[str], contest_id: str = "") -> dict[str, str]:
    """Return a mapping from normalized task name to original column name."""
    mapping: dict[str, str] = {}
    normalized_exclusions = {normalize_name(col) for col in NON_TASK_COLUMNS}
    contest_lower = (contest_id or "").lower()

    # BOI scores by day, so `day1` / `day2` are tasks there rather than headers.
    if "boi-" in contest_lower:
        for day_col in ("day1", "day2"):
            normalized_exclusions.discard(day_col)

    for column in columns:
        norm = normalize_name(column)
        if norm and norm not in normalized_exclusions:
            mapping[norm] = column
    return mapping


def calculate_percentile(model_score: float, human_scores: Sequence[float]) -> float | None:
    """Percent of contestants the model strictly outscores."""
    if not human_scores:
        return None
    better = sum(1 for score in human_scores if model_score > score)
    return (better / len(human_scores)) * 100


def medal_from_cutoffs(
    total: float,
    gold_cutoff: float | None,
    silver_cutoff: float | None,
    bronze_cutoff: float | None,
) -> str | None:
    """Upstream's cutoff ladder: ``None`` only when the contest publishes none."""
    if gold_cutoff is not None and total >= gold_cutoff:
        return "Gold"
    if silver_cutoff is not None and total >= silver_cutoff:
        return "Silver"
    if bronze_cutoff is not None and total >= bronze_cutoff:
        return "Bronze"
    if any(cutoff is not None for cutoff in (gold_cutoff, silver_cutoff, bronze_cutoff)):
        return "None"
    return None


def _to_numeric(value: Any) -> float:
    """``pd.to_numeric(errors="coerce").fillna(0.0)`` for a single cell."""
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if number != number else number  # NaN -> 0.0


def score_contest(
    rankings: Sequence[Mapping[str, Any]],
    model_scores: Mapping[str, float],
    contest_id: str = "",
    gold_cutoff: float | None = None,
    silver_cutoff: float | None = None,
    bronze_cutoff: float | None = None,
) -> dict:
    """Rank one model's contest total against that contest's contestants.

    *model_scores* maps a problem's ``task_name`` to the model's score on it, for
    the problems of this contest that were actually evaluated.
    """
    columns: list[str] = []
    for row in rankings:
        for key in row:
            if key not in columns:
                columns.append(key)
    task_column_map = identify_task_columns(columns, contest_id)

    matched_columns: list[str] = []
    matched_scores: list[float] = []
    for task_name, score in model_scores.items():
        column = task_column_map.get(normalize_name(str(task_name)))
        if column:
            matched_columns.append(column)
            matched_scores.append(score)

    if matched_columns:
        model_total = float(sum(matched_scores))
    else:
        # No per-task columns: fall back to the contest total, against which the
        # model's own total is every evaluated problem summed.
        fallback = next((col for col in _FALLBACK_TOTAL_COLUMNS if col in columns), None)
        if fallback is None:
            return {
                "human_percentile": None,
                "medal": None,
                "model_total": float(sum(model_scores.values())),
                "matched_columns": [],
                "n_contestants": len(rankings),
            }
        matched_columns = [fallback]
        model_total = float(sum(model_scores.values()))

    human_totals = [
        sum(_to_numeric(row.get(column)) for column in matched_columns) for row in rankings
    ]

    return {
        "human_percentile": calculate_percentile(model_total, human_totals),
        "medal": medal_from_cutoffs(model_total, gold_cutoff, silver_cutoff, bronze_cutoff),
        "model_total": model_total,
        "matched_columns": matched_columns,
        "n_contestants": len(rankings),
    }
