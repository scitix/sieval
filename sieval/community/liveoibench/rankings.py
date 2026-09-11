# adapted from https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/generate_rankings.py
# (normalize_name, normalize_contest_identifier, NON_TASK_COLUMNS,
#  identify_task_columns, calculate_percentile, build_problem_to_contest_map,
#  the contest-routing block of group_problems_by_contest, and compute_human_metrics)
"""Ranking a model against the contest's human contestants.

Divergence from upstream:

* Upstream operates on a pandas DataFrame built from ``contestants_ranking``;
  here the rows arrive as the list of dicts that column actually holds, so
  ``df.columns`` becomes the union of row keys and the vectorized column sum
  becomes a per-row sum. The numbers are the same.
* Codeforces Elo is not ported — it is a separate upstream fit against
  contestant CF ratings.
* Upstream's USACO ``-combined`` contests are scored from promotion thresholds
  in ``data/USACO/{year}/{contest}/contest_info.json``, a file that lives in the
  upstream repository rather than in the published dataset. Without it upstream
  itself returns no medal and no percentile for those contests, which is what
  :func:`score_contest` returns; see the ``-combined`` branch there.

Contest identity is **not** derivable from the problem id alone. USACO keys its
contestant rows by division (``USACO-2023-December_Contest-platinum`` /
``-combined``), so :func:`build_problem_to_contest_map` reads the authoritative
mapping out of the contestant table's own ``problems`` column and
:func:`resolve_contest_id` falls back to upstream's division ladder. On the
published data every problem is listed by exactly one contest row.

Human totals are re-summed over **only** the tasks the model was scored on, so a
contest whose problems were partly filtered out (interactive ones are) still
compares like with like. Medal cutoffs are the contest's published ones and are
not rescaled, which is upstream's behaviour.
"""

import json
import re
from typing import Any, Iterable, Mapping, Sequence

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
_RECALC_TOTAL_COLUMNS = ("Recalculated_Total", "recalculated_total")
_RANK_COLUMNS = ("Rank", "rank")
_PROMOTED_DIVISIONS = {"bronze", "silver", "gold"}

# Upstream renames the three Canadian rounds when matching the human leaderboard.
_CCO_ROUND_ALIASES = {
    "Canadian_Computing_Competition_Junior": "Junior",
    "Canadian_Computing_Competition_Senior": "Senior",
    "Canadian_Computing_Olympiad": "contest",
}


def normalize_name(value: str) -> str:
    """Normalize task column names for consistent matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_contest_identifier(contest_base: str) -> str:
    """Upstream ``normalize_contest_identifier``: rename the CCO rounds.

    The published contestant table happens to key CCO by its *un*-renamed round,
    which is why :func:`resolve_contest_id` keeps upstream's fallback to the raw
    identifier. Ported so the ladder is the same one upstream walks.
    """
    if not contest_base.startswith("CCO-"):
        return contest_base
    parts = contest_base.split("-", 2)
    if len(parts) != 3:
        return contest_base
    prefix, year, rest = parts
    for source, replacement in _CCO_ROUND_ALIASES.items():
        if rest.startswith(source):
            rest = rest.replace(source, replacement, 1)
            break
    return f"{prefix}-{year}-{rest}"


def usaco_division(problem_id: str) -> str | None:
    """The division token a USACO problem id carries, or ``None`` elsewhere.

    ``USACO-2023-US_Open_Contest-bronze_FEB`` -> ``"bronze"``. Upstream reads it
    off the same fourth ``-`` segment, which is also the ``task_name`` column.
    """
    if not problem_id.startswith("USACO-"):
        return None
    parts = problem_id.split("-", 3)
    if len(parts) < 4:
        return None
    return parts[3].split("_", 1)[0].lower()


def build_problem_to_contest_map(
    contest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    """Upstream ``build_problem_to_contest_map``: problem id -> contest id.

    Driven by each contestant row's ``problems`` column, which is the only place
    the USACO division split is stated. The division filter is upstream's and is
    a no-op on consistent data — a ``-combined`` row lists only bronze/silver/gold
    problems and a ``-platinum`` row only platinum ones.
    """
    mapping: dict[str, str] = {}
    for row in contest_rows:
        contest_id = row.get("contest_id")
        problems = row.get("problems")
        if not contest_id or problems is None:
            continue
        if isinstance(problems, str):
            try:
                problems = json.loads(problems)
            except json.JSONDecodeError:
                continue
        if not isinstance(problems, (list, tuple)):
            continue
        for problem in problems:
            problem_id = str(problem)
            division = usaco_division(problem_id)
            if division is None:
                mapping[problem_id] = contest_id
            elif contest_id.endswith("-combined"):
                if division in _PROMOTED_DIVISIONS:
                    mapping[problem_id] = contest_id
            elif contest_id.endswith("-platinum"):
                if division == "platinum":
                    mapping[problem_id] = contest_id
            else:
                mapping[problem_id] = contest_id
    return mapping


def resolve_contest_id(
    problem_id: str,
    contest_base: str,
    problem_to_contest: Mapping[str, str],
    available_contests: Sequence[str] | set[str] = (),
) -> str | None:
    """Which contest row one problem is ranked against; upstream's ladder.

    *contest_base* is ``{competition}-{year}-{round}``, the identifier the
    problem id alone yields. ``None`` means the problem has no human ranking —
    upstream's ``continue``, which drops a USACO problem whose division token is
    not one of the four.
    """
    available = set(available_contests)
    contest_id = problem_to_contest.get(problem_id)
    base_norm = normalize_contest_identifier(contest_base)

    if not contest_id:
        division = usaco_division(problem_id)
        if division is not None:
            if division == "platinum":
                contest_id = f"{base_norm}-platinum"
            elif division in _PROMOTED_DIVISIONS:
                contest_id = f"{base_norm}-combined"
            else:
                return None
        else:
            contest_id = base_norm

    if contest_id not in available and contest_id != base_norm and base_norm in available:
        contest_id = base_norm
    if contest_id not in available and contest_base in available:
        contest_id = contest_base
    return contest_id


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


def _numeric_or_none(value: Any) -> float | None:
    """``pd.to_numeric(errors="coerce")`` for a single cell, keeping NaN as None.

    The recalculated-total branch *drops* unparseable rows rather than zeroing
    them (upstream ``.dropna()``), so it needs the distinction ``_to_numeric``
    throws away.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _result(
    human_percentile: float | None,
    medal: str | None,
    model_total: float,
    matched_columns: list[str],
    n_contestants: int,
) -> dict:
    return {
        "human_percentile": human_percentile,
        "medal": medal,
        "model_total": model_total,
        "matched_columns": matched_columns,
        "n_contestants": n_contestants,
    }


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
    model_sum = float(sum(model_scores.values()))

    # Upstream short-circuits a USACO `-combined` contest into
    # `compute_usaco_combined_metrics`, which compares each division's total
    # against the promotion thresholds in that contest's `contest_info.json`.
    # That file is part of the upstream repository, not of the published
    # dataset, so upstream on this data reaches `thresholds_present = False` and
    # returns no medal and no percentile. Returning anything here -- the row's
    # own medal cutoffs, say -- would be a number upstream does not compute.
    if contest_id.startswith("USACO-") and contest_id.endswith("-combined"):
        return _result(None, None, model_sum, [], len(rankings))

    # No contestant rows: upstream's `if df.empty` branch still awards a medal
    # from the published cutoffs, and only the percentile is unavailable. Every
    # USACO contest in the release takes this path.
    if not rankings:
        medal = medal_from_cutoffs(model_sum, gold_cutoff, silver_cutoff, bronze_cutoff)
        return _result(None, medal, model_sum, [], 0)

    columns: list[str] = []
    for row in rankings:
        for key in row:
            if key not in columns:
                columns.append(key)

    # Upstream prefers a `Recalculated_Total` column wherever one exists: the
    # contest's own total re-derived over the tasks LiveOIBench scores, which is
    # not the same number as the sum of the per-task columns (it differs for
    # 20-83% of contestants on the 10 contests that carry it). The model side
    # stays the sum of every problem evaluated in the contest.
    recalc_column = next((col for col in _RECALC_TOTAL_COLUMNS if col in columns), None)
    if recalc_column is not None:
        human_totals = [
            value
            for value in (_numeric_or_none(row.get(recalc_column)) for row in rankings)
            if value is not None
        ]
        human_percentile = calculate_percentile(model_sum, human_totals)
        # On this branch only, upstream re-derives the Canadian Computing
        # Olympiad percentile from the rank column. No published CCO contest
        # carries `Recalculated_Total`, so this is unreachable on the released
        # data; ported because dropping it would make the port wrong for a
        # release that does.
        if "canadian_computing_olympiad" in contest_id.lower() and any(
            col in columns for col in _RANK_COLUMNS
        ):
            rank_column = next(col for col in _RANK_COLUMNS if col in columns)
            n_ranked = sum(
                1 for row in rankings if _numeric_or_none(row.get(rank_column)) is not None
            )
            if n_ranked:
                model_rank = sum(1 for total in human_totals if total > model_sum) + 1
                model_rank = max(1, min(model_rank, n_ranked))
                human_percentile = ((n_ranked - model_rank) / n_ranked) * 100
        medal = medal_from_cutoffs(model_sum, gold_cutoff, silver_cutoff, bronze_cutoff)
        return _result(human_percentile, medal, model_sum, [recalc_column], len(rankings))

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
            return _result(None, None, model_sum, [], len(rankings))
        matched_columns = [fallback]
        model_total = model_sum

    human_totals = [
        sum(_to_numeric(row.get(column)) for column in matched_columns) for row in rankings
    ]

    return _result(
        calculate_percentile(model_total, human_totals),
        medal_from_cutoffs(model_total, gold_cutoff, silver_cutoff, bronze_cutoff),
        model_total,
        matched_columns,
        len(rankings),
    )
