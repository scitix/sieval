"""Unit tests for ranking a model against a contest's human contestants.

The load-bearing behaviour is that humans are re-totaled over **only** the tasks
the model was scored on: sieval filters interactive problems out, so a contest is
often partial, and comparing a partial model total against full human totals
would understate every model by however much it skipped.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import json

from sieval.community.liveoibench.rankings import (
    build_problem_to_contest_map,
    calculate_percentile,
    identify_task_columns,
    medal_from_cutoffs,
    normalize_contest_identifier,
    normalize_name,
    resolve_contest_id,
    score_contest,
)

# Shape of the published `contestants_ranking` payload.
RANKINGS = [
    {
        "Rank": 1,
        "Contestant": "A",
        "Country": "CN",
        "alpha": 100,
        "beta": 100,
        "Total": 200,
    },
    {
        "Rank": 2,
        "Contestant": "B",
        "Country": "US",
        "alpha": 100,
        "beta": 20,
        "Total": 120,
    },
    {
        "Rank": 3,
        "Contestant": "C",
        "Country": "PL",
        "alpha": 40,
        "beta": 0,
        "Total": 40,
    },
    {"Rank": 4, "Contestant": "D", "Country": "JP", "alpha": 0, "beta": 0, "Total": 0},
]


def test_percentile_counts_only_contestants_strictly_below():
    # Upstream is `model_score > human_scores`, so tying the field beats nobody.
    assert calculate_percentile(40, [0, 40, 120, 200]) == 25.0
    assert calculate_percentile(0, [0, 40, 120, 200]) == 0.0
    assert calculate_percentile(201, [0, 40, 120, 200]) == 100.0


def test_percentile_of_an_empty_field_is_unknown_not_zero():
    assert calculate_percentile(100, []) is None


def test_medal_ladder_is_inclusive_at_each_cutoff():
    assert medal_from_cutoffs(236, 236, 191, 145) == "Gold"
    assert medal_from_cutoffs(235, 236, 191, 145) == "Silver"
    assert medal_from_cutoffs(145, 236, 191, 145) == "Bronze"
    assert medal_from_cutoffs(144, 236, 191, 145) == "None"


def test_a_contest_publishing_no_cutoffs_yields_no_medal_rather_than_none_the_string():
    assert medal_from_cutoffs(500, None, None, None) is None


def test_header_columns_are_not_mistaken_for_tasks():
    mapping = identify_task_columns(list(RANKINGS[0]))
    assert set(mapping) == {"alpha", "beta"}


def test_boi_scores_by_day_so_day_columns_are_tasks_there():
    columns = ["Rank", "Contestant", "Day1", "Day2"]
    assert identify_task_columns(columns, "BOI-2025-contest") == {
        "day1": "Day1",
        "day2": "Day2",
    }
    assert identify_task_columns(columns, "IOI-2025-contest") == {}


def test_normalize_name_strips_everything_but_alphanumerics():
    assert normalize_name("Cow-libi (Silver)") == "cowlibisilver"


def test_humans_are_retotaled_over_only_the_tasks_the_model_attempted():
    # Model scored 100 on `alpha` alone. Against alpha-only human totals
    # (100, 100, 40, 0) it beats two; against full totals it would beat one.
    result = score_contest(RANKINGS, {"alpha": 100})
    assert result["model_total"] == 100
    assert result["matched_columns"] == ["alpha"]
    assert result["human_percentile"] == 50.0


def test_scoring_every_task_uses_every_column():
    result = score_contest(RANKINGS, {"alpha": 100, "beta": 100})
    assert result["model_total"] == 200
    assert sorted(result["matched_columns"]) == ["alpha", "beta"]
    # Ties with the top contestant, so it beats the other three.
    assert result["human_percentile"] == 75.0


def test_task_names_match_columns_case_and_punctuation_insensitively():
    rankings = [{"Rank": 1, "Cow-libi": 100, "Total": 100}]
    result = score_contest(rankings, {"cow_libi": 50})
    assert result["matched_columns"] == ["Cow-libi"]


def test_unmatched_task_names_fall_back_to_the_contest_total():
    result = score_contest(RANKINGS, {"gamma": 130})
    assert result["matched_columns"] == ["Total"]
    assert result["model_total"] == 130
    # Against Total (200, 120, 40, 0) a 130 beats three.
    assert result["human_percentile"] == 75.0


def test_a_field_with_no_usable_column_at_all_reports_no_percentile():
    result = score_contest([{"Rank": 1}], {"gamma": 10})
    assert result["human_percentile"] is None
    assert result["model_total"] == 10


def test_non_numeric_and_missing_cells_count_as_zero():
    rankings = [
        {"alpha": "-"},  # withdrew
        {"alpha": None},
        {},  # column absent for this row
        {"alpha": "50"},  # numeric strings still count
    ]
    result = score_contest(rankings, {"alpha": 10})
    assert result["human_percentile"] == 75.0


def test_medals_use_the_contests_published_cutoffs():
    result = score_contest(
        RANKINGS,
        {"alpha": 100, "beta": 100},
        gold_cutoff=200.0,
        silver_cutoff=120.0,
        bronze_cutoff=40.0,
    )
    assert result["medal"] == "Gold"
    assert result["n_contestants"] == 4


# --------------------------------------------------------------------------- #
# Contest identity — USACO splits one round into two rankings
# --------------------------------------------------------------------------- #
def test_the_contest_map_reads_the_division_split_out_of_the_problems_column():
    """The only place the split is stated is the contestant row's own list."""
    rows = [
        {
            "contest_id": "USACO-2025-January_Contest-platinum",
            "problems": json.dumps(["USACO-2025-January_Contest-platinum_Cow"]),
        },
        {
            "contest_id": "USACO-2025-January_Contest-combined",
            "problems": json.dumps(
                [
                    "USACO-2025-January_Contest-bronze_Moo",
                    "USACO-2025-January_Contest-gold_Hay",
                ]
            ),
        },
        {"contest_id": "IOI-2025-contest", "problems": ["IOI-2025-contest-beechtree"]},
    ]
    mapping = build_problem_to_contest_map(rows)
    assert (
        mapping["USACO-2025-January_Contest-platinum_Cow"]
        == "USACO-2025-January_Contest-platinum"
    )
    assert (
        mapping["USACO-2025-January_Contest-bronze_Moo"]
        == "USACO-2025-January_Contest-combined"
    )
    assert mapping["IOI-2025-contest-beechtree"] == "IOI-2025-contest"


def test_a_division_listed_by_the_wrong_row_is_not_mapped():
    """Upstream's filter: a `-combined` row never claims a platinum problem."""
    rows = [
        {
            "contest_id": "USACO-2025-January_Contest-combined",
            "problems": json.dumps(["USACO-2025-January_Contest-platinum_Cow"]),
        }
    ]
    assert build_problem_to_contest_map(rows) == {}


def test_the_contestant_table_wins_over_the_id_derived_contest():
    """Upstream reads the mapping out of the table first and only derives one
    when the table lists nothing.

    On the published data the two routes happen to agree everywhere, so nothing
    else here would notice the table being ignored — but the table is the only
    one of the two that can follow a release that regroups a round.
    """
    mapping = build_problem_to_contest_map(
        [{"contest_id": "IOI-2025-day2", "problems": ["IOI-2025-contest-beechtree"]}]
    )
    assert (
        resolve_contest_id(
            "IOI-2025-contest-beechtree",
            "IOI-2025-contest",
            mapping,
            {"IOI-2025-day2", "IOI-2025-contest"},
        )
        == "IOI-2025-day2"
    )


def test_an_unlisted_usaco_problem_falls_back_to_its_division_suffix():
    resolved = resolve_contest_id(
        "USACO-2025-January_Contest-platinum_Cow",
        "USACO-2025-January_Contest",
        {},
        {"USACO-2025-January_Contest-platinum"},
    )
    assert resolved == "USACO-2025-January_Contest-platinum"


def test_a_non_usaco_problem_resolves_to_the_contest_its_id_names():
    assert (
        resolve_contest_id("IOI-2025-contest-beechtree", "IOI-2025-contest", {}, set())
        == "IOI-2025-contest"
    )


def test_the_cco_rename_falls_back_to_the_identifier_the_table_actually_uses():
    """Upstream renames the CCO rounds and the published table does not, so the
    ladder has to come back to the raw identifier or CCO stops matching."""
    raw = "CCO-2024-Canadian_Computing_Competition_Senior"
    assert normalize_contest_identifier(raw) == "CCO-2024-Senior"
    assert resolve_contest_id(f"{raw}-x", raw, {}, {raw}) == raw


# --------------------------------------------------------------------------- #
# Recalculated_Total — upstream prefers it wherever it exists
# --------------------------------------------------------------------------- #
def test_a_recalculated_total_column_replaces_the_per_task_sum():
    """The two are not the same number: the column is the contest's own
    re-derivation, and on the published data it differs from the per-task sum
    for 20-83% of contestants on each of the 10 contests that carry it."""
    rankings = [
        {"alpha": 0, "beta": 0, "Recalculated_Total": 300},
        {"alpha": 100, "beta": 100, "Recalculated_Total": 10},
    ]
    result = score_contest(rankings, {"alpha": 100, "beta": 0})
    assert result["matched_columns"] == ["Recalculated_Total"]
    # Against the column the model's 100 beats only the 10 -> 50th. Summing the
    # task columns would have compared it against 0 and 200 instead; the column
    # name above is what discriminates, the percentile pins the values.
    assert result["human_percentile"] == 50.0


def test_the_recalculated_branch_drops_unparseable_rows_rather_than_zeroing_them():
    """Upstream's `.dropna()`. Zeroing instead would invent contestants the
    model outscores, which inflates the percentile."""
    rankings = [
        {"Recalculated_Total": 300},
        {"Recalculated_Total": None},
        {"Recalculated_Total": "n/a"},
    ]
    result = score_contest(rankings, {"alpha": 10})
    # One usable contestant, who beat the model. Zeroing the other two would
    # have reported 66.7 instead.
    assert result["human_percentile"] == 0.0


def test_rows_without_the_column_still_take_the_per_task_branch():
    result = score_contest(RANKINGS, {"alpha": 100, "beta": 100})
    assert result["matched_columns"] == ["alpha", "beta"]


# --------------------------------------------------------------------------- #
# Contests that publish cutoffs but no contestants
# --------------------------------------------------------------------------- #
def test_an_empty_field_still_awards_the_medal_its_cutoffs_imply():
    """Upstream's `df.empty` branch. Every USACO row in the release is this
    shape, so reporting no medal here would lose all 22 of them."""
    result = score_contest(
        [], {"alpha": 100}, gold_cutoff=90.0, silver_cutoff=60.0, bronze_cutoff=30.0
    )
    assert result["human_percentile"] is None
    assert result["medal"] == "Gold"
    assert result["n_contestants"] == 0


def test_a_usaco_combined_contest_reports_nothing():
    """Its metric is a promotion threshold that the published dataset does not
    carry; upstream returns no medal and no percentile for it."""
    result = score_contest(
        [],
        {"bronze_Moo": 100},
        contest_id="USACO-2025-January_Contest-combined",
        gold_cutoff=800.0,
        silver_cutoff=750.0,
        bronze_cutoff=700.0,
    )
    assert result["human_percentile"] is None
    assert result["medal"] is None
