"""Unit tests for ranking a model against a contest's human contestants.

The load-bearing behaviour is that humans are re-totaled over **only** the tasks
the model was scored on: sieval filters interactive problems out, so a contest is
often partial, and comparing a partial model total against full human totals
would understate every model by however much it skipped.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

from sieval.community.liveoibench.rankings import (
    calculate_percentile,
    identify_task_columns,
    medal_from_cutoffs,
    normalize_name,
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
