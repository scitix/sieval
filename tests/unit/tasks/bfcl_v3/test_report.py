"""Cover the two BFCL v3 group rollups: what each mean is over, and what carries
an interval.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest

from sieval.community.bfcl_v3 import (
    calculate_unweighted_accuracy,
    calculate_weighted_accuracy,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
)


def test_unweighted_is_the_mean_of_rates_not_the_pooled_rate():
    # 1/1 and 0/99: the pooled rate is 0.01, the unweighted mean is 0.5.
    # Every cell carries all THREE keys upstream reads, `display_accuracy`
    # included -- both helpers dereference it unconditionally, so a two-key
    # cell raises `KeyError` rather than returning a wrong number.
    cells = [
        {"accuracy": 1.0, "total_count": 1, "display_accuracy": 1.0},
        {"accuracy": 0.0, "total_count": 99, "display_accuracy": 0.0},
    ]
    assert calculate_unweighted_accuracy(cells)["accuracy"] == pytest.approx(0.5)
    assert calculate_weighted_accuracy(cells)["accuracy"] == pytest.approx(0.01)


@pytest.mark.anyio
async def test_non_live_rollup_nests_simple_ast(non_live_report):
    # Percentage points throughout: simple 100 (400 rows), java 0 (100),
    # javascript 0 (50)  ->  simple_ast = (100 + 0 + 0) / 3
    assert non_live_report["simple_ast"] == pytest.approx(100 / 3)
    # ast_summary = mean(simple_ast, multiple, parallel, parallel_multiple)
    assert non_live_report["ast_summary"] == pytest.approx(
        (100 / 3 + 100.0 + 100.0 + 100.0) / 4
    )
    # overall additionally folds in irrelevance
    assert non_live_report["non_live_overall_acc"] == pytest.approx(
        (100 / 3 + 100.0 + 100.0 + 100.0 + 100.0) / 5
    )
    assert non_live_report["score"] == non_live_report["non_live_overall_acc"]


@pytest.mark.anyio
async def test_non_live_headline_carries_no_interval(non_live_report):
    for key in (
        "score_ci95",
        "non_live_overall_acc_ci95",
        "simple_ast_ci95",
        "ast_summary_ci95",
    ):
        assert key not in non_live_report, (
            f"{key} is an unweighted mean of category rates; an interval on it "
            "would bracket a different statistic than the number beside it."
        )
    # And the population key that would have to sit beside such an interval is
    # absent too -- the pair is emitted whole or not at all.
    assert "n_problems" not in non_live_report


@pytest.mark.anyio
async def test_non_live_per_category_rates_all_carry_the_triple(non_live_report):
    units = non_live_report["ci95_units"]
    for category in (
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "java",
        "javascript",
        "irrelevance",
    ):
        assert f"{category}_ci95" in non_live_report
        assert f"n_{category}" in non_live_report
        assert units[category] == f"n_{category}"


@pytest.mark.anyio
async def test_live_headline_carries_an_interval(live_report):
    assert "score_ci95" in live_report
    assert "n_problems" in live_report
    assert live_report["ci95_units"]["score"] == "n_problems"
    assert "ast_summary_ci95" in live_report
    # The headline and its aliased twin are ONE number, carrying ONE interval.
    assert live_report["live_overall_acc"] == live_report["score"]
    assert live_report["live_overall_acc_ci95"] == live_report["score_ci95"]
    assert live_report["ci95_units"]["live_overall_acc"] == "n_problems"


@pytest.mark.anyio
async def test_live_rollups_are_pooled_over_rows_not_means_of_rates(live_report):
    """Weighted, so both rollups are the rate over the union of their rows.

    The six categories run from 16.7 to 80.1 at sizes from 16 to 1053, so the
    unweighted mean of them (41.1) is nowhere near the pooled rate -- which is
    what makes the choice of helper visible in the number rather than only in
    the source.
    """
    assert live_report["live_overall_acc"] == pytest.approx(100 * 1426 / 2251)
    assert live_report["ast_summary"] == pytest.approx(100 * 982 / 1351)
    # `ast_summary` is over the four AST categories only: the two relevance sets
    # join at the overall, and folding them in here would make it the headline.
    assert live_report["n_ast"] == 1351
    assert live_report["n_problems"] == 2251
    assert live_report["ci95_units"]["ast_summary"] == "n_ast"


@pytest.mark.anyio
async def test_every_live_interval_brackets_the_rate_printed_beside_it(live_report):
    """The published rate and its bounds have to be in the same units.

    Both come off the same per-sample list: the rate is that list's mean scaled
    to percentage points, the bounds are estimated from it as a probability and
    scaled the same way. Scale the list instead of the rate and the estimator
    reads every rate above one point as a saturated set, which still publishes
    a triple -- a plausible-looking interval that brackets 100 rather than the
    number it is printed beside.
    """
    units = live_report["ci95_units"]
    assert units, "a live report with no declared interval proves nothing here"
    for metric in units:
        low, high = live_report[f"{metric}_ci95"]
        assert low < live_report[metric] < high, (
            f"{metric} is {live_report[metric]} but its interval is [{low}, {high}]"
        )
        assert 0.0 <= low < high <= 100.0


@pytest.mark.anyio
async def test_a_missing_sample_scores_wrong_not_excluded(non_live_report_short):
    # 399 of simple's 400 came back, all correct. The denominator is still 400.
    assert non_live_report_short["simple"] == pytest.approx(100 * 399 / 400)
    assert non_live_report_short["n_simple"] == 400


@pytest.mark.anyio
async def test_a_failed_sample_moves_nothing_but_the_fail_count(
    non_live_report_short, non_live_report_with_fails
):
    """The denominator is DECLARED, so charging a failure is already done.

    A failed sample has no category to be attributed to -- one that failed in
    `preprocess` never read a row -- and under this policy it does not need
    one: it is missing from its category's numerator and the denominator never
    moved. So `fails` is the only key that reacts to it.
    """
    assert non_live_report_short["fails"] == 0
    assert non_live_report_with_fails["fails"] == 1
    assert non_live_report_with_fails["score"] == non_live_report_short["score"]
    assert non_live_report_with_fails["simple"] == non_live_report_short["simple"]


@pytest.mark.anyio
async def test_both_groups_declare_the_column_and_the_population(
    non_live_report, live_report
):
    """`score_key` is built from the group key, so no static reader resolves it.

    It names a key the report writes, and the report is the only place that can
    be checked -- which is why it is asserted here rather than left to the
    preflight, whose rule 4 skips a computed key.
    """
    for report, group in ((non_live_report, "non_live"), (live_report, "live")):
        assert report[SCORE_KEY_FIELD] == f"{group}_overall_acc"
        assert report[SCORE_KEY_FIELD] in report
        assert report[report[SCORE_KEY_FIELD]] == report["score"]
        assert report[DENOMINATOR_FIELD] == DENOMINATOR_REQUESTED
