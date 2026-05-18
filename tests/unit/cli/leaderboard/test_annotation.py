"""Tests for the per-cell annotation helper.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import dataclasses

import pytest

from sieval.cli.leaderboard.annotation import (
    annotate_cell,
    display_precision,
)


class TestAnnotateCell:
    def test_pass_within_tolerance(self) -> None:
        result = annotate_cell(observed=48.1, reference=48.3, tolerance=3.0)
        assert result.observed == 48.1
        assert result.reference == 48.3
        assert result.diff == pytest.approx(-0.2)
        assert result.tolerance == 3.0
        assert result.status == "pass"

    def test_fail_over_tolerance(self) -> None:
        result = annotate_cell(observed=80.1, reference=85.3, tolerance=3.0)
        assert result.status == "fail"
        assert result.diff == pytest.approx(-5.2)
        assert result.tolerance == 3.0

    def test_exact_tolerance_boundary_passes(self) -> None:
        # |diff| == tolerance should pass (inclusive).
        result = annotate_cell(observed=50.0, reference=47.0, tolerance=3.0)
        assert result.status == "pass"
        assert result.diff == pytest.approx(3.0)

    def test_exact_tolerance_negative_boundary_passes(self) -> None:
        result = annotate_cell(observed=47.0, reference=50.0, tolerance=3.0)
        assert result.status == "pass"
        assert result.diff == pytest.approx(-3.0)

    def test_just_over_tolerance_fails(self) -> None:
        result = annotate_cell(observed=50.01, reference=47.0, tolerance=3.0)
        assert result.status == "fail"

    def test_zero_diff(self) -> None:
        result = annotate_cell(observed=73.0, reference=73.0, tolerance=3.0)
        assert result.status == "pass"
        assert result.diff == 0.0

    def test_correlation_scale_tolerance(self) -> None:
        # metric units different (correlation coefficient)
        result = annotate_cell(observed=0.82, reference=0.85, tolerance=0.03)
        assert result.status == "pass"
        assert result.diff == pytest.approx(-0.03)

    def test_to_dict_carries_tolerance(self) -> None:
        d = annotate_cell(observed=0.82, reference=0.85, tolerance=0.03).to_dict()
        assert d["tolerance"] == 0.03
        assert set(d.keys()) == {
            "observed",
            "reference",
            "diff",
            "tolerance",
            "status",
        }

    def test_frozen_dataclass(self) -> None:
        result = annotate_cell(observed=1.0, reference=1.0, tolerance=0.1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = "fail"  # type: ignore[misc]


class TestDisplayPrecision:
    """Precision tracks tolerance magnitude so diffs never collapse to 0."""

    @pytest.mark.parametrize(
        ("tolerance", "expected"),
        [
            (3.0, 1),
            (1.0, 1),
            (0.5, 2),
            (0.3, 2),
            (0.1, 2),
            (0.05, 3),
            (0.03, 3),
            (0.01, 3),
            (0.001, 4),
        ],
    )
    def test_scales_with_tolerance(self, tolerance: float, expected: int) -> None:
        assert display_precision(tolerance) == expected

    def test_clamped_at_six(self) -> None:
        assert display_precision(1e-20) == 6

    def test_nonpositive_falls_back(self) -> None:
        # defensive: annotate_cell rejects these upstream
        assert display_precision(0.0) == 1
        assert display_precision(-1.0) == 1
