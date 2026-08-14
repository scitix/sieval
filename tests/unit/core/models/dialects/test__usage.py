"""Tests for the usage lifting shared by the OpenAI-shaped dialects."""

from types import SimpleNamespace

import pytest

from sieval.core.models.dialects._usage import usage_stats


class TestOptionalDetails:
    def test_details_are_lifted_when_reported(self):
        raw = SimpleNamespace(
            total_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=6,
                accepted_prediction_tokens=3,
                rejected_prediction_tokens=1,
            ),
        )

        stats = usage_stats(raw, 10, 20)

        assert stats.cached_tokens == 4
        assert stats.reasoning_tokens == 6
        assert stats.accepted_prediction_tokens == 3
        assert stats.rejected_prediction_tokens == 1

    def test_absent_detail_objects_read_as_none_never_zero(self):
        """The reason the fields are optional: silence is not a measurement."""
        stats = usage_stats(SimpleNamespace(total_tokens=30), 10, 20)

        assert stats.cached_tokens is None
        assert stats.reasoning_tokens is None
        assert stats.accepted_prediction_tokens is None
        assert stats.rejected_prediction_tokens is None

    def test_a_reported_zero_stays_zero(self):
        """A server saying "nothing was cached" measured something."""
        raw = SimpleNamespace(
            total_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )

        assert usage_stats(raw, 10, 20).cached_tokens == 0

    def test_a_partial_detail_object_only_blanks_the_missing_field(self):
        raw = SimpleNamespace(
            total_tokens=30,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=6),
        )

        stats = usage_stats(raw, 10, 20)

        assert stats.reasoning_tokens == 6
        assert stats.accepted_prediction_tokens is None

    @pytest.mark.parametrize(
        "value",
        [-1, True, "4", 1.5, None],
        ids=["negative", "bool", "str", "float", "none"],
    )
    def test_a_malformed_detail_reads_as_unreported_and_never_raises(self, value):
        """Observational counts must not cost a reply that was already billed."""
        raw = SimpleNamespace(
            total_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=value),
        )

        assert usage_stats(raw, 10, 20).cached_tokens is None

    def test_no_subset_relation_is_enforced(self):
        """``reasoning <= output`` is an OpenAI convention, not a wire guarantee.

        A server counting reasoning outside its completion count is exactly the
        one whose reported total exceeds the computed one, so rejecting the
        pair would discard the case the field exists to describe.
        """
        raw = SimpleNamespace(
            total_tokens=200,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=170),
        )

        stats = usage_stats(raw, 10, 20)

        assert stats.reasoning_tokens == 170
        assert stats.total_tokens == 30
        assert stats.reported_total_tokens == 200


class TestReportedTotal:
    def test_a_divergent_total_is_recorded(self):
        assert (
            usage_stats(SimpleNamespace(total_tokens=99), 10, 20).reported_total_tokens
            == 99
        )

    def test_an_agreeing_total_is_dropped(self):
        """Agreement is the common case and carries nothing worth storing."""
        assert (
            usage_stats(SimpleNamespace(total_tokens=30), 10, 20).reported_total_tokens
            is None
        )

    def test_an_absent_total_is_dropped(self):
        assert usage_stats(SimpleNamespace(), 10, 20).reported_total_tokens is None

    def test_the_computed_total_never_takes_the_reported_value(self):
        assert usage_stats(SimpleNamespace(total_tokens=99), 10, 20).total_tokens == 30
