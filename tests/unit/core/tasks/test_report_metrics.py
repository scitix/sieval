"""``ReportMetrics`` and how a task class finds its declaration.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from typing import ClassVar

import pytest

from sieval.core.tasks.report_metrics import (
    RUNNER_GEN_METRICS,
    ReportMetrics,
    class_path,
    declared_report_metrics,
    report_metrics,
)


class TestReportMetrics:
    def test_a_name_matches_as_a_name(self) -> None:
        declared = report_metrics("score", patterns=("score_{category}",))
        assert declared.match("score") == "name"

    def test_a_pattern_matches_a_group_from_the_data(self) -> None:
        declared = report_metrics("score", patterns=("score_{category}",))
        assert declared.match("score_stem") == "pattern"
        assert declared.match("score_social_sciences") == "pattern"

    def test_a_placeholder_is_not_empty(self) -> None:
        assert report_metrics(patterns=("score_{category}",)).match("score_") is None

    def test_a_pattern_matches_the_whole_name(self) -> None:
        declared = report_metrics(patterns=("score_{category}",))
        assert declared.match("xscore_stem") is None
        assert declared.match("n_problems") is None

    def test_literal_parts_are_not_regex(self) -> None:
        declared = report_metrics(patterns=("pass@1_{language}",))
        assert declared.match("pass@1_cpp") == "pattern"
        assert declared.match("passX1_cpp") is None

    def test_two_placeholders(self) -> None:
        declared = report_metrics(patterns=("turn_{turn}_{grade}_accuracy",))
        assert declared.match("turn_1_strict_accuracy") == "pattern"

    @pytest.mark.parametrize("pattern", ["{category}", "score", ""])
    def test_a_pattern_needs_a_literal_and_a_placeholder(self, pattern: str) -> None:
        with pytest.raises(ValueError, match="literal part"):
            report_metrics(patterns=(pattern,))

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            report_metrics("")

    def test_union_keeps_both_and_drops_repeated_patterns(self) -> None:
        a = report_metrics("score", patterns=("score_{x}",))
        b = report_metrics("fails", patterns=("score_{x}", "n_{x}"))
        assert (a | b) == ReportMetrics(
            frozenset({"score", "fails"}), ("score_{x}", "n_{x}")
        )
        assert (a | {"n"}).names == {"score", "n"}

    def test_union_with_a_str_is_refused(self) -> None:
        # Iterating a str would add one name per character.
        with pytest.raises(TypeError, match="wrap the name"):
            report_metrics("fails") | "score"

    def test_json_sorts_names_and_keeps_pattern_order(self) -> None:
        declared = report_metrics("score", "fails", patterns=("b_{x}", "a_{x}"))
        assert declared.to_json() == {
            "names": ["fails", "score"],
            "patterns": ["b_{x}", "a_{x}"],
        }


async def _report(_self, _finals, _fails):
    return {}


class _Base:
    tags: ClassVar[frozenset[str]] = frozenset({"gen"})
    report_metrics = report_metrics("score")
    report = _report


class _Inherits(_Base):
    pass


class _OwnReport(_Base):
    async def report(self, finals, fails):
        return {"other": 1.0}


class _OwnReportDeclared(_Base):
    report_metrics = report_metrics("other")

    async def report(self, finals, fails):
        return {"other": 1.0}


class _Unknown(_Base):
    report_metrics = None


class _Clp:
    tags: ClassVar[frozenset[str]] = frozenset({"clp"})

    async def report(self, finals, fails):
        return {}


class TestDeclaredReportMetrics:
    def test_a_subclass_inherits_a_declaration_with_the_report(self) -> None:
        declared = declared_report_metrics(_Inherits)
        assert declared is not None
        assert declared.names == {"score"} | RUNNER_GEN_METRICS

    def test_a_report_below_the_declaration_is_unknown(self) -> None:
        # Its keys are its own; the declaration above never saw them.
        assert declared_report_metrics(_OwnReport) is None

    def test_a_report_declared_beside_it_is_known(self) -> None:
        declared = declared_report_metrics(_OwnReportDeclared)
        assert declared is not None and "other" in declared.names

    def test_none_declares_unknown(self) -> None:
        assert declared_report_metrics(_Unknown) is None

    def test_a_table_entry_declares_the_class_it_names(self) -> None:
        table = {class_path(_Clp): report_metrics("acc")}
        declared = declared_report_metrics(_Clp, table)
        # No rollouts, so the runner adds nothing.
        assert declared == report_metrics("acc")

    def test_the_class_attribute_wins_over_the_table(self) -> None:
        table = {class_path(_Base): report_metrics("from_table")}
        declared = declared_report_metrics(_Base, table)
        assert declared is not None and "score" in declared.names

    def test_no_declaration_is_unknown(self) -> None:
        assert declared_report_metrics(_Clp) is None

    def test_a_patched_report_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A report defined in another module than the class: `setattr` patching.
        import json

        monkeypatch.setattr(_Base, "report", json.dumps)
        assert declared_report_metrics(_Base) is None

    def test_a_declaration_that_is_not_report_metrics_is_refused(self) -> None:
        class Wrong(_Base):
            report_metrics = frozenset({"score"})

        with pytest.raises(TypeError, match="ReportMetrics or None"):
            declared_report_metrics(Wrong)

    def test_a_malformed_declaration_is_refused_below_a_nearer_report(self) -> None:
        # The report shadows it, but the author still wrote a broken declaration.
        class Wrong(_Base):
            report_metrics = frozenset({"score"})

        class Below(Wrong):
            async def report(self, finals, fails):
                return {}

        with pytest.raises(TypeError, match="ReportMetrics or None"):
            declared_report_metrics(Below)
