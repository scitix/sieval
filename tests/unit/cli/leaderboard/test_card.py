"""Tests for the alignment card loader.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from pathlib import Path

import pytest

from sieval.cli.leaderboard.card import AlignmentCard, load_card


@pytest.fixture
def card_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample-card.md"
    path.write_text(
        """---
reference:
  kind: tr
  source: "arXiv:0000.00000v1"
  title: "Sample Paper"
tolerance: 3.0
reference_scores:
  model-a:
    task_x: 50.0
    task_y: 72.5
  model-b:
    task_x: 48.0
---

# Sample card

Prose body unused by the loader.
""",
        encoding="utf-8",
    )
    return path


class TestLoadCardHappyPath:
    def test_returns_alignment_card_with_all_fields(self, card_file: Path) -> None:
        card = load_card(card_file)
        assert isinstance(card, AlignmentCard)
        assert card.kind == "tr"
        assert card.source == "arXiv:0000.00000v1"
        assert card.title == "Sample Paper"
        assert card.tolerance == 3.0
        assert card.reference_scores == {
            "model-a": {"task_x": 50.0, "task_y": 72.5},
            "model-b": {"task_x": 48.0},
        }
        assert card.path == card_file

    def test_user_defined_kind_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.md"
        path.write_text(
            "---\n"
            "reference:\n"
            "  kind: user-defined\n"
            '  source: "internal-qwen3-eval-20260415"\n'
            '  title: "Internal SFT baseline"\n'
            "tolerance: 3.0\n"
            "reference_scores: {m: {t: 42.0}}\n"
            "---\n",
            encoding="utf-8",
        )
        card = load_card(path)
        assert card.kind == "user-defined"
        assert card.source == "internal-qwen3-eval-20260415"


class TestLoadCardErrors:
    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_card(tmp_path / "nope.md")

    def test_missing_frontmatter(self, tmp_path: Path) -> None:
        path = tmp_path / "no-fm.md"
        path.write_text("# No frontmatter here\n")
        with pytest.raises(ValueError, match="frontmatter"):
            load_card(path)

    def test_missing_reference_field(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\ntolerance: 3.0\nreference_scores: {model-a: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="reference"):
            load_card(path)

    def test_missing_reference_kind(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {source: s, title: t}\ntolerance: 3.0\n"
            "reference_scores: {m: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="reference.kind must be one of"):
            load_card(path)

    def test_invalid_reference_kind(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: internal, source: s, title: t}\ntolerance: 3.0\n"
            "reference_scores: {m: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="reference.kind must be one of"):
            load_card(path)

    def test_missing_reference_source(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, title: x}\ntolerance: 3.0\n"
            "reference_scores: {m: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="reference.source"):
            load_card(path)

    def test_missing_reference_title(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, source: s}\ntolerance: 3.0\n"
            "reference_scores: {m: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="reference.title"):
            load_card(path)

    def test_missing_tolerance(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, source: s, title: t}\n"
            "reference_scores: {m: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="tolerance"):
            load_card(path)

    def test_missing_reference_scores(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, source: s, title: t}\ntolerance: 3.0\n---\n"
        )
        with pytest.raises(ValueError, match="reference_scores"):
            load_card(path)

    def test_reference_scores_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, source: s, title: t}\ntolerance: 3.0\n"
            "reference_scores: {}\n---\n"
        )
        with pytest.raises(ValueError, match="reference_scores.*empty"):
            load_card(path)

    def test_tolerance_rejects_bool(self, tmp_path: Path) -> None:
        """``True`` is numerically ``1`` and would otherwise pass the ``> 0``
        check, but "tolerance: true" is clearly a schema error."""
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, source: s, title: t}\ntolerance: true\n"
            "reference_scores: {m: {t: 1}}\n---\n"
        )
        with pytest.raises(ValueError, match="tolerance"):
            load_card(path)

    def test_score_rejects_bool(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text(
            "---\nreference: {kind: tr, source: s, title: t}\ntolerance: 3.0\n"
            "reference_scores: {m: {t: true}}\n---\n"
        )
        with pytest.raises(ValueError, match="score must be numeric"):
            load_card(path)
