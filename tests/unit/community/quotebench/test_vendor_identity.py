"""The vendored QuoteBench modules must stay byte-identical to upstream.

Naming ``sieval/community/<pkg>/`` explicitly on a ruff or formatter invocation
overrides the exclude and rewrites byte-identical upstream code; nothing else
notices. The markers are the sharpest case: ``LMARK``/``RMARK`` delimit the exact
literal text every task is about, so a unicode literal quietly normalized to
ASCII would rewrite all 56 prompts while leaving the code perfectly readable.

Digests are of the upstream files at the pin, taken from the upstream tree
rather than from these copies -- a digest read off the copy would pin whatever
was copied, including a corrupted copy.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import hashlib
import unicodedata
from pathlib import Path

import pytest

PIN = "693325a671e65f889e5cd9d83965db9cc3b26dc2"

EXPECTED_SHA256 = {
    "core.py": "1fb1538b0ae41a2422d341b9bab1608fdddb31df5c35ae299f1fbbbd2ab7b755",
    "scenarios.py": "ac8c574df60c073c31cb95c295783d102da49e7d77a684a8a10a3ad9bd84c909",
    "shellesc.py": "faa9188024731d06b932f29e843e619abd073be41bd7856856265c7b9aa76d86",
}


def _vendor_dir() -> Path:
    import sieval.community.quotebench as pkg

    assert pkg.__file__ is not None
    return Path(pkg.__file__).parent


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA256))
def test_vendored_file_is_byte_identical_to_upstream(name: str) -> None:
    digest = hashlib.sha256((_vendor_dir() / name).read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[name], (
        f"{name} diverged from upstream {PIN}; a formatter run naming "
        f"community/ is the usual cause"
    )


def test_only_the_three_needed_modules_are_vendored() -> None:
    """Execution machinery is the evaluator's half. Vendoring `harness.py` here
    would put an executor one import away from a task."""
    present = {p.name for p in _vendor_dir().glob("*.py")}
    assert present == {"__init__.py", *EXPECTED_SHA256}


def test_markers_are_the_upstream_codepoints() -> None:
    from sieval.community.quotebench.core import LMARK, RMARK

    assert (ord(LMARK), ord(RMARK)) == (0x27EA, 0x27EB)
    assert unicodedata.name(LMARK) == "MATHEMATICAL LEFT DOUBLE ANGLE BRACKET"
    assert unicodedata.name(RMARK) == "MATHEMATICAL RIGHT DOUBLE ANGLE BRACKET"


def test_markers_survive_in_the_instructions_that_carry_literal_text() -> None:
    """A normalization that ate the markers would leave the code readable and
    every prompt wrong, so assert on the rendered instructions, not the constants.

    48 of the 56 instructions carry marked literal text. The 8 that do not are
    exactly the ``find-glob`` and ``bulk-rename`` families, and that is structural
    rather than incidental: those two put their hazards in the fixture -- hostile
    filenames already on disk that the command has to survive -- instead of
    handing the model a literal string to reproduce. Pinning the split, not just
    the count, keeps this test biting if a marker is lost from a family that
    should have one.
    """
    from sieval.community.quotebench.core import LMARK, RMARK
    from sieval.community.quotebench.scenarios import all_tasks

    tasks = all_tasks()
    assert len(tasks) == 56
    unmarked = {
        t.scenario
        for t in tasks
        if not (LMARK in t.instruction and RMARK in t.instruction)
    }
    assert unmarked == {"find-glob", "bulk-rename"}
    marked = [t for t in tasks if LMARK in t.instruction and RMARK in t.instruction]
    assert len(marked) == 48
    assert {t.scenario for t in marked} == {t.scenario for t in tasks} - unmarked


def test_frozen_core_shape() -> None:
    from sieval.community.quotebench.scenarios import all_tasks

    tasks = all_tasks()
    assert len(tasks) == 56
    assert len({t.scenario for t in tasks}) == 14
    assert sorted({t.tier for t in tasks}) == [0, 1, 2, 3]
    assert len({t.task_id for t in tasks}) == 56
