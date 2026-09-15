"""Tests for scripts/post_edit_checks.py.

The script is called by three agent harnesses after a file edit. Tests cover
which checks each path selects, and that the script never fails the caller —
a post-edit check informs, it does not block the edit.

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "post_edit_checks.py"


def _load():
    spec = importlib.util.spec_from_file_location("post_edit_checks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["post_edit_checks"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("sieval/tasks/foo.py", {"stubs", "meta", "ruff", "ty"}),
        ("sieval/datasets/bar.py", {"stubs", "meta", "ruff", "ty"}),
        ("sieval/tasks/__init__.pyi", {"stubs", "ruff", "ty"}),
        ("sieval/core/runners/baz.py", {"ruff", "ty"}),
        ("tests/unit/test_x.py", {"ruff", "ty"}),
        ("README.md", set()),
        ("pyproject.toml", set()),
    ],
)
def test_selects_checks_by_path(path: str, expected: set[str]):
    mod = _load()
    assert mod.select_checks(path) == expected


def test_absolute_paths_match_the_same_way():
    """Harnesses pass absolute paths; the sieval/ prefix must still match."""
    mod = _load()
    assert mod.select_checks("/home/u/repo/sieval/tasks/foo.py") == {
        "stubs",
        "meta",
        "ruff",
        "ty",
    }


def test_similar_paths_outside_sieval_do_not_match():
    """A vendored or nested `sieval/tasks/` elsewhere is still a task file, but
    a lookalike like `mysieval/tasks/` is not."""
    mod = _load()
    assert "meta" not in mod.select_checks("mysieval/tasks/foo.py")


def test_empty_path_selects_nothing():
    mod = _load()
    assert mod.select_checks("") == set()


def test_main_exits_zero_on_unmatched_path():
    mod = _load()
    assert mod.main(["README.md"]) == 0


def test_main_exits_zero_with_no_argument():
    mod = _load()
    assert mod.main([]) == 0


def test_main_exits_zero_even_when_a_check_fails(monkeypatch):
    """A failing lint must not fail the caller's edit."""
    mod = _load()
    monkeypatch.setattr(mod, "_run", lambda *a, **k: 1)
    assert mod.main(["sieval/tasks/foo.py"]) == 0
