"""Unit tests for unpacking the LiveOIBench test corpus.

The load-bearing behaviour is atomicity. The unpack writes >30 GB and takes long
enough to be interrupted — Ctrl-C, a full disk, an OOM kill — and both the
script's own skip check and the dataset loader's ``require_tests`` guard read a
directory's *existence* as "this problem is materialized". A problem written in
place would therefore come back from a crash looking complete, and the missing
cases would surface much later as a subtask naming a test that is not there.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    """Import the script by path; `scripts/` is not a package."""
    path = ROOT / "scripts" / "materialize_liveoibench_tests.py"
    spec = importlib.util.spec_from_file_location("_materialize_liveoibench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materialize = _load_module()

TESTS = {
    "a1": {"input": "1\n", "output": "2\n"},
    "a2": {"input": "3\n", "output": "4\n"},
}


def test_a_problems_cases_land_in_the_upstream_layout(tmp_path):
    out_dir = tmp_path / "IOI" / "2025" / "contest" / "beechtree" / "tests"
    materialize.write_problem_tests(out_dir, TESTS)
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "a1.in",
        "a1.out",
        "a2.in",
        "a2.out",
    ]
    assert (out_dir / "a2.out").read_text() == "4\n"


def test_an_interrupted_problem_leaves_no_directory_to_mistake_for_a_whole_one(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "IOI" / "2025" / "contest" / "beechtree" / "tests"

    real_write = Path.write_text
    state = {"n": 0}

    def exploding_write(self, data, *args, **kwargs):
        state["n"] += 1
        if state["n"] > 2:
            raise KeyboardInterrupt("simulated Ctrl-C mid-problem")
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exploding_write)
    with pytest.raises(KeyboardInterrupt):
        materialize.write_problem_tests(out_dir, TESTS)

    # The guard that matters: nothing named `tests` exists, so neither the
    # script's skip check nor `require_tests` reads this problem as done.
    assert not out_dir.exists()
    assert out_dir.with_name("tests.partial").is_dir()


def test_a_crashed_partial_is_discarded_rather_than_merged(tmp_path):
    out_dir = tmp_path / "IOI" / "2025" / "contest" / "beechtree" / "tests"
    stale = out_dir.with_name("tests.partial")
    stale.mkdir(parents=True)
    (stale / "ghost.in").write_text("from an earlier crash\n")

    materialize.write_problem_tests(out_dir, TESTS)
    assert not (out_dir / "ghost.in").exists()
    assert not stale.exists()


def test_overwriting_replaces_the_whole_directory(tmp_path):
    out_dir = tmp_path / "IOI" / "2025" / "contest" / "beechtree" / "tests"
    materialize.write_problem_tests(out_dir, {"old": {"input": "x", "output": "y"}})
    materialize.write_problem_tests(out_dir, TESTS)
    assert not (out_dir / "old.in").exists()
    assert (out_dir / "a1.in").read_text() == "1\n"
