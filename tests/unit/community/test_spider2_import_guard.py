"""The vendored Spider 2.0 evaluator's import side effects, and their undoing.

Upstream's ``evaluate.py`` is a command-line entry point as well as a comparison
library, so at module scope it does ``sys.stdout = TeeOutput("log.txt")`` and
points ``sys.stderr`` at the same object. Importing it therefore opens (and
truncates) a file in whatever the current directory happens to be, and redirects
the whole process's output for the rest of the run. The file is vendored
byte-identical, so the fix lives in the package ``__init__``, which is the only
door into it.

Both halves have to be checked **out of process**: a package is imported once
per interpreter, so a test that ran in this one would be asserting about an
import that already happened, under a ``log.txt`` this suite may itself have
created. Two children cover the two shapes — a clean directory and one that
already holds a ``log.txt`` worth keeping — and each answers every question, so
the interpreter start-up is paid twice rather than six times.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("google.cloud.bigquery", reason="requires the `spider2` extra")

#: Repo root: <root>/tests/unit/community/<this file>.
_ROOT = Path(__file__).parents[3]

#: Content of a `log.txt` that was already there. Upstream opens the name with
#: mode "w", so a guard that merely restores the streams still loses this.
_SENTINEL = "a log that belongs to someone else\n"

_CHILD = """
import json
import sys
from pathlib import Path

before = (sys.stdout, sys.stderr)
import sieval.community.spider2 as spider2

log = Path("log.txt")
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "stdout_restored": sys.stdout is before[0],
            "stderr_restored": sys.stderr is before[1],
            "log_exists": log.exists(),
            "log_text": log.read_text() if log.exists() else None,
            # Listed while building this payload, i.e. before the report is
            # written, so it holds what the import left and nothing else.
            "leftovers": sorted(p.name for p in Path().iterdir()),
            "exports": sorted(spider2.__all__),
            "compare_is_callable": callable(spider2.compare_pandas_table),
        }
    )
)
"""


def _probe(directory: Path) -> dict:
    """Import the package in a fresh interpreter whose cwd is *directory*."""
    report = directory / "report.json"
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, str(report)],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={"PYTHONPATH": str(_ROOT), "PATH": "/usr/bin:/bin", "HOME": str(directory)},
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(report.read_text())


@pytest.fixture(scope="module")
def in_a_clean_directory(tmp_path_factory) -> dict:
    return _probe(tmp_path_factory.mktemp("clean"))


@pytest.fixture(scope="module")
def over_an_existing_log(tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("with-log")
    (directory / "log.txt").write_text(_SENTINEL)
    return _probe(directory)


def test_the_import_gives_stdout_and_stderr_back(in_a_clean_directory):
    """Otherwise every later print in the process — this session's progress
    bar, another task's logs, a worker's traceback — goes into upstream's file
    instead, and nothing says so."""
    assert in_a_clean_directory["stdout_restored"] is True
    assert in_a_clean_directory["stderr_restored"] is True


def test_the_import_leaves_no_log_behind(in_a_clean_directory):
    """The handle upstream opened is closed and the empty file removed, so a
    run directory does not acquire a stray `log.txt` from a grader import."""
    assert in_a_clean_directory["log_exists"] is False
    assert in_a_clean_directory["leftovers"] == []


def test_a_log_that_was_already_there_survives_intact(over_an_existing_log):
    """The destructive half: upstream opens the name with mode "w".

    Restoring the streams alone would still have truncated this file, and the
    directory the grader imports from is the user's, not ours.
    """
    assert over_an_existing_log["log_exists"] is True
    assert over_an_existing_log["log_text"] == _SENTINEL
    assert over_an_existing_log["stdout_restored"] is True
    # The stash is moved back, not copied — no `log.txt.sieval-*` left over.
    assert over_an_existing_log["leftovers"] == ["log.txt"]


def test_the_comparison_is_still_reachable_through_the_guard(in_a_clean_directory):
    """The guard undoes the side effects without costing the exports — a
    wrapper that swallowed the import would pass every test above."""
    assert in_a_clean_directory["exports"] == [
        "compare_multi_pandas_table",
        "compare_pandas_table",
        "extract_sql_query",
        "load_gold_csv",
        "resolve_gold_paths",
    ]
    assert in_a_clean_directory["compare_is_callable"] is True
