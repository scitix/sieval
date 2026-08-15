"""Task-test collection tweaks: NLTK staging under xdist, and mutation testing.

**NLTK staging.** ``sieval/community/ifbench/instructions_util.py`` calls
``download_nltk_resources()`` at module scope, so the corpora arrive with whoever
imports the vendored fork first. Serial that is fine; under ``-n auto`` several
workers import it while collecting ``test__ifbench_fixed_checkers.py`` and race to
unzip into one shared directory. The losers read a half-written archive and die
with ``FileExistsError`` / ``EOFError`` / ``LookupError`` — none of them the one
upstream's helper catches — and a collection error takes the whole run with it.
Repro outside CI: 4 concurrent cold imports, 3 fail; warmed first, 4 pass.

The warm import below runs at conftest *import* time, not in a fixture: the race
is between test-module imports during collection, which finishes before any
fixture runs. It imports the module the tests import, so no second corpus list
lives here. It is skipped outside xdist (one process cannot race itself, and the
import costs ~3.4s), and only the first worker to take the lock imports — the rest
return on a ``PYTEST_XDIST_TESTRUNUID``-keyed sentinel, so their imports still run
in parallel against corpora already on disk, and the sentinel cannot outlive its
run. ``ImportError`` is swallowed so a box without the ``ifbench`` extra still
fails at that one test module rather than taking this directory down.

**Mutation testing.**
``test_import_does_not_pull_the_optional_grader`` proves an optional dependency
is lazy-imported, which only means anything in an interpreter where that
dependency was never imported — so it runs one out of process
(``tests/unit/tasks/_import_probe.py``). Under mutation testing that child
inherits an environment pointing at mutmut's instrumented ``mutants/`` tree,
whose trampolines require mutmut's in-process runtime config — unavailable in a
bare interpreter — so the import crashes for reasons unrelated to what the test
verifies. These tests also exercise no ``sieval/core`` mutants, so skipping them
under mutation testing loses no kill power. ``MUTANT_UNDER_TEST`` is set by
mutmut for every pytest run it drives and is absent otherwise.

Skipping happens before fixture setup, so the probe interpreter never starts.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import contextlib
import os
import tempfile
from pathlib import Path

import pytest
from filelock import FileLock


def _warm_stem() -> Path | None:
    """Where this run's lock and sentinel live, or ``None`` outside xdist."""
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return None
    run_id = os.environ.get("PYTEST_XDIST_TESTRUNUID", "unknown")
    # Per-uid so two users on one box never contend for each other's lock file;
    # per-run so the sentinel cannot vouch for a previous run's corpora.
    return Path(tempfile.gettempdir()) / f"sieval-ifbench-nltk-{os.getuid()}-{run_id}"


def _warm_ifbench_nltk_corpora() -> None:
    stem = _warm_stem()
    if stem is None:
        return
    sentinel = Path(f"{stem}.warm")
    with FileLock(f"{stem}.lock"):
        if sentinel.exists():
            return
        with contextlib.suppress(ImportError):
            import sieval.community.ifbench.instructions  # noqa: F401
        # Written even when the import failed: it records that the one serialized
        # attempt happened, which is all the other workers wait on. Without the
        # extra they would otherwise queue up one at a time for the same
        # ImportError.
        sentinel.touch()


_warm_ifbench_nltk_corpora()


def pytest_sessionfinish():
    # Collection is long over, so the sentinel has no readers left. Left behind,
    # these accumulate one pair per parallel run until the box reboots.
    stem = _warm_stem()
    if stem is None:
        return
    for path in (Path(f"{stem}.warm"), Path(f"{stem}.lock")):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def pytest_collection_modifyitems(items):
    if not os.environ.get("MUTANT_UNDER_TEST"):
        return
    skip = pytest.mark.skip(
        reason="fresh-interpreter lazy-import check cannot run against "
        "mutmut's instrumented tree"
    )
    for item in items:
        if item.name.startswith("test_import_does_not_pull_"):
            item.add_marker(skip)
