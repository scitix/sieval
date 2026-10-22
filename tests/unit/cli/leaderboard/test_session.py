"""The deprecated ``sieval.cli.leaderboard.session`` alias (#194).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import sys

import pytest

import sieval.cli.resolution as resolution
import sieval.cli.session as session

_SHIM = "sieval.cli.leaderboard.session"


def _import_shim():
    """Import the shim afresh: the warning fires once, at module execution."""
    sys.modules.pop(_SHIM, None)
    return importlib.import_module(_SHIM)


def test_old_path_warns_with_the_new_one():
    with pytest.warns(DeprecationWarning, match=r"sieval\.cli\.session"):
        _import_shim()


@pytest.mark.parametrize(
    ("name", "home"),
    [
        ("EvalSession", session),
        ("arun_session", session),
        ("run_session", session),
        ("resolve_deterministic", session),
        ("unwrap_proxies", session),
        ("RootConfigDict", session),
        ("ModelConfigDict", session),
        ("resolve_task_class", resolution),
        ("resolve_dataset_class", resolution),
        ("resolve_class", resolution),
    ],
)
def test_old_path_hands_out_the_same_objects(name, home):
    """Identity, not equality: ``isinstance`` and patching need the very same
    class and function the new path holds."""
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    assert getattr(shim, name) is getattr(home, name)


def test_shim_exports_exactly_the_documented_names():
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    assert set(shim.__all__) == {
        "EvalSession",
        "ModelConfigDict",
        "RootConfigDict",
        "arun_session",
        "resolve_class",
        "resolve_dataset_class",
        "resolve_deterministic",
        "resolve_task_class",
        "run_session",
        "unwrap_proxies",
    }


def test_package_no_longer_exports_aggregation_api():
    import sieval.cli.leaderboard as package

    for gone in ("AlignmentCard", "scan_runs", "build_matrix", "leaderboard_app"):
        assert not hasattr(package, gone)
