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


def test_a_global_the_old_module_held_is_forwarded():
    """A ``None`` here silently drops the deterministic seed (#204)."""
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    with pytest.warns(DeprecationWarning, match=r"sieval\.cli\.session"):
        seed = getattr(shim, "DETERMINISTIC_DEFAULT_SEED", None)
    assert seed is not None
    assert seed is session.DETERMINISTIC_DEFAULT_SEED


def test_every_engine_global_resolves_through_the_alias():
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    names = [n for n in vars(session) if not n.startswith("__")]
    with pytest.warns(DeprecationWarning):
        for name in names:
            assert getattr(shim, name) is getattr(session, name), name


def test_a_resolver_not_in_all_is_forwarded_from_resolution():
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    name = next(n for n in vars(resolution) if not hasattr(session, n))
    with pytest.warns(DeprecationWarning):
        assert getattr(shim, name) is getattr(resolution, name)


def test_an_unknown_name_still_raises_attribute_error():
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    assert getattr(shim, "no_such_name", None) is None
    with pytest.raises(AttributeError, match="no_such_name"):
        shim.no_such_name  # noqa: B018


def test_a_dunder_the_engine_holds_is_not_forwarded(monkeypatch):
    """``inspect.unwrap`` and the import system probe dunders; they must not
    see the engine's."""
    monkeypatch.setattr(session, "__wrapped__", object(), raising=False)
    with pytest.warns(DeprecationWarning):
        shim = _import_shim()
    assert not hasattr(shim, "__wrapped__")
