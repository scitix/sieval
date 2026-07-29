"""Contract tests for the `ModuleIsolation` helper in `tests/conftest.py`.

Every assertion here is about `sys.modules` mechanics, so the subject is a
synthetic package built on disk rather than `sieval.tasks` / `sieval.datasets`.
Re-importing a real task module would re-run `@sieval_task` and trip the
duplicate-name guard, which is precisely why the five adopting fixtures have to
clear a registry first — coupling these tests to that would test the fixtures
again instead of the helper they share. The synthetic package mirrors the one
structural feature that matters: a lazy `__getattr__` that caches each export in
the package's own `__dict__`.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import importlib
import sys
from collections.abc import Iterator

import pytest

from tests.conftest import ModuleIsolation

_PKG = "module_isolation_probe"

_INIT = """
__all__ = ["Thing"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    import importlib

    module = importlib.import_module(f"{__name__}.thing")
    value = getattr(module, name)
    globals()[name] = value
    return value
"""


@pytest.fixture(scope="module", autouse=True)
def _probe_package(tmp_path_factory) -> Iterator[None]:
    """Put a synthetic package with a lazy `__getattr__` on `sys.path`."""
    root = tmp_path_factory.mktemp("module_isolation")
    pkg = root / _PKG
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_INIT)
    (pkg / "alpha.py").write_text('VALUE = "alpha"\n')
    (pkg / "beta.py").write_text('VALUE = "beta"\n')
    (pkg / "thing.py").write_text("class Thing:\n    pass\n")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(root))
        _purge()
        importlib.invalidate_caches()


def _purge() -> None:
    for name in [n for n in sys.modules if n == _PKG or n.startswith(f"{_PKG}.")]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _fresh_probe_modules() -> Iterator[None]:
    """No probe module survives into or out of a test."""
    _purge()
    yield
    _purge()


def _import(*suffixes: str):
    """Import the package plus the named submodules, returning them in order."""
    return tuple(
        importlib.import_module(_PKG if not s else f"{_PKG}.{s}")
        for s in ("", *suffixes)
    )


def test_evict_unbinds_the_parent_attribute():
    """A dropped module must not stay reachable as an attribute of its parent."""
    pkg, alpha = _import("alpha")
    assert pkg.__dict__["alpha"] is alpha  # the import bound it

    isolation = ModuleIsolation((f"{_PKG}.",))
    isolation.snapshot()
    isolation.evict()

    assert f"{_PKG}.alpha" not in sys.modules
    assert "alpha" not in pkg.__dict__, "evicted module still bound on its parent"


def test_evict_lets_from_package_import_re_execute_the_module():
    """`from pkg import submodule` must miss, not hit a dropped copy.

    The import system consults `hasattr(package, name)` before it considers an
    import, so a parent attribute surviving `evict()` short-circuits the whole
    point of evicting: the module body never re-executes and a decorator inside
    it never re-runs against the registry the caller just cleared.
    """
    _, alpha = _import("alpha")

    isolation = ModuleIsolation((f"{_PKG}.",))
    isolation.snapshot()
    isolation.evict()

    # The statement form is the assertion: `import_module` looks the name up in
    # `sys.modules` and would never touch the parent attribute under test. `ty`
    # cannot resolve a package this module writes to disk at runtime.
    from module_isolation_probe import (  # ty: ignore[unresolved-import]
        alpha as reimported,
    )

    assert reimported is not alpha, "stale parent attribute served a dropped module"
    assert sys.modules[f"{_PKG}.alpha"] is reimported


def test_restore_rebinds_the_snapshotted_parent_attribute():
    """After restore, parent attribute and cache agree on the original object."""
    pkg, alpha = _import("alpha")

    isolation = ModuleIsolation((f"{_PKG}.",))
    isolation.snapshot()
    isolation.evict()
    importlib.import_module(f"{_PKG}.alpha")  # a copy the fixture will discard
    isolation.restore()

    assert sys.modules[f"{_PKG}.alpha"] is alpha
    assert pkg.__dict__["alpha"] is alpha, "parent still bound to the discarded copy"


def test_restore_unbinds_modules_the_test_imported():
    """A module imported on top of the snapshot leaves no trace behind."""
    (pkg,) = _import()
    assert f"{_PKG}.beta" not in sys.modules  # deliberately outside the snapshot

    isolation = ModuleIsolation((f"{_PKG}.",))
    isolation.snapshot()
    importlib.import_module(f"{_PKG}.beta")
    isolation.restore()

    assert f"{_PKG}.beta" not in sys.modules
    assert "beta" not in pkg.__dict__, "extra module left bound on its parent"


def test_restore_leaves_an_unrelated_same_named_attribute_alone():
    """The unbind is identity-checked, so it cannot clobber a test's own stub."""
    (pkg,) = _import()
    isolation = ModuleIsolation((f"{_PKG}.",))
    isolation.snapshot()
    importlib.import_module(f"{_PKG}.beta")
    sentinel = object()
    pkg.__dict__["beta"] = sentinel  # e.g. a monkeypatched stand-in
    isolation.restore()

    assert pkg.__dict__["beta"] is sentinel


def test_lazy_export_cache_moves_with_the_modules():
    """A cached export must not outlive the module copy it was resolved from."""
    (pkg,) = _import()
    original = pkg.Thing  # resolved through the lazy __getattr__ and cached
    original_module = sys.modules[f"{_PKG}.thing"]

    isolation = ModuleIsolation((f"{_PKG}.",), lazy_packages=(_PKG,))
    isolation.snapshot()
    isolation.evict()
    assert "Thing" not in pkg.__dict__, "cached export survived eviction"

    assert pkg.Thing is not original  # re-resolved against the fresh copy
    isolation.restore()

    assert pkg.__dict__["Thing"] is original, "export outlived its module"
    assert sys.modules[f"{_PKG}.thing"] is original_module


def test_restore_tolerates_a_parent_that_is_gone():
    """Teardown must not raise when the parent package is no longer cached.

    `restore()` runs after the caller has already put its registries back, so a
    `KeyError` here would surface as a teardown error on an otherwise green test.
    """
    _, alpha = _import("alpha")

    isolation = ModuleIsolation((f"{_PKG}.",))
    isolation.snapshot()
    isolation.evict()
    del sys.modules[_PKG]

    isolation.restore()  # must not raise

    assert sys.modules[f"{_PKG}.alpha"] is alpha


def test_exclude_keeps_a_subtree_out_of_scope():
    """An excluded module is neither evicted nor unbound."""
    pkg, alpha, beta = _import("alpha", "beta")

    isolation = ModuleIsolation((f"{_PKG}.",), exclude=(f"{_PKG}.beta",))
    isolation.snapshot()
    isolation.evict()

    assert f"{_PKG}.alpha" not in sys.modules
    assert sys.modules[f"{_PKG}.beta"] is beta
    assert pkg.__dict__["beta"] is beta
