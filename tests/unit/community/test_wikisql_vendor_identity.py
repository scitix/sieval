"""The vendored WikiSQL scorer must stay byte-identical to upstream.

Split out from ``test_wikisql.py`` so it runs **without the `wikisql` extra**.
That file has to import the package, which reaches ``babel`` through upstream's
``dbengine``, so it skips wherever the extra is absent — and CI installs eight
dependency groups, not including that one. These checks need no import at all:
they locate the package with ``find_spec`` (which resolves a path without
executing the module) and read the files off disk.

Keeping them runnable there is the point. ``Query.__eq__`` **is** the
``lf_accuracy`` metric, so a reformat of ``query.py`` is a silent scoring-layer
edit, and this is its only enforcer — `ruff`, `ty`, `mypy` and every behavioural
test are all indifferent to it. It has already fired once: `[tool.ruff]` excludes
``sieval/community``, but naming the path explicitly on a ``ruff format``
command overrides that and rewrote this file.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import hashlib
import importlib.util
import pathlib

#: Located, not imported -- `find_spec` resolves the package path without
#: executing its `__init__`, which is what keeps this module free of the
#: optional dependency.
_SPEC = importlib.util.find_spec("sieval.community.wikisql")
assert _SPEC is not None and _SPEC.submodule_search_locations is not None
_VENDOR_DIR = pathlib.Path(_SPEC.submodule_search_locations[0])

#: sha256 of upstream's `lib/query.py` at the pinned commit
#: (cffb423077756d04c1bac5bcd45167c86903fbcb).
_UPSTREAM_QUERY_SHA = "f539150bea6cd07a5dca226abcced6f9d356d216f5c3d70107693613f1fbeb25"
#: sha256 of upstream's `lib/common.py` at the same commit.
_UPSTREAM_COMMON_SHA = (
    "21079d6e99246eb9bfa8689b7548af9747f1b96c71e45be0522038a3486fde98"
)

_PATCHED_IMPORT = "from .common import detokenize"
_UPSTREAM_IMPORT = "from lib.common import detokenize"


def test_query_py_is_upstream_verbatim_but_for_the_import():
    """`query.py` must differ from upstream in exactly one line.

    Asserted by hash rather than by eye: reverting the one intended adaptation
    and re-hashing is the only check that notices a reformat, and `Query.__eq__`
    is the metric this repo publishes as `lf_accuracy`.
    """
    text = (_VENDOR_DIR / "query.py").read_text()
    assert text.count(_PATCHED_IMPORT) == 1
    assert _UPSTREAM_IMPORT not in text
    reverted = text.replace(_PATCHED_IMPORT, _UPSTREAM_IMPORT)
    assert hashlib.sha256(reverted.encode()).hexdigest() == _UPSTREAM_QUERY_SHA


def test_common_py_is_upstream_verbatim():
    text = (_VENDOR_DIR / "common.py").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == _UPSTREAM_COMMON_SHA


def test_identity_checks_ran_against_real_files():
    """Guards the guards: a wrong `_VENDOR_DIR` would make both tests vacuous.

    `find_spec` returning a stale or foreign path, or the files being renamed,
    would otherwise surface as a hash mismatch that reads like tampering.
    """
    assert (_VENDOR_DIR / "query.py").is_file()
    assert (_VENDOR_DIR / "common.py").is_file()
    assert _VENDOR_DIR.name == "wikisql"
    assert _VENDOR_DIR.parent.name == "community"
