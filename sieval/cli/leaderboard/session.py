"""
Deprecated alias of :mod:`sieval.cli.session` (and the class resolvers of
:mod:`sieval.cli.resolution`), re-exporting the same objects. Delete after 0.11.

Other engine globals (e.g. ``DETERMINISTIC_DEFAULT_SEED``) are forwarded on
access. Assigning here does not reach the engine; patch :mod:`sieval.cli.session`.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import warnings

import sieval.cli.resolution as _resolution
import sieval.cli.session as _session
from sieval import __version__ as __version__
from sieval.cli.resolution import (
    resolve_class,
    resolve_dataset_class,
    resolve_task_class,
)
from sieval.cli.session import (
    EvalSession,
    ModelConfigDict,
    RootConfigDict,
    arun_session,
    resolve_deterministic,
    run_session,
    unwrap_proxies,
)

_DEPRECATION = (
    "sieval.cli.leaderboard.session is deprecated and will be removed after "
    "0.11; import from sieval.cli.session (resolvers: sieval.cli.resolution)."
)

warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)


def __getattr__(name: str):
    if name.startswith("__"):  # keep protocol probes (`__path__`, `__wrapped__`) local
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for home in (_session, _resolution):
        try:
            value = getattr(home, name)
        except AttributeError:
            continue
        warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
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
]
