"""
Deprecated alias of :mod:`sieval.cli.session` (and the class resolvers of
:mod:`sieval.cli.resolution`), re-exporting the same objects. Delete after 0.11.

``__all__`` lists the documented names. Any other attribute the engine module
holds, such as ``DETERMINISTIC_DEFAULT_SEED``, is forwarded on access, so code
that read a global of the 0.10 module keeps getting it. Assigning to this module
does not reach the engine: patch :mod:`sieval.cli.session` instead.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import warnings
from typing import Any

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


def __getattr__(name: str) -> Any:
    """Forward a name this alias does not hold to the engine module."""
    if name.startswith("__"):
        # Module protocol probes (`__path__`, `__wrapped__`) stay this module's.
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
