"""
Deprecated alias of :mod:`sieval.cli.session` (and the class resolvers of
:mod:`sieval.cli.resolution`), re-exporting the same objects. Delete after 0.11.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import warnings

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

warnings.warn(
    "sieval.cli.leaderboard.session is deprecated and will be removed after "
    "0.11; import from sieval.cli.session (resolvers: sieval.cli.resolution).",
    DeprecationWarning,
    stacklevel=2,
)

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
