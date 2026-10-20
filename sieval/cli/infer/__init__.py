"""
CLI subpackage for inference service management.

Re-exports live consumers of submodule symbols:
  - ``infer_app`` — used by ``sieval/cli/main.py``
  - ``HandleHeldError`` / ``cleanup_model`` / ``launch_model`` /
    ``resolve_infer_config`` / ``remove_run_handle_dir`` /
    ``resolve_keep_logs`` / ``run_handle_dir`` —
    used by ``sieval/cli/run.py``

Everything else goes direct to the owning submodule (``.lifecycle``,
``.recipe``, ``.commands``) — imports imply public API, so we don't
broadcast names no caller is actually pulling.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.cli.infer.commands import infer_app
from sieval.cli.infer.lifecycle import (
    HandleHeldError,
    cleanup_model,
    launch_model,
    remove_run_handle_dir,
    resolve_keep_logs,
    run_handle_dir,
)
from sieval.cli.infer.recipe import resolve_infer_config

__all__ = [
    "HandleHeldError",
    "cleanup_model",
    "infer_app",
    "launch_model",
    "remove_run_handle_dir",
    "resolve_infer_config",
    "resolve_keep_logs",
    "run_handle_dir",
]
