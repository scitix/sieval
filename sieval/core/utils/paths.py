"""On-disk path resolution for sieval-managed data.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import os
from pathlib import Path


def resolve_data_dir(cli_flag: str | None) -> Path:
    """Resolve download root: CLI flag > ``SIEVAL_DATA_DIR`` env > ``~/.sieval/data``.

    ``--data-dir`` is per-invocation. Cross-command consistency (``sieval
    dataset download`` ↔ ``sieval eval``) relies on ``SIEVAL_DATA_DIR`` in
    the environment — a one-off ``--data-dir /custom`` on ``download`` writes
    to ``/custom`` but the eval path still reads from the env-var root.
    """
    if cli_flag is not None:
        return Path(cli_flag).expanduser()
    env = os.environ.get("SIEVAL_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".sieval" / "data"
