"""
Leaderboard YAML catalog — backs `sieval leaderboard list`.

Malformed files surface in the list with ``error`` set rather than
aborting the whole scan.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class LeaderboardSummary:
    """One row in the `sieval leaderboard list` output."""

    name: str
    path: Path
    # Count of ``tasks:`` keys, not ``datasets:`` — a sieval task is a
    # fixed (dataset, prompt, postprocess, metric) bundle, so one dataset
    # may back several tasks with different metrics.
    models: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    alignment_card: str | None = None
    # Set on parse failure, non-mapping root, or schema violation (e.g.
    # non-string ``alignment.card``). Rows with ``error`` render as
    # ``[malformed]`` in text mode.
    error: str | None = None


def scan_leaderboards(directory: Path) -> list[LeaderboardSummary]:
    """Non-recursive ``*.yaml`` / ``*.yml`` scan; missing dir returns ``[]``."""
    if not directory.is_dir():
        return []

    paths = sorted({*directory.glob("*.yaml"), *directory.glob("*.yml")})
    return [_summarize(p) for p in paths if p.is_file()]


def _require_mapping(data: dict, key: str, errors: list[str]) -> dict:
    """Return ``data[key]`` if a mapping; append an error and return ``{}`` otherwise.

    Missing key and explicit ``None`` are treated as absent (no error).
    """
    val = data.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        errors.append(f"{key} must be a mapping, got {type(val).__name__}")
        return {}
    return val


def _summarize(path: Path) -> LeaderboardSummary:
    name = path.stem
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return LeaderboardSummary(
            name=name, path=path, error=f"yaml parse error: {exc}"
        )
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError subclasses ValueError, not OSError — catch it
        # explicitly so a stray binary file in leaderboards/ surfaces as a
        # row error instead of aborting the whole scan.
        return LeaderboardSummary(name=name, path=path, error=f"read error: {exc}")

    if not isinstance(data, dict):
        return LeaderboardSummary(name=name, path=path, error="root is not a mapping")

    errors: list[str] = []
    models_block = _require_mapping(data, "models", errors)
    tasks_block = _require_mapping(data, "tasks", errors)
    alignment_block = _require_mapping(data, "alignment", errors)

    card = alignment_block.get("card")
    if card is not None and not isinstance(card, str):
        errors.append(f"alignment.card must be a string, got {type(card).__name__}")
        card = None

    return LeaderboardSummary(
        name=name,
        path=path,
        models=sorted(models_block.keys()),
        tasks=sorted(tasks_block.keys()),
        alignment_card=card,
        error="; ".join(errors) if errors else None,
    )
