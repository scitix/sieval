"""
CLI subpackage for leaderboard display and cross-run aggregation.

Re-exports the public API:
  - ``from sieval.cli.leaderboard import leaderboard_app``
  - ``from sieval.cli.leaderboard import scan_runs, build_matrix, ...``
  - ``from sieval.cli.leaderboard import AlignmentCard, load_card``

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.cli.leaderboard.card import AlignmentCard, load_card
from sieval.cli.leaderboard.catalog import LeaderboardSummary, scan_leaderboards
from sieval.cli.leaderboard.commands import leaderboard_app
from sieval.cli.leaderboard.scanner import (
    LeaderboardMatrix,
    LeaderboardResult,
    RunInfo,
    build_matrix,
    resolve_model_name,
    scan_runs,
)

__all__ = [
    "AlignmentCard",
    "LeaderboardMatrix",
    "LeaderboardResult",
    "LeaderboardSummary",
    "RunInfo",
    "build_matrix",
    "leaderboard_app",
    "load_card",
    "resolve_model_name",
    "scan_leaderboards",
    "scan_runs",
]
