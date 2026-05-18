"""Verify 'sieval eval' shortcut stays in lockstep with 'sieval leaderboard run'.

The top-level `sieval eval` command is intentionally registered as a thin
alias of `sieval leaderboard run`: both Typer commands are bound to the same
Python callable, so options, positional arguments, and defaults cannot drift
between them. These tests lock that invariant in three layers:

1. Identity — both Typer ``CommandInfo.callback`` entries are the same object.
2. Signature — ``inspect.signature`` agrees (trivially true given (1), but a
   useful explicit check if (1) is ever relaxed).
3. Help surface — the rendered ``--help`` option names match, catching any
   future divergence in user-visible flag text even if the callable is cloned.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import inspect

from typer.testing import CliRunner

from sieval.cli import app
from sieval.cli.leaderboard import leaderboard_app

runner = CliRunner()


def _find_command(typer_app, name: str):
    for cmd in typer_app.registered_commands:
        if cmd.name == name:
            return cmd
    raise AssertionError(f"Command {name!r} not registered on {typer_app!r}")


def test_eval_shortcut_is_same_callable_as_leaderboard_run():
    """The shortcut must be bound to the *same* callable — not a wrapper."""
    eval_cmd = _find_command(app, "eval")
    lb_run_cmd = _find_command(leaderboard_app, "run")
    assert eval_cmd.callback is lb_run_cmd.callback, (
        "'sieval eval' and 'sieval leaderboard run' must share one callable "
        "object so options/args cannot drift; got distinct callables "
        f"{eval_cmd.callback!r} vs {lb_run_cmd.callback!r}."
    )


def test_eval_shortcut_signature_matches_leaderboard_run():
    """Defensive check — signatures must agree even if identity is ever relaxed."""
    eval_cmd = _find_command(app, "eval")
    lb_run_cmd = _find_command(leaderboard_app, "run")
    assert inspect.signature(eval_cmd.callback) == inspect.signature(
        lb_run_cmd.callback
    )


def _extract_option_names(help_text: str) -> set[str]:
    """Return the set of '--flag' tokens from a --help dump."""
    tokens: set[str] = set()
    for line in help_text.splitlines():
        for word in line.strip().split():
            if word.startswith("--") and len(word) > 2:
                tokens.add(word.rstrip(",.:;"))
    return tokens


def test_eval_shortcut_matches_leaderboard_run_options():
    """User-visible --help option surface must match."""
    eval_result = runner.invoke(app, ["eval", "--help"])
    lb_result = runner.invoke(app, ["leaderboard", "run", "--help"])
    assert eval_result.exit_code == 0
    assert lb_result.exit_code == 0

    eval_opts = _extract_option_names(eval_result.stdout)
    lb_opts = _extract_option_names(lb_result.stdout)

    # Ignore typer-injected meta flags (--help, --install-completion, etc.)
    meta = {"--help", "--install-completion", "--show-completion"}
    assert (eval_opts - meta) == (lb_opts - meta), (
        f"Shortcut drifted from resource verb.\n"
        f"Only in eval: {eval_opts - lb_opts}\n"
        f"Only in leaderboard run: {lb_opts - eval_opts}"
    )
