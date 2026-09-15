"""Run post-edit checks for a single edited file.

Claude Code, opencode, and Codex each fire a hook after a file edit. Keeping
the logic here rather than inlining it three times means a pattern change
cannot drift between them — and drift here is silent, since the failure mode
is a file class quietly no longer being checked.

Always exits 0: a post-edit check informs the agent, it does not block the
edit that already happened.

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A Python source or stub file.
PY_RE = re.compile(r"\.pyi?$")
# A file under sieval/tasks/ or sieval/datasets/, at any depth, matching both
# repo-relative and absolute paths as passed by the different harnesses.
REGISTRY_RE = re.compile(r"(^|/)sieval/(tasks|datasets)/")


def select_checks(path: str) -> set[str]:
    """Return the check names that apply to ``path``."""
    if not path:
        return set()

    checks: set[str] = set()
    is_py = bool(PY_RE.search(path))
    in_registry = bool(REGISTRY_RE.search(path))

    if is_py:
        checks.update({"ruff", "ty"})
    if is_py and in_registry:
        checks.add("stubs")
    if in_registry and path.endswith(".py"):
        checks.add("meta")
    return checks


def _run(cmd: list[str]) -> int:
    """Run a command from the repo root, streaming output. Never raises."""
    try:
        return subprocess.run(cmd, cwd=ROOT).returncode
    except OSError as exc:
        print(f"post-edit check could not run {cmd[0]}: {exc}", file=sys.stderr)
        return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return 0

    path = args[0]
    checks = select_checks(path)

    if "stubs" in checks:
        _run([sys.executable, str(ROOT / "scripts" / "sync_package_stubs.py")])
    if "meta" in checks:
        _run([sys.executable, str(ROOT / "scripts" / "sync_meta_index.py")])
    if "ruff" in checks:
        _run(["ruff", "check", path])
    if "ty" in checks:
        _run(["ty", "check", path])

    return 0


if __name__ == "__main__":
    sys.exit(main())
