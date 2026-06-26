"""Import-discipline test for the HMMT Feb 2026 task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import subprocess
import sys


def test_import_does_not_pull_math_verify():
    code = (
        "import sys\n"
        "import sieval.tasks.hmmt_feb_2026_0shot_gen\n"
        "assert 'math_verify' not in sys.modules, "
        "'math_verify must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules don't mask the
    # check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
