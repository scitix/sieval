"""Import-discipline test for the DROP k-shot task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import subprocess
import sys


def test_import_does_not_pull_drop_eval_backend():
    # drop_eval pulls scipy; importing the task for registration must not.
    code = (
        "import sys\n"
        "import sieval.tasks.drop_kshot_gen\n"
        "assert 'sieval.community.simple_evals.drop_eval' not in sys.modules, "
        "'drop_eval backend must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules
    # don't mask the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
