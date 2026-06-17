"""Import-discipline test for the IFEval task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import subprocess
import sys


def test_import_does_not_pull_evaluation_lib():
    # evaluation_lib pulls absl/langdetect/nltk; registration must not import it.
    code = (
        "import sys\n"
        "import sieval.tasks.ifeval_0shot_gen\n"
        "assert 'sieval.community.instruction_following_eval.evaluation_lib' "
        "not in sys.modules, 'evaluation_lib must be lazy-imported'\n"
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
