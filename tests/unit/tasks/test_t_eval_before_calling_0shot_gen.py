"""Import-discipline test for the t_eval before-calling task.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import subprocess
import sys


def test_import_does_not_pull_sentence_transformers():
    # Importing the task module (for registration) must NOT eagerly import the
    # heavy embedding backend — it is only needed when eval_thought is enabled.
    code = (
        "import sys\n"
        "import sieval.tasks.t_eval_before_calling_0shot_gen\n"
        "assert 'sentence_transformers' not in sys.modules, "
        "'sentence_transformers must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules don't mask
    # the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
