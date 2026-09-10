"""The timeout predicate, and the false positives a substring test let through.

Each `not_a_timeout` case below contains the word "timeout" somewhere, so the
whole file passes trivially against `"timeout" in msg.lower()` -- which is what
these assertions exist to reject.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.tasks._code_eval_msg import CODE_EVAL_TIMEOUT_PREFIXES, is_timeout_message

# Every timeout spelling the code-eval service emits, verbatim from
# `vendor/code-evaluator/app/exec_*.py`.
TIMEOUTS = [
    "failed: subprocess timeout: 3.0s",  # exec_py_code / exec_py_test wall
    "failed: case timeout: 6.0s",  # exec_py_test, per case
    "failed: compile timeout: 6.0s",  # exec_py_test, per-case compile
    "failed: build timeout",  # exec_lang, compile wall
    "failed: timeout",  # exec_js / exec_ts wall
]

# Messages that are NOT the service stopping a clock, but DO contain the word --
# so a substring test counts every one of them as a timeout.
NOT_TIMEOUTS = [
    # The program raised; the class name lands in the interpolated tail.
    "failed: [TimeoutError] deadline exceeded",
    "failed: [ValueError] timeout must be positive",
    # LiveCodeBench prints the program's own stdout next to the expectation.
    "failed: output ['timeout'] != expect ['ok']",
    "failed: output mismatch: got 'timeout' expected 'done'",
    # A c++ diagnostic quoting an identifier (exec_lang / MultiPL-E).
    "failed [build exit 1]: error: no member named 'timeout' in 'Config'",
    # A non-zero exit whose stderr mentions it.
    "failed [exit 1]: Traceback ...\nTimeoutError: x",
]


@pytest.mark.parametrize("msg", TIMEOUTS)
def test_service_timeouts_are_recognized(msg):
    assert is_timeout_message(msg) is True


@pytest.mark.parametrize("msg", NOT_TIMEOUTS)
def test_the_word_elsewhere_in_a_message_is_not_a_timeout(msg):
    assert "timeout" in msg.lower(), "case must be a substring-test false positive"
    assert is_timeout_message(msg) is False


@pytest.mark.parametrize("msg", ["", None, "failed: compile error", "ok"])
def test_absent_and_unrelated_messages(msg):
    assert is_timeout_message(msg) is False


def test_case_and_surrounding_whitespace_do_not_matter():
    assert is_timeout_message("  FAILED: Subprocess Timeout: 3.0s  ") is True


def test_every_prefix_is_itself_recognized():
    """No prefix is shadowed by another, and each one is reachable."""
    for prefix in CODE_EVAL_TIMEOUT_PREFIXES:
        assert is_timeout_message(prefix) is True
