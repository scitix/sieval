"""The timeout predicate: the false positives a substring test lets through, and
the one true timeout a prefix set is easy to leave out.

Every `NOT_TIMEOUTS` case contains the word, so this file passes trivially
against `"timeout" in msg.lower()` -- rejecting them is what the assertions are
for. `failed: [CaseTimeout]` is the mirror case: a real wall that arrives
formatted as a class name, which the substring test caught by accident.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.tasks._code_eval_msg import (
    CODE_EVAL_TIMEOUT_PREFIXES,
    exception_class_name,
    is_timeout_message,
)

# Every timeout spelling the code-eval service emits, verbatim from
# `vendor/code-evaluator/app/exec_*.py`.
TIMEOUTS = [
    "failed: subprocess timeout: 3.0s",  # exec_py_code / exec_py_test wall
    "failed: case timeout: 6.0s",  # exec_py_test, per case
    "failed: compile timeout: 6.0s",  # exec_py_test, per-case compile
    # The per-case wall landing past `_unsafe_execute`'s own handler, so the
    # worker's outer `except (Exception, CaseTimeout)` formats it as a class
    # name. The trailing space is the empty `{e}`; the message is verbatim.
    "failed: [CaseTimeout] ",
    "failed: build timeout",  # exec_lang compile wall (MultiPL-E)
    "failed: timeout",  # exec_js / exec_ts / exec_lang run wall
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


def test_a_late_case_timeout_is_not_confused_with_a_raised_one():
    """The two class-name shapes differ only by which class, and split opposite ways.

    `CaseTimeout` is service-internal, so submitted code cannot produce the name;
    `TimeoutError` is the program's. Getting this pair backwards is the whole
    reason the predicate is not a substring test in either direction.
    """
    assert is_timeout_message("failed: [CaseTimeout] ") is True
    assert is_timeout_message("failed: [TimeoutError] deadline exceeded") is False


@pytest.mark.parametrize("msg", ["", None, "failed: compile error", "ok"])
def test_absent_and_unrelated_messages(msg):
    assert is_timeout_message(msg) is False


def test_case_and_surrounding_whitespace_do_not_matter():
    assert is_timeout_message("  FAILED: Subprocess Timeout: 3.0s  ") is True


def test_every_prefix_is_itself_recognized():
    """No prefix is shadowed by another, and each one is reachable."""
    for prefix in CODE_EVAL_TIMEOUT_PREFIXES:
        assert is_timeout_message(prefix) is True


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ("failed: [MemoryError] unable to allocate 8 GiB", "memoryerror"),
        (
            "failed: [ModuleNotFoundError] No module named 'numba'",
            "modulenotfounderror",
        ),
        # A subclass: the service names the concrete class, not the builtin.
        ("failed: [ZipImportError] bad local file header", "zipimporterror"),
        ("  FAILED: [AssertionError] x  ", "assertionerror"),
        ("failed: [CaseTimeout] ", "casetimeout"),
        # Not the exception shape at all.
        ("failed: subprocess timeout: 3.0s", None),
        ("failed: compile error", None),
        # The word in the TAIL, which is the program's own output.
        ("failed: output ['memoryerror'] != expect ['ok']", None),
        # Malformed: opened but never closed.
        ("failed: [MemoryError unable to allocate", None),
        ("", None),
        (None, None),
    ],
)
def test_exception_class_name_reads_only_the_slot(msg, expected):
    assert exception_class_name(msg) == expected


def test_class_families_are_matched_not_just_the_builtin():
    """The counters match a family, so a subclass the service names still counts.

    A bare `"importerror" in msg` caught `ZipImportError` by accident; testing
    the slot for equality with the builtin would silently drop it, turning a
    precision fix into a completeness regression.
    """
    zip_err = exception_class_name("failed: [ZipImportError] bad local file header")
    assert zip_err is not None and zip_err.endswith("importerror")

    oom = exception_class_name("failed: [OutOfMemoryError] cuda ran out")
    assert oom is not None and oom.endswith("memoryerror")

    # ...while the tail no longer reaches either counter.
    tail = exception_class_name("failed: [ValueError] simulated memoryerror path")
    assert tail == "valueerror"
    assert not tail.endswith("memoryerror")
