"""Unit tests for the shared MultiPL-E dataset loader.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.datasets._multipl_e import (
    HUMANEVAL_LANGUAGES,
    MBPP_LANGUAGES,
    normalize_languages,
)


def _he(languages=None, config=None):
    return normalize_languages(
        languages, config, suite="humaneval", available=HUMANEVAL_LANGUAGES
    )


def _mbpp(languages=None, config=None):
    return normalize_languages(
        languages, config, suite="mbpp", available=MBPP_LANGUAGES
    )


def test_suites_differ_only_by_dart():
    # Upstream translated HumanEval to Dart and MBPP not. Asserted because the
    # two tuples are written out separately, and a copy-paste that unified them
    # would make `mbpp-dart` look loadable.
    assert set(HUMANEVAL_LANGUAGES) - set(MBPP_LANGUAGES) == {"dart"}
    assert set(MBPP_LANGUAGES) - set(HUMANEVAL_LANGUAGES) == set()
    assert len(HUMANEVAL_LANGUAGES) == 24
    assert len(MBPP_LANGUAGES) == 23


def test_tags_are_upstream_registry_names_not_english():
    # The tags ARE the HuggingFace config names, so an English name is an
    # unresolvable config rather than a language that scores badly.
    for tag in ("jl", "ml", "rkt", "adb", "rb", "hs", "cs", "sh", "pl"):
        assert tag in HUMANEVAL_LANGUAGES
    for english in ("julia", "ocaml", "racket", "ada", "ruby", "haskell", "bash"):
        assert english not in HUMANEVAL_LANGUAGES


def test_no_python_config():
    # MultiPL-E translates OUT of Python, so a `py` config would be plain
    # HumanEval -- which this repo already has as its own task.
    assert "py" not in HUMANEVAL_LANGUAGES
    assert "python" not in HUMANEVAL_LANGUAGES
    assert "py" not in MBPP_LANGUAGES


def test_default_selects_every_language():
    assert _he() == HUMANEVAL_LANGUAGES
    assert _mbpp() == MBPP_LANGUAGES
    # `config="all"` is MMMLU's spelling for the same thing.
    assert _he(config="all") == HUMANEVAL_LANGUAGES


def test_an_empty_language_list_is_refused_rather_than_read_as_all():
    """Omitting the argument means "all"; an empty list means a config bug.

    The two readings are 24 languages of inference apart, so the silent one is
    the expensive one — and a caller that computed a selection and came up with
    nothing is far likelier than one who spelled "everything" as `[]`.
    """
    for select in (_he, _mbpp):
        with pytest.raises(ValueError, match="empty list"):
            select([])


def test_bare_tags_and_full_config_names_both_resolve():
    assert _he(["cpp", "js"]) == ("cpp", "js")
    assert _he(config="humaneval-rs") == ("rs",)
    assert _mbpp(config="mbpp-cpp") == ("cpp",)


def test_order_follows_upstream_not_the_caller():
    # Row order decides sample ids, so a resume needs the same order from the
    # same set however the caller happened to spell it.
    assert (
        _he(["ts", "cpp", "adb"])
        == _he(["adb", "ts", "cpp"])
        == (
            "adb",
            "cpp",
            "ts",
        )
    )


def test_duplicates_collapse():
    assert _he(["cpp", "cpp", "js"]) == ("cpp", "js")


def test_english_language_name_is_rejected():
    with pytest.raises(ValueError, match="julia"):
        _he(["julia"])


def test_cross_suite_config_is_rejected():
    # `mbpp-cpp` asked of HumanEval would load rows the registered task is not
    # bound to, so it is refused rather than silently re-pointed at `cpp`.
    with pytest.raises(ValueError, match="mbpp-cpp"):
        _he(["mbpp-cpp"])
    with pytest.raises(ValueError, match="humaneval-cpp"):
        _mbpp(["humaneval-cpp"])


def test_dart_is_rejected_for_mbpp_only():
    assert _he(["dart"]) == ("dart",)
    with pytest.raises(ValueError, match="dart"):
        _mbpp(["dart"])


def test_error_names_the_expected_vocabulary():
    # The message has to teach the tag/English-name distinction, since that is
    # the mistake it exists to catch.
    with pytest.raises(ValueError) as excinfo:
        _he(["racket"])
    message = str(excinfo.value)
    assert "registry tags" in message
    assert "rkt" in message
