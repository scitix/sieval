"""Tests for sieval/datasets/ruler/_cwe.py — CWE example synthesis."""

import pytest

try:
    import tiktoken as _tiktoken  # noqa: F401

    _ruler_deps = True
except ImportError:
    _ruler_deps = False

_needs_ruler_deps = pytest.mark.skipif(
    not _ruler_deps, reason="ruler deps group not installed"
)

if _ruler_deps:
    from sieval.datasets.ruler._cwe import _get_example


@_needs_ruler_deps
def test_get_example_missing_fallback_pool_raises_clear_error():
    """num_words beyond the small `words` pool needs the english_words.json
    fallback (randle_words); when it's missing/empty this must raise a clear
    error naming the file and the download command, not the bare
    ValueError('Sample larger than population...') from random.sample."""
    with pytest.raises(FileNotFoundError, match="english_words.json"):
        _get_example(
            num_words=10,
            words=["a", "b", "c"],
            randle_words=[],
            common_repeats=1,
            uncommon_repeats=1,
            common_nums=1,
            random_seed=42,
        )


@_needs_ruler_deps
def test_get_example_uses_fallback_pool_when_available():
    """Same oversized-num_words case succeeds once randle_words is populated."""
    context, common = _get_example(
        num_words=5,
        words=["a", "b", "c"],
        randle_words=[f"word{i}" for i in range(10)],
        common_repeats=1,
        uncommon_repeats=1,
        common_nums=2,
        random_seed=42,
    )
    assert context
    assert len(common) == 2
