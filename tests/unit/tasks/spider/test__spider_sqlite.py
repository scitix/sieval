"""Spider 1.0's execution bounds, pinned against what its gold actually needs.

The *guards* — read-only, no ATTACH, the progress-handler deadline, the text
factory — are one shared contract and are tested with it, in
`tests/unit/tasks/test__sqlite_exec.py`. What stays here is the pair of numbers
that contract has no opinion about: a deadline and a row cap are measured over a
corpus, and this is Spider 1.0's corpus.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.tasks.spider._spider_sqlite import DEFAULT_DEADLINE_S, DEFAULT_MAX_ROWS


def test_default_bounds_do_not_bind_on_a_realistic_result():
    """The bounds must sit above real gold results, not truncate them.

    Measured over every dev gold on both grading paths: 20,662 rows / 0.486 s
    against the shipped databases, and 92,450 rows / 0.359 s across the 40,167
    executions of the distilled test suites. This pins the constants so a later
    'tightening' has to argue with a failing test rather than silently rescoring
    the benchmark.
    """
    assert DEFAULT_MAX_ROWS > 92_450
    assert DEFAULT_DEADLINE_S > 0.486
