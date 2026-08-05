"""
Registration contract for the PlatinumBench SingleOp 0-shot task.

Behaviour is shared with the other four math subsets and tested in
``test__base.py``; this module pins what is specific to this leaf.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.tasks.platinum_bench.platinum_singleop_0shot_gen import (
    PlatinumSingleOpZeroShotGenTask,
)

from .conftest import assert_leaf_meta


def test_meta():
    assert_leaf_meta(
        PlatinumSingleOpZeroShotGenTask,
        name="platinum_singleop_0shot_gen",
        subset="singleop",
        kept=150,
        total=159,
    )
