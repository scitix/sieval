"""
Registration contract for the PlatinumBench MultiArith 0-shot task.

Behaviour is shared with the other four math subsets and tested in
``test__base.py``; this module pins what is specific to this leaf.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.tasks.platinum_bench.platinum_multiarith_0shot_gen import (
    PlatinumMultiArithZeroShotGenTask,
)

from .conftest import assert_leaf_meta


def test_meta():
    assert_leaf_meta(
        PlatinumMultiArithZeroShotGenTask,
        name="platinum_multiarith_0shot_gen",
        subset="multiarith",
        kept=170,
        total=174,
    )


def test_subset_uses_the_bare_config_name():
    # Upstream's own `check_prediction` list carries both the bare `multiarith`
    # and a legacy `math_eval__multiarith` spelling. The HF config name is the
    # bare one, and that is what selects the data.
    assert PlatinumMultiArithZeroShotGenTask.subset == "multiarith"
