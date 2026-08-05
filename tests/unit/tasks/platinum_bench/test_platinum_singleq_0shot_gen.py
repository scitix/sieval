"""
Registration contract for the PlatinumBench SingleEq 0-shot task.

Behaviour is shared with the other four math subsets and tested in
``test__base.py``; this module pins what is specific to this leaf.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.datasets.platinum_bench import PLATINUM_SUBSETS
from sieval.tasks.platinum_bench.platinum_singleq_0shot_gen import (
    PlatinumSingleEqZeroShotGenTask,
)

from .conftest import assert_leaf_meta


def test_meta():
    assert_leaf_meta(
        PlatinumSingleEqZeroShotGenTask,
        name="platinum_singleq_0shot_gen",
        subset="singleq",
        kept=100,
        total=109,
    )


def test_subset_uses_the_config_spelling_not_the_upstream_comment_spelling():
    # Upstream's own docstring calls this benchmark "singleeq", but the HF config
    # — and upstream's `check_prediction` list — spell it "singleq". The config
    # name is what selects the data, so the "corrected" spelling would 404.
    assert PlatinumSingleEqZeroShotGenTask.subset == "singleq"
    assert "singleeq" not in PLATINUM_SUBSETS
