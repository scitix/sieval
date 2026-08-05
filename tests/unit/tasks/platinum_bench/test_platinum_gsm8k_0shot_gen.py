"""
Registration contract for the PlatinumBench GSM8K 0-shot task.

Behaviour is shared with the other four math subsets and tested in
``test__base.py``; this module pins what is specific to this leaf.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks.meta import get_task_meta
from sieval.tasks.gsm8k_0shot_gen import GSM8KZeroShotGenTask
from sieval.tasks.platinum_bench.platinum_gsm8k_0shot_gen import (
    PlatinumGSM8KZeroShotGenTask,
)

from .conftest import assert_leaf_meta


def test_meta():
    assert_leaf_meta(
        PlatinumGSM8KZeroShotGenTask,
        name="platinum_gsm8k_0shot_gen",
        subset="gsm8k",
        kept=268,
        total=300,
    )


def test_is_not_the_existing_gsm8k_task_in_disguise():
    # `gsm8k_0shot_gen` runs the full uncleaned split through the DeepSeek-Math
    # protocol: different questions, prompt and extractor. The two names must
    # never be conflated on a leaderboard, and the dataset FK is what keeps them
    # apart mechanically.
    platinum = get_task_meta(PlatinumGSM8KZeroShotGenTask)
    flat = get_task_meta(GSM8KZeroShotGenTask)
    assert flat.name == "gsm8k_0shot_gen"
    assert flat.dataset == "gsm8k"
    assert platinum.name == "platinum_gsm8k_0shot_gen"
    assert platinum.dataset == "platinum_bench"
