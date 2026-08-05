"""
Registration contract for the PlatinumBench SVAMP 0-shot task.

Behaviour is shared with the other four math subsets and tested in
``test__base.py``; this module pins what is specific to this leaf.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks.meta import get_task_meta
from sieval.tasks.platinum_bench.platinum_svamp_0shot_gen import (
    PlatinumSVAMPZeroShotGenTask,
)

from .conftest import assert_leaf_meta


def test_meta():
    assert_leaf_meta(
        PlatinumSVAMPZeroShotGenTask,
        name="platinum_svamp_0shot_gen",
        subset="svamp",
        kept=265,
        total=300,
    )


def test_is_the_most_heavily_cleaned_math_subset():
    # 35 of 300 rejected — the highest of the five, which is consistent with
    # SVAMP being the subset built by perturbing problem surface forms. Pinned
    # because a silently-changed count is the failure mode that looks like a
    # score shift.
    reference_impl = get_task_meta(PlatinumSVAMPZeroShotGenTask).reference_impl
    assert reference_impl is not None
    assert "265 rows kept of 300 (35 rejected)" in reference_impl.notes
