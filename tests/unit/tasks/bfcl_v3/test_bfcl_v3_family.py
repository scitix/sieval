"""One contract over the four registered BFCL v3 leaves.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from types import SimpleNamespace

import pytest

from sieval.core.tasks.meta import get_task_class, get_task_meta
from sieval.tasks.bfcl_v3._base import BfclV3Task
from tests.conftest import MockChatModel, MockDataset

NAMES = (
    "bfcl_v3_non_live_0shot_gen",
    "bfcl_v3_non_live_0shot_gen_fc",
    "bfcl_v3_live_0shot_gen",
    "bfcl_v3_live_0shot_gen_fc",
)


@pytest.mark.parametrize("name", NAMES)
def test_every_leaf_resolves_by_name(name):
    """A nested task that imports but does not RESOLVE is the failure mode the
    subpackage rule exists for -- a bare KeyError from `sieval task show`."""
    assert get_task_class(name) is not None


@pytest.mark.parametrize("name", NAMES)
def test_every_leaf_declares_the_metadata_the_rules_require(name):
    meta = get_task_meta(get_task_class(name))
    assert meta.model_type == "chat"
    assert meta.reference_kind == "value"
    assert meta.n_shot == 0
    # `reference_impl` is optional on `TaskMeta`, so the narrowing is an
    # assertion in its own right: a leaf that shipped without one would
    # otherwise fail on an attribute error rather than on the thing under test.
    assert meta.reference_impl is not None
    assert meta.reference_impl.url.endswith(
        "ea13468e4423454d0c213704fb87cf7cb3990433/berkeley-function-call-leaderboard"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [(n, n.endswith("_fc")) for n in NAMES],
)
def test_only_the_fc_leaves_demand_the_tools_capability(name, expected):
    # Direct attribute access, not `getattr(..., False)`: `requires` is a
    # `ClassVar[TaskRequirements]` that every task inherits, so a default would
    # turn Task 1's field going missing into a silently passing assertion for
    # the two Prompt leaves.
    cls = get_task_class(name)
    assert cls.requires.function_tools is expected


class _ToollessChatModel(MockChatModel):
    """A chat model whose runtime plan offers no `function_tools`.

    Not reachable through a registered dialect today -- the only one that denies
    the capability is `openai_completions`, whose completion input kind trips a
    different check first -- so the runtime-plan branch is how the gate is
    exercised at all.
    """

    @property
    def runtime_plan(self):
        return SimpleNamespace(
            dialect_id="openai_chat",
            available_capabilities=frozenset(),
            capability_minimums={},
        )


def test_an_fc_leaf_is_refused_at_construction_when_tools_are_unavailable():
    """What the FC leaves' notes claim, asserted where it happens.

    The refusal is `Task.__init__`'s, so it lands at prelaunch reconciliation --
    before a request is made and before a result directory exists. Static schema
    validation does NOT see it: `validate_eval_config` runs without importing or
    instantiating a task, so a config naming an FC leaf passes it clean.
    """
    with pytest.raises(ValueError, match="function_tools"):
        get_task_class("bfcl_v3_live_0shot_gen_fc")(MockDataset(), _ToollessChatModel())


def test_the_prompt_leaves_construct_on_the_same_toolless_model():
    """The other half of the pair: without it, a gate that refused EVERY model
    would pass the test above just as happily."""
    get_task_class("bfcl_v3_live_0shot_gen")(MockDataset(), _ToollessChatModel())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfcl_v3_non_live_0shot_gen", "bfcl_v3_non_live"),
        ("bfcl_v3_non_live_0shot_gen_fc", "bfcl_v3_non_live"),
        ("bfcl_v3_live_0shot_gen", "bfcl_v3_live"),
        ("bfcl_v3_live_0shot_gen_fc", "bfcl_v3_live"),
    ],
)
def test_the_two_groups_bind_different_datasets(name, expected):
    """The FK is not declared -- `@sieval_task` resolves it by reverse lookup
    from the sample `TypedDict` the leaf binds, so this is what proves the two
    groups really did bind different sample types rather than both inheriting
    one."""
    assert get_task_meta(get_task_class(name)).dataset == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfcl_v3_non_live_0shot_gen", "non_live"),
        ("bfcl_v3_non_live_0shot_gen_fc", "non_live"),
        ("bfcl_v3_live_0shot_gen", "live"),
        ("bfcl_v3_live_0shot_gen_fc", "live"),
    ],
)
def test_each_leaf_rolls_up_under_its_own_group(name, expected):
    """Which group base a leaf inherits is a separate fact from which dataset it
    binds, and only this one is about the arithmetic.

    The two bases differ in how the headline is computed -- the non-live one
    takes an unweighted mean of five category rates and publishes no interval,
    the live one pools over rows and does -- and `GROUP_KEY` is what selects the
    rollup and names `score_key`. Nothing else here would notice a leaf wired to
    the wrong base while still binding the right sample type: the dataset FK
    comes from the generic arg, the capability from the protocol mixin, and the
    rollups are exercised against the bases rather than through the leaves. A
    live leaf on the non-live base would publish a mean of rates over live data
    under a live-sounding score key, and pass everything else.
    """
    cls = get_task_class(name)
    assert issubclass(cls, BfclV3Task)
    group_key = cls.GROUP_KEY
    assert group_key == expected
