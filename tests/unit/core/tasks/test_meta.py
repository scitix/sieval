"""Tests for sieval.core.tasks.meta — task metadata value types and registry.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from typing import TypedDict

import pytest

from sieval.core.datasets.meta import (
    _VALID_LEVEL2,
    DATASET_REGISTRY,
    SAMPLE_TO_DATASET,
    Category,
    DatasetMeta,
    Level1Category,
)
from sieval.core.tasks import Task
from sieval.core.tasks.meta import (
    TASK_REGISTRY,
    EvalMode,
    ReferenceImpl,
    TaskMeta,
    get_task_meta,
    get_task_run_identity,
    iter_task_metas,
    sieval_task,
    task_meta_from_dict,
    task_meta_to_dict,
)
from tests.conftest import ModuleIsolation

# ── Stub sample type + dataset class for reverse-lookup tests ──


class _StubSample(TypedDict):
    x: int


class _StubDataset:
    """Minimal stand-in registered via SAMPLE_TO_DATASET for unit tests."""

    _sieval_dataset_meta = DatasetMeta(
        name="stub_dataset",
        display_name="Stub",
        description="stub",
        # Wire-format normalization: DatasetMeta.source is tuple[str, ...]
        # always (the decorator accepts a bare string and normalizes, but
        # direct construction like this stub must pass the tuple form).
        source=("hf:stub/stub",),
        categories=(Category(Level1Category.MATHEMATICS),),
    )


class _StubTask(Task[_StubSample, None, None, None, None, dict[str, float]]):
    """Bare Task subclass used as a @sieval_task decoration target.

    Inheriting from Task satisfies the decorator's `type[Task]` bound and
    supplies the `tags` / `model_type` ClassVar declarations ty needs to
    resolve attribute accesses after decoration. The stage methods are inert
    stand-ins rather than left abstract: run identity is projected off an
    *instance*, so `_identity_of` needs a class `object.__new__` will accept.
    """

    async def preprocess(self, *args, **kwargs): ...
    async def infer(self, *args, **kwargs): ...
    async def postprocess(self, *args, **kwargs): ...
    async def feedback(self, *args, **kwargs): ...
    async def report(self, *args, **kwargs): ...


def _identity_of(cls: type[Task]):
    """Project run identity for a decoration-target stub.

    `get_task_run_identity` takes an instance, because `n_shot` is read off
    the task rather than the class. These stubs hold no `__init__` state it
    touches, so `object.__new__` stands in for a constructed task and keeps a
    pure projection test free of a dataset/model pair.
    """
    return get_task_run_identity(object.__new__(cls))


@pytest.fixture(autouse=True)
def _stub_sample_mapping():
    """Ensure _StubSample → _StubDataset is registered for every test."""
    SAMPLE_TO_DATASET[_StubSample] = _StubDataset
    DATASET_REGISTRY["stub_dataset"] = _StubDataset._sieval_dataset_meta
    yield
    SAMPLE_TO_DATASET.pop(_StubSample, None)
    DATASET_REGISTRY.pop("stub_dataset", None)


def test_level1_category_is_str_enum():
    assert Level1Category.MATHEMATICS.value == "Mathematics"
    assert Level1Category.MATHEMATICS == "Mathematics"


def test_level1_category_has_expected_members():
    expected = {
        "Language",
        "Knowledge",
        "Logic",
        "Mathematics",
        "Code",
        "Agent",
    }
    assert {cn.value for cn in Level1Category} == expected


def test_valid_level2_covers_all_level1_categories():
    """Every Level1Category has an entry in the closed-vocabulary map."""
    assert set(_VALID_LEVEL2.keys()) == set(Level1Category)


def test_category_frozen():
    cat = Category(Level1Category.MATHEMATICS, "CompetitionMath")
    with pytest.raises((AttributeError, TypeError)):
        cat.level1 = Level1Category.CODE  # type: ignore[misc]


def test_category_default_level2_is_none():
    cat = Category(Level1Category.LOGIC)
    assert cat.level2 is None


def test_category_accepts_positional_and_keyword_args():
    pos = Category(Level1Category.CODE, "CodeGeneration")
    kw = Category(level1=Level1Category.CODE, level2="CodeGeneration")
    assert pos == kw


def test_eval_mode_is_str_enum():
    assert EvalMode.GEN.value == "gen"


def test_eval_mode_has_expected_members():
    assert {m.value for m in EvalMode} == {"gen", "ppl", "clp"}


def test_eval_mode_clp_value_and_roundtrip():
    assert EvalMode.CLP.value == "clp"
    assert EvalMode("clp") is EvalMode.CLP


def test_reference_impl_requires_source_and_url():
    ri = ReferenceImpl(
        source="simple-evals", url="https://github.com/x/y/blob/abc/f.py"
    )
    assert ri.notes == ""


def test_task_meta_constructs():
    meta = TaskMeta(
        name="dummy",
        display_name="Dummy",
        description="A dummy task",
        dataset="stub_dataset",
        eval_mode=EvalMode.GEN,
    )
    assert meta.name == "dummy"
    assert meta.dataset == "stub_dataset"
    assert meta.n_shot == 0
    assert meta.tags == ()
    assert meta.deps_group is None
    assert meta.status == "stable"
    assert meta.reference_impl is None


@pytest.fixture(autouse=True)
def _clean_registry():
    """Give every test an empty task registry *and* an empty ``sieval.tasks``
    module cache — see ``ModuleIsolation`` for why the two must move together.

    Only submodules are in scope; ``sieval.tasks`` itself stays cached so its
    ``__path__`` keeps the identity that path-injecting tests rely on. Its lazy
    export cache is still declared, because staying cached is exactly what lets
    it hold a class resolved from a copy that ``restore()`` later discards. The
    dataset registries are deliberately untouched — ``_stub_sample_mapping``
    above seeds ``SAMPLE_TO_DATASET`` for every test in this file.
    """
    from sieval.core.tasks.meta import _TASK_CLASSES

    reg_snapshot = dict(TASK_REGISTRY)
    cls_snapshot = dict(_TASK_CLASSES)
    modules = ModuleIsolation(("sieval.tasks.",), lazy_packages=("sieval.tasks",))
    modules.snapshot()

    TASK_REGISTRY.clear()
    _TASK_CLASSES.clear()
    modules.evict()
    try:
        yield
    finally:
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(reg_snapshot)
        _TASK_CLASSES.clear()
        _TASK_CLASSES.update(cls_snapshot)
        modules.restore()


def test_clean_registry_fixture_leaves_task_modules_reimportable():
    """Lock the `_clean_registry` contract that the rest of this file rests on.

    Registry and `sieval.tasks` module cache must both start empty, so
    `import_all_tasks()` genuinely re-runs every `@sieval_task` decorator. When
    the cache was left populated, tasks another test file had already imported
    at module level came back unregistered — silently, and only for those tasks.

    The cache half of that only has teeth once something has populated it, so
    running this file alone leaves the second assertion trivially true. It earns
    its keep in a full-suite or multi-file run, where an earlier file has already
    imported a task module.

    Also pins the eponymous-*filename* convention `get_task_class()` depends on
    (`sieval/core/tasks/meta.py`): every registered name must be the last segment
    of some loaded `sieval.tasks.*` module, or that lookup cannot resolve its
    class. Depth is deliberately unconstrained — flat and subpackage-hosted both
    satisfy it. What stays banned is a task whose defining file is named
    something other than the task, which no amount of walking can find.
    """
    import sys

    from sieval.core.tasks.meta import import_all_tasks
    from sieval.meta import load_index

    assert not TASK_REGISTRY
    assert not [name for name in sys.modules if name.startswith("sieval.tasks.")]

    import_all_tasks()

    _, tasks = load_index()
    assert tasks, "shipped index is non-empty"
    missing = {t.name for t in tasks} - set(TASK_REGISTRY)
    assert not missing, f"indexed tasks left unregistered: {sorted(missing)}"
    loaded_stems = {
        module.rpartition(".")[2]
        for module in sys.modules
        if module.startswith("sieval.tasks.")
    }
    off_convention = sorted(set(TASK_REGISTRY) - loaded_stems)
    assert not off_convention, (
        f"tasks not defined in an eponymous module: {off_convention}"
    )


def test_sieval_task_attaches_meta_and_registers():
    @sieval_task(
        name="pilot",
        display_name="Pilot",
        description="pilot task",
        eval_mode=EvalMode.GEN,
    )
    class PilotTask(_StubTask):
        pass

    meta = get_task_meta(PilotTask)
    assert meta.name == "pilot"
    assert meta.dataset == "stub_dataset"
    assert TASK_REGISTRY["pilot"] is meta
    assert PilotTask._sieval_task_meta is meta  # type: ignore[attr-defined]


def test_sieval_task_rejects_duplicate_name():
    @sieval_task(
        name="dup",
        display_name="A",
        description="x",
        eval_mode=EvalMode.GEN,
    )
    class A(_StubTask):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @sieval_task(
            name="dup",
            display_name="B",
            description="y",
            eval_mode=EvalMode.GEN,
        )
        class B(_StubTask):
            pass


def test_iter_task_metas_returns_registered():
    @sieval_task(
        name="t1",
        display_name="T1",
        description="x",
        eval_mode=EvalMode.GEN,
    )
    class T1(_StubTask):
        pass

    names = [m.name for m in iter_task_metas()]
    assert names == ["t1"]


def test_get_task_meta_raises_for_unregistered():
    class Plain:
        pass

    with pytest.raises(AttributeError):
        get_task_meta(Plain)


# ===================================================================
# get_task_run_identity — the subset persisted into run meta.json
# ===================================================================
class TestGetTaskRunIdentity:
    def test_projects_the_declared_subset(self):
        @sieval_task(
            name="ident",
            display_name="Ident",
            description="identity task",
            eval_mode=EvalMode.CLP,
            n_shot=5,
            tags=("chinese", "multiple-choice"),
            deps_group="extra",
            model_type="gen",
            reference_impl=ReferenceImpl(source="upstream", url="https://x.invalid"),
            status="experimental",
        )
        class IdentTask(_StubTask):
            pass

        assert _identity_of(IdentTask) == {
            "name": "ident",
            "display_name": "Ident",
            "dataset": "stub_dataset",
            "eval_mode": "clp",
            "n_shot": 5,
            "tags": ["chinese", "multiple-choice"],
            "status": "experimental",
        }

    def test_omits_fields_excluded_on_purpose(self):
        """`description` / `deps_group` / `model_type` / `reference_impl` are
        deliberately not persisted — assert their absence so a well-meaning
        "complete the subset" change has to delete a test to land."""

        @sieval_task(
            name="excluded",
            display_name="Excluded",
            description="has every optional field set",
            eval_mode=EvalMode.GEN,
            deps_group="extra",
            model_type="chat",
            reference_impl=ReferenceImpl(source="upstream", url="https://x.invalid"),
        )
        class ExcludedTask(_StubTask):
            pass

        identity = _identity_of(ExcludedTask)
        assert identity is not None
        assert set(identity) == {
            "name",
            "display_name",
            "dataset",
            "eval_mode",
            "n_shot",
            "tags",
            "status",
        }

    def test_is_json_shaped(self):
        """`eval_mode` serializes as its enum *value* and `tags` as a list, so
        the projection can go straight through `orjson.dumps`."""

        @sieval_task(
            name="jsonshape",
            display_name="JSON",
            description="x",
            eval_mode=EvalMode.PPL,
            tags=("english",),
        )
        class JsonTask(_StubTask):
            pass

        identity = _identity_of(JsonTask)
        assert identity is not None
        assert identity["eval_mode"] == "ppl"
        assert type(identity["eval_mode"]) is str
        assert identity["tags"] == ["english"]
        assert json.loads(json.dumps(identity)) == identity

    def test_descriptive_tags_not_the_synthesized_protocol_set(self):
        """`tags` is the author-declared `TaskMeta.tags`, not the `cls.tags`
        protocol set the decorator synthesizes from `eval_mode` + `n_shot`."""

        @sieval_task(
            name="tagsplit",
            display_name="Tags",
            description="x",
            eval_mode=EvalMode.GEN,
            n_shot=3,
            tags=("english",),
        )
        class TagTask(_StubTask):
            pass

        identity = _identity_of(TagTask)
        assert identity is not None
        assert identity["tags"] == ["english"]
        assert TagTask.tags == frozenset({"gen", "few_shot"})

    def test_returns_none_for_undecorated_class(self):
        """Fail-soft: `Task` subclasses are not required to be decorated, and
        `write_run_meta` is documented as never raising."""

        class Undecorated(_StubTask):
            pass

        assert _identity_of(Undecorated) is None

    def test_does_not_inherit_identity_from_a_decorated_parent(self):
        """An undecorated subclass is a *different* task. `get_task_meta` reads
        through the MRO and so still resolves the parent's meta; run identity
        deliberately does not, or the subclass's results would be persisted
        under its parent's name."""

        @sieval_task(
            name="parent",
            display_name="Parent",
            description="x",
            eval_mode=EvalMode.GEN,
        )
        class Parent(_StubTask):
            pass

        class Child(Parent):
            pass

        assert _identity_of(Parent) is not None
        assert _identity_of(Child) is None
        # The MRO-reading accessor is unchanged — this is a divergence, not a
        # migration.
        assert get_task_meta(Child).name == "parent"

    def test_n_shot_is_the_run_not_the_declaration(self):
        """A run directory answers "what did this run do", so the shot count
        persisted there is the instance's, not the class's advertisement.

        The override has to be per *instance*: the decorator assigns
        `cls.n_shot` after the class body executes, so a class-level value
        would simply be overwritten by the declared one.
        """

        @sieval_task(
            name="knob",
            display_name="Knob",
            description="x",
            eval_mode=EvalMode.CLP,
            n_shot=5,
        )
        class KnobTask(_StubTask):
            pass

        # What a real task's `__init__` does with its shot-count knob.
        task = object.__new__(KnobTask)
        task.n_shot = 3

        identity = get_task_run_identity(task)
        assert identity is not None
        assert identity["n_shot"] == 3
        # Only the projection moves: the declaration and the catalog row it
        # feeds still say 5.
        assert get_task_meta(KnobTask).n_shot == 5
        assert KnobTask.n_shot == 5

    def test_class_level_n_shot_cannot_shadow_the_declaration(self):
        """The decorator is the single declaration, so it wins over the body.

        Guards the reading that a subclass could advertise one count while
        `@sieval_task` declares another — it cannot, and a run that wants a
        different count shadows per instance instead.
        """

        @sieval_task(
            name="bodyshadow",
            display_name="BodyShadow",
            description="x",
            eval_mode=EvalMode.CLP,
            n_shot=5,
        )
        class BodyShadowTask(_StubTask):
            n_shot = 3  # overwritten by the decorator

        assert BodyShadowTask.n_shot == 5
        identity = _identity_of(BodyShadowTask)
        assert identity is not None
        assert identity["n_shot"] == 5

    def test_declared_n_shot_stands_without_a_knob(self):
        """`@sieval_task` seeds `cls.n_shot`, so a task with no shot-count knob
        keeps projecting what it declared, with no code of its own — the common
        case."""

        @sieval_task(
            name="noknob",
            display_name="NoKnob",
            description="x",
            eval_mode=EvalMode.CLP,
            n_shot=4,
        )
        class NoKnobTask(_StubTask):
            pass

        identity = _identity_of(NoKnobTask)
        assert identity is not None
        assert identity["n_shot"] == 4

    def test_rejects_a_class(self):
        """Passing a class resolves to no metadata, which the fail-soft `None`
        path would report as "undecorated" and silently drop the block. The
        previous signature took a class, so this is a reachable mistake."""

        @sieval_task(
            name="classarg",
            display_name="ClassArg",
            description="x",
            eval_mode=EvalMode.GEN,
        )
        class ClassArgTask(_StubTask):
            pass

        with pytest.raises(TypeError, match="instance, not a class"):
            get_task_run_identity(ClassArgTask)  # ty: ignore[invalid-argument-type]


def _valid_kwargs(**overrides):
    base = {
        "name": "x",
        "display_name": "X",
        "description": "desc",
        "eval_mode": EvalMode.GEN,
    }
    base.update(overrides)
    return base


def test_validation_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):

        @sieval_task(**_valid_kwargs(name=""))
        class T(_StubTask):
            pass


def test_validation_rejects_empty_description():
    with pytest.raises(ValueError, match="description"):

        @sieval_task(**_valid_kwargs(description=""))
        class T(_StubTask):
            pass


def test_validation_rejects_empty_display_name():
    with pytest.raises(ValueError, match="display_name"):

        @sieval_task(**_valid_kwargs(display_name=""))
        class T(_StubTask):
            pass


def test_validation_rejects_overlong_description():
    with pytest.raises(ValueError, match="description"):

        @sieval_task(**_valid_kwargs(description="x" * 121))
        class T(_StubTask):
            pass


def test_validation_rejects_negative_n_shot():
    with pytest.raises(ValueError, match="n_shot"):

        @sieval_task(**_valid_kwargs(n_shot=-1))
        class T(_StubTask):
            pass


def test_validation_requires_pinned_commit_url_for_github_ref_impl():
    bad = ReferenceImpl(
        source="simple-evals", url="https://github.com/x/y/blob/main/f.py"
    )
    with pytest.raises(ValueError, match="pinned commit"):

        @sieval_task(**_valid_kwargs(reference_impl=bad))
        class T(_StubTask):
            pass


def test_validation_accepts_pinned_github_ref_impl():
    good = ReferenceImpl(
        source="simple-evals",
        url="https://github.com/x/y/blob/0123456789abcdef0123456789abcdef01234567/f.py",
    )

    @sieval_task(**_valid_kwargs(reference_impl=good))
    class T(_StubTask):
        pass

    assert get_task_meta(T).reference_impl == good


def test_validation_allows_non_github_ref_impl_url():
    """Non-github URLs bypass commit-pinning check (not applicable)."""
    ri = ReferenceImpl(source="official", url="https://example.org/paper.pdf")

    @sieval_task(**_valid_kwargs(reference_impl=ri))
    class T(_StubTask):
        pass

    assert get_task_meta(T).reference_impl == ri


def test_validation_does_not_trigger_pinned_check_on_non_github_host_with_github_substring():  # noqa: E501
    """URLs whose path (not host) contains 'github.com' must not trip the gate."""
    ri = ReferenceImpl(
        source="official",
        url="https://example.com/github.com-mirror/path",
    )

    @sieval_task(**_valid_kwargs(reference_impl=ri))
    class T(_StubTask):
        pass

    assert get_task_meta(T).reference_impl == ri


def test_validation_rejects_www_github_com_unpinned():
    bad = ReferenceImpl(
        source="simple-evals",
        url="https://www.github.com/x/y/blob/main/f.py",
    )
    with pytest.raises(ValueError, match="pinned commit"):

        @sieval_task(**_valid_kwargs(reference_impl=bad))
        class T(_StubTask):
            pass


def test_validation_rejects_raw_githubusercontent_unpinned():
    bad = ReferenceImpl(
        source="x",
        url="https://raw.githubusercontent.com/owner/repo/main/path/f.py",
    )
    with pytest.raises(ValueError, match="pinned commit"):

        @sieval_task(**_valid_kwargs(reference_impl=bad))
        class T(_StubTask):
            pass


def test_validation_accepts_raw_githubusercontent_pinned():
    good = ReferenceImpl(
        source="x",
        url="https://raw.githubusercontent.com/owner/repo/0123456789abcdef0123456789abcdef01234567/path/f.py",
    )

    @sieval_task(**_valid_kwargs(reference_impl=good))
    class T(_StubTask):
        pass

    assert get_task_meta(T).reference_impl == good


def test_validation_rejects_gist_unpinned():
    bad = ReferenceImpl(
        source="x",
        url="https://gist.github.com/user/abcdef1234567890abcdef1234567890",
    )
    with pytest.raises(ValueError, match="pinned commit"):

        @sieval_task(**_valid_kwargs(reference_impl=bad))
        class T(_StubTask):
            pass


def test_validation_accepts_gist_pinned():
    good = ReferenceImpl(
        source="x",
        url="https://gist.github.com/user/0123456789abcdef0123456789abcdef01234567/fedcba9876543210fedcba9876543210fedcba98",
    )

    @sieval_task(**_valid_kwargs(reference_impl=good))
    class T(_StubTask):
        pass

    assert get_task_meta(T).reference_impl == good


def test_sieval_task_overrides_classvar_tags_silently():
    """Hand-written ClassVar tags on a decorated class are silently overwritten.

    `sieval/tasks/CLAUDE.md` forbids setting tags manually on decorated
    classes; the decorator unconditionally replaces `cls.tags` with the
    synthesized protocol set. This regression test pins that behavior so
    mixing the legacy pattern can't partially leak through.
    """
    from typing import ClassVar

    @sieval_task(**_valid_kwargs(name="ovr", eval_mode=EvalMode.GEN, n_shot=0))
    class T(_StubTask):
        tags: ClassVar[frozenset[str]] = frozenset({"stale", "manual"})

    assert T.tags == frozenset({"gen", "zero_shot"})


def test_task_meta_to_dict_roundtrips_basic_fields():
    meta = TaskMeta(
        name="s",
        display_name="S",
        description="d",
        dataset="stub_dataset",
        eval_mode=EvalMode.GEN,
        n_shot=3,
        tags=("a", "b"),
        deps_group="math",
        model_type="chat",
    )
    d = task_meta_to_dict(meta)
    assert json.dumps(d) is not None
    assert d["name"] == "s"
    assert d["dataset"] == "stub_dataset"
    assert d["eval_mode"] == "gen"
    assert d["n_shot"] == 3
    assert d["tags"] == ["a", "b"]


def test_sieval_task_sets_protocol_tags_from_eval_mode_and_n_shot():
    @sieval_task(**_valid_kwargs(name="p_gen_zero", eval_mode=EvalMode.GEN, n_shot=0))
    class TGenZero(_StubTask):
        pass

    assert TGenZero.tags == frozenset({"gen", "zero_shot"})

    @sieval_task(**_valid_kwargs(name="p_ppl_few", eval_mode=EvalMode.PPL, n_shot=5))
    class TPplFew(_StubTask):
        pass

    assert TPplFew.tags == frozenset({"ppl", "few_shot"})

    @sieval_task(**_valid_kwargs(name="p_gen_few", eval_mode=EvalMode.GEN, n_shot=3))
    class TGenFew(_StubTask):
        pass

    assert TGenFew.tags == frozenset({"gen", "few_shot"})

    @sieval_task(**_valid_kwargs(name="p_clp_zero", eval_mode=EvalMode.CLP, n_shot=0))
    class TClpZero(_StubTask):
        pass

    assert TClpZero.tags == frozenset({"clp", "zero_shot"})


def test_sieval_task_sets_class_model_type():
    @sieval_task(**_valid_kwargs(name="mt_chat", model_type="chat"))
    class TChat(_StubTask):
        pass

    assert TChat.model_type == "chat"

    @sieval_task(**_valid_kwargs(name="mt_none", model_type=None))
    class TNone(_StubTask):
        pass

    # model_type=None means "don't touch the class attr" — decorator must not
    # write None onto the class, so subclasses can inherit from Task base.
    assert "model_type" not in TNone.__dict__


def test_sieval_task_descriptive_tags_do_not_leak_to_class_tags():
    @sieval_task(
        **_valid_kwargs(
            name="desc",
            eval_mode=EvalMode.GEN,
            n_shot=0,
            tags=("english", "open-ended"),
        )
    )
    class T(_StubTask):
        pass

    # cls.tags is the synthesized *protocol* vocabulary only
    assert T.tags == frozenset({"gen", "zero_shot"})
    # the *descriptive* tuple is preserved on _sieval_task_meta
    assert get_task_meta(T).tags == ("english", "open-ended")


def test_task_meta_to_dict_serializes_reference_impl():
    ri = ReferenceImpl(
        source="simple-evals",
        url="https://github.com/x/y/blob/0123456789abcdef0123456789abcdef01234567/f.py",
        notes="n",
    )
    meta = TaskMeta(
        name="r",
        display_name="R",
        description="d",
        dataset="stub_dataset",
        eval_mode=EvalMode.GEN,
        reference_impl=ri,
    )
    d = task_meta_to_dict(meta)
    assert d["reference_impl"]["source"] == "simple-evals"
    assert d["reference_impl"]["notes"] == "n"


def test_sieval_task_rejects_unregistered_sample_type():
    """A Task whose sample type has no registered Dataset must fail."""

    class _UnknownSample(TypedDict):
        y: str

    with pytest.raises(ValueError, match="No @sieval_dataset found"):

        @sieval_task(**_valid_kwargs(name="orphan"))
        class T(Task[_UnknownSample, None, None, None, None, dict[str, float]]):
            pass


def test_sieval_task_rejects_task_without_generic_args():
    """A bare Task subclass (no generic args) must fail at registration."""

    class _BareTask(Task):
        pass

    with pytest.raises(ValueError, match="no concrete generic args"):

        @sieval_task(**_valid_kwargs(name="bare"))
        class T(_BareTask):
            pass


def test_iter_task_entries_returns_class_meta_pairs():
    @sieval_task(
        name="entry1",
        display_name="E1",
        description="x",
        eval_mode=EvalMode.GEN,
    )
    class E1(_StubTask):
        pass

    from sieval.core.tasks.meta import iter_task_entries

    entries = list(iter_task_entries())
    assert len(entries) == 1
    cls, meta = entries[0]
    assert cls is E1
    assert meta.name == "entry1"


def test_lookup_task_returns_registered():
    from sieval.core.tasks.meta import lookup_task

    @sieval_task(
        name="lookup_test_task",
        display_name="Lookup Test Task",
        description="A task for lookup testing.",
        eval_mode=EvalMode.GEN,
    )
    class _LookupTask(_StubTask):
        pass

    found = lookup_task("lookup_test_task")
    assert found is not None
    assert found.name == "lookup_test_task"


def test_lookup_task_returns_none_for_unknown():
    from sieval.core.tasks.meta import lookup_task

    assert lookup_task("nonexistent_yyy") is None


def test_tasks_for_dataset_yields_matching():
    from sieval.core.tasks.meta import tasks_for_dataset

    @sieval_task(
        name="ds_filter_task_1",
        display_name="DS Filter Task 1",
        description="First task for filter test.",
        eval_mode=EvalMode.GEN,
    )
    class _DsFilterTask1(_StubTask):
        pass

    @sieval_task(
        name="ds_filter_task_2",
        display_name="DS Filter Task 2",
        description="Second task for filter test.",
        eval_mode=EvalMode.PPL,
    )
    class _DsFilterTask2(_StubTask):
        pass

    results = list(tasks_for_dataset("stub_dataset"))
    assert len(results) == 2
    assert all(t.dataset == "stub_dataset" for t in results)


def test_tasks_for_dataset_empty_for_unknown():
    from sieval.core.tasks.meta import tasks_for_dataset

    assert list(tasks_for_dataset("nonexistent")) == []


def test_get_task_class_returns_registered_class():
    """An already-registered name resolves straight out of `_TASK_CLASSES`.

    The target is registered by this test rather than read off the shipped
    index, so the assertion is exact (identity, not `issubclass`) and does not
    shift with whichever task happens to sort first.
    """
    from sieval.core.tasks.meta import get_task_class

    @sieval_task(
        name="lookup_pilot",
        display_name="Lookup Pilot",
        description="registered target for get_task_class",
        eval_mode=EvalMode.GEN,
    )
    class _LookupPilotTask(_StubTask):
        pass

    assert get_task_class("lookup_pilot") is _LookupPilotTask


def test_get_task_class_lazy_imports_unregistered_module(tmp_path):
    """A name missing from `_TASK_CLASSES` resolves by importing
    `sieval.tasks.{name}` — the one-task-per-eponymous-module convention that
    lets point lookups skip a full `import_all_tasks()`.
    """
    import sys

    import sieval.tasks
    from sieval.core.tasks.meta import TASK_REGISTRY, get_task_class

    name = "lazy_task_for_test"
    # Decorating in a real module file (rather than in this test body) is what
    # makes the import the only path to registration.
    (tmp_path / f"{name}.py").write_text(
        "from sieval.core.tasks.meta import EvalMode, sieval_task\n"
        f"from {__name__} import _StubTask\n"
        "\n"
        "\n"
        "@sieval_task(\n"
        f'    name="{name}",\n'
        '    display_name="Lazy",\n'
        '    description="lazy-import target",\n'
        "    eval_mode=EvalMode.GEN,\n"
        ")\n"
        "class LazyForTestTask(_StubTask):\n"
        "    pass\n"
    )
    sieval.tasks.__path__.insert(0, str(tmp_path))
    try:
        assert name not in TASK_REGISTRY  # only the import can register it
        cls = get_task_class(name)
        assert cls.__name__ == "LazyForTestTask"
        assert TASK_REGISTRY[name].display_name == "Lazy"
    finally:
        sieval.tasks.__path__.remove(str(tmp_path))
        sys.modules.pop(f"sieval.tasks.{name}", None)
        # The successful import also bound the submodule on the package; drop it
        # so nothing outside this test can reach a module whose file is gone.
        sieval.tasks.__dict__.pop(name, None)


def test_get_task_class_lazy_imports_subpackage_hosted_module(tmp_path):
    """A benchmark that outgrew a flat layout keeps the eponymous filename inside
    its subpackage (`sieval/tasks/CLAUDE.md`, `arc/`), so the lookup must fall
    back to scanning subpackages for `{subpkg}.{name}`. Without that fallback the
    module exists and imports fine but `get_task_class()` raises KeyError, taking
    `sieval task show` and by-name task resolution down with it.
    """
    import sys

    import sieval.tasks
    from sieval.core.tasks.meta import TASK_REGISTRY, get_task_class

    subpkg, name = "pkg_for_test", "nested_task_for_test"
    (tmp_path / subpkg).mkdir()
    (tmp_path / subpkg / "__init__.py").write_text("")
    (tmp_path / subpkg / f"{name}.py").write_text(
        "from sieval.core.tasks.meta import EvalMode, sieval_task\n"
        f"from {__name__} import _StubTask\n"
        "\n"
        "\n"
        "@sieval_task(\n"
        f'    name="{name}",\n'
        '    display_name="Nested",\n'
        '    description="subpackage-hosted lazy-import target",\n'
        "    eval_mode=EvalMode.GEN,\n"
        ")\n"
        "class NestedForTestTask(_StubTask):\n"
        "    pass\n"
    )
    sieval.tasks.__path__.insert(0, str(tmp_path))
    try:
        assert name not in TASK_REGISTRY  # only the import can register it
        # The flat path must genuinely miss first, or this proves nothing.
        assert not (tmp_path / f"{name}.py").exists()
        cls = get_task_class(name)
        assert cls.__name__ == "NestedForTestTask"
        assert TASK_REGISTRY[name].display_name == "Nested"
    finally:
        sieval.tasks.__path__.remove(str(tmp_path))
        sys.modules.pop(f"sieval.tasks.{subpkg}.{name}", None)
        sys.modules.pop(f"sieval.tasks.{subpkg}", None)
        # The successful import also bound the submodule on the package; drop it
        # so nothing outside this test can reach a module whose file is gone.
        sieval.tasks.__dict__.pop(subpkg, None)


def test_get_task_class_raises_key_error_on_unknown_name():
    """Unregistered names raise KeyError — a programmer-error signal.

    Also covers the exhausted-scan path: neither the flat module nor any
    subpackage hosts the name, and the lookup still ends in KeyError rather than
    leaking a ModuleNotFoundError from the last candidate it tried.
    """
    import pytest

    from sieval.core.tasks.meta import get_task_class

    with pytest.raises(KeyError):
        get_task_class("nonexistent_task_name_zzz")


def test_get_task_class_surfaces_nested_import_error(tmp_path):
    """Regression: a task module whose own import fails (missing third-party
    dep) must surface the real ModuleNotFoundError, not a cryptic KeyError.
    Previously `contextlib.suppress(ModuleNotFoundError)` swallowed both the
    "no such task module" case AND nested dependency failures, leaving users
    staring at `KeyError: 'mytask'` when the actual cause was e.g.
    `No module named 'math_verify'`.
    """
    import sys

    import pytest

    import sieval.tasks
    from sieval.core.tasks.meta import get_task_class

    # Build a fake task module that imports a library that definitely isn't
    # installed, and inject its directory into sieval.tasks.__path__.
    task_file = tmp_path / "broken_task_for_test.py"
    task_file.write_text("import definitely_missing_lib_xyz\n")
    sieval.tasks.__path__.insert(0, str(tmp_path))
    try:
        with pytest.raises(ModuleNotFoundError) as excinfo:
            get_task_class("broken_task_for_test")
        # The raised error must name the nested dep, not the task module.
        assert excinfo.value.name == "definitely_missing_lib_xyz"
    finally:
        sieval.tasks.__path__.remove(str(tmp_path))
        sys.modules.pop("sieval.tasks.broken_task_for_test", None)


class TestTaskMetaRoundTrip:
    """`task_meta_from_dict` is the reverse of `task_meta_to_dict`.

    It had no field-level tests: 84 mutants survived, meaning any field could be
    read from the wrong key, dropped, or swapped with its neighbour and nothing
    would notice. `meta/index.json` is how every consumer outside this process
    learns what a task is, so a mis-mapped field is not a local error.
    """

    def _full(self) -> TaskMeta:
        # Every field distinct, so a swap between any two is visible.
        return TaskMeta(
            name="a_name",
            display_name="A Display Name",
            description="a description",
            dataset="a_dataset",
            eval_mode=EvalMode.GEN,
            n_shot=7,
            tags=("t1", "t2"),
            deps_group="a_group",
            model_type="chat",
            reference_impl=ReferenceImpl(
                source="a_source", url="https://example.com/x", notes="a note"
            ),
            status="experimental",
        )

    def test_round_trip_preserves_every_field(self):
        meta = self._full()
        assert task_meta_from_dict(task_meta_to_dict(meta)) == meta

    def test_each_field_lands_in_its_own_slot(self):
        # Round-trip equality alone cannot catch a *symmetric* swap, so the
        # fields are also read back individually.
        got = task_meta_from_dict(task_meta_to_dict(self._full()))
        assert got.name == "a_name"
        assert got.display_name == "A Display Name"
        assert got.description == "a description"
        assert got.dataset == "a_dataset"
        assert got.eval_mode is EvalMode.GEN
        assert got.n_shot == 7
        assert got.tags == ("t1", "t2")
        assert got.deps_group == "a_group"
        assert got.model_type == "chat"
        assert got.status == "experimental"

    def test_reference_impl_survives_the_round_trip(self):
        got = task_meta_from_dict(task_meta_to_dict(self._full()))
        assert got.reference_impl is not None
        assert got.reference_impl.source == "a_source"
        assert got.reference_impl.url == "https://example.com/x"
        assert got.reference_impl.notes == "a note"

    def test_eval_mode_comes_back_as_the_enum(self):
        # The dict carries its raw value; left as a string, a consumer comparing
        # against EvalMode would silently never match.
        got = task_meta_from_dict(task_meta_to_dict(self._full()))
        assert isinstance(got.eval_mode, EvalMode)

    def test_tags_come_back_as_a_tuple(self):
        # TaskMeta is frozen+slots; a list would make it unhashable and mutable
        # through the caller's own reference.
        got = task_meta_from_dict(task_meta_to_dict(self._full()))
        assert isinstance(got.tags, tuple)


class TestTaskMetaFromDictDefaults:
    """Absent optional keys fall back to the documented defaults.

    `index.json` rows are release-authored and omit fields sitting at their
    default, so a wrong default here silently rewrites what a task claims to be.
    """

    def _minimal(self) -> dict:
        return {
            "name": "n",
            "display_name": "d",
            "description": "desc",
            "dataset": "ds",
            "eval_mode": EvalMode.GEN.value,
        }

    def test_n_shot_defaults_to_zero(self):
        assert task_meta_from_dict(self._minimal()).n_shot == 0

    def test_tags_default_to_empty(self):
        assert task_meta_from_dict(self._minimal()).tags == ()

    def test_optional_strings_default_to_none(self):
        got = task_meta_from_dict(self._minimal())
        assert got.deps_group is None
        assert got.model_type is None

    def test_status_defaults_to_stable(self):
        # An omitted status must not downgrade a task — consumers gate on this.
        assert task_meta_from_dict(self._minimal()).status == "stable"

    def test_absent_reference_impl_is_none(self):
        assert task_meta_from_dict(self._minimal()).reference_impl is None

    def test_reference_impl_notes_default_to_empty(self):
        payload = self._minimal() | {
            "reference_impl": {"source": "s", "url": "https://example.com/u"}
        }
        ref = task_meta_from_dict(payload).reference_impl
        assert ref is not None
        assert ref.notes == ""

    def test_present_values_beat_the_defaults(self):
        payload = self._minimal() | {
            "n_shot": 5,
            "tags": ["x"],
            "deps_group": "g",
            "model_type": "gen",
            "status": "experimental",
        }
        got = task_meta_from_dict(payload)
        assert (got.n_shot, got.tags, got.deps_group, got.model_type, got.status) == (
            5,
            ("x",),
            "g",
            "gen",
            "experimental",
        )
