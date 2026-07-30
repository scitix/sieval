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
    iter_task_metas,
    sieval_task,
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
    """Bare Task subclass used only as a @sieval_task decoration target.

    Tests never instantiate these stubs, so the abstract methods are left
    unimplemented; inheriting from Task satisfies the decorator's `type[Task]`
    bound and supplies the `tags` / `model_type` ClassVar declarations ty needs
    to resolve attribute accesses after decoration.
    """


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

    Also pins the eponymous-module convention `get_task_class()` depends on
    (`sieval/core/tasks/meta.py`): every registered name must be reachable as
    `sieval.tasks.{name}`, or `sieval task list` cannot resolve its class. Note
    this is deliberately stricter than the lazy export map, which also accepts
    subpackage-hosted tasks as `"subpkg.module_stem"` (`sieval/tasks/__init__.py`
    "Subpackage .py modules"). Such a task would import and export fine but be
    unreachable through `get_task_class()`, so the flat layout is the binding
    constraint until that lookup learns to walk subpackages.
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
    off_convention = [
        name for name in TASK_REGISTRY if f"sieval.tasks.{name}" not in sys.modules
    ]
    assert not off_convention, (
        f"tasks not defined in an eponymous module: {sorted(off_convention)}"
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


def test_get_task_class_raises_key_error_on_unknown_name():
    """Unregistered names raise KeyError — a programmer-error signal."""
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
