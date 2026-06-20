"""
Unit tests for sieval.core.datasets.meta.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from typing import TypedDict

import pytest

from sieval.core.datasets import Dataset
from sieval.core.datasets.meta import (
    DATASET_REGISTRY,
    SAMPLE_TO_DATASET,
    Category,
    DatasetMeta,
    Level1Category,
    dataset_meta_from_dict,
    dataset_meta_to_dict,
    get_dataset_meta,
    iter_dataset_metas,
    sieval_dataset,
)


class FooSample(TypedDict):
    prompt: str


class BarSample(TypedDict):
    prompt: str


@pytest.fixture(autouse=True)
def _clean_registries():
    """Each test starts with empty registries; restore afterwards.

    Also tracks `sieval.datasets.*` entries in `sys.modules` so that a test
    that triggers `import_all_datasets()` does not leave modules cached —
    otherwise later tests that rely on the real dataset decorators running
    (e.g. Task-side reverse-lookup) will see an empty `SAMPLE_TO_DATASET`.
    """
    import sys

    saved_ds = DATASET_REGISTRY.copy()
    saved_map = SAMPLE_TO_DATASET.copy()
    preloaded_modules = {
        name for name in sys.modules if name.startswith("sieval.datasets.")
    }
    DATASET_REGISTRY.clear()
    SAMPLE_TO_DATASET.clear()
    yield
    DATASET_REGISTRY.clear()
    DATASET_REGISTRY.update(saved_ds)
    SAMPLE_TO_DATASET.clear()
    SAMPLE_TO_DATASET.update(saved_map)
    for name in list(sys.modules):
        if name.startswith("sieval.datasets.") and name not in preloaded_modules:
            del sys.modules[name]


def test_sieval_dataset_happy_path():
    @sieval_dataset(
        name="foo",
        display_name="Foo Bench",
        description="A test dataset.",
        source="hf:org/foo",
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        tags=("english",),
    )
    class FooDataset(Dataset[FooSample]):
        pass

    meta = get_dataset_meta(FooDataset)
    assert isinstance(meta, DatasetMeta)
    assert meta.name == "foo"
    assert meta.display_name == "Foo Bench"
    assert DATASET_REGISTRY["foo"] is meta
    assert SAMPLE_TO_DATASET[FooSample] is FooDataset
    assert list(iter_dataset_metas()) == [meta]


def test_dataset_meta_to_dict_roundtrip():
    @sieval_dataset(
        name="foo",
        display_name="Foo",
        description="Test.",
        source="hf:org/foo",
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    class FooDataset(Dataset[FooSample]):
        pass

    payload = dataset_meta_to_dict(get_dataset_meta(FooDataset))
    assert payload == {
        "name": "foo",
        "display_name": "Foo",
        "description": "Test.",
        "source": ["hf:org/foo"],
        "categories": [{"level1": "Mathematics", "level2": "CompetitionMath"}],
        "tags": [],
        "deps_group": None,
        "license": None,
        "checksums": {},
    }


def test_duplicate_name_raises():
    @sieval_dataset(
        name="foo",
        display_name="Foo",
        description="Test.",
        source="hf:org/foo",
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    class FooDataset(Dataset[FooSample]):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @sieval_dataset(
            name="foo",
            display_name="Foo Again",
            description="Test.",
            source="hf:org/bar",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class FooDuplicateDataset(Dataset[BarSample]):
            pass


def test_duplicate_sample_type_raises():
    @sieval_dataset(
        name="foo",
        display_name="Foo",
        description="Test.",
        source="hf:org/foo",
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    class FooDataset(Dataset[FooSample]):
        pass

    with pytest.raises(ValueError, match="sample type .* already bound"):

        @sieval_dataset(
            name="foo2",
            display_name="Foo2",
            description="Test.",
            source="hf:org/foo2",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class FooDupSampleDataset(Dataset[FooSample]):
            pass


def test_validate_empty_name():
    with pytest.raises(ValueError, match="name must be non-empty"):

        @sieval_dataset(
            name="",
            display_name="X",
            description="X",
            source="hf:o/x",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_empty_display_name():
    with pytest.raises(ValueError, match="display_name must be non-empty"):

        @sieval_dataset(
            name="x",
            display_name="",
            description="X",
            source="hf:o/x",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_empty_description():
    with pytest.raises(ValueError, match="description must be non-empty"):

        @sieval_dataset(
            name="x",
            display_name="X",
            description="",
            source="hf:o/x",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_description_too_long():
    with pytest.raises(ValueError, match="description exceeds"):

        @sieval_dataset(
            name="x",
            display_name="X",
            description="x" * 101,
            source="hf:o/x",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_empty_categories():
    with pytest.raises(ValueError, match="at least one category"):

        @sieval_dataset(
            name="x",
            display_name="X",
            description="X",
            source="hf:o/x",
            categories=(),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_invalid_level2():
    with pytest.raises(ValueError, match="is not valid for"):

        @sieval_dataset(
            name="x",
            display_name="X",
            description="X",
            source="hf:o/x",
            categories=(Category(Level1Category.MATHEMATICS, "BogusLevel2"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_bad_source_scheme():
    with pytest.raises(ValueError, match="must use scheme"):

        @sieval_dataset(
            name="x",
            display_name="X",
            description="X",
            source="bogus:not-a-real-scheme",
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_validate_tuple_source_all_checked():
    with pytest.raises(ValueError, match="must use scheme"):

        @sieval_dataset(
            name="x",
            display_name="X",
            description="X",
            source=("hf:o/ok", "bogus:bad"),
            categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        )
        class _D(Dataset[FooSample]):
            pass


def test_cross_registry_collision_dataset_vs_task():
    """sieval_dataset raises if name already in TASK_REGISTRY."""
    from sieval.core.tasks.meta import TASK_REGISTRY

    TASK_REGISTRY["shared_name"] = object()  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="already registered as a Task"):

            @sieval_dataset(
                name="shared_name",
                display_name="X",
                description="X",
                source="hf:o/x",
                categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
            )
            class _D(Dataset[FooSample]):
                pass
    finally:
        TASK_REGISTRY.pop("shared_name", None)


def test_import_all_datasets_imports_submodules():
    from sieval.core.datasets.meta import import_all_datasets

    import_all_datasets()
    # Discriminating: after walking sieval.datasets, at least one @sieval_dataset
    # must have fired (11 pilots registered post-Task-4). A no-op implementation
    # would leave both registries empty and fail here.
    assert len(DATASET_REGISTRY) >= 1
    assert len(SAMPLE_TO_DATASET) >= 1


def test_extract_sample_type_rejects_non_generic_base():
    """A class whose only reachable generic arg is a TypeVar (from PEP 695
    `Dataset[TSample]`'s implicit `Generic[TSample]` base) must fail, not
    silently register a bare TypeVar as the sample type."""
    from sieval.core.datasets.meta import extract_sample_type

    class PlainDataset(Dataset):
        pass

    with pytest.raises(ValueError, match="no concrete generic args"):
        extract_sample_type(PlainDataset)


def test_extract_sample_type_walks_mro_through_intermediate_base():
    """An intermediate abstract base carrying `Dataset[Sample]` must be
    reachable from a concrete subclass via MRO walk (regression test for
    patterns like `_GLRBBaseDataset(Dataset[GLRBSample], ABC)`)."""
    from sieval.core.datasets.meta import extract_sample_type

    class _IntermediateBase(Dataset[FooSample]):
        pass

    class ConcreteDataset(_IntermediateBase):
        pass

    assert extract_sample_type(ConcreteDataset) is FooSample


def test_dataset_meta_accepts_license():
    meta = DatasetMeta(
        name="foo",
        display_name="Foo",
        description="x",
        source=("hf:org/foo",),
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        license="CC-BY-4.0",
    )
    assert meta.license == "CC-BY-4.0"


def test_dataset_meta_license_defaults_to_none():
    meta = DatasetMeta(
        name="foo",
        display_name="Foo",
        description="x",
        source=("hf:org/foo",),
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    assert meta.license is None


def test_single_source_serialized_as_array():
    """Decorator-accepted bare string is normalized to a 1-tuple; wire format
    still serializes as a 1-element array."""
    meta = DatasetMeta(
        name="foo",
        display_name="Foo",
        description="x",
        source=("hf:org/foo",),
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    payload = dataset_meta_to_dict(meta)
    assert payload["source"] == ["hf:org/foo"]


def test_sieval_dataset_decorator_normalizes_bare_string_source():
    """`@sieval_dataset(source="hf:org/foo")` must land on the meta as a tuple."""

    @sieval_dataset(
        name="norm_bare",
        display_name="Norm Bare",
        description="Test normalization.",
        source="hf:org/norm",
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    class _NormBareDS(Dataset[FooSample]):
        pass

    meta = get_dataset_meta(_NormBareDS)
    assert meta.source == ("hf:org/norm",)
    assert isinstance(meta.source, tuple)


def test_sieval_dataset_decorator_accepts_list_source():
    """Lists are accepted for convenience and coerced to tuple."""

    @sieval_dataset(
        name="norm_list",
        display_name="Norm List",
        description="Test list source.",
        source=["url:a", "url:b"],
        categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    )
    class _NormListDS(Dataset[FooSample]):
        pass

    meta = get_dataset_meta(_NormListDS)
    assert meta.source == ("url:a", "url:b")
    assert isinstance(meta.source, tuple)


def test_multi_source_preserved_as_array():
    meta = DatasetMeta(
        name="bar",
        display_name="Bar",
        description="x",
        source=("url:a", "url:b"),
        categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    )
    payload = dataset_meta_to_dict(meta)
    assert payload["source"] == ["url:a", "url:b"]


def test_dict_includes_license():
    meta = DatasetMeta(
        name="foo",
        display_name="Foo",
        description="x",
        source=("hf:org/foo",),
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
        license="MIT",
    )
    payload = dataset_meta_to_dict(meta)
    assert payload["license"] == "MIT"


def test_lookup_dataset_returns_registered():
    from sieval.core.datasets.meta import lookup_dataset

    # Register a fresh dataset in the clean registry (autouse fixture cleared it)
    @sieval_dataset(
        name="lookup_test_ds",
        display_name="Lookup Test DS",
        description="A dataset for lookup test.",
        source="hf:org/lookup",
        categories=(Category(Level1Category.MATHEMATICS, "CompetitionMath"),),
    )
    class _LookupDS(Dataset[FooSample]):
        pass

    meta = lookup_dataset("lookup_test_ds")
    assert meta is not None
    assert meta.name == "lookup_test_ds"


def test_lookup_dataset_returns_none_for_unknown():
    from sieval.core.datasets.meta import lookup_dataset

    assert lookup_dataset("nonexistent_zzz") is None


def test_sieval_dataset_rejects_colliding_url_basenames():
    """Two url: sources in the same dataset with the same basename would
    overwrite each other on disk. Reject at decorator time."""

    class ClashSample(TypedDict):
        x: str

    with pytest.raises(ValueError, match="colliding basenames"):

        @sieval_dataset(
            name="clash_test",
            display_name="Clash Test",
            description="x",
            source=(
                "url:https://a.example.com/data.csv",
                "url:https://b.example.com/data.csv",
            ),
            categories=(Category(Level1Category.LOGIC, "BasicLogic"),),
        )
        class ClashDataset(Dataset[ClashSample]):
            def load(self, name_or_path, **kwargs):
                raise NotImplementedError


def test_sieval_dataset_accepts_different_url_basenames():
    """Different basenames should not trigger the validation."""

    class OkSample(TypedDict):
        x: str

    @sieval_dataset(
        name="ok_test_distinct_basenames",
        display_name="OK",
        description="x",
        source=(
            "url:https://a.example.com/train.csv",
            "url:https://a.example.com/dev.csv",
        ),
        categories=(Category(Level1Category.LOGIC, "BasicLogic"),),
    )
    class OkDataset(Dataset[OkSample]):
        def load(self, name_or_path, **kwargs):
            raise NotImplementedError

    # Must not have raised; cleanup: remove from registry to avoid polluting later tests
    DATASET_REGISTRY.pop("ok_test_distinct_basenames", None)
    SAMPLE_TO_DATASET.pop(OkSample, None)


class CkSample(TypedDict):
    prompt: str


def test_checksums_normalized_sorted_and_round_trips():
    digest = "sha256:" + "a" * 64

    @sieval_dataset(
        name="ck",
        display_name="Ck",
        description="checksum dataset.",
        source="url:https://example.com/y.csv",
        categories=(Category(Level1Category.CODE, "CodeGeneration"),),
        checksums={"y.csv": digest},
    )
    class CkDataset(Dataset[CkSample]):
        pass

    meta = get_dataset_meta(CkDataset)
    assert meta.checksums == (("y.csv", digest),)

    wire = dataset_meta_to_dict(meta)
    assert wire["checksums"] == {"y.csv": digest}
    assert dataset_meta_from_dict(wire).checksums == meta.checksums
    # absent key defaults to empty tuple
    no_ck = {k: v for k, v in wire.items() if k != "checksums"}
    assert dataset_meta_from_dict(no_ck).checksums == ()


def test_checksums_bad_format_rejected():
    with pytest.raises(ValueError, match="sha256"):

        @sieval_dataset(
            name="ckbad",
            display_name="CkBad",
            description="bad checksum.",
            source="url:https://example.com/y.csv",
            categories=(Category(Level1Category.CODE, "CodeGeneration"),),
            checksums={"y.csv": "deadbeef"},
        )
        class CkBadDataset(Dataset[CkSample]):
            pass


def test_checksums_key_must_be_url_basename():
    with pytest.raises(ValueError, match="basename"):

        @sieval_dataset(
            name="ckkey",
            display_name="CkKey",
            description="bad key.",
            source="url:https://example.com/y.csv",
            categories=(Category(Level1Category.CODE, "CodeGeneration"),),
            checksums={"other.csv": "sha256:" + "a" * 64},
        )
        class CkKeyDataset(Dataset[CkSample]):
            pass
