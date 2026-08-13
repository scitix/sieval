"""
Unit tests for Model.with_args, lifecycle, and meta() derivation logic.

Covers parent limiter wiring, nested derivation, shared wrapper ownership, and
meta() field presence — paths not exercised by test_model.py.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import anyio
import pytest

from sieval.core.models import GenModel

# These tests never execute a request.  The compatibility wrapper's real
# dialect is therefore the most faithful fixture for pool/limiter derivation;
# retaining an old ``Transport.CAPABILITIES`` double would exercise the
# capability system that RFC #25 intentionally removed.
StubGenModel = GenModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_gen():
    """GenModel with a concurrency limiter (total_tokens=64)."""
    return StubGenModel(model="base-gen", api_key="fake", concurrency_limit=64)


@pytest.fixture
def base_gen_no_limit():
    """GenModel without any concurrency limiter."""
    return StubGenModel(model="base-gen-unlimited", api_key="fake")


# ===================================================================
# TestModelDerivation
# ===================================================================
class TestModelDerivation:
    # ------------------------------------------------------------------
    # with_args — parent limiter wiring
    # ------------------------------------------------------------------

    def test_with_args_creates_child_with_parent_limiter(self, base_gen):
        """with_args(concurrency_limit=32) → child._parent_limiter is base _limiter."""
        child = base_gen.with_args(concurrency_limit=32)

        assert child._limiter is not None
        assert child._limiter.total_tokens == 32
        # Parent's own limiter becomes the child's _parent_limiter
        assert child._parent_limiter is base_gen._limiter

    def test_with_args_preserves_external_shared_parent_limiter(self):
        """A derived local cap keeps an externally supplied shared quota visible."""
        parent = anyio.CapacityLimiter(64)
        base = StubGenModel(
            model="base-gen-external-parent",
            api_key="fake",
            parent_limiter=parent,
        )

        child = base.with_args(concurrency_limit=32)

        assert child.pool.shared_limiter is parent
        assert child._parent_limiter is parent
        assert child.get_quota_info()["parent"] == {
            "available": 64,
            "total": 64,
        }

    def test_with_args_without_concurrency_limit_shares_limiters(self, base_gen):
        """with_args() without concurrency_limit keeps the same limiter refs."""
        child = base_gen.with_args(temperature=0.7)

        # Shares the exact same limiter object — no new limiter created
        assert child._limiter is base_gen._limiter
        # parent_limiter is also unchanged (base_gen has no parent, so None)
        assert child._parent_limiter is base_gen._parent_limiter

    def test_with_args_without_concurrency_limit_no_base_limiter(
        self, base_gen_no_limit
    ):
        """with_args() on an unlimited model → child also has no limiter."""
        child = base_gen_no_limit.with_args(temperature=0.5)

        assert child._limiter is None
        assert child._parent_limiter is None

    def test_with_args_new_limiter_total_tokens(self, base_gen):
        """Child limiter capacity equals the requested concurrency_limit."""
        child = base_gen.with_args(concurrency_limit=16)

        assert child._limiter.total_tokens == 16

    def test_with_args_shares_transport(self, base_gen):
        """Derived models reuse the same Transport (same client, same wire)."""
        child = base_gen.with_args(temperature=0.7)

        assert child._transport is base_gen._transport
        assert child.capabilities == base_gen.capabilities

    @pytest.mark.parametrize(
        "key",
        [
            "api_base",
            "api_key",
            "authorization",
            "base_url",
            "connection_family",
            "max_retries",
            "service_role",
            "transport",
        ],
    )
    def test_with_args_rejects_binding_resource_keys(self, base_gen, key):
        with pytest.raises(ValueError, match="cannot change binding resources"):
            base_gen.with_args(**{key: "replacement"})

    @pytest.mark.parametrize(
        ("container", "key"),
        [
            ("extra_body", "authorization"),
            ("extra_wire_params", "connection_family"),
        ],
    )
    def test_request_builder_rejects_resources_inside_wire_extensions(
        self, base_gen, container, key
    ):
        with pytest.raises(ValueError, match="cannot contain binding resource"):
            base_gen._build_generate_request(
                "prompt",
                **{container: {key: "secret-or-resource"}},
            )

    # ------------------------------------------------------------------
    # Nested derivation
    # ------------------------------------------------------------------

    def test_nested_derivation_child_has_parent_limiter(self, base_gen):
        """base → child1 (limit=32) → child1.with_args(no limit): child2 inherits."""
        child1 = base_gen.with_args(concurrency_limit=32)
        # Further derivation without new concurrency_limit is allowed
        child2 = child1.with_args(temperature=0.1)

        # child2 shares child1's limiter
        assert child2._limiter is child1._limiter
        # child2 also shares child1's parent_limiter (which points to base_gen._limiter)
        assert child2._parent_limiter is not None
        assert child2._parent_limiter is base_gen._limiter

    def test_nested_derivation_with_new_limit_raises(self, base_gen):
        """base → child1 (limit=32) → child1.with_args(limit=16) must raise."""
        child1 = base_gen.with_args(concurrency_limit=32)

        with pytest.raises(ValueError, match="multi-level"):
            child1.with_args(concurrency_limit=16)

    def test_nested_derivation_with_new_limit_raises_from_unlimited_base(
        self, base_gen_no_limit
    ):
        """A local cap remains a derivation even without a shared root cap."""
        child1 = base_gen_no_limit.with_args(concurrency_limit=32)

        assert child1.pool.shared_limiter is None
        assert child1._parent_limiter is None
        with pytest.raises(ValueError, match="multi-level"):
            child1.with_args(concurrency_limit=16)

    @pytest.mark.parametrize("concurrency_limit", [True, 0])
    def test_with_args_rejects_invalid_concurrency_limit(
        self, base_gen, concurrency_limit
    ):
        with pytest.raises(ValueError, match="positive integer"):
            base_gen.with_args(concurrency_limit=concurrency_limit)

    @pytest.mark.anyio
    async def test_close_via_derived_wrapper_invalidates_compat_pool_sibling(
        self, base_gen
    ):
        child = base_gen.with_args(concurrency_limit=32)
        compat = base_gen.as_compat_type(GenModel)

        await child.aclose()

        assert base_gen.pool.is_closed
        for sibling in (base_gen, child):
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await sibling.__aenter__()
            with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
                await sibling.agenerate("prompt")
        with pytest.raises(RuntimeError, match="canonical model's ConnectionPool"):
            await compat.__aenter__()
        with pytest.raises(RuntimeError, match="ConnectionPool is closing"):
            await compat.agenerate("prompt")
        with pytest.raises(RuntimeError, match="canonical Model does not own"):
            await compat.aclose()

        # Closing through an owning sibling remains idempotent and does not revive it.
        await child.aclose()
        assert base_gen.pool.is_closed

    # ------------------------------------------------------------------
    # meta()
    # ------------------------------------------------------------------

    def test_model_meta_contains_model_field(self, base_gen):
        """meta() must return a dict with a 'model' key."""
        m = base_gen.meta()

        assert "model" in m
        assert m["model"] == "base-gen"

    def test_model_meta_contains_api_base(self, base_gen):
        """meta() 'api_base' is None when no api_base was given."""
        m = base_gen.meta()

        assert "api_base" in m
        assert m["api_base"] is None

    def test_model_meta_contains_default_params(self, base_gen):
        """meta() 'default_params' reflects kwargs passed at construction."""
        model = StubGenModel(
            model="test-params", api_key="fake", temperature=0.5, top_p=0.9
        )
        m = model.meta()

        assert "default_params" in m
        assert m["default_params"]["temperature"] == 0.5
        assert m["default_params"]["top_p"] == 0.9

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_binding_rejects_non_finite_default_params(self, value):
        """Rejected at bind time, so no model call can be spent on it first."""
        with pytest.raises(ValueError, match="non-finite"):
            StubGenModel(model="bad-param", api_key="fake", custom=value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_request_builder_rejects_non_finite_sampling_values(self, base_gen, value):
        with pytest.raises(ValueError, match="finite"):
            base_gen._build_generate_request("prompt", temperature=value)

    def test_model_meta_api_base_set(self):
        """meta() 'api_base' reflects the value passed at construction."""
        model = StubGenModel(
            model="remote", api_key="fake", api_base="http://localhost:9000"
        )
        m = model.meta()

        assert m["api_base"] == "http://localhost:9000"

    def test_model_meta_after_with_args_inherits_model_name(self, base_gen):
        """Derived model's meta() still reports the original model name."""
        child = base_gen.with_args(temperature=0.3)
        m = child.meta()

        assert m["model"] == "base-gen"

    def test_model_meta_after_with_args_reflects_overridden_kwargs(self, base_gen):
        """Derived model's meta() 'default_params' shows overridden kwargs."""
        child = base_gen.with_args(temperature=0.3)
        m = child.meta()

        assert m["default_params"].get("temperature") == 0.3


# ===================================================================
# TestModelExtra
# ===================================================================
class TestModelExtra:
    """extra: init, property, with_args propagation, meta() exposure."""

    def test_default_extra_is_empty(self):
        """Model without extra returns empty dict."""
        model = StubGenModel(model="test", api_key="fake")
        assert model.extra == {}

    def test_init_stores_extra(self):
        """Model with extra stores and returns it."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        assert model.extra == extra

    def test_with_args_preserves_extra(self):
        """with_args() without extra preserves the original."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        child = model.with_args(temperature=0.5)
        assert child.extra == extra

    def test_with_args_overrides_extra(self):
        """with_args(extra=...) replaces the original."""
        old = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        new = {"sequence_wrappers": {"rna": "<rna>{seq}</rna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=old)
        child = model.with_args(extra=new)
        assert child.extra == new

    def test_with_args_sets_extra_on_plain_model(self):
        """with_args(extra=...) on a model without extra sets it."""
        model = StubGenModel(model="test", api_key="fake")
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        child = model.with_args(extra=extra)
        assert child.extra == extra
        # Original unchanged
        assert model.extra == {}

    def test_meta_includes_extra(self):
        """meta() includes extra when configured."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        m = model.meta()
        assert m["extra"] == extra

    def test_meta_omits_extra_when_empty(self):
        """meta() omits extra key when no extra is configured."""
        model = StubGenModel(model="test", api_key="fake")
        m = model.meta()
        assert "extra" not in m

    def test_extra_not_in_kwargs(self):
        """extra must NOT leak into _kwargs (would be sent to API)."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        assert "extra" not in model._kwargs

    def test_extra_not_in_meta_default_params(self):
        """meta()['default_params'] must not contain extra."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        assert "extra" not in model.meta()["default_params"]

    def test_with_args_extra_not_in_child_kwargs(self):
        """with_args(extra=...) must not leak into child _kwargs."""
        model = StubGenModel(model="test", api_key="fake")
        child = model.with_args(
            extra={"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}},
            temperature=0.5,
        )
        assert "extra" not in child._kwargs
        assert child._kwargs.get("temperature") == 0.5
