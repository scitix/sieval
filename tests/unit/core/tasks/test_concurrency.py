"""
Unit tests for sieval/core/tasks/concurrency.py.

Covers: compute_stream_buffer_capacity, prepare_limiters (_validate_limit),
get_limiter_for, min_limit.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from typing import Literal

import anyio
import pytest

from sieval.core.tasks.concurrency import (
    compute_stream_buffer_capacity,
    get_limiter_for,
    min_limit,
    prepare_limiters,
)
from sieval.core.tasks.consts import TaskAction


# compute_stream_buffer_capacity
# ===================================================================
class TestComputeStreamBufferCapacity:
    def test_minimum_floor_and_type(self):
        cap = compute_stream_buffer_capacity(
            record_each_stage=False, global_limit=None, per_stage={}
        )
        assert cap >= 32
        assert isinstance(cap, int)

    def test_record_each_stage_increases_capacity(self):
        cap_no = compute_stream_buffer_capacity(
            record_each_stage=False, global_limit=64, per_stage={}
        )
        cap_yes = compute_stream_buffer_capacity(
            record_each_stage=True, global_limit=64, per_stage={}
        )
        assert cap_yes > cap_no

    def test_feedback_stage_amplifies_capacity(self):
        cap_without = compute_stream_buffer_capacity(
            record_each_stage=False, global_limit=64, per_stage={}
        )
        cap_with = compute_stream_buffer_capacity(
            record_each_stage=False, global_limit=64, per_stage={"feedback": 32}
        )
        assert cap_with > cap_without

    def test_no_global_limit_uses_per_stage_max(self):
        cap = compute_stream_buffer_capacity(
            record_each_stage=False, global_limit=None, per_stage={"infer": 100}
        )
        assert cap >= 32


# ===================================================================
# prepare_limiters
# ===================================================================
class TestPrepareLimiters:
    @pytest.mark.anyio
    async def test_global_limit_and_empty_cases(self):
        global_limiter, stage_limiters = prepare_limiters(None, {})
        assert global_limiter is None
        assert stage_limiters == {}

        global_limiter, _ = prepare_limiters(32, {})
        assert global_limiter is not None
        assert global_limiter.total_tokens == 32

    @pytest.mark.anyio
    async def test_stage_limits_create_limiters_for_known_actions(self):
        _, stage_limiters = prepare_limiters(None, {"infer": 16, "feedback": 8})
        assert "infer" in stage_limiters
        assert stage_limiters["infer"].total_tokens == 16
        assert "feedback" in stage_limiters

    @pytest.mark.anyio
    async def test_stage_limit_effective_min(self):
        """Effective stage limit = min(stage, global)."""
        for global_limit, stage_limit, expected in [(10, 50, 10), (100, 20, 20)]:
            _, stage_limiters = prepare_limiters(global_limit, {"infer": stage_limit})
            assert stage_limiters["infer"].total_tokens == expected

    def test_invalid_limits_raise(self):
        for global_limit, stage_limits in [(0, {}), (None, {"infer": 0})]:
            with pytest.raises(ValueError, match=">=1"):
                prepare_limiters(global_limit, stage_limits)

    @pytest.mark.anyio
    async def test_unknown_stage_key_ignored(self):
        """Keys not in TaskAction enum values are silently skipped."""
        _, stage_limiters = prepare_limiters(None, {"unknown_stage": 16})
        assert "unknown_stage" not in stage_limiters


# ===================================================================
# get_limiter_for
# ===================================================================
class TestGetLimiterFor:
    @pytest.mark.anyio
    async def test_none_action_returns_empty_composite(self):
        composite = get_limiter_for(None, None, {})
        # Empty composite — should not raise
        async with composite:
            pass

    @pytest.mark.anyio
    async def test_composite_limiter_combinations(self):
        # (
        #   action,
        #   has_global_limiter,
        #   has_stage_limiter,
        #   expect_global_borrowed_inside_context,
        #   expect_stage_borrowed_inside_context,
        # )
        cases: list[tuple[TaskAction | Literal["infer"], bool, bool, bool, bool]] = [
            # Only global limiter applies.
            (TaskAction.INFER, True, False, True, False),
            # Only stage limiter applies.
            (TaskAction.INFER, False, True, False, True),
            # Both limiters apply for infer.
            (TaskAction.INFER, True, True, True, True),
            # String action alias should behave like TaskAction.INFER.
            ("infer", False, True, False, True),
            # No stage limiter for postprocess in this stage map.
            (TaskAction.POSTPROCESS, True, False, True, False),
        ]
        for (
            action,
            has_global,
            has_stage,
            expect_global_borrowed,
            expect_stage_borrowed,
        ) in cases:
            global_lim = anyio.CapacityLimiter(10) if has_global else None
            stage_lim = anyio.CapacityLimiter(4) if has_stage else None
            stage_map = (
                {"infer": stage_lim} if has_stage and stage_lim is not None else {}
            )

            composite = get_limiter_for(action, global_lim, stage_map)
            async with composite:
                if global_lim is not None:
                    assert (global_lim.borrowed_tokens == 1) is expect_global_borrowed
                if stage_lim is not None:
                    assert (stage_lim.borrowed_tokens == 1) is expect_stage_borrowed
            if global_lim is not None:
                assert global_lim.borrowed_tokens == 0


# ===================================================================
# min_limit
# ===================================================================
class TestMinLimit:
    def test_min_limit_cases(self):
        # (lhs_limit, rhs_limit, expected_min_limit)
        cases = [
            # Both unset -> unset.
            (None, None, None),
            # One unset -> the other side.
            (None, 10, 10),
            (5, None, 5),
            # Both set -> numeric min.
            (3, 7, 3),
            (7, 3, 3),
            # Equal limits -> same value.
            (4, 4, 4),
        ]
        for a, b, expected in cases:
            assert min_limit(a, b) == expected
