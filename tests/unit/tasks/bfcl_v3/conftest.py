"""Shared rows for the BFCL v3 task family.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
import types

import pytest

from sieval.tasks.bfcl_v3._base import BfclV3LiveTask, BfclV3NonLiveTask
from tests.conftest import MockChatModel, MockDataset


@pytest.fixture
def simple_row():
    return {
        "id": "simple_0",
        "category": "simple",
        "language": "Python",
        "question": [{"role": "user", "content": "Area of a 10x5 triangle?"}],
        "function": json.dumps(
            [
                {
                    "name": "calculate_triangle_area",
                    "description": "Area of a triangle.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "base": {"type": "integer", "description": "base"},
                            "height": {"type": "integer", "description": "height"},
                        },
                        "required": ["base", "height"],
                    },
                }
            ]
        ),
        "ground_truth": json.dumps(
            [{"calculate_triangle_area": {"base": [10], "height": [5]}}]
        ),
    }


@pytest.fixture
def dotted_row(simple_row):
    """A function name containing a dot -- the FC/Prompt divergence probe."""
    row = dict(simple_row)
    functions = json.loads(row["function"])
    functions[0]["name"] = "geometry.triangle_area"
    row["function"] = json.dumps(functions)
    row["ground_truth"] = json.dumps(
        [{"geometry.triangle_area": {"base": [10], "height": [5]}}]
    )
    return row


@pytest.fixture
def java_row(simple_row):
    """A `java` row -- the cohort upstream's schema preprocessing rewrites.

    100 non-live rows are Java and 50 more are JavaScript; on those, upstream
    restates every parameter as a `string` before the schema reaches the model,
    because the model is being asked for Java source rather than for JSON.
    """
    row = dict(simple_row)
    row["id"] = "java_0"
    row["category"] = "java"
    row["language"] = "Java"
    functions = json.loads(row["function"])
    functions[0]["parameters"]["properties"]["tags"] = {
        "type": "ArrayList",
        "description": "tags",
        "items": {"type": "String"},
    }
    row["function"] = json.dumps(functions)
    return row


@pytest.fixture
def system_turn_row(simple_row):
    """A row that already opens with its own system turn.

    92 live rows do. Upstream merges the schema block into that turn; a port
    that prepends a second one instead sends a two-system-turn conversation no
    upstream run ever produced.
    """
    row = dict(simple_row)
    row["id"] = "live_simple_0"
    row["category"] = "live_simple"
    row["question"] = [
        {"role": "system", "content": "You are a geometry tutor."},
        {"role": "user", "content": "Area of a 10x5 triangle?"},
    ]
    return row


@pytest.fixture
def irrelevance_row(simple_row):
    row = dict(simple_row)
    row["id"] = "irrelevance_0"
    row["category"] = "irrelevance"
    row["ground_truth"] = None
    return row


# --------------------------------------------------------------------------
# The report. `report` reads one thing off a finalized context -- its judgement
# record -- so these stand in for one rather than driving a runner.
# --------------------------------------------------------------------------


def _ctx(category: str, correct: bool, index: int):
    return types.SimpleNamespace(
        feedback_result={
            "reference": None,
            "rollouts": [{"index": 0, "correct": correct}],
            "n_rollouts": 1,
            "n_correct": int(correct),
            "extra": {"category": category, "id": f"{category}_{index}"},
        }
    )


def _finals(spec: dict[str, tuple[int, int]]) -> list:
    """One context per sample that came back; the first *n_correct* are right.

    *spec* maps a category to `(n_returned, n_correct)`. A category returning
    fewer rows than it declares is how a pipeline failure reaches the report:
    the failed samples are simply absent, with nothing naming their category.
    """
    return [
        _ctx(category, index < correct, index)
        for category, (returned, correct) in spec.items()
        for index in range(returned)
    ]


def _group_task(cls):
    """A group base with no protocol mixin -- `report` reads neither hook."""
    return cls(MockDataset(), MockChatModel())


#: Every non-live category complete, and the three `simple_ast` members at
#: DIFFERENT rates: a pooled rate over their 550 rows is 72.7, an unweighted
#: mean of the three is 33.3, so the two rollup rules cannot be confused.
_NON_LIVE_FULL = {
    "simple": (400, 400),
    "multiple": (200, 200),
    "parallel": (200, 200),
    "parallel_multiple": (200, 200),
    "java": (100, 0),
    "javascript": (50, 0),
    "irrelevance": (240, 240),
}

#: Every live category complete, at six rates that are deliberately unequal and
#: none of them 0 or 1. Unequal, so a weighted rollup (63.3) and an unweighted
#: one (41.1) are far apart; strictly interior, so every interval comes off the
#: Wilson branch rather than a saturated-set fallback.
_LIVE_SPREAD = {
    "live_simple": (258, 129),
    "live_multiple": (1053, 843),
    "live_parallel": (16, 4),
    "live_parallel_multiple": (24, 6),
    "live_irrelevance": (882, 441),
    "live_relevance": (18, 3),
}


@pytest.fixture
async def non_live_report():
    return await _group_task(BfclV3NonLiveTask).report(_finals(_NON_LIVE_FULL), [])


@pytest.fixture
async def non_live_report_short():
    """One of `simple`'s 400 rows never came back, and no other category ran."""
    return await _group_task(BfclV3NonLiveTask).report(
        _finals({"simple": (399, 399)}), []
    )


@pytest.fixture
async def non_live_report_with_fails():
    """The same run, with the missing row present as a FAILED sample."""
    return await _group_task(BfclV3NonLiveTask).report(
        _finals({"simple": (399, 399)}), [object()]
    )


@pytest.fixture
async def live_report():
    return await _group_task(BfclV3LiveTask).report(_finals(_LIVE_SPREAD), [])
