"""Shared rows for the BFCL v3 task family.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json

import pytest


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
def irrelevance_row(simple_row):
    row = dict(simple_row)
    row["id"] = "irrelevance_0"
    row["category"] = "irrelevance"
    row["ground_truth"] = None
    return row
