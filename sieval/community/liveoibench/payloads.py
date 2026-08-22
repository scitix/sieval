# adapted from https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/process_dataset.py
# (_normalize_code_mapping, and the subtasks parse in write_problem_assets)
"""Parsing the JSON-in-a-column payloads the LiveOIBench parquets carry.

Divergence from upstream: upstream decodes these while writing files to disk and
raises ``DatasetError`` on a malformed subtasks blob; here the same decoding
happens at load time. A malformed *code* bundle still reads as "this problem
ships none", which is upstream's behaviour and the common case — 362 of 403
problems have no grader bundle at all.
"""

import json
import math


def load_subtasks(raw_subtasks: str | None) -> dict[str, dict]:
    """Parse the ``subtasks`` column into upstream's ``{id: subtask}`` mapping."""
    parsed = json.loads(raw_subtasks or "{}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Malformed subtasks payload: {type(parsed).__name__}")
    return parsed


def load_code_bundle(raw_value: object) -> dict[str, str]:
    """Parse a ``grader_code`` / ``starter_code`` cell into filename → contents."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return {}
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    if not isinstance(raw_value, str):
        return {}
    stripped = raw_value.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(name): str(contents) for name, contents in parsed.items()}
