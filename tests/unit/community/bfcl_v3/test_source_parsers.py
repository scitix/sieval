"""`ast_parse` must actually dispatch to the vendored source-text parsers.

The identity test proves the vendored bytes have not drifted; it says nothing
about whether the Java/JavaScript branches run. These assert real behaviour
through `ast_parse`'s public surface, covering all three branches (Python,
Java, JavaScript) plus the unsupported-language error path.

Expected values were derived by running the parsers, not assumed: both
tree-sitter parsers return every argument value as a string (`"true"`, not
`True`), which is upstream's behaviour, not a porting artifact.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest

from sieval.community.bfcl_v3 import ast_parse


def test_java_source_is_parsed_into_a_call_dict():
    assert ast_parse(
        '[TaskManager.removeTask(task="Meeting", urgent=true)]', "Java"
    ) == [{"TaskManager.removeTask": {"task": "Meeting", "urgent": "true"}}]


def test_javascript_source_is_parsed_into_a_call_dict():
    assert ast_parse('[doTask(name="Meeting", urgent=true)]', "JavaScript") == [
        {"doTask": {"name": "Meeting", "urgent": "true"}}
    ]


def test_python_source_is_still_parsed_by_ast():
    assert ast_parse('[doTask(name="Meeting", urgent=True)]', "Python") == [
        {"doTask": {"name": "Meeting", "urgent": True}}
    ]


def test_unknown_language_still_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        ast_parse('[doTask(name="Meeting")]', "Ruby")
