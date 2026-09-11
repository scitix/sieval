"""BFCL v3 (single-turn) grader, vendored from ShishirPatil/gorilla.

Source: https://github.com/ShishirPatil/gorilla
Revision: v1.3 (ea13468e4423454d0c213704fb87cf7cb3990433)
License: Apache-2.0

Upstream's evaluation surface is `bfcl_eval`, which imports its own submodules
by absolute path. A vendored copy cannot keep those imports, so rewriting them
is the only edit made to any vendored file -- with one exception: `parser.py`'s
`ast.BinOp` branch calls `_safe_eval.safe_eval` where upstream `eval`s a
model-authored expression. Each vendored module records upstream's blob SHA at
the pin and its exact edits, so the copy can be re-derived from a fresh checkout.

Whole-file: `ast_checker.py`, `type_mappings.py`, `type_convertor/*.py`,
`source_parser/*.py`, and `prompts.py` (upstream's `constants/default_prompts`).
Partial -- named subsets of a larger upstream file, listed in each module's own
docstring: `parser.py`, `tool_convert.py`, `output_checks.py`, `preprocess.py`,
`aggregate.py`.

First-party: `_model_config.py` supplies the `MODEL_CONFIG_MAPPING` global
`ast_checker.convert_func_name` reads, since upstream's 2044-line registry has no
entry for sieval's models. `_tables.py` holds the category tables shared by the
dataset loader and the tasks, neither of which may reach into the other's private
module. `_safe_eval.py` is the deviation `parser.py` carries, held outside that
file so the boundary stays lintable -- vendored code is not.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import TYPE_CHECKING

from ._model_config import set_underscore_to_dot
from ._tables import (
    CALL_EXPECTED,
    GOLDLESS_CATEGORIES,
    LANGUAGE_BY_CATEGORY,
    LIVE_COUNTS,
    NON_LIVE_COUNTS,
)
from .aggregate import calculate_unweighted_accuracy, calculate_weighted_accuracy
from .ast_checker import ast_checker
from .output_checks import is_empty_output, is_function_calling_format_output
from .preprocess import (
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from .prompts import DEFAULT_SYSTEM_PROMPT
from .tool_convert import convert_to_tool

if TYPE_CHECKING:
    # A module `__getattr__` types *every* attribute of the package as its own
    # return type, so `ast_parse` would resolve to `object` and every call site
    # would read as calling a non-callable. An explicitly declared symbol takes
    # precedence over the fallback; `TYPE_CHECKING` is False at runtime, so the
    # import below stays deferred.
    from .parser import ast_parse


__all__ = [
    "ast_checker",
    "ast_parse",
    "convert_to_tool",
    "is_function_calling_format_output",
    "is_empty_output",
    "func_doc_language_specific_pre_processing",
    "system_prompt_pre_processing_chat_model",
    "calculate_weighted_accuracy",
    "calculate_unweighted_accuracy",
    "DEFAULT_SYSTEM_PROMPT",
    "set_underscore_to_dot",
    "NON_LIVE_COUNTS",
    "LIVE_COUNTS",
    "GOLDLESS_CATEGORIES",
    "CALL_EXPECTED",
    "LANGUAGE_BY_CATEGORY",
]


def __getattr__(name: str) -> object:
    """`ast_parse` is deferred because `parser.py` is the only module here that
    reaches tree-sitter, which lives in the optional `bfcl-v3` extra. Importing
    it eagerly makes that extra a hard requirement for importing this package at
    all -- and `sieval/datasets/` imports it for two pure-data tables, so the
    whole dataset registry died on a base install.
    """
    if name == "ast_parse":
        from .parser import ast_parse

        return ast_parse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
