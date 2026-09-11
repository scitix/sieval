"""Vendored verbatim from ShishirPatil/gorilla @ v1.3 (ea13468e).

Upstream path: bfcl_eval/model_handler/utils.py
Upstream blob: f78acc240bee3c43b18de3d4195fbf6ee1a207f1
License: Apache-2.0

Symbols taken (bodies unchanged): `ast_parse`, `resolve_ast_call`,
`resolve_ast_by_type`. This is a partial extraction, not a whole-file copy --
the rest of `model_handler/utils.py` (prompt construction, decoders for other
model styles, retry helpers) is out of scope for this port.

Deviations:
1. The module needs no `bfcl_eval.*` imports at all for these three
   functions (`import ast` is stdlib), so their bodies needed no rewrite.
2. Upstream's `utils.py:13-14` imports `parse_java_function_call` and
   `parse_javascript_function_call` from `bfcl_eval.model_handler.parser.*`.
   Rewritten to `from .source_parser.java_parser import
   parse_java_function_call` and the `js_parser` twin -- same rename as
   `source_parser/`'s own docstrings explain (this package's `parser.py`
   already occupies upstream's directory name).
3. `resolve_ast_by_type`'s `ast.BinOp` branch calls `safe_eval` where upstream
   calls `eval(ast.unparse(value))`. That node is model output, so upstream's
   line executes model-authored code -- `f(x=__import__('os').system('...')
   + 0)` runs the call and hands back an ordinary-looking number. `safe_eval`
   computes the same value for every expression that does not execute
   something, and refuses the rest; the argument for where it lives, and for
   the non-executing shapes it also refuses, is in `_safe_eval.py`.

   The `ast.Lambda` branch below keeps upstream's `eval` untouched, because it
   cannot reach it: `Lambda.body` is a single expression node, so `value.body[0]`
   raises `TypeError` while the argument is still being built. Hardening a
   branch that cannot execute would be a deviation bought for nothing, and the
   raise it produces today is already the decode failure upstream scores.

The Java/JavaScript branches of `ast_parse` dispatch to those two functions:
a `java`/`javascript`-category sample's model output is Java/JavaScript
source text, not Python call syntax, so `ast_parse` parses it with the
matching tree-sitter grammar rather than `ast.parse`.
"""

import ast

from ._safe_eval import safe_eval
from .source_parser.java_parser import parse_java_function_call
from .source_parser.js_parser import parse_javascript_function_call


def ast_parse(input_str: str, language: str="Python") -> list[dict]:
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    elif language == "Java":
        return parse_java_function_call(
            input_str[1:-1]
        )  # Remove the [ and ] from the string
    elif language == "JavaScript":
        return parse_javascript_function_call(input_str[1:-1])
    else:
        raise NotImplementedError(f"Unsupported language: {language}")


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = safe_eval(value)  # DEVIATION 3 -- upstream: eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output
