"""Vendored verbatim from ShishirPatil/gorilla @ v1.3 (ea13468e).

Upstream path: bfcl_eval/model_handler/utils.py
Upstream blob: f78acc240bee3c43b18de3d4195fbf6ee1a207f1
License: Apache-2.0

Symbols taken (bodies unchanged): `ast_parse`, `resolve_ast_call`,
`resolve_ast_by_type`. A partial extraction -- the rest of
`model_handler/utils.py` (prompt construction, other model styles' decoders,
retry helpers) is out of scope.

Deviations:
1. The two source-parser imports are rewritten to `.source_parser.*`, renamed
   from upstream's `parser/` because this file already occupies that name.
   Nothing else needed rewriting: these three functions reach no `bfcl_eval.*`.
2. `resolve_ast_by_type`'s `ast.BinOp` branch calls `safe_eval` where upstream
   calls `eval(ast.unparse(value))`. That node is model output, so upstream's
   line executes model-authored code -- `f(x=__import__('os').system('...')
   + 0)` runs the call and hands back an ordinary-looking number. `safe_eval`
   computes the same value for every expression that does not execute
   something; what it refuses, and why it lives outside this file, is in
   `_safe_eval.py`.

   The `ast.Lambda` branch keeps upstream's `eval` because it cannot reach it:
   `Lambda.body` is a single expression node, so `value.body[0]` raises
   `TypeError` first -- and that raise is already the decode failure upstream
   scores. Hardening an unreachable branch buys nothing.

`ast_parse` dispatches `java`/`javascript` rows to the tree-sitter parsers
rather than `ast.parse`: their model output is source text, not Python call
syntax.
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
