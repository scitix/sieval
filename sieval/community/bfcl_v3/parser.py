"""Vendored verbatim from ShishirPatil/gorilla @ v1.3 (ea13468e).

Upstream path: bfcl_eval/model_handler/utils.py
Upstream blob: f78acc240bee3c43b18de3d4195fbf6ee1a207f1
License: Apache-2.0

Symbols taken (bodies unchanged): `ast_parse`, `resolve_ast_call`,
`resolve_ast_by_type`. This is a partial extraction, not a whole-file copy --
the rest of `model_handler/utils.py` (prompt construction, decoders for other
model styles, retry helpers) is out of scope for this port.

Only deviation: the module needs no `bfcl_eval.*` imports at all for these
three functions (`import ast` is stdlib), so there was nothing to rewrite.

Note on the Java/JavaScript branches of `ast_parse`: upstream dispatches to
`bfcl_eval.model_handler.parser.java_parser.parse_java_function_call` and the
`js_parser` twin, neither of which is vendored here. Their bodies are kept
verbatim (unreachable rather than deleted, per the "removing a helper is a
deviation too" rule), because sieval never calls `ast_parse` with
`language="Java"` or `"JavaScript"` -- the category's language only ever
selects the type converter inside `ast_checker`, not the parser's language
(see `_tables.py`'s `LANGUAGE_BY_CATEGORY` docstring). Calling `ast_parse`
with either of those two language values would raise `NameError` on the
undefined parser call, exactly as it would if the names were simply typos;
that failure mode is intentional rather than silent, since it is unreachable
from the intended call path.
"""

import ast


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
        output = eval(ast.unparse(value))
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
