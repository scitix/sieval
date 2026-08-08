"""
Unit tests for the shared sympy execution guards.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from sympy.parsing.sympy_parser import parse_expr

from sieval.community._sympy_guards import (
    MAX_EXPONENT,
    evaluable,
    quotes_free,
    sympy_globals,
)


def test_sympy_globals_has_no_builtins():
    assert sympy_globals()["__builtins__"] == {}


def test_sympy_globals_keeps_the_sympy_dialect():
    """Clearing builtins must not narrow what a legitimate answer may say.

    `auto_symbol` rewrites an unknown callable into `Function(...)`, so a
    namespace holding only the answer aliases fails every real `sin(pi*x/5)`.
    """
    ns = sympy_globals()
    for name in ("Function", "sin", "pi", "sqrt", "Symbol"):
        assert name in ns
    assert parse_expr("sin(pi*x/5)", global_dict=sympy_globals()) is not None


def test_cleared_namespace_blocks_the_direct_payload(tmp_path):
    target = tmp_path / "direct"
    with pytest.raises(AttributeError):
        parse_expr(
            f"__import__('os').system('touch {target}')", global_dict=sympy_globals()
        )
    assert not target.exists()


def test_cleared_namespace_alone_is_not_enough(tmp_path):
    """Pins why `quotes_free` exists: a nested parse gets the builtins back.

    If this ever stops executing, the quote screen has become unnecessary and
    the restriction it imposes should be revisited rather than kept on faith.
    """
    target = tmp_path / "nested"
    payload = f"eval('__import__(\\'os\\').system(\\'touch {target}\\')')"
    parse_expr(payload, global_dict=sympy_globals())
    assert target.exists()
    assert not quotes_free(payload)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("x**2 + 1", True),
        ("sin(pi/6)", True),
        ("eval('x')", False),
        ('eval("x")', False),
        ("'quoted'", False),
    ],
)
def test_quotes_free(text, expected):
    assert quotes_free(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2 + 2", True),  # no `**` at all — early return
        ("x**2", True),
        ("(x**2)**3", True),  # left-nested is fine
        (f"2**{MAX_EXPONENT - 1}", True),
        ("9**9**9", False),  # right-nested tower
        (f"2**{MAX_EXPONENT + 1}", False),
    ],
)
def test_evaluable(text, expected):
    assert evaluable(text) is expected


def test_evaluable_defaults_match_bare_parse_expr():
    """Called with no local/transformations, as deepseek_math does."""
    assert evaluable("x**2 + 1") is True
    assert evaluable("9**9**9") is False
