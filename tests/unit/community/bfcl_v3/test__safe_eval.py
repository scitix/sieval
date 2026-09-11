"""Cover the guard that replaces `eval` in the BFCL v3 argument resolver.

Two properties, and neither is worth much alone: the guard must refuse what
executes (or it is not a guard) *and* agree with upstream on everything that
does not (or it is a scoring change wearing a safety label).

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import ast

import pytest

from sieval.community.bfcl_v3._safe_eval import MAX_RESULT_SIZE, safe_eval
from sieval.community.bfcl_v3.parser import ast_parse, resolve_ast_by_type


def _argument(source: str) -> ast.expr:
    """The `x=` argument node of a one-call snippet, as the resolver sees it."""
    return ast.parse(source, mode="eval").body.keywords[0].value  # ty: ignore


def test_a_payload_in_an_argument_does_not_run(tmp_path):
    """The whole point, observed as a side effect rather than as a refusal.

    Asserting only that a `ValueError` comes back would pass against a guard
    that refuses *after* evaluating. The marker file is what distinguishes
    "refused" from "ran, then complained": under the `eval` this replaces, the
    file exists and the decoder returns an ordinary-looking number.
    """
    marker = tmp_path / "payload-ran"
    source = f"f(x=__import__('pathlib').Path({str(marker)!r}).write_text('x') + 0)"

    with pytest.raises(ValueError, match="refusing to evaluate Call"):
        resolve_ast_by_type(_argument(source))

    assert not marker.exists(), "the payload executed before being refused"


def test_the_decoder_refuses_rather_than_executing_a_hostile_reply(tmp_path):
    """Through the real entry point, because that is what `postprocess` calls.

    The refusal has to surface as a decode failure -- an exception out of
    `ast_parse` -- so the sample scores wrong. A reply that puts executable code
    in an argument slot did not produce the function call it was asked for.
    """
    marker = tmp_path / "payload-ran"
    reply = f"[f(x=__import__('pathlib').Path({str(marker)!r}).write_text('x') + 0)]"

    with pytest.raises(ValueError, match="refusing"):
        ast_parse(reply, "Python")

    assert not marker.exists()


#: Every shape upstream's `eval(ast.unparse(node))` computes without executing
#: anything. The expected values are upstream's, so a divergence shows up as a
#: wrong value rather than as a refusal.
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("f(x=1 + 2)", 3),
        ("f(x=60 * 60)", 3600),
        ("f(x=10 / 4)", 2.5),
        ("f(x=7 // 2)", 3),
        ("f(x=7 % 3)", 1),
        ("f(x=2 ** 10)", 1024),
        ("f(x=-3 + 1)", -2),
        ("f(x='a' + 'b')", "ab"),
        ("f(x=[1] + [2])", [1, 2]),
        ("f(x=(1,) + (2,))", (1, 2)),
        ("f(x=1 << 4)", 16),
        ("f(x=6 & 3)", 2),
        ("f(x=6 | 3)", 7),
        ("f(x=6 ^ 3)", 5),
        ("f(x=1.5 * 2)", 3.0),
        ("f(x=(1 + 2) * (3 - 1))", 6),
        ("f(x=[1 + 1, 2 * 2])", [2, 4]),
        ("f(x={'a': 1 + 1})", {"a": 2}),
    ],
)
def test_a_non_executing_expression_keeps_upstreams_value(source, expected):
    node = _argument(source)
    result = resolve_ast_by_type(node)
    assert result == expected
    # `1 == 1.0` and `True == 1`, so equality alone would let a type change
    # through -- and the checker compares types.
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    "source",
    [
        "f(x=__import__('os').getpid() + 0)",
        "f(x=open('/etc/passwd').read() + '')",
        "f(x=[].__class__ + 0)",
        "f(x=undefined_name + 1)",
        "f(x=(lambda: 1)() + 1)",
        "f(x=[i for i in range(3)] + [])",
        # The spread must sit inside a BinOp to reach this guard at all: a bare
        # `{**a}` argument is handled by upstream's own `ast.Dict` branch, which
        # refuses it a different way (`Exception: Unsupported AST type`).
        "f(x={**{'a': 1}} | {'b': 2})",
        "f(x=f'a' + 'b')",
    ],
)
def test_an_expression_that_could_name_anything_is_refused(source):
    with pytest.raises(ValueError, match="refusing"):
        resolve_ast_by_type(_argument(source))


@pytest.mark.parametrize(
    "source",
    [
        "f(x=9 ** 9 ** 9)",  # 369 million digits
        "f(x='a' * 10 ** 9)",  # a gigabyte of string
        "f(x=1 << 10 ** 9)",
        "f(x=[0] * 10 ** 9)",
    ],
)
def test_an_unbounded_expression_is_refused_before_it_is_computed(source):
    """A hang here is the failure, not an error.

    Decoding runs inline on the session's event loop with no timeout around it,
    so computing any of these would stall every other sample in the run. The
    test can only assert that it returns at all -- which it does only because
    the cost is screened before the operator runs.
    """
    refused = "refusing an expression|refusing an intermediate"
    with pytest.raises(ValueError, match=refused):
        resolve_ast_by_type(_argument(source))


@pytest.mark.parametrize(
    "source",
    [
        "f(x='%.400000000f' % 1.0)",  # ~400MB of string from two tiny operands
        "f(x='%.20f' % 1.0)",  # 22 chars -- admissible by SIZE, refused anyway
        "f(x=b'%.400000000f' % 1.0)",
    ],
)
def test_percent_formatting_on_a_string_is_refused_outright(source):
    """Refused by SHAPE, not by size -- which is the whole point.

    A precision field sets the result size independently of both operands, so
    there is nothing for `_refuse_if_unbounded` to measure and `_bounded` would
    see the value only after the allocation it exists to prevent. The 22-char
    case is the discriminating one: it is far under `MAX_RESULT_SIZE`, so a
    post-hoc size check admits it. Only a categorical refusal rejects it, and
    only a categorical refusal bounds the 400MB sibling before it is built.
    """
    with pytest.raises(ValueError, match="refusing `%` formatting"):
        resolve_ast_by_type(_argument(source))


def test_integer_modulo_is_still_admissible():
    """`%` on integers is bounded by its right operand, so it keeps working."""
    assert safe_eval(_argument("f(x=10 % 3)")) == 1
    assert safe_eval(_argument("f(x=10 ** 100 % 7)")) == 10**100 % 7


def test_the_bound_admits_far_more_than_any_real_argument():
    """Evidence the cap does not bind: the pinned largest literal is 13 digits."""
    assert safe_eval(_argument("f(x=2 ** 4096)")) == 2**4096
    assert safe_eval(_argument("f(x=1617262800000 * 2)")) == 3234525600000
    assert MAX_RESULT_SIZE >= 1 << 20


def test_the_lambda_branch_still_cannot_reach_its_eval():
    """Upstream's second `eval` is dead, and this fails if it stops being dead.

    `ast.Lambda.body` is a single expression node, so upstream's
    `value.body[0].value` raises while the argument to `eval` is still being
    built. That is why the branch was left byte-identical: hardening a line that
    cannot execute buys nothing. If someone later "repairs" the subscript, this
    test fails -- which is the point, because the repair would install a live
    `eval` over model output.
    """
    with pytest.raises(TypeError, match="not subscriptable"):
        resolve_ast_by_type(_argument("f(x=lambda a: a + 1)"))
