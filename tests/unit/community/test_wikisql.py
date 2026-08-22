"""Vendored WikiSQL harness: byte-identity, engine semantics, index guard.

The engine cases are hand-computed against upstream's documented behaviour
(``lib/table.py`` lowercases string cells on INSERT; ``lib/dbengine.py``
lowercases string condition values on SELECT and coerces against a ``real``
column through babel), so a regression shows up as a wrong *value* rather than
as a crash.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import hashlib
import pathlib

import pytest

import sieval.community.wikisql as wikisql_pkg
from sieval.community.wikisql import (
    DBEngine,
    InvalidQueryIndex,
    Query,
    detokenize,
    get_id,
)

# --- byte-identity with upstream ------------------------------------------

#: Resolved from the imported package, not from the CWD: a relative path would
#: make the identity tests silently depend on where pytest was invoked, and the
#: whole point of them is that they cannot be satisfied by accident.
_VENDOR_DIR = pathlib.Path(wikisql_pkg.__file__).parent

#: sha256 of upstream's `lib/query.py` at the pinned commit
#: (cffb423077756d04c1bac5bcd45167c86903fbcb).
_UPSTREAM_QUERY_SHA = "f539150bea6cd07a5dca226abcced6f9d356d216f5c3d70107693613f1fbeb25"
#: sha256 of upstream's `lib/common.py` at the same commit.
_UPSTREAM_COMMON_SHA = (
    "21079d6e99246eb9bfa8689b7548af9747f1b96c71e45be0522038a3486fde98"
)

_PATCHED_IMPORT = "from .common import detokenize"
_UPSTREAM_IMPORT = "from lib.common import detokenize"


def test_query_py_is_upstream_verbatim_but_for_the_import():
    """`query.py` must differ from upstream in exactly one line.

    Asserted by hash rather than by eye: a later "tidy-up" of the vendored
    scorer is a silent scoring change, and reverting the one adaptation and
    re-hashing is the only check that notices. `Query.__eq__` IS the
    lf_accuracy metric, so this is the metric's definition being pinned.
    """
    text = (_VENDOR_DIR / "query.py").read_text()
    assert text.count(_PATCHED_IMPORT) == 1
    assert _UPSTREAM_IMPORT not in text
    reverted = text.replace(_PATCHED_IMPORT, _UPSTREAM_IMPORT)
    assert hashlib.sha256(reverted.encode()).hexdigest() == _UPSTREAM_QUERY_SHA


def test_common_py_is_upstream_verbatim():
    text = (_VENDOR_DIR / "common.py").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == _UPSTREAM_COMMON_SHA


def test_upstream_op_tables_are_unchanged():
    """The index encodings the prompt teaches and the engine indexes."""
    assert Query.agg_ops == ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    assert Query.cond_ops == ["=", ">", "<", "OP"]


def test_detokenize_still_reachable():
    """Kept because upstream ships it; `query.py` imports it at module scope."""
    assert detokenize({"gloss": ["a", "b"], "after": [" ", ""]}) == "a b"


# --- Query equality is the lf_accuracy metric ------------------------------


def test_conditions_compare_unordered_by_default():
    """Upstream's default; the leaderboard marks the ordered reading with `*`."""
    a = Query.from_dict({"sel": 1, "agg": 0, "conds": [[0, 0, "x"], [2, 1, "y"]]})
    b = Query.from_dict({"sel": 1, "agg": 0, "conds": [[2, 1, "y"], [0, 0, "x"]]})
    assert a == b


def test_ordered_reading_is_order_sensitive():
    a = Query.from_dict(
        {"sel": 1, "agg": 0, "conds": [[0, 0, "x"], [2, 1, "y"]]}, ordered=True
    )
    b = Query.from_dict(
        {"sel": 1, "agg": 0, "conds": [[2, 1, "y"], [0, 0, "x"]]}, ordered=True
    )
    assert a != b


def test_condition_values_compare_case_insensitively_and_across_types():
    """`str(cond).lower()` on both sides, so 1998 and "1998" are equal.

    Load-bearing: gold condition values are str, int AND float depending on the
    row, so a type-sensitive comparison would fail rows nobody got wrong.
    """
    a = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, 1998]]})
    b = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, "1998"]]})
    c = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, "TERRENCE"]]})
    d = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, "terrence"]]})
    assert a == b
    assert c == d


def test_query_never_equals_none():
    """Upstream's `qp == qg` with `qp` None, when a prediction is unusable.

    `==` and this operand order are both deliberate: that is the expression
    `evaluate.py` evaluates, and it must come out False rather than raising.
    `__eq__` returns NotImplemented both ways, so Python falls back to identity.
    """
    unusable = None  # what upstream leaves `qp` as
    q = Query.from_dict({"sel": 0, "agg": 0, "conds": []})
    assert (unusable == q) is False


# --- the engine ------------------------------------------------------------

_HEADER = ["Player", "Nationality", "Points"]
_TYPES = ["text", "text", "real"]
_ROWS = [
    ["Terrence Ross", "United States", 12],
    ["Jose Calderon", "Spain", 8],
    ["Chris Bosh", "United States", 22],
]


def _engine() -> DBEngine:
    return DBEngine.from_table("1-234-5", _TYPES, _ROWS)


def test_get_id_matches_upstream_naming():
    assert get_id("1-10015132-16") == "table_1_10015132_16"


def test_select_with_text_condition_is_case_insensitive():
    """Cells are lowercased on INSERT and values on SELECT — both sides."""
    with _engine() as engine:
        q = Query.from_dict({"sel": 1, "agg": 0, "conds": [[0, 0, "TERRENCE ROSS"]]})
        assert engine.execute_query("1-234-5", q) == ["united states"]


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        (0, [12, 8, 22]),  # no aggregation
        (1, [22]),  # MAX
        (2, [8]),  # MIN
        (3, [3]),  # COUNT
        (4, [42]),  # SUM
        (5, [14]),  # AVG
    ],
)
def test_every_aggregation_operator(agg, expected):
    """All six appear in the gold data, so all six must run."""
    with _engine() as engine:
        q = Query.from_dict({"sel": 2, "agg": agg, "conds": []})
        assert engine.execute_query("1-234-5", q) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [(0, [12]), (1, [22]), (2, [8])],
)
def test_comparison_operators_on_a_real_column(op, expected):
    with _engine() as engine:
        q = Query.from_dict({"sel": 2, "agg": 0, "conds": [[2, op, 12]]})
        assert engine.execute_query("1-234-5", q) == expected


def test_string_value_against_a_real_column_is_coerced_through_babel():
    """Upstream's `parse_decimal` path — it fires on 414 gold test conditions."""
    with _engine() as engine:
        q = Query.from_dict({"sel": 0, "agg": 0, "conds": [[2, 0, "12"]]})
        assert engine.execute_query("1-234-5", q) == ["terrence ross"]


def test_thousands_separated_value_against_a_real_column():
    """`parse_decimal` is why "1,234" works where `float()` would raise."""
    with DBEngine.from_table("t", ["real"], [[1234]]) as engine:
        q = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, "1,234"]]})
        assert engine.execute_query("t", q) == [1234]


def test_unparseable_value_against_a_real_column_falls_back_to_the_digit_regex():
    """Upstream's second chance: `num_re` after NumberFormatError."""
    with DBEngine.from_table("t", ["real"], [[7]]) as engine:
        q = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, "~7 pts"]]})
        assert engine.execute_query("t", q) == [7]


def test_no_conditions_selects_every_row():
    """131 gold test rows have an empty `conds`."""
    with _engine() as engine:
        q = Query.from_dict({"sel": 1, "agg": 0, "conds": []})
        assert engine.execute_query("1-234-5", q) == [
            "united states",
            "spain",
            "united states",
        ]


def test_a_condition_matching_nothing_returns_an_empty_list():
    with _engine() as engine:
        q = Query.from_dict({"sel": 1, "agg": 0, "conds": [[0, 0, "nobody"]]})
        assert engine.execute_query("1-234-5", q) == []


def test_two_conditions_on_one_column_keep_upstream_s_last_value_wins():
    """An upstream quirk, preserved deliberately.

    `where_map` is keyed by column, so a second condition on the same column
    overwrites the first value while both clauses stay in the SQL — the query
    becomes `col0 = :col0 AND col0 > :col0`. Reproducing it matters because
    scoring must match upstream's, quirks included.
    """
    with DBEngine.from_table("t", ["real"], [[5], [9]]) as engine:
        q = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 0, 5], [0, 1, 1]]})
        # Both clauses bind 1: `col0 = 1 AND col0 > 1` matches nothing.
        assert engine.execute_query("t", q) == []


# --- the index guard (the one behavioural divergence) ----------------------


@pytest.mark.parametrize(
    ("field", "form"),
    [
        ("sel", {"sel": "0", "agg": 0, "conds": []}),
        ("sel", {"sel": 1.0, "agg": 0, "conds": []}),
        ("sel", {"sel": True, "agg": 0, "conds": []}),
        ("sel", {"sel": None, "agg": 0, "conds": []}),
        ("agg", {"sel": 0, "agg": "0", "conds": []}),
        ("agg", {"sel": 0, "agg": False, "conds": []}),
        ("cond column", {"sel": 0, "agg": 0, "conds": [["0", 0, "x"]]}),
        ("cond operator", {"sel": 0, "agg": 0, "conds": [[0, "0", "x"]]}),
    ],
)
def test_non_integer_indices_are_rejected(field, form):
    """A non-int index is string-formatted into the SQL text upstream.

    `bool` is in here on purpose: it satisfies `isinstance(v, int)` but renders
    as `colTrue`, so accepting it would query a column that does not exist.
    """
    with _engine() as engine:
        q = Query.from_dict(form)
        with pytest.raises(InvalidQueryIndex, match=field):
            engine.execute_query("1-234-5", q)


@pytest.mark.parametrize(
    ("field", "form"),
    [
        ("sel", {"sel": -1, "agg": 0, "conds": []}),
        ("sel", {"sel": 3, "agg": 0, "conds": []}),
        ("agg", {"sel": 0, "agg": -1, "conds": []}),
        ("agg", {"sel": 0, "agg": 6, "conds": []}),
        ("cond column", {"sel": 0, "agg": 0, "conds": [[-1, 0, "x"]]}),
        ("cond column", {"sel": 0, "agg": 0, "conds": [[3, 0, "x"]]}),
        ("cond operator", {"sel": 0, "agg": 0, "conds": [[0, -1, "x"]]}),
        ("cond operator", {"sel": 0, "agg": 0, "conds": [[0, 4, "x"]]}),
    ],
)
def test_out_of_range_indices_are_rejected(field, form):
    """Negative is the dangerous half: Python wraps, so `agg_ops[-1]` is AVG.

    That path raises nothing upstream — it silently scores a query the model
    never asked for, which is why the guard checks the lower bound and not only
    the upper one.
    """
    with _engine() as engine:
        q = Query.from_dict(form)
        with pytest.raises(InvalidQueryIndex, match=field):
            engine.execute_query("1-234-5", q)


def test_negative_agg_would_alias_to_avg_without_the_guard():
    """Pins WHY the lower bound exists, by showing what it prevents.

    If this ever stops raising, the guard has been removed and `agg = -1` is
    quietly computing an average.
    """
    assert Query.agg_ops[-1] == "AVG"
    with _engine() as engine, pytest.raises(InvalidQueryIndex):
        engine.execute_query(
            "1-234-5", Query.from_dict({"sel": 2, "agg": -1, "conds": []})
        )


def test_sql_injection_through_sel_is_rejected_not_executed():
    payload = "0 AS result FROM table_1_234_5; DROP TABLE table_1_234_5; --"
    with _engine() as engine:
        with pytest.raises(InvalidQueryIndex):
            engine.execute_query(
                "1-234-5",
                Query.from_dict({"sel": payload, "agg": 0, "conds": []}),
            )
        # The table is still there.
        q = Query.from_dict({"sel": 0, "agg": 3, "conds": []})
        assert engine.execute_query("1-234-5", q) == [3]


def test_condition_values_are_bound_not_interpolated():
    """Values need no guard because upstream already binds them as parameters."""
    payload = "x'; DROP TABLE table_1_234_5; --"
    with _engine() as engine:
        q = Query.from_dict({"sel": 0, "agg": 0, "conds": [[1, 0, payload]]})
        assert engine.execute_query("1-234-5", q) == []
        assert engine.execute_query(
            "1-234-5", Query.from_dict({"sel": 0, "agg": 3, "conds": []})
        ) == [3]


def test_op_index_three_is_accepted_and_left_to_sqlite():
    """`OP` is in upstream's cond_ops and renders as invalid SQL.

    Not rejected by the guard: SQLite raising is upstream's behaviour, and
    keeping it means a model emitting `OP` stays distinguishable from one
    emitting an out-of-range integer.
    """
    with _engine() as engine:
        q = Query.from_dict({"sel": 0, "agg": 0, "conds": [[0, 3, "x"]]})
        with pytest.raises(Exception) as excinfo:
            engine.execute_query("1-234-5", q)
        assert not isinstance(excinfo.value, InvalidQueryIndex)


def test_missing_table_raises_lookup_error():
    with _engine() as engine, pytest.raises(LookupError):
        engine.execute_query(
            "9-999-9", Query.from_dict({"sel": 0, "agg": 0, "conds": []})
        )
