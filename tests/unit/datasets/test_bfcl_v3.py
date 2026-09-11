"""The shared BFCL v3 single-turn loader: the index join and its guards.

What is covered here is the join key (the universal index, not the whole id),
the per-category row counts, the guards that refuse an ambiguous or incomplete
join, and the one thing neither can check -- that a goldless category stays
goldless when it is loaded next to a category that has gold. Also pinned: the
package holding the loader's category tables must import without the optional
`bfcl-v3` extra, since every dataset in the repo is registered through the same
loop.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sieval.community.bfcl_v3 import GOLDLESS_CATEGORIES
from sieval.datasets._bfcl_v3 import load_categories

_ROOT = Path(__file__).parents[3]


def _write(root, category, rows, gold=None):
    (root / f"BFCL_v3_{category}.json").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    if gold is not None:
        answers = root / "possible_answer"
        answers.mkdir(exist_ok=True)
        (answers / f"BFCL_v3_{category}.json").write_text(
            "\n".join(json.dumps(g) for g in gold), encoding="utf-8"
        )


def test_gold_is_joined_onto_its_prompt_row(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [{"name": "f"}],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": [{"f": {"a": [1]}}]}],
    )
    row = load_categories(str(tmp_path), {"simple": 1})["test"][0]
    assert row["id"] == "simple_0"
    assert row["language"] == "Python"
    assert json.loads(row["ground_truth"]) == [{"f": {"a": [1]}}]
    assert row["question"] == [{"role": "user", "content": "hi"}]


def test_a_goldless_category_needs_no_answer_file(tmp_path):
    assert "irrelevance" in GOLDLESS_CATEGORIES
    _write(
        tmp_path,
        "irrelevance",
        [
            {
                "id": "irrelevance_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [{"name": "f"}],
            }
        ],
    )
    row = load_categories(str(tmp_path), {"irrelevance": 1})["test"][0]
    assert row["ground_truth"] is None


def test_a_goldless_category_stays_goldless_beside_one_that_has_gold(tmp_path):
    """The gold table is per-category, and the universal indices of two
    categories overlap -- `irrelevance_0` and `simple_0` both key on 0. A gold
    table built once for the whole call would hand every goldless row its
    neighbour's answer, which reads downstream as a real gold rather than as an
    error: the sample TypedDict documents `None` as "this row has no gold".
    """
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "set it to 50"}]],
                "function": [{"name": "set_volume", "parameters": {}}],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": [{"set_volume": {"volume": [50]}}]}],
    )
    _write(
        tmp_path,
        "irrelevance",
        [
            {
                "id": "irrelevance_0",
                "question": [[{"role": "user", "content": "how are you"}]],
                "function": [{"name": "set_volume", "parameters": {}}],
            }
        ],
    )

    rows = list(load_categories(str(tmp_path), {"simple": 1, "irrelevance": 1})["test"])

    # Every category's rows, not just the last one's.
    assert [row["id"] for row in rows] == ["simple_0", "irrelevance_0"]
    by_id = {row["id"]: row for row in rows}
    assert json.loads(by_id["simple_0"]["ground_truth"]) == [
        {"set_volume": {"volume": [50]}}
    ]
    assert by_id["irrelevance_0"]["ground_truth"] is None
    # `function` is the tool schema the whole grading path reads, and is declared
    # `str` because pyarrow cannot type its nesting.
    for row in rows:
        assert isinstance(row["function"], str)
        assert json.loads(row["function"]) == [{"name": "set_volume", "parameters": {}}]


def test_java_and_javascript_carry_their_language(tmp_path):
    for category, language in (("java", "Java"), ("javascript", "JavaScript")):
        _write(
            tmp_path,
            category,
            [
                {
                    "id": f"{category}_0",
                    "question": [[{"role": "user", "content": "x"}]],
                    "function": [{"name": "f"}],
                }
            ],
            gold=[{"id": f"{category}_0", "ground_truth": [{"f": {}}]}],
        )
        row = load_categories(str(tmp_path), {category: 1})["test"][0]
        assert row["language"] == language


def test_a_moved_row_count_raises(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="expected 400"):
        load_categories(str(tmp_path), {"simple": 400})


def test_a_moved_gold_row_count_raises(tmp_path):
    """Upstream asserts prompt and gold are the same length. Guarding only the
    prompt side would let a re-upload that adds gold rows through: the loader
    joins on the index, so the extra rows are simply never looked up and the
    column is rescored with nothing to show for it.
    """
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
        ],
        gold=[{"id": f"simple_{i}", "ground_truth": []} for i in range(3)],
    )
    with pytest.raises(ValueError, match="3 possible_answer rows"):
        load_categories(str(tmp_path), {"simple": 1})


def test_a_non_numeric_universal_index_raises(tmp_path):
    """The id shape moving while the counts hold is exactly what a pin bump can
    do, and it must not surface as a bare `int()` failure with no category, row
    or file in it.
    """
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0a",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
        ],
        gold=[{"id": "simple_0a", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="where its universal index should be") as e:
        load_categories(str(tmp_path), {"simple": 1})
    assert "'simple_0a'" in str(e.value)
    assert "'simple'" in str(e.value)
    assert "'0a'" in str(e.value)


def test_a_multi_turn_row_is_refused(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [
                    [{"role": "user", "content": "a"}],
                    [{"role": "user", "content": "b"}],
                ],
                "function": [],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="turns"):
        load_categories(str(tmp_path), {"simple": 1})


def test_gold_joins_on_the_universal_index_not_the_whole_id(tmp_path):
    """Upstream sorts both files on the universal index and pairs them
    positionally, so a row whose sub-indices disagree between the two files is
    still graded. `live_multiple_1052-79-0` / `live_multiple_1052-279-0` is that
    row at the pinned revision; joining on the whole id would drop it.
    """
    _write(
        tmp_path,
        "live_multiple",
        [
            {
                "id": "live_multiple_1052-79-0",
                "question": [[{"role": "user", "content": "volume 50"}]],
                "function": [{"name": "set_volume"}],
            }
        ],
        gold=[
            {
                "id": "live_multiple_1052-279-0",
                "ground_truth": [{"set_volume": {"volume": [50]}}],
            }
        ],
    )
    row = load_categories(str(tmp_path), {"live_multiple": 1})["test"][0]
    assert row["id"] == "live_multiple_1052-79-0"
    assert json.loads(row["ground_truth"]) == [{"set_volume": {"volume": [50]}}]


def _two_rows(ids):
    return [
        {
            "id": row_id,
            "question": [[{"role": "user", "content": row_id}]],
            "function": [],
        }
        for row_id in ids
    ]


def test_a_repeated_universal_index_in_the_prompt_file_raises(tmp_path):
    """Two rows sharing the join key make the pairing ambiguous. Upstream cannot
    see this -- it only checks that the two files are the same length.

    The gold file here is well-formed, so only the prompt-side guard can fire.
    """
    _write(
        tmp_path,
        "live_multiple",
        _two_rows(["live_multiple_0-1-0", "live_multiple_0-2-0"]),
        gold=[
            {"id": "live_multiple_0-1-0", "ground_truth": []},
            {"id": "live_multiple_1-1-0", "ground_truth": []},
        ],
    )
    with pytest.raises(
        ValueError, match="shared by more than one prompt row"
    ) as raised:
        load_categories(str(tmp_path), {"live_multiple": 2})
    # The colliding index and its ids, not a bare count of excess rows.
    assert "{0: ['live_multiple_0-1-0', 'live_multiple_0-2-0']}" in str(raised.value)
    assert "1 universal index " in str(raised.value)


def test_a_repeated_universal_index_in_the_gold_file_raises(tmp_path):
    """The mirror of the test above: the prompt file is well-formed and the
    repeat is on the gold side, so only the possible_answer guard can fire.
    """
    _write(
        tmp_path,
        "live_multiple",
        _two_rows(["live_multiple_0-1-0", "live_multiple_1-1-0"]),
        gold=[
            {"id": "live_multiple_0-1-0", "ground_truth": []},
            {"id": "live_multiple_0-2-0", "ground_truth": []},
        ],
    )
    with pytest.raises(ValueError, match="shared by more than one possible_answer row"):
        load_categories(str(tmp_path), {"live_multiple": 2})


def test_the_dataset_registry_loads_without_the_bfcl_v3_extra():
    """Both loaders import `sieval.community.bfcl_v3` for two pure-data tables,
    and `import_all_datasets()` imports every dataset module in a bare loop with
    no per-module `except`. So anything that package pulls in at import time is a
    hard requirement for *every* dataset in the repo -- `sieval dataset list`
    included -- and tree-sitter, which only the vendored `parser.py` needs, is in
    the optional `bfcl-v3` extra.

    Out of process with the extra blocked at `sys.meta_path`: this venv has
    tree-sitter installed (CI's `INSTALL_GROUPS` carries `-G bfcl-v3` so the
    Java/JavaScript decode path is exercised), so an in-process check would pass
    with the eager import restored.
    """
    child = f"""
import sys

class _NoTreeSitter:
    def find_spec(self, name, path=None, target=None):
        if name == "tree_sitter" or name.startswith("tree_sitter."):
            raise ModuleNotFoundError("No module named " + name)
        return None

sys.meta_path.insert(0, _NoTreeSitter())

import sieval

# The editable install may name a different checkout than the tree this test
# was loaded from; without this the child would probe that one instead.
assert sieval.__file__.startswith({str(_ROOT)!r}), sieval.__file__

from sieval.core.datasets.meta import import_all_datasets

import_all_datasets()
assert "sieval.datasets.bfcl_v3_non_live" in sys.modules
assert "sieval.datasets.bfcl_v3_live" in sys.modules
"""
    inherited = os.environ.get("PYTHONPATH")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(_ROOT), *([inherited] if inherited else [])]
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, (
        "the dataset registry must import without the optional `bfcl-v3` "
        f"extra:\n{completed.stderr}"
    )


def test_a_gold_miss_raises(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
        ],
        gold=[{"id": "simple_99", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="no possible_answer"):
        load_categories(str(tmp_path), {"simple": 1})
