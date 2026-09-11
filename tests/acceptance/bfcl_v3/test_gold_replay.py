"""Replay BFCL v3 gold as the reply, and require the grader to accept it.

Each row's `possible_answer` IS an accepted answer, so grading it must return
correct. That checks the dataset join, the category routing and the vendored
checker together, against real data, with no model spend -- and it is the only
check here that can distinguish "the port is wired up right" from "the model is
good".

Restricted to `language == "Python"`. Java and JavaScript gold is not in the
decoded form: `ast_checker` requires every parameter there to arrive as a raw
source-text string for the type converters to parse, while gold holds native
values. Replaying it would mean writing a Java/JS serializer inside a test, and
its bugs would be indistinguishable from grader bugs.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
import pathlib

import pytest

from sieval.community.bfcl_v3 import GOLDLESS_CATEGORIES
from sieval.core.utils.hf import maybe_resolve_hf_path
from sieval.datasets import BfclV3LiveDataset, BfclV3NonLiveDataset
from sieval.tasks.bfcl_v3._base import grade_single_turn

#: The snapshot both datasets are pinned to; `maybe_resolve_hf_path` rewrites it
#: to `$SIEVAL_DATA_DIR/gorilla-llm/...`.
SNAPSHOT = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

#: Rows where upstream's gold contradicts upstream's own schema, so no reply can
#: be both the gold answer and schema-valid. Each was read individually; see the
#: table in this module's commit message. Asserted by EQUALITY: a pin bump that
#: fixes one must fail here and be updated in the same commit.
UPSTREAM_GOLD_DEFECTS = frozenset(
    {
        "simple_363",
        "live_simple_106-63-0",
        "live_simple_112-68-0",
        "live_multiple_507-149-4",
        "live_multiple_862-181-3",
        "live_multiple_964-207-0",
        "live_multiple_1038-265-0",
    }
)


def _resolve(node):
    """Gold nests alternatives inside dicts AND lists; unwrap every level."""
    if isinstance(node, dict):
        return {
            key: _first(accepted)
            for key, accepted in node.items()
            if _first(accepted) is not None
        }
    if isinstance(node, list):
        return [_resolve(item) for item in node]
    return node


def _first(accepted):
    """One concrete value out of a gold alternatives list.

    A leaf is *usually* a list of acceptable values, but three leaves in 3641
    rows are a bare scalar, so a non-list is passed through rather than indexed.
    `""` means "omitting is acceptable" and is skipped here; whether omission is
    actually permitted is the schema's call, made in `_replay`.
    """
    if not isinstance(accepted, list):
        return accepted
    for candidate in accepted:
        if candidate != "":
            return _resolve(candidate)
    return None


def _replay(gold_entry: list, functions: list) -> list[dict]:
    """Turn one `ground_truth` entry into the decoded form a model would yield.

    Reads the schema because gold alone is ambiguous: `""` among a parameter's
    alternatives licenses omitting it, but only where the schema does not mark it
    required -- and some parameters gold offers are not in the schema at all, so
    naming them is an `unexpected_param`.
    """
    schemas = {fn["name"]: fn for fn in functions}
    replayed = []
    for call in gold_entry:
        for name, params in call.items():
            parameters = schemas.get(name, {}).get("parameters", {})
            required = set(parameters.get("required", ()))
            args = {}
            for param, accepted in params.items():
                if not accepted:
                    continue
                # `""` licenses omitting the parameter, but only where the
                # schema does not require it -- dropping a required one is
                # `missing_required`, and emitting one the schema never declares
                # is `unexpected_param`, so both sides need the schema.
                omittable = isinstance(accepted, list) and "" in accepted
                if omittable and param not in required:
                    continue
                value = _first(accepted)
                if value is not None:
                    args[param] = value
            replayed.append({name: args})
    return replayed


@pytest.mark.skipif(
    not pathlib.Path(maybe_resolve_hf_path(SNAPSHOT)).is_dir(),
    reason=f"needs the downloaded BFCL v3 snapshot ({SNAPSHOT})",
)
def test_replaying_gold_as_the_reply_grades_correct():
    missed = set()
    graded = 0
    for dataset in (BfclV3NonLiveDataset(SNAPSHOT), BfclV3LiveDataset(SNAPSHOT)):
        test_set = dataset.test_set
        # Bound rather than iterated inline: the property is `HFDataset | None`,
        # and a missing split would otherwise walk zero rows on the way to the
        # count assertion below, reporting a moved pin for an absent split.
        assert test_set is not None, f"{type(dataset).__name__} has no test split"
        for row in test_set:
            if row["category"] in GOLDLESS_CATEGORIES or row["language"] != "Python":
                continue
            graded += 1
            decoded = _replay(
                json.loads(row["ground_truth"]), json.loads(row["function"])
            )
            if not grade_single_turn(
                row["function"],
                decoded,
                row["ground_truth"],
                row["language"],
                row["category"],
                False,
            ):
                missed.add(row["id"])

    assert graded == 2351, (
        f"Expected 2351 gradeable Python rows, walked {graded}. The snapshot pin "
        "moved, or the goldless/language filter changed -- either way the defect "
        "list below was measured against a different corpus."
    )
    assert missed == UPSTREAM_GOLD_DEFECTS, (
        "Replaying gold as the reply must grade correct.\n"
        f"  newly failing (OUR defect, fix it): "
        f"{sorted(missed - UPSTREAM_GOLD_DEFECTS)}\n"
        f"  newly passing (upstream fixed it, update the set): "
        f"{sorted(UPSTREAM_GOLD_DEFECTS - missed)}"
    )
