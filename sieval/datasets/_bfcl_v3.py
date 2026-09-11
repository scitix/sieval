"""Shared loader for the BFCL v3 single-turn category files.

One HF repo holds every category as its own JSONL file, with ground truth in a
parallel `possible_answer/` file. Three categories ship no gold -- `irrelevance`,
`live_irrelevance` and `live_relevance` are scored on whether a call was
produced, not on which call it was.

The join key is the **universal index** -- the `<index>` of
`<category>_<index>[-<sub>-<sub>]` -- not the whole `id`. That is upstream's key:
its runner sorts both files on `(category, universal_index)` and pairs them
positionally, discarding the sub-indices. It is load-bearing on exactly one row
at the pinned revision, where prompt `live_multiple_1052-79-0` is graded against
gold `live_multiple_1052-279-0`: one sub-index is an upstream typo, invisible to
upstream because it never enters the key. Joining on the whole `id` would score
1052 of `live_multiple`'s 1053 rows and publish a column upstream never did.

An index join equals upstream's positional one only while the key is unique per
joined file, so the loader asserts that instead of assuming it -- stricter than
upstream, which only checks the two files are the same length. Not asserted for
the goldless three, which are never joined: `live_relevance` ships one row twice,
18 rows over 17 distinct ids.

Row counts are asserted per category on the gold file as well as the prompt file,
which together cover upstream's own `len(prompt) == len(possible_answer)`. The
revision pin stops a silent re-upload but not a pin bump, so a count that moves
fails here rather than quietly rescoring a leaderboard column.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
from collections.abc import Mapping
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.bfcl_v3 import GOLDLESS_CATEGORIES, LANGUAGE_BY_CATEGORY

#: Pinned snapshot. Verified row-for-row against the harness's own
#: `bfcl_eval/data/` at gorilla v1.3.
BFCL_V3_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _universal_index(row_id: str, category: str, what: str) -> int:
    """The join key of one row: the index in `<category>_<index>[-<sub>-<sub>]`.

    Mirrors the index that upstream's own sort key extracts -- it splits on the
    last underscore and keeps only what precedes the first dash, so the two
    sub-indices the live categories carry are discarded here exactly as they
    are there.
    """
    index = row_id.rsplit("_", 1)[-1].split("-")[0]
    try:
        return int(index)
    except ValueError:
        raise ValueError(
            f"BFCL v3 {what} row {row_id!r} in category {category!r} has "
            f"{index!r} where its universal index should be. The index is the "
            "join key upstream sorts both files on, so an id whose shape moved "
            "cannot be joined at all; re-verify the category against upstream."
        ) from None


def _index_rows(rows: list[dict], category: str, what: str) -> dict[int, dict]:
    by_index: dict[int, dict] = {}
    collisions: dict[int, list[str]] = {}
    for row in rows:
        index = _universal_index(row["id"], category, what)
        if index in by_index:
            collisions.setdefault(index, [by_index[index]["id"]]).append(row["id"])
        else:
            by_index[index] = row

    if collisions:
        # The ids, not just a count: on a pin bump this is the difference
        # between a one-minute diagnosis and a manual scan of 1053 rows.
        examples = {index: collisions[index] for index in sorted(collisions)[:3]}
        plural = "index" if len(collisions) == 1 else "indices"
        raise ValueError(
            f"BFCL v3 category {category!r} has {len(collisions)} universal "
            f"{plural} shared by more than one {what} row (e.g. {examples}). "
            "The index is the join key and upstream's runner pairs the two "
            "files positionally under it, so a repeat makes the pairing "
            "ambiguous rather than merely odd."
        )
    return by_index


def load_categories(name_or_path: str, counts: Mapping[str, int]) -> HFDatasetDict:
    root = Path(name_or_path)
    rows: list[dict] = []
    for category, expected in counts.items():
        prompts = _read_jsonl(root / f"BFCL_v3_{category}.json")
        if len(prompts) != expected:
            raise ValueError(
                f"BFCL v3 category {category!r} has {len(prompts)} rows at the "
                f"pinned revision, expected {expected}. A count that moved means "
                "the pin was bumped; re-verify the category against upstream and "
                "update the table deliberately, do not relax this check."
            )

        gold_by_index: dict[int, str] = {}
        if category not in GOLDLESS_CATEGORIES:
            # Only a category that is actually joined needs a unique key. The
            # goldless three are not joined, and one of them must not be held to
            # this: `live_relevance` ships `live_relevance_3-3-0` twice, so it
            # has 18 rows and 17 distinct ids. Upstream scores all 18 and so do
            # we -- deduplicating here would publish a denominator upstream
            # never used.
            prompt_by_index = _index_rows(prompts, category, "prompt")
            gold = _read_jsonl(root / "possible_answer" / f"BFCL_v3_{category}.json")
            if len(gold) != expected:
                raise ValueError(
                    f"BFCL v3 category {category!r} has {len(gold)} "
                    f"possible_answer rows at the pinned revision, expected "
                    f"{expected}. Upstream asserts the prompt and gold files are "
                    "the same length, so a gold file that moved on its own would "
                    "rescore the column without a single prompt changing."
                )
            answers = _index_rows(gold, category, "possible_answer")
            gold_by_index = {
                index: json.dumps(entry["ground_truth"], ensure_ascii=False)
                for index, entry in answers.items()
            }
            missing = set(prompt_by_index) - set(gold_by_index)
            if missing:
                examples = [prompt_by_index[i]["id"] for i in sorted(missing)[:3]]
                raise ValueError(
                    f"BFCL v3 category {category!r} has {len(missing)} prompt rows "
                    f"with no possible_answer entry (e.g. {examples}). The join key "
                    "is the universal index; a miss means the two files disagree."
                )

        for row in prompts:
            turns = row["question"]
            # Every phase-1 category is single-turn. Asserting beats taking [0]:
            # a multi-turn file loaded here would silently score only its first
            # turn and look like a very poor model.
            if len(turns) != 1:
                raise ValueError(
                    f"BFCL v3 row {row['id']!r} in {category!r} has {len(turns)} "
                    "turns; the single-turn datasets accept exactly one. "
                    "Multi-turn categories are phase 2 and must not be loaded here."
                )
            rows.append(
                {
                    "id": row["id"],
                    "category": category,
                    "language": LANGUAGE_BY_CATEGORY.get(category, "Python"),
                    "question": turns[0],
                    "function": json.dumps(row["function"], ensure_ascii=False),
                    "ground_truth": gold_by_index.get(
                        _universal_index(row["id"], category, "prompt")
                    ),
                }
            )

    return HFDatasetDict({"test": HFDataset.from_list(rows)})
