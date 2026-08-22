"""WikiSQL dataset loader (Salesforce).

WikiSQL (Zhong et al., 2017, arXiv:1709.00103) is 80,654 crowd-sourced
natural-language questions over 24,241 Wikipedia tables, each paired with the
SQL query that answers it. The queries live in a restricted grammar --
``SELECT [agg] col FROM table WHERE col op value [AND ...]`` -- so upstream
represents the ground truth as a **logical form** (``sel``/``agg``/``conds``
indices) rather than as SQL text, and its ``evaluate.py`` scores predictions in
that same form.

This loader materialises the two published evaluation splits, ``test`` (15,878
questions over 5,230 tables) and ``validation`` (8,421 over 2,716) -- the two
columns every leaderboard row reports. ``train`` (56,355) is deliberately not
materialised: nothing consumes it yet, and denormalising it would add roughly
150 MB of Arrow. Adding it is a one-line change to ``_SPLITS`` if a k-shot
sibling ever wants exemplars.

**No SQLite files.** Upstream's ``data.tar.bz2`` ships ``{split}.db`` alongside
the JSONL, and its ``DBEngine`` reads them. This loader ignores them and carries
each table's ``types``/``rows`` on the row instead, letting
``sieval.community.wikisql`` rebuild the table in memory through upstream's own
``create_table``. Verified rather than assumed -- all 15,878 test gold queries
return identical results either way, the declared schema of all 7,946 test+dev
tables matches the ``types`` column, and upstream's own example predictions
score identically. It drops ~120 MB of binary SQLite from the download and
leaves the engine no filesystem reach.

**Each row carries its own table** (the 5,230 test tables denormalise across
15,878 questions). A shared side table would have to reach the task outside the
framework's per-sample flow, which the stage contract forbids; a self-contained
row is also what makes a single sample replayable on its own.

Two columns are JSON strings because their dtypes cannot be a typed Arrow
column, not as a shortcut:

* ``sql_json`` -- ``conds`` is a list of ``[col, op, value]`` triples whose
  value is ``str``, ``int`` or ``float`` depending on the row (16,424 / 4,957 /
  465 of them across the test split), so the inner list has no single type.
* ``rows_json`` -- cells are ``str`` or ``int``, and mixed *within* a column:
  ``real``-typed columns hold 51,471 ints and 103,971 strings across test.

The ``_json`` suffix is this repo's existing signal for a forced serialisation
(cf. ``sysbench``'s ``turns_json``). Every other upstream field keeps its own
name and dtype -- no cast needed, upstream ships these dtypes.

``page_title`` / ``section_title`` / ``caption`` are absent on 329 of 5,230 test
tables (181 of 2,716 dev), ``name`` on 3,956, and ``page_id`` on 1,274, hence
``| None`` on each. They are carried but unused by the scorer: the engine derives
its SQL identifier from ``table_id``, never from ``name``.

Licensing: the upstream repository is BSD-3-Clause, which covers both the
harness (vendored in ``sieval.community.wikisql``) and the data files
distributed in its tree. The tables are derived from Wikipedia (CC-BY-SA-3.0);
a redistributor should honour both.

References:

* Paper: <https://arxiv.org/abs/1709.00103>
* Harness: <https://github.com/salesforce/WikiSQL>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import os
import tarfile
from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

#: Upstream commit the data tarball is addressed at. The repository is archived
#: (2025-10-06), so this is also its final state -- but the commit is pinned
#: rather than the branch anyway: a branch would let the bytes move under a
#: checksum that no longer matches, which fails closed only after a download.
WIKISQL_COMMIT = "cffb423077756d04c1bac5bcd45167c86903fbcb"

WIKISQL_URL = (
    "https://raw.githubusercontent.com/salesforce/WikiSQL/"
    f"{WIKISQL_COMMIT}/data.tar.bz2"
)

_ARCHIVE = "data.tar.bz2"

#: HF split name -> upstream file stem. Upstream calls the held-out split
#: ``dev``; this maps it to the conventional ``validation`` so `fewshot_split`
#: and friends read the same here as everywhere else, while ``test`` keeps its
#: name. ``train`` is omitted -- see the module docstring.
_SPLITS = {"test": "test", "validation": "dev"}


class WikiSQLDatasetSample(TypedDict):
    # -- question side (upstream's `{split}.jsonl`)
    phase: int
    table_id: str
    question: str
    #: `{"sel": int, "agg": int, "conds": [[col, op, value], ...]}`, upstream's
    #: `sql` object verbatim. A JSON string because `conds` values are mixed
    #: str/int/float; feed it to `Query.from_dict(json.loads(...))`.
    sql_json: str
    # -- table side (upstream's `{split}.tables.jsonl`), denormalised
    header: list[str]
    #: Per-column SQL type, `text` or `real`. Load-bearing: the engine coerces
    #: a condition value against a `real` column through babel.
    types: list[str]
    #: `[[cell, ...], ...]`. A JSON string because cells are mixed str/int.
    rows_json: str
    page_title: str | None
    section_title: str | None
    caption: str | None
    name: str | None
    page_id: int | None


@sieval_dataset(
    name="wikisql",
    display_name="WikiSQL",
    description=("Questions over Wikipedia tables paired with their SQL logical form."),
    source=f"url:{WIKISQL_URL}",
    checksums={
        _ARCHIVE: "sha256:755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881",  # noqa: E501
    },
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "sql", "tabular", "text-to-sql"),
    # The upstream repository's own license, covering the harness and the data
    # files in its tree. The tables derive from Wikipedia (CC-BY-SA-3.0).
    license="BSD-3-Clause",
)
class WikiSQLDataset(Dataset[WikiSQLDatasetSample]):
    """WikiSQL, one row per question with its table attached.

    ``name_or_path`` may be the downloaded directory or the tarball itself, so a
    run works both from ``sieval dataset download wikisql`` and from a
    hand-placed copy.
    """

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        path = (
            os.path.join(name_or_path, _ARCHIVE)
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        dataset_dict = HFDatasetDict()
        # One pass over the archive per split, reading only that split's two
        # members. bz2 has no random access, so a member is reached by
        # decompressing up to it; two seeks beat holding every split in memory.
        with tarfile.open(path, "r:bz2") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            for split, stem in _SPLITS.items():
                questions = _read_jsonl(tar, members, f"data/{stem}.jsonl")
                tables = {
                    t["id"]: t
                    for t in _read_jsonl(tar, members, f"data/{stem}.tables.jsonl")
                }
                rows = [_row(q, tables[q["table_id"]]) for q in questions]
                if not rows:
                    raise ValueError(
                        f"WikiSQL produced an empty {split!r} split from "
                        f"{path!r}; re-run `sieval dataset download wikisql`."
                    )
                dataset_dict[split] = HFDataset.from_list([dict(r) for r in rows])
        return ensure_dataset_dict(dataset_dict)


def _read_jsonl(tar: tarfile.TarFile, members: dict, name: str) -> list[dict]:
    """Decode one JSONL member of the archive."""
    member = members.get(name)
    if member is None:
        raise ValueError(
            f"{name!r} is not in the WikiSQL archive; found "
            f"{sorted(members)!r}. The download may be truncated or replaced."
        )
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"could not read {name!r} from the WikiSQL archive")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row(question: dict, table: dict) -> WikiSQLDatasetSample:
    """One upstream question + its table -> one row."""
    return {
        "phase": question["phase"],
        "table_id": question["table_id"],
        "question": question["question"],
        "sql_json": json.dumps(question["sql"], ensure_ascii=False),
        "header": table["header"],
        "types": table["types"],
        "rows_json": json.dumps(table["rows"], ensure_ascii=False),
        # `.get()` rather than `[]`: these five are absent on a measured
        # fraction of tables (see the module docstring), and absent is the same
        # thing as null here -- upstream carries no distinction between them.
        "page_title": table.get("page_title"),
        "section_title": table.get("section_title"),
        "caption": table.get("caption"),
        "name": table.get("name"),
        "page_id": table.get("page_id"),
    }
