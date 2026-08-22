"""Spider 1.0 dataset loader (Yale LILY).

Spider (Yu et al., EMNLP 2018) is the reference cross-domain text-to-SQL set:
1,034 dev questions over 20 databases, each paired with a gold SQL query.
Scoring is execution-based, so the SQLite databases are part of the dataset
rather than an optional extra — which is why this loader stages a 206 MB archive
instead of a table of rows.

**Provenance.** Upstream distributes ``spider_data.zip`` through Google Drive
only, which has no stable direct-download URL. This loader pulls a
checksum-pinned mirror instead. The archive was verified against the official
row set before pinning: ``dev.json`` carries exactly 1,034 rows over 20 distinct
``db_id``s, every one of those databases ships its ``.sqlite``, and row 0 is
Spider's canonical "How many singers do we have?" /
``SELECT count(*) FROM singer``. The sha256 is what makes the mirror
reproducible; if it ever fails, re-verify against the official release rather
than re-pinning blind.

**The ``sql`` column is dropped at load.** Upstream ships a pre-parsed query tree
per row whose ``except``/``intersect``/``union`` slots are sometimes null and
sometimes nested objects. Arrow cannot infer a schema across that
(``ArrowInvalid: cannot mix list and non-list, non-null values``), so
``load_dataset("json", ...)`` fails on the file outright. Nothing is lost: the
vendored ``process_sql.get_sql`` rebuilds the same tree from the ``query``
string, which is what the grader does anyway.

``train_others.json`` (1,659 rows from other sources) is not loaded — the
``train`` split here is ``train_spider.json`` alone, matching what "Spider train"
means in the literature. Neither split is consumed by ``spider_0shot_gen``,
which evaluates ``validation``.

References:

* Paper: <https://arxiv.org/abs/1809.08887>
* Harness: <https://github.com/taoyds/spider>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import zipfile
from pathlib import Path
from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

#: Hub revision of the checksum-pinned mirror carrying upstream's archive.
SPIDER_ARCHIVE_REVISION = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
SPIDER_ARCHIVE_SHA256 = (
    "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
)
ARCHIVE_BASENAME = "spider_data.zip"
#: Every member of the archive lives under this single top-level directory.
_ROOT = "spider_data"
#: Written after a complete extraction; its presence is what makes a second
#: load cheap. Leading dot so it cannot collide with a db_id.
_MARKER = ".sieval-extracted"
#: Upstream ships this pre-parsed query tree; Arrow cannot type it. See above.
_UNREPRESENTABLE = "sql"


class SpiderDatasetSample(TypedDict):
    db_id: str
    query: str
    query_toks: list[str]
    query_toks_no_value: list[str]
    question: str
    question_toks: list[str]


@sieval_dataset(
    name="spider",
    display_name="Spider 1.0",
    description="Cross-domain text-to-SQL; 1,034 dev questions over 20 databases.",
    source=(
        f"url:https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/"
        f"{SPIDER_ARCHIVE_REVISION}/{ARCHIVE_BASENAME}"
    ),
    checksums={ARCHIVE_BASENAME: f"sha256:{SPIDER_ARCHIVE_SHA256}"},
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "text-to-sql", "code-exec"),
    license="CC-BY-SA-4.0",
)
class SpiderDataset(Dataset[SpiderDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        _ = kwargs
        root = self._extract(Path(name_or_path))
        # Captured so the task can reach the databases; `copy.copy`-based clones
        # (slice/shuffle) preserve these, matching SciCodeDataset.h5_path.
        self._db_dir = str(root / "database")
        self._tables_json_path = str(root / "tables.json")
        return HFDatasetDict(
            {
                "train": self._read_split(root / "train_spider.json"),
                # Upstream's `dev.json`, deliberately exposed as `test`. Two
                # reasons, and the mismatch is worth the friction: the runner
                # evaluates `Dataset.test_set`, which reads exactly this key, so
                # any other name would silently evaluate nothing; and Spider's
                # dev set is the split the literature reports, because the real
                # test set was held out for years. The archive does ship
                # `test.json`, but nothing here reads it — adding it would take
                # this name and push the reported split somewhere unevaluated.
                "test": self._read_split(root / "dev.json"),
            }
        )

    @property
    def db_dir(self) -> str | None:
        """Directory of per-database SQLite files, or ``None`` if never loaded."""
        return getattr(self, "_db_dir", None)

    @property
    def tables_json_path(self) -> str | None:
        """Upstream's schema/foreign-key file, or ``None`` if never loaded."""
        return getattr(self, "_tables_json_path", None)

    @staticmethod
    def _read_split(path: Path) -> HFDataset:
        rows = json.loads(path.read_text(encoding="utf-8"))
        projected = [
            {k: v for k, v in row.items() if k != _UNREPRESENTABLE} for row in rows
        ]
        if not projected:
            raise ValueError(
                f"Spider split {path.name!r} is empty; re-stage the archive with "
                "'sieval dataset download spider --force'."
            )
        return HFDataset.from_list(projected)

    @staticmethod
    def _extract(staged: Path) -> Path:
        archive = staged if staged.is_file() else staged / ARCHIVE_BASENAME
        root = (staged.parent if staged.is_file() else staged) / _ROOT
        if (root / _MARKER).is_file():
            return root
        if not archive.is_file():
            raise FileNotFoundError(
                f"Spider archive not found at {str(archive)!r}. Run "
                "'sieval dataset download spider' to stage spider_data.zip."
            )
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(root.parent)
        (root / _MARKER).touch()
        return root
