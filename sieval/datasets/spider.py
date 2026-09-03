"""Spider 1.0 dataset loader (Yale LILY).

Spider (Yu et al., EMNLP 2018) is the reference cross-domain text-to-SQL set:
1,034 dev questions over 20 databases, each paired with a gold SQL query.
Scoring is execution-based, so the SQLite databases are part of the dataset
rather than an optional extra — which is why this loader stages ~1.5 GB of
archives instead of a table of rows. Two of them, for two different jobs:

* ``spider_data.zip`` (206 MB) — the question rows, ``tables.json``, and one
  SQLite database per ``db_id``. Everything except the headline metric reads it.
* ``testsuite_databases.zip`` (1.27 GB) — 25 to 60 *distilled* variants of each
  of those databases, from ``taoyds/test-suite-sql-eval``. Same schemas,
  different rows, chosen so that a query which merely happens to agree with the
  gold on the shipped database disagrees on one of these. This is what makes
  test-suite accuracy the metric it is, and it is why the download is large.

**Provenance.** Upstream distributes both archives through Google Drive only.
``spider_data.zip`` has no stable direct-download URL, so this loader pulls a
checksum-pinned mirror, verified against the official row set before pinning:
``dev.json`` carries exactly 1,034 rows over 20 distinct ``db_id``s, every one
of those databases ships its ``.sqlite``, and row 0 is Spider's canonical "How
many singers do we have?" / ``SELECT count(*) FROM singer``.

The checksum pins the bytes; what it cannot pin is *availability*. If the mirror
disappears, the recovery path is upstream's own Drive ``spider_data.zip``
(linked from <https://yale-lily.github.io/spider>), staged by hand and checked
against the same sha256 — not a swap to whichever mirror is reachable. The
widely-used ``xlangai/spider`` on the Hub is **not** a substitute: it publishes
the question rows only, without the ``.sqlite`` databases execution grading
needs.

``testsuite_databases.zip`` has **no mirror at all** — the Hub's several
"SPIDER" datasets are unrelated sets sharing the name — so it is fetched from
Drive directly, which works without cookies because the file is public and the
URL carries ``confirm=t``. That is a weaker provenance story and the checksum is
what carries it: a Drive link that starts serving something else fails the
download rather than quietly changing a published score. If it dies, upstream's
README (<https://github.com/taoyds/test-suite-sql-eval>) has the replacement.

**The ``sql`` column is dropped at load.** Upstream ships a pre-parsed query tree
per row whose ``except``/``intersect``/``union`` slots are sometimes null and
sometimes nested objects. Arrow cannot infer a schema across that
(``ArrowInvalid: cannot mix list and non-list, non-null values``), so
``load_dataset("json", ...)`` fails on the file outright. Nothing is lost: the
vendored ``process_sql.get_sql`` rebuilds the same tree from the ``query``
string, which is what the grader does anyway.

``train_others.json`` (1,659 rows from other sources) is not loaded — the
``train`` split here is ``train_spider.json`` alone, matching what "Spider train"
means in the literature. Neither training file is consumed by
``spider_0shot_gen``, which evaluates the ``test`` split — upstream's ``dev.json``
under the key the runner reads. See :meth:`SpiderDataset.load` for why dev lands
under that name.

References:

* Paper: <https://arxiv.org/abs/1809.08887>
* Harness: <https://github.com/taoyds/spider>
* Test suites: <https://github.com/taoyds/test-suite-sql-eval>,
  <https://arxiv.org/abs/2010.02840>

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

#: The distilled test suites, from ``taoyds/test-suite-sql-eval``. 1.27 GB,
#: 3,194 SQLite databases over 28 ``db_id``s — every one of the 20 dev
#: databases, 25 to 60 variants each.
TEST_SUITE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w&export=download&confirm=t"
)
TEST_SUITE_SHA256 = "9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96"
#: What the staged file is CALLED, which is not a choice: ``URLHandler`` names a
#: download after the URL's last path segment, and Google Drive's is
#: ``/download`` — every identifying part of that URL is a query parameter.
#: Renaming it would mean a filename override in the shared downloader plus a
#: matching change to the checksum-key rule, so pinning the odd name here is the
#: smaller surface. The checksum key must equal it, so the two are one constant.
TEST_SUITE_BASENAME = "download"
#: Where the test suites are extracted, relative to the staged directory. The
#: archive has NO single top-level directory — it opens straight onto
#: ``database/`` and ``__MACOSX/`` — so unlike the main archive this one is
#: extracted into a directory we name rather than one it carries. Named to read
#: as a sibling of ``spider_data``, which is the name its archive dictates.
_TEST_SUITE_ROOT = "spider_test_suite"
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
        f"{SPIDER_ARCHIVE_REVISION}/{ARCHIVE_BASENAME}",
        f"url:{TEST_SUITE_URL}",
    ),
    checksums={
        ARCHIVE_BASENAME: f"sha256:{SPIDER_ARCHIVE_SHA256}",
        TEST_SUITE_BASENAME: f"sha256:{TEST_SUITE_SHA256}",
    },
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "text-to-sql", "code-exec"),
    license="CC-BY-SA-4.0",
)
class SpiderDataset(Dataset[SpiderDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        _ = kwargs
        staged = Path(name_or_path)
        root = self._extract(staged)
        # Captured so the task can reach the databases; `copy.copy`-based clones
        # (slice/shuffle) preserve these, matching SciCodeDataset.h5_path.
        self._db_dir = str(root / "database")
        self._tables_json_path = str(root / "tables.json")
        self._test_suite_db_dir = str(self._extract_test_suite(staged) / "database")
        return HFDatasetDict(
            {
                "train": self._read_split(root / "train_spider.json"),
                # Upstream's `dev.json`, deliberately exposed as `test`, and the
                # mismatch is worth the friction: the runner evaluates
                # `Dataset.test_set`, which reads exactly this key, so any other
                # name would silently evaluate nothing — and Spider's dev set is
                # the split the literature reports, because the real test set was
                # held out for years. The archive does ship `test.json`, but
                # adding it would take this name and push the reported split
                # somewhere unevaluated.
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

    @property
    def test_suite_db_dir(self) -> str | None:
        """Directory of distilled test-suite databases, or ``None`` if unloaded.

        Holds one subdirectory per ``db_id``, each with 25-60 ``.sqlite`` files.
        These have the *same schemas* as ``db_dir``'s databases and different
        rows, which is the whole point — and also why they are kept in a
        separate tree rather than merged in.
        """
        return getattr(self, "_test_suite_db_dir", None)

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

    @staticmethod
    def _extract_test_suite(staged: Path) -> Path:
        """Stage the distilled test suites, returning their root directory.

        Separate from :meth:`_extract` rather than folded into it, because the
        two archives are shaped differently: this one has no top-level directory
        of its own, so the destination is named here instead of being read off
        the archive.
        """
        root = (staged.parent if staged.is_file() else staged) / _TEST_SUITE_ROOT
        if (root / _MARKER).is_file():
            return root
        archive = root.parent / TEST_SUITE_BASENAME
        if not archive.is_file():
            raise FileNotFoundError(
                f"Spider test-suite databases not found at {str(archive)!r}. Run "
                "'sieval dataset download spider' to stage them; they are 1.3 GB "
                "and back the headline metric, so they are not optional."
            )
        with zipfile.ZipFile(archive) as handle:
            # `__MACOSX/` is AppleDouble metadata from however upstream zipped
            # this, ~48 entries mirroring `database/`. Skipped to keep the tree
            # readable rather than for correctness: the grader globs
            # `database/<db_id>/` only, so nothing under `__MACOSX/` was ever
            # reachable from it.
            handle.extractall(
                root,
                members=[n for n in handle.namelist() if not n.startswith("__MACOSX")],
            )
        (root / _MARKER).touch()
        return root
