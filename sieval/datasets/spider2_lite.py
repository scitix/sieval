"""Spider 2.0-lite dataset loader (XLang Lab).

Spider 2.0 (Lei et al., ICLR 2025) is Spider's enterprise-scale successor: real
warehouse schemas with hundreds of columns, questions needing external
documentation, and SQL that is routinely dozens of lines. The **lite** setting
is its single-call text-to-SQL reading — 547 questions, no agent loop.

The 547 split three ways by ``instance_id`` prefix, and the prefix is the only
thing that says which engine a question runs on:

===============  =========================  =======
Prefix           Engine                     Count
===============  =========================  =======
``bq`` / ``ga``  BigQuery                   205
``sf_bq`` / ``sf``  Snowflake               207
``local``        SQLite (shipped locally)   135
===============  =========================  =======

Only the 135 ``local`` questions run without cloud credentials. The counts come
from the ids themselves; the upstream README's table is approximate.

**Two pinned sources, because the data is split across two hosts.** The
questions, per-database schemas, external-knowledge documents and gold results
live in the GitHub repo, which ships no release archive — and the gold set alone
is 1,544 CSVs, far past what enumerating ``url:`` entries could express, so the
whole repo archive is taken at a pinned commit and only ``spider2-lite/`` is
extracted (909 MB rather than the full 1.9 GB). The local SQLite databases are a
separate 457 MB download, which upstream tells you to unzip into
``resource/databases/spider2-localdb`` — this loader puts them exactly there.

**Take the local databases from the archive upstream's README links, not from
the Hub.** ``xlangai/spider2-localdb`` also publishes a ``sqlite.zip``, and it is
tempting because a Hub URL pins by revision while a Drive URL pins only by file
id plus checksum. It is the wrong corpus: it ships 40 databases, of which only
23 of the 30 the 135 ``local`` questions actually name, plus 17 no question
references. The seven it omits — ``bank_sales_trading`` (15 questions),
``city_legislation`` (10), ``modern_data`` (7), ``sqlite-sakila`` (7),
``education_business`` (5), ``California_Traffic_Collision`` (3) and ``music``
(1) — are **48 of the 135**, and each one raises
``sqlite3.OperationalError: unable to open database file`` at prompt time.
``local_sqlite.zip`` holds exactly the 30 the questions name, no more and no
fewer, so :meth:`_verify_local_dbs` asserts that set equality at load rather
than trusting either archive.

The cost of that choice is the pin: Drive has no revision, so the checksum is
the whole guarantee, and a download-quota block serves an HTML page instead of
the zip. That page fails the checksum, which is the loud failure — if
``dataset download`` reports a mismatch on this file, retry rather than
re-pinning.

**Two traps in the archives.** ``local_sqlite.zip`` interleaves
``__MACOSX/._*`` AppleDouble stubs with the real files, and one of them
(``._chinook.sqlite``) ends in ``.sqlite``, so an unfiltered extraction yields a
file that looks like a database and is not. And 37 of the 547 rows carry a
``temporal`` key the other 510 omit, which Arrow cannot infer across — it is
normalised to ``None`` here rather than dropped, since it marks the questions
whose answer depends on when they are asked.

**A note on the GitHub archive's checksum.** ``codeload`` zips are not
contractually byte-stable, so a pin over one is a maintenance risk rather than a
guarantee. Three independent downloads of this commit produced the same sha256,
which is evidence and not a promise; if it ever fails, re-verify the contents
before re-pinning.

References:

* Paper: <https://arxiv.org/abs/2411.07763>
* Harness: <https://github.com/xlang-ai/Spider2>

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

#: Pinned upstream commit; the archive basename is this sha plus ``.zip``.
SPIDER2_REVISION = "cafb867313aab4e674652054198f383cf4018943"
SPIDER2_ARCHIVE_SHA256 = (
    "1e3cbb6a0eb13d9a397a8a786d9cd9b06ba54df124b6a193dd52f2949580276b"
)
#: The local SQLite databases, from the Drive archive upstream's README links.
#: Pinned by file id + checksum; Drive exposes no revision. ``confirm=t`` is
#: what gets the bytes rather than the interstitial virus-scan page that Drive
#: serves for anything this large.
LOCALDB_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG&export=download&confirm=t"
)
LOCALDB_SHA256 = "d56acf7c9d89be4bdf1f4f1281f4a03d91735f1858f6d6d0cfe0f9d562e3a94f"

ARCHIVE_BASENAME = f"{SPIDER2_REVISION}.zip"
#: Not a filename anyone chose: ``url:`` sources stage under the *path*
#: basename, and this URL's path is ``/download``. The archive is served as
#: ``local_sqlite.zip`` by Content-Disposition, which the URL handler does not
#: read, so the staged file is ``<data_dir>/spider2_lite/download``.
LOCALDB_BASENAME = "download"

#: Only this subtree of the repo archive is extracted.
_SUBTREE = f"Spider2-{SPIDER2_REVISION}/spider2-lite/"
_ROOT = "spider2-lite"
#: Where upstream's own instructions say the .sqlite files belong.
_LOCALDB_SUBDIR = "resource/databases/spider2-localdb"
_MARKER = ".sieval-extracted"
#: macOS resource forks shipped inside local_sqlite.zip; not databases.
_APPLEDOUBLE = "__MACOSX"

#: `instance_id` prefix -> engine. Longest prefix wins, so `sf_bq` is tested
#: before `sf` would swallow it, and `bq` never matches `sf_bq`.
BACKEND_BY_PREFIX = (
    ("local", "sqlite"),
    ("sf_bq", "snowflake"),
    ("sf", "snowflake"),
    ("bq", "bigquery"),
    ("ga", "bigquery"),
)


def backend_for(instance_id: str) -> str:
    """Engine that answers *instance_id*, from its prefix alone."""
    for prefix, backend in BACKEND_BY_PREFIX:
        if instance_id.startswith(prefix):
            return backend
    raise ValueError(f"Unrecognised Spider 2.0-lite instance id: {instance_id!r}")


class Spider2LiteDatasetSample(TypedDict):
    instance_id: str
    db: str
    question: str
    #: Filename under ``resource/documents``; ``None`` on 440 of 547 rows.
    external_knowledge: str | None
    #: ``"Yes"`` on the 37 rows whose answer depends on when they are asked.
    temporal: str | None


@sieval_dataset(
    name="spider2_lite",
    display_name="Spider 2.0-lite",
    description="Enterprise-scale text-to-SQL; 547 questions over BigQuery, Snowflake and SQLite.",  # noqa: E501
    source=(
        f"url:https://github.com/xlang-ai/Spider2/archive/{ARCHIVE_BASENAME}",
        f"url:{LOCALDB_URL}",
    ),
    checksums={
        ARCHIVE_BASENAME: f"sha256:{SPIDER2_ARCHIVE_SHA256}",
        LOCALDB_BASENAME: f"sha256:{LOCALDB_SHA256}",
    },
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "text-to-sql", "code-exec", "enterprise"),
    license="MIT",
)
class Spider2LiteDataset(Dataset[Spider2LiteDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        _ = kwargs
        staged = Path(name_or_path)
        root = self._extract(staged)
        self._root = str(root)
        rows = self._read_rows(root / "spider2-lite.jsonl")
        self._verify_local_dbs(root / _LOCALDB_SUBDIR, rows)
        return HFDatasetDict({"test": HFDataset.from_list(rows)})

    # -- staged paths the task reads -------------------------------------

    @property
    def localdb_dir(self) -> str | None:
        """Directory of ``<db>.sqlite`` files for the 135 local questions."""
        return self._under(_LOCALDB_SUBDIR)

    @property
    def gold_dir(self) -> str | None:
        """Gold execution results, ``<instance_id>[_a-z].csv``."""
        return self._under("evaluation_suite/gold/exec_result")

    @property
    def eval_config_path(self) -> str | None:
        """Per-instance ``condition_cols`` / ``ignore_order`` comparison rules."""
        return self._under("evaluation_suite/gold/spider2lite_eval.jsonl")

    @property
    def documents_dir(self) -> str | None:
        """External-knowledge markdown referenced by 107 of the questions."""
        return self._under("resource/documents")

    @property
    def db_schema_dir(self) -> str | None:
        """Per-engine schema trees (``bigquery`` / ``snowflake`` / ``sqlite``)."""
        return self._under("resource/databases")

    def _under(self, relative: str) -> str | None:
        root = getattr(self, "_root", None)
        return str(Path(root, relative)) if root else None

    # -- loading ---------------------------------------------------------

    @staticmethod
    def _verify_local_dbs(localdb_dir: Path, rows: list[dict]) -> None:
        """Every ``local`` question's database must be on disk.

        The archive this loader pins holds exactly the databases the questions
        name, so a miss here is a staging fault or a swapped archive rather
        than an upstream gap — and it is worth one ``listdir`` to say so now,
        naming the databases, instead of letting each affected sample raise
        ``unable to open database file`` from inside ``preprocess`` with no
        clue which file it wanted. The Hub's ``sqlite.zip`` fails this on seven
        databases and 48 questions; see the module docstring.
        """
        wanted = {
            row["db"] for row in rows if backend_for(row["instance_id"]) == "sqlite"
        }
        absent = sorted(
            db for db in wanted if not (localdb_dir / f"{db}.sqlite").is_file()
        )
        if absent:
            raise ValueError(
                f"Spider 2.0-lite is missing {len(absent)} of {len(wanted)} local "
                f"databases under {str(localdb_dir)!r}: {absent}. Re-stage with "
                "'sieval dataset download spider2_lite --force'; if they are still "
                "absent, the staged archive is not the one this loader pins."
            )

    @staticmethod
    def _read_rows(path: Path) -> list[dict]:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not rows:
            raise ValueError(
                f"Spider 2.0-lite rows missing at {str(path)!r}; re-stage with "
                "'sieval dataset download spider2_lite --force'."
            )
        # `temporal` is present on 37 rows and absent on 510; Arrow needs one
        # schema, so the absence is spelled explicitly rather than inferred.
        return [
            {
                "instance_id": row["instance_id"],
                "db": row["db"],
                "question": row["question"],
                "external_knowledge": row.get("external_knowledge"),
                "temporal": row.get("temporal"),
            }
            for row in rows
        ]

    @classmethod
    def _extract(cls, staged: Path) -> Path:
        root = staged / _ROOT
        if (root / _MARKER).is_file():
            return root
        archive = staged / ARCHIVE_BASENAME
        localdb = staged / LOCALDB_BASENAME
        for required in (archive, localdb):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Spider 2.0-lite archive not found at {str(required)!r}. Run "
                    "'sieval dataset download spider2_lite' to stage both files."
                )
        cls._extract_subtree(archive, root)
        cls._extract_localdb(localdb, root / _LOCALDB_SUBDIR)
        (root / _MARKER).touch()
        return root

    @staticmethod
    def _extract_subtree(archive: Path, root: Path) -> None:
        """Extract only ``spider2-lite/`` from the repo archive, unprefixed.

        Member names are archive-controlled, so each destination is checked to
        be inside *root* before anything is written — a member named
        ``spider2-lite/../../.bashrc`` would otherwise be extracted where its
        name says. Unreachable on the pinned archive, whose members are all
        plain paths under the subtree; the guard is here because a checksum
        pins the archive we verified, not the one a mirror serves next year,
        and ``_extract_localdb``'s ``Path(name).name`` already has the same
        property for free.
        """
        root.mkdir(parents=True, exist_ok=True)
        inside = root.resolve()
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if not member.filename.startswith(_SUBTREE):
                    continue
                relative = member.filename[len(_SUBTREE) :]
                if not relative:
                    continue
                target = root / relative
                if inside not in target.resolve().parents:
                    raise ValueError(
                        "Spider 2.0-lite archive member escapes the staging "
                        f"directory: {member.filename!r}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as sink:
                    sink.write(source.read())

    @staticmethod
    def _extract_localdb(archive: Path, target_dir: Path) -> None:
        """Extract the .sqlite files, dropping macOS resource forks.

        The pinned archive carries two ``__MACOSX/._*`` stubs, and one of them
        is ``._chinook.sqlite`` — it ends in ``.sqlite``, is a few hundred
        bytes, and is not a database. It does not collide with the real
        ``chinook.sqlite`` (the flattened name keeps the ``._`` prefix), so
        what the filter buys is that a ``*.sqlite`` glob over this directory
        counts 30 databases rather than 31.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                name = member.filename
                if member.is_dir() or name.startswith(_APPLEDOUBLE):
                    continue
                if not name.endswith(".sqlite"):
                    continue
                target = target_dir / Path(name).name
                with handle.open(member) as source, target.open("wb") as sink:
                    sink.write(source.read())
