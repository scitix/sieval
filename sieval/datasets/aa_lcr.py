"""AA-LCR dataset loader (Artificial Analysis Long Context Reasoning).

AA-LCR is a 100-question benchmark of hard, text-only reasoning questions, each
answered against a set of real-world documents (~100k tokens per set, 234
documents across 30 sets) whose answers must be reasoned across multiple
sources. The Hub repo ships ``AA-LCR_Dataset.csv`` (one row per question) plus
``extracted_text/AA-LCR_extracted-text.zip`` holding the pre-extracted document
text at ``lcr/{document_category}/{document_set_id}/{filename}``.

The upstream ``hf:`` mirror stages both files; ``load`` reads the CSV and, for
each question, pulls its documents straight from the archive (no on-disk
extraction — the CMMLU loader pattern) in ``data_source_filenames`` order, which
is significant: the card requires documents be prompted in filename order. The
ordered document texts are attached to each sample as ``documents`` so the task
is self-contained, mirroring DROP's context-in-sample approach.

``answer`` is kept verbatim from the CSV (sometimes a single value, sometimes a
``\\n``-separated ranked list); the LLM equality checker consumes it as-is.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import csv
import io
import unicodedata
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
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (current `main` at integration time).
AA_LCR_REVISION = "bdae010bbce259820c0e34c1d7cce210d966fb75"

_CSV_FILENAME = "AA-LCR_Dataset.csv"
_ZIP_RELPATH = Path("extracted_text") / "AA-LCR_extracted-text.zip"
# Documents live under this top-level dir inside the archive.
_ZIP_DOC_ROOT = "lcr"

# ZIP general-purpose bit 11: filename/comment are UTF-8 (vs. legacy cp437).
_ZIP_UTF8_FLAG = 0x800


def _member_utf8_name(info: zipfile.ZipInfo) -> str:
    """Return *info*'s intended UTF-8 name, NFC-normalized.

    Two archive quirks are handled so lookups match the CSV's
    ``data_source_filenames`` verbatim:

    * Encoding: ``zipfile`` decodes names as UTF-8 only when bit 0x800 is set,
      else cp437. This archive omits the flag on UTF-8 names, so cp437-decoded
      entries are round-tripped back through cp437 bytes and re-decoded as UTF-8;
      genuine cp437 (or already-UTF-8) names fall through unchanged.
    * Normalization: some members are stored NFD (e.g. ``ş`` as ``s`` + combining
      cedilla) while the CSV is NFC — so the recovered name is normalized to NFC.
    """
    if info.flag_bits & _ZIP_UTF8_FLAG:
        name = info.filename
    else:
        try:
            name = info.filename.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            name = info.filename
    return unicodedata.normalize("NFC", name)


class AALCRDatasetSample(TypedDict):
    question_id: int
    document_category: str
    document_set_id: str
    question: str
    answer: str
    # Document texts in `data_source_filenames` order (prompt order matters).
    documents: list[str]
    # Raw semicolon-joined filenames, kept for provenance.
    data_source_filenames: str
    input_tokens: int


@sieval_dataset(
    name="aa_lcr",
    display_name="AA-LCR",
    description="AA-LCR — 100-question long-context multi-document reasoning.",
    source=f"hf:ArtificialAnalysis/AA-LCR@{AA_LCR_REVISION}",
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "long-context", "reasoning", "open-ended"),
    license="apache-2.0",
)
class AALCRDataset(Dataset[AALCRDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        _ = kwargs
        # `hf:` stages the repo as a directory; that is the only layout the
        # runtime hands us, so accept only it (fail loudly if the files are
        # absent rather than probing speculative alternative layouts).
        staged = Path(name_or_path)
        csv_path = staged / _CSV_FILENAME
        zip_path = staged / _ZIP_RELPATH

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"AA-LCR CSV not found at {str(csv_path)!r}. Run "
                "'sieval dataset download aa_lcr' to stage the dataset."
            )
        if not zip_path.is_file():
            raise FileNotFoundError(
                f"AA-LCR extracted-text archive not found at {str(zip_path)!r}. "
                "Run 'sieval dataset download aa_lcr' to stage the dataset."
            )

        with (
            zip_path.open("rb") as zip_bytes,
            zipfile.ZipFile(zip_bytes) as archive,
            csv_path.open(newline="", encoding="utf-8") as csv_file,
        ):
            member_index = self._index_members(archive)
            rows = [
                self._build_sample(row, archive, member_index)
                for row in csv.DictReader(csv_file)
            ]

        dataset = HFDatasetDict({"test": HFDataset.from_list([{**r} for r in rows])})
        dataset = ensure_dataset_dict(dataset)
        if len(dataset["test"]) == 0:
            raise ValueError(
                f"AA-LCR produced an empty 'test' split from {str(csv_path)!r}; "
                "check that the dataset has been downloaded via "
                "'sieval dataset download aa_lcr'."
            )
        return dataset

    def _build_sample(
        self,
        row: dict[str, str],
        archive: zipfile.ZipFile,
        member_index: dict[str, zipfile.ZipInfo],
    ) -> AALCRDatasetSample:
        category = (row["document_category"] or "").strip()
        set_id = (row["document_set_id"] or "").strip()
        filenames_raw = row["data_source_filenames"] or ""
        filenames = [f.strip() for f in filenames_raw.split(";") if f.strip()]
        documents = [
            self._read_document(archive, member_index, category, set_id, filename)
            for filename in filenames
        ]
        return {
            "question_id": int(row["question_id"]),
            "document_category": category,
            "document_set_id": set_id,
            "question": row["question"] or "",
            "answer": row["answer"] or "",
            "documents": documents,
            "data_source_filenames": filenames_raw,
            "input_tokens": int(row["input_tokens"]),
        }

    @staticmethod
    def _index_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
        """Map each file member's UTF-8 path to its ``ZipInfo``.

        The archive stores names without the UTF-8 flag (bit 0x800), so
        ``zipfile`` decodes the original UTF-8 bytes as cp437 — mangling
        non-ASCII names (e.g. ``EU’s`` -> ``EUΓÇÖs``). We recover the intended
        name by re-encoding cp437 back to bytes and decoding as UTF-8 so lookups
        match the CSV's ``data_source_filenames`` verbatim; members must be
        opened by ``ZipInfo`` (their raw name is the mojibake form).
        """
        index: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            index[_member_utf8_name(info)] = info
        return index

    @staticmethod
    def _read_document(
        archive: zipfile.ZipFile,
        member_index: dict[str, zipfile.ZipInfo],
        category: str,
        set_id: str,
        filename: str,
    ) -> str:
        member = unicodedata.normalize(
            "NFC", f"{_ZIP_DOC_ROOT}/{category}/{set_id}/{filename}"
        )
        info = member_index.get(member)
        if info is None:
            raise FileNotFoundError(
                f"AA-LCR document {member!r} missing from the extracted-text "
                "archive; the CSV and archive are out of sync."
            )
        with archive.open(info) as raw_file:
            return io.TextIOWrapper(raw_file, encoding="utf-8").read()
