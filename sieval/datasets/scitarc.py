"""SciTaRC dataset loader (JHU-CLSP).

SciTaRC (Wang et al., COLM 2026, arXiv:2603.08910) is 371 expert-authored
questions over raw LaTeX tables lifted from arXiv papers. It targets composite
reasoning — chaining selection, iteration, arithmetic and conditionals — rather
than single-cell lookup, and every question ships an expert-written pseudo-code
plan so planning can be scored apart from execution. The Hub repo exposes a
single ``test`` split; this loader mirrors it as-is.

Schema, measured over all 371 rows at the pinned revision: seven columns, all
strings or nested string lists. ``paper``, ``question`` and ``answer`` are
populated on every row; ``plan`` is null on exactly one, hence ``str | None``.
``relevant_tables`` is non-empty everywhere, so no row prompts against a blank
table block.

``relevant_tables`` and ``tables`` are ``list[list[str]]`` — a list of tables,
each a list of LaTeX source lines that already carry their own newlines, joined
inner-with-nothing and outer-with-a-blank-line by
``sieval.community.scitarc.get_table_text``. Re-joining them with ``"\\n"``
would double every line break.

``fulltext`` (~50 KB per row) and ``tables`` (every table in the paper, not just
the ones the question needs) are kept rather than dropped: they are what a
future full-paper or table-retrieval reading would consume, and the whole split
is under 14 MB. ``scitarc_0shot_gen`` reads ``relevant_tables`` only, matching
upstream's ``generate.py``.

Licensing is split: the data is CC-BY-NC-4.0 (non-commercial), while upstream's
harness code — vendored in ``sieval.community.scitarc`` — is MIT.

References:

* Paper: <https://arxiv.org/abs/2603.08910>
* Harness: <https://github.com/JHU-CLSP/SciTaRC>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (`main` at integration time).
SCITARC_REVISION = "8be5661f333c5d2342f264b763396524d30fb98c"


class SciTaRCDatasetSample(TypedDict):
    paper: str
    relevant_tables: list[list[str]]
    tables: list[list[str]]
    fulltext: str
    question: str
    answer: str
    plan: str | None


@sieval_dataset(
    name="scitarc",
    display_name="SciTaRC",
    description=(
        "SciTaRC — composite multi-step QA over raw LaTeX tables from arXiv papers."
    ),
    source=f"hf:jhu-clsp/SciTaRC@{SCITARC_REVISION}",
    categories=(
        Category(Level1Category.LOGIC, "ComplexLogic"),
        Category(Level1Category.KNOWLEDGE, "STEM"),
    ),
    tags=("english", "tabular", "open-ended"),
    # The DATA license, from the Hub card and the repo's own badge split. The
    # harness code is MIT; that governs `sieval.community.scitarc`, not this.
    license="CC-BY-NC-4.0",
)
class SciTaRCDataset(Dataset[SciTaRCDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))
        if "test" in dataset and len(dataset["test"]) == 0:
            raise ValueError(
                "SciTaRC produced an empty 'test' split; check that the dataset "
                "has been downloaded via `sieval dataset download scitarc`."
            )
        return dataset
