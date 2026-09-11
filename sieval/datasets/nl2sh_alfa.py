"""NL2SH-ALFA dataset loader (Westenfelder et al., NAACL 2025).

The test set for natural-language-to-Bash translation: 300 English instructions,
each with two manually verified commands, curated over 100+ hours by MIT-CSAIL's
ALFA group. Scoring is execution-based, through the InterCode-ALFA harness.

**300 rows, not 600.** The card and the paper both say "600 instruction-command
pairs", and at the pinned revision the ``test`` config holds exactly **300 rows**
with difficulty 0/1/2 at 100 each. 600 counts *pairs*: 300 instructions times the
two verified commands (``bash``, ``bash2``). The harness agrees -- its
``index_to_img`` split table sums to 300 -- and upstream's own scripts divide by
``len(dataset)``. This matters beyond bookkeeping: ``index_to_img`` raises for
``index >= 300`` and ``submit_command`` swallows that into a score of **0**, so a
caller who believes the 600 figure scores zero on half a run with no error raised.

**Two ground truths, from two places, and they disagree on two rows.** The
harness does not grade against this Hub dataset. It grades against its own
vendored copy of the same table (``sieval/community/intercode_alfa/assets/``),
while upstream's README and notebooks prompt the model from the Hub's ``nl``. At
the pinned revision and commit the two copies differ on five of 300 rows:

* index 38 -- Hub ``echo -n 'hello' | base64`` vs harness ``echo 'hello' | base64``
  (``aGVsbG8=`` against ``aGVsbG8K``, so the compared bytes differ);
* index 100 -- ``awk 'length < 20' ...`` vs ``awk 'length < 40' ...``
  (a different line set);
* indices 150, 190, 284 -- reworded instructions, identical commands.

So this loader carries **both**: the Hub's ``nl`` / ``bash`` / ``bash2`` as
published, and the harness's ``query`` / ``gold`` / ``gold2`` as graded, joined
positionally. That is deliberately visible rather than hidden inside the grader,
where a later reader would assume the Hub's column was the one scored. Tasks
prompt from ``nl`` and grade against ``gold``, which is what upstream does.

The join key is **row position** -- nothing else relates the two sources, and
``index_to_img`` is positional too -- so :meth:`NL2SHAlfaDataset.load` asserts
both sides are 300 rows before pairing them.

**``test`` only.** The ``train`` config (40,639 rows) is an unverified
fine-tuning set with a different two-column schema and no consumer here; it stays
reachable from the pinned source for anyone who wants it
(``load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")``). Both Hub
configs name their single split ``train``, so the mapping to sieval's ``test``
is spelled explicitly below.

References:

* Paper: <https://arxiv.org/abs/2502.06858>
* Dataset: <https://huggingface.co/datasets/westenfelder/NL2SH-ALFA>
* Harness: <https://github.com/westenfelder/InterCode-ALFA>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.community.intercode_alfa import IMAGE_SPLITS, fs_id_for_index, gold_table
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

NL2SH_ALFA_REVISION = "a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba"

#: The Hub config holding the verified test rows. Its split is named ``train``,
#: which is why both names appear in the `load_dataset` call below.
_TEST_CONFIG = "test"
_HUB_SPLIT = "train"

#: Row count at the pinned revision, and the harness's own split table sum.
EXPECTED_TEST_ROWS = sum(IMAGE_SPLITS)


class NL2SHAlfaDatasetSample(TypedDict):
    """One NL2SH-ALFA test row: the Hub's columns plus the harness's own.

    ``nl`` is the prompt and ``gold`` is the graded reference -- from two
    different upstream files, which disagree on five of the 300 rows (see the
    module docstring). ``query`` is the harness's copy of the instruction,
    carried because it is what upstream's LLM-judge FEH prompt interpolates;
    the embedding FEH never reads it. ``bash2`` / ``gold2`` are the second
    verified command, which the harness never reads at all.

    ``fs_id`` is the 1-based filesystem image this row is scored in, derived
    from its position by ``index_to_img``.
    """

    nl: str
    bash: str
    bash2: str
    difficulty: int
    query: str
    gold: str
    gold2: str
    fs_id: int


@sieval_dataset(
    name="nl2sh_alfa",
    display_name="NL2SH-ALFA",
    description=(
        "Natural language to Bash; 300 verified instructions, execution-graded."
    ),
    source=f"hf:westenfelder/NL2SH-ALFA@{NL2SH_ALFA_REVISION}",
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "nl2bash", "code-exec"),
    license="MIT",
)
class NL2SHAlfaDataset(Dataset[NL2SHAlfaDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        hub = ensure_dataset(
            load_dataset(name_or_path, _TEST_CONFIG, split=_HUB_SPLIT, **kwargs)
        )
        graded = gold_table()
        if len(hub) != EXPECTED_TEST_ROWS or len(graded) != EXPECTED_TEST_ROWS:
            raise ValueError(
                f"NL2SH-ALFA expects {EXPECTED_TEST_ROWS} rows on both sides of a "
                f"positional join, got {len(hub)} from the Hub config "
                f"{_TEST_CONFIG!r} and {len(graded)} from the vendored harness "
                "tables. Row position is the only key relating them, so a count "
                "mismatch means one side moved and the pairing can no longer be "
                "trusted -- re-pin the revision or the harness commit, do not "
                "truncate to the shorter side."
            )
        paired = HFDataset.from_list(
            [
                {
                    **row,
                    "query": gold["query"],
                    "gold": gold["gold"],
                    "gold2": gold["gold2"],
                    "fs_id": fs_id_for_index(index),
                }
                for index, (row, gold) in enumerate(zip(hub, graded, strict=True))
            ]
        )
        # Upstream's single split is named `train`; the runner evaluates
        # `Dataset.test_set`, which reads exactly this key, so the rename is what
        # makes the 300 verified rows the ones that get scored.
        return HFDatasetDict({"test": paired})
