"""Multi-IF dataset loader (multi-turn, multilingual instruction following).

Multi-IF extends IFEval into three-turn conversations across eight languages
(English, French, Hindi, Portuguese, Spanish, Russian, Italian, Chinese),
4,501 conversations in all. Each later turn adds a constraint while keeping the
earlier ones, so a turn is graded against the *accumulated* constraint set --
verified against the pinned revision: turn ``t``'s ``instruction_id_list`` is a
prefix-extension of turn ``t-1``'s for every row, with no exceptions.

The Hub repo ships one file, ``multiIF_20241018.csv``, with nine turn columns
(``turn_{1,2,3}_{prompt,instruction_id_list,kwargs}``) plus ``key``,
``language``, and three eval-time placeholders (``turns``, ``responses``,
``turn_index``) that are empty in the release and are dropped here -- upstream
writes them back as it walks the conversation, which is state this loader has no
reason to carry.

The nine flat columns are reshaped into a ``turns`` list because the turn count
genuinely varies: 56 rows (French 30, English 13, Hindi 13) have no third turn,
and upstream skips those rows when scoring turn 3 rather than padding them. A
list makes that the natural ``len(turns) == 2``.

Field-by-field, relative to the CSV:

* ``turn_N_prompt`` is a JSON-encoded chat message. Every one of the 13,447
  turn prompts in the pinned revision is ``{"role": "user", ...}`` with no other
  keys, so only ``content`` is kept and the role is re-attached by the task.
* ``turn_N_instruction_id_list`` is a JSON list of strings, decoded here.
* ``turn_N_kwargs`` is a JSON list of *JSON strings* -- double-encoded upstream,
  and each inner object holds only the keys its own checker needs (24 distinct
  keys across the set). The outer list is decoded; the inner objects are
  deliberately left encoded, because decoding them would make Arrow unify 24
  sparse struct fields across every constraint. IFEval hits the same wall and
  works around it by stripping ``None``s at use time; keeping the payload opaque
  avoids it outright. The task decodes each element immediately before handing
  it to ``build_description``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from pathlib import Path
from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (current `main` at integration time).
MULTI_IF_REVISION = "0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa"

_CSV_FILENAME = "multiIF_20241018.csv"
_MAX_TURNS = 3


class MultiIFTurn(TypedDict):
    """One user turn and the constraints its response is graded against.

    Attributes:
        prompt: The user message text (``content`` of the CSV's JSON message).
        instruction_id_list: Constraint ids for this turn, cumulative -- it
            includes every earlier turn's constraints as a prefix.
        kwargs: Per-constraint arguments, positionally aligned with
            ``instruction_id_list`` and still JSON-encoded (see module docstring).
    """

    prompt: str
    instruction_id_list: list[str]
    kwargs: list[str]


class MultiIFDatasetSample(TypedDict):
    key: str
    language: str
    # Three turns, except for the 56 rows that ship only two.
    turns: list[MultiIFTurn]


@sieval_dataset(
    name="multi_if",
    display_name="Multi-IF",
    description=(
        "Multi-IF — 4,501 three-turn, eight-language instruction-following "
        "conversations extending IFEval."
    ),
    source=f"hf:facebook/Multi-IF@{MULTI_IF_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("multilingual", "multi-turn", "open-ended"),
    # The Hub dataset card's own license, which is NOT the Apache-2.0 of the
    # facebookresearch/Multi-IF *code* repo that ships the evaluator.
    license="CC-BY-NC-2.0",
)
class MultiIFDataset(Dataset[MultiIFDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # `hf:` stages the repo as a directory; accept only that layout and fail
        # loudly rather than probing speculative alternatives.
        csv_path = Path(name_or_path) / _CSV_FILENAME
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Multi-IF CSV not found at {str(csv_path)!r}. Run "
                "'sieval dataset download multi_if' to stage the dataset."
            )

        # `keep_default_na=False` so an absent third turn arrives as "" rather
        # than a float nan, matching upstream's own read of this CSV and letting
        # one emptiness check cover both spellings.
        raw = load_dataset(
            "csv",
            data_files={"test": str(csv_path)},
            keep_default_na=False,
            **kwargs,
        )
        rows = [self._build_sample(row) for row in ensure_dataset_dict(raw)["test"]]

        dataset = ensure_dataset_dict(
            HFDatasetDict({"test": HFDataset.from_list([{**r} for r in rows])})
        )
        if len(dataset["test"]) == 0:
            raise ValueError(
                f"Multi-IF produced an empty 'test' split from {str(csv_path)!r}; "
                "check that the dataset has been downloaded via "
                "'sieval dataset download multi_if'."
            )
        return dataset

    def _build_sample(self, row: dict) -> MultiIFDatasetSample:
        turns: list[MultiIFTurn] = []
        for index in range(1, _MAX_TURNS + 1):
            turn = self._build_turn(row, index)
            if turn is None:
                # Only the third turn is ever absent in the pinned revision, and
                # a conversation cannot resume after a gap -- so stop rather than
                # skip, which would silently splice turn 3 onto turn 1.
                break
            turns.append(turn)
        if not turns:
            # Every row of the pinned revision has at least two turns, so an empty
            # conversation means the CSV is not the one this loader pins. Fail
            # here rather than letting the task raise IndexError on `turns[0]`.
            raise ValueError(
                f"Multi-IF row {str(row['key'])!r} has no first turn; expected "
                f"'turn_1_prompt' to be a JSON chat message, got "
                f"{str(row['turn_1_prompt'])!r}."
            )
        return {
            "key": row["key"],
            "language": row["language"],
            "turns": turns,
        }

    def _build_turn(self, row: dict, index: int) -> MultiIFTurn | None:
        prompt = row[f"turn_{index}_prompt"]
        # Upstream treats both "" and the literal string "None" as "no turn".
        if not prompt or prompt == "None":
            return None
        return {
            "prompt": json.loads(prompt)["content"],
            "instruction_id_list": json.loads(row[f"turn_{index}_instruction_id_list"]),
            # Elements stay JSON-encoded on purpose (see module docstring).
            "kwargs": json.loads(row[f"turn_{index}_kwargs"]),
        }
