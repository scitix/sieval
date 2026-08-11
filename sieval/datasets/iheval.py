"""IHEval dataset loader — the instruction-hierarchy benchmark, flattened.

Upstream ships the benchmark as 47 separate ``input_data.json`` files under
``iheval/``, one per *cell*, where a cell is a
``(category, subtask, setting, variant)`` tuple:

* **category** — ``rule-following``, ``task-execution``, ``safety``, ``tool-use``
* **subtask** — the nine tasks, two or three per category
* **setting** — ``reference`` (all instructions merged into one user message),
  ``aligned`` (low-priority inputs agree with the system message), ``conflict``
  (they contradict it)
* **variant** — the strictness knob, e.g. ``system_translation_strong`` vs
  ``..._weak``. Upstream calls this the *prompt setting*; it is a directory name
  with no shared vocabulary across subtasks.

They are loaded into a single ``test`` split, ordered by cell path and then by
upstream row order, with the four coordinates as columns. Scoring needs every
cell of a subtask *together* (a task's headline is a mean over its settings), so
splitting them into separate registered datasets would put the aggregation
outside any task's reach. Narrow with ``filter`` on ``category`` / ``subtask`` /
``setting`` when a run wants one slice.

Two columns are JSON-encoded strings rather than structured fields, because
their shape is per-subtask and Arrow needs one schema per column:

* ``answer_json`` — ``{instruction_id_list, kwargs}`` for rule-following,
  ``{access_code, label, system_prompt}`` for safety, ``{task, content}`` for
  ``get-webpage``, a bare string for translation / verb-extract / slack-user,
  and a string *or* list of accepted names for lang-detect.
* ``tool_json`` — ``{definition, call, return}`` for the tool-use cells, empty
  everywhere else.

Two upstream fields are dropped rather than carried as columns nothing consumes.
Neither is read by anything under upstream's ``src/``, and neither is an answer
key:

* ``summary`` (1,920 lang-detect rows) — the summary the *conflicting* user
  instruction asks for, so it is reference material for the instruction the model
  is supposed to refuse, not for grading it.
* ``original_prompt`` (1,082 rows) — the pre-conflict phrasing the row was built
  from, provenance for how the benchmark was constructed.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
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

IHEVAL_REVISION = "89d71d7b9522740e0d1c871752fd3cdbcee13131"

# Upstream's data root inside the repo. Everything else in the snapshot is
# paper figures.
_DATA_SUBDIR = "iheval"

# The three settings, in the order the paper reports them. Used to sort cells so
# a run's sample order does not depend on the filesystem's alphabetical accident
# ("aligned" < "conflict" < "reference" would bury the baseline last).
_SETTING_ORDER = ("reference", "aligned", "conflict")


class IHEvalDatasetSample(TypedDict):
    """One IHEval row, tagged with the cell it came from.

    Attributes:
        uid: ``<category>/<subtask>/<setting>/<variant>#<sample_id>``. Upstream
            ids are only unique within a cell (and are ints in some cells,
            strings in others), so this is the stable per-row key.
        category: One of the four scenarios.
        subtask: One of the nine tasks.
        setting: ``reference`` / ``aligned`` / ``conflict``.
        variant: Upstream's prompt-setting directory name within the setting.
        sample_id: Upstream's own ``id``, stringified.
        system: The system message, or ``""`` when the cell has none (every
            ``reference`` cell, which merges it into the user message instead).
        conversation_history: Alternating user/assistant turns preceding the
            request, oldest first. Empty except in multi-turn rule-following.
        instruction: The current user message.
        tool_json: JSON ``{definition, call, return}`` for a prefilled tool
            round-trip, or ``""`` when the row has no tool.
        answer_json: JSON-encoded answer key; shape varies by subtask.
    """

    uid: str
    category: str
    subtask: str
    setting: str
    variant: str
    sample_id: str
    system: str
    conversation_history: list[str]
    instruction: str
    tool_json: str
    answer_json: str


def _cell_sort_key(cell: tuple[str, str, str, str]) -> tuple[str, str, int, str]:
    category, subtask, setting, variant = cell
    return (category, subtask, _SETTING_ORDER.index(setting), variant)


def _iter_cells(data_root: Path) -> list[tuple[tuple[str, str, str, str], Path]]:
    """Find every ``<category>/<subtask>/<setting>/<variant>/input_data.json``.

    Globbed at a fixed depth rather than recursively: the same tree also holds
    the source corpora each subtask was built from (``mgsm_en_es.json``,
    ``ontonotes_250.json``, ``probe_questions.json``, ...), which are not cells
    and have no ``answer`` field.
    """
    cells = []
    for path in data_root.glob("*/*/*/*/input_data.json"):
        variant = path.parent.name
        setting = path.parent.parent.name
        subtask = path.parent.parent.parent.name
        category = path.parent.parent.parent.parent.name
        if setting not in _SETTING_ORDER:
            continue
        cells.append(((category, subtask, setting, variant), path))
    if not cells:
        raise FileNotFoundError(
            f"No IHEval cells under {data_root}. "
            "Run `sieval dataset download iheval` first."
        )
    return sorted(cells, key=lambda item: _cell_sort_key(item[0]))


def _read_cell(
    cell: tuple[str, str, str, str], path: Path
) -> list[IHEvalDatasetSample]:
    category, subtask, setting, variant = cell
    rows: list[IHEvalDatasetSample] = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        sample_id = str(raw["id"])
        system = raw.get("system")
        tool = raw.get("tool")
        rows.append(
            IHEvalDatasetSample(
                uid=f"{category}/{subtask}/{setting}/{variant}#{sample_id}",
                category=category,
                subtask=subtask,
                setting=setting,
                variant=variant,
                sample_id=sample_id,
                system=system if system is not None else "",
                conversation_history=list(raw.get("conversation_history") or []),
                instruction=raw["instruction"],
                tool_json=json.dumps(tool, ensure_ascii=False) if tool else "",
                answer_json=json.dumps(raw["answer"], ensure_ascii=False),
            )
        )
    return rows


@sieval_dataset(
    name="iheval",
    display_name="IHEval",
    description=(
        "Instruction hierarchy — 9 tasks under aligned and conflicting inputs."
    ),
    source=f"hf:zhihz0535/IHEval@{IHEVAL_REVISION}",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("english", "chinese", "spanish", "open-ended", "safety", "tool-use"),
    license="CC-BY-NC-ND-4.0",
)
class IHEvalDataset(Dataset[IHEvalDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # Read the staged files directly instead of `load_dataset`: the 47 cells
        # have mutually incompatible schemas (`answer` is a dict, a string, or a
        # list depending on the subtask), so no single Arrow schema spans them
        # until the per-subtask fields are JSON-encoded here.
        data_root = Path(name_or_path) / _DATA_SUBDIR
        rows: list[IHEvalDatasetSample] = []
        for cell, path in _iter_cells(data_root):
            rows.extend(_read_cell(cell, path))
        # No train split: IHEval is 0-shot only and every cell is eval data.
        return HFDatasetDict({"test": HFDataset.from_list([dict(row) for row in rows])})
