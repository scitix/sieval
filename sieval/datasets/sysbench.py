"""SysBench — 500 Chinese multi-turn sessions with per-turn constraint checklists.

Upstream: PKU-Baichuan-MLSystemLab/SysBench, paper arXiv:2408.10943 "SysBench: Can
Large Language Models Follow System Messages?". Each session pairs one hand-written
system prompt with a 5-turn conversation; every turn carries a ``criteria`` checklist
(the atomic constraints that turn's reply must satisfy, drawn from the system prompt)
and an ``alignment`` label saying whether the turn's user request agrees with the
system prompt or pulls against it.

**One row is one SESSION, not one turn**, because the benchmark's headline protocol
feeds the model its own prior replies: ``eval_system_bench.py``'s ``do_infer`` appends
each generated answer to ``messages`` before asking the next turn, so the five turns are
one sequential walk and cannot be sampled or scored independently. That is also what
makes the session-level metric (SSR) a property of a row rather than a reassembly.
The ground-truth assistant turns are carried alongside anyway, because upstream's
``eval_system_bench_with_gt.py`` ablation replaces the history with them, and the task
exposes that as an explicitly-named mode.

``turns`` is carried as a JSON string. That is not laziness about the schema: a criteria
dict is keyed by variable integer ids, so a typed column would differ in shape from row
to row and Arrow would either reject it or widen every row to the union. A JSON string
keeps one flat, stable columnar schema across the whole set.

**Upstream declares no license.** Verified at the pinned commit: the repository tree has
no LICENSE file and GitHub reports no license for it, so ``license`` is null rather than
guessed. sieval does not redistribute the data — ``source`` points at upstream's own
file and ``sieval dataset download`` fetches it — but a downstream user should read that
as "all rights reserved by default" and check with the authors before redistributing it.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
import os
from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

#: Upstream commit the data file and the judge prompt are both pinned to.
SYSBENCH_COMMIT = "627ffa8010d00e270426975b33b1fb7a0a635602"

#: The upstream data file, addressed at the pinned commit rather than at a branch: a
#: branch would let the rows move under a checksum that no longer matches, which fails
#: closed but only after a download.
SYSBENCH_URL = (
    "https://raw.githubusercontent.com/PKU-Baichuan-MLSystemLab/SysBench/"
    f"{SYSBENCH_COMMIT}/datas/system_benchmark_eval_datas.json"
)

_DATA_FILE = "system_benchmark_eval_datas.json"


class SysBenchDatasetSample(TypedDict):
    session_id: int
    domain: str
    scenario: str
    system_prompt: str
    n_turns: int
    # JSON list of {"user", "assistant", "alignment", "criteria"} — the ground-truth
    # assistant reply is kept for the with-GT-history ablation, never for scoring.
    turns_json: str


@sieval_dataset(
    name="sysbench",
    display_name="SysBench (system-message following)",
    description=(
        "500 Chinese 5-turn sessions with per-turn constraint checklists, one "
        "row per session."
    ),
    source=f"url:{SYSBENCH_URL}",
    checksums={
        _DATA_FILE: "sha256:d58186c227d8284e153619005cea704010e7290b4d3329c23445c6b504de1af2",  # noqa: E501
    },
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("chinese", "multi-turn", "open-ended", "system-following"),
    # Upstream declares none -- see the module docstring. Null is the measured
    # answer here, not a missing field.
    license=None,
)
class SysBenchDataset(Dataset[SysBenchDatasetSample]):
    """SysBench, one row per session.

    ``name_or_path`` may be the downloaded directory or the JSON file itself, so a run
    works both from ``sieval dataset download`` and from a hand-placed copy.
    """

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        path = (
            os.path.join(name_or_path, _DATA_FILE)
            if os.path.isdir(name_or_path)
            else name_or_path
        )
        with open(path, encoding="utf-8") as fh:
            sessions = json.load(fh)

        rows = [_session_row(session) for session in sessions]
        rows.sort(key=lambda r: r["session_id"])

        ds = HFDataset.from_list([dict(row) for row in rows])
        # Both splits are the same rows: SysBench ships no train/test division, and a
        # task asking for `train` should get the data rather than an empty split.
        return HFDatasetDict({"train": ds, "test": ds})


def _session_row(session: dict) -> SysBenchDatasetSample:
    """One upstream session -> one row, turns in conversation order."""
    prompt_infos = session.get("prompt_infos", {})
    turns: list[dict] = []
    for i, msg in enumerate(session["messages"]):
        if msg["role"] != "assistant":
            continue
        user_msg = session["messages"][i - 1]
        info = prompt_infos.get(user_msg["content"], {})
        turns.append(
            {
                "user": user_msg["content"],
                "assistant": msg["content"],
                "alignment": _text(info.get("alignment")),
                # A turn upstream left unlabelled keeps its place in the walk and is
                # skipped by the scorer -- it still shapes the history of the turns
                # after it, so dropping it would change what they were asked from.
                "criteria": info.get("criteria", {}),
            }
        )
    return {
        "session_id": session["system_id"],
        "domain": _text(session.get("领域")),
        "scenario": _text(session.get("场景")),
        "system_prompt": _text(session["system_prompt"]),
        "n_turns": len(turns),
        "turns_json": json.dumps(turns, ensure_ascii=False),
    }


def _text(value: object) -> str:
    """Coerce to str, mapping absent/NaN to "".

    Upstream's ``场景`` column is NaN for some sessions rather than absent, and a NaN
    float in a string column makes Arrow reject the batch.
    """
    return value if isinstance(value, str) else ""
