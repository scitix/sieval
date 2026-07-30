"""Unified RULER dataset: 13 subtasks in one loader.

Each call to ``load()`` targets one subtask (or ``"all"`` to concatenate all
13). Every emitted row carries ``subtask`` and ``context_length`` fields so
``RulerZeroShotGenTask.report()`` can group and score without any external
aggregation command.

The 13 canonical subtask names (transcribed from NVIDIA/RULER's synthetic.yaml
@ab17b78; the two QA subtasks are RULER's separate qa config):
    niah_single_1, niah_single_2, niah_single_3,
    niah_multikey_1, niah_multikey_2, niah_multikey_3,
    niah_multivalue, niah_multiquery,
    vt, cwe, fwe, qa_squad, qa_hotpotqa

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from typing import NotRequired, TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._cwe import load_cwe
from ._fwe import load_fwe
from ._niah import _NIAH_SUBTASK_KWARGS, load_niah
from ._qa import load_qa
from ._shared import (
    _HOTPOTQA_REPO_ID,
    _HOTPOTQA_REVISION,
    _RULER_DATA_SHA,
    resolve_reserve_think_budget,
    tokens_to_generate,
)
from ._vt import load_vt

_ALL_SUBTASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_squad",
    "qa_hotpotqa",
)

# Subtasks that read from the base data dir rather than ``<data_dir>/ruler/``:
# FWE is fully synthetic (no external files) and qa_hotpotqa is fetched from HF.
# Everything else reads staged files under the ``ruler`` staging subdir.
_BASE_DIR_SUBTASKS = frozenset({"fwe", "qa_hotpotqa"})


def _subtask_data_path(name_or_path: str, subtask: str) -> str:
    """Resolve the data dir a single subtask reads from.

    Centralizes the ``/ruler`` staging-subdir rule so every entry point — a
    direct single-subtask ``load``, ``"all"``, and explicit subtask lists — maps
    the path identically. Previously only the aggregate branches appended
    ``/ruler``, so a direct ``subtask="niah_single_2"`` looked in the wrong dir
    and raised ``FileNotFoundError``.
    """
    return name_or_path if subtask in _BASE_DIR_SUBTASKS else f"{name_or_path}/ruler"


# Each subtask's RULER task name, for the per-subtask generation budget
# (tokens_to_generate). All 8 NIAH variants share "niah".
_NON_NIAH_RULER_TASK = {
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_squad": "qa",
    "qa_hotpotqa": "qa",
}


def _ruler_task_name(subtask: str) -> str:
    return "niah" if subtask in _NIAH_SUBTASK_KWARGS else _NON_NIAH_RULER_TASK[subtask]


class RulerDatasetSample(TypedDict):
    index: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str
    subtask: str
    context_length: int
    gen_budget: int  # per-subtask generation cap (tokens_to_generate)
    think_budget: NotRequired[int]  # thinking budget, added if enable_thinking=True
    enable_thinking: NotRequired[bool]  # whether thinking mode is enabled
    # Whether gen_budget already includes think_budget (see tokens_to_generate).
    # infer() reads this instead of re-deriving the decision from context_length,
    # so the loader stays the single owner of the reserve rule.
    think_budget_reserved: NotRequired[bool]
    token_position_answer: NotRequired[int]  # NIAH only


@sieval_dataset(
    name="ruler",
    display_name="RULER",
    description=(
        "RULER long-context benchmark: 13 subtasks (NIAH ×8, VT, CWE, FWE, QA ×2)."
    ),
    source=(
        # Staged flat as <data_dir>/ruler/<basename> like url: sources, so the
        # scheme carries the filename only — no directory component to mislead.
        "local:PaulGrahamEssays.json.gz",
        # CWE falls back to this pool once num_words exceeds the wonderwords vocab
        # (~8k words), matching upstream (see _cwe.py `_get_example`).
        f"url:https://media.githubusercontent.com/media/NVIDIA/RULER/{_RULER_DATA_SHA}/scripts/data/synthetic/json/english_words.json",
        "url:https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        f"hf:{_HOTPOTQA_REPO_ID}@{_HOTPOTQA_REVISION}",
    ),
    checksums={
        "english_words.json": "sha256:affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",  # noqa: E501
        "dev-v2.0.json": "sha256:80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",  # noqa: E501
    },
    categories=(Category(Level1Category.LANGUAGE, "SemanticUnderstanding"),),
    tags=("english", "open-ended", "long-context"),
    license="Apache-2.0",
    deps_group="ruler",
)
class RulerDataset(Dataset[RulerDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        subtask: str | list[str] | None = None,
        max_seq_length: int = 4096,
        tokenizer_type: str = "openai",
        tokenizer_path: str = "cl100k_base",
        num_samples: int = 500,
        random_seed: int = 42,
        remove_newline_tab: bool = False,
        enable_thinking: bool = False,
        think_budget: int = 0,
        model_name: str = "qwen3",
        reserve_think_budget: bool | None = None,
        # NIAH needle/haystack config is fixed per subtask by _NIAH_SUBTASK_KWARGS,
        # so it is not exposed as a load() knob (a stray value would do nothing).
        # CWE-specific
        freq_cw: int = 30,
        freq_ucw: int = 3,
        num_cw: int = 10,
        num_fewshot: int = 1,
        # VT-specific
        num_chains: int = 1,
        num_hops: int = 4,
        # FWE-specific
        alpha: float = 2.0,
        coded_wordlen: int = 6,
        vocab_size: int = -1,
        # QA-specific
        pre_samples: int = 0,
        **kwargs,
    ) -> HFDatasetDict:
        if subtask is None:
            raise ValueError("RulerDataset.load requires `subtask`")

        # A single subtask, an explicit list, and "all" all funnel through one
        # dispatch loop — a single subtask is just a one-element target list. This
        # keeps the ``/ruler`` staging-path rule (``_subtask_data_path``) and the
        # budget stamping identical across every entry point: no recursion, no
        # re-listed kwargs, and no way for the aggregate and single-subtask paths
        # to drift apart.
        if subtask == "all":
            targets = list(_ALL_SUBTASKS)
        elif isinstance(subtask, list):
            targets = subtask
        else:
            targets = [subtask]

        splits = []
        for st in targets:
            data_path = _subtask_data_path(name_or_path, st)

            if st in _NIAH_SUBTASK_KWARGS:
                niah_kwargs = _NIAH_SUBTASK_KWARGS[st]
                rows = load_niah(
                    data_path,
                    max_seq_length=max_seq_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_path=tokenizer_path,
                    num_samples=num_samples,
                    random_seed=random_seed,
                    remove_newline_tab=remove_newline_tab,
                    enable_thinking=enable_thinking,
                    think_budget=think_budget,
                    model_name=model_name,
                    reserve_think_budget=reserve_think_budget,
                    num_needle_k=niah_kwargs["num_needle_k"],
                    num_needle_v=niah_kwargs["num_needle_v"],
                    num_needle_q=niah_kwargs["num_needle_q"],
                    type_haystack=niah_kwargs["type_haystack"],
                    type_needle_k=niah_kwargs["type_needle_k"],
                    type_needle_v=niah_kwargs["type_needle_v"],
                )
            elif st == "vt":
                rows = load_vt(
                    data_path,
                    max_seq_length=max_seq_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_path=tokenizer_path,
                    num_samples=num_samples,
                    random_seed=random_seed,
                    remove_newline_tab=remove_newline_tab,
                    enable_thinking=enable_thinking,
                    think_budget=think_budget,
                    model_name=model_name,
                    reserve_think_budget=reserve_think_budget,
                    num_chains=num_chains,
                    num_hops=num_hops,
                    type_haystack="noise",
                )
            elif st == "cwe":
                rows = load_cwe(
                    data_path,
                    max_seq_length=max_seq_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_path=tokenizer_path,
                    num_samples=num_samples,
                    random_seed=random_seed,
                    remove_newline_tab=remove_newline_tab,
                    enable_thinking=enable_thinking,
                    think_budget=think_budget,
                    model_name=model_name,
                    reserve_think_budget=reserve_think_budget,
                    freq_cw=freq_cw,
                    freq_ucw=freq_ucw,
                    num_cw=num_cw,
                    num_fewshot=num_fewshot,
                )
            elif st == "fwe":
                rows = load_fwe(
                    data_path,
                    max_seq_length=max_seq_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_path=tokenizer_path,
                    num_samples=num_samples,
                    random_seed=random_seed,
                    remove_newline_tab=remove_newline_tab,
                    enable_thinking=enable_thinking,
                    think_budget=think_budget,
                    model_name=model_name,
                    reserve_think_budget=reserve_think_budget,
                    alpha=alpha,
                    coded_wordlen=coded_wordlen,
                    vocab_size=vocab_size,
                )
            elif st in ("qa_squad", "qa_hotpotqa"):
                qa_dataset = "squad" if st == "qa_squad" else "hotpotqa"
                rows = load_qa(
                    data_path,
                    dataset=qa_dataset,
                    max_seq_length=max_seq_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_path=tokenizer_path,
                    num_samples=num_samples,
                    random_seed=random_seed,
                    remove_newline_tab=remove_newline_tab,
                    enable_thinking=enable_thinking,
                    think_budget=think_budget,
                    model_name=model_name,
                    reserve_think_budget=reserve_think_budget,
                    pre_samples=pre_samples,
                )
            else:
                raise ValueError(
                    f"Unknown subtask {st!r}. Valid subtasks: {_ALL_SUBTASKS} or 'all'."
                )

            gen_budget = tokens_to_generate(
                _ruler_task_name(st),
                enable_thinking=enable_thinking,
                think_budget=think_budget,
                model_name=model_name,
                context_length=max_seq_length,
                for_dataset=True,
                reserve_think_budget=reserve_think_budget,
            )
            rows = _stamp(
                rows,
                subtask=st,
                context_length=max_seq_length,
                gen_budget=gen_budget,
                think_budget=think_budget,
                enable_thinking=enable_thinking,
                think_budget_reserved=resolve_reserve_think_budget(
                    max_seq_length, reserve_think_budget
                ),
            )
            splits.append(HFDataset.from_list(rows))

        return HFDatasetDict({"test": concatenate_datasets(splits)})


def _stamp(
    rows: list[dict],
    *,
    subtask: str,
    context_length: int,
    gen_budget: int,
    think_budget: int = 0,
    enable_thinking: bool = False,
    think_budget_reserved: bool = False,
) -> list[dict]:
    for row in rows:
        row["subtask"] = subtask
        row["context_length"] = context_length
        row["gen_budget"] = gen_budget
        if enable_thinking:
            row["think_budget"] = think_budget
            row["enable_thinking"] = True
            row["think_budget_reserved"] = think_budget_reserved
    return rows
