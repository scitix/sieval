"""Unified RULER dataset: 13 subtasks in one loader.

Each call to ``load()`` targets one subtask (or ``"all"`` to concatenate all
13). Every emitted row carries ``subtask`` and ``context_length`` fields so
``RulerZeroShotGenTask.report()`` can group and score without any external
aggregation command.

The 13 canonical subtask names mirror ``synthetic.yaml``:
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
from ._shared import _HOTPOTQA_REVISION, _RULER_DATA_SHA
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


class RulerDatasetSample(TypedDict):
    index: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str
    subtask: str
    context_length: int
    token_position_answer: NotRequired[int]  # NIAH only


@sieval_dataset(
    name="ruler",
    display_name="RULER",
    description=(
        "RULER long-context benchmark: 13 subtasks (NIAH ×8, VT, CWE, FWE, QA ×2)."
    ),
    source=(
        "local:paul_graham_essays/PaulGrahamEssays.json.gz",
        f"url:https://media.githubusercontent.com/media/NVIDIA/RULER/{_RULER_DATA_SHA}/scripts/data/synthetic/json/english_words.json",
        "url:https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        f"hf:hotpotqa/hotpot_qa@{_HOTPOTQA_REVISION}",
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
        # NIAH-specific (ignored for non-NIAH subtasks)
        num_needle_k: int = 1,
        num_needle_v: int = 1,
        num_needle_q: int = 1,
        type_haystack: str = "essay",
        type_needle_k: str = "words",
        type_needle_v: str = "numbers",
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

        # Aggregate entry points — an explicit list or "all" — load each single
        # subtask and concatenate. Recurse with the *base* ``name_or_path``; the
        # per-subtask ``/ruler`` staging path is resolved once, below, so every
        # entry point (including a direct single-subtask load) maps it the same way.
        if isinstance(subtask, list) or subtask == "all":
            targets = subtask if isinstance(subtask, list) else _ALL_SUBTASKS
            splits = [
                self.load(
                    name_or_path,
                    subtask=st,
                    max_seq_length=max_seq_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_path=tokenizer_path,
                    num_samples=num_samples,
                    random_seed=random_seed,
                    remove_newline_tab=remove_newline_tab,
                    enable_thinking=enable_thinking,
                    think_budget=think_budget,
                    model_name=model_name,
                    num_needle_k=num_needle_k,
                    num_needle_v=num_needle_v,
                    num_needle_q=num_needle_q,
                    type_haystack=type_haystack,
                    type_needle_k=type_needle_k,
                    type_needle_v=type_needle_v,
                    freq_cw=freq_cw,
                    freq_ucw=freq_ucw,
                    num_cw=num_cw,
                    num_fewshot=num_fewshot,
                    num_chains=num_chains,
                    num_hops=num_hops,
                    alpha=alpha,
                    coded_wordlen=coded_wordlen,
                    vocab_size=vocab_size,
                    pre_samples=pre_samples,
                )["test"]
                for st in targets
            ]
            return HFDatasetDict({"test": concatenate_datasets(list(splits))})

        data_path = _subtask_data_path(name_or_path, subtask)

        if subtask in _NIAH_SUBTASK_KWARGS:
            niah_kwargs = _NIAH_SUBTASK_KWARGS[subtask]
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
                num_needle_k=niah_kwargs["num_needle_k"],
                num_needle_v=niah_kwargs["num_needle_v"],
                num_needle_q=niah_kwargs["num_needle_q"],
                type_haystack=niah_kwargs["type_haystack"],
                type_needle_k=niah_kwargs["type_needle_k"],
                type_needle_v=niah_kwargs["type_needle_v"],
            )
        elif subtask == "vt":
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
                num_chains=num_chains,
                num_hops=num_hops,
                type_haystack="noise",
            )
        elif subtask == "cwe":
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
                freq_cw=freq_cw,
                freq_ucw=freq_ucw,
                num_cw=num_cw,
                num_fewshot=num_fewshot,
            )
        elif subtask == "fwe":
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
                alpha=alpha,
                coded_wordlen=coded_wordlen,
                vocab_size=vocab_size,
            )
        elif subtask in ("qa_squad", "qa_hotpotqa"):
            qa_dataset = "squad" if subtask == "qa_squad" else "hotpotqa"
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
                pre_samples=pre_samples,
            )
        else:
            raise ValueError(
                f"Unknown subtask {subtask!r}. "
                f"Valid subtasks: {_ALL_SUBTASKS} or 'all'."
            )

        rows = _stamp(rows, subtask=subtask, context_length=max_seq_length)
        return HFDatasetDict({"test": HFDataset.from_list(rows)})


def _stamp(rows: list[dict], *, subtask: str, context_length: int) -> list[dict]:
    for row in rows:
        row["subtask"] = subtask
        row["context_length"] = context_length
    return rows
