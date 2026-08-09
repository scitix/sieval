"""
Shared fixtures for the PlatinumBench task tests.

All five leaf tasks are 2-line subclasses of one base, so the model stub and the
row/task factories live here instead of being copied into six files.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import EvalMode
from sieval.core.tasks.meta import get_task_class, get_task_meta
from sieval.datasets.platinum_bench import (
    PlatinumBenchDataset,
    PlatinumBenchDatasetSample,
)
from sieval.tasks.platinum_bench._base import (
    PLATINUM_UPSTREAM_URL,
    PlatinumMathGenTask,
)

# Both columns are shaped like the real ones, because the wording is load-bearing:
# the no-CoT column really does open its last sentence with "Then, provide" — a
# leftover from the CoT variant — and that exact substring is what upstream's
# o1 rewrite targets. A fixture without it would make the `no_cot_o1` tests pass
# vacuously.
_PROMPT_HEAD = "Solve the following math word problem.\n\nWhat is 40 + 2?\n\n"
_PROMPT_TAIL = (
    'the final answer as a single integer in the format "Answer: XXX" with no '
    "extra formatting."
)
COT_PROMPT = f"{_PROMPT_HEAD}Think step-by-step. Then, provide {_PROMPT_TAIL}"
NO_COT_PROMPT = f"{_PROMPT_HEAD}Then, provide {_PROMPT_TAIL}"
O1_PROMPT = f"{_PROMPT_HEAD}Provide {_PROMPT_TAIL}"


class CapturingChatModel(ChatModel):
    """Returns a fixed completion and records the request the model would send.

    ``last_kwargs`` is the **merged** result, not what ``infer()`` passed. The
    override point (``_agenerate_impl``) sits below ``ChatModel``'s own
    ``{**self._kwargs, **kwargs}``, so recording the call-time kwargs alone
    would make every "the task injects no decoding params" assertion vacuous —
    a task-side value that silently outranks the configured one would look
    identical to no value at all. Reproducing the merge here is what lets a test
    assert which side wins.
    """

    def __init__(
        self,
        text: str = "Answer: 42",
        texts: list[str] | None = None,
        **model_kwargs,
    ):
        super().__init__(model="mock-chat", api_key="fake", **model_kwargs)
        self.last_kwargs: dict[str, object] = {}
        self._texts = list(texts) if texts is not None else [text]

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = {**self._kwargs, **kwargs}
        # Honouring `n` is what lets a test tell "the task asked for n" apart
        # from "the stub happened to hand back a list".
        requested = self.last_kwargs.get("n", 1)
        n = requested if isinstance(requested, int) and requested > 0 else 1
        texts = (self._texts * n)[:n] if len(self._texts) < n else self._texts[:n]
        return ModelOutput(model=self.meta(), texts=texts)

    async def _alogprobs_impl(
        self,
        prompt,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def make_sample(
    *,
    subset: str = "gsm8k",
    target: str = "42",
    strategy: str = "math",
) -> PlatinumBenchDatasetSample:
    return {
        "subset": subset,
        "cleaning_status": "consensus",
        "platinum_prompt": COT_PROMPT,
        "platinum_prompt_no_cot": NO_COT_PROMPT,
        "platinum_target": [target],
        "original_target": [target],
        "platinum_parsing_strategy": strategy,
    }


def make_dataset(*subsets: str) -> PlatinumBenchDataset:
    """A dataset holding one row per named subset, built without touching the hub.

    The loader merges every config and the caller narrows it, so *which* subsets
    are present is the whole wiring contract: pass one for a correctly narrowed
    dataset, several to stand in for an un-narrowed merged split, none for an
    empty one.
    """
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict as HFDatasetDict

    rows = [dict(make_sample(subset=s)) for s in subsets]
    return PlatinumBenchDataset(
        _hf_dict=HFDatasetDict(
            {"test": HFDataset.from_list(rows) if rows else HFDataset.from_dict({})}
        )
    )


def assert_leaf_meta(
    task_cls: type[PlatinumMathGenTask],
    *,
    name: str,
    subset: str,
    kept: int,
    total: int,
) -> None:
    """Assert the registration contract every leaf shares.

    Lives here rather than in each leaf test so a copy-paste slip between two
    leaves (swapped subset, stale row count) fails in exactly one place.
    """
    meta = get_task_meta(task_cls)
    assert meta.name == name
    assert task_cls.subset == subset
    # The FK is resolved from the *base* class's sample generic via the MRO —
    # the leaves declare no generic of their own.
    assert meta.dataset == "platinum_bench"
    assert meta.eval_mode is EvalMode.GEN
    assert meta.n_shot == 0
    assert meta.model_type == "chat"
    # All 120 math cells of the paper's Table 3 reproduce exactly (see `_base`).
    assert meta.status == "stable"
    assert meta.tags == ("english", "math-word-problems", "open-ended")

    assert meta.reference_impl is not None
    assert meta.reference_impl.source == "MadryLab/platinum-benchmarks"
    assert meta.reference_impl.url == PLATINUM_UPSTREAM_URL
    notes = meta.reference_impl.notes
    assert f"Subset '{subset}'" in notes
    # kept + rejected == total, spelled out so a stale count cannot agree with
    # itself.
    assert f"{kept} rows kept of {total}" in notes
    assert f"({total - kept} rejected)" in notes
    # The row count is the one number a leaderboard reader compares, so it must
    # also be in the one-line description.
    assert str(kept) in meta.description

    # Subpackage-hosted tasks must still resolve by name — this is what
    # `sieval task show` / `sieval eval` go through.
    assert get_task_class(name) is task_cls

    # A leaf adds nothing but its subset; behaviour lives in the shared base, so
    # the five subsets cannot silently drift apart. `tags` / `model_type` /
    # `n_shot` are seeded onto the class by `@sieval_task`, not written by the
    # leaf. Private names are excluded: `_sieval_task_meta` comes from the
    # decorator and `_abc_impl` from ABCMeta. Every stage method is public, so an
    # override still trips this.
    own = {key for key in vars(task_cls) if not key.startswith("_")}
    assert own == {"subset", "tags", "model_type", "n_shot"}


def make_task[T: PlatinumMathGenTask](
    task_cls: type[T],
    *,
    text: str = "Answer: 42",
    texts: list[str] | None = None,
    subset: str | None = None,
    model_kwargs: dict[str, object] | None = None,
    **task_kwargs,
) -> tuple[T, CapturingChatModel]:
    """Instantiate *task_cls* against a dataset for *subset* (default: its own).

    ``model_kwargs`` seeds the model's own request params — the `models:` /
    ``infer_args`` side — so a test can assert what survives the merge.
    """
    model = CapturingChatModel(text=text, texts=texts, **(model_kwargs or {}))
    dataset = make_dataset(subset if subset is not None else task_cls.subset)
    return task_cls(dataset, model, **task_kwargs), model
