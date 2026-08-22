"""Ag-LiveCodeBench-X — the multi-language LiveCodeBench subset from Agnostics.

Upstream (NUPRL, `Agnostics <https://arxiv.org/abs/2508.04865>`_) took
LiveCodeBench v5's 880 problems, kept the **499 that specify and check their
answer purely through stdin/stdout**, and dropped the 381 that ship Python
starter code. What is left is language-agnostic: the same problem statement and
the same test cases score a solution written in any language, which is why the
set can be pointed at Lua, R, Julia, OCaml or Fortran without re-authoring
anything.

Three columns and no more — there is no ``starter_code``, no ``metadata`` and no
``public_test_cases`` split here, so every case is a private one and the
call-based (``fn_name``) grading path LiveCodeBench also carries cannot arise.
``private_test_cases`` keeps LiveCodeBench's own encoding (base64 -> zlib ->
pickle -> JSON); decoded, the 499 problems are ~8 GB of test data, which is why
it travels compressed.

Upstream publishes **no license**: the Hub card carries no license field and the
scripts repo has no ``LICENSE`` file, so ``license`` is ``None`` rather than a
guess. See the task's ``reference_impl.notes`` for what that costs in port
fidelity.

AI-Generated Code - Claude Opus 5 (Anthropic)
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

AG_LIVECODEBENCH_X_REVISION = "15bef6bf966893163ea9d22f25b7c58fec06a949"


class AgLiveCodeBenchXDatasetSample(TypedDict):
    question_id: str
    question_content: str
    private_test_cases: str


@sieval_dataset(
    name="ag_livecodebench_x",
    display_name="Ag-LiveCodeBench-X",
    description=(
        "LiveCodeBench v5's 499 stdin/stdout problems, prepared by Agnostics "
        "for any language."
    ),
    source=f"hf:nuprl/Ag-LiveCodeBench-X@{AG_LIVECODEBENCH_X_REVISION}",
    categories=(
        Category(Level1Category.CODE, "CodeGeneration"),
        Category(Level1Category.CODE, "MultiLanguageSupport"),
    ),
    # No `python` tag, unlike `livecodebench_code_generation`: the target
    # language is the task's `language` argument, not a property of the data.
    tags=("english", "code-exec"),
    # Upstream declares none (no Hub license field, no LICENSE in the scripts
    # repo). `None` is the accurate record; do not infer one from LiveCodeBench,
    # whose own label ('cc') covers a different artifact.
    license=None,
)
class AgLiveCodeBenchXDataset(Dataset[AgLiveCodeBenchXDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # One call for both a Hub id and a staged directory: the release is plain
        # parquet under `data/`, with no loader script and no config to select,
        # so `load_dataset` resolves either form on its own. No `trust_remote_code`
        # for the same reason -- unlike `livecodebench_code_generation`, whose Hub
        # copy ships a builder script.
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))

        # A window that filters everything away would otherwise be reported as a
        # score of 0.0 rather than as a mistake.
        if any(len(split) == 0 for split in dataset.values()):
            raise ValueError(
                "Ag-LiveCodeBench-X dataset is empty after loading "
                f"(name_or_path={name_or_path!r}); the release has a single "
                "'test' split of 499 problems."
            )

        return dataset
