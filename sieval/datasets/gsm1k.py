"""
GSM1k loader — Scale AI's from-scratch re-do of GSM8K, built to detect overfitting.

GSM1k is 1205 grade-school word problems written by human annotators with no LLM
or synthetic assistance, mirroring GSM8K's difficulty and answer-magnitude
distribution. The benchmark exists to be read as a *pair* with GSM8K, not on its
own: a model's GSM8K − GSM1k gap estimates how much of its GSM8K score is
memorization rather than reasoning. The paper measures drops of up to 8% and a
Spearman r² of 0.36 between a model's likelihood of generating a GSM8k example
and its gap.

**Release history, because the pinned snapshot ships an empty dataset card while
the paper says the data is withheld.** The paper (Nov 2024) declined to publish
GSM1k "to prevent a similar problem of data contamination occurring in the
future" and precommitted to release on the earlier of two triggers — three
open-source models of different lineages reaching 95% accuracy, or June 2025 —
with its datasheet stating "The dataset (yet unreleased) will be released with
the MIT license." The pinned snapshot was uploaded to the ScaleAI org on
2025-03-31/04-01, i.e. that release. So `license="MIT"` here is the datasheet's
commitment for the *data*, not the eval repo's code license.

**Provenance verified**, since an empty card is not evidence: every one of the 50
questions in `gsm1k_public_50.csv` — the sample Scale published in its eval repo
while the full set was still withheld — appears verbatim in this snapshot's
`test` split with an identical answer (50/50 found, 0 answer mismatches). The row
count also matches the paper exactly: "GSM1k consists of 1205 problems".

**What that does not establish**, and it matters when reading a paired diff
against the paper's own column: whether this 2025 release is the *same* 1205
problems the 2024 paper scored. The 50 published rows and the count both match,
but item-level identity with the withheld set behind Table 1 cannot be checked
from here. A live run gives that question weight — under upstream's own exemplar
protocol the GSM8k halves land within +0.54 and +2.54 of Table 1 while the GSM1k
halves land +3.03 and −1.90, i.e. the validated-pipeline half agrees and the
GSM1k half disagrees in *opposite directions* per model. Contamination is not the
explanation: every checkpoint involved predates this release, so none could have
trained on it. See `gsm1k_kshot_base_gen`'s docstring for the numbers.

Schema, measured at the pinned revision: `question` and `answer` are both
strings and need no cast — no cast needed, upstream ships this dtype. All 1205
answers are bare integers (1-6 characters, every one matching `-?[0-9]+`), with
no thousands separators, no `####` delimiter and **no worked solution** — the
released data carries final answers only. Two consequences for tasks: the gold
needs no `answer.split("####")` step (unlike `openai/gsm8k`), and GSM1k cannot
supply chain-of-thought few-shot exemplars of its own, which is why
`gsm1k_kshot_base_gen` borrows GSM8K's the way upstream's harness does.

There is a `test` split and nothing else — no train set, so nothing to hold out.

References:

* Paper (v4 — the revision the task docstrings' targets are read from; v3
  published different Table 1 values for several rows):
  <https://arxiv.org/abs/2405.00332v4>
* Eval harness + public 50-example sample: <https://github.com/scaleapi/gsm1k_eval>

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

GSM1K_REVISION = "bc09569d09a614b9b530edc7f076fb214ac10493"


class GSM1KDatasetSample(TypedDict):
    question: str
    answer: str


@sieval_dataset(
    name="gsm1k",
    display_name="GSM1k",
    description="Grade School Math 1k - 1205 human-written GSM8K mirror problems.",
    source=f"hf:ScaleAI/gsm1k@{GSM1K_REVISION}",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("english", "math-word-problems", "open-ended"),
    license="MIT",
)
class GSM1KDataset(Dataset[GSM1KDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # One unnamed config, unlike openai/gsm8k's "main" / "socratic", so there
        # is no config argument to forward.
        dataset = load_dataset(name_or_path, **kwargs)
        return ensure_dataset_dict(dataset)
