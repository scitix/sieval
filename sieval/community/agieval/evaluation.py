# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# adapted from https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/src/evaluation.py
# leaderboard groupings from https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/post_process_and_evaluation.py
"""AGIEval per-sample verdict + the groupings its leaderboard averages over.

Three comparison rules (upstream ``evaluate_single_sample``): set-of-letters for
the multi-answer MCQ subsets, math equivalence for the two cloze subsets, exact
string equality for everything else.

The leaderboard groups here are **not** the prompt-language families in
:mod:`.dataset_loader`, and the difference is not cosmetic: ``gaokao-english`` is
prompted in English but averaged as Chinese (it is a Chinese Gaokao paper), and
``gaokao-mathqa`` is Chinese in both. Upstream's driver keeps two separate lists
for exactly this reason.

Deltas from upstream: ``convert_to_set(None)`` returns an empty *set* (upstream
returns an empty *dict* — a typo, ``{}``); every comparison it feeds reaches the
same verdict either way. A ``None`` prediction (sieval's "could not extract")
needs no special case: it compares unequal under all three rules.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from .dataset_loader import MATH_OUTPUT_SUBSETS, MULTI_CHOICE_SUBSETS
from .math_equivalence import is_equiv

#: The 8 English MCQ subsets upstream averages for its AGIEval-en leaderboard.
#: `gaokao-english` is excluded here and counted with the Chinese group below.
LEADERBOARD_EN_MCQ_SUBSETS: tuple[str, ...] = (
    "aqua-rat",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-math",
    "sat-en",
    "sat-en-without-passage",
)

#: The 11 Chinese MCQ subsets upstream averages for its AGIEval-zh leaderboard.
LEADERBOARD_ZH_MCQ_SUBSETS: tuple[str, ...] = (
    "logiqa-zh",
    "jec-qa-kd",
    "jec-qa-ca",
    "gaokao-chinese",
    "gaokao-english",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-physics",
    "gaokao-mathqa",
)


def convert_to_set(item: str | list | None) -> set[str]:
    if isinstance(item, list):
        return set(item)
    if isinstance(item, str):
        return {item}
    if item is None:
        return set()
    raise ValueError("Input can't parse:", item)


def evaluate_single_sample(
    subset: str, prediction: str | list | None, label: str | None
) -> bool:
    if subset in MULTI_CHOICE_SUBSETS:
        return convert_to_set(prediction) == convert_to_set(label)
    if subset in MATH_OUTPUT_SUBSETS:
        return is_equiv(prediction, label)
    return prediction == label
