"""AGIEval dataset loader (v1.1) with per-subset selection.

AGIEval is 21 files under ``data/v1_1`` of the official repo — 19 MCQ subsets and
2 cloze subsets (``math``, ``gaokao-mathcloze``) — drawn from human admission and
qualification exams (Gaokao, SAT, LSAT, LSAT-adjacent law exams, AQuA-RAT, MATH).
There is no combined config and no official HF mirror carrying the cloze subsets,
so each subset is fetched as its own commit-pinned, checksummed ``.jsonl``.

Which subsets get loaded is the main knob, since the 21 span four languages ×
formats and nobody runs all of them by accident:

* ``subsets=["math", "sat-math"]`` — exact names (see :data:`SUBSETS`);
* ``group="math"`` — a named group (see :data:`SUBSET_GROUPS`);
* neither — all 21.

Rows keep upstream's field names and nullability (``passage`` / ``question`` /
``options`` / ``label`` / ``answer`` / ``other``) and gain ``subset``, which is
absent from the raw rows but decides prompt, parsing and scoring downstream.
Concatenating the 21 files forces three normalizations, all verdict-neutral and
all guarded — see :meth:`AGIEvalDataset.load`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
from typing import TypedDict, override

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Features, List, Value, concatenate_datasets, load_dataset

from sieval.community.agieval.dataset_loader import (
    MATH_OUTPUT_SUBSETS,
    MATH_SUBSETS,
    SUBSETS,
)
from sieval.community.agieval.evaluation import (
    LEADERBOARD_EN_MCQ_SUBSETS,
    LEADERBOARD_ZH_MCQ_SUBSETS,
)
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

# Pin the data to an immutable commit: `data/v1_1` is the current release (v1.0
# is still in the repo under `data/v1`), and a bare branch URL would not survive
# the checksums below.
AGIEVAL_COMMIT = "84ab72d94318290aad2e4ec820d535a95a1f7552"
_DATA_BASE_URL = (
    f"https://raw.githubusercontent.com/ruixiangcui/AGIEval/{AGIEVAL_COMMIT}/data/v1_1"
)

#: Named subset groups accepted by ``group=``. ``en-mcq`` / ``zh-mcq`` are
#: upstream's own leaderboard groupings (AGIEval-en / AGIEval-zh, MCQ only);
#: ``math`` is sieval's, and is the reason this loader exists in this shape.
SUBSET_GROUPS: dict[str, tuple[str, ...]] = {
    "all": SUBSETS,
    "math": MATH_SUBSETS,
    "en-mcq": LEADERBOARD_EN_MCQ_SUBSETS,
    "zh-mcq": LEADERBOARD_ZH_MCQ_SUBSETS,
}

# One uniform schema for all 21 files. Needed because the per-file inferred
# schemas genuinely disagree: `options` / `label` / `answer` / `passage` are
# all-null in some subsets (inferred as `null`), `label` is a list in the two
# jec-qa files, `answer` is absent from the three sat files, and `other` is a
# struct whose members differ per subset.
_FEATURES = Features(
    {
        "subset": Value("string"),
        "passage": Value("string"),
        "question": Value("string"),
        "options": List(Value("string")),
        "label": Value("string"),
        "answer": Value("string"),
        # Upstream's provenance bag: `solution` (math / aqua-rat / sat-*),
        # `source` (gaokao-*), `level` + `type` (math). Never read by AGIEval's
        # own prompt/parse/score path; kept because the MATH solutions and
        # difficulty levels have no other home.
        "other": {
            "solution": Value("string"),
            "source": Value("string"),
            "level": Value("string"),
            "type": Value("string"),
        },
    }
)

_OTHER_KEYS = ("solution", "source", "level", "type")


class AGIEvalDatasetSample(TypedDict):
    subset: str
    passage: str | None
    question: str
    options: list[str]
    label: str | None
    answer: str | None
    other: dict


@sieval_dataset(
    name="agieval",
    display_name="AGIEval",
    description="AGIEval v1.1 — 21 human exam subsets (Gaokao, SAT, LSAT, MATH).",
    source=tuple(f"url:{_DATA_BASE_URL}/{subset}.jsonl" for subset in SUBSETS),
    checksums={
        "aqua-rat.jsonl": "sha256:cff42e946e6082dacb27285dae19cb4be98408ab760fe43c6c462e543be50572",  # noqa: E501
        "gaokao-biology.jsonl": "sha256:789baadad69c998743302143a3e1c2022a2aca785bbe75644877a336242c7e69",  # noqa: E501
        "gaokao-chemistry.jsonl": "sha256:4fb8a4b881f652a908545447a062c7ba458be2ba793a45fc11b892053406704e",  # noqa: E501
        "gaokao-chinese.jsonl": "sha256:1ddcf8fa15e07a25589796dc1c72a341c2d874af8de41970262d66693f95285f",  # noqa: E501
        "gaokao-english.jsonl": "sha256:2de1b1e5d9718d908ffb46665e949bff83bef956ad8b7de16ea412e058162b01",  # noqa: E501
        "gaokao-geography.jsonl": "sha256:1170cb39171d6dfc35bd52a52505e5a120cf400abc697f282d0739db6181713b",  # noqa: E501
        "gaokao-history.jsonl": "sha256:27350771d399814fc69dd6d7e6ce115b7913ae208e04e385a7e6fcdf51a6b8b7",  # noqa: E501
        "gaokao-mathcloze.jsonl": "sha256:088675c147794970a3ed25c7147a3bbc59715d6813837d5f483f46dcb1b5008d",  # noqa: E501
        "gaokao-mathqa.jsonl": "sha256:d246f12752d121289ef55cbf1bcf954243cefb65d110f71d09358e24065c808f",  # noqa: E501
        "gaokao-physics.jsonl": "sha256:9f8f91b35b5cc2d3ba67b9c6f31bc72c51caaaa58a4a46375690d3b8b127a82b",  # noqa: E501
        "jec-qa-ca.jsonl": "sha256:704efb9943cb827811d883163d893d483451ddaaf9f595f94866e51e70e20785",  # noqa: E501
        "jec-qa-kd.jsonl": "sha256:fec1d0d85d480ee6c23371af59b2305732d8561296cd878b5367b65ebfc4a467",  # noqa: E501
        "logiqa-en.jsonl": "sha256:63d0e8efaca1944e7eac5037903650f92ca9c036155a66b4e95bf2aa06da1702",  # noqa: E501
        "logiqa-zh.jsonl": "sha256:0e5f6548932ce6cd388d8432d015db9baa468731c94409c98d20902832bb099c",  # noqa: E501
        "lsat-ar.jsonl": "sha256:3b3e3fe09a07c695326adb82f38da0d67d5bbaadab41551f198c3102f1ea9dc8",  # noqa: E501
        "lsat-lr.jsonl": "sha256:c6acb4d843db7b515da4d853e52bb8e6e5f776910f2e35a1805634db24bf94ea",  # noqa: E501
        "lsat-rc.jsonl": "sha256:0eed491b3099d66b8110d4fabb98ce43285a6ac61ea5643285eec11b8abce202",  # noqa: E501
        "math.jsonl": "sha256:43e783af2025318125a96a970a0df37941124a5c0dabea382a12ce1b04651a11",  # noqa: E501
        "sat-en-without-passage.jsonl": "sha256:77eb57bb6f6f39466d5d169de5253e77755ef521ad6692f582307672affbf593",  # noqa: E501
        "sat-en.jsonl": "sha256:33fe87b32ff16ae7ba52b27bb398190b082253b3617a3eb9737005262ba12dd4",  # noqa: E501
        "sat-math.jsonl": "sha256:9cde4c0522b6196852a5562db6e72bf8817b4802822db0fc10013d1becdab3c3",  # noqa: E501
    },
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "chinese", "multiple-choice", "open-ended"),
    # The closest single SPDX id for the aggregate, not the repo's headline MIT,
    # and NOT a floor -- one of the terms is narrower than any SPDX id expresses.
    # Upstream's data/v1_1/LICENSE reproduces each source exam's own terms: Gaokao
    # / SAT / LSAT / MATH MIT, aqua-rat Apache-2.0, logiqa-en + logiqa-zh (1,302 of
    # the 7,272 rows; group=all, en-mcq, zh-mcq) CC BY-NC-SA 4.0, and jec-qa-kd +
    # jec-qa-ca (1,012 rows; group=all, zh-mcq) academic research only, commercial
    # use "strictly prohibited", plus a required citation of arXiv:1911.12011.
    # That last section is easy to miss -- upstream files it under a duplicated
    # `# MATH` header, and the tell is `Link: https://jecqa.thunlp.org/`. Because
    # academic-only grants strictly less than CC BY-NC-SA 4.0's NonCommercial
    # share-and-adapt, no single value bounds the aggregate: a selection narrows
    # which terms apply, so redistributing one means reading that LICENSE, not
    # this field.
    license="CC-BY-NC-SA-4.0",
)
class AGIEvalDataset(Dataset[AGIEvalDatasetSample]):
    """AGIEval v1.1, one ``test`` split concatenated from the selected subsets."""

    @override
    def load(
        self,
        name_or_path: str,
        subsets: list[str] | None = None,
        group: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        """Load the selected subsets from ``<name_or_path>/<subset>.jsonl``.

        Exactly one selection may be given: *subsets* (exact names) or *group*
        (a :data:`SUBSET_GROUPS` key). Neither loads all 21. Selection order is
        always :data:`SUBSETS` order, so the concatenation — and every sample id
        derived from it — does not depend on how the argument was spelled.

        Three normalizations, each required to concatenate the files at all:

        * ``label`` is a 1-element **list** in ``jec-qa-kd`` / ``jec-qa-ca`` and a
          string everywhere else; the list is unwrapped. A longer one (v1.0 had
          genuine multi-label rows) raises rather than silently changing what gets
          compared.
        * ``other.level`` is an ``int64`` in ``math.jsonl`` and absent elsewhere;
          stringified so the struct has one dtype across subsets. No other column
          is cast — upstream already ships them as strings.
        * ``options`` is ``null`` on the two cloze subsets (1,118 rows with no
          choices to show) and a list of strings on the other 19; coerced to
          ``[]`` for one dtype. Verdict-neutral only because no cloze prompt
          renders options, which is why it is guarded by subset: a *silent* ``[]``
          on an MCQ subset would render a question with no choices and score
          whatever the parser made of the reply.
        """
        selected = self._select_subsets(subsets, group)

        parts: list[HFDataset] = []
        for subset in selected:
            path = os.path.join(name_or_path, f"{subset}.jsonl")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"AGIEval subset file not found: {path}. Stage the data with "
                    "`sieval dataset download agieval`, and point the dataset's "
                    "`path` at the directory holding the <subset>.jsonl files."
                )
            raw = ensure_dataset(
                load_dataset("json", data_files=path, split="train", **kwargs)
            )
            parts.append(
                raw.map(
                    lambda row, s=subset: self._normalize(row, s),
                    features=_FEATURES,
                    remove_columns=raw.column_names,
                )
            )

        combined = concatenate_datasets(parts)
        if len(combined) == 0:
            raise ValueError(
                f"AGIEval produced an empty test split for subsets={selected!r}; "
                "check the staged .jsonl files."
            )
        return HFDatasetDict({"test": combined})

    @staticmethod
    def _select_subsets(subsets: list[str] | None, group: str | None) -> list[str]:
        if subsets is not None and group is not None:
            raise ValueError(
                "AGIEval: pass either `subsets` (exact names) or `group` "
                f"(one of {sorted(SUBSET_GROUPS)}), not both."
            )
        if group is not None:
            if group not in SUBSET_GROUPS:
                raise ValueError(
                    f"AGIEval: unknown group {group!r}; "
                    f"expected one of {sorted(SUBSET_GROUPS)}."
                )
            chosen = set(SUBSET_GROUPS[group])
        elif subsets is not None:
            unknown = [s for s in subsets if s not in SUBSETS]
            if unknown:
                raise ValueError(
                    f"AGIEval: unknown subset(s) {unknown!r}; "
                    f"expected names from {list(SUBSETS)}."
                )
            if not subsets:
                raise ValueError("AGIEval: `subsets` is empty; omit it to load all.")
            chosen = set(subsets)
        else:
            chosen = set(SUBSETS)
        return [subset for subset in SUBSETS if subset in chosen]

    @staticmethod
    def _normalize(row: dict, subset: str) -> AGIEvalDatasetSample:
        label = row.get("label")
        if isinstance(label, list):
            if len(label) != 1:
                raise ValueError(
                    f"AGIEval subset {subset!r}: expected a single-answer label, "
                    f"got {label!r}. The pinned v1.1 data has none; a multi-label "
                    "row means the source changed and scoring must be revisited."
                )
            label = label[0]
        other = row.get("other") or {}
        options = row.get("options")
        if not options and subset not in MATH_OUTPUT_SUBSETS:
            raise ValueError(
                f"AGIEval subset {subset!r}: row has no `options`, but only the "
                f"cloze subsets {list(MATH_OUTPUT_SUBSETS)} may omit them. On an "
                "MCQ subset the question would be prompted with no answer choices "
                "and scored anyway, so the shape change must be reviewed rather "
                "than normalized away. The pinned v1.1 data has no such row."
            )
        return {
            "subset": subset,
            "passage": row.get("passage"),
            "question": row["question"],
            # null on the two cloze subsets, which have no options to show; the
            # guard above keeps this from silently emptying an MCQ row.
            "options": options or [],
            "label": label,
            "answer": row.get("answer"),
            "other": {
                key: None if other.get(key) is None else str(other[key])
                for key in _OTHER_KEYS
            },
        }
