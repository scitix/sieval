"""Shared loader logic for the MultiPL-E benchmark family.

MultiPL-E is HumanEval and MBPP machine-translated into other programming
languages, published as one HuggingFace dataset with one config per
(suite, language) pair -- ``humaneval-cpp``, ``mbpp-rs``, and so on. The two
suites share an identical schema and an identical loading contract, so the
processing lives here and the two dataset modules differ only in metadata, the
config prefix, and their language set.

**A language is named by upstream's registry tag, not by its English name**:
``jl`` not ``julia``, ``ml`` not ``ocaml``, ``rkt`` not ``racket``, ``adb`` not
``ada``. Those tags are the config names, so a wrong one is an unresolvable
config rather than a language that scores badly.

Python is deliberately absent from both suites: MultiPL-E translates *out of*
Python, so ``humaneval-py`` would be plain HumanEval -- which this repo already
has as ``human_eval_0shot_gen`` / ``_base_gen``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import TypedDict

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset

from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict

# nuprl/MultiPL-E main HEAD (dataset last modified 2025-07-15; static since).
MULTIPL_E_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"

# Upstream's registry tags, per suite. These differ: HumanEval was translated to
# Dart and MBPP was not, so the sets are listed separately rather than shared
# with a subtraction -- an asymmetry in upstream's data is not a rule to encode.
HUMANEVAL_LANGUAGES: tuple[str, ...] = (
    "adb",
    "clj",
    "cpp",
    "cs",
    "d",
    "dart",
    "elixir",
    "go",
    "hs",
    "java",
    "jl",
    "js",
    "lua",
    "ml",
    "php",
    "pl",
    "r",
    "rb",
    "rkt",
    "rs",
    "scala",
    "sh",
    "swift",
    "ts",
)
MBPP_LANGUAGES: tuple[str, ...] = tuple(
    lang for lang in HUMANEVAL_LANGUAGES if lang != "dart"
)


class MultiPLESampleFields(TypedDict):
    """Upstream's eight columns, kept under upstream's own names.

    Each suite declares its own ``TypedDict`` with these fields rather than
    sharing this one: the sample type is the reverse-lookup key ``@sieval_task``
    resolves a task's dataset through, so one shared type would make the two
    suites indistinguishable to the registry. This exists to state the schema
    once, and is not itself registered.

    ``prompt`` is a partial program that stops at the function's opening -- the
    model continues it. ``tests`` closes that function and adds the suite's
    assertions, so the graded program is ``prompt + completion + tests``.
    ``stop_tokens`` truncates a base model's continuation.

    ``original`` is an absolute path on the authors' own cluster and is useless
    outside it; it is carried because dropping a column upstream ships is a
    divergence with no benefit, not because anything reads it.
    """

    name: str
    language: str
    prompt: str
    doctests: str
    original: str
    prompt_terminology: str
    tests: str
    stop_tokens: list[str]


def normalize_languages(
    languages: list[str] | None,
    config: str | None,
    *,
    suite: str,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the requested language tags, in upstream's declared order.

    Accepts a bare tag (``cpp``) or a full config name (``humaneval-cpp``) so a
    caller who has read the HuggingFace config list can paste from it. A config
    naming the *other* suite is rejected rather than silently re-pointed: the
    suite is fixed by which dataset class this is, and honouring the config
    would load rows the registered task was never bound to.

    Order comes from *available* rather than from the caller, so two runs
    requesting the same set in a different order produce the same row order --
    which is what keeps a resume's sample ids stable.

    An EMPTY ``languages`` list is refused rather than read as "all". The two
    readings are 24 languages of inference apart, and an empty list reaches here
    from a config that computed a selection and came up with nothing far more
    often than from an author who meant the whole suite -- which omitting the
    argument already says, unambiguously.
    """
    if languages is not None and not languages:
        raise ValueError(
            f"MultiPL-E {suite}: `languages` is an empty list, which names no "
            f"language to run. Omit it (or pass `config='all'`) to run all "
            f"{len(available)} languages; list registry tags to run a subset."
        )

    requested: list[str] = []
    if config is not None and config != "all":
        requested.append(config)
    if languages is not None:
        requested.extend(languages)

    if not requested:
        return available

    prefix = f"{suite}-"
    tags: set[str] = set()
    unknown: list[str] = []
    for raw in requested:
        tag = str(raw).strip()
        if tag.startswith(prefix):
            tag = tag[len(prefix) :]
        elif "-" in tag:
            # A config for the other suite, e.g. `mbpp-cpp` asked of HumanEval.
            unknown.append(raw)
            continue
        if tag in available:
            tags.add(tag)
        else:
            unknown.append(raw)

    if unknown:
        raise ValueError(
            f"Unknown MultiPL-E {suite} language(s): {', '.join(sorted(unknown))}. "
            f"Expected upstream registry tags (not English names): "
            f"{', '.join(available)}."
        )
    return tuple(lang for lang in available if lang in tags)


def load_multipl_e(
    name_or_path: str,
    *,
    suite: str,
    available: tuple[str, ...],
    languages: list[str] | None = None,
    config: str | None = None,
    eval_split: str | None = None,
    **kwargs,
) -> HFDatasetDict:
    """Load one or more of *suite*'s language configs as a single DatasetDict.

    All selected languages land in one dataset, the way ``MMMLUDataset`` returns
    every locale together rather than one dataset per locale. The rows already
    carry a ``language`` column, so a task recovers the per-language breakdown
    from the data instead of from how it was loaded.
    """
    selected = normalize_languages(languages, config, suite=suite, available=available)

    by_split: dict[str, list[HFDataset]] = {}
    for lang in selected:
        dataset = load_dataset(name_or_path, f"{suite}-{lang}", **kwargs)
        dataset = apply_eval_split(ensure_dataset_dict(dataset), eval_split)
        for split, split_dataset in dataset.items():
            if len(split_dataset) > 0:
                by_split.setdefault(str(split), []).append(split_dataset)

    processed = HFDatasetDict()
    for split, split_datasets in by_split.items():
        processed[split] = (
            split_datasets[0]
            if len(split_datasets) == 1
            else concatenate_datasets(split_datasets)
        )

    if not processed or all(len(split) == 0 for split in processed.values()):
        raise ValueError(
            f"MultiPL-E {suite} loader produced an empty dataset for languages "
            f"{', '.join(selected)}; check the language tags, eval_split, or schema."
        )
    return processed
