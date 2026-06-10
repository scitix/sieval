"""RULER NIAH (needle-in-a-haystack) synthetic dataset.

One parameterized loader covering all eight RULER NIAH variants (single_1/2/3,
multikey_1/2/3, multivalue, multiquery) via ``load()`` args. Synthesis is ported
from OpenCompass ``opencompass/datasets/ruler/ruler_niah.py``: build a haystack,
insert key/value needles at sampled depths, grow the haystack until it fills
``max_seq_length`` (measured with a tiktoken/HF tokenizer), and emit
``{prompt, answer}`` rows. The bound task does inference + substring scoring.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import gzip
import json
import os
import random
import re
import uuid
from typing import TypedDict, override

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.datasets.ruler._common import build_tokenizer

_CORPUS_FILE = "PaulGrahamEssays.json.gz"

_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
_REPEAT_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)
_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} "
    "afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for "
    "{query} mentioned in the provided text? The special magic {type_needle_v} "
    "for {query} mentioned in the provided text are"
)


class RulerNiahDatasetSample(TypedDict):
    prompt: str
    answer: list[str]


@sieval_dataset(
    name="ruler_niah",
    display_name="RULER NIAH",
    description="RULER needle-in-a-haystack: retrieve magic values from long context.",
    source=("local:paul_graham_essays/PaulGrahamEssays.json.gz",),
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "open-ended", "long-context"),
    license="Apache-2.0",
    deps_group="ruler",
)
class RulerNiahDataset(Dataset[RulerNiahDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        max_seq_length: int = 4096,
        tokens_to_generate: int = 128,
        tokenizer_model: str = "gpt-4",
        num_samples: int = 500,
        random_seed: int = 42,
        num_needle_k: int = 1,
        num_needle_v: int = 1,
        num_needle_q: int = 1,
        type_haystack: str = "essay",
        type_needle_k: str = "words",
        type_needle_v: str = "numbers",
        remove_newline_tab: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = build_tokenizer(tokenizer_model)
        random.seed(random_seed)
        np.random.seed(random_seed)
        num_needle_k = max(num_needle_k, num_needle_q)

        haystack = self._build_haystack(name_or_path, type_haystack)
        words = _word_pool()
        depths = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

        def gen(num_haystack: int) -> tuple[str, list[str]]:
            return _generate_input_output(
                num_haystack=num_haystack,
                haystack=haystack,
                words=words,
                depths=depths,
                random_seed=random_seed,
                num_needle_k=num_needle_k,
                num_needle_v=num_needle_v,
                num_needle_q=num_needle_q,
                type_haystack=type_haystack,
                type_needle_k=type_needle_k,
                type_needle_v=type_needle_v,
            )

        num_haystack = self._fit_haystack_size(
            gen=gen,
            tokenizer=tokenizer,
            haystack=haystack,
            type_haystack=type_haystack,
            max_seq_length=max_seq_length,
            tokens_to_generate=tokens_to_generate,
        )

        rows = []
        incremental = _incremental(type_haystack, max_seq_length)
        for _ in range(num_samples):
            used = num_haystack
            while True:
                try:
                    prompt, answer = gen(used)
                    length = len(tokenizer.encode(prompt)) + tokens_to_generate
                    assert length <= max_seq_length, "exceeds max_seq_length"
                    break
                except Exception:
                    if used > incremental:
                        used -= incremental
                    else:
                        prompt, answer = gen(used)
                        break
            if remove_newline_tab:
                prompt = " ".join(
                    prompt.replace("\n", " ").replace("\t", " ").strip().split()
                )
            rows.append({"prompt": prompt, "answer": answer})

        return HFDatasetDict({"test": HFDataset.from_list(rows)})

    def _build_haystack(self, name_or_path: str, type_haystack: str):
        if type_haystack == "essay":
            path = os.path.join(name_or_path, _CORPUS_FILE)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                text = json.load(f)["text"]
            return re.sub(r"\s+", " ", text).split(" ")
        if type_haystack == "repeat":
            return _REPEAT_HAYSTACK
        if type_haystack == "needle":
            return _NEEDLE
        raise NotImplementedError(f"{type_haystack} is not implemented.")

    def _fit_haystack_size(
        self,
        *,
        gen,
        tokenizer,
        haystack,
        type_haystack: str,
        max_seq_length: int,
        tokens_to_generate: int,
    ) -> int:
        """RULER's tokens-per-haystack estimate + binary search for the largest fit.

        The essay haystack now repeats on overflow (see ``_generate_input_output``),
        so the search is no longer capped at the corpus size.
        """
        incremental = _incremental(type_haystack, max_seq_length)
        sample_prompt, _ = gen(incremental)
        tokens_per_haystack = len(tokenizer.encode(sample_prompt)) / incremental
        estimated_max = int((max_seq_length / tokens_per_haystack) * 3)

        lower, upper = incremental, max(estimated_max, incremental * 2)
        optimal: int | None = None
        while lower <= upper:
            mid = (lower + upper) // 2
            prompt, _ = gen(mid)
            total = len(tokenizer.encode(prompt)) + tokens_to_generate
            if total <= max_seq_length:
                optimal = mid
                lower = mid + 1
            else:
                upper = mid - 1
        return optimal if optimal is not None else incremental


def _word_pool() -> list[str]:
    import wonderwords

    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    return sorted(set(words))


def _incremental(type_haystack: str, max_seq_length: int) -> int:
    if type_haystack == "essay":
        return 500
    if max_seq_length < 4096:
        return 5
    return 25


def _random_value(type_needle: str, words: list[str]) -> str:
    if type_needle == "numbers":
        return str(random.randint(10**6, 10**7 - 1))
    if type_needle == "words":
        return random.choice(words)
    if type_needle == "uuids":
        return str(uuid.UUID(int=random.getrandbits(128), version=4))
    raise NotImplementedError(f"{type_needle} is not implemented.")


def _generate_input_output(
    *,
    num_haystack: int,
    haystack,
    words: list[str],
    depths: list[int],
    random_seed: int,
    num_needle_k: int,
    num_needle_v: int,
    num_needle_q: int,
    type_haystack: str,
    type_needle_k: str,
    type_needle_v: str,
) -> tuple[str, list[str]]:
    keys: list[str] = []
    values: list[list[str]] = []
    needles: list[str] = []
    for _ in range(num_needle_k):
        keys.append(_random_value(type_needle_k, words))
        value: list[str] = []
        for _ in range(num_needle_v):
            value.append(_random_value(type_needle_v, words))
            needles.append(
                _NEEDLE.format(
                    type_needle_v=type_needle_v, key=keys[-1], value=value[-1]
                )
            )
        values.append(value)

    random.Random(random_seed).shuffle(needles)

    if type_haystack == "essay":
        # Repeat the essay when more words are needed than the corpus holds
        # (RULER behaviour); otherwise slice. Keeps very large contexts fillable.
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            repeats = (num_haystack + len(haystack) - 1) // len(haystack)
            text = " ".join((haystack * repeats)[:num_haystack])
        document_sents = [s.strip() for s in text.split(". ") if s]
        insertion_positions = (
            [0]
            + sorted(
                int(len(document_sents) * (depth / 100))
                for depth in random.sample(depths, len(needles))
            )
            + [len(document_sents)]
        )
        pieces: list[str] = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            pieces.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                pieces.append(needles[i - 1])
        context = " ".join(pieces)
    else:
        if type_haystack == "repeat":
            sentences = [haystack] * num_haystack
        else:  # needle
            sentences = [
                haystack.format(
                    type_needle_v=type_needle_v,
                    key=_random_value(type_needle_k, words),
                    value=_random_value(type_needle_v, words),
                )
                for _ in range(num_haystack)
            ]
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles, strict=True):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (
        ", ".join(queries[:-1]) + ", and " + queries[-1]
        if len(queries) > 1
        else queries[0]
    )

    template = _TEMPLATE
    tnv = type_needle_v
    if num_needle_q * num_needle_v == 1:
        template = (
            template.replace("Some", "A")
            .replace("are all", "is")
            .replace("are", "is")
            .replace("answers", "answer")
        )
        tnv = tnv[:-1]  # singularize

    input_text = template.format(type_needle_v=tnv, context=context, query=query)
    return input_text, answers
