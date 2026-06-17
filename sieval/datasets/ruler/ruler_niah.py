"""RULER NIAH (needle-in-a-haystack) synthetic dataset.

One parameterized loader covering all eight RULER NIAH variants (single_1/2/3,
multikey_1/2/3, multivalue, multiquery) via ``load()`` args. Synthesis is ported
from OpenCompass ``opencompass/datasets/ruler/ruler_niah.py``: build a haystack,
insert key/value needles at sampled depths, grow the haystack until it fills
``max_seq_length`` (measured with a tiktoken/HF tokenizer), and emit
``{prompt, answer}`` rows. The bound task does inference + substring scoring.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import random
import uuid
from typing import TypedDict, override

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.ruler.datasets.constants import TASKS
from sieval.community.ruler.scripts.tokenizer import select_tokenizer
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._common import _NEEDLE, _build_haystack


class RulerNiahDatasetSample(TypedDict):
    index: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str
    token_position_answer: int


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
        tokens_to_generate: int = TASKS['niah']['tokens_to_generate'],
        tokenizer_type: str = 'openai',
        tokenizer_path: str = 'cl100k_base',
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
        tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

        random.seed(random_seed)
        np.random.seed(random_seed)

        num_needle_k = max(num_needle_k, num_needle_q)

        haystack = _build_haystack(name_or_path, type_haystack)
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
            used_haystack = num_haystack
            while True:
                try:
                    input_text, answer = gen(used_haystack)
                    length = (
                        len(tokenizer.text_to_tokens(input_text)) + tokens_to_generate
                    )
                    assert length <= max_seq_length, "exceeds max_seq_length"
                    break
                except Exception:
                    if used_haystack > incremental:
                        used_haystack -= incremental
                    else:
                        input_text, answer = gen(used_haystack)
                        break
            if remove_newline_tab:
                input_text = " ".join(
                    [input_text.replace("\n", " ").replace("\t", " ").strip().split()]
                )
            # use first 10 char of answer prefix to locate it
            answer_prefix_index = input_text.rfind(TASKS['niah']['answer_prefix'][:10])
            answer_prefix = input_text[answer_prefix_index:]
            input_text = input_text[:answer_prefix_index]
            # find answer position in text
            index = input_text.find(answer[0])
            token_position_answer = len(tokenizer.text_to_tokens(input_text[:index]))
            rows.append(
                {
                    "index": index,
                    "input": input_text,
                    "outputs": answer,
                    "length": length,
                    "answer_prefix": answer_prefix,
                    'token_position_answer': token_position_answer,
                }
            )

        return HFDatasetDict({"test": HFDataset.from_list(rows)})

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
        tokens_per_haystack = len(tokenizer.text_to_tokens(sample_prompt)) / incremental

        estimated_max_questions = int((max_seq_length / tokens_per_haystack) * 3)

        # Binary search for optimal haystack size

        lower_bound = incremental
        upper_bound = max(estimated_max_questions, incremental * 2)
        optimal_haystack: int | None = None
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            prompt, _ = gen(mid)
            total_tokens = len(tokenizer.text_to_tokens(prompt)) + tokens_to_generate

            if total_tokens <= max_seq_length:
                optimal_haystack = mid
                lower_bound = mid + 1
            else:
                upper_bound = mid - 1
        return optimal_haystack if optimal_haystack is not None else incremental

def _ensure_punkt() -> None:
    """Ensure NLTK's ``punkt_tab`` sentence tokenizer is present.

    ``sent_tokenize`` (used for the ``essay`` haystack) loads ``punkt_tab`` on
    nltk >= 3.9. Mirrors RULER's ``prepare.py``: probe first, download only when
    missing, so the one-time fetch happens during data generation rather than at
    eval time.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

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

def _generate_random_number(num_digits=7) -> str:
    lower_bound_bound = 10**(num_digits - 1)
    upper_bound_bound = 10**num_digits - 1
    return str(random.randint(lower_bound_bound, upper_bound_bound))

def _generate_random_word(words) -> str:
    word = random.choice(words)
    return word

def _generate_random_uuid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def _random_value(type_needle: str, words: list[str]) -> str:
    if type_needle == "numbers":
        return _generate_random_number()
    if type_needle == "words":
        return _generate_random_word(words)
    if type_needle == "uuids":
        return _generate_random_uuid()
    else:
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

    # Context
    if type_haystack == "essay":
        # Repeat the essay when more words are needed than the corpus holds
        # (RULER behaviour); otherwise slice. Keeps very large contexts fillable.
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            repeats = (num_haystack + len(haystack) - 1) // len(haystack)
            text = " ".join((haystack * repeats)[:num_haystack])

        _ensure_punkt()
        from nltk.tokenize import sent_tokenize

        document_sents = sent_tokenize(text.strip())
        insertion_positions = (
            [0]
            + sorted(
                int(len(document_sents) * (depth / 100))
                for depth in random.sample(depths, len(needles))
            )
            + [len(document_sents)]
        )
        document_sents_list: list[str] = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                document_sents_list.append(needles[i - 1])
        context = " ".join(document_sents_list)
    else:
        if type_haystack == "noise":
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

    ## Query and Answer
    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (
        ", ".join(queries[:-1]) + ", and " + queries[-1]
        if len(queries) > 1
        else queries[0]
    )

    # RULER bakes answer_prefix into the template before generation (prepare.py),
    # so the prompt ends with the answer cue and the loader can split it back off.
    # The singularization below must therefore also rewrite the prefix's "are".
    template = TASKS['niah']['template'] + TASKS['niah']['answer_prefix']
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
