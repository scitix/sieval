"""RULER common-words-extraction (CWE) synthetic dataset.

Aggregation task: the prompt is a long numbered list of words in which a handful
of "common" words repeat far more often than the rest; the model must report the
most frequent ones. Synthesis is ported from OpenCompass
``opencompass/datasets/ruler/ruler_cwe.py``: draw words from ``wonderwords``,
repeat common/uncommon words at configured frequencies, prepend a one-shot
example, and grow the list to fill ``max_seq_length`` (measured with a
tiktoken/HF tokenizer). Emits ``{prompt, answer}`` rows; the bound task does
inference + substring scoring.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import random
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

_TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more often "
    "than others. Memorize the ones that appear most often.\n{context}\n"
    "Question: What are the 10 most common words in the above list? Answer: The "
    "top 10 words that appear most often in the list are:"
)


class RulerCweDatasetSample(TypedDict):
    prompt: str
    answer: list[str]


@sieval_dataset(
    name="ruler_cwe",
    display_name="RULER CWE",
    description="RULER common words extraction: report the most frequent words.",
    source=(),
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "open-ended", "long-context"),
    license="Apache-2.0",
    deps_group="ruler",
)
class RulerCweDataset(Dataset[RulerCweDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        max_seq_length: int = 4096,
        tokens_to_generate: int = 120,
        tokenizer_model: str = "gpt-4",
        freq_cw: int = 30,
        freq_ucw: int = 3,
        num_cw: int = 10,
        num_samples: int = 500,
        random_seed: int = 42,
        remove_newline_tab: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = build_tokenizer(tokenizer_model)
        random.seed(random_seed)
        np.random.seed(random_seed)
        words = _word_pool(random_seed)

        def gen(num_words: int) -> tuple[str, list[str]]:
            return _generate_input_output(
                num_words=num_words,
                words=words,
                max_seq_length=max_seq_length,
                freq_cw=freq_cw,
                freq_ucw=freq_ucw,
                num_cw=num_cw,
                random_seed=random_seed,
            )

        incremental = 10
        num_words = self._binary_search_words(
            gen=gen,
            tokenizer=tokenizer,
            vocab_size=len(words),
            max_seq_length=max_seq_length,
            tokens_to_generate=tokens_to_generate,
            incremental=incremental,
        )

        rows = []
        for _ in range(num_samples):
            used_words = num_words
            while True:
                try:
                    prompt, answer = gen(used_words)
                    length = len(tokenizer.encode(prompt)) + tokens_to_generate
                    assert length <= max_seq_length, "exceeds max_seq_length"
                    break
                except Exception:
                    if used_words > incremental:
                        used_words -= incremental
                    else:
                        prompt, answer = gen(used_words)
                        break
            if remove_newline_tab:
                prompt = " ".join(
                    prompt.replace("\n", " ").replace("\t", " ").strip().split()
                )
            rows.append({"prompt": prompt, "answer": answer})

        return HFDatasetDict({"test": HFDataset.from_list(rows)})

    def _binary_search_words(
        self,
        *,
        gen,
        tokenizer,
        vocab_size: int,
        max_seq_length: int,
        tokens_to_generate: int,
        incremental: int,
    ) -> int:
        """RULER's tokens-per-word estimate + binary search for the largest fit.

        RULER falls back to a large ``english_words.json`` pool when the optimal
        word count exceeds the wonderwords vocabulary; that file is an unavailable
        git-LFS stub here, so the search is capped at ``vocab_size`` and a warning
        is logged when the estimate wanted more (the context can't be fully filled
        from wonderwords alone — relevant only at very large ``max_seq_length``).
        """
        from loguru import logger

        # Estimate tokens-per-word from a fixed 4096-word sample (RULER constant).
        sample_text, _ = gen(min(4096, vocab_size))
        tokens_per_word = len(tokenizer.encode(sample_text)) / min(4096, vocab_size)
        estimated_max = int(max_seq_length // tokens_per_word) * 2

        lower = incremental
        upper = max(estimated_max, incremental * 2)
        if upper > vocab_size:
            logger.warning(
                f"RULER CWE: estimated word count {upper} exceeds wonderwords "
                f"vocab {vocab_size}; capping (RULER would extend via "
                f"english_words.json, unavailable here). Prompts at "
                f"max_seq_length={max_seq_length} may underfill."
            )
            upper = vocab_size

        optimal: int | None = None
        while lower <= upper:
            mid = (lower + upper) // 2
            text, _ = gen(mid)
            total = len(tokenizer.encode(text)) + tokens_to_generate
            if total <= max_seq_length:
                optimal = mid
                lower = mid + 1
            else:
                upper = mid - 1
        return optimal if optimal is not None else incremental


def _word_pool(random_seed: int) -> list[str]:
    import wonderwords

    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
    words = sorted(set(nouns + adjs + verbs))
    random.Random(random_seed).shuffle(words)
    return words


def _get_example(
    *,
    num_words: int,
    words: list[str],
    common_repeats: int,
    uncommon_repeats: int,
    common_nums: int,
    random_seed: int,
) -> tuple[str, list[str]]:
    word_list_full = random.sample(words, num_words)
    common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    random.Random(random_seed).shuffle(word_list)
    context = " ".join(f"{i + 1}. {word}" for i, word in enumerate(word_list))
    return context, common


def _generate_input_output(
    *,
    num_words: int,
    words: list[str],
    max_seq_length: int,
    freq_cw: int,
    freq_ucw: int,
    num_cw: int,
    random_seed: int,
) -> tuple[str, list[str]]:
    if max_seq_length < 4096:
        context_example, answer_example = _get_example(
            num_words=20,
            words=words,
            common_repeats=3,
            uncommon_repeats=1,
            common_nums=num_cw,
            random_seed=random_seed,
        )
        context, answer = _get_example(
            num_words=num_words,
            words=words,
            common_repeats=6,
            uncommon_repeats=1,
            common_nums=num_cw,
            random_seed=random_seed,
        )
    else:
        context_example, answer_example = _get_example(
            num_words=40,
            words=words,
            common_repeats=10,
            uncommon_repeats=3,
            common_nums=num_cw,
            random_seed=random_seed,
        )
        context, answer = _get_example(
            num_words=num_words,
            words=words,
            common_repeats=freq_cw,
            uncommon_repeats=freq_ucw,
            common_nums=num_cw,
            random_seed=random_seed,
        )

    input_example = _TEMPLATE.format(context=context_example, query="") + " ".join(
        f"{i + 1}. {word}" for i, word in enumerate(answer_example)
    )
    input_text = _TEMPLATE.format(context=context, query="")
    return input_example + "\n" + input_text, answer
