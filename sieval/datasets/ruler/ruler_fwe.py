"""RULER FWE (frequent words extraction) synthetic dataset.

Prompt synthesis is ported from NVIDIA RULER's
``scripts/data/synthetic/freq_words_extraction.py`` (Zipfian coded-word
generation, two-phase length fitting), refactored into a sieval Dataset loader.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import random
import string
from typing import TypedDict, override

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from scipy.special import zeta

from sieval.community.ruler.scripts.tokenizer import select_tokenizer
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)

from ._common import ruler_task, thinking_prefill


class RulerFweDatasetSample(TypedDict):
    index: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str


_DEFAULT_TOKENS_TO_GENERATE = ruler_task("freq_words_extraction")["tokens_to_generate"]


@sieval_dataset(
    name="ruler_fwe",
    display_name="RULER FWE",
    description="RULER frequent words extraction: report the top frequent coded words.",
    source=(),
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "open-ended", "long-context"),
    license="Apache-2.0",
    deps_group="ruler",
)
class RulerFweDataset(Dataset[RulerFweDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        max_seq_length: int = 4096,
        tokens_to_generate: int = _DEFAULT_TOKENS_TO_GENERATE,
        tokenizer_type: str = "openai",
        tokenizer_path: str = "cl100k_base",
        alpha: float = 2.0,
        coded_wordlen: int = 6,
        vocab_size: int = -1,
        num_samples: int = 500,
        random_seed: int = 42,
        remove_newline_tab: bool = False,
        enable_thinking: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Reserve budget for any assistant-turn prefill (e.g. Qwen3 thinking tags).
        thinking_overhead = len(
            tokenizer.text_to_tokens(thinking_prefill(tokenizer_path, enable_thinking))
        )

        input_max_len = max_seq_length - tokens_to_generate - thinking_overhead
        vocab_size = input_max_len // 50 if vocab_size == -1 else vocab_size

        _, _, num_example_words = _generate_input_output(
            input_max_len,
            tokenizer=tokenizer,
            coded_wordlen=coded_wordlen,
            vocab_size=vocab_size,
            incremental=input_max_len // 32,
            alpha=alpha,
            random_seed=random_seed,
        )

        rows = []
        for index in range(num_samples):
            input_text, answer, _ = _generate_input_output(
                input_max_len,
                tokenizer=tokenizer,
                num_words=num_example_words,
                coded_wordlen=coded_wordlen,
                vocab_size=vocab_size,
                incremental=input_max_len // 32,
                alpha=alpha,
                random_seed=random_seed,
            )

            length = (
                len(tokenizer.text_to_tokens(input_text))
                + tokens_to_generate
                + thinking_overhead
            )

            if remove_newline_tab:
                input_text = " ".join(
                    input_text.replace("\n", " ").replace("\t", " ").strip().split()
                )
            answer_prefix_index = input_text.rfind(
                ruler_task("freq_words_extraction")["answer_prefix"][:10]
            )
            answer_prefix = input_text[answer_prefix_index:]
            input_text = input_text[:answer_prefix_index]
            rows.append(
                {
                    "index": index,
                    "input": input_text,
                    "outputs": answer,
                    "length": length,
                    "answer_prefix": answer_prefix,
                }
            )

        return HFDatasetDict({"test": HFDataset.from_list(rows)})


def _generate_input_output(
    max_len: int,
    *,
    tokenizer,
    num_words: int = -1,
    coded_wordlen: int = 6,
    vocab_size: int = 2000,
    incremental: int = 10,
    alpha: float = 2.0,
    random_seed: int = 42,
) -> tuple[str, list[str], int]:
    vocab = [
        "".join(random.choices(string.ascii_lowercase, k=coded_wordlen))
        for _ in range(vocab_size)
    ]
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(set(vocab))
    random.Random(random_seed).shuffle(vocab)
    vocab[0] = "..."  # treat the top-ranked entry as noise

    def gen_text(n_words: int) -> tuple[str, list[str]]:
        k = np.arange(1, len(vocab) + 1)
        sampled_cnt = n_words * (k**-alpha) / zeta(alpha)
        sampled_words = [
            [w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int), strict=True)
        ]
        flat = [x for wlst in sampled_words for x in wlst]
        random.Random(random_seed).shuffle(flat)

        template = (
            ruler_task("freq_words_extraction")["template"]
            + ruler_task("freq_words_extraction")["answer_prefix"]
        )
        text = template.format(context=" ".join(flat), query="")
        return text, vocab[1:4]

    if num_words > 0:
        num_words = num_words
        text, answer = gen_text(num_words)
        while len(tokenizer.text_to_tokens(text)) > max_len:
            num_words -= incremental
            text, answer = gen_text(num_words)
    else:
        num_words = max_len // coded_wordlen
        text, answer = gen_text(num_words)
        while len(tokenizer.text_to_tokens(text)) < max_len:
            num_words += incremental
            text, answer = gen_text(num_words)
        num_words -= incremental
    text, answer = gen_text(num_words)
    return text, answer, num_words
