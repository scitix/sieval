"""RULER frequent-words-extraction (FWE) synthetic dataset.

Aggregation task: the prompt is a stream of coded words drawn from a Zipfian
distribution (a few words dominate, most are rare, with ``...`` injected as
noise); the model must name the three most frequent coded words. Synthesis is
ported from OpenCompass ``opencompass/datasets/ruler/ruler_fwe.py``: build a
random coded vocabulary, sample word counts as ``k^-alpha / zeta(alpha)``, and
grow the stream to fill ``max_seq_length`` (measured with a tiktoken/HF
tokenizer). Emits ``{prompt, answer}`` rows; the bound task does inference +
substring scoring.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import random
import string
from typing import TypedDict, override

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from scipy.special import zeta

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.datasets.ruler._common import build_tokenizer

_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. "
    "Find the three most frequently appeared coded words. {context}\nQuestion: "
    "Do not provide any explanation. Please ignore the dots '....'. What are the "
    "three most frequently appeared words in the above coded text? Answer: "
    "According to the coded text above, the three most frequently appeared words "
    "are:"
)


class RulerFweDatasetSample(TypedDict):
    prompt: str
    answer: list[str]


@sieval_dataset(
    name="ruler_fwe",
    display_name="RULER FWE",
    description="RULER frequent words extraction: report the top-3 coded words.",
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
        tokens_to_generate: int = 50,
        tokenizer_model: str = "gpt-4",
        alpha: float = 2.0,
        coded_wordlen: int = 6,
        vocab_size: int = -1,
        num_samples: int = 500,
        random_seed: int = 42,
        remove_newline_tab: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = build_tokenizer(tokenizer_model)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # RULER reserves the generation budget before sizing the context: the
        # coded-word stream is built to fill (max_seq_length - tokens_to_generate),
        # and vocab_size is derived from that reduced length.
        input_max_len = max_seq_length - tokens_to_generate
        resolved_vocab = input_max_len // 50 if vocab_size == -1 else vocab_size

        # Calibrate the number of words once, then reuse it for every sample.
        _, _, num_words = _generate_input_output(
            input_max_len,
            tokenizer=tokenizer,
            coded_wordlen=coded_wordlen,
            vocab_size=resolved_vocab,
            incremental=input_max_len // 32,
            alpha=alpha,
            random_seed=random_seed,
        )

        rows = []
        for _ in range(num_samples):
            prompt, answer, _ = _generate_input_output(
                input_max_len,
                tokenizer=tokenizer,
                num_words=num_words,
                coded_wordlen=coded_wordlen,
                vocab_size=resolved_vocab,
                incremental=input_max_len // 32,
                alpha=alpha,
                random_seed=random_seed,
            )
            if remove_newline_tab:
                prompt = " ".join(
                    prompt.replace("\n", " ").replace("\t", " ").strip().split()
                )
            rows.append({"prompt": prompt, "answer": answer})

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
        return _TEMPLATE.format(context=" ".join(flat), query=""), vocab[1:4]

    if num_words > 0:
        text, answer = gen_text(num_words)
        while len(tokenizer.encode(text)) > max_len:
            num_words -= incremental
            text, answer = gen_text(num_words)
    else:
        num_words = max_len // coded_wordlen
        text, answer = gen_text(num_words)
        while len(tokenizer.encode(text)) < max_len:
            num_words += incremental
            text, answer = gen_text(num_words)
        num_words -= incremental
    text, answer = gen_text(num_words)
    return text, answer, num_words
