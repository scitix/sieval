"""Frequent Words Extraction (FWE) synthesis helpers for RULER."""

import random
import string

import numpy as np

from sieval.community.ruler.scripts.tokenizer import select_tokenizer

from ._shared import ruler_task, tokens_to_generate


def load_fwe(
    _name_or_path: str,
    *,
    max_seq_length: int,
    tokenizer_type: str,
    tokenizer_path: str,
    num_samples: int,
    random_seed: int,
    remove_newline_tab: bool,
    enable_thinking: bool,
    think_budget: int = 0,
    model_name: str = "qwen3",
    alpha: float,
    coded_wordlen: int,
    vocab_size: int,
) -> list[dict]:
    from scipy.special import zeta

    gen_budget = tokens_to_generate("freq_words_extraction", enable_thinking=enable_thinking, think_budget=think_budget, model_name=model_name)
    tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

    random.seed(random_seed)
    np.random.seed(random_seed)

    input_max_len = max_seq_length - gen_budget
    if vocab_size == -1:
        vocab_size = input_max_len // 50

    _, _, num_example_words = _generate_input_output(
        input_max_len,
        tokenizer=tokenizer,
        coded_wordlen=coded_wordlen,
        vocab_size=vocab_size,
        incremental=input_max_len // 32,
        alpha=alpha,
        random_seed=random_seed,
        zeta=zeta,
    )

    fwe_answer_prefix = ruler_task("freq_words_extraction")["answer_prefix"]
    rows: list[dict] = []
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
            zeta=zeta,
        )
        length = (
            len(tokenizer.text_to_tokens(input_text))
            + gen_budget
        )
        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )
        answer_prefix_index = input_text.rfind(fwe_answer_prefix[:10])
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
    return rows


def _generate_input_output(
    max_len: int,
    *,
    tokenizer,
    zeta,
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
    vocab[0] = "..."

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
