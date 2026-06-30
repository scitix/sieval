"""Common Words Extraction (CWE) synthesis helpers for RULER."""

import json
import os
import random

from sieval.community.ruler.scripts.tokenizer import select_tokenizer

from ._shared import ruler_task, tokens_to_generate


def load_cwe(
    name_or_path: str,
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
    freq_cw: int,
    freq_ucw: int,
    num_cw: int,
    num_fewshot: int,
) -> list[dict]:
    gen_budget = tokens_to_generate("common_words_extraction", enable_thinking=enable_thinking, think_budget=think_budget, model_name=model_name)
    tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

    random.seed(random_seed)

    words = _word_pool(random_seed)

    randle_words: list[str] = []
    randle_path = os.path.join(name_or_path, "english_words.json")
    if os.path.exists(randle_path):
        with open(randle_path) as f:
            randle_words = list(json.load(f).values())

    def gen(num_words: int) -> tuple[str, list[str]]:
        return _generate_input_output(
            num_words=num_words,
            words=words,
            max_seq_length=max_seq_length,
            freq_cw=freq_cw,
            freq_ucw=freq_ucw,
            num_cw=num_cw,
            random_seed=random_seed,
            num_fewshot=num_fewshot,
            randle_words=randle_words,
        )

    incremental = 10
    num_words = _binary_search_words(
        gen=gen,
        tokenizer=tokenizer,
        vocab_size=len(words),
        max_seq_length=max_seq_length,
        tokens_to_generate=gen_budget,
        incremental=incremental,
    )

    cwe_answer_prefix = ruler_task("common_words_extraction")["answer_prefix"]

    rows: list[dict] = []
    for index in range(num_samples):
        used_words = num_words
        while True:
            try:
                input_text, answer = gen(used_words)
                length = (
                    len(tokenizer.text_to_tokens(input_text))
                    + gen_budget
                )
                assert length <= max_seq_length, "exceeds max_seq_length"
                break
            except Exception:
                if used_words > incremental:
                    used_words -= incremental
                else:
                    break
        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )
        answer_prefix_index = input_text.rfind(cwe_answer_prefix[:10])
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


def _binary_search_words(
    *,
    gen,
    tokenizer,
    vocab_size: int,
    max_seq_length: int,
    tokens_to_generate: int,
    incremental: int,
) -> int:
    from loguru import logger

    sample_text, _ = gen(min(4096, vocab_size))
    tokens_per_word = len(tokenizer.text_to_tokens(sample_text)) / min(4096, vocab_size)
    estimated_max_words = int(max_seq_length // tokens_per_word) * 2
    lower_bound = incremental
    upper_bound = max(estimated_max_words, incremental * 2)
    if upper_bound > vocab_size:
        logger.warning(
            f"RULER CWE: estimated word count {upper_bound} exceeds wonderwords "
            f"vocab {vocab_size}; capping. Prompts at "
            f"max_seq_length={max_seq_length} may underfill."
        )
        upper_bound = vocab_size
    optimal: int | None = None
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, _ = gen(mid)
        total_tokens = len(tokenizer.text_to_tokens(input_text)) + tokens_to_generate
        if total_tokens <= max_seq_length:
            optimal = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
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
    randle_words: list[str],
    common_repeats: int,
    uncommon_repeats: int,
    common_nums: int,
    random_seed: int,
) -> tuple[str, list[str]]:
    if num_words <= len(words):
        word_list_full = random.sample(words, num_words)
    else:
        word_list_full = random.sample(randle_words, num_words)
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
    num_fewshot: int,
    randle_words: list[str],
) -> tuple[str, list[str]]:
    few_shots = []
    if max_seq_length < 4096:
        for _ in range(num_fewshot):
            context_example, answer_example = _get_example(
                num_words=20,
                words=words,
                randle_words=randle_words,
                common_repeats=3,
                uncommon_repeats=1,
                common_nums=num_cw,
                random_seed=random_seed,
            )
            few_shots.append((context_example, answer_example))
        context, answer = _get_example(
            num_words=num_words,
            words=words,
            randle_words=randle_words,
            common_repeats=6,
            uncommon_repeats=1,
            common_nums=num_cw,
            random_seed=random_seed,
        )
    else:
        for _ in range(num_fewshot):
            context_example, answer_example = _get_example(
                num_words=40,
                words=words,
                randle_words=randle_words,
                common_repeats=10,
                uncommon_repeats=3,
                common_nums=num_cw,
                random_seed=random_seed,
            )
            few_shots.append((context_example, answer_example))
        context, answer = _get_example(
            num_words=num_words,
            words=words,
            randle_words=randle_words,
            common_repeats=freq_cw,
            uncommon_repeats=freq_ucw,
            common_nums=num_cw,
            random_seed=random_seed,
        )
    _template = (
        ruler_task("common_words_extraction")["template"]
        + ruler_task("common_words_extraction")["answer_prefix"]
    )
    for n in range(len(few_shots)):
        shot_answer = " ".join(
            f"{i + 1}. {word}" for i, word in enumerate(few_shots[n][1])
        )
        few_shots[n] = (
            _template.format(num_cw=num_cw, context=few_shots[n][0], query="")
            + " "
            + shot_answer
        )
    few_shots_text = "\n".join(few_shots)
    input_text = _template.format(num_cw=num_cw, context=context, query="")
    return few_shots_text + "\n" + input_text, answer
