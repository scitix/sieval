"""NIAH (Needle-in-a-Haystack) synthesis helpers for RULER."""

import random

import numpy as np

from sieval.community.ruler.scripts.tokenizer import select_tokenizer

from ._shared import (
    _NEEDLE,
    _NIAH_DEPTHS,
    _build_haystack,
    _ensure_punkt,
    ruler_task,
    tokens_to_generate,
)

# NIAH subtask → load() kwargs, from synthetic.yaml.
_NIAH_SUBTASK_KWARGS: dict[str, dict] = {
    "niah_single_1": {
        "type_haystack": "noise",
        "type_needle_k": "words",
        
        "type_needle_v": "numbers",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
    },
    "niah_single_2": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
    },
    "niah_single_3": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "uuids",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
    },
    "niah_multikey_1": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 4,
        "num_needle_v": 1,
        "num_needle_q": 1,
    },
    "niah_multikey_2": {
        "type_haystack": "needle",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
    },
    "niah_multikey_3": {
        "type_haystack": "needle",
        "type_needle_k": "uuids",
        "type_needle_v": "uuids",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
    },
    "niah_multivalue": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 1,
        "num_needle_v": 4,
        "num_needle_q": 1,
    },
    "niah_multiquery": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 4,
    },
}


def load_niah(
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
    num_needle_k: int,
    num_needle_v: int,
    num_needle_q: int,
    type_haystack: str,
    type_needle_k: str,
    type_needle_v: str,
) -> list[dict]:
    gen_budget = tokens_to_generate("niah", enable_thinking=enable_thinking, think_budget=think_budget, model_name=model_name)
    tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

    random.seed(random_seed)
    np.random.seed(random_seed)

    num_needle_k = max(num_needle_k, num_needle_q)
    haystack = _build_haystack(name_or_path, type_haystack)
    words = _niah_word_pool()

    def gen(num_haystack: int) -> tuple[str, list[str]]:
        return _generate_input_output(
            num_haystack=num_haystack,
            haystack=haystack,
            words=words,
            depths=_NIAH_DEPTHS,
            random_seed=random_seed,
            num_needle_k=num_needle_k,
            num_needle_v=num_needle_v,
            num_needle_q=num_needle_q,
            type_haystack=type_haystack,
            type_needle_k=type_needle_k,
            type_needle_v=type_needle_v,
        )

    num_haystack = _fit_haystack_size(
        gen=gen,
        tokenizer=tokenizer,
        haystack=haystack,
        type_haystack=type_haystack,
        max_seq_length=max_seq_length,
        tokens_to_generate=gen_budget,
    )

    incremental = _incremental(type_haystack, max_seq_length)
    niah_answer_prefix_template = ruler_task("niah")["answer_prefix"]

    rows: list[dict] = []
    for _ in range(num_samples):
        used_haystack = num_haystack
        while True:
            try:
                input_text, answer = gen(used_haystack)
                length = (
                    len(tokenizer.text_to_tokens(input_text))
                    + gen_budget
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
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )
        answer_prefix_index = input_text.rfind(niah_answer_prefix_template[:10])
        answer_prefix = input_text[answer_prefix_index:]
        input_text = input_text[:answer_prefix_index]
        index_in_text = input_text.find(answer[0])
        token_position_answer = len(
            tokenizer.text_to_tokens(input_text[:index_in_text])
        )
        rows.append(
            {
                "index": index_in_text,
                "input": input_text,
                "outputs": answer,
                "length": length,
                "answer_prefix": answer_prefix,
                "token_position_answer": token_position_answer,
            }
        )
    return rows


def _incremental(type_haystack: str, max_seq_length: int) -> int:
    if type_haystack == "essay":
        return 500
    if max_seq_length < 4096:
        return 5
    return 25


def _fit_haystack_size(
    *,
    gen,
    tokenizer,
    haystack,
    type_haystack: str,
    max_seq_length: int,
    tokens_to_generate: int,
) -> int:
    incremental = _incremental(type_haystack, max_seq_length)
    sample_prompt, _ = gen(incremental)
    tokens_per_haystack = len(tokenizer.text_to_tokens(sample_prompt)) / incremental
    estimated_max = int((max_seq_length / tokens_per_haystack) * 3)
    lower_bound = incremental
    upper_bound = max(estimated_max, incremental * 2)
    optimal: int | None = None
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        prompt, _ = gen(mid)
        total = len(tokenizer.text_to_tokens(prompt)) + tokens_to_generate
        if total <= max_seq_length:
            optimal = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    return optimal if optimal is not None else incremental


def _niah_word_pool() -> list[str]:
    import wonderwords

    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    return sorted(set(words))


def _generate_random_number(num_digits: int = 7) -> str:
    lower = 10 ** (num_digits - 1)
    upper = 10**num_digits - 1
    return str(random.randint(lower, upper))


def _generate_random_word(words: list[str]) -> str:
    return random.choice(words)


def _generate_random_uuid() -> str:
    import uuid

    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def _random_value(type_needle: str, words: list[str]) -> str:
    if type_needle == "numbers":
        return _generate_random_number()
    if type_needle == "words":
        return _generate_random_word(words)
    if type_needle == "uuids":
        return _generate_random_uuid()
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
        else:
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

    template = ruler_task("niah")["template"] + ruler_task("niah")["answer_prefix"]
    tnv = type_needle_v
    if num_needle_q * num_needle_v == 1:
        template = (
            template.replace("Some", "A")
            .replace("are all", "is")
            .replace("are", "is")
            .replace("answers", "answer")
        )
        tnv = tnv[:-1]

    input_text = template.format(type_needle_v=tnv, context=context, query=query)
    return input_text, answers
