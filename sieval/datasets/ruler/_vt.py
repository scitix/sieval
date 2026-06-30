"""Variable Tracking (VT) synthesis helpers for RULER."""

import heapq
import random
import string

import numpy as np

from sieval.community.ruler.scripts.tokenizer import select_tokenizer

from ._shared import (
    _VT_DEPTHS,
    _build_haystack,
    _ensure_punkt,
    ruler_task,
    tokens_to_generate,
)


def load_vt(
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
    num_chains: int,
    num_hops: int,
    type_haystack: str,
) -> list[dict]:
    gen_budget = tokens_to_generate("variable_tracking", enable_thinking=enable_thinking, think_budget=think_budget, model_name=model_name)
    tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

    random.seed(random_seed)
    np.random.seed(random_seed)

    haystack = _build_haystack(name_or_path, type_haystack)

    icl_example = _synthesize(
        tokenizer=tokenizer,
        num_samples=1,
        max_seq_length=500,
        num_chains=num_chains,
        num_hops=num_hops,
        tokens_to_generate=0,
        add_fewshot=True,
        icl_example=None,
        remove_newline_tab=False,
        type_haystack=type_haystack,
        haystack=haystack,
        final_output=False,
    )[0]

    return _synthesize(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
        num_chains=num_chains,
        num_hops=num_hops,
        tokens_to_generate=gen_budget,
        add_fewshot=True,
        icl_example=icl_example,
        remove_newline_tab=remove_newline_tab,
        type_haystack=type_haystack,
        haystack=haystack,
        final_output=True,
    )


def _synthesize(
    *,
    tokenizer,
    num_samples: int,
    max_seq_length: int,
    num_chains: int,
    num_hops: int,
    tokens_to_generate: int,
    icl_example: dict | None,
    remove_newline_tab: bool,
    type_haystack: str,
    haystack,
    final_output: bool = False,
    add_fewshot: bool = True,
) -> list[dict]:
    is_icl = add_fewshot and (icl_example is None)

    if icl_example is not None:
        incremental = 500 if type_haystack == "essay" else 10
        if type_haystack != "essay" and max_seq_length < 4096:
            incremental = 5
    else:
        incremental = 50 if type_haystack == "essay" else 5

    example_tokens = 0
    icl_text: str | None = None
    if add_fewshot and (icl_example is not None):
        icl_text = icl_example["input"] + " " + " ".join(icl_example["outputs"]) + "\n"
        example_tokens = len(tokenizer.text_to_tokens(icl_text))

    def gen(num_noises: int) -> tuple[str, list[str]]:
        return _generate_input_output(
            num_noises=num_noises,
            num_chains=num_chains,
            num_hops=num_hops,
            is_icl=is_icl,
            type_haystack=type_haystack,
            haystack=haystack,
        )

    num_noises = _binary_search_noises(
        gen=gen,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        tokens_to_generate=tokens_to_generate,
        example_tokens=example_tokens,
        incremental=incremental,
    )

    rows: list[dict] = []
    for index in range(num_samples):
        used_noises = num_noises
        while True:
            try:
                input_text, answer = gen(used_noises)
                if add_fewshot and (icl_text is not None):
                    cutoff = input_text.index(
                        ruler_task("variable_tracking")["template"][:20]
                    )
                    input_text = (
                        input_text[:cutoff]
                        + _randomize_icl(icl_text, num_hops)
                        + "\n"
                        + input_text[cutoff:]
                    )
                if remove_newline_tab:
                    input_text = " ".join(
                        input_text.replace("\n", " ").replace("\t", " ").strip().split()
                    )
                length = (
                    len(tokenizer.text_to_tokens(input_text))
                    + tokens_to_generate
                )
                assert length <= max_seq_length, "exceeds max_seq_length"
                break
            except Exception:
                if used_noises > incremental:
                    used_noises -= incremental
                else:
                    break
        if final_output:
            answer_prefix_index = input_text.rfind(
                ruler_task("variable_tracking")["answer_prefix"][:10]
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
        else:
            rows.append(
                {
                    "index": index,
                    "input": input_text,
                    "outputs": answer,
                    "length": length,
                }
            )
    return rows


def _binary_search_noises(
    *,
    gen,
    tokenizer,
    max_seq_length: int,
    tokens_to_generate: int,
    example_tokens: int,
    incremental: int,
) -> int:
    sample_text, _ = gen(incremental)
    sample_tokens = len(tokenizer.text_to_tokens(sample_text))
    tokens_per_haystack = sample_tokens / incremental
    estimated_max = int((max_seq_length / tokens_per_haystack) * 3)
    lower_bound, upper_bound = incremental, max(estimated_max, incremental * 2)
    optimal: int | None = None
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        text, _ = gen(mid)
        total = (
            len(tokenizer.text_to_tokens(text)) + example_tokens + tokens_to_generate
        )
        if total <= max_seq_length:
            optimal = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    return optimal if optimal is not None else incremental


def _generate_chains(
    num_chains: int, num_hops: int, is_icl: bool = False
) -> tuple[list[list[str]], list[list[str]]]:
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [
        "".join(random.choices(string.ascii_uppercase, k=k)).upper()
        for _ in range((num_hops + 1) * num_chains)
    ]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(random.choices(string.ascii_uppercase, k=k)).upper())
    vars_ret: list[list[str]] = []
    chains_ret: list[list[str]] = []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i : i + num_hops + 1]
        vars_ret.append(this_vars)
        if is_icl:
            this_chain = [f"VAR {this_vars[0]} = 12345"]
        else:
            this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret


def _shuffle_sublists_heap(lst: list[list[str]]) -> list[str]:
    heap: list[tuple[float, int, int]] = []
    for i in range(len(lst)):
        heapq.heappush(heap, (random.random(), i, 0))
    result: list[str] = []
    while heap:
        _, list_idx, elem_idx = heapq.heappop(heap)
        result.append(lst[list_idx][elem_idx])
        if elem_idx + 1 < len(lst[list_idx]):
            heapq.heappush(heap, (random.random(), list_idx, elem_idx + 1))
    return result


def _generate_input_output(
    *,
    num_noises: int,
    num_chains: int,
    num_hops: int,
    type_haystack: str,
    haystack,
    is_icl: bool = False,
) -> tuple[str, list[str]]:
    variables, chains = _generate_chains(num_chains, num_hops, is_icl=is_icl)
    value = chains[0][0].split("=")[-1].strip()
    if type_haystack == "essay":
        from nltk.tokenize import sent_tokenize

        text = " ".join(haystack[:num_noises])
        _ensure_punkt()
        document_sents = sent_tokenize(text.strip())
        chains_flat = _shuffle_sublists_heap(chains)
        insertion_positions = (
            [0]
            + sorted(
                int(len(document_sents) * (depth / 100))
                for depth in random.sample(_VT_DEPTHS, len(chains_flat))
            )
            + [len(document_sents)]
        )
        document_sents_list: list[str] = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(chains_flat):
                document_sents_list.append(chains_flat[i - 1].strip() + ".")
        context = " ".join(document_sents_list)
    elif type_haystack == "noise":
        sentences = [haystack] * num_noises
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences)), len(chain)))
            for insert_pi, j in zip(positions, range(len(chain)), strict=True):
                sentences.insert(insert_pi + j, chain[j])
        context = "\n".join(sentences)
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")
    context = context.replace(". \n", ".\n")
    template = (
        ruler_task("variable_tracking")["template"]
        + ruler_task("variable_tracking")["answer_prefix"]
    )
    input_text = template.format(context=context, query=value, num_v=num_hops + 1)
    return input_text, variables[0]


def _randomize_icl(icl_example: str, num_hops: int) -> str:
    icl_tgt = icl_example.strip().split()[-num_hops - 1 :]
    for item in icl_tgt:
        new_item = "".join(random.choices(string.ascii_uppercase, k=len(item))).upper()
        icl_example = icl_example.replace(item, new_item)
    icl_example = icl_example.replace("12345", str(np.random.randint(10000, 99999)))
    return icl_example
