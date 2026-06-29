"""QA (SQuAD / HotpotQA) synthesis helpers for RULER."""

import json
import os
import random

from sieval.community.ruler.scripts.tokenizer import select_tokenizer
from sieval.core.utils.hf import ensure_dataset

from ._shared import (
    _DOCUMENT_PROMPT,
    _HOTPOTQA_REVISION,
    _SQUAD_FILE,
    ruler_task,
    tokens_to_generate,
)


def load_qa(
    name_or_path: str,
    *,
    dataset: str,
    max_seq_length: int,
    tokenizer_type: str,
    tokenizer_path: str,
    num_samples: int,
    random_seed: int,
    remove_newline_tab: bool,
    enable_thinking: bool,
    think_budget: int = 0,
    pre_samples: int,
) -> list[dict]:
    gen_budget = tokens_to_generate("qa", enable_thinking=enable_thinking, think_budget=think_budget)
    tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

    random.seed(random_seed)

    if dataset == "squad":
        qas, docs = _read_squad(os.path.join(name_or_path, _SQUAD_FILE))
    elif dataset == "hotpotqa":
        qas, docs = _read_hotpotqa(name_or_path)
    else:
        raise NotImplementedError(f"{dataset} is not implemented.")

    def gen(index: int, num_docs: int) -> tuple[str, list[str]]:
        return _generate_input_output(
            index=index,
            num_docs=num_docs,
            qas=qas,
            docs=docs,
            random_seed=random_seed,
        )

    incremental = 10
    num_docs = _fit_num_docs(
        gen=gen,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        tokens_to_generate=gen_budget,
        incremental=incremental,
    )

    qa_answer_prefix = ruler_task("qa")["answer_prefix"]

    rows: list[dict] = []
    for index in range(num_samples):
        used_docs = num_docs
        while True:
            try:
                input_text, answer = gen(index + pre_samples, used_docs)
                length = (
                    len(tokenizer.text_to_tokens(input_text))
                    + gen_budget
                )
                assert length <= max_seq_length, f"{length} exceeds max_seq_length"
                break
            except AssertionError:
                if used_docs > incremental:
                    used_docs -= incremental
        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )
        answer_prefix_index = input_text.rfind(qa_answer_prefix[:10])
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


def _fit_num_docs(
    *,
    gen,
    tokenizer,
    max_seq_length: int,
    tokens_to_generate: int,
    incremental: int = 10,
) -> int:
    sample_input_text, _ = gen(0, incremental)
    tokens_per_doc = len(tokenizer.text_to_tokens(sample_input_text)) / incremental
    estimated_max_docs = int((max_seq_length / tokens_per_doc) * 3)
    lower_bound = incremental
    upper_bound = max(estimated_max_docs, incremental * 2)
    optimal: int | None = None
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, _ = gen(0, mid)
        total_tokens = len(tokenizer.text_to_tokens(input_text)) + tokens_to_generate
        if total_tokens <= max_seq_length:
            optimal = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    return optimal if optimal is not None else incremental


def _read_squad(path: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(set(total_docs))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )
    return total_qas, total_docs


def _read_hotpotqa(name_or_path: str) -> tuple[list[dict], list[str]]:
    from datasets import load_dataset as hf_load_dataset

    try:
        raw = hf_load_dataset(name_or_path, "distractor", split="validation")
    except (ValueError, FileNotFoundError):
        raw = hf_load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
            revision=_HOTPOTQA_REVISION,
        )
    data = ensure_dataset(raw)
    total_docs_set: dict[str, int] = {}
    for row in data:
        ctx = row["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"], strict=True):
            doc = f"{title}\n{''.join(sents)}"
            if doc not in total_docs_set:
                total_docs_set[doc] = len(total_docs_set)
    total_docs = sorted(total_docs_set, key=lambda d: total_docs_set[d])
    total_docs_dict = {d: i for i, d in enumerate(total_docs)}
    total_qas = []
    for row in data:
        ctx = row["context"]
        context_indices = [
            total_docs_dict[f"{t}\n{''.join(s)}"]
            for t, s in zip(ctx["title"], ctx["sentences"], strict=True)
        ]
        total_qas.append(
            {
                "query": row["question"],
                "outputs": [row["answer"]],
                "context": context_indices,
            }
        )
    return total_qas, total_docs


def _generate_input_output(
    *,
    index: int,
    num_docs: int,
    qas: list[dict],
    docs: list[str],
    random_seed: int,
) -> tuple[str, list[str]]:
    curr = qas[index]
    curr_q = curr["query"]
    curr_a = curr["outputs"]
    curr_docs = curr["context"]
    curr_more = curr.get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i in range(len(docs)) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs,
                    max(0, num_docs - len(curr_docs) - len(curr_more)),
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [docs[idx] for idx in all_docs]
    else:
        repeats = (num_docs + len(docs) - 1) // len(docs)
        all_docs = (docs * repeats)[:num_docs]
    random.Random(random_seed).shuffle(all_docs)
    context = "\n\n".join(
        _DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)
    )
    template = ruler_task("qa")["template"] + ruler_task("qa")["answer_prefix"]
    input_text = template.format(context=context, query=curr_q)
    return input_text, curr_a
