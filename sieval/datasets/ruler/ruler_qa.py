"""RULER QA synthetic dataset (multi-document question answering).

One parameterized loader covering both RULER QA variants — ``dataset="squad"``
and ``dataset="hotpotqa"`` — selected via a ``load()`` arg. Synthesis is ported
from OpenCompass ``opencompass/datasets/ruler/ruler_qa.py``: read the source QA
pairs and their gold documents, pad each question with distractor documents up to
``max_seq_length`` (measured with a tiktoken/HF tokenizer), shuffle, and emit
``{prompt, answer}`` rows. The bound task does inference + substring scoring.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import json
import os
import random
from typing import TypedDict, override

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.ruler.datasets.constants import TASKS
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.datasets.ruler._common import build_tokenizer

_SQUAD_FILE = "dev-v2.0.json"
_HOTPOTQA_FILE = "hotpot_dev_distractor_v1.json"

_TEMPLATE = (
    "Answer the question based on the given documents. Only give me the answer "
    "and do not output any other words.\n\nThe following are given documents.\n\n"
    "{context}\n\nAnswer the question based on the given documents. Only give me "
    "the answer and do not output any other words.\n\nQuestion: {query} Answer:"
)
_DOCUMENT_PROMPT = "Document {i}:\n{document}"


class RulerQaDatasetSample(TypedDict):
    index: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str


@sieval_dataset(
    name="ruler_qa",
    display_name="RULER QA",
    description="RULER QA: answer over many distractor documents.",
    source=(
        "url:https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "url:http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    ),
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "open-ended", "long-context"),
    license="Apache-2.0",
    deps_group="ruler",
)
class RulerQaDataset(Dataset[RulerQaDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        dataset: str = "squad",
        max_seq_length: int = 4096,
        tokens_to_generate: int = 32,
        tokenizer_model: str = "gpt-4",
        num_samples: int = 500,
        pre_samples: int = 0,
        random_seed: int = 42,
        remove_newline_tab: bool = True,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = build_tokenizer(tokenizer_model)
        random.seed(random_seed)
        np.random.seed(random_seed)

        if dataset == "squad":
            qas, docs = _read_squad(os.path.join(name_or_path, _SQUAD_FILE))
        elif dataset == "hotpotqa":
            qas, docs = _read_hotpotqa(os.path.join(name_or_path, _HOTPOTQA_FILE))
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

        # Find the perfect num_docs
        incremental = 10
        num_docs = self._fit_num_docs(
            gen=gen,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            tokens_to_generate=tokens_to_generate,
            incremental=incremental,
        )

        # Generate samples
        rows = []
        for index in range(num_samples):
            used_docs = num_docs
            while True:
                try:
                    input_text, answer = gen(index + pre_samples, used_docs)
                    length = len(tokenizer.encode(input_text)) + tokens_to_generate
                    assert length <= max_seq_length, f"{length} exceeds max_seq_length"
                    break
                except AssertionError:
                    if used_docs > incremental:
                        used_docs -= incremental

            if remove_newline_tab:
                input_text = " ".join(
                    input_text.replace("\n", " ").replace("\t", " ").strip().split()
                )
            # Locate the answer prefix by its first 10 chars and split it off.
            qa_answer_prefix = str(TASKS["qa"]["answer_prefix"])
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

        return HFDatasetDict({"test": HFDataset.from_list(rows)})

    def _fit_num_docs(
        self,
        *,
        gen,
        tokenizer,
        max_seq_length: int,
        tokens_to_generate: int,
        incremental: int = 10,
    ) -> int:
        # Estimate tokens per question to determine a reasonable upper bound.
        sample_input_text, _ = gen(0, incremental)
        sample_tokens = len(tokenizer.encode(sample_input_text))
        tokens_per_doc = sample_tokens / incremental

        estimated_max_docs = int((max_seq_length / tokens_per_doc) * 3)

        # Binary search for optimal haystack size.
        lower_bound = incremental
        upper_bound = max(estimated_max_docs, incremental * 2)

        optimal_num_docs = None

        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            input_text, _ = gen(0, mid)
            total_tokens = len(tokenizer.encode(input_text)) + tokens_to_generate

            if total_tokens <= max_seq_length:
                # This size works, can we go larger?
                optimal_num_docs = mid
                lower_bound = mid + 1
            else:
                # Too large, need to go smaller
                upper_bound = mid - 1

        return optimal_num_docs if optimal_num_docs is not None else incremental


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


def _read_hotpotqa(path: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(set(total_docs))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
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
        # Repeat DOCS as many times as needed and slice to num_docs
        repeats = (num_docs + len(docs) - 1) // len(docs)  # Ceiling division
        all_docs = (docs * repeats)[:num_docs]

    random.Random(random_seed).shuffle(all_docs)
    context = "\n\n".join(
        _DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)
    )
    input_text = _TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a
