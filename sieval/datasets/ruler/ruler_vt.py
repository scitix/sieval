"""RULER variable-tracking (VT) synthetic dataset.

Multi-hop tracing: the prompt hides one or more chains of variable assignments
(``VAR X = 12345`` then ``VAR Y = VAR X`` …) inside repeated noise sentences;
the model must name every variable that ultimately resolves to a given value.

Ported from original NVIDIA RULER ``scripts/data/synthetic/variable_tracking.py``
(not the OpenCompass reduction), so it reproduces RULER's two distinguishing
behaviours:

* **Binary-search sizing** — estimate tokens-per-noise once, then binary-search
  the noise count that fills ``max_seq_length`` (vs a linear scan).
* **Built-in 1-shot ICL** — RULER first synthesizes a small worked example
  (``max_seq_length=500``), then prepends a per-sample *randomized* copy of it
  (fresh variable names + value via :func:`_randomize_icl`) before each prompt.

Emits ``{prompt, answer}`` rows; the bound task does inference + substring
scoring (``string_match_all``).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import random
import string
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

_NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)
# Template head (RULER locates the ICL insertion point by this prefix) and the
# combined template + answer_prefix that the model actually sees.
_TEMPLATE_HEAD = "Memorize and track t"
_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the "
    "following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above. Answer: According to the "
    "chain(s) of variable assignment in the text above, {num_v} variables are "
    "assigned the value {query}, they are: "
)


class RulerVtDatasetSample(TypedDict):
    prompt: str
    answer: list[str]


@sieval_dataset(
    name="ruler_vt",
    display_name="RULER VT",
    description="RULER variable tracking: trace multi-hop variable assignments.",
    source=(),
    categories=(Category(Level1Category.LOGIC, "TextualReasoning"),),
    tags=("english", "open-ended", "long-context"),
    license="Apache-2.0",
    deps_group="ruler",
)
class RulerVtDataset(Dataset[RulerVtDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        *,
        max_seq_length: int = 4096,
        tokens_to_generate: int = 30,
        tokenizer_model: str = "gpt-4",
        num_chains: int = 1,
        num_hops: int = 4,
        num_samples: int = 500,
        random_seed: int = 42,
        remove_newline_tab: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = build_tokenizer(tokenizer_model)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 1) Synthesize the 1-shot ICL example (small, is_icl=True), then flatten
        #    it to a worked string "input + ' ' + answer + '\n'" (RULER form).
        icl_row = self._synthesize(
            tokenizer=tokenizer,
            num_samples=1,
            max_seq_length=500,
            num_chains=num_chains,
            num_hops=num_hops,
            tokens_to_generate=0,
            is_icl_gen=True,
            icl_example=None,
            remove_newline_tab=False,
        )[0]
        icl_example = icl_row["prompt"] + " " + " ".join(icl_row["answer"]) + "\n"

        # 2) Synthesize the real samples, each prefixed with a randomized ICL copy.
        rows = self._synthesize(
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            num_chains=num_chains,
            num_hops=num_hops,
            tokens_to_generate=tokens_to_generate,
            is_icl_gen=False,
            icl_example=icl_example,
            remove_newline_tab=remove_newline_tab,
        )

        return HFDatasetDict({"test": HFDataset.from_list(rows)})

    def _synthesize(
        self,
        *,
        tokenizer,
        num_samples: int,
        max_seq_length: int,
        num_chains: int,
        num_hops: int,
        tokens_to_generate: int,
        is_icl_gen: bool,
        icl_example: str | None,
        remove_newline_tab: bool,
    ) -> list[dict]:
        # Incremental matches RULER: icl-prefixed generation steps by 10
        # (5 for <4096), the ICL example itself steps by 5.
        if icl_example is not None:
            incremental = 5 if max_seq_length < 4096 else 10
        else:
            incremental = 5

        example_tokens = (
            len(tokenizer.encode(icl_example)) if icl_example is not None else 0
        )

        def gen(num_noises: int) -> tuple[str, list[str]]:
            return _generate_input_output(
                num_noises, num_chains, num_hops, is_icl=is_icl_gen
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
        for _ in range(num_samples):
            used_noises = num_noises
            while True:
                try:
                    prompt, answer = gen(used_noises)
                    if icl_example is not None:
                        # Insert a per-sample randomized ICL copy before the body.
                        cutoff = prompt.index(_TEMPLATE_HEAD)
                        prompt = (
                            prompt[:cutoff]
                            + _randomize_icl(icl_example, num_hops)
                            + "\n"
                            + prompt[cutoff:]
                        )
                    if remove_newline_tab:
                        prompt = " ".join(
                            prompt.replace("\n", " ").replace("\t", " ").strip().split()
                        )
                    length = len(tokenizer.encode(prompt)) + tokens_to_generate
                    assert length <= max_seq_length, "exceeds max_seq_length"
                    break
                except Exception:
                    if used_noises > incremental:
                        used_noises -= incremental
                    else:
                        break
            rows.append({"prompt": prompt, "answer": answer})
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
    """RULER's tokens-per-noise estimate + binary search for the largest fit."""
    sample_text, _ = gen(incremental)
    sample_tokens = len(tokenizer.encode(sample_text))
    tokens_per_noise = sample_tokens / incremental
    estimated_max = int((max_seq_length / tokens_per_noise) * 3)

    lower, upper = incremental, max(estimated_max, incremental * 2)
    optimal: int | None = None
    while lower <= upper:
        mid = (lower + upper) // 2
        text, _ = gen(mid)
        total = len(tokenizer.encode(text)) + example_tokens + tokens_to_generate
        if total <= max_seq_length:
            optimal = mid
            lower = mid + 1
        else:
            upper = mid - 1
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


def _generate_input_output(
    num_noises: int, num_chains: int, num_hops: int, is_icl: bool = False
) -> tuple[str, list[str]]:
    variables, chains = _generate_chains(num_chains, num_hops, is_icl=is_icl)
    value = chains[0][0].split("=")[-1].strip()

    sentences = [_NOISE] * num_noises
    for chain in chains:
        positions = sorted(random.sample(range(len(sentences)), len(chain)))
        for insert_pi, j in zip(positions, range(len(chain)), strict=True):
            sentences.insert(insert_pi + j, chain[j])
    context = "\n".join(sentences)
    context = context.replace(". \n", ".\n")

    input_text = _TEMPLATE.format(context=context, query=value, num_v=num_hops + 1)
    return input_text, variables[0]


def _randomize_icl(icl_example: str, num_hops: int) -> str:
    """Refresh the worked example: new variable names for the answer + a new value.

    Mirrors RULER's ``randomize_icl`` — replace the last ``num_hops + 1``
    whitespace tokens (the answer variable names) with fresh uppercase strings,
    and swap the literal root value ``12345`` for a new one.
    """
    icl_tgt = icl_example.strip().split()[-num_hops - 1 :]
    for item in icl_tgt:
        new_item = "".join(random.choices(string.ascii_uppercase, k=len(item))).upper()
        icl_example = icl_example.replace(item, new_item)
    icl_example = icl_example.replace("12345", str(np.random.randint(10000, 99999)))
    return icl_example
