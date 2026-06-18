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
import json
import os
import random
from typing import TypedDict, override

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.ruler.datasets.constants import TASKS
from sieval.community.ruler.scripts.tokenizer import select_tokenizer
from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)


class RulerCweDatasetSample(TypedDict):
    index: int
    input: str
    outputs: list[str]
    length: int
    answer_prefix: str


@sieval_dataset(
    name="ruler_cwe",
    display_name="RULER CWE",
    description="RULER common words extraction: report the most frequent words.",
    source=(
        "url:https://media.githubusercontent.com/media/NVIDIA/RULER/main/scripts/data/synthetic/json/english_words.json",
    ),
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
        tokens_to_generate: int = TASKS['common_words_extraction'][
            'tokens_to_generate'
        ],
        tokenizer_type: str = 'openai',
        tokenizer_path: str = 'cl100k_base',
        freq_cw: int = 30,
        freq_ucw: int = 3,
        num_cw: int = 10,
        num_samples: int = 500,
        random_seed: int = 42,
        num_fewshot: int = 1,
        remove_newline_tab: bool = False,
        enable_thinking: bool = False,
        **kwargs,
    ) -> HFDatasetDict:
        tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

        random.seed(random_seed)
        np.random.seed(random_seed)
        words = _word_pool(random_seed)

        # Overflow vocabulary (RULER's english_words.json), staged by
        # ``sieval dataset download`` into ``<datadir>/ruler_cwe/`` from the
        # ``url:`` source. Only consumed when more words are needed than the
        # wonderwords pool holds; load it when present, else fall back to empty.
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
                randle_words=randle_words
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

        # Generate samples
        rows = []
        # Account for thinking tags overhead when enable_thinking=False
        thinking_overhead = 0
        if enable_thinking is False:
            thinking_overhead = len(tokenizer.text_to_tokens("<think>\n\n</think>\n\n"))

        for index in range(num_samples):
            used_words = num_words
            while True:
                try:
                    input_text, answer = gen(used_words)
                    length = (
                        len(tokenizer.text_to_tokens(input_text)) + tokens_to_generate + thinking_overhead
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

            # use first 10 char of answer prefix to locate it
            answer_prefix_index = input_text.rfind(
                TASKS['common_words_extraction']['answer_prefix'][:10]
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
        from loguru import logger

        # Estimate tokens-per-word from a fixed 4096-word sample (RULER constant).
        sample_text, _ = gen(min(4096, vocab_size))
        tokens_per_word = len(tokenizer.text_to_tokens(sample_text)) / min(
            4096, vocab_size
        )

        estimated_max_words = int(max_seq_length // tokens_per_word) * 2

        lower_bound = incremental
        upper_bound = max(estimated_max_words, incremental * 2)

        if upper_bound > vocab_size:
            logger.warning(
                f"RULER CWE: estimated word count {upper_bound} exceeds wonderwords "
                f"vocab {vocab_size}; capping (RULER would extend via "
                f"english_words.json, unavailable here). Prompts at "
                f"max_seq_length={max_seq_length} may underfill."
            )
            upper_bound = vocab_size

        optimal_num_words: int | None = None

        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            input_text, _ = gen(mid)
            total_tokens = (
                len(tokenizer.text_to_tokens(input_text)) + tokens_to_generate
            )
            if total_tokens <= max_seq_length:
                optimal_num_words = mid
                lower_bound = mid + 1
            else:
                upper_bound = mid - 1
        return optimal_num_words if optimal_num_words is not None else incremental


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
    # RULER bakes answer_prefix into the template before generation
    # (prepare.py: ``template = model_template.format(task_template) + answer_prefix``),
    # so the prompt ends with the answer cue and the loader can split it back off.
    _template = (
        TASKS['common_words_extraction']['template']
        + TASKS['common_words_extraction']['answer_prefix']
    )
    for n in range(len(few_shots)):
        shot_answer = ' '.join(
            f"{i + 1}. {word}" for i, word in enumerate(few_shots[n][1])
        )
        few_shots[n] = (
            _template.format(num_cw=num_cw, context=few_shots[n][0], query='')
            + ' '
            + shot_answer
        )
    few_shots = "\n".join(few_shots)
    input_text = _template.format(num_cw=num_cw, context=context, query='')
    return few_shots + "\n" + input_text, answer
