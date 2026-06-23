# adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/prompts/code_generation.py
import json
from pathlib import Path

_FEWSHOT_DIR = Path(__file__).parent / "few_shot_examples" / "generation"

# Upstream loads these few-shot pools at module level; we only fix the path to
# be package-relative. Each pool is the 2 upstream examples verbatim plus 1
# sieval-authored example (marked with a `_source` key, ignored by the template)
# so the base-model template can reach a 3-shot setting — upstream ships only 2.
with (_FEWSHOT_DIR / "func.json").open(encoding="utf-8") as _f:
    func = json.load(_f)

with (_FEWSHOT_DIR / "stdin.json").open(encoding="utf-8") as _f:
    stdin = json.load(_f)


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_GEMINI = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."

    SYSTEM_MESSAGE_GEMINITHINK = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_DEEPSEEK = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_CODEQWEN = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    SYSTEM_MESSAGE_QWEN_QWQ = "<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n<|im_start|>user"

    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def get_generic_question_template_answer(question: dict, cot: bool = False) -> str:
    prompt = f"### Question:\n{question['question_content']}\n\n"
    if question["starter_code"]:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    if cot:
        prompt += (
            "### Answer:\n"
            "Let's think step by step first, provide a concise plan:\n"
            "```text\n1. ...\n2. ...\n3. ...\n```\n"
            "Then provide the final solution: (use the provided format with backticks)\n\n"
        )
    else:
        prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


# adapted from upstream `get_base_model_question_template_answer`. Two divergences
# from upstream, both documented on the consuming task: (1) generalized from
# upstream's hardcoded single example to `n_shot` in-context examples; (2) takes a
# `dict` instead of a `CodeGenerationProblem` (matching this repo's
# `get_generic_question_template_answer` convention). Decomposed into prefix +
# target so the fixed few-shot prefix can be cached once per run instead of rebuilt
# per sample; `get_base_model_question_template_answer` is retained as the faithful
# entry point and, for `n_shot == 1`, its output is byte-identical to upstream.
def _format_base_example(example: dict, has_starter: bool) -> str:
    prompt = ""
    prompt += "### Question\n"
    prompt += example["question"]
    prompt += "\n\n"
    if has_starter:
        prompt += "### Starter Code\n"
        prompt += example["sample_code"]
        prompt += "\n\n"
    prompt += "### Answer\n\n"
    prompt += example["answer"]
    if example["answer"]:
        prompt += "\n\n"
    return prompt


def get_base_model_fewshot_prefix(has_starter: bool, n_shot: int = 1) -> str:
    """Fixed `n_shot` in-context prefix (no target). Cache once per `has_starter`."""
    examples_json = func if has_starter else stdin
    if n_shot < 0:
        raise ValueError(f"n_shot must be >= 0, got {n_shot}")
    if n_shot > len(examples_json):
        raise ValueError(
            f"n_shot={n_shot} exceeds the {len(examples_json)} available few-shot "
            f"examples for {'starter-code' if has_starter else 'stdin'} problems."
        )
    return "".join(
        _format_base_example(example, has_starter)
        for example in examples_json[:n_shot]
    )


def get_base_model_target_block(question_content: str, starter_code: str) -> str:
    """Per-sample trailing block: the target question with an empty answer."""
    return _format_base_example(
        {
            "question": question_content,
            "sample_code": starter_code,
            "answer": "",
        },
        bool(starter_code),
    )


def get_base_model_question_template_answer(question: dict, n_shot: int = 1) -> str:
    has_starter = bool(question["starter_code"])
    return get_base_model_fewshot_prefix(has_starter, n_shot) + (
        get_base_model_target_block(question["question_content"], question["starter_code"])
    )
