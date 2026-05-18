# adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/prompts/code_generation.py
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
