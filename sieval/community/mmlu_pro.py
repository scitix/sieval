# adapted from https://github.com/open-compass/opencompass/blob/568572803ab108eb0e2ae73b770d965b7de078de/opencompass/configs/datasets/mmlu_pro/mmlu_pro_0shot_cot_gen_08c1de.py
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.

Question:\n
{question}

Options:\n
{options_str}

""".strip()

CHOICES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]
