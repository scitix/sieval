"""MBPP prompt helpers adapted from lm-evaluation-harness.

The fixed task_id 2/3/4 few-shot examples are copied verbatim (including their
``\r\n`` line endings) from lm-evaluation-harness:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mbpp/utils.py
lm-evaluation-harness is distributed under the MIT License
(Copyright (c) 2020 EleutherAI). The example data itself originates from the
MBPP dataset (Austin et al., 2021), CC-BY-4.0.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from typing import TypedDict


class MBPPFewShotSample(TypedDict):
    task_id: int
    text: str
    code: str
    test_list: list[str]
    is_fewshot: bool


def list_fewshot_samples() -> list[MBPPFewShotSample]:
    return [
        {
            "task_id": 2,
            "text": (
                "Write a function to find the similar elements from the given two "
                "tuple lists."
            ),
            "code": (
                "def similar_elements(test_tup1, test_tup2):\r\n"
                "  res = tuple(set(test_tup1) & set(test_tup2))\r\n"
                "  return (res) "
            ),
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                (
                    "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) "
                    "== (13, 14)"
                ),
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": (
                "import math\r\n"
                "def is_not_prime(n):\r\n"
                "    result = False\r\n"
                "    for i in range(2,int(math.sqrt(n)) + 1):\r\n"
                "        if n % i == 0:\r\n"
                "            result = True\r\n"
                "    return result"
            ),
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": (
                "Write a function to find the largest integers from a given list "
                "of numbers using heap queue algorithm."
            ),
            "code": (
                "import heapq as hq\r\n"
                "def heap_queue_largest(nums,n):\r\n"
                "  largest_nums = hq.nlargest(n, nums)\r\n"
                "  return largest_nums"
            ),
            "test_list": [
                (
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, "
                    "22, 58],3)==[85, 75, 65] "
                ),
                (
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, "
                    "22, 58],2)==[85, 75] "
                ),
                (
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, "
                    "22, 58],5)==[85, 75, 65, 58, 35]"
                ),
            ],
            "is_fewshot": True,
        },
    ]
