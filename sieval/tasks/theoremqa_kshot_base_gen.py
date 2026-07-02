"""
TheoremQA k-shot base generative task.

This implementation intentionally tracks the original TheoremQA vLLM
evaluation path: official short-form examples, matching stop tokens, and the
upstream answer_clean matcher. For score reproduction, configure the model
layer with the official decoding values: temperature=0, top_p=1, and
max_tokens=2048.
The few-shot prompt preserves official runtime artifacts, including the
approximation symbol and the control characters produced by non-raw LaTeX
escapes in the upstream examples.py.
Two common anomaly classes are therefore expected compatibility artifacts:
long chain-of-thought outputs can finish with reason="length", and repeated
"The answer is" triggers can be treated as ICL leakage by the upstream cleaner.

Known implementation deviations are import/runtime safety and typed
groundtruth plumbing: numeric eval is sandboxed instead of using upstream bare
eval, latex2sympy2 falls back to latex2sympy2_extended when the original
package is unavailable, and numeric groundtruths are derived from the
dataset's Answer_type field instead of the official loader's runtime Python
types to reproduce the same typed comparison inputs.

The Qwen2.5 technical report Table 2 lists TheoremQA as a 5-shot base-model
benchmark: Qwen2.5-72B scores 42.4, while 42.8 belongs to Qwen2-72B. By
matching the original TheoremQA runner strictly, this task reproduces a nearby
Qwen2.5-72B score under SiEval.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import ast
import contextlib
import math
import re
from collections.abc import Callable
from importlib import import_module
from math import cos, e, exp, factorial, log, pi, sin, sqrt
from typing import TypedDict, override

from sieval.core.models import ModelOutput
from sieval.core.tasks import EvalMode, ReferenceImpl, Task, sieval_task
from sieval.datasets import TheoremQADatasetSample


def _load_latex2sympy():
    # Official TheoremQA imports latex2sympy2. Python >=3.12 environments may
    # only have the compatible latex2sympy2_extended fallback, whose parser can
    # differ from upstream on edge cases.
    try:
        return import_module("latex2sympy2").latex2sympy
    except ModuleNotFoundError:
        return import_module("latex2sympy2_extended").latex2sympy


_LATEX2SYMPY: Callable[[str], object] | None = None


def _get_latex2sympy() -> Callable[[str], object]:
    global _LATEX2SYMPY
    if _LATEX2SYMPY is None:
        _LATEX2SYMPY = _load_latex2sympy()
    return _LATEX2SYMPY


E = 2.718
# Upstream uses bare eval(num). SiEval keeps eval sandboxed for task runtime
# safety, so builtins such as abs/round/pow are intentionally unavailable.
_EVAL_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "log": log,
    "pi": pi,
    "factorial": factorial,
    "exp": exp,
    "e": e,
    "E": E,
}

_DIRECT_ANSWER_TRIGGERS = ["The answer is:", "The answer is", "the answer is"]
_STOP_TOKENS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction:",
    "Response:",
    "<start_of_turn>",
    "[INST]",
    "\n\nProblem",
    "Problem:",
]
_THEOREMQA_RUN_URL = "https://github.com/TIGER-AI-Lab/TheoremQA/blob/acfc9686aa9b49f3c8f189364a9a9ee9c53da039/run.py"  # noqa: E501

# These strings mirror the official runtime examples, including artifacts from
# upstream non-raw string escapes. The model-facing prompt is intentionally more
# important here than source readability.
_THEOREMQA_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "In a 10 Gigabit Ethernet network, the average size of a frame is "
        "1500 bytes. If a burst of noise lasting 1ms interrupts the network, "
        "how many frames are lost?",
        "First, calculate the data rate in bytes/s:\n\n"
        "10 Gigabit/s * (1 Byte / 8 bits) = 1.25 * 10^9 Bytes/s\n\n"
        "Next, calculate the data loss in bytes due to the noise:\n\n"
        "1 ms * 1.25 * 10^9 Bytes/s = 1.25 * 10^6 Bytes\n\n"
        "Finally, divide the data loss by the average frame size to get the "
        "number of frames lost:\n\n"
        "1.25 * 10^6 Bytes / 1500 Bytes/frame \u2248 833.33 frames\n"
        "The answer is 833.33",
    ),
    (
        "Given x = 0.157, what is the value of x \\times "
        "\\frac{\\prod_{n=1}^\\infty (1 - \\frac{x^2}{n^2 \\pi^2})}"
        "{\\sin(x)}?",
        "To evaluate the expression $x \\times "
        "\\frac{\\prod_{n=1}^{\\infty} (1 - \\frac{x^2}{n^2 \\pi^2})}"
        "{\\sin(x)}$ given x = 0.157, we first recognize that the product "
        "in the numerator is related to the sine function through the "
        "Euler's reflection formula for the sine function, which can be "
        "expressed as:\n\n"
        "$$\\sin(x) = x \\prod_{n=1}^{\\infty} \\left(1 - "
        "\\frac{x^2}{n^2 \\pi^2}\\right)$$\n\n"
        "Therefore, the given expression simplifies to: $x \\times "
        "\\frac{\\sin(x)}{\\sin(x)}$\n\n"
        "Because sin(x) in the numerator and denominator cancels out, the "
        "expression simplifies further to just x.\n\n"
        "So, given x = 0.157, the value of the expression is 0.157. "
        "This result is derived from the properties of the sine function "
        "and does not require computational evaluation.\n"
        "The answer is 0.157",
    ),
    (
        "Consider the basis C of \\mathbb{R}^2 consisting of vectors "
        "u_1 = [2, 4] and u_2 = [1, -1]. If y = [8, 12], find the "
        "C-coordinate vector of y.",
        "The goal is to express y as a linear combination of the basis "
        "vectors of C, i.e., $y = a\\cdot u_1 + b\\cdot u_2$, where a and b "
        "are the scalar coefficients that we want to find. These coefficients "
        "will form the C-coordinate vector of y, which we'll denote as "
        "$[a, b]_C$.\n\n"
        "Given:\n"
        "- $u_1 = [2, 4]$,\n"
        "- $u_2 = [1, -1]$,\n"
        "- $y = [8, 12]$.\n\n"
        "We need to solve the system of linear equations:\n"
        "2a + 1b = 8\n"
        "4a - 1b = 12\n\n"
        "Let's solve this system of equations to find a and b.\n\n"
        "The solution to the system of equations is $a = \\frac{10}{3} and "
        "b = \\frac{4}{3}$. Therefore, the C-coordinate vector of y in the "
        "basis consisting of vectors u_1 = [2, 4] and u_2 = [1, -1] is "
        "$\\left[\\frac{10}{3}, \\frac{4}{3}\\right]_C$. \n"
        "Let's calculate the numerical value of "
        "$\\left[\frac{10}{3}, \frac{4}{3}\right]_C$ as [3.33, 1.33].\n"
        "The answer is [3.33, 1.33]",
    ),
    (
        "One can draw a simple, connected planar graph with 200 vertices and "
        "397 edges. Is this statement Trur or False?",
        "To determine the answer, we can use Euler's formula for planar "
        "graphs, which states that for any finite, connected, planar graph, "
        "$V - E + F = 2$, where V is the number of vertices, E is the number "
        "of edges, and F is the number of faces.\n\n"
        "Given the modified question, we have V = 200 vertices and E = 397 "
        "edges. We want to find if we can have a graph that satisfies these "
        "conditions, adhering to Euler's formula.\n\n"
        "First, let's rearrange Euler's formula to solve for F:  F = E - V + 2\n\n"
        "Substituting the given values: F = 397 - 200 + 2,  F = 199\n\n"
        "This means a graph with 200 vertices and 397 edges would have 199 "
        "faces. However, to determine the truth of this possibility, we "
        "should check if this graph doesn't violate any other planar graph "
        "constraints, particularly regarding the number of edges.\n\n"
        "For a simple, connected planar graph, there's also a relationship "
        "between vertices, edges, and faces given by the inequality: "
        "$E \\leq 3V - 6$\n\n"
        "Substituting V = 200 gives: $E \\leq 3*200 - 6 = 594$\n\n"
        "With E = 397, the condition $E \\leq 594$ is satisfied, meaning "
        "it's theoretically possible in terms of the edge condition for a "
        "planar graph.\n\n"
        "Therefore, one can draw a simple, connected planar graph with 200 "
        "vertices and 397 edges, resulting in 199 faces, without violating "
        "the conditions for it to be planar according to both Euler's formula "
        "and the constraint on the maximum number of edges.\n"
        "The answer is True",
    ),
    (
        "Given a finite group G, and a collection of permutations H on a set. "
        "Then (a) there always exists H such that G is isomorphic to H; "
        "(b) for any H, G is isomorphic to H; (c) G can never be isomorphic "
        "to H; (d) none of the above. Which option is correct?",
        "This is based on Cayley's theorem, which states that every group G "
        "is isomorphic to a subgroup of the symmetric group acting on G. \n"
        "In other words, for every finite group G, there exists a collection "
        "of permutations H (which in this context, can be thought of as the "
        "set of permutations representing the action of G on itself) such "
        "that G is isomorphic to H.\n\n"
        "Therefore, there always exists H such that G is isomorphic to H.\n"
        "The answer is (a)",
    ),
)

_DEFAULT_FEW_SHOT_COUNT = len(_THEOREMQA_EXAMPLES)


class Feedback(TypedDict):
    correct: bool
    pred: str
    answer: str


def floatify(num: str):
    try:
        num_float = float(num)
        if num_float.is_integer():
            return round(num_float)
        return num_float
    except Exception:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    return gt - eps <= pred <= gt + eps


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string: str):
        code_string = code_string.replace("\\pi", "\u03c0")
        code_string = re.sub(r"(?<![\d}])\\?\u03c0", "3.14", code_string)
        code_string = re.sub(r"(\d)(\\?\u03c0)", r"\1*3.14", code_string)
        code_string = re.sub(r"\{(\\?\u03c0)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?\u03c0)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("\u00a5", "")
    pred_str = pred_str.replace("\u00b0C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("\u00b0", "")
    return pred_str


def number_it(num):
    if isinstance(num, (int, float)):
        return num

    num = clean_units(num)
    with contextlib.suppress(Exception):
        num = str(_get_latex2sympy()(num))

    if floatify(num) is not None:
        return floatify(num)

    try:
        num_value = eval(num, _EVAL_GLOBALS)
        if isinstance(num_value, (list, tuple)):
            num_value = num_value[0]
        if floatify(num_value) is not None:
            return floatify(num_value)
        return None
    except Exception:
        return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    if len(pred) != len(gt):
        return False
    if any(not isinstance(x, (int, float)) for x in pred):
        return False
    pred = sorted(pred)
    gt = sorted(gt)
    return all(compare_two_numbers(p, g) for p, g in zip(pred, gt, strict=True))


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any(option in pred.lower() for option in ["yes", "true"]):
        pred = "True"
    elif any(option in pred.lower() for option in ["no", "false"]):
        pred = "False"
    elif any(
        option in pred.lower() for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    ):
        pass
    elif answer_flag:
        pred = pred.split("=")[-1].strip()
        pred = clean_units(pred)
        try:
            tmp = str(_get_latex2sympy()(pred))
            pred = str(eval(tmp, _EVAL_GLOBALS))
        except Exception:
            if re.match(r"-?[\d.]+\s\D+$", pred) or re.match(
                r"-?[\d.]+\s[^\s]+$",
                pred,
            ):
                pred = pred.split(" ")[0]
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred)
        pred = preds[-1] if len(preds) >= 1 else ""

    return pred


def answer_clean(direct_answer_trigger_for_fewshot: list[str], pred: str):
    pred = pred.strip("\n")

    icl = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            icl = True
    if icl:
        pred = pred.split("\n\n")[0]

    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")

    pred = extract_theoremqa_answer(pred, answer_flag)
    pred = pred.rstrip(".").rstrip("/")

    return pred


def compare_answer_with_groundtruth(
    answer: str,
    groundtruth_str: str,
    groundtruth_num=None,
):
    if groundtruth_str.lower() in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]:
        return groundtruth_str.lower() in answer.lower()
    if answer.lower() == groundtruth_str.lower():
        return True
    if groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return compare_two_numbers(number_it(answer), groundtruth_num)
        if answer.startswith("(") and answer.endswith(")"):
            try:
                answer_list = list(eval(answer, _EVAL_GLOBALS))
                answer_list = [number_it(a) for a in answer_list]
            except Exception:
                return False
            return compare_two_list(answer_list, groundtruth_num)
        return False
    return False


def _get_short_format(qas: list[tuple[str, str]]):
    tmp = "You are supposed to provide a solution to a given problem.\n\n"
    for q, a in qas:
        tmp += f"\nProblem:\n{q}\nSolution:\n{a}\n"
    prefix = "\nProblem:\n{query}\nSolution:\n"

    return tmp, prefix


def _parse_groundtruth_num(answer: str, answer_type: str):
    if answer_type == "integer":
        return int(answer)
    if answer_type == "float":
        return float(answer)
    if answer_type == "list of integer":
        value = ast.literal_eval(answer)
        return [int(v) for v in value]
    if answer_type == "list of float":
        value = ast.literal_eval(answer)
        return [float(v) for v in value]
    return None


def _groundtruth_args(sample: TheoremQADatasetSample):
    answer = str(sample["Answer"])
    try:
        groundtruth_num = _parse_groundtruth_num(answer, sample["Answer_type"])
    except Exception:
        groundtruth_num = None
    return answer, groundtruth_num


def _normalize_k(k: int | None) -> int:
    if k is None:
        return _DEFAULT_FEW_SHOT_COUNT
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}: {k!r}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > len(_THEOREMQA_EXAMPLES):
        raise ValueError(
            f"k must be <= {len(_THEOREMQA_EXAMPLES)} because only "
            "that many built-in TheoremQA examples are available."
        )
    return k


@sieval_task(
    name="theoremqa_kshot_base_gen",
    display_name="TheoremQA (k-shot, base generative)",
    description="TheoremQA k-shot ICL with official short prompt and matcher.",
    eval_mode=EvalMode.GEN,
    n_shot=_DEFAULT_FEW_SHOT_COUNT,
    tags=("english", "open-ended", "theorem-driven"),
    deps_group="math",
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="TIGER-AI-Lab/TheoremQA",
        url=_THEOREMQA_RUN_URL,
        notes=(
            "Prompt follows official short-form examples by default; k can "
            "select any prefix of the built-in examples. answer_clean and "
            "numeric matching mirror official utils.py/number_utils.py."
        ),
    ),
)
class TheoremQAKShotBaseGenTask(
    Task[TheoremQADatasetSample, str, ModelOutput, str, Feedback, dict[str, float]]
):
    def __init__(
        self, dataset, model, name: str | None = None, *, k: int | None = None
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = _normalize_k(k)
        self._prompt_no_input: str | None = None
        self._prompt_prefix: str | None = None

    @override
    async def setup(self) -> None:
        self._prompt_no_input, self._prompt_prefix = self._build_prompt_parts()

    @override
    async def preprocess(self, raw, ctx):
        prompt_no_input, prefix = self._get_prompt_parts()
        return prompt_no_input + prefix.format(query=raw["Question"])

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, stop=_STOP_TOKENS)

    @override
    async def postprocess(self, inf, ctx):
        text = inf.texts[0] if inf.texts else ""
        return answer_clean(_DIRECT_ANSWER_TRIGGERS, text)

    @override
    async def feedback(self, post, ctx):
        answer, groundtruth_num = _groundtruth_args(ctx.raw_sample)
        return True, {
            "correct": compare_answer_with_groundtruth(
                post,
                answer,
                groundtruth_num,
            ),
            "pred": post,
            "answer": answer,
        }

    @override
    async def report(self, finals, fails) -> dict[str, float]:
        count = len(finals)
        if count == 0:
            return {
                "score": 0.0,
                "accuracy": 0.0,
                "fails": len(fails),
                "empty": 0,
            }

        correct = sum(1 for ctx in finals if ctx.feedback_result["correct"])
        empty = sum(1 for ctx in finals if ctx.feedback_result["pred"] == "")
        accuracy = 100 * correct / count
        return {
            "score": accuracy,
            "accuracy": accuracy,
            "fails": len(fails),
            "empty": empty,
        }

    def _build_prompt_parts(self) -> tuple[str, str]:
        used_examples = list(_THEOREMQA_EXAMPLES[: self._k])
        return _get_short_format(used_examples)

    def _get_prompt_parts(self) -> tuple[str, str]:
        if self._prompt_no_input is None or self._prompt_prefix is None:
            self._prompt_no_input, self._prompt_prefix = self._build_prompt_parts()
        return self._prompt_no_input, self._prompt_prefix
