# Adapted from DeepSeek-AI DeepSeek-Math, pinned commit:
# https://github.com/deepseek-ai/DeepSeek-Math/tree/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation
# Sources: data_processing/answer_extraction.py, eval/eval_utils.py, eval/eval_script.py.
"""
DeepSeek-Math answer extraction and answer equivalence.

Faithful port of the DeepSeek-Math answer-handling utilities from the pinned
commit (`data_processing/answer_extraction.py`, `eval/eval_utils.py`,
`eval/eval_script.py`), serving both the GSM8K 0-shot task and the MATH few-shot
task (per-consumer breakdown below):

* `extract_answer` (with `extract_boxed_answers` / `extract_program_output` /
  `strip_string`) — pull the final answer out of a model's reasoning: last
  ``\\boxed{...}`` if present, else the text after ``"he answer is"``, else the
  last number; then normalize.
* `math_equal` / `is_correct` — string, then numeric (with percentage /
  interval / matrix / equation handling), then sympy symbolic equivalence.

Shared across the DeepSeek-Math GSM8K and MATH tasks:
* `sieval.tasks.gsm8k_0shot_gen` calls `extract_answer(exhaust=False)` (=
  DeepSeek's `extract_last_single_answer`) and `is_correct` (=
  `eval_last_single_answer`).
* `sieval.tasks.hendrycks_math_kshot_base_gen` calls the Minerva 4-shot prompt
  (`few_shot_prompt` / `format_prompt` / `STOP_WORDS`), the multi-answer
  extractors (`extract_math_answer` / `extract_math_few_shot_cot_answer`), and
  `eval_math` — all vendored byte-faithfully from the same pinned commit
  (`few_shot_prompts/cot_minerva_math_4_shot.py` + the `math-cot-test` path).

Deviations from upstream:
- `math_equal` is only ever called with the default `timeout=False` (via
  `eval_math` / `is_correct` and the GSM8K path), so the
  `symbolic_equal_process` / `call_with_timeout` multiprocessing path is unused
  here; both are kept verbatim so `math_equal` stays byte-faithful and callable
  with `timeout=True`. (`symbolic_equal`'s `parse_latex` works once
  `antlr4-python3-runtime` is pinned to 4.11.0 in the `[math]` extra — sympy
  1.14's LaTeX grammar requires that ANTLR runtime.)
- The two debug `print` statements in `is_correct`'s list-branch ``'2,3,4'``
  guard are dropped (they fire during normal scoring — a library must not write
  to stdout). The lone `print(item)` before the final `NotImplementedError`
  (unreachable for GSM8K's single-string answers) is kept verbatim; control flow
  is otherwise byte-faithful.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import multiprocessing
import re
from copy import deepcopy
from math import isclose
from typing import Union

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string

def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers

def extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if '```output' in pred_str:
        pred_str = pred_str.split('```output')[-1]
    if '```' in pred_str:
        pred_str = pred_str.split('```')[0]
    output = pred_str.strip()
    return output

def extract_answer(pred_str, exhaust=False):
    pred = []
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = [tmp.split('$. I hope', 1)[0].strip()]
    elif 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif ('he answer is' in pred_str):
        pred = [pred_str.split('he answer is')[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else: # use the last number
            pattern = '-?\d*\.?\d+'
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if(len(ans) >= 1):
                ans = ans[-1]
            else:
                ans = ''
            if ans:
                pred.append(ans)

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""

def parse_digits(num):
    # format: 234.23 || 23%
    num = regex.sub(',', '', str(num))
    try:
        return float(num)
    except:
        if num.endswith('%'):
            num = num[:-1]
            if num.endswith('\\'):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None

def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s
    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def math_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: bool = True,
                timeout: bool = False,
                ) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if str(prediction) == str(reference):
        return True

    try: # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, abs_tol=1e-3):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if regex.match(r'(\(|\[).+(\)|\])', prediction) is not None and regex.match(r'(\(|\[).+(\)|\])', reference) is not None:
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                return True

    if (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}")) and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}")) and \
        (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}")) and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")):
        pred_lines = [line.strip() for line in prediction[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\") if line.strip()]
        ref_lines = [line.strip() for line in reference[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\") if line.strip()]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count('=') == 1 and reference.count('=') == 1:
        pred = prediction.split('=')
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split('=')
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count('=') == 1 and len(prediction.split('=')[0].strip()) <= 2 and '=' not in reference:
        if math_equal(prediction.split('=')[1], reference, include_percentage, is_close):
            return True
    elif reference.count('=') == 1 and len(reference.split('=')[0].strip()) <= 2 and '=' not in prediction:
        if math_equal(prediction, reference.split('=')[1], include_percentage, is_close):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False

def is_correct(item, pred_key='prediction', prec=1e-3):
    pred = item[pred_key]
    ans = item['answer']
    if isinstance(pred, list) and isinstance(ans, list):
        pred_matched = set()
        ans_matched = set()
        for i in range(len(pred)):
            for j in range(len(ans)):
                item_cpy = deepcopy(item)
                item_cpy.update({
                    pred_key: pred[i],
                    'answer': ans[j]
                })
                if is_correct(item_cpy, pred_key=pred_key, prec=prec):
                    pred_matched.add(i)
                    ans_matched.add(j)
        return len(pred_matched) == len(pred) and len(ans_matched) == len(ans)
    elif isinstance(pred, str) and isinstance(ans, str):
        if '\\cup' in pred and '\\cup' in ans:
            item = deepcopy(item)
            item.update({
                pred_key: pred.split('\\cup'),
                'answer': ans.split('\\cup'),
            })
            return is_correct(item, pred_key=pred_key, prec=prec)
        else:
            label = False
            try:
                label = abs(float(regex.sub(r',', '', str(pred))) - float(regex.sub(r',', '', str(ans)))) < prec
            except:
                pass
            label = label or (ans and pred == ans) or math_equal(pred, ans)
            return label
    else:
        print(item, flush=True)
        raise NotImplementedError()


# --- MATH (CoT) path: Minerva 4-shot prompt + multi-answer extraction/scoring ---
# Vendored from the same pinned commit: few_shot_prompts/cot_minerva_math_4_shot.py
# (MinervaMathPrompt) and the math-cot-test extract_math_answer / eval_math path.

few_shot_prompt = """Problem:
Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative.
Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.
Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."""


STOP_WORDS = ["\nProblem:"]


def format_prompt(task_input: str, task_output: str = "") -> str:
    prompt = f"{few_shot_prompt}\n\nProblem:\n{task_input}\n\nSolution:\n{task_output}"
    return prompt.rstrip()


def extract_math_answer(question, reasoning, task):
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if 'separated by commas' in question and all(ch not in ans for ch in '()[]'):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())
    return answer

def extract_math_few_shot_cot_answer(question, reasoning, task):
    if 'Problem:' in reasoning:
        reasoning = reasoning.split("Problem:", 1)[0]
    return extract_math_answer(question, reasoning, task)

def eval_math(item, pred_key='prediction', prec=1e-3):
    pred = item[pred_key]
    if pred_key == 'program_output' and isinstance(pred, str):
        pred = [pred]
    ans = item['answer']
    if isinstance(pred, list) and isinstance(ans, list):
        # for some questions in MATH, `reference` repeats answers
        _ans = []
        for a in ans:
            if a not in _ans:
                _ans.append(a)
        ans = _ans
        # some predictions for MATH questions also repeats answers
        _pred = []
        for a in pred:
            if a not in _pred:
                _pred.append(a)
        # some predictions mistakenly box non-answer strings
        pred = _pred[-len(ans):]

    item.update({
        pred_key: pred,
        'answer': ans
    })
    return is_correct(item, pred_key=pred_key, prec=prec)
