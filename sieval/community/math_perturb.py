# Adapted from the MATH-Perturb Team's MATH-Perturb, pinned commit:
# https://github.com/Kaffaljidhmah2/MATH-Perturb/tree/df4840f680fce405c9449008564574961c7f4df1
# Sources: evaluate.py, evaluation/answer_extraction.py, evaluation/eval_utils.py,
# evaluation/eval_script.py. Upstream is Apache-2.0 (LICENSE at the pinned commit).
"""
MATH-Perturb answer extraction and answer equivalence.

Faithful port of MATH-Perturb's own evaluation package at the pinned commit,
carrying its single entry point `answer_check(problem, solution_str,
ground_truth, dataset_type)` and the two extractors it composes.

**This is a fork of DeepSeek-Math's evaluation code, not a copy of it**, which is
why it lives beside `sieval.community.deepseek_math` instead of importing from
it. Upstream says so itself (`evaluation/README.md`: "modified from
DeepSeek-Math/evaluation") and lists four classes of change; the ported diff
against DeepSeek-Math's `b8b0f8ce` is:

* `abs_tol` 1e-3 -> `MAX_ABS_TOL` = 1e-7, at both `math_equal` call sites and as
  the `prec` `answer_check` passes into `eval_math`. Upstream's stated reason:
  the looser tolerance called 1/5120 equal to 1/28800.
* `parse_latex` is wrapped to pass `backend="lark"` rather than sympy's default
  ANTLR backend. A different grammar, not a different spelling of one — see the
  measurement in `sieval/tasks/_math_perturb_base.py`.
* Unicode normalization (`_fix_unicode`, called from `strip_string`), added
  because o1 / o3-mini emit `√`, `π`, `²`, `−` and sans-bold digits.
* LaTeX handling fixes: `\\boxed(...)` / `\\boxed[...]` and `boxed {` spacing,
  `{,}` thousands separators (`parse_digits`, `is_correct`), spacing macros
  `\\ ` / `\\,` / `\\:` / `\\;` / `\\quad` stripped, and three upstream lines
  *disabled* — `strip_string` no longer deletes all spaces or `\\cdot`.
* `extract_math_answer` gains `\\text{or}` / bare `and` / bare `or` splits, and
  `extract_math_perturb_ground_truth_answer` is new: the gold-side extractor,
  splitting on a bare `" or "` because some MATH-Perturb labels are multi-valued.

Ported surface: everything `answer_check` reaches, which is `evaluate.py` whole
plus, from `evaluation/`, `strip_string` and its four `_fix_*` helpers,
`extract_boxed_answers` / `extract_program_output` / `extract_answer`, both
`extract_math*` functions, `MAX_ABS_TOL` / `parse_latex` / `parse_digits` /
`is_digit` / `math_equal` / `symbolic_equal` and `eval_math` / `is_correct`.
Upstream's unreached helpers (`extract_program`, `parse_ground_truth`,
`parse_question`, `run_execute`, `normalize_prediction`, `math_equal_process`,
and the dozen benchmark-specific extractors DeepSeek-Math left in
`answer_extraction.py`) are not carried, matching how `deepseek_math` is scoped
here. `numpy` is not imported for the same reason: only `normalize_prediction`
used it.

Deviations from upstream (documented, not silent):

- **`symbolic_equal` does not execute model output.** Upstream inherits
  DeepSeek-Math's hole unchanged: a prediction is parsed with a bare
  `parse_expr`, whose default namespace carries `__builtins__`, and when both
  parsers fail `_parse` returns the *raw string*, which reaches `N` — and `N`
  sympifies it with sympy's own default namespace. (Only `N`: `simplify(a - b)`
  raises `TypeError` first, since sympy's arithmetic dunders sympify strictly.)
  Either route runs `__import__('os').system(...)` supplied as a boxed answer,
  and the grader still reports the sample wrong, so nothing in the run looks
  unusual. The same two changes `deepseek_math` takes close it here: `parse_expr`
  runs under `_sympy_guards` (cleared namespace, quote screen, unevaluated
  exponent pre-parse), and an unparseable answer becomes `None` and refuses the
  comparison rather than reaching `N` as text. Both modules import the guards
  from `_sympy_guards` so a new escape route closes in one place.
  The measured cost of the guards on this benchmark's own data is in
  `sieval/tasks/_math_perturb_base.py`.
- `math_equal` is only ever called with the default `timeout=False` (via
  `eval_math` / `is_correct`), so the `symbolic_equal_process` /
  `call_with_timeout` multiprocessing path is unused; both are kept verbatim so
  `math_equal` stays byte-faithful and callable with `timeout=True`. sieval bounds
  grading in the worker process instead (`GRADE_TIMEOUT`).
- Four `print` statements that fire during normal scoring are dropped — a library
  must not write to stdout. They are `_fix_unicode`'s per-conversion `DEBUG:`
  line (and with it the `before` local, which fed nothing else), the `'2,3,4'`
  guard inside `is_correct`'s list branch, and `evaluate.py`'s "Multi-valued
  ground truth:" / "multi-valued prediction:" notices. The first also costs
  `extract_ground_truth_answer`'s `ground_truth_answer_stripped` local, which
  only that notice read. All four are pure observation: no branch depends on
  them, so control flow is byte-faithful. The lone `print(item)` before
  `is_correct`'s final `NotImplementedError` is kept verbatim (unreachable for
  the list/list and str/str shapes `answer_check` produces), as in
  `deepseek_math`.
- Upstream's `test_parse_latex()` self-check and its `__main__` block are not
  carried as module code; the same assertion runs as a unit test.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import multiprocessing
import re
from copy import deepcopy
from math import isclose
from typing import Union

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex as parse_latex_core
from sympy.parsing.sympy_parser import parse_expr

from ._sympy_guards import evaluable, quotes_free, sympy_globals

# --- evaluation/answer_extraction.py ---


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

def _fix_unicode(string):
    # SIEVAL DIVERGENCE: upstream keeps a `before = string` here and prints a
    # `DEBUG: Unicode conversion:` line at the end when the two differ. Both are
    # dropped -- a library must not write to stdout, and nothing else read the
    # local.

    # square root
    pattern = re.compile(r'√(\([^()]*\)|[A-Za-z0-9]+)')
    string = pattern.sub(lambda m: r'\sqrt{' + m.group(1) + '}', string)

    # cube root
    pattern = re.compile(r'∛(\([^()]*\)|[A-Za-z0-9]+)')
    string = pattern.sub(lambda m: r'\sqrt[3]{' + m.group(1) + '}', string)

    # other fonts of digits
    math_sans_bold_digits = {
        '𝟬': '0', '𝟭': '1', '𝟮': '2', '𝟯': '3', '𝟰': '4',
        '𝟱': '5', '𝟲': '6', '𝟳': '7', '𝟴': '8', '𝟵': '9',

        '𝟢': '0', '𝟣': '1', '𝟤': '2', '𝟥': '3', '𝟦': '4',
        '𝟧': '5', '𝟨': '6', '𝟩': '7', '𝟪': '8', '𝟫': '9',
    }
    for unicode_digit, ascii_digit in math_sans_bold_digits.items():
        string = string.replace(unicode_digit, ascii_digit)

    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3',
        '₄': '4', '₅': '5', '₆': '6', '₇': '7',
        '₈': '8', '₉': '9', 'ₙ': 'n'
    }
    for subchar, digit in subscript_map.items():
        string = string.replace(subchar, f"_{{{digit}}}")

    # other replacements
    replacements = {
        '²': '^{2}',
        '³': '^{3}',
        'ⁿ': '^{n}',
        'π': '\\pi ',
        '∞': '\\infty ',
        '⎣': '\\lfloor ',
        '⎦': '\\rfloor ',
        '–': '-', ## (en dash) U+2013 to (hyphen) U+002D
        '−': '-', ## (minus) U+2212 to (hyphen) U+002D
        '∪': '\\cup ',
        '∩': '\\cap ',
        '·': '\\cdot ',
        '×': '\\times ',
        ' ': ' ',
        '⁄': '/',
        '\xa0': ' ',
        '½': '\\frac{1}{2}',
        '∏': '\\prod ',
        '∑': '\\sum ',
    }

    for unicode_char, latex_equiv in replacements.items():
        string = string.replace(unicode_char, latex_equiv)

    return string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = re.sub(r'(?<!\\)\\ ', '', string) # remove "\\ " but not "\\\\ ".
    string = string.replace("\\,", "")
    string = string.replace("\\:", "")
    string = string.replace("\\;", "")
    string = string.replace("\\quad", "")

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

    ## fix for o1 and o3-mini: these models may use unicode characters for some operators.
    string = _fix_unicode(string)


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
    # string = string.replace("\\cdot", "")

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
    #string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def extract_boxed_answers(text):

    ### sometimes extra spaces are added between `boxed`` and the brackets `{}`.
    text = re.sub(r"boxed\s+\{", "boxed{", text)
    text = re.sub(r"boxed\s+\[", "boxed[", text)
    text = re.sub(r"boxed\s+\(", "boxed(", text)

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

    ### o3-mini may output boxed(), boxed[] instead of boxed{}. This is a hack to fix it.
    left_brackets = ['(', '[', '{']
    right_brackets = [')', ']', '}']
    if len(answers) == 0:
        for piece in text.split('boxed(')[1:]:
            n = 0
            for i in range(len(piece)):
                if piece[i] in left_brackets:
                    n += 1
                elif piece[i] in right_brackets:
                    n -= 1
                    if n < 0:
                        if i + 1 < len(piece) and piece[i + 1] == '%':
                            answers.append(piece[: i + 1])
                        else:
                            answers.append(piece[:i])
                        break
        for piece in text.split('boxed[')[1:]:
            n = 0
            for i in range(len(piece)):
                if piece[i] in left_brackets:
                    n += 1
                elif piece[i] in right_brackets:
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
            #print("warning: fall back to the last number", flush=True)
            #print([pred_str], flush=True)
            pattern = '-?\d*\.?\d+'
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if(len(ans) >= 1):
                ans = ans[-1]
            else:
                ans = ''
            if ans:
                pred.append(ans)
                #print(ans, flush=True)

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

def extract_math_answer(question, reasoning, task):
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if 'separated by commas' in question and all(ch not in ans for ch in '()[]'):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        elif regex.search(r"\\text\{\s*or\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*or\s*\}", "[SEP]", ans).split("[SEP]")])
        elif regex.search(r"\s+and\s+", ans):
            answer.extend([a.strip() for a in regex.sub(r"\s+and\s+", "[SEP]", ans).split("[SEP]")])
        elif regex.search(r"\s+or\s+", ans):
            answer.extend([a.strip() for a in regex.sub(r"\s+or\s+", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())
    return answer

def extract_math_perturb_ground_truth_answer(question, reasoning, task):
    """
        Hack for the labels with multiple answers separated by ' or '
    """
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if 'separated by commas' in question and all(ch not in ans for ch in '()[]'):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        elif regex.search(r" or ", ans):
            answer.extend([a.strip() for a in regex.sub(r" or ", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())
    return answer


# --- evaluation/eval_utils.py ---

MAX_ABS_TOL = 1e-7

def parse_latex(s):
    return parse_latex_core(s, backend='lark') # the backend does not require antlr4-python3-runtime==4.11

def parse_digits(num):
    # format: 234.23 || 23%
    num = regex.sub('\{,\}', '', str(num))
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


def _guarded_parse_expr(s):
    """`parse_expr` with the three guards in `_sympy_guards` applied.

    SIEVAL DIVERGENCE (execution safety). Upstream calls bare `parse_expr(s)`
    on model output, whose default namespace carries `__builtins__`, so a
    prediction of `__import__('os').system(...)` runs. See `_sympy_guards`.
    """
    if not quotes_free(s) or not evaluable(s):
        raise ValueError("refused by sieval: unsafe to hand to sympy")
    return parse_expr(s, global_dict=sympy_globals())


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
                        if isclose(item, prediction, abs_tol=MAX_ABS_TOL):
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


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, _guarded_parse_expr]:
            try:
                return f(s)
            except:
                pass
        # SIEVAL DIVERGENCE (execution safety). Upstream returns `s` here, the
        # raw model output, which reaches `N` below -- and `N` sympifies it with
        # sympy's own default namespace, not the caller's. (Not `simplify(a-b)`:
        # the subtraction raises TypeError first.) That alone defeats the guards
        # above -- with `__import__` resolvable a payload needs no quote -- so an
        # unparseable answer becomes None and the comparison is refused.
        return None
    a = _parse(a)
    b = _parse(b)
    if a is None or b is None:
        return False

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=MAX_ABS_TOL):
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


# --- evaluation/eval_script.py ---

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
                digit_pred = regex.sub('\{,\}', '', str(pred)) # support numbers like 18{,}234
                digit_ans = regex.sub('\{,\}', '', str(ans))
                digit_pred = regex.sub(',', '', digit_pred)
                digit_ans = regex.sub(',', '', digit_ans)
                label = abs(float(digit_pred) - float(digit_ans)) < prec
            except:
                pass
            label = label or (ans and pred == ans) or math_equal(pred, ans)
            return label
    else:
        print(item, flush=True)
        raise NotImplementedError()

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


# --- evaluate.py ---

def extract_ground_truth_answer(problem, ground_truth, dataset_type):
    """
        Extracts the ground truth answer from the problem statement and ground_truth string.
        Args:
            problem (str): The problem statement.
            ground_truth (str): The ground truth answer string.
            dataset_type (str): The type of dataset, either 'perturb' or 'original'.
        Returns:
            ground_truth_answer_extracted (list): The extracted ground truth answer.
    """
    assert dataset_type in ['perturb', 'original'], "dataset_type must be either 'perturb' or 'original'"
    if dataset_type == 'perturb':
        ground_truth_answer = ground_truth
        if isinstance(ground_truth_answer, int) or isinstance(ground_truth_answer, float):
            ground_truth_answer = str(ground_truth_answer)

        # SIEVAL DIVERGENCE: upstream computes `strip_string(ground_truth_answer)`
        # here and prints "Multi-valued ground truth:" when it differs from the
        # extracted head. Both dropped -- no branch read either.
        ground_truth_wrapped = "\\boxed{" + ground_truth_answer + "}"
        ground_truth_answer_extracted = extract_math_perturb_ground_truth_answer(problem, ground_truth_wrapped, task='')

    elif dataset_type == 'original':
        ground_truth_answer_extracted = extract_math_answer(problem, ground_truth, task='')

    return ground_truth_answer_extracted

def extract_predicted_answer(problem, solution_str):
    """
        Extracts the predicted answer from the solution string.
        Args:
            problem (str): The problem statement.
            solution_str (str): The solution string containing the predicted answer.
        Returns:
            unique_prediction (list): The extracted predicted answer, with duplicates removed.
    """
    prediction = extract_math_answer(problem, solution_str, task='')

    unique_prediction = list(dict.fromkeys(prediction)) # remove duplicates but preserve order
    # SIEVAL DIVERGENCE: upstream prints "multi-valued prediction:" here. Dropped.

    return unique_prediction


def answer_check(problem, solution_str, ground_truth, dataset_type):
    """
        Checks if the predicted answer matches the ground truth answer.
        Args:
            problem (str): The problem statement.
            solution_str (str): The solution string containing the predicted answer.
            ground_truth (str): The ground truth answer string.
            dataset_type (str): The type of dataset, either 'perturb' or 'original'.
        Returns:
            is_correct (bool): True if the predicted answer matches the ground truth answer, False otherwise.
    """
    ground_truth_answer_extracted = extract_ground_truth_answer(problem, ground_truth, dataset_type)
    prediction_extracted = extract_predicted_answer(problem, solution_str)

    inp = {
        'answer': ground_truth_answer_extracted,
        'prediction': prediction_extracted
    }

    is_correct = eval_math(inp, prec=MAX_ABS_TOL)
    return is_correct
