"""
PlatinumBench answer parsing and scoring, vendored from the upstream harness.

Byte-faithful port of ``get_parse_fn`` and ``check_prediction`` from
MadryLab/platinum-benchmarks ``src/utils.py`` at commit
``8fd2f82e63c49ea1cca4266f4dded82b7ddbcb55``:
https://github.com/MadryLab/platinum-benchmarks/blob/8fd2f82e63c49ea1cca4266f4dded82b7ddbcb55/src/utils.py

Upstream code is licensed CC-BY-4.0 (``LICENSE`` in that repo); this module
carries the attribution forward. Only trailing whitespace was stripped — the
commented-out lines, the unused ``prompt`` parameter and the dead-code branch
noted below are all upstream's and are kept so this file diffs 1:1.

Deliberately kept whole: all five parsing strategies ship, not just the ``math``
one the currently-shipped tasks use, so a later subset needs no re-port.

Two upstream behaviours the *caller* has to absorb, because they are expressed
upstream as exceptions swallowed by ``run_benchmark.py``'s bare ``except``:

* ``parse_fn_math`` raises ``AttributeError`` when the model output holds no
  digit at all (``re.search(...).group()`` on ``None``). Upstream records
  ``prediction='parsing error'`` and ``correct=False``.
* ``check_prediction``'s ``prediction != 'Parsing error'`` guard never fires
  against that lowercase ``'parsing error'``, so upstream still reaches
  ``float('parsing error')`` and raises ``ValueError`` — same bare ``except``,
  same ``correct=False``. The guard is dead code upstream. Kept for fidelity;
  callers must not rely on it.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import re


def get_parse_fn(parsing_strategy):
    def parse_fn_math(output):
        """Used for singleop, singleeq, multiarith, gsm8k, and svamp"""
        return re.sub(r"\.0+$", "", (re.search(r'-?[0-9.]*[0-9]', output.replace('*','').replace('#','').lower().split('answer: ')[-1].replace(',', '')).group()))

    def parse_fn_multiple_choice(output):
        """Used for mmlu math and winograd schema challenge"""
        #return output.replace('*', '').lower().split("answer: ")[-1].replace(".", "").strip()[0:1].lower()

        x = output.replace('*', '').lower().split("answer: ")[-1].replace(".", "").strip()

        pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(pattern, x)
        if match:
            return match.group(1)[0:1]
        else:
            return x[0:1]

    def parse_bbh_multiple_choice(output):
        """Used for BBH multiple choice questions, where the answer is in the form (A)"""
        result = output.replace('*', '').replace('#', '').lower().split('answer: ')[-1].replace('.', '').replace('\'', '').replace('\"', '').strip().lower()
        result = re.search(r'\([a-z]\)', result).group(0)
        return result

    def parse_fn_text(output):
        """Used by DROP and hotpotqa, where the answer is a string"""
        return (output.replace("#","").replace("*","").replace("\"", "").replace('\xa0', ' ')
                      .lower().split("answer: ")[-1].split('\n')[0].replace(",", "")
                      .replace(".","").split("}")[0].strip())

    def parse_fn_squad(output):
        """Like rext parsing, but explicitly handles the case when there is text after n/a"""
        output_clean = parse_fn_text(output)
        if output_clean.startswith('n/a '):
            return 'n/a'
        return output_clean

    def create_parse_fn(specific_parsing_fn):

        def parse_fn(output):
            # tex_pattern = r'\\boxed\{([^{}]+)\}|\\boxed\{\\text\{([^}]+)\}\}'
            tex_pattern = r'\\boxed\{(\\text\{)?([^\\{}]+)\}'

            # If answer is on the last line as expected, run as usual
            if "answer:" in output.lower().replace("*", ""):
                # If the answer is wrapped in latex (e.g., \boxed{...}), extract the content
                answer_section = output.lower().split("answer: ")[-1]
                if re.search(tex_pattern, answer_section):
                    match = re.search(tex_pattern, answer_section).group(2)
                    output = "Answer: " + match
            elif re.search(tex_pattern, output):
                # If the answer is not on the last line, try to recover by looking for a box
                output = "Answer: " + re.search(tex_pattern, output).group(2)
            else:
                # Otherwise, just return the last line
                last_line = output.strip("\n").split("\n")[-1].lower()
                output = "Answer: " + last_line
            return specific_parsing_fn(output)

        return parse_fn


    if parsing_strategy == 'math':
        return create_parse_fn(parse_fn_math)
    elif parsing_strategy == 'multiple_choice':
        return create_parse_fn(parse_fn_multiple_choice)
    elif parsing_strategy == 'bbh_multiple_choice':
        return create_parse_fn(parse_bbh_multiple_choice)
    elif parsing_strategy == 'text':
        return create_parse_fn(parse_fn_text)
    elif parsing_strategy == 'squad':
        return create_parse_fn(parse_fn_squad)
    else:
        raise ValueError(f"Invalid parsing strategy: {parsing_strategy}")

def check_prediction(prediction, platinum_target, prompt, dataset_name):
    math_datasets = ['math_eval__multiarith', 'math_eval__singleop', 'math_eval__singleq', 'gsm8k', 'svamp',
                 'multiarith', 'singleop', 'singleq', 'bbh_object_counting']
    if dataset_name in math_datasets and prediction != 'Parsing error':
        correct = float(platinum_target[0]) == float(prediction)
    else:
        correct = prediction in platinum_target
    return correct
