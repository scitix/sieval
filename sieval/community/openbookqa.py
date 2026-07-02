"""OpenBookQA prompt template and answer extraction, adapted from OpenCompass.

Vendored from open-compass/opencompass @ 5767b748:
  - configs/datasets/obqa/obqa_gen_9069e4.py — prompt template (``main`` variant)
  - utils/text_postprocessors.py — ``first_option_postprocess``

``first_option_postprocess`` is reproduced verbatim, including upstream's
missing-comma quirk between the ``故选`` and ``只有选项`` patterns (the two
adjacent f-strings implicitly concatenate into one pattern). The only
mechanical change is the ``r`` string prefix to silence the W605 invalid-escape
warning; the compiled regexes are byte-for-byte identical to upstream.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

import re

# `main` variant of _template in obqa_gen_9069e4.py. The `additional`/fact1
# variant ("Given the fact: {fact1}\n...") is intentionally not vendored.
OBQA_PROMPT_TEMPLATE = (
    "Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
)

OBQA_OPTIONS = "ABCD"


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    patterns = [
        rf'答案是?\s*([{options}])',
        rf'答案是?\s*：\s*([{options}])',
        rf'答案是?\s*:\s*([{options}])',
        rf'答案选项应?该?是\s*([{options}])',
        rf'答案选项应?该?为\s*([{options}])',
        rf'答案应该?是\s*([{options}])',
        rf'答案应该?选\s*([{options}])',
        rf'答案选项为?\s*：\s*([{options}])',
        rf'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'答案选项是?\s*:\s*([{options}])',
        rf'答案为\s*([{options}])',
        rf'答案选\s*([{options}])',
        rf'选择?\s*([{options}])',
        rf'故选?\s*([{options}])'
        rf'只有选?项?\s?([{options}])\s?是?对',
        rf'只有选?项?\s?([{options}])\s?是?错',
        rf'只有选?项?\s?([{options}])\s?不?正确',
        rf'只有选?项?\s?([{options}])\s?错误',
        rf'说法不?对选?项?的?是\s?([{options}])',
        rf'说法不?正确选?项?的?是\s?([{options}])',
        rf'说法错误选?项?的?是\s?([{options}])',
        rf'([{options}])\s?是正确的',
        rf'([{options}])\s?是正确答案',
        rf'选项\s?([{options}])\s?正确',
        rf'所以答\s?([{options}])',
        rf'所以\s?([{options}][.。$]?$)',
        rf'所有\s?([{options}][.。$]?$)',
        rf'[\s，：:,]([{options}])[。，,\.]?$',
        rf'[\s，,：:][故即]([{options}])[。\.]?$',
        rf'[\s，,：:]因此([{options}])[。\.]?$',
        rf'[是为。]\s?([{options}])[。\.]?$',
        rf'因此\s?([{options}])[。\.]?$',
        rf'显然\s?([{options}])[。\.]?$',
        r'答案是\s?(\S+)(?:。|$)',
        r'答案应该是\s?(\S+)(?:。|$)',
        r'答案为\s?(\S+)(?:。|$)',
        rf'(?i)ANSWER\s*:\s*([{options}])',
        rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        rf'^选项\s?([{options}])',
        rf'^([{options}])\s?选?项',
        rf'(\s|^)[{options}][\s。，,：:\.$]',
        r'1.\s?(.*?)$',
        rf'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        rf'([{options}]):',
        rf'([{options}])',
    ]

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''
