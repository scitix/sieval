# Sources (SciTaRC, JHU-CLSP), pinned to commit
# d96f4e7b0d312cf1bfc2cc16345cd150ba0fa78d:
#   - get_table_text, create_language_prompt, extract_answer_language (verbatim):
#     generate.py
#   - EVAL_PROMPT (verbatim, `.strip()`ed as upstream's loader does):
#     eval_prompt.txt, loaded by evaluate.py TableQAEvaluator.load_prompt
#   - parse_response (upstream's ladder, plus the two deviations below):
#     evaluate.py TableQAEvaluator.parse_response
#   - normalize_text and the equality it feeds (verbatim): exact_match.py
#
# LICENSE: upstream's CODE is MIT, permissive and compatible with this
# repository's Apache-2.0, so this file needs no per-file SPDX marker (contrast
# `gsm_plus.py`, whose share-alike terms do). How that is known, since it is
# weaker than a LICENSE file: at the pinned commit the repo holds nine files,
# none a LICENSE/COPYING/NOTICE, the GitHub API reports `"license": null`, and
# the sole statement is the README's badge pair — "Code License: MIT" and "Data
# License: CC BY-NC 4.0". Only the code half is relied on here; the CC-BY-NC
# data half governs `SciTaRCDataset`, which declares it.
"""SciTaRC prompt + grading assets.

SciTaRC (Wang et al., 2026, arXiv:2603.08910) is a 371-question expert-authored
benchmark for composite question answering over raw LaTeX tables lifted from
arXiv papers. This module vendors upstream's Direct-QA prompt (``plan_mode
none`` x ``exec_mode language``), its ``Answer:`` extractor, its grader prompt,
and the two scorers its published table reports side by side: an LLM grader
whose ternary score is binarised, and a strict exact match.

Only the Direct-QA cell is vendored. The other five cells of upstream's
``plan_mode`` x ``exec_mode`` grid are out of scope for this module: the two
``code`` cells run model-authored Python through a bare ``exec`` (see
``scitarc_0shot_gen``'s docstring), and the ``auto``/``oracle`` plan cells are a
four-model ablation (upstream Table 4) rather than the leaderboard protocol.

Deviations from upstream (@ d96f4e7b):

* :func:`parse_response` returns a ``parsed`` flag alongside the score. Upstream
  defaults an unreadable reply to ``0.0``, which is indistinguishable from a
  grader that read the answer and scored it wrong; the flag separates them so
  the caller can count ``n_grader_unparsed``. It is purely additive — every
  score this returns is the score upstream computes.
* It returns ``(score, reasoning, parsed)`` rather than upstream's dict, whose
  third key was the truncated reply. The caller persists the grader's whole
  ``ModelOutput`` instead, which subsumes it.
* :func:`parse_response` reads only what follows the LAST ``[Evaluation Start]``.
  Upstream's parser may assume the prompt is absent from the text it reads,
  because its completion endpoint returns only the continuation. A chat grader
  handed the same prompt can quote it back, and the template's format example,
  ``{"reasoning": "Brief explanation", "score": 1.0}``, sits before the
  ``[Evaluation End]`` truncation point and matches the very pattern Method 1
  looks for — which takes the FIRST brace-run, so an echo outranks a genuine
  grade that follows it. That is the one reply shape misreading *high*, so it is
  closed at the boundary rather than by tightening the score patterns, which
  stay upstream's byte for byte because they are what the published column was
  computed with. A no-op on every reply upstream could produce (none carries the
  marker). Replayed over a stored 371-row run it proves reachable but harmless:
  4 of 369 replies restated ``[Evaluation Start]`` before their verdict, so the
  cut keeps it, and **no verdict changed** — though a grader inclined to restate
  the tag is one nudge from restating the example with it.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import re
from collections.abc import Sequence

#: ``eval_prompt.txt``, as ``TableQAEvaluator.load_prompt`` hands it to
#: ``str.format`` — upstream reads the file and ``.strip()``s it, so the
#: trailing newline is not part of the rendered prompt. The doubled braces are
#: upstream's own escaping of the JSON example against that ``format`` call.
EVAL_PROMPT = """You are evaluating a table QA system. Compare the predicted answer against the ground truth answer.

SCORING RULES (follow exactly):
- 1.0: CORRECT - Prediction contains the same core factual information as ground truth
- 0.5: PARTIALLY CORRECT - Prediction has some correct information but missing key parts
- 0.0: INCORRECT - Prediction is factually wrong or completely unrelated

CRITICAL REQUIREMENTS:
- ALL components in ground truth MUST be mentioned in prediction
- If ground truth has multiple parts, prediction must include ALL parts
- Missing ANY component = maximum score 0.5
- Consider what the QUESTION is asking for - additional mentions can be explanatory context (OK) or extra answer components (NOT OK)
- Adding extra components not in ground truth = maximum score 0.5
- Numbers must be nearly identical - only final digit can differ by ±1 (17.25% vs 17.26% = OK, 17.25% vs 16.75% = WRONG, 4.2 vs 3.8 = WRONG)

IMPORTANT GUIDELINES:
- IGNORE spelling errors, typos, and minor word variations
- IGNORE extra formatting, phrases, or explanations if core facts are correct
- IGNORE word order differences and minor rephrasing
- FOCUS ONLY on whether the key factual content matches
- Model names must match exactly

EXAMPLES:
- Ground: "Japanese" | Prediction: "None. HTML2Text performs best in Japanese" → 1.0 (Japanese is mentioned)
- Ground: "Humanitarian" | Prediction: "The humanitanian domain" → 1.0 (same meaning despite typo)
- Ground: "Adam" | Prediction: "Adam optimizer with 0.001 learning rate" → 1.0 (explanatory context OK)
- Ground: "ResNet-50" | Prediction: "ResNet-50 and DenseNet-121" → 0.5 (extra answer component)
- Ground: "French, German" | Prediction: "French shows improvement" → 0.5 (missing German component)
- Ground: "4.2" | Prediction: "approximately 3.8" → 0.0 (wrong number)

CRITICAL: Return ONLY this exact JSON format with start/end tags:
[Evaluation Start]
{{"reasoning": "Brief explanation", "score": 1.0}}
[Evaluation End]

Now evaluate:

Question: {question}
Ground Truth: {ground_truth}
Prediction: {prediction}

[Evaluation Start]"""

#: The score upstream's summary counts as correct. Its ternary scale also awards
#: 0.5 for a partially correct answer, which `accuracy` does not credit at all.
CORRECT_SCORE = 1.0

# `evaluate.py` parse_response, Method 1: the first brace-delimited run that
# mentions a "score" key. `[^}]*` keeps it to a single flat object.
_JSON_SCORE_RE = r'\{[^}]*"score"[^}]*\}'

# `evaluate.py` parse_response, Method 2 — tried in order against the reply and
# then its lowercased copy, first parse wins.
_SCORE_PATTERNS = [
    r'"score":\s*([\d.]+)',
    r'score["\']?\s*[:=]\s*([\d.]+)',
    r"\*\*final\s+score\*\*:?\s*([\d.]+)",
    r"\*\*score\*\*:?\s*([\d.]+)",
    r"\*\*scoring\*\*:?\s*([\d.]+)",
    r"final\s+score:?\s*([\d.]+)",
    r"score:?\s*([\d.]+)",
    r"scoring:?\s*([\d.]+)",
    r"FINAL\s+SCORE:?\s*([\d.]+)",
    r"SCORE:?\s*([\d.]+)",
    r"SCORING:?\s*([\d.]+)",
    r"my\s+score:?\s*([\d.]+)",
    r"the\s+score:?\s*([\d.]+)",
    r"overall\s+score:?\s*([\d.]+)",
]

# `evaluate.py` parse_response, Method 3.
_REASONING_PATTERNS = [
    r'"reasoning":\s*"([^"]*)"',
    r'"reasoning":\s*\'([^\']*)\'',
    r'reasoning["\']?\s*[:=]\s*["\']([^"\']*)["\']',
    r"\*\*reasoning\*\*:?\s*([^\n\*]+)",
    r"reasoning:?\s*([^\n]+)",
]

#: What upstream stores when Method 3 finds nothing.
NO_REASONING = "No reasoning extracted from response"


def get_table_text(relevant_tables: Sequence[Sequence[str]]) -> str:
    """Flatten the LaTeX table sources into the prompt's table block.

    Each table arrives as a list of source lines that already carry their own
    newlines, so the lines are joined with nothing and the tables with a blank
    line.
    """
    return "\n\n".join("".join(table) for table in relevant_tables)


def create_language_prompt(
    question: str, relevant_tables: Sequence[Sequence[str]]
) -> str:
    """Upstream's ``plan_mode=none`` x ``exec_mode=language`` prompt.

    The persona, the table block and the ``Answer:`` instruction are one string
    upstream, so they stay one string here — sieval sends it as a single user
    turn rather than splitting a system message out of it.
    """
    return f"""You are a helpful science assistant who answers questions about information in tables.

Here is the relevant tabular data:

{get_table_text(relevant_tables)}

You may think through the question step by step. Your final response should be "Answer:" followed by the answer.

Question: {question}
"""


def extract_answer_language(raw_response: str) -> str:
    """Take everything after the first ``Answer:``, or the whole reply.

    ``DOTALL`` makes ``(.*)$`` run to the end of the reply, so anything the
    model writes after its answer line is part of the answer — including a
    second ``Answer:``. That is upstream's behaviour and it is what the
    published exact-match column was computed against, so it is preserved
    rather than tightened.
    """
    m = re.search(
        r"(?:^|\n)\s*(?:final\s*)?answer\s*:\s*(.*)$",
        raw_response,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else raw_response.strip()


def parse_response(response: str) -> tuple[float, str, bool]:
    """Extract ``(score, reasoning, parsed)`` from a grader reply.

    Upstream's three-method ladder, in order: a flat JSON object carrying a
    ``score`` key; failing that, a list of loose ``score: <number>`` patterns
    tried against the reply and then its lowercased copy; failing that, the
    score stays ``0.0``. ``reasoning`` is read from the JSON when that path
    won, else from its own pattern list, else :data:`NO_REASONING`.

    ``parsed`` is sieval's addition — ``False`` means no method found a number,
    so the ``0.0`` is a read failure rather than a verdict. The caller counts
    those as ``n_grader_unparsed`` and still scores them incorrect, which is
    what upstream's undifferentiated ``0.0`` does.

    The reply is first cut to the text after the last ``[Evaluation Start]``,
    which is where upstream's completion endpoint would have begun; see the
    module docstring for why a chat grader makes that necessary.
    """
    raw_score = 0.0
    reasoning = ""
    parsed = False

    response = response.strip()
    # Cut to the continuation upstream's completion endpoint would have returned,
    # so an echoed template cannot be read as a verdict (module docstring).
    if "[Evaluation Start]" in response:
        response = response.rsplit("[Evaluation Start]", 1)[1].strip()
    if "[Evaluation End]" in response:
        response = response.split("[Evaluation End]")[0].strip()

    # Method 1: JSON.
    try:
        json_match = re.search(_JSON_SCORE_RE, response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "score" in data:
                return (
                    float(data["score"]),
                    str(data.get("reasoning", "")).strip(),
                    True,
                )
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        # TypeError is sieval's: `json.loads('{"score": null}')` parses, and
        # `float(None)` raises a type error upstream's tuple does not catch.
        # Upstream would crash the whole batch; here the reply reads as
        # unparsed, which is what it is.
        pass

    # Method 2: loose score patterns.
    for text_to_search in [response, response.lower()]:
        for pattern in _SCORE_PATTERNS:
            score_match = re.search(pattern, text_to_search, re.IGNORECASE)
            if score_match:
                try:
                    raw_score = float(score_match.group(1))
                    parsed = True
                    break
                except ValueError:
                    continue
        else:
            continue
        break

    # Method 3: reasoning, when the JSON path did not supply one.
    if not reasoning:
        for pattern in _REASONING_PATTERNS:
            reasoning_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                break

    if not reasoning:
        reasoning = NO_REASONING

    return raw_score, reasoning, parsed


def normalize_text(s: str | None) -> str:
    """``exact_match.py``'s normalisation: stringify and strip, nothing else.

    Upstream leaves the ``.lower()`` commented out, so exact match is
    case-sensitive — which is most of why its published column runs so far
    below the grader's.
    """
    if s is None:
        return ""
    return str(s).strip()


def exact_match(prediction: str | None, ground_truth: str | None) -> bool:
    """Strict equality of the two normalised strings."""
    return normalize_text(prediction) == normalize_text(ground_truth)
