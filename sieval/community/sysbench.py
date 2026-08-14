# Upstream: https://github.com/PKU-Baichuan-MLSystemLab/SysBench/blob/627ffa8010d00e270426975b33b1fb7a0a635602/utils.py
#           https://github.com/PKU-Baichuan-MLSystemLab/SysBench/blob/627ffa8010d00e270426975b33b1fb7a0a635602/eval_system_bench.py
#
# The judge prompt below is upstream's `get_eval_pattern`, reproduced verbatim in
# Chinese, down to its two internal inconsistencies (the assistant `<role:>` line of a
# history round is emitted on one line where the user's is on three; the current turn's
# two blocks use the three-line form). Upstream declares no license (no LICENSE file at
# the pinned commit, and GitHub reports none), so this is quoted as the specification of
# the measurement rather than redistributed as a work: it IS the published protocol, and
# a translated or tidied prompt would measure something else while still being called
# SysBench.
"""SysBench judge prompt and constraint-satisfaction scoring kernel.

SysBench (arXiv:2408.10943, "Can Large Language Models Follow System Messages?")
scores a multi-turn reply against the turn's checklist of atomic constraints: a
grader LLM answers 是/否 per constraint, and the three published rates nest —

* **CSR** constraint satisfaction rate, the mean per-turn fraction of constraints met;
* **ISR** instruction satisfaction rate, the fraction of turns meeting *every* one;
* **SSR** session satisfaction rate, the fraction of sessions whose turns all do.

:func:`build_judge_prompt` takes the same ``(messages, criteria)`` arguments upstream's
``get_eval_pattern`` takes, and slices them the same way — ``messages[0]`` is the system
prompt, ``messages[1:-2]`` the history, ``messages[-2:]`` the turn being judged. Taking
the cumulative message list rather than pre-split parts is deliberate: the split is part
of what is being ported, and doing it here keeps it checkable against upstream line by
line instead of trusting a caller to reproduce it.

One deviation from upstream @ 627ffa8, in grader-reply handling, which does not change
what is scored:

* **Reply parsing.** Upstream runs ``eval(eval_response[7:-3])`` — Python ``eval``, not
  ``json.loads``, on a fixed slice that assumes the reply is exactly a ```json fence.
  The dict it asks for has bare integer keys (``1: "……"``), which is a valid *Python*
  literal and not valid JSON, so ``eval`` is load-bearing rather than incidental. This
  port reads the verdicts by regex instead: it does not execute text a grader wrote, and
  it does not break when a grader's fence is off by a character. Upstream then asserts
  the verdict keys are exactly the criteria keys and retries up to 10 times (temperature
  0 for the first five, 0.5 after) before letting the session fail; here an unresolved
  constraint scores not-satisfied and is counted in ``n_unparsed``, so a grader wobble
  costs one constraint and stays visible instead of costing a whole session.
  :func:`aggregate_turn` keeps unresolved (``None``) apart from refused (``False``) for
  exactly that reason — otherwise an unreadable reply scores a silent 0.0,
  indistinguishable from a reply that satisfied nothing.

**Upstream's one rule-based check is unreachable, and is therefore not ported.**
``utils.character_count`` opens with ``assert len(chinese_character_count) <= 0`` and
``assert len(character_count) <= 0`` immediately after the two regexes that populate
them, so every input the function's own body would decide instead raises
``AssertionError``, and its only reachable return is the ``-1`` "not my rule"
sentinel. ``run.sh`` invokes plain ``python`` (no ``-O``), so the asserts are live in
the published protocol. Its call site compounds this: both eval scripts call
``character_count(criteria_content, answer)`` against a definition of
``character_count(answer, criteria_content)`` — the arguments are swapped, so the
regexes scan the model's reply rather than the constraint text. Both defects point the
same way, so **every constraint reaches the judge**, which is what upstream's own
numbers measure. Reviving the rule is a *scoring change*: measured on 459 shipped
constraints it claims 13 of them (2.83%), which bounds the CSR difference at 2.83
points — small, but a divergence to declare rather than a repair to slip in. The
unqualified task tracks upstream, defects included; a ``_fixed`` sibling would be
licensed by that measurement, not by the defect's existence.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

#: A single ``<id>: <verdict>`` pair, with or without quotes on either side. Upstream's
#: requested format leaves the key bare; real graders quote it about as often as not.
_YESNO = re.compile(r'["\']?(\d+)["\']?\s*[:：]\s*["\']?(是|否|yes|no|YES|NO|Yes|No)')

#: The ``评判结果`` object. Scoped first so that digits inside the ``评判理由`` prose
#: (a quoted constraint, a count) cannot be read as verdicts.
_VERDICT_BLOCK = re.compile(r"评判结果[\"']?\s*[:：]\s*\{(.+?)\}", re.S)

#: Upstream computes ``all(value in {"是", "否"} ...)`` and discards the result — the
#: statement has no ``assert``, so the verdict vocabulary is never actually enforced.
#: Accepting the English pair too costs nothing and covers a grader that answers in it.
_PASS = frozenset({"是", "yes", "YES", "Yes"})


def build_judge_prompt(
    messages: Sequence[dict],
    criteria: dict[str, Any],
) -> str:
    """Reproduce upstream ``get_eval_pattern``.

    ``messages`` is the cumulative conversation ending at the turn being judged:
    ``[system, user_1, assistant_1, ..., user_t, assistant_t]``, where the assistant
    entries are whatever history the protocol supplies (the model's own replies for the
    headline run, the dataset's for the with-GT ablation). ``criteria`` is that turn's
    checklist, keyed as upstream keys it.
    """
    system_prompt = messages[0]["content"]
    historys = messages[1:-2]
    current_dialogs = messages[-2:]
    criteria_pattern = "\n".join(
        f"{criteria[cid]['criteria_id']}. {criteria[cid]['criteria_content']}"
        f" | {criteria[cid]['criteria_type']}"
        for cid in criteria
    )

    pattern = f"""
# 背景与目标
你现在是一个大模型结果评估专家。以下，你将会面对一个大模型system prompt遵循能力评估任务。
我将会给出你对应的system prompt、历史对话、当前轮次的问题、和当前轮次的回答
需要你准确判断当前轮次的回答是否合格。
为了让你的评判足够准确，我还将会给出评判细则，需要你准确判断并告知评判细则当中每一条约束的遵循情况。

# 对话信息
## system prompt
<system prompt>
{system_prompt}
</system prompt>

## 历史对话轮次

"""

    for index in range(len(historys) // 2):
        user_message = historys[index * 2]
        assistant_message = historys[index * 2 + 1]
        # The asymmetry between the two `<role:>` lines is upstream's, not a typo here:
        # the user's spans three lines and the assistant's is inlined. It reaches the
        # grader either way, so it is reproduced rather than tidied.
        pattern += f"""
<round-{index + 1}>
<role:>
{user_message["role"]}
</role>
<content>
{user_message["content"]}
</content>
<role:>{assistant_message["role"]}</role>
<content>
{assistant_message["content"]}
</content>
</round-{index + 1}>

"""

    pattern += f"""
## 当前待评估的对话轮次
<role:>
{current_dialogs[0]["role"]}
</role>
<content>
{current_dialogs[0]["content"]}
</content>

<role:>
{current_dialogs[1]["role"]}
</role>
<content>
{current_dialogs[1]["content"]}
</content>

# 评判细则
<评判细则>
{criteria_pattern}
</评判细则>
"""
    pattern += """
请你认真阅读上述system prompt设定与历史对话轮次，并严格以评判细则为评判标准，针对评判细则当中的逐条要求，判断当前对话轮次的回答是否遵循。
请以json格式回答，包含两个字段：评判理由、评判结果（评判结果为一个dict，dict的key评判细则的序号，value为对应的评判结果（是/否））
输出格式如下：
'''json
{
  "评判理由": "……",
  "评判结果": {
    1: "……",
    ……
  }
}
'''

"""
    return pattern


def parse_verdict(text: str, criteria_ids: Sequence[str]) -> dict[str, bool | None]:
    """Extract ``{criteria_id: satisfied}`` from the judge reply.

    A constraint the reply did not resolve maps to ``None``, never to ``False``.
    """
    block = text
    m = _VERDICT_BLOCK.search(text)
    if m:
        block = m.group(1)
    found = {str(cid): (v in _PASS) for cid, v in _YESNO.findall(block)}
    return {str(cid): found.get(str(cid)) for cid in criteria_ids}


def aggregate_turn(verdicts: dict[str, bool | None]) -> tuple[float, bool, int, int]:
    """One turn's ``(csr, all_satisfied, n_satisfied, n_unparsed)``.

    An unresolved constraint counts as not satisfied — an unreadable verdict must
    never inflate a score — but is returned separately so grader format drift stays
    distinguishable from a reply that genuinely failed its checklist.
    """
    n = len(verdicts)
    n_satisfied = sum(1 for v in verdicts.values() if v)
    n_unparsed = sum(1 for v in verdicts.values() if v is None)
    return (
        n_satisfied / n if n else 0.0,
        n > 0 and n_satisfied == n,
        n_satisfied,
        n_unparsed,
    )


def aggregate_metrics(
    turns: Iterable[tuple[int, float, bool]],
) -> dict[str, float]:
    """Pool per-turn results into the paper's three nested rates.

    ``turns`` is ``(session_id, csr, all_satisfied)`` per turn. CSR and ISR are means
    over turns; SSR is over sessions, and a session counts only when **every** one of
    its turns satisfied everything — so a session is scored on the turns present in the
    run, which is why a subset selection must keep whole sessions to keep SSR readable.
    """
    turns = list(turns)
    n = len(turns)
    if n == 0:
        return {"csr": 0.0, "isr": 0.0, "ssr": 0.0, "n_turns": 0.0, "n_sessions": 0.0}
    by_session: dict[int, list[bool]] = defaultdict(list)
    for sid, _csr, full in turns:
        by_session[sid].append(full)
    return {
        "csr": sum(t[1] for t in turns) / n,
        "isr": sum(1 for t in turns if t[2]) / n,
        "ssr": sum(1 for fulls in by_session.values() if all(fulls)) / len(by_session),
        "n_turns": float(n),
        "n_sessions": float(len(by_session)),
    }
