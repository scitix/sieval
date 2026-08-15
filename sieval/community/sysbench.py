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

* **CSR** constraint satisfaction rate, the fraction of *constraints* met, pooled over
  every constraint in the run rather than averaged over turns — see
  :func:`aggregate_metrics` for why the two differ and which one Table 2 is;
* **ISR** instruction satisfaction rate, the fraction of turns meeting *every* one;
* **SSR** session *stability* rate — the mean normalised count of consecutive turns,
  from the start of a session, that met every constraint. Not "sessions whose turns
  all do": see :func:`aggregate_metrics` for the formula and for how upstream
  computes it.

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
  constraint scores not-satisfied and is counted in ``n_grader_unparsed``, so a grader wobble
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

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
    # Blocks are read LAST to FIRST, and the first one that resolves anything wins.
    # Both orderings occur and neither is safe alone: the requested output format is
    # quoted in the prompt, so a grader that restates it *before* answering emits a
    # decoy `评判结果` whose placeholder verdicts parse as nothing (anchoring on the
    # first match reads the decoy and scores the whole turn unresolved), while a grader
    # that answers and *then* restates the format emits the decoy last. Taking the last
    # block unconditionally loses that second turn entirely. Skipping a block that
    # yields no verdict handles both, and a trailing restatement of a verdict already
    # given is the same verdict either way.
    for block in reversed(_VERDICT_BLOCK.findall(text) or [text]):
        found = {str(cid): (v in _PASS) for cid, v in _YESNO.findall(block)}
        if any(str(cid) in found for cid in criteria_ids):
            return {str(cid): found.get(str(cid)) for cid in criteria_ids}
    return {str(cid): None for cid in criteria_ids}


def aggregate_turn(verdicts: dict[str, bool | None]) -> tuple[float, bool, int, int]:
    """One turn's ``(csr, all_satisfied, n_satisfied, n_grader_unparsed)``.

    An unresolved constraint counts as not satisfied — an unreadable verdict must
    never inflate a score — but is returned separately so grader format drift stays
    distinguishable from a reply that genuinely failed its checklist.
    """
    n = len(verdicts)
    n_satisfied = sum(1 for v in verdicts.values() if v)
    n_grader_unparsed = sum(1 for v in verdicts.values() if v is None)
    return (
        n_satisfied / n if n else 0.0,
        n > 0 and n_satisfied == n,
        n_satisfied,
        n_grader_unparsed,
    )


def session_prefixes(
    turns: Iterable[tuple[int, int, int, int]],
    session_n_turns: Mapping[int, int] | None = None,
) -> dict[int, tuple[int, int]]:
    """Per session, ``(leading run of fully-satisfied turns, declared turn count)``.

    The one place the prefix is defined. ``ssr`` is the mean of ``prefix / declared``
    over these, and Table 4's per-position ``R_t`` columns are the fraction of them
    with ``prefix >= t`` — two readings of one quantity, which is why they are not
    computed twice. ``turns`` is as :func:`aggregate_metrics` takes it.
    """
    full_by_session: dict[int, dict[int, bool]] = defaultdict(dict)
    for sid, index, n_satisfied, n_criteria in turns:
        full_by_session[sid][index] = n_criteria > 0 and n_satisfied == n_criteria
    declared = dict(session_n_turns or {})
    prefixes: dict[int, tuple[int, int]] = {}
    for sid, by_index in full_by_session.items():
        # `or max(...)`: a 0 or missing declared count would divide by zero or
        # over-credit; the turns actually seen are the only honest fallback.
        n_declared = declared.get(sid) or max(by_index)
        prefix = 0
        # A gap stops the run as surely as a failure does -- `.get` is None there.
        while by_index.get(prefix + 1):
            prefix += 1
        prefixes[sid] = (prefix, n_declared)
    return prefixes


def aggregate_metrics(
    turns: Iterable[tuple[int, int, int, int]],
    session_n_turns: Mapping[int, int] | None = None,
) -> dict[str, float]:
    """Pool per-turn results into the paper's three published rates.

    ``turns`` is ``(session_id, turn_index, n_satisfied, n_criteria)`` per turn, with
    ``turn_index`` 1-based. Raw counts, not a per-turn rate: which of the two averages
    below comes out is decided here, and a rate has already thrown away what decides it.
    ``session_n_turns`` gives each session's **declared** turn count; a session missing
    from it falls back to the highest index seen for it.

    **CSR pools over constraints; it is not the mean of the per-turn rates.** Turns
    carry between 1 and 11 constraints (mean 2.38 on the released set), so
    ``Σ satisfied / Σ constraints`` and ``mean(satisfied/constraints)`` are different
    numbers — up to ~1.9 points apart on published runs, enough to swap two adjacent
    rows of Table 2. The published tables are the former: upstream generates them in
    ``plot/tab6_csr_full.py`` (``res[i] = column[1] / column[0]``, i.e. 遵循数量 over
    约束总量 from the per-constraint-type sheet), wired to the reported CSR by
    ``plot/eval_output.py``. Paper §3.3's ``1/mn`` prefactor describes the ISR/SSR
    outer average over turns; reading it onto CSR as well produces the macro variant,
    which is returned as ``csr_macro`` rather than dropped — it is the finer signal
    when turns are the unit of interest, it is just not what Table 2 reports.

    ISR *is* a mean over turns: upstream averages the per-turn all-satisfied indicator
    ``是否可用`` (``plot/tab3_align.py``), so its unit is the turn, not the constraint.

    **SSR is not a session-level all-or-nothing rate.** §3.3 defines it as
    ``(1/mn) Σ_i Σ_α (⋀_{j≤α} ⋀_k s_ijk)`` — a sum over *every* prefix length α, so it
    is the mean normalised count of consecutive fully-satisfied turns from the start of
    a session. Upstream computes the same quantity in ``utils.analysis_eval_results``:
    ``count_continuous_round`` holds one count per α, and ``cal_continuous_avg``
    telescopes them to ``Σ_α Q_α``, which is ``5 × SSR`` on the released set. Reading
    SSR as "sessions whose turns all do" reports only the α = n term — the smallest of
    them — and lands ~20 points below Table 2 on a run of published shape.

    Dividing by the **declared** count is the other half of that: a walk that ended
    early cannot have satisfied turns it never reached, so an unreached turn shortens
    the prefix instead of dropping out of the denominator. Scoring on the turns merely
    *present* lets an infrastructure failure at turn 3 report a perfect session.
    """
    turns = list(turns)
    n = len(turns)
    if n == 0:
        return {
            "csr": 0.0,
            "csr_macro": 0.0,
            "isr": 0.0,
            "ssr": 0.0,
            "n_turns": 0.0,
            "n_sessions": 0.0,
            "n_criteria": 0.0,
        }
    prefixes = session_prefixes(turns, session_n_turns)
    ssr = sum(prefix / n_declared for prefix, n_declared in prefixes.values())
    n_criteria = sum(t[3] for t in turns)
    n_satisfied = sum(t[2] for t in turns)
    return {
        "csr": n_satisfied / n_criteria if n_criteria else 0.0,
        "csr_macro": sum(t[2] / t[3] for t in turns if t[3]) / n,
        "isr": sum(1 for t in turns if t[3] > 0 and t[2] == t[3]) / n,
        "ssr": ssr / len(prefixes),
        "n_turns": float(n),
        "n_sessions": float(len(prefixes)),
        "n_criteria": float(n_criteria),
    }
