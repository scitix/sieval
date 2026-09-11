"""The AIME 2025 tool task and its no-tool sibling as one controlled pair.

Everything this file asserts is about the PAIR: the same prompt template object,
the same answer pattern object, the same dataset rows, the same verifier reached
through the same inherited `feedback`. A copy of any of them would score the same
today and drift tomorrow, at which point the two published numbers stop being a
difference and become two unrelated measurements.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import asyncio
import inspect

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.tasks._math_tool_base as tool_base
import sieval.tasks.aime_2025_0shot_gen as sibling
from sieval.community.simple_evals.common import ANSWER_PATTERN
from sieval.community.simple_evals.math_eval import QUERY_TEMPLATE
from sieval.core.tasks import TaskContext
from sieval.core.tasks.meta import get_task_class, get_task_meta
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    interval_declaration_problems,
)
from sieval.datasets.aime_2025 import AIME2025Dataset
from sieval.tasks._math_tool_base import TOOL_SYSTEM_PROMPT
from sieval.tasks._math_verify import verify_answer
from sieval.tasks.aime_2025_0shot_gen_tool import AIME2025ZeroShotGenToolTask
from tests.unit.tasks.test_math_tool_base import _ScriptedModel, _ScriptedSandbox

PROBLEM = "What is 6 times 7?"
ANSWER = "42"

#: Every counter `_tool_metrics` publishes. Named once so the two tests below
#: assert the same set: one lists it on an empty report, the other compares the
#: empty report's key set against a populated one.
TOOL_COUNTERS = frozenset(
    {
        "n_tool_calls",
        "n_rollouts_using_tool",
        "n_execution_errors",
        "n_tool_timeouts",
        "n_budget_exhausted",
        "n_discarded_tails",
        "n_sandbox_unreachable",
    }
)


def _raw_sample() -> dict[str, str]:
    return {"question": PROBLEM, "answer": ANSWER}


def _dataset(rows: list[dict[str, str]]) -> AIME2025Dataset:
    """One-row dataset in the shape every task in this family is built over."""
    hf = HFDataset.from_list(rows)
    return AIME2025Dataset(_hf_dict=HFDatasetDict({"train": hf, "test": hf}))


def _build_tool_task(*, replies=(), stdouts=(), n: int = 1, **kwargs):
    # The same single-row construction `test_math_pass_at_k_family.py` uses, so
    # the tool task and its no-tool sibling are built over identical rows.
    task = AIME2025ZeroShotGenToolTask(
        _dataset([_raw_sample()]), _ScriptedModel(list(replies)), n=n, **kwargs
    )
    # Swapped after construction: `MathToolTask.__init__` builds a real
    # `SandboxClient`, and the loop only ever reaches the sandbox through
    # `self._sandbox`, so this one attribute is the whole seam.
    task._sandbox = _ScriptedSandbox(list(stdouts))
    return task


def test_the_task_is_registered_under_its_variant_name():
    cls = get_task_class("aime_2025_0shot_gen_tool")
    assert cls is AIME2025ZeroShotGenToolTask


def test_it_reuses_the_no_tool_sibling_s_prompt_and_pattern():
    # The delta is only readable if everything except the affordance is
    # identical. Asserting object identity with the sibling's constants, not
    # equality with a copy -- a copy drifts.
    #
    # The sibling names these as module-scope locals rather than class
    # attributes, so there is no class attribute on it to compare against. What
    # the pair must agree on is the rendered query, which
    # `test_the_queries_match_the_sibling_s_byte_for_byte` pins directly.
    assert AIME2025ZeroShotGenToolTask.query_template is QUERY_TEMPLATE
    assert AIME2025ZeroShotGenToolTask.answer_pattern is ANSWER_PATTERN


def test_it_grades_through_the_sibling_s_verifier():
    # The base's `feedback` is what makes the pair comparable, so the leaf must
    # not define its own. A leaf that overrode `feedback` with a hand-rolled
    # verifier would keep every other test in this file green while the two
    # headline numbers stopped being a difference.
    #
    # Object identity, not a module-name comparison -- two modules naming the
    # same string prove nothing about which function object is called.
    assert AIME2025ZeroShotGenToolTask.feedback is tool_base.MathToolTask.feedback
    assert "feedback" not in AIME2025ZeroShotGenToolTask.__dict__
    assert tool_base.verify_answer is verify_answer
    assert sibling.verify_answer is verify_answer


def test_the_leaf_declares_its_dependencies_at_the_decorator():
    # `deps_group="math"` is what installs the verifier's backend; the sibling
    # declares the same one, and a tool task is not a reason to widen it. Read
    # off the class meta rather than the decorator AST so this asserts what the
    # registry actually published.
    meta = get_task_meta(AIME2025ZeroShotGenToolTask)
    assert meta.name == "aime_2025_0shot_gen_tool"
    assert meta.deps_group == "math"
    assert meta.model_type == "chat"
    assert meta.status == "experimental"
    assert meta.n_shot == 0


def test_max_tool_calls_has_no_default():
    # Deliberately keyword-only and required: the value is a strict resume field
    # and the pilot has not measured what it should be, so nothing may pick one
    # silently. `max_tool_calls` is NOT a shot count -- `check_task_shot_knobs`
    # would reject it if it were spelled like one.
    with pytest.raises(TypeError):
        _build_tool_task()


def test_max_tool_calls_is_keyword_only():
    # Positional use would silently bind to whatever parameter happens to sit
    # there after a signature edit. Asserted through the parameter's KIND rather
    # than by calling positionally: a single positional call only proves the
    # argument did not land in `max_tool_calls`, which a keyword-or-positional
    # parameter would also satisfy at the wrong slot count.
    param = inspect.signature(AIME2025ZeroShotGenToolTask.__init__).parameters[
        "max_tool_calls"
    ]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_a_zero_or_negative_budget_is_rejected():
    # A zero budget would spend a full run to measure a model that was never
    # allowed to call the tool. Rejected loudly rather than scored.
    for budget in (0, -1):
        with pytest.raises(ValueError, match="max_tool_calls"):
            _build_tool_task(max_tool_calls=budget)


def test_report_declares_its_headline_and_denominator():
    task = _build_tool_task(max_tool_calls=2)
    report = asyncio.run(task.report([], []))
    # Asserted on the EMPTY path too: the declarations must be on every return,
    # and an empty-run guard is the branch that most often forgets them.
    assert report[SCORE_KEY_FIELD] == "pass@1"
    assert report[DENOMINATOR_FIELD] == DENOMINATOR_REQUESTED
    assert report["n_tool_calls"] == 0.0


def test_the_tool_counters_are_present_even_when_no_tool_was_called():
    # The counters are the affordance's own evidence: a run where the model never
    # called the tool and a run whose counts were dropped both read as "no tool
    # calls" if the keys are missing, so the zeroes have to be published rather
    # than omitted.
    report = asyncio.run(_build_tool_task(max_tool_calls=2).report([], []))

    assert set(report) >= TOOL_COUNTERS
    assert all(report[key] == 0.0 for key in TOOL_COUNTERS)


def test_a_populated_run_reports_the_same_key_set_as_an_empty_one():
    # Forced by the family's shared report contract rather than by this leaf, but
    # asserted here because this leaf is the one whose counters could be gated
    # behind `if finals:` without any other test noticing.
    task = _build_tool_task(max_tool_calls=2)
    empty = asyncio.run(task.report([], []))
    populated = asyncio.run(
        task.report([TaskContext(sample_id=0, raw_sample=_raw_sample())], [])
    )

    assert set(empty) == set(populated)
    assert interval_declaration_problems(empty) == []
    assert interval_declaration_problems(populated) == []


def test_the_protocol_prompt_reaches_the_model_through_preprocess():
    # The leaf's contract with the base is the two constants above plus the
    # system turn. A leaf that overrode `preprocess` and dropped the system
    # message would leave the model without the protocol it is being scored on.
    task = _build_tool_task(max_tool_calls=2)
    pre = asyncio.run(task.preprocess(_raw_sample(), task.make_context(0)))

    assert pre["prompt"][0] == {"role": "system", "content": TOOL_SYSTEM_PROMPT}
    assert PROBLEM in pre["prompt"][1]["content"]
    assert pre["reference"] == ANSWER


def test_an_uncounted_final_reports_null_counters_not_zeroes():
    """A resumed run keeps the status without the payload; `0.0` would be a lie.

    `postprocess_result is None` is a real, reachable state: `record_each_stage=
    False` plus a resume leaves the final with its judgement and its status but
    no per-rollout extras. Summing nothing over it yields 0.0, which is byte-for-
    byte what a run that genuinely never called the tool yields -- so the counts
    must go null instead, the way every other "could not measure" in this tree
    does. The headline is deliberately NOT withheld: it comes from the
    judgements, which do survive the resume.
    """
    task = _build_tool_task(max_tool_calls=2)
    report = asyncio.run(
        task.report([TaskContext(sample_id=0, raw_sample=_raw_sample())], [])
    )

    assert all(report[key] is None for key in TOOL_COUNTERS)
    # Still a real report: the declarations and the headline survive.
    assert report[SCORE_KEY_FIELD] == "pass@1"
    assert report[DENOMINATOR_FIELD] == DENOMINATOR_REQUESTED
    assert report["score"] is not None
    assert interval_declaration_problems(report) == []


def test_an_empty_run_still_reports_measured_zeroes():
    # The other side of the same distinction: no finals at all means nothing was
    # left uncounted, so the tallies are a genuine 0.0 rather than null.
    report = asyncio.run(_build_tool_task(max_tool_calls=2).report([], []))
    assert all(report[key] == 0.0 for key in TOOL_COUNTERS)


def test_report_counts_sandbox_failures_separately_from_code_errors():
    task = _build_tool_task(max_tool_calls=2)
    # Built through the stage transitions rather than assigned: `TaskContext` is
    # frozen, and `Final` is a status the runner only reaches by passing through
    # the earlier ones -- a context that claimed FINAL without them would not be
    # a shape any run produces.
    ctx = (
        TaskContext(sample_id=0, raw_sample=_raw_sample())
        .to_preprocessed({"prompt": []})
        .to_inferred(None)
        .to_postprocessed(
            {
                "rollouts": [
                    {
                        "index": 0,
                        "prediction": "42",
                        "extra": {
                            "n_tool_calls": 2,
                            "stop_reason": "sandbox_unreachable",
                            # One program that ran and failed, one call that
                            # never ran at all.
                            "n_execution_errors": 1,
                            "n_tool_timeouts": 0,
                            "n_discarded_tails": 0,
                            "n_sandbox_unreachable": 1,
                        },
                    }
                ]
            }
        )
        .to_feedback({"rollouts": [{"index": 0, "correct": True}]})
        .to_final()
    )
    report = asyncio.run(task.report([ctx], []))

    assert report["n_execution_errors"] == 1.0
    assert report["n_sandbox_unreachable"] == 1.0
    assert report["n_tool_calls"] == 2.0
    # Reported, not folded into the score: a rollout that never reached the
    # service is still scored here, and the stop reason is what lets an operator
    # drop it rather than read the pass rate as a measurement of the affordance.
    assert report["score"] is not None


def test_the_sandbox_record_carries_both_the_version_and_full_service():
    # Rediscovered by a reader rather than re-derived: the trajectory has to say
    # the version AND whether it describes every call, because one string cannot
    # say both. Asserted on the serialized dict, since that is what reaches disk.
    task = _build_tool_task(
        max_tool_calls=2, replies=["```python\nprint(6 * 7)\n```", r"\boxed{42}"], n=1
    )
    boxed = asyncio.run(
        task.infer({"prompt": [{"role": "user", "content": "q"}]}, task.make_context(0))
    )
    assert boxed.value.sandbox == {
        "service_version": _ScriptedSandbox.service_version,
        "fully_served": True,
    }


def test_the_queries_match_the_sibling_s_byte_for_byte():
    # The one place the pair could still diverge without any shared object
    # changing: the leaf reads its problem off `question` where most of the
    # family reads `problem`, so a wrong field would KeyError here rather than
    # silently prompt on the gold.
    tool_task = _build_tool_task(max_tool_calls=2)
    raw = _raw_sample()
    tool_pre = asyncio.run(tool_task.preprocess(raw, tool_task.make_context(0)))
    sibling_pre = asyncio.run(
        sibling.AIME2025ZeroShotGenTask(_dataset([raw]), _ScriptedModel([])).preprocess(
            raw, TaskContext(sample_id=0, raw_sample=raw)
        )
    )

    assert tool_pre["prompt"][1] == sibling_pre["prompt"][0]
    assert tool_pre["reference"] == sibling_pre["reference"]
