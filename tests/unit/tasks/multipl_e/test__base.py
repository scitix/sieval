"""Unit tests for the shared MultiPL-E protocol, grading and reporting.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import dataclasses

import pytest

from sieval.core.models import ModelOutput
from sieval.core.tasks import TaskContext, build_prediction_record
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.tasks.multipl_e._base import (
    DEFAULT_GRADE_WALL,
    EVALUATOR_LANG_BY_TAG,
    GRADE_WALL_HEADROOM,
    _postprocess_chat_completion,
    stop_at_stop_token,
)
from sieval.tasks.multipl_e.multipl_e_humaneval_0shot_base_gen import (
    MultiPLEHumanEvalZeroShotBaseGenTask,
)
from sieval.tasks.multipl_e.multipl_e_humaneval_0shot_gen import (
    MultiPLEHumanEvalZeroShotGenTask,
)
from tests.unit.tasks.multipl_e.conftest import (
    CapturingChatModel,
    CapturingGenModel,
    EvaluatorDouble,
    UnreachableEvaluator,
    dataset_for,
    judgement,
    row,
)


def _base_task(rows=None, **kwargs):
    rows = rows or [row()]
    return MultiPLEHumanEvalZeroShotBaseGenTask(
        dataset_for(rows), CapturingGenModel(), **kwargs
    )


def _chat_task(rows=None, **kwargs):
    rows = rows or [row()]
    return MultiPLEHumanEvalZeroShotGenTask(
        dataset_for(rows), CapturingChatModel(), **kwargs
    )


async def _swap(task, evaluator):
    """Replace the real http client, as the LiveCodeBench tests do."""
    await task._http_client.aclose()
    task._http_client = evaluator
    return evaluator


# ---- language vocabulary -------------------------------------------------


def test_every_upstream_tag_maps_to_an_evaluator_language():
    from sieval.datasets._multipl_e import HUMANEVAL_LANGUAGES, MBPP_LANGUAGES

    # An unmapped tag would reach setup() as a confusing internal error rather
    # than as "the evaluator cannot run this".
    assert set(HUMANEVAL_LANGUAGES) <= set(EVALUATOR_LANG_BY_TAG)
    assert set(MBPP_LANGUAGES) <= set(EVALUATOR_LANG_BY_TAG)


def test_mapping_uses_the_evaluators_existing_spelling():
    # The evaluator names languages in English (`javascript`, not `js`), which
    # is the side that has to match for dispatch to work.
    assert EVALUATOR_LANG_BY_TAG["js"] == "javascript"
    assert EVALUATOR_LANG_BY_TAG["ts"] == "typescript"
    assert EVALUATOR_LANG_BY_TAG["sh"] == "bash"
    assert EVALUATOR_LANG_BY_TAG["pl"] == "perl"
    assert EVALUATOR_LANG_BY_TAG["jl"] == "julia"
    assert EVALUATOR_LANG_BY_TAG["ml"] == "ocaml"
    assert EVALUATOR_LANG_BY_TAG["rkt"] == "racket"
    assert EVALUATOR_LANG_BY_TAG["adb"] == "ada"


# ---- stop-token truncation ----------------------------------------------


def test_stop_at_stop_token_takes_the_earliest_match():
    assert stop_at_stop_token("a\n}b\nfunction c", ["\nfunction ", "\n}"]) == "a"


def test_stop_at_stop_token_leaves_untouched_text_alone():
    assert stop_at_stop_token("  return 1;", ["\n}"]) == "  return 1;"


def test_stop_at_stop_token_can_empty_a_completion():
    # Upstream's warning made concrete: scanning text that begins with a stop
    # token truncates to nothing, which is why the prompt is never scanned.
    assert stop_at_stop_token("\n}", ["\n}"]) == ""


# ---- chat postprocessing -------------------------------------------------


def test_chat_postprocess_strips_a_leading_fence():
    reply = "```\nlong f() {\n  return 1;\n}\n```"
    assert _postprocess_chat_completion(reply, "long f() {\n", []) == (
        "long f() {\n  return 1;\n}"
    )


def test_chat_postprocess_ignores_a_fence_that_is_not_leading():
    # Upstream only looks at the start of the reply; a mid-reply fence is left
    # in place. Kept deliberately -- it is upstream's rule, not an oversight.
    reply = "here:\n```\nlong f() { }\n```"
    assert "```" in _postprocess_chat_completion(reply, "long f() {\n", [])


def test_chat_postprocess_only_scans_past_the_prompt_length():
    # The repeated prefix carries this language's stop token. Scanning from
    # zero would truncate the whole program away; upstream's character split is
    # what prevents that.
    prompt = "long f() {\n"
    reply = prompt + "  return 1;\n}\nextra"
    out = _postprocess_chat_completion(reply, prompt, ["\n}"])
    assert out == prompt + "  return 1;"
    assert out.startswith(prompt)


# ---- program assembly ----------------------------------------------------


@pytest.mark.anyio
async def test_completion_protocol_prepends_the_dataset_prompt():
    task = _base_task()
    evaluator = await _swap(task, EvaluatorDouble())
    try:
        raw = row()
        await task.feedback(
            build_prediction_record(["  return s.length();"]),
            TaskContext(sample_id=0, raw_sample=raw),
        )
        program = evaluator.bodies[0]["code"]
        # Upstream's assembly, separator included (evaluation/src/main.py).
        assert program == raw["prompt"] + "  return s.length();" + "\n" + raw["tests"]
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_chat_protocol_discards_the_dataset_prompt():
    # Upstream's blank-prompt rule: the MODEL's copy of the prefix compiles.
    # Prepending the dataset prompt would define the function twice, which does
    # not compile in most of these languages -- a whole-benchmark zero.
    task = _chat_task()
    evaluator = await _swap(task, EvaluatorDouble())
    try:
        raw = row()
        whole_program = raw["prompt"] + "  return s.length();"
        await task.feedback(
            build_prediction_record([whole_program]),
            TaskContext(sample_id=0, raw_sample=raw),
        )
        program = evaluator.bodies[0]["code"]
        assert program == whole_program + "\n" + raw["tests"]
        assert program.count(raw["prompt"]) == 1
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_unextractable_prediction_is_still_run():
    # A blank completion is graded as a real build failure rather than skipped,
    # so the verdict comes from the evaluator instead of from a missing record.
    task = _base_task()
    evaluator = await _swap(task, EvaluatorDouble())
    try:
        raw = row()
        await task.feedback(
            build_prediction_record([None]),
            TaskContext(sample_id=0, raw_sample=raw),
        )
        assert evaluator.bodies[0]["code"] == raw["prompt"] + "\n" + raw["tests"]
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_feedback_sends_the_language_and_suite_alias():
    task = _base_task([row("sh")])
    evaluator = await _swap(task, EvaluatorDouble())
    try:
        await task.feedback(
            build_prediction_record(["x"]),
            TaskContext(sample_id=0, raw_sample=row("sh")),
        )
        body = evaluator.bodies[0]
        assert body["lang"] == "bash"
        assert body["source"] == "human-eval"
        # No `timeout` key: the per-language default on the evaluator is the
        # budget, and one number here would charge c++ bash's wall.
        assert "timeout" not in body
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_explicit_timeout_is_forwarded_with_headroom():
    task = _base_task(timeout=7.0)
    evaluator = await _swap(task, EvaluatorDouble())
    try:
        await task.feedback(
            build_prediction_record(["x"]),
            TaskContext(sample_id=0, raw_sample=row()),
        )
        assert evaluator.bodies[0]["timeout"] == 7.0
        # Above the server's own budget, so a slow compile reads as a grade
        # rather than as a network timeout.
        assert evaluator.deadlines[0] == pytest.approx(7.0 + GRADE_WALL_HEADROOM)
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_grade_request_is_never_unbounded():
    # Grading is synchronous on one shared event loop, so an unbounded wait
    # stalls the session rather than the sample. Covers the default path, where
    # the budget is the evaluator's own per-language one and this side does not
    # know it.
    task = _base_task()
    evaluator = await _swap(task, EvaluatorDouble())
    try:
        await task.feedback(
            build_prediction_record(["x"]),
            TaskContext(sample_id=0, raw_sample=row()),
        )
        wall = evaluator.deadlines[0]
        assert wall is not None
        assert wall == pytest.approx(DEFAULT_GRADE_WALL)
        # Has to clear the evaluator's largest row: c++'s 15s run plus its
        # separate 60s compile.
        assert wall > 75.0
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_judgement_persists_the_language():
    # So a per-language rate is recomputable from the shard files alone.
    task = _base_task([row("pl")])
    await _swap(task, EvaluatorDouble())
    try:
        _, record = await task.feedback(
            build_prediction_record(["x"]),
            TaskContext(sample_id=0, raw_sample=row("pl")),
        )
        assert record["extra"]["language"] == "pl"
        assert record["rollouts"][0]["extra"]["language"] == "pl"
    finally:
        await task.shutdown()


# ---- infer wiring --------------------------------------------------------


@pytest.mark.anyio
async def test_completion_infer_sends_the_rows_own_stop_tokens():
    # Stop tokens are per language, so they must come off the sample rather
    # than from a task constant.
    task = _base_task(n=3)
    try:
        raw = row("js", stop_tokens=("\nfunction ", "\n//"))
        ctx = TaskContext(sample_id=0, raw_sample=raw)
        pre = await task.preprocess(raw, ctx)
        await task.infer(pre, ctx)
        req = task.model.last_req
        assert req.sampling.stop == ("\nfunction ", "\n//")
        assert req.sampling.n == 3
        assert pre["prompt"] == raw["prompt"]
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_chat_infer_sends_no_stop_tokens():
    # The reply is asked to repeat the prefix, whose text contains this
    # language's stop tokens -- passing them would cut it off mid-prefix.
    task = _chat_task()
    try:
        raw = row()
        ctx = TaskContext(sample_id=0, raw_sample=raw)
        pre = await task.preprocess(raw, ctx)
        await task.infer(pre, ctx)
        assert task.model.last_req.sampling.stop is None
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_chat_prompt_carries_upstreams_instruction():
    task = _chat_task()
    try:
        raw = row()
        pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
        text = "\n".join(m["content"] for m in pre["prompt"])
        # The "repeat it exactly" instruction is what makes the blank-prompt
        # grading path work at all.
        assert "do not alter the prefix but repeat it exactly" in text
        assert raw["prompt"] in text
        assert "cpp" in text
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_completion_postprocess_truncates_at_the_stop_token():
    task = _base_task()
    try:
        raw = row()
        inferred = ModelOutput(
            model=task.model.meta(), texts=["  return 1;\n}\nlong g() {}", "   "]
        )
        post = await task.postprocess(
            inferred,
            TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred),
        )
        predictions = [r.get("prediction") for r in post["rollouts"]]
        assert predictions[0] == "  return 1;"
        # Whitespace-only survives truncation as itself, not as None: it is
        # what the model produced, and the evaluator gives it a real verdict.
        assert predictions[1] == "   "
    finally:
        await task.shutdown()


# ---- setup probe ---------------------------------------------------------


@pytest.mark.anyio
async def test_setup_passes_when_every_language_is_advertised():
    task = _base_task([row("cpp"), row("sh")])
    await _swap(task, EvaluatorDouble(languages=("cpp", "bash")))
    try:
        await task.setup()
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_setup_refuses_a_language_the_evaluator_cannot_run():
    # The failure this exists to prevent: without it the run completes, costs a
    # full generation budget, and reports pass@1 = 0 with no errors.
    task = _base_task([row("cpp"), row("rkt")])
    await _swap(task, EvaluatorDouble(languages=("cpp", "bash")))
    try:
        with pytest.raises(ValueError) as excinfo:
            await task.setup()
        message = str(excinfo.value)
        assert "rkt" in message
        assert "racket" in message  # names the toolchain to deploy
        assert "languages" in message  # names the way to narrow the run
        assert "cpp" in message  # names what does work
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_setup_refuses_an_evaluator_without_the_capability_endpoint():
    task = _base_task()
    await _swap(task, EvaluatorDouble(languages=None))
    try:
        with pytest.raises(RuntimeError, match="languages"):
            await task.setup()
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_setup_refuses_an_unreachable_evaluator():
    task = _base_task()
    await _swap(task, UnreachableEvaluator())
    try:
        with pytest.raises(RuntimeError, match="cannot reach"):
            await task.setup()
    finally:
        await task.shutdown()


# ---- report --------------------------------------------------------------


def _finals(*specs, rows):
    out = []
    for i, (language, verdicts) in enumerate(specs):
        out.append(
            dataclasses.replace(
                TaskContext(sample_id=i, raw_sample=rows[i]),
                feedback_result=judgement(language, *verdicts),
            )
        )
    return out


@pytest.mark.anyio
async def test_report_publishes_per_language_rates_and_a_macro():
    rows = [row("cpp"), row("sh"), row("js")]
    task = _base_task(rows, n=2)
    try:
        report = await task.report(
            _finals(
                ("cpp", [(True, ""), (True, "")]),
                ("sh", [(True, ""), (False, "failed [exit 1]: x")]),
                ("js", [(False, "failed [exit 1]: x")] * 2),
                rows=rows,
            ),
            [],
        )
        assert report["pass@1_cpp"] == pytest.approx(100.0)
        assert report["pass@1_sh"] == pytest.approx(50.0)
        assert report["pass@1_js"] == pytest.approx(0.0)
        assert report["n_languages"] == 3
        # Unweighted over languages; the headline pools size-weighted. Equal
        # here only because each language has one problem -- asserted anyway so
        # the two keys cannot silently become one number.
        assert report["pass@1_macro"] == pytest.approx(50.0)
        assert report["pass@1"] == pytest.approx(50.0)
        assert report["score"] == report["pass@1"]
        assert report["score_key"] == "pass@1"
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_report_headline_pools_size_weighted_while_macro_does_not():
    # Two problems in one language, one in another: the two aggregations must
    # disagree, which is the whole reason both are published.
    rows = [row("cpp", name="a"), row("cpp", name="b"), row("sh")]
    task = _base_task(rows)
    try:
        report = await task.report(
            _finals(
                ("cpp", [(True, "")]),
                ("cpp", [(True, "")]),
                ("sh", [(False, "failed [exit 1]: x")]),
                rows=rows,
            ),
            [],
        )
        assert report["pass@1"] == pytest.approx(200 / 3)  # 2 of 3 problems
        assert report["pass@1_macro"] == pytest.approx(50.0)  # 100 and 0
        assert report["pass@1"] != report["pass@1_macro"]
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_report_charges_a_pipeline_fail_as_wrong_and_places_its_language():
    # DENOMINATOR_REQUESTED. The failed sample has no judgement to read a
    # language from, so this also covers reading it off the live dataset.
    rows = [row("cpp"), row("sh")]
    task = _base_task(rows)
    try:
        report = await task.report(
            _finals(("cpp", [(True, "")]), rows=rows),
            [TaskContext(sample_id=1, raw_sample=rows[1])],
        )
        assert report["fails"] == 1
        assert report["pass@1"] == pytest.approx(50.0)
        assert report["denominator_policy"] == "requested"
        assert report["pass@1_sh"] == pytest.approx(0.0)
        assert report["n_problems_sh"] == 1
        assert report["n_languages"] == 2
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_report_buckets_failures_by_stage():
    rows = [row("cpp"), row("cpp", name="b"), row("cpp", name="c")]
    task = _base_task(rows)
    try:
        report = await task.report(
            _finals(
                ("cpp", [(False, "failed: timeout")]),
                ("cpp", [(False, "failed [build exit 1]: syntax")]),
                ("cpp", [(False, "failed [exit 1]: assertion")]),
                rows=rows,
            ),
            [],
        )
        assert report["timeouts"] == 1
        assert report["n_build_errors"] == 1
        assert report["n_execution_errors"] == 1
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_report_tolerates_a_null_message():
    # A null msg is absent on disk; bucketing must not crash on it.
    rows = [row("cpp")]
    task = _base_task(rows)
    try:
        report = await task.report(_finals(("cpp", [(False, None)]), rows=rows), [])
        assert report["n_execution_errors"] == 1
    finally:
        await task.shutdown()


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.anyio
async def test_report_interval_declarations_are_consistent(n):
    # What the runner checks at report-write time. Guards the `|` merge of
    # three fragments: a plain merge can replace `ci95_units` wholesale and
    # leave intervals with no unit.
    rows = [row("cpp"), row("sh"), row("js"), row("pl")]
    task = _base_task(rows, n=n)
    try:
        report = await task.report(
            _finals(
                ("cpp", [(True, "")] * n),
                ("sh", [(True, "")] * n),
                ("js", [(False, "failed [exit 1]: x")] * n),
                ("pl", [(False, "failed [exit 1]: x")] * n),
                rows=rows,
            ),
            [],
        )
        assert interval_declaration_problems(report) == []
        lo, hi = report["score_ci95"]
        assert lo < report["score"] < hi
        assert report["n_problems"] == 4
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_empty_report_declares_without_faking_a_population():
    task = _base_task()
    try:
        report = await task.report([], [])
        assert report["score"] == 0.0
        assert report["score_key"] == "pass@1"
        assert report["denominator_policy"] == "requested"
        # No zeroed population, which would read as a measured one.
        assert "n_problems" not in report
        assert "score_ci95" not in report
        assert interval_declaration_problems(report) == []
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_k_greater_than_n_is_refused():
    with pytest.raises(ValueError, match="pass@2"):
        _base_task(k=2, n=1)
