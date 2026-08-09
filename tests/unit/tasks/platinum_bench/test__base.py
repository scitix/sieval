"""
Unit tests for the shared PlatinumBench math task base.

The five leaf tasks add nothing but a ``subset`` string, so every behavioural
assertion lives here and is parametrized over the leaves where the leaf identity
matters. Per-leaf metadata is pinned in the leaf test modules.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset

from sieval.community.platinum_bench import check_prediction
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.tasks.platinum_bench._base import (
    MATH_PARSING_STRATEGY,
    PLATINUM_REFERENCE_NOTES,
    PLATINUM_UPSTREAM_URL,
)
from sieval.tasks.platinum_bench.platinum_gsm8k_0shot_gen import (
    PlatinumGSM8KZeroShotGenTask,
)
from sieval.tasks.platinum_bench.platinum_multiarith_0shot_gen import (
    PlatinumMultiArithZeroShotGenTask,
)
from sieval.tasks.platinum_bench.platinum_singleop_0shot_gen import (
    PlatinumSingleOpZeroShotGenTask,
)
from sieval.tasks.platinum_bench.platinum_singleq_0shot_gen import (
    PlatinumSingleEqZeroShotGenTask,
)
from sieval.tasks.platinum_bench.platinum_svamp_0shot_gen import (
    PlatinumSVAMPZeroShotGenTask,
)

from .conftest import (
    COT_PROMPT,
    NO_COT_PROMPT,
    O1_PROMPT,
    make_dataset,
    make_sample,
    make_task,
)

LEAVES = (
    PlatinumGSM8KZeroShotGenTask,
    PlatinumSVAMPZeroShotGenTask,
    PlatinumMultiArithZeroShotGenTask,
    PlatinumSingleOpZeroShotGenTask,
    PlatinumSingleEqZeroShotGenTask,
)


def _ctx(**kwargs) -> TaskContext:
    kwargs.setdefault("sample_id", 0)
    kwargs.setdefault("raw_sample", make_sample())
    return TaskContext(**kwargs)


# ---------------------------------------------------------------------------
# The leaf set: five distinct subsets, all on the vendored float-compare branch
# ---------------------------------------------------------------------------


def test_leaves_bind_five_distinct_subsets():
    subsets = [cls.subset for cls in LEAVES]
    assert sorted(subsets) == [
        "gsm8k",
        "multiarith",
        "singleop",
        "singleq",
        "svamp",
    ]


@pytest.mark.parametrize("task_cls", LEAVES, ids=lambda c: c.subset)
def test_every_subset_is_on_the_upstream_math_branch(task_cls):
    # `check_prediction` selects float-equality vs string-membership from a
    # hardcoded dataset-name list. A subset spelled outside that list would score
    # by membership — "42" vs ["42"] still passes, so nothing else would notice.
    assert check_prediction("42.0", ["42"], "prompt", task_cls.subset) is True


# ---------------------------------------------------------------------------
# setup() — the wrong-subset wiring guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_cls", LEAVES, ids=lambda c: c.subset)
@pytest.mark.anyio
async def test_setup_accepts_a_matching_dataset(task_cls):
    task, _ = make_task(task_cls)
    await task.setup()  # must not raise


@pytest.mark.anyio
async def test_setup_rejects_a_sibling_subsets_dataset():
    # Narrowing to the wrong subset would score real answers against the wrong
    # questions and look merely bad, not broken.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask, subset="svamp")
    with pytest.raises(ValueError, match="scores the 'gsm8k' subset"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_rejects_an_un_narrowed_merged_dataset():
    # The loader merges all 14 configs, so forgetting the filter operation is the
    # likeliest wiring mistake — and the one that silently mixes subsets.
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    task = type(task)(make_dataset("gsm8k", "svamp", "drop"), model)
    with pytest.raises(ValueError, match=r"carries \['drop', 'gsm8k', 'svamp'\]"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_error_names_the_filter_operation():
    # The message has to carry the fix; the wiring is no longer guessable from
    # the task name alone.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask, subset="svamp")
    with pytest.raises(ValueError, match=r"filter: \{by: subset, value: gsm8k\}"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_rejects_a_dataset_without_a_subset_column(monkeypatch):
    # A different Dataset class entirely (no `subset` column) must fail the same
    # way rather than crash inside `unique()`.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    monkeypatch.setattr(
        type(task.dataset),
        "test_set",
        property(lambda _self: HFDataset.from_dict({"other": [1]})),
    )
    with pytest.raises(ValueError, match=r"carries \[\]"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_rejects_an_empty_dataset():
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    task = type(task)(make_dataset(), model)
    with pytest.raises(ValueError, match=r"carries \[\]"):
        await task.setup()


# ---------------------------------------------------------------------------
# preprocess() — prompt comes from the row, one user turn
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_preprocess_sends_the_rows_cot_prompt_verbatim():
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample()
    pre = await task.preprocess(raw, _ctx(raw_sample=raw))
    assert pre["prompt"] == [{"role": "user", "content": COT_PROMPT}]
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == ["42"]


@pytest.mark.anyio
async def test_preprocess_no_cot_variant_reads_the_other_column():
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask, prompt_variant="no_cot")
    raw = make_sample()
    pre = await task.preprocess(raw, _ctx(raw_sample=raw))
    assert pre["prompt"][0]["content"] == NO_COT_PROMPT


@pytest.mark.anyio
async def test_o1_variant_rewrites_only_the_no_cot_column():
    # Upstream's o1 snapshots get `'Then, provide' -> 'Provide'` applied to the
    # no-CoT column. Asserting the whole string, not just the absence of "Then,",
    # is what catches the edit being applied to the CoT column instead — both
    # columns contain the substring, so a membership check would pass either way.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask, prompt_variant="no_cot_o1")
    raw = make_sample()
    pre = await task.preprocess(raw, _ctx(raw_sample=raw))
    assert pre["prompt"] == [{"role": "user", "content": O1_PROMPT}]


@pytest.mark.anyio
@pytest.mark.parametrize("variant", ["cot", "no_cot"])
async def test_non_o1_variants_keep_the_leftover_conjunction(variant):
    # Only `no_cot_o1` rewrites. Upstream's other 22 published rows were run
    # against the unedited wording, so editing it here would silently move them.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask, prompt_variant=variant)
    raw = make_sample()
    pre = await task.preprocess(raw, _ctx(raw_sample=raw))
    assert "Then, provide" in pre["prompt"][0]["content"]


def test_unknown_prompt_variant_is_rejected_at_construction():
    with pytest.raises(ValueError, match="prompt_variant must be one of"):
        make_task(PlatinumGSM8KZeroShotGenTask, prompt_variant="chain_of_thought")


@pytest.mark.anyio
async def test_preprocess_rejects_a_non_math_parsing_strategy():
    # The strategy is a data column, so a future revision can change it. A
    # silently-wrong parser would read as a model regression.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample(strategy="multiple_choice")
    with pytest.raises(ValueError, match="parsing strategy changed"):
        await task.preprocess(raw, _ctx(raw_sample=raw))


def test_math_strategy_constant_matches_upstreams_spelling():
    assert MATH_PARSING_STRATEGY == "math"


# ---------------------------------------------------------------------------
# infer() — the prompt and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_infer_injects_no_decode_params():
    # Every decoding param, max_tokens included, is a model-layer asset. The task
    # must inject none of them: `agenerate` merges `{**model_kwargs, **kwargs}`,
    # so a task-side value silently outranks whatever `models:` / `infer_args`
    # configured — and max_tokens is the one knob this benchmark most needs a
    # caller to turn, since the score is budget-sensitive.
    #
    # `n` is the single exception, and not a decoding param: it is the sampling
    # budget the task validated `k` against, so it has to reach the model or
    # pass@k is computed over a draw that never happened.
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample()
    pre = await task.preprocess(raw, _ctx(raw_sample=raw))
    await task.infer(pre, _ctx(raw_sample=raw))
    assert model.last_kwargs == {"n": 1}


@pytest.mark.anyio
async def test_infer_lets_the_configured_budget_reach_the_request():
    # The regression this pins: with a task-side default the merge discarded the
    # configured value, so a user raising the budget for a thinking model got
    # upstream's 6000 anyway and the truncation they were fixing persisted.
    task, model = make_task(
        PlatinumGSM8KZeroShotGenTask,
        model_kwargs={"max_tokens": 32000, "temperature": 0.5},
    )
    raw = make_sample()
    pre = await task.preprocess(raw, _ctx(raw_sample=raw))
    await task.infer(pre, _ctx(raw_sample=raw))
    assert model.last_kwargs == {"max_tokens": 32000, "temperature": 0.5, "n": 1}


# ---------------------------------------------------------------------------
# postprocess() — vendored extraction, None on failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: 42", "42"),
        ("Reasoning.\nAnswer: 1,000", "1000"),
        ("The total is $\\boxed{18.0}$.", "18"),
        ("**Answer: -7**", "-7"),
    ],
)
@pytest.mark.anyio
async def test_postprocess_extracts_via_upstream_parser(text, expected):
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    inf = ModelOutput(model=model.meta(), texts=[text])
    post = await task.postprocess(inf, _ctx())
    assert post["rollouts"][0]["prediction"] == expected


@pytest.mark.anyio
async def test_postprocess_records_none_when_output_has_no_digit():
    # Upstream raises AttributeError here and its runner swallows it into the
    # string 'parsing error'. `None` is the protocol's spelling of
    # "could not extract" — never "" and never a sentinel string.
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    inf = ModelOutput(model=model.meta(), texts=["I cannot solve this."])
    post = await task.postprocess(inf, _ctx())
    assert post["rollouts"][0]["prediction"] is None


@pytest.mark.anyio
async def test_postprocess_handles_an_empty_completion():
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    inf = ModelOutput(model=model.meta(), texts=[])
    post = await task.postprocess(inf, _ctx())
    assert post["rollouts"][0]["prediction"] is None


# ---------------------------------------------------------------------------
# feedback() — vendored scoring, exact float equality
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_feedback_scores_a_correct_answer():
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample(target="1000")
    inf = ModelOutput(model=model.meta(), texts=["Answer: 1,000"])
    ctx = _ctx(raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    finalize, fb = await task.feedback(post, ctx)
    assert finalize is True
    assert fb["reference"] == ["1000"]
    assert fb["rollouts"][0]["correct"] is True
    assert fb["extra"]["subset"] == "gsm8k"


@pytest.mark.anyio
async def test_feedback_accepts_a_float_spelling_of_an_integer_gold():
    # "18.0" vs gold "18": upstream compares floats, and the parser strips the
    # trailing ".0" anyway. Both layers have to agree for this to pass.
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample(target="18")
    inf = ModelOutput(model=model.meta(), texts=["Answer: 18.0"])
    ctx = _ctx(raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)
    assert fb["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_feedback_scores_a_wrong_answer():
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample(target="42")
    inf = ModelOutput(model=model.meta(), texts=["Answer: 7"])
    ctx = _ctx(raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_feedback_comparison_is_exact_not_tolerant():
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample(target="42")
    inf = ModelOutput(model=model.meta(), texts=["Answer: 42.0001"])
    ctx = _ctx(raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_feedback_treats_a_none_prediction_as_wrong():
    task, model = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample()
    inf = ModelOutput(model=model.meta(), texts=["I cannot solve this."])
    ctx = _ctx(raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    assert post["rollouts"][0]["prediction"] is None
    _, fb = await task.feedback(post, ctx)
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_feedback_survives_an_absent_prediction_key():
    # A None field is ABSENT once the record round-trips through disk, and the
    # resume path reads it back from disk. Indexing would raise KeyError there
    # while passing every in-memory test.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample()
    post = build_prediction_record([None])
    post["rollouts"][0].pop("prediction", None)
    _, fb = await task.feedback(post, _ctx(raw_sample=raw))
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_feedback_treats_an_unparseable_number_as_wrong():
    # The parse regex accepts any run of digits and dots, so "1.2.3" reaches
    # float() and raises. Upstream swallows the same ValueError; wrong, not error.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    raw = make_sample()
    post = build_prediction_record(["1.2.3"])
    _, fb = await task.feedback(post, _ctx(raw_sample=raw))
    assert fb["rollouts"][0]["correct"] is False


# ---------------------------------------------------------------------------
# report() — accuracy plus upstream's error count
# ---------------------------------------------------------------------------


def _final(sample_id: int, correct: bool) -> TaskContext:
    return TaskContext(
        sample_id=sample_id,
        raw_sample=make_sample(),
        feedback_result=build_judgement_record(
            ["42"], [build_rollout_judgement(0, correct)]
        ),
    )


@pytest.mark.anyio
async def test_report_emits_accuracy_and_error_count():
    # `errors` is upstream's headline unit — how many of the subset's questions
    # the model got wrong — so results are directly comparable to its tables.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    finals = [_final(0, True), _final(1, True), _final(2, False), _final(3, False)]
    assert await task.report(finals, []) == {
        "score": 50.0,
        "fails": 0,
        "accuracy": 50.0,
        "errors": 2,
        # `score` is upstream's first-rollout accuracy, not a sampling metric.
        "score_key": "accuracy",
    }


@pytest.mark.anyio
async def test_report_counts_fails_as_errors():
    # Upstream has no failure bucket — every row it fetched gets a verdict — so
    # a pipeline failure counts as wrong to keep the denominators equal.
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    finals = [_final(0, True)]
    fails = [TaskContext(sample_id=1, raw_sample=make_sample())]
    assert await task.report(finals, fails) == {
        "score": 50.0,
        "fails": 1,
        "accuracy": 50.0,
        "errors": 1,
        "score_key": "accuracy",
    }


@pytest.mark.anyio
async def test_report_on_an_empty_run():
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask)
    assert await task.report([], []) == {
        "score": 0.0,
        "fails": 0,
        "accuracy": 0.0,
        "errors": 0,
        "score_key": "accuracy",
    }


# ---------------------------------------------------------------------------
# Shared reference-impl notes
# ---------------------------------------------------------------------------


def test_upstream_url_is_commit_pinned():
    assert PLATINUM_UPSTREAM_URL.startswith(
        "https://github.com/MadryLab/platinum-benchmarks/blob/"
        "8fd2f82e63c49ea1cca4266f4dded82b7ddbcb55/"
    )


def test_reference_notes_carry_the_repro_decoding_protocol():
    # `sieval task show` is the only place these surface, and upstream's sampling
    # protocol (not greedy) cannot be inferred from the code.
    assert "temperature=0.5" in PLATINUM_REFERENCE_NOTES
    assert "no seed" in PLATINUM_REFERENCE_NOTES
    assert "max_tokens=6000" in PLATINUM_REFERENCE_NOTES
    assert "prompt_variant" in PLATINUM_REFERENCE_NOTES


def test_reference_notes_carry_the_validation_record():
    # `status="stable"` is a claim about reproduction; the evidence for it has to
    # travel with the task, not live only in a merged PR body.
    assert "all 120 math error counts" in PLATINUM_REFERENCE_NOTES
    assert "platinum-bench-paper-cache" in PLATINUM_REFERENCE_NOTES
    assert "platinum-bench-paper-version" in PLATINUM_REFERENCE_NOTES
    # Which prompt/temperature reproduces which published row — the recipe is
    # useless without this, since three of the four combinations are exceptions.
    for variant in ("cot", "no_cot", "no_cot_o1"):
        assert variant in PLATINUM_REFERENCE_NOTES
    # The pinned revision is NOT the one the paper's numbers are computed on, so a
    # reader comparing against Table 3 has to be told the row counts differ.
    assert "968 -> 953" in PLATINUM_REFERENCE_NOTES
    # The replay stubs the model layer, so the notes must not let it be read as
    # end-to-end evidence — the live run is a separate claim.
    assert "run live end-to-end" in PLATINUM_REFERENCE_NOTES


def test_reference_notes_warn_that_the_token_budget_suits_non_thinking_models():
    # A thinking model can spend all 6000 tokens reasoning and return an empty
    # answer, which scores as an error indistinguishable from a wrong one unless
    # the reader is told to check `anomalies.json`. Measured, not theorized.
    assert "sized for non-thinking models" in PLATINUM_REFERENCE_NOTES
    assert "truncated_output" in PLATINUM_REFERENCE_NOTES


def test_reference_notes_state_that_the_budget_is_the_callers():
    # The task forwards no max_tokens, so a run at some backend's default budget
    # silently under-scores. `sieval task show` is the only place a caller learns
    # they have to set it — and that 6000 is the value that reproduces upstream.
    assert "Infer prereqs" in PLATINUM_REFERENCE_NOTES
    assert "max_tokens=6000" in PLATINUM_REFERENCE_NOTES
    assert "budget-sensitive" in PLATINUM_REFERENCE_NOTES


# --- n / k sampling wiring -------------------------------------------------
#
# Three regressions that shipped together in the first draft of this feature,
# none of them visible to a report-key assertion.


@pytest.mark.anyio
async def test_infer_forwards_n_to_the_model():
    """Otherwise `n=4` enables the sampling metrics over a one-rollout draw."""
    task, model = make_task(PlatinumGSM8KZeroShotGenTask, k=4, n=4)
    pre = await task.preprocess(make_sample(), None)
    await task.infer(pre, None)
    assert model.last_kwargs.get("n") == 4


@pytest.mark.anyio
async def test_postprocess_keeps_every_rollout():
    """`texts[0]` would discard n-1 rollouts that were generated and paid for."""
    task, model = make_task(PlatinumGSM8KZeroShotGenTask, k=4, n=4)
    out = ModelOutput(
        model=model.meta(),
        texts=["Answer: 42", "Answer: 42", "Answer: 42", "Answer: 7"],
    )
    post = await task.postprocess(out, None)
    assert [r["prediction"] for r in post["rollouts"]] == ["42", "42", "42", "7"]


@pytest.mark.anyio
async def test_feedback_grades_every_rollout():
    task, _ = make_task(PlatinumGSM8KZeroShotGenTask, k=4, n=4)
    post = build_prediction_record(["42", "42", "7", "7"])
    raw = make_sample()
    _, judgement = await task.feedback(post, _ctx(raw_sample=raw))
    assert [r["correct"] for r in judgement["rollouts"]] == [True, True, False, False]
