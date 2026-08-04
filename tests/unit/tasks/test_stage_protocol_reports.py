"""Report-parity tests for the tasks migrated to the stage-output protocol.

Each task's ``report()`` was rewritten to read protocol records instead of a
bespoke feedback shape. The numbers must not have moved, so every expectation
here is computed by hand from the fixture rather than from the implementation.

Covers the pilot set: mmlu_pro, gpqa_diamond, aime_2026, hmmt_feb_2026,
livecodebench, ifeval, simpleqa_verified.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from dataclasses import replace

import pytest

from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)


def _final(feedback, *, postprocess=None, preprocess=None) -> TaskContext:
    """A FINAL-stage context carrying *feedback* as its judgement record."""
    ctx = TaskContext(sample_id=0, raw_sample={})
    ctx = ctx.to_preprocessed(preprocess if preprocess is not None else {"prompt": "p"})
    ctx = ctx.to_inferred("inf")
    ctx = ctx.to_postprocessed(
        postprocess if postprocess is not None else build_prediction_record(["x"])
    )
    ctx = ctx.to_feedback(feedback)
    return ctx.to_final()


def _task(cls, **kwargs):
    """Build a task without touching a dataset or model.

    ``report()`` reads only the contexts it is handed plus constructor knobs, so
    bypassing ``__init__`` keeps these tests free of dataset/model fixtures.
    """
    task = object.__new__(cls)
    for key, value in kwargs.items():
        setattr(task, key, value)
    return task


@pytest.mark.anyio
class TestMMLUProReport:
    async def test_overall_and_per_category_scores(self):
        from sieval.tasks.mmlu_pro_0shot_gen import MMLUProZeroShotGenTask

        finals = [
            _final(
                build_judgement_record(
                    "A",
                    [build_rollout_judgement(0, correct)],
                    extra={"category": category},
                )
            )
            for correct, category in [
                (True, "physics"),
                (False, "physics"),
                (True, "law"),
                (True, "law"),
            ]
        ]
        report = await _task(MMLUProZeroShotGenTask).report(finals, [])

        assert report["score"] == pytest.approx(75.0)  # 3 of 4
        assert report["score_physics"] == pytest.approx(50.0)  # 1 of 2
        assert report["score_law"] == pytest.approx(100.0)  # 2 of 2
        assert report["fails"] == 0

    async def test_empty_finals_scores_zero(self):
        from sieval.tasks.mmlu_pro_0shot_gen import MMLUProZeroShotGenTask

        report = await _task(MMLUProZeroShotGenTask).report([], [])
        assert report["score"] == 0.0
        assert report["fails"] == 0


@pytest.mark.anyio
class TestGPQADiamondReport:
    async def test_score_counts_correct_samples(self):
        from sieval.tasks.gpqa_diamond_0shot_gen import GPQADiamondZeroShotGenTask

        finals = [
            _final(build_judgement_record("A", [build_rollout_judgement(0, correct)]))
            for correct in [True, False, False, False]
        ]
        report = await _task(GPQADiamondZeroShotGenTask).report(finals, [])
        assert report["score"] == pytest.approx(25.0)
        assert report["fails"] == 0

    async def test_empty_finals_scores_zero(self):
        from sieval.tasks.gpqa_diamond_0shot_gen import GPQADiamondZeroShotGenTask

        report = await _task(GPQADiamondZeroShotGenTask).report([], [_final(None)])
        assert report["score"] == 0.0
        assert report["fails"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("sieval.tasks.aime_2024_0shot_gen", "AIME2024ZeroShotGenTask"),
        ("sieval.tasks.aime_2025_0shot_gen", "AIME2025ZeroShotGenTask"),
        ("sieval.tasks.aime_2026_0shot_gen", "AIME2026ZeroShotGenTask"),
        ("sieval.tasks.hmmt_feb_2025_0shot_gen", "HMMTFeb2025ZeroShotGenTask"),
        ("sieval.tasks.hmmt_feb_2026_0shot_gen", "HMMTFeb2026ZeroShotGenTask"),
        ("sieval.tasks.hmmt_nov_2025_0shot_gen", "HMMTNov2025ZeroShotGenTask"),
        ("sieval.tasks.math_500_0shot_gen", "MATH500ZeroShotGenTask"),
        ("sieval.tasks.imo_answer_bench_0shot_gen", "IMOAnswerBenchZeroShotGenTask"),
    ],
)
class TestMathPassAtKReports:
    """All eight pass@k math tasks share one implementation shape."""

    @staticmethod
    def _load(module_name, class_name):
        import importlib

        return getattr(importlib.import_module(module_name), class_name)

    async def test_pass_at_1_averages_per_sample_rates(self, module_name, class_name):
        cls = self._load(module_name, class_name)
        # Two samples, n=2: one fully correct, one half correct -> (1 + 0.5) / 2.
        finals = [
            _final(
                build_judgement_record(
                    "42",
                    [
                        build_rollout_judgement(0, True),
                        build_rollout_judgement(1, True),
                    ],
                )
            ),
            _final(
                build_judgement_record(
                    "42",
                    [
                        build_rollout_judgement(0, True),
                        build_rollout_judgement(1, False),
                    ],
                )
            ),
        ]
        report = await _task(cls, _k=1, _n=2).report(finals, [])
        assert report["score"] == pytest.approx(75.0)
        assert report["pass@1"] == pytest.approx(75.0)
        assert "pass@2" not in report
        assert report["fails"] == 0

    async def test_pass_at_k_uses_the_unbiased_estimator(self, module_name, class_name):
        cls = self._load(module_name, class_name)
        # n=2, c=1 -> pass@2 = 1 - (2-1-0)/(2-0) * (2-1-1)/(2-1) = 1.0
        finals = [
            _final(
                build_judgement_record(
                    "42",
                    [
                        build_rollout_judgement(0, True),
                        build_rollout_judgement(1, False),
                    ],
                )
            )
        ]
        report = await _task(cls, _k=2, _n=2).report(finals, [])
        assert report["pass@1"] == pytest.approx(50.0)
        assert report["pass@2"] == pytest.approx(100.0)

    async def test_fails_dilute_the_score(self, module_name, class_name):
        cls = self._load(module_name, class_name)
        finals = [
            _final(build_judgement_record("42", [build_rollout_judgement(0, True)]))
        ]
        # One correct final + one failed sample -> 1/2 of the requested set.
        report = await _task(cls, _k=1, _n=1).report(finals, [_final(None)])
        assert report["score"] == pytest.approx(50.0)
        assert report["fails"] == 1

    async def test_empty_run_keeps_the_full_key_set(self, module_name, class_name):
        cls = self._load(module_name, class_name)
        report = await _task(cls, _k=2, _n=2).report([], [])
        assert report == {"score": 0.0, "fails": 0, "pass@1": 0.0, "pass@2": 0.0}


@pytest.mark.anyio
class TestLiveCodeBenchReport:
    @staticmethod
    def _rollout(index, correct, msg=""):
        return build_rollout_judgement(
            index,
            correct,
            extra={"msg": msg, "n_cases": 3, "n_passed": 3},
        )

    async def test_pass_at_1_and_timeout_counting(self):
        from sieval.tasks.livecodebench_code_generation_0shot_gen import (
            LiveCodeBenchCodeGenerationZeroShotGenTask as Cls,
        )

        finals = [
            _final(
                build_judgement_record(
                    None, [self._rollout(0, True), self._rollout(1, False)]
                )
            ),
            _final(
                build_judgement_record(
                    None,
                    [
                        self._rollout(0, False, "failed: subprocess timeout: 6.0s"),
                        self._rollout(1, False, "failed: output 1 != expect 2"),
                    ],
                )
            ),
        ]
        report = await _task(Cls, _k=1, _n=2).report(finals, [])

        assert report["score"] == pytest.approx(25.0)  # (0.5 + 0.0) / 2
        assert report["pass@1"] == pytest.approx(25.0)
        assert report["timeouts"] == 1
        assert report["fails"] == 0

    async def test_timeout_match_is_case_insensitive(self):
        from sieval.tasks.livecodebench_code_generation_0shot_gen import (
            LiveCodeBenchCodeGenerationZeroShotGenTask as Cls,
        )

        finals = [
            _final(
                build_judgement_record(
                    None, [self._rollout(0, False, "failed: TIMEOUT 6s")]
                )
            )
        ]
        report = await _task(Cls, _k=1, _n=1).report(finals, [])
        assert report["timeouts"] == 1

    async def test_empty_run_scores_zero(self):
        from sieval.tasks.livecodebench_code_generation_0shot_gen import (
            LiveCodeBenchCodeGenerationZeroShotGenTask as Cls,
        )

        report = await _task(Cls, _k=1, _n=1).report([], [])
        assert report == {"score": 0.0, "fails": 0}


@pytest.mark.anyio
class TestIFEvalReport:
    @staticmethod
    def _final(strict_followed, loose_followed):
        # Mirrors the task: both readings in `metrics`, per-instruction lists in
        # `extra`, headline derived from `metrics` rather than recomputed.
        def rate(followed):
            return sum(followed) / len(followed) if followed else 0.0

        metrics: dict[str, bool | float] = {
            "strict_follow_all": all(strict_followed),
            "strict_instruction_level": rate(strict_followed),
            "loose_follow_all": all(loose_followed),
            "loose_instruction_level": rate(loose_followed),
        }
        detail = {
            "strict": {"follow_instruction_list": strict_followed},
            "loose": {"follow_instruction_list": loose_followed},
        }
        correct = bool(metrics["strict_follow_all"])
        score = float(metrics["strict_instruction_level"])
        return _final(
            build_judgement_record(
                ["a", "b"],
                [build_rollout_judgement(0, correct, score=score, metrics=metrics)],
                score=score,
                metrics=metrics,
                extra={"key": 1, **detail},
            )
        )

    async def test_prompt_and_instruction_level_accuracies(self):
        from sieval.tasks.ifeval_0shot_gen import IFEvalZeroShotGenTask

        finals = [
            self._final([True, True], [True, True]),
            self._final([True, False], [True, True]),
        ]
        report = await _task(IFEvalZeroShotGenTask).report(finals, [])

        # strict: 1 of 2 prompts fully followed; 3 of 4 instructions followed.
        assert report["strict_prompt_level_accuracy"] == pytest.approx(50.0)
        assert report["strict_accuracy"] == pytest.approx(50.0)
        assert report["strict_instruction_level_accuracy"] == pytest.approx(75.0)
        # loose: both prompts fully followed, all 4 instructions.
        assert report["loose_prompt_level_accuracy"] == pytest.approx(100.0)
        assert report["loose_instruction_level_accuracy"] == pytest.approx(100.0)
        # headline is the strict prompt-level accuracy
        assert report["score"] == pytest.approx(50.0)
        assert report["fails"] == 0

    async def test_instruction_level_pools_counts_rather_than_averaging_samples(self):
        from sieval.tasks.ifeval_0shot_gen import IFEvalZeroShotGenTask

        # Sample A follows 1 of 1; sample B follows 1 of 3. Pooled = 2/4 = 50%,
        # whereas averaging per-sample rates would give (100 + 33.3) / 2 = 66.7%.
        finals = [
            self._final([True], [True]),
            self._final([True, False, False], [True, False, False]),
        ]
        report = await _task(IFEvalZeroShotGenTask).report(finals, [])
        assert report["strict_instruction_level_accuracy"] == pytest.approx(50.0)

    async def test_empty_finals_report_zeros_instead_of_dividing_by_zero(self):
        from sieval.tasks.ifeval_0shot_gen import IFEvalZeroShotGenTask

        report = await _task(IFEvalZeroShotGenTask).report([], [_final(None)])
        assert report["score"] == 0.0
        assert report["strict_instruction_level_accuracy"] == 0.0
        assert report["loose_prompt_level_accuracy"] == 0.0
        assert report["fails"] == 1


@pytest.mark.anyio
class TestSimpleQAVerifiedReport:
    @staticmethod
    def _final(*grades):
        # report() reads only extra["grade"]; the grader_output the real path also
        # writes is irrelevant to F1 aggregation and omitted here.
        return _final(
            build_judgement_record(
                "gold",
                [
                    build_rollout_judgement(
                        index, grade == "CORRECT", extra={"grade": grade}
                    )
                    for index, grade in enumerate(grades)
                ],
            )
        )

    async def test_f1_is_aggregated_from_the_three_way_grades(self):
        from sieval.tasks.simpleqa_verified_0shot_gen import (
            SimpleQAVerifiedZeroShotGenTask as Cls,
        )

        finals = [
            self._final("CORRECT"),
            self._final("CORRECT"),
            self._final("INCORRECT"),
            self._final("NOT_ATTEMPTED"),
        ]
        report = await _task(Cls, _n=1).report(finals, [])

        # correct 2/4 = 0.5; attempted 3/4 so accuracy-given-attempted = 2/3.
        # F1 = 2 * 0.5 * (2/3) / (0.5 + 2/3) = 0.5714...
        assert report["correct"] == pytest.approx(50.0)
        assert report["incorrect"] == pytest.approx(25.0)
        assert report["not_attempted"] == pytest.approx(25.0)
        assert report["accuracy_given_attempted"] == pytest.approx(200 / 3)
        assert report["f1"] == pytest.approx(57.142857, abs=1e-4)
        assert report["score"] == report["f1"]
        assert report["n_graded"] == 4

    async def test_failed_samples_count_as_not_attempted_per_requested_attempt(self):
        from sieval.tasks.simpleqa_verified_0shot_gen import (
            SimpleQAVerifiedZeroShotGenTask as Cls,
        )

        finals = [self._final("CORRECT", "CORRECT")]
        # n=2 with one failed sample adds 2 NOT_ATTEMPTED, so correct = 2/4.
        report = await _task(Cls, _n=2).report(finals, [_final(None)])
        assert report["n_graded"] == 2
        assert report["correct"] == pytest.approx(50.0)
        assert report["not_attempted"] == pytest.approx(50.0)
        assert report["fails"] == 1

    async def test_tolerates_a_context_with_no_feedback(self):
        from sieval.tasks.simpleqa_verified_0shot_gen import (
            SimpleQAVerifiedZeroShotGenTask as Cls,
        )

        empty = replace(_final(None), feedback_result=None)
        report = await _task(Cls, _n=1).report([empty], [])
        assert report["n_graded"] == 0
        assert report["score"] == 0.0
