"""The report metrics of each built-in task, by the class that defines ``report``.

A table rather than an attribute on each class: a task module's source is in its
stage identities, so an attribute there would rerun every resumed directory from
``postprocess``. ``sieval.meta`` is outside the grader walk. Plugin tasks
declare on the class. Each entry is checked against the reports that
``tests/unit/tasks`` runs (``tests/unit/tasks/conftest.py``).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from types import MappingProxyType

from sieval.core.tasks.report_metrics import (
    HEALTH_METRICS,
    SAMPLING_METRICS,
    SAMPLING_VOTE_METRICS,
    ReportMetrics,
    declared_report_metrics,
    report_metrics,
)

_SAMPLED = SAMPLING_METRICS | HEALTH_METRICS
_VOTED = _SAMPLED | SAMPLING_VOTE_METRICS

#: The math competitions: a sampling block with votes, headline `pass@1`.
_COMPETITION = report_metrics("score", "fails") | _VOTED
#: The code tasks: a sampling block without votes, and timeouts.
_CODE = report_metrics("score", "fails", "timeouts") | _SAMPLED
#: One accuracy over problems, with its population and extraction health.
_ACCURACY = report_metrics("score", "accuracy", "fails", "n_problems") | HEALTH_METRICS
#: A judged three-way grade: correct, incorrect, and what was graded.
_JUDGED = (
    report_metrics(
        "score", "accuracy", "correct", "incorrect", "fails", "n_graded", "n_problems"
    )
    | HEALTH_METRICS
)
#: ARC's shared report.
_ARC = report_metrics("score", "acc", "fails", "n_problems")
#: MMLU-style macro: the headline and one score per category.
_CATEGORY = report_metrics("score", "fails", patterns=("score_{category}",))
#: C-Eval and CMMLU: the subject macro, and one rate per category that has a
#: subject in the run, named by the category.
_SUBJECT_MACRO = report_metrics("score", "overall", "fails", "n_subjects")
_CEVAL_CATEGORIES = ("stem", "social_science", "humanities", "other")
_CMMLU_CATEGORIES = (*_CEVAL_CATEGORIES, "china_specific")
#: IFEval-style strict and loose rates.
_INSTRUCTION = report_metrics(
    "score",
    "fails",
    "n_problems",
    patterns=(
        "{grade}_prompt_level_accuracy",
        "{grade}_instruction_level_accuracy",
    ),
)
#: BFCL's two families share a report and differ in their categories.
_BFCL = report_metrics("score", "fails", "ast_summary", "n_problems", "n_ast")

_P = "sieval.tasks"

#: Each built-in class defining ``report``, by import path.
BUILTIN_REPORT_METRICS: MappingProxyType[str, ReportMetrics] = MappingProxyType(
    {
        f"{_P}._math_perturb_base.MathPerturbZeroShotGenTask": report_metrics(
            "score",
            "fails",
            patterns=(
                "score_seed_{split}",
                "n_problems_seed_{split}",
                "score_type_{type}",
            ),
        )
        | _SAMPLED,
        f"{_P}._nl2sh_alfa_base.NL2SHAlfaSharedZeroShotGenTask": report_metrics(
            "score",
            "accuracy",
            "fails",
            "mean_reward",
            "n_gold_timeouts",
            "n_model_nonzero_exit",
            "n_model_timeouts",
            "n_p3_command_match",
            "n_p3_output_match",
            "n_p3_empty_output",
            "n_p3_embed",
            "n_problems",
        )
        | HEALTH_METRICS,
        f"{_P}._quotebench_base.QuoteBenchTask": report_metrics(
            "score",
            "pass_rate_pct",
            "fails",
            "n_problems",
            "n_pass",
            "n_environment_invalid",
            "n_timeout",
            "n_shell_syntax",
            "n_tool_error",
            "n_runtime_error",
            "n_silent_wrong",
            "n_unknown_class",
            patterns=("pass_rate_pct_tier{tier}", "pass_rate_pct_{scenario}"),
        )
        | HEALTH_METRICS,
        f"{_P}.aa_lcr_0shot_gen.AALCRZeroShotGenTask": _JUDGED,
        f"{_P}.advanced_if_0shot_gen.AdvancedIFZeroShotGenTask": report_metrics(
            "score",
            "overall_pass_rate",
            "macro_pass_rate",
            "micro_pass_rate",
            "fails",
            "n_graded",
            "n_grader_unparsed",
            "n_problems",
            "n_rubric_checks",
            patterns=(
                "{benchmark}_pass_rate",
                "{benchmark}_micro_pass_rate",
                "{benchmark}_macro_pass_rate",
                "{benchmark}_n_grader_unparsed",
                "{benchmark}_n_graded",
            ),
        )
        | HEALTH_METRICS,
        f"{_P}.ag_livecodebench_x_0shot_gen.AgLiveCodeBenchXZeroShotGenTask": (
            report_metrics("score", "fails", "n_run_errors") | _SAMPLED
        ),
        f"{_P}.agieval_0shot_gen.AGIEvalZeroShotGenTask": report_metrics(
            "score",
            "fails",
            "n_subsets",
            "macro_en_mcq",
            "macro_zh_mcq",
            "macro_math",
            patterns=("score_{subset}",),
        ),
        f"{_P}.aime_2024_0shot_gen.AIME2024ZeroShotGenTask": _COMPETITION,
        f"{_P}.aime_2025_0shot_gen.AIME2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.aime_2026_0shot_gen.AIME2026ZeroShotGenTask": _COMPETITION,
        f"{_P}.apex_2025_0shot_gen.Apex2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.apex_shortlist_2025_0shot_gen.ApexShortlist2025ZeroShotGenTask": (
            _COMPETITION
        ),
        f"{_P}.arc.arc_challenge_kshot_clp.ARCChallengeFewShotClpTask": _ARC,
        f"{_P}.arc.arc_challenge_kshot_ppl.ARCChallengeFewShotPplTask": _ARC,
        f"{_P}.arc.arc_easy_kshot_clp.ARCEasyFewShotClpTask": _ARC,
        f"{_P}.arc.arc_easy_kshot_ppl.ARCEasyFewShotPplTask": _ARC,
        f"{_P}.bfcl_v3._base.BfclV3NonLiveTask": _BFCL
        | report_metrics(
            "non_live_overall_acc",
            "simple_ast",
            "simple",
            "multiple",
            "parallel",
            "parallel_multiple",
            "java",
            "javascript",
            "irrelevance",
            "n_simple",
            "n_multiple",
            "n_parallel",
            "n_parallel_multiple",
            "n_java",
            "n_javascript",
            "n_irrelevance",
        ),
        f"{_P}.bfcl_v3._base.BfclV3LiveTask": _BFCL
        | report_metrics(
            "live_overall_acc",
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
            "live_irrelevance",
            "live_relevance",
            "n_live_simple",
            "n_live_multiple",
            "n_live_parallel",
            "n_live_parallel_multiple",
            "n_live_irrelevance",
            "n_live_relevance",
        ),
        f"{_P}.browsecomp_0shot_gen.BrowseCompZeroShotGenTask": _JUDGED,
        f"{_P}.brumo_2025_0shot_gen.BRUMO2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.c_eval_kshot_clp.CEvalFewShotCLPTask": _SUBJECT_MACRO
        | _CEVAL_CATEGORIES,
        f"{_P}.cmimc_2025_0shot_gen.CMIMC2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.cmmlu_kshot_clp.CMMLUFewShotClpTask": _SUBJECT_MACRO | _CMMLU_CATEGORIES,
        f"{_P}.complex_constraints_0shot_gen.ComplexConstraintsZeroShotGenTask": (
            report_metrics(
                "score",
                "task_pass_rate",
                "criterion_pass_rate_macro",
                "criterion_pass_rate_micro",
                "fails",
                "n_criteria_graded",
                "n_graded",
                "n_grader_unparsed",
                "n_problems",
            )
            | HEALTH_METRICS
        ),
        f"{_P}.drop_kshot_gen.DROPFewShotGenTask": report_metrics(
            "score", "em", "f1", "fails", "n_problems"
        ),
        f"{_P}.gpqa_diamond_0shot_gen.GPQADiamondZeroShotGenTask": report_metrics(
            "score", "fails", "n_problems"
        )
        | HEALTH_METRICS,
        f"{_P}.gsm1k_0shot_gen.GSM1KZeroShotGenTask": _ACCURACY,
        f"{_P}.gsm1k_kshot_base_gen.GSM1KFewShotBaseGenTask": report_metrics(
            "score", "strict_exact_match", "flexible_exact_match", "fails", "n_problems"
        )
        | HEALTH_METRICS,
        f"{_P}.gsm8k_0shot_gen.GSM8KZeroShotGenTask": _ACCURACY | _VOTED,
        f"{_P}.gsm8k_kshot_base_gen.GSM8KFewShotBaseGenTask": report_metrics(
            "score", "exact_match", "flexible_exact_match", "fails"
        )
        | _VOTED,
        f"{_P}.gsm_plus_0shot_gen.GSMPlusZeroShotGenTask": _ACCURACY
        | report_metrics(
            "score_wo_critical_thinking",
            "n_problems_wo_critical_thinking",
            patterns=("score_{perturbation}",),
        ),
        f"{_P}.hellaswag_kshot_ppl.HellaSwagFewShotPPLTask": report_metrics(
            "score", "acc", "acc_norm", "fails", "n_problems"
        ),
        f"{_P}.hendrycks_math_kshot_base_gen.HendrycksMathFewShotBaseGenTask": (
            _ACCURACY | _VOTED
        ),
        f"{_P}.hle_0shot_gen.HLEZeroShotGenTask": report_metrics(
            "score",
            "accuracy",
            "calibration_error",
            "confidence_interval",
            "fails",
            "n",
            "n_graded",
            "n_grader_unparsed",
            "n_problems",
        )
        | HEALTH_METRICS,
        f"{_P}.hmmt_feb_2025_0shot_gen.HMMTFeb2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.hmmt_feb_2026_0shot_gen.HMMTFeb2026ZeroShotGenTask": _COMPETITION,
        f"{_P}.hmmt_nov_2025_0shot_gen.HMMTNov2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.human_eval_0shot_base_gen.HumanEvalZeroShotBaseGenTask": _CODE,
        f"{_P}.human_eval_0shot_gen.HumanEvalZeroShotGenTask": _CODE,
        f"{_P}.ifbench_0shot_gen.IFBenchZeroShotGenTask": _INSTRUCTION,
        f"{_P}.ifeval_0shot_gen.IFEvalZeroShotGenTask": _INSTRUCTION
        | report_metrics(patterns=("{grade}_accuracy",)),
        f"{_P}.iheval_0shot_gen.IHEvalZeroShotGenTask": report_metrics(
            "score",
            "fails",
            "reference_cells_degraded",
            patterns=(
                "score_{setting}",
                "score_{subtask}_{setting}",
                "cell_{subtask}_{setting}_{variant}",
                "diff_{setting}",
                "abs_diff_{setting}",
            ),
        ),
        f"{_P}.imo_answer_bench_0shot_gen.IMOAnswerBenchZeroShotGenTask": (
            _COMPETITION
        ),
        f"{_P}.inverse_ifeval_0shot_gen.InverseIFEvalZeroShotGenTask": report_metrics(
            "score",
            "fails",
            "n_graded",
            "n_grader_unparsed",
            patterns=("score_{group}", "n_{group}"),
        )
        | _SAMPLED,
        f"{_P}.livecodebench_code_generation_0shot_gen."
        "LiveCodeBenchCodeGenerationZeroShotGenTask": _CODE,
        f"{_P}.livecodebench_code_generation_kshot_base_gen."
        "LiveCodeBenchCodeGenerationFewShotBaseGenTask": _CODE,
        f"{_P}.liveoibench_0shot_gen.LiveOIBenchZeroShotGenTask": report_metrics(
            "score",
            "relative_score",
            "pass_rate",
            "tests_passed_pct",
            "human_percentile",
            "medal_rate",
            "gold_rate",
            "fails",
            "n",
            "k",
            "n_short",
            "n_problems",
            "n_scored",
            "n_contests",
            "n_contests_ranked",
            "n_contests_medalled",
        )
        | HEALTH_METRICS,
        f"{_P}.math_500_0shot_gen.MATH500ZeroShotGenTask": _COMPETITION,
        f"{_P}.mbpp_kshot_base_gen.MBPPFewShotBaseGenTask": _CODE,
        f"{_P}.mmlu_0shot_gen.MMLUZeroShotGenTask": _CATEGORY
        | report_metrics("n_problems")
        | HEALTH_METRICS,
        f"{_P}.mmlu_kshot_clp.MMLUFewShotCLPTask": _CATEGORY
        | report_metrics("n_problems"),
        f"{_P}.mmlu_pro_0shot_gen.MMLUProZeroShotGenTask": _CATEGORY
        | report_metrics("n_problems")
        | HEALTH_METRICS,
        f"{_P}.mmmlu_kshot_clp.MMMLUKShotClpTask": report_metrics(
            "score",
            "score_mmmlu",
            "fails",
            "n_problems",
            patterns=(
                "score_locale_{locale}",
                "score_locale_{locale}_category_{category}",
                "score_locale_{locale}_subject_{subject}",
            ),
        ),
        f"{_P}.multi_if_0shot_gen.MultiIFZeroShotGenTask": report_metrics(
            "score",
            "fails",
            "all_turns_all_languages_overall",
            patterns=(
                "turn_{turn}_prompts_number",
                "turn_{turn}_{grade}_prompt_level_accuracy",
                "turn_{turn}_{grade}_instruction_level_accuracy",
                "turn_{turn}_{language}_overall",
            ),
        ),
        f"{_P}.multipl_e._base.MultiPLETask": report_metrics(
            "score",
            "fails",
            "timeouts",
            "n_build_errors",
            "n_execution_errors",
            "n_languages",
            "pass@1_macro",
            patterns=("pass@1_{language}", "n_problems_{language}"),
        )
        | _SAMPLED,
        f"{_P}.openbookqa_kshot_gen.OpenBookQAFewShotGenTask": _ACCURACY,
        f"{_P}.platinum_bench._base.PlatinumMathGenTask": _ACCURACY
        | report_metrics("errors")
        | _VOTED,
        f"{_P}.ruler_0shot_gen.RulerZeroShotGenTask": report_metrics(
            "score",
            "fails",
            patterns=("score_{length}", "score_{subtask}_{length}"),
        ),
        f"{_P}.scicode_0shot_gen.SciCodeZeroShotGenTask": report_metrics(
            "score",
            "main_problem_accuracy",
            "sub_problem_accuracy",
            "correct_problems",
            "correct_steps",
            "total_problems",
            "total_steps",
            "unevaluated_steps",
            "empty_extractions",
            "import_errors",
            "memory_errors",
            "timeouts",
            "fails",
            "n_problems",
        ),
        f"{_P}.scitarc_0shot_gen.SciTaRCZeroShotGenTask": report_metrics(
            "score",
            "accuracy",
            "exact_match",
            "partial",
            "fails",
            "n",
            "n_graded",
            "n_grader_unparsed",
            "n_problems",
        )
        | HEALTH_METRICS,
        f"{_P}.simpleqa_verified_0shot_gen.SimpleQAVerifiedZeroShotGenTask": (
            report_metrics(
                "score",
                "f1",
                "accuracy_given_attempted",
                "correct",
                "incorrect",
                "not_attempted",
                "fails",
                "n_graded",
                "n_problems",
            )
            | HEALTH_METRICS
        ),
        f"{_P}.smt_2025_0shot_gen.SMT2025ZeroShotGenTask": _COMPETITION,
        f"{_P}.spider.spider_0shot_gen.SpiderZeroShotGenTask": report_metrics(
            "score",
            "execution_accuracy",
            "exact_match",
            "test_suite_accuracy",
            "fails",
            "n",
            "n_execution_errors",
            "n_parser_rejected",
            "n_problems",
            patterns=("test_suite_accuracy_{level}", "n_{level}"),
        )
        | HEALTH_METRICS,
        f"{_P}.spider2_lite_0shot_gen.Spider2LiteZeroShotGenTask": report_metrics(
            "score",
            "execution_accuracy",
            "fails",
            "n",
            "n_execution_errors",
            "n_missing_credentials",
            patterns=(
                "execution_accuracy_{backend}",
                "n_{backend}",
                "n_missing_credentials_{backend}",
            ),
        )
        | HEALTH_METRICS,
        f"{_P}.sysbench_0shot_gen.SysBenchZeroShotGenTask": report_metrics(
            "score",
            "csr",
            "isr",
            "ssr",
            "csr_macro",
            "csr_graded",
            "ungradeable_rate",
            "fails",
            "n_criteria_graded",
            "n_criteria_unaligned",
            "n_criteria_untyped",
            "n_grader_unparsed",
            "n_grader_unparsed_turns",
            "n_sessions",
            "n_turns",
            "n_turns_unaligned",
            patterns=(
                "csr_{alignment}",
                "isr_{alignment}",
                "csr_{alignment}_n_criteria",
                "isr_{alignment}_n_turns",
                "csr_type_{type}",
                "csr_type_{type}_n_criteria",
                "turn_{turn}_csr",
                "turn_{turn}_isr",
                "turn_{turn}_n_turns",
                "turn_{turn}_n_criteria",
                "turn_{turn}_isr_cumulative",
                "turn_{turn}_isr_cumulative_n_sessions",
            ),
        )
        | HEALTH_METRICS,
        f"{_P}.t_eval_before_calling_0shot_gen.TEvalBeforeCallingZeroShotGenTask": (
            report_metrics(
                "thought",
                "name",
                "args_precision",
                "args_recall",
                "args_f1_score",
                "parse_rate",
                "args_precision_parsed",
                "args_recall_parsed",
                "args_f1_score_parsed",
                "fails",
                "n_graded",
                "n_parsed",
            )
        ),
        f"{_P}.theoremqa_kshot_base_gen.TheoremQAKShotBaseGenTask": _ACCURACY
        | report_metrics("empty")
        | _VOTED,
        f"{_P}.ugmathbench_0shot_gen_fixed.UGMathBenchZeroShotGenFixedTask": (
            report_metrics(
                "score",
                "aacc",
                "eacc",
                "cacc",
                "delta",
                "relative_delta",
                "fails",
                "incomplete_problems",
                "n_problems",
                "n_versions",
                "n_versions_judged",
                "unattributed_fails",
                "unattributed_finals",
                patterns=("eacc_{subject}",),
            )
            | _VOTED
            | HEALTH_METRICS
        ),
        f"{_P}.wikisql_0shot_gen.WikiSQLZeroShotGenTask": report_metrics(
            "score",
            "ex_accuracy",
            "lf_accuracy",
            "fails",
            "n",
            "n_execution_errors",
        )
        | HEALTH_METRICS,
    }
)


def task_report_metrics(task_cls: type) -> ReportMetrics | None:
    """*task_cls*'s report metrics: its own declaration, else this table's.

    ``None`` when neither describes the class that owns its ``report``.
    """
    return declared_report_metrics(task_cls, BUILTIN_REPORT_METRICS)
