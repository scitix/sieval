"""Acceptance test: pilot tasks registered, index round-trips.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from pathlib import Path

from sieval.core.models.requirements import (
    InputKind,
    InputModality,
    NamedModelBinding,
    RequirementContext,
)
from sieval.core.tasks.meta import (
    TASK_REGISTRY,
    import_all_tasks,
    iter_task_entries,
    task_meta_to_dict,
)

PILOT_NAMES = {
    "aime_2024_0shot_gen",
    "aime_2025_0shot_gen",
    "math_500_0shot_gen",
    "drop_kshot_gen",
    "gpqa_diamond_0shot_gen",
    "human_eval_0shot_gen",
    "livecodebench_code_generation_0shot_gen",
    "mmlu_0shot_gen",
    "mmlu_pro_0shot_gen",
    "ifeval_0shot_gen",
    "t_eval_before_calling_0shot_gen",
}

JUDGE_TASK_NAMES = {
    "aa_lcr_0shot_gen",
    "advanced_if_0shot_gen",
    "browsecomp_0shot_gen",
    "complex_constraints_0shot_gen",
    "hle_0shot_gen",
    "inverse_ifeval_0shot_gen",
    "simpleqa_verified_0shot_gen",
    "sysbench_0shot_gen",
}


def test_all_pilot_tasks_registered():
    import_all_tasks()
    registered = set(TASK_REGISTRY.keys())
    missing = PILOT_NAMES - registered
    assert not missing, f"pilot tasks missing from registry: {missing}"


def test_registered_model_tasks_have_composed_input_and_modality_requirements():
    """Golden: every model-calling catalog entry has an executable input shape."""

    import_all_tasks()
    missing: list[str] = []
    mismatched: list[str] = []
    for task_cls, meta in iter_task_entries():
        requires = task_cls.requires
        if requires.input is None or not requires.input_modalities:
            missing.append(meta.name)
            continue
        expected = InputKind.CHAT if meta.model_type == "chat" else InputKind.COMPLETION
        if requires.input is not expected:
            mismatched.append(meta.name)

    assert not missing, f"tasks missing composed modality requirements: {missing}"
    assert not mismatched, f"tasks with mismatched input requirements: {mismatched}"


def test_registered_ppl_and_clp_tasks_declare_scoring_semantics():
    import_all_tasks()
    for task_cls, meta in iter_task_entries():
        if meta.eval_mode.value == "ppl":
            assert task_cls.requires.input_scoring, meta.name
            assert not task_cls.requires.sampled_logprobs, meta.name
        elif meta.eval_mode.value == "clp":
            assert task_cls.requires.sampled_logprobs, meta.name
            assert task_cls.requires.min_top_logprobs is not None, meta.name


def test_registered_clp_hooks_use_normalized_logprobs_task_arg():
    import_all_tasks()
    binding = NamedModelBinding("candidate", "root", "model", "config")
    context = RequirementContext(
        model_bindings={"candidate": binding}, task_args={"logprobs": 37}
    )

    for task_cls, meta in iter_task_entries():
        if meta.eval_mode.value != "clp":
            continue
        (requirement,) = task_cls.model_requirements_for(context)
        assert requirement.requires.min_top_logprobs == 37, meta.name
        assert requirement.source_task == meta.name


def test_llm_judge_tasks_expose_grader_binding_before_construction():
    import_all_tasks()
    candidate = NamedModelBinding("candidate", "candidate-root", "model", "model")
    grader = NamedModelBinding("grader", "grader-root", "judge", "judge")
    context = RequirementContext(
        model_bindings={"candidate": candidate, "grader": grader}
    )

    seen: set[str] = set()
    for task_cls, meta in iter_task_entries():
        if meta.name not in JUDGE_TASK_NAMES:
            continue
        seen.add(meta.name)
        requirements = task_cls.model_requirements_for(context)
        assert {item.role for item in requirements} == {"candidate", "grader"}
        grader_requirement = next(
            item for item in requirements if item.role == "grader"
        )
        assert grader_requirement.requires.input is InputKind.CHAT
        assert grader_requirement.binding is grader

    assert seen == JUDGE_TASK_NAMES


def test_constructor_model_roles_have_matching_hooks_and_injection_support():
    """Constructor roles independently agree with pre-construction hooks."""
    from inspect import signature

    from sieval.cli.resolution import TASK_MODEL_ROLES

    import_all_tasks()
    known_roles = frozenset(TASK_MODEL_ROLES)
    expected = {
        **{name: frozenset({"grader"}) for name in JUDGE_TASK_NAMES},
        "agieval_0shot_gen": frozenset({"extractor"}),
    }
    observed: dict[str, frozenset[str]] = {}

    for task_cls, meta in iter_task_entries():
        parameters = signature(task_cls.__init__).parameters
        constructor_roles = frozenset(parameters) & known_roles
        if not constructor_roles:
            continue

        observed[meta.name] = constructor_roles
        assert "models_by_role" in parameters, (
            f"{meta.name} accepts model role(s) {sorted(constructor_roles)} but "
            "cannot receive normalized models_by_role"
        )

        bindings = {
            "candidate": NamedModelBinding(
                "candidate", "candidate-root", "model", "model"
            ),
            **{
                role: NamedModelBinding(
                    role, f"{role}-root", f"{role}-model", f"{role}-config"
                )
                for role in constructor_roles
            },
        }
        requirements = task_cls.model_requirements_for(
            RequirementContext(model_bindings=bindings)
        )
        requirements_by_role = {item.role: item for item in requirements}
        hook_roles = frozenset(requirements_by_role) & known_roles

        assert hook_roles == constructor_roles, (
            f"{meta.name} constructor roles {sorted(constructor_roles)} disagree "
            f"with hook roles {sorted(hook_roles)}"
        )
        for role in constructor_roles:
            requirement = requirements_by_role[role]
            assert requirement.binding is bindings[role], (meta.name, role)
            assert requirement.requires.input is InputKind.CHAT, (meta.name, role)

    assert observed == expected


def test_hle_full_dataset_adds_candidate_image_modality_requirement():
    import_all_tasks()
    task_cls = next(
        cls for cls, meta in iter_task_entries() if meta.name == "hle_0shot_gen"
    )
    context = RequirementContext(
        model_bindings={
            "candidate": NamedModelBinding(
                "candidate", "candidate-root", "model", "model"
            ),
            "grader": NamedModelBinding("grader", "grader-root", "judge", "judge"),
        },
        dataset_config={"text_only": False},
    )

    requirements = task_cls.model_requirements_for(context)
    candidate = next(item for item in requirements if item.role == "candidate")
    assert candidate.requires.input_modalities == {
        InputModality.TEXT,
        InputModality.IMAGE,
    }


def test_index_json_matches_current_registry():
    """Committed index.json must match fresh serialization of both registries."""
    from sieval.core.datasets.meta import (
        DATASET_REGISTRY,
        dataset_meta_to_dict,
        import_all_datasets,
    )

    import_all_datasets()
    import_all_tasks()
    fresh_datasets = sorted(
        (dataset_meta_to_dict(m) for m in DATASET_REGISTRY.values()),
        key=lambda d: d["name"],
    )
    fresh_tasks = sorted(
        (task_meta_to_dict(m) for m in TASK_REGISTRY.values()),
        key=lambda d: d["name"],
    )
    fresh_payload = {
        "schema_version": 1,
        "datasets": fresh_datasets,
        "tasks": fresh_tasks,
    }

    index_path = Path(__file__).parents[4] / "sieval" / "meta" / "index.json"
    committed = json.loads(index_path.read_text())

    assert committed == fresh_payload, (
        "sieval/meta/index.json is stale; "
        "run 'python scripts/sync_meta_index.py' and commit."
    )


def test_pilot_tasks_retain_protocol_tags_and_model_type():
    """Migrated pilot tasks must still expose Task.tags (protocol) and
    Task.model_type as class attributes — runner.py and session.py depend on
    these.
    """
    from sieval.tasks.aime_2024_0shot_gen import AIME2024ZeroShotGenTask
    from sieval.tasks.drop_kshot_gen import DROPFewShotGenTask
    from sieval.tasks.mmlu_0shot_gen import MMLUZeroShotGenTask

    assert AIME2024ZeroShotGenTask.tags == frozenset({"gen", "zero_shot"})
    assert AIME2024ZeroShotGenTask.model_type == "chat"

    assert DROPFewShotGenTask.tags == frozenset({"gen", "few_shot"})
    assert DROPFewShotGenTask.model_type == "chat"

    assert MMLUZeroShotGenTask.tags == frozenset({"gen", "zero_shot"})
    assert MMLUZeroShotGenTask.model_type == "chat"


def test_pilot_tasks_have_dataset_fk_resolving_to_registered_dataset():
    from sieval.core.datasets.meta import DATASET_REGISTRY, import_all_datasets

    import_all_datasets()
    import_all_tasks()

    expected = {
        "aime_2024_0shot_gen": "aime_2024",
        "aime_2025_0shot_gen": "aime_2025",
        "math_500_0shot_gen": "math_500",
        "drop_kshot_gen": "drop",
        "gpqa_diamond_0shot_gen": "gpqa_diamond",
        "human_eval_0shot_gen": "human_eval",
        "livecodebench_code_generation_0shot_gen": "livecodebench_code_generation",
        "mmlu_0shot_gen": "mmlu",
        "mmlu_pro_0shot_gen": "mmlu_pro",
        "ifeval_0shot_gen": "ifeval",
        "t_eval_before_calling_0shot_gen": "t_eval_before_calling",
    }
    for task_name, expected_dataset in expected.items():
        meta = TASK_REGISTRY[task_name]
        assert meta.dataset == expected_dataset, (
            f"{task_name}: expected {expected_dataset}, got {meta.dataset}"
        )
        assert expected_dataset in DATASET_REGISTRY, (
            f"dataset {expected_dataset} not in registry"
        )
