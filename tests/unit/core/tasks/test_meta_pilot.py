"""Acceptance test: pilot tasks registered, index round-trips.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from pathlib import Path

from sieval.core.tasks.meta import (
    TASK_REGISTRY,
    import_all_tasks,
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


def test_all_pilot_tasks_registered():
    import_all_tasks()
    registered = set(TASK_REGISTRY.keys())
    missing = PILOT_NAMES - registered
    assert not missing, f"pilot tasks missing from registry: {missing}"


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
