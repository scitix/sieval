"""One import-discipline contract, over every task with an optional grader.

Importing a task module registers it, and registration is paid by things that
never grade anything: `sieval task list`, the meta index, and any eval that
fails before `feedback()`. So a grading dependency behind an optional
dependency group must be imported inside `feedback()`, never at module scope —
otherwise the group stops being optional.

The check only means anything in an interpreter where the dependency was never
imported, which is why it runs out of process. It runs there **once** for the
whole family rather than once per task: 23 fresh interpreters cost ~60s, and
~2.4s of each was re-importing the same `sieval.tasks` base. `_import_probe.py`
holds that reasoning and the isolation it needs.

Each task still gets its own test, so a failure names the task rather than the
family, and so does each forbidden name — the manifest below is the contract.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = Path(__file__).with_name("_import_probe.py")

# task module (under `sieval.tasks.`) -> modules its registration must not pull.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    # math_verify is behind the `math` group and is slow to import.
    "aime_2024_0shot_gen": ("math_verify",),
    "aime_2025_0shot_gen": ("math_verify",),
    "aime_2026_0shot_gen": ("math_verify",),
    "apex_2025_0shot_gen": ("math_verify",),
    "apex_shortlist_2025_0shot_gen": ("math_verify",),
    "brumo_2025_0shot_gen": ("math_verify",),
    "cmimc_2025_0shot_gen": ("math_verify",),
    "hmmt_feb_2025_0shot_gen": ("math_verify",),
    "hmmt_feb_2026_0shot_gen": ("math_verify",),
    "hmmt_nov_2025_0shot_gen": ("math_verify",),
    "imo_answer_bench_0shot_gen": ("math_verify",),
    "math_500_0shot_gen": ("math_verify",),
    "smt_2025_0shot_gen": ("math_verify",),
    "ugmathbench_0shot_gen_fixed": ("math_verify",),
    # drop_eval pulls scipy.
    "drop_kshot_gen": ("sieval.community.simple_evals.drop_eval",),
    # evaluation_lib pulls the optional IFBench scorers.
    "ifbench_0shot_gen": ("sieval.community.ifbench.evaluation_lib",),
    # The repair module subclasses the vendored checkers and reaches NLTK
    # through them. Its underscore keeps it out of the task *index*, not out of
    # `import_all_tasks`' scan, so it owes the same discipline as a public one.
    "ifbench_0shot_gen_fixed": (
        "sieval.community.ifbench.evaluation_lib",
        "sieval.tasks._ifbench_fixed_checkers",
        "nltk",
    ),
    # evaluation_lib pulls absl/langdetect/nltk.
    "ifeval_0shot_gen": ("sieval.community.instruction_following_eval.evaluation_lib",),
    # multi_if's fork adds emoji and a 3.5k-line checker; `_ensure_punkt_tab`
    # imports nltk when called, so registration must not drag that in either.
    "multi_if_0shot_gen": ("sieval.community.multi_if.evaluation_lib", "nltk"),
    # The `_fixed` repairs additionally reach the vendored checkers + langdetect.
    "ifeval_0shot_gen_fixed": (
        "sieval.community.instruction_following_eval.evaluation_lib",
        "sieval.community.instruction_following_eval_fixed",
        "langdetect",
    ),
    "multi_if_0shot_gen_fixed": (
        "sieval.community.multi_if.evaluation_lib",
        "sieval.community.instruction_following_eval_fixed",
        "langdetect",
        "nltk",
    ),
    # Both IHEval grader modules sit behind the `iheval` group.
    "iheval_0shot_gen": (
        "sieval.community.iheval",
        "sieval.community.instruction_following_eval.evaluation_lib",
    ),
    # The embedding backend is only needed when eval_thought is enabled.
    "t_eval_before_calling_0shot_gen": ("sentence_transformers",),
    # latex2sympy2 is behind the `math` group.
    "theoremqa_kshot_base_gen": ("latex2sympy2", "latex2sympy2_extended"),
}


@pytest.fixture(scope="session")
def import_probe() -> dict[str, dict]:
    """Run the whole manifest through one fresh interpreter."""
    manifest = {f"sieval.tasks.{task}": list(deps) for task, deps in FORBIDDEN.items()}
    completed = subprocess.run(
        [sys.executable, str(_PROBE)],
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"import probe exited {completed.returncode}\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("task", FORBIDDEN)
def test_import_does_not_pull_the_optional_grader(
    task: str, import_probe: dict[str, dict]
) -> None:
    result = import_probe[f"sieval.tasks.{task}"]
    assert "error" not in result, f"importing {task} raised {result.get('error')}"
    assert not result["present"], (
        f"registering {task} imports {', '.join(result['present'])} at module "
        f"scope; move the import inside feedback() so the optional dependency "
        f"group stays optional"
    )
