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
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = Path(__file__).with_name("_import_probe.py")
#: Repo root: <root>/tests/unit/tasks/<this file>.
_ROOT = Path(__file__).parents[3]

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
    # Both Spider graders parse SQL, and the two parsers pull the `spider`
    # group: the vendored `process_sql` imports nltk, and the test-suite
    # evaluator's `parse` imports sqlparse. The prompt path needs neither, which
    # is why the hardened connection lives in `sieval.tasks._sqlite_exec` --
    # importing it from `_spider_exec` would put both back at module scope.
    # Keyed by module path under `sieval.tasks.`, so a task in a subpackage
    # spells the subpackage too -- the manifest builds `sieval.tasks.<key>`.
    "spider.spider_0shot_gen": (
        "sieval.community.spider",
        "sieval.community.spider_test_suite",
        "sieval.tasks.spider._spider_exec",
        "sieval.tasks.spider._spider_test_suite",
        "nltk",
        "sqlparse",
    ),
    # Spider 2.0's vendored evaluator reaches google.cloud at module scope, and
    # importing it has side effects on top of that -- upstream's `evaluate.py`
    # hijacks the process's stdout/stderr and truncates `log.txt` in the current
    # directory, which the package wrapper undoes. Registration must pay neither,
    # so `extract_sql_query` and the comparison are imported inside the stages
    # that use them. (`pandas` is deliberately not on this list: the *loader*
    # imports HuggingFace `datasets`, which pulls it in regardless, so naming it
    # here would fail for a reason that has nothing to do with the grader.)
    "spider2_lite_0shot_gen": (
        "sieval.community.spider2",
        "google.cloud.bigquery",
        "snowflake.connector",
    ),
    # The embedding backend is only needed when eval_thought is enabled.
    "t_eval_before_calling_0shot_gen": ("sentence_transformers",),
    # latex2sympy2 is behind the `math` group.
    "theoremqa_kshot_base_gen": ("latex2sympy2", "latex2sympy2_extended"),
    # The vendored engine reaches babel (upstream's `parse_decimal`) at module
    # scope, so the whole package is forbidden and not just the third-party name.
    "wikisql_0shot_gen": ("sieval.community.wikisql", "babel"),
}


@pytest.fixture(scope="session")
def import_probe() -> dict[str, dict]:
    """Run the whole manifest through one fresh interpreter.

    The child gets ``PYTHONPATH`` pointed at the tree this test was loaded from,
    prepended so it beats the editable install. Without it the child resolves
    `sieval` through that install — i.e. to whichever checkout `pip install -e`
    happened to name — so running the suite from a **git worktree** probes the
    *other* tree. A task the worktree adds is then reported as
    ``ModuleNotFoundError``, which this file's assertion cannot tell apart from a
    genuine import-discipline violation: the same red, for a task whose imports
    are fine. CI never saw it, since CI runs from a normal checkout.
    """
    manifest = {f"sieval.tasks.{task}": list(deps) for task, deps in FORBIDDEN.items()}
    inherited = os.environ.get("PYTHONPATH")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(_ROOT), *([inherited] if inherited else [])]
        ),
    }
    completed = subprocess.run(
        [sys.executable, str(_PROBE)],
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
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
