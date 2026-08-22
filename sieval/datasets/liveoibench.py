"""LiveOIBench dataset — informatics-olympiad problems with official test data.

403 problems from 72 contests across 14 olympiads (2023–2025), published as three
HuggingFace repos that this dataset stages together:

* problems and metadata (920 KB) — statements, limits, grader/starter bundles;
* contestant results (559 KB) — per-contest human rankings and medal cutoffs,
  kept on :attr:`contestants_path` for the task's report to rank against;
* official test cases (**33.5 GB**) — one parquet per year, each a single row
  group, so there is no cheap per-problem read out of them.

Because of that last point the test corpus is **materialized once** into
per-problem directories — the layout upstream's ``process_dataset.py`` produces,
so an existing ``$LIVEOIBENCH_ROOT/data`` tree can be pointed at directly. Rows
carry the path rather than the payload: at ~83 MB of tests per problem, a column
holding them would put the whole corpus through the record pipeline.

The ``subtasks`` column is joined in from the test parquets, where it lives; it
is ~20 KB per year, so column projection reads it without touching ``tests``.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import os
from typing import TypedDict, override

import pyarrow.parquet as pq
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# `load_subtasks` / `load_code_bundle` are ports of upstream's own payload
# decoding and live in community/ next to the rest of it; the task reads the same
# two functions from there rather than reaching into this module.

LIVEOIBENCH_REVISION = "0e87d7c7a667979e00df41889656d73f1d3ce153"
LIVEOIBENCH_TESTS_REVISION = "05731838fc460d3235c70506ee177aeb68a610cf"
LIVEOIBENCH_CONTESTANTS_REVISION = "fdcc2ba59670b06ec55ee61bc771ed855ba6dba6"

PROBLEMS_FILE = os.path.join("data", "liveoibench_v1.parquet")
CONTESTANTS_FILE = os.path.join("data", "contest_results.parquet")
TEST_YEARS = ("2023", "2024", "2025")


def year_parquet_name(year: str) -> str:
    """The test parquet for one year.

    Deliberately not spelled ``test_*``: pytest collects any imported callable
    with that prefix as a test case, so a test module using this helper would
    fail to collect on the helper itself.
    """
    return f"liveoibench_testcases_v1_{year}.parquet"


def parse_problem_id(problem_id: str) -> tuple[str, str, str, str]:
    """Split ``competition-year-round-task``; upstream ``parse_problem_id``.

    The task token is everything after the third segment, so a task name that
    itself contains ``-`` (``grading-server``) survives. It is the same value the
    ``task_name`` column carries, and it is what the prompt's code fence and the
    extractor's first pattern are built from.
    """
    parts = problem_id.split("-")
    if len(parts) < 4:
        raise ValueError(
            f"Invalid problem_id {problem_id!r}; expected at least 4 tokens."
        )
    return parts[0], parts[1], parts[2], "-".join(parts[3:])


def problem_tests_dir(tests_root: str, problem_id: str) -> str:
    """Where one problem's ``.in`` / ``.out`` files live once materialized.

    Upstream's reconstructed layout
    (``{competition}/{year}/{round}/{task}/tests``), so a tree already built by
    ``process_dataset.py`` can be used as *tests_root* unchanged.
    """
    competition, year, round_name, task = parse_problem_id(problem_id)
    return os.path.join(tests_root, competition, year, round_name, task, "tests")


class LiveOIBenchDatasetSample(TypedDict):
    problem_id: str
    competition: str
    contest: str
    task_name: str
    problem_statement: str
    time_limit: float
    memory_limit: float
    task_type: str
    difficulty: int
    algorithms: str
    grader_code: str
    starter_code: str
    # Joined from the test parquet: `{id: {"score": int, "testcases": [name]}}`.
    subtasks: str
    # Derived. `contest_id` is the join key into the contestant table, whose own
    # column is spelled that way; `tests_dir` is where this problem's test files
    # were materialized.
    contest_id: str
    tests_dir: str


@sieval_dataset(
    name="liveoibench",
    display_name="LiveOIBench",
    description=(
        "403 olympiad problems with official tests, subtask rubrics and human "
        "rankings; 33.5 GB of tests."
    ),
    source=(
        f"hf:LiveOIBench/LiveOIBench@{LIVEOIBENCH_REVISION}",
        f"hf:LiveOIBench/LiveOIBench_tests@{LIVEOIBENCH_TESTS_REVISION}",
        f"hf:LiveOIBench/LiveOIBench_contestants@{LIVEOIBENCH_CONTESTANTS_REVISION}",
    ),
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "cpp", "code-exec"),
    license="cc-by-4.0",
)
class LiveOIBenchDataset(Dataset[LiveOIBenchDatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        task_type: str | None = "batch",
        year: str | None = None,
        competition: str | None = None,
        tests_root: str | None = None,
        require_tests: bool = True,
        **kwargs,
    ) -> HFDatasetDict:
        """Load the problems, joined with their subtasks and test locations.

        *task_type* defaults to ``"batch"`` — the 380 problems the evaluator's
        C++ path can grade. The 23 ``interactive`` ones need an interactor
        process it does not run yet, and are filtered out rather than scored
        zero, which would understate every model invisibly. Pass ``None`` to keep
        every problem.

        *tests_root* defaults to a ``tests/`` tree beside the staged parquet
        files; point it at an existing ``$LIVEOIBENCH_ROOT/data`` to reuse one.
        *require_tests* is the guard that turns a missing corpus into an
        actionable error instead of a run that fails 380 times.
        """
        staged_root = os.path.dirname(os.path.normpath(name_or_path))
        problems_path = os.path.join(name_or_path, PROBLEMS_FILE)
        tests_repo = os.path.join(staged_root, "LiveOIBench_tests")
        contestants_repo = os.path.join(staged_root, "LiveOIBench_contestants")

        # Captured for the task, the way SciCode captures its h5 path;
        # copy.copy-based clones (slice/shuffle/...) preserve the attribute.
        self._contestants_path = os.path.join(contestants_repo, CONTESTANTS_FILE)
        self._tests_root = tests_root or os.path.join(tests_repo, "materialized")

        table = pq.read_table(problems_path)
        rows: list[dict] = table.to_pylist()

        subtasks_by_id = _read_subtasks(tests_repo)
        kept: list[dict] = []
        for row in rows:
            problem_id = row["problem_id"]
            if task_type is not None and row["task_type"] != task_type:
                continue
            competition_name, problem_year, round_name, _task = parse_problem_id(
                problem_id
            )
            if year is not None and problem_year != year:
                continue
            if competition is not None and competition_name != competition:
                continue
            subtasks = subtasks_by_id.get(problem_id)
            if subtasks is None:
                # A problem with no subtask rubric has no score to compute; that
                # is a broken join, not an empty result.
                raise ValueError(
                    f"No subtasks found for {problem_id!r} in {tests_repo!r}; "
                    "the problem and test parquets are out of step."
                )
            kept.append(
                row
                | {
                    "subtasks": subtasks,
                    "contest_id": "-".join(
                        (competition_name, problem_year, round_name)
                    ),
                    "tests_dir": problem_tests_dir(self._tests_root, problem_id),
                }
            )

        if not kept:
            raise ValueError(
                "LiveOIBench dataset is empty after filtering "
                f"(task_type={task_type!r}, year={year!r}, "
                f"competition={competition!r})."
            )

        if require_tests:
            missing = [
                row["problem_id"] for row in kept if not os.path.isdir(row["tests_dir"])
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(kept)} problems have no materialized "
                    f"test directory under {self._tests_root!r} "
                    f"(e.g. {missing[0]!r}). The 33.5 GB test corpus ships as "
                    "three single-row-group parquets and has to be unpacked "
                    "once:\n"
                    "    python scripts/materialize_liveoibench_tests.py\n"
                    "Pass tests_root= to reuse a tree built by upstream's "
                    "process_dataset.py, or require_tests=False to load prompts "
                    "without grading."
                )

        return ensure_dataset_dict(HFDatasetDict({"test": HFDataset.from_list(kept)}))

    @property
    def contestants_path(self) -> str | None:
        """Staged path of the contestant-results parquet, for ranking a model.

        ``None`` when the dataset was built from a pre-loaded dict rather than
        via :meth:`load`.
        """
        return getattr(self, "_contestants_path", None)

    @property
    def tests_root(self) -> str | None:
        """Root of the materialized test tree; ``None`` as for
        :attr:`contestants_path`."""
        return getattr(self, "_tests_root", None)


def _read_subtasks(tests_repo: str) -> dict[str, str]:
    """Read every year's ``subtasks`` column, skipping the ``tests`` column.

    Each year file is one row group whose ``tests`` column is 6–17 GB and whose
    ``subtasks`` column is ~20 KB, so projecting to two columns is the difference
    between a 20 KB read and the whole corpus.
    """
    subtasks: dict[str, str] = {}
    for year in TEST_YEARS:
        path = os.path.join(tests_repo, year_parquet_name(year))
        if not os.path.exists(path):
            continue
        table = pq.read_table(path, columns=["problem_id", "subtasks"])
        for problem_id, payload in zip(
            table.column("problem_id").to_pylist(),
            table.column("subtasks").to_pylist(),
            strict=True,
        ):
            subtasks[problem_id] = payload or "{}"
    if not subtasks:
        raise FileNotFoundError(
            f"No LiveOIBench test parquet found under {tests_repo!r}; "
            "run `sieval dataset download liveoibench` first."
        )
    return subtasks
