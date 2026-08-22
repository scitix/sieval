"""Unit tests for the LiveOIBench dataset loader.

Covers the three-repo join (problems + subtasks + contestants), the derived
columns the task reads, and the two guards that turn a silent under-run into an
error: a missing test corpus and an empty filter.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sieval.datasets.liveoibench import (
    LiveOIBenchDataset,
    parse_problem_id,
    problem_tests_dir,
    year_parquet_name,
)

BATCH_ID = "IOI-2025-contest-beechtree"
INTERACTIVE_ID = "IOI-2025-contest-longesttrip"
DASHED_ID = "CEOI-2024-contest-grading-server"

SUBTASKS = {"1": {"task": "Subtask 1", "score": 100, "testcases": ["t1"]}}


def _problem(problem_id: str, task_type: str = "batch") -> dict:
    # The year is read out of `problem_id`, the way the loader reads it.
    _, _, _, task = parse_problem_id(problem_id)
    return {
        "id": 0,
        "problem_id": problem_id,
        "competition": problem_id.split("-")[0],
        "contest": "contest",
        "task_name": task,
        "problem_statement": f"# {task}\n",
        "time_limit": 1.5,
        "memory_limit": 2048.0,
        "task_type": task_type,
        "difficulty": 12,
        "algorithms": '["greedy"]',
        "setup_script": "",
        "evaluation_script": "",
        "grader_code": "",
        "starter_code": "",
    }


def _stage(root, problems: list[dict], *, with_tests: bool = True) -> str:
    """Write the three staged repos the way `sieval dataset download` would."""
    problems_dir = root / "LiveOIBench" / "data"
    problems_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(problems), problems_dir / "liveoibench_v1.parquet"
    )

    tests_repo = root / "LiveOIBench_tests"
    tests_repo.mkdir(parents=True, exist_ok=True)
    by_year: dict[str, list[dict]] = {}
    for problem in problems:
        year = parse_problem_id(problem["problem_id"])[1]
        by_year.setdefault(year, []).append(
            {
                "id": 0,
                "problem_id": problem["problem_id"],
                "subtasks": json.dumps(SUBTASKS),
            }
        )
    for year, rows in by_year.items():
        pq.write_table(pa.Table.from_pylist(rows), tests_repo / year_parquet_name(year))

    contestants_dir = root / "LiveOIBench_contestants" / "data"
    contestants_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contest_id": "IOI-2025-contest",
                    "gold_cutoff": 90.0,
                    "silver_cutoff": 60.0,
                    "bronze_cutoff": 30.0,
                    "division": "Division 1",
                    "contestants_ranking": json.dumps([{"Rank": 1, "beechtree": 100}]),
                    "date": "2025-08-01",
                    "num_contestants": 1,
                    "problems": json.dumps([BATCH_ID]),
                }
            ]
        ),
        contestants_dir / "contest_results.parquet",
    )

    tests_root = tests_repo / "materialized"
    if with_tests:
        for problem in problems:
            case_dir = problem_tests_dir(str(tests_root), problem["problem_id"])
            os.makedirs(case_dir, exist_ok=True)
            with open(os.path.join(case_dir, "t1.in"), "w") as f:
                f.write("1 2\n")
            with open(os.path.join(case_dir, "t1.out"), "w") as f:
                f.write("3\n")
    return str(root / "LiveOIBench")


# --------------------------------------------------------------------------- #
# Identifier parsing
# --------------------------------------------------------------------------- #
def test_the_task_token_keeps_its_dashes():
    # `task_name` is everything after competition/year/round, so a hyphenated
    # task name must not be truncated -- it names the prompt's code fence.
    assert parse_problem_id(DASHED_ID) == ("CEOI", "2024", "contest", "grading-server")


def test_a_malformed_identifier_raises():
    with pytest.raises(ValueError, match="expected at least 4 tokens"):
        parse_problem_id("IOI-2025-contest")


def test_tests_live_in_upstreams_reconstructed_layout():
    # Matching upstream means an existing `process_dataset.py` tree can be reused.
    assert problem_tests_dir("/root", BATCH_ID) == (
        "/root/IOI/2025/contest/beechtree/tests"
    )


# --------------------------------------------------------------------------- #
# Loading and joining
# --------------------------------------------------------------------------- #
def test_load_joins_subtasks_and_derives_the_columns_the_task_reads(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)])
    dataset = LiveOIBenchDataset(name_or_path=path)

    (row,) = list(dataset.test_set)
    assert json.loads(row["subtasks"]) == SUBTASKS
    assert row["contest_id"] == "IOI-2025-contest"
    assert row["tests_dir"].endswith("IOI/2025/contest/beechtree/tests")
    assert row["task_name"] == "beechtree"


def test_the_contestant_parquet_is_reachable_from_the_dataset(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)])
    dataset = LiveOIBenchDataset(name_or_path=path)
    assert dataset.contestants_path is not None
    assert os.path.exists(dataset.contestants_path)


def test_interactive_problems_are_filtered_out_by_default(tmp_path):
    # Filtered rather than scored zero: the evaluator has no interactor process,
    # and a zero would understate every model without saying so.
    path = _stage(
        tmp_path,
        [_problem(BATCH_ID), _problem(INTERACTIVE_ID, task_type="interactive")],
    )
    assert [
        r["problem_id"] for r in LiveOIBenchDataset(name_or_path=path).test_set
    ] == [BATCH_ID]
    both = LiveOIBenchDataset(name_or_path=path, task_type=None).test_set
    assert len(both) == 2


def test_year_and_competition_filters_apply(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID), _problem(DASHED_ID)])
    by_year = LiveOIBenchDataset(name_or_path=path, year="2024").test_set
    assert [r["problem_id"] for r in by_year] == [DASHED_ID]
    by_competition = LiveOIBenchDataset(name_or_path=path, competition="IOI").test_set
    assert [r["problem_id"] for r in by_competition] == [BATCH_ID]


def test_a_filter_matching_nothing_raises_instead_of_scoring_an_empty_set(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)])
    with pytest.raises(ValueError, match="empty after filtering"):
        LiveOIBenchDataset(name_or_path=path, year="1999")


def test_a_missing_test_corpus_names_the_command_that_builds_it(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)], with_tests=False)
    with pytest.raises(FileNotFoundError, match="materialize_liveoibench_tests"):
        LiveOIBenchDataset(name_or_path=path)


def test_prompts_can_be_loaded_without_the_corpus(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)], with_tests=False)
    dataset = LiveOIBenchDataset(name_or_path=path, require_tests=False)
    assert len(dataset.test_set) == 1


def test_an_external_tests_root_is_honoured(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)], with_tests=False)
    external = tmp_path / "upstream_root"
    case_dir = problem_tests_dir(str(external), BATCH_ID)
    os.makedirs(case_dir)
    dataset = LiveOIBenchDataset(name_or_path=path, tests_root=str(external))
    assert list(dataset.test_set)[0]["tests_dir"] == case_dir


def test_a_problem_absent_from_the_test_parquet_is_a_loud_join_failure(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)])
    # A second problem in the problems parquet only -- the two repos out of step.
    problems_dir = tmp_path / "LiveOIBench" / "data"
    pq.write_table(
        pa.Table.from_pylist([_problem(BATCH_ID), _problem(DASHED_ID)]),
        problems_dir / "liveoibench_v1.parquet",
    )
    with pytest.raises(ValueError, match="out of step"):
        LiveOIBenchDataset(name_or_path=path)


def test_a_missing_year_parquet_names_the_download_command(tmp_path):
    path = _stage(tmp_path, [_problem(BATCH_ID)])
    os.remove(tmp_path / "LiveOIBench_tests" / year_parquet_name("2025"))
    with pytest.raises(FileNotFoundError, match="sieval dataset download"):
        LiveOIBenchDataset(name_or_path=path)
