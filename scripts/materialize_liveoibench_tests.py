"""Unpack the LiveOIBench test corpus into per-problem test directories.

The published test data is three parquet files (2023/2024/2025, 33.5 GB total),
each a **single row group** whose ``tests`` column holds every problem's cases as
one JSON blob per row. That layout has no cheap random access: reading one
problem's tests means decompressing the whole column chunk. So the corpus is
unpacked once, into the directory layout upstream's ``process_dataset.py``
produces:

    {out}/{competition}/{year}/{round}/{task}/tests/{case}.in
    {out}/{competition}/{year}/{round}/{task}/tests/{case}.out

Rows stream one at a time, so peak memory is one problem's tests (~83 MB on
average) rather than the file. Expect >30 GB written and a long run; already
materialized problems are skipped unless ``--overwrite`` is given.

Equivalent to upstream's ``process_dataset.py --skip-metadata``, minus the
problem-metadata tree that sieval reads from the parquet directly.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# Import sieval from THIS checkout rather than through the editable install,
# which may point at a different worktree.
sys.path.insert(0, str(ROOT))

from sieval.datasets.liveoibench import (  # noqa: E402
    TEST_YEARS,
    problem_tests_dir,
    year_parquet_name,
)


def materialize_year(
    parquet_path: Path, tests_root: Path, overwrite: bool, verbose: bool = True
) -> tuple[int, int]:
    """Unpack one year file. Returns (problems written, problems skipped)."""
    written = skipped = 0
    parquet_file = pq.ParquetFile(parquet_path)
    # batch_size=1 keeps one problem's payload in memory at a time; the column
    # projection drops `subtasks`, which the dataset loader reads separately.
    for batch in parquet_file.iter_batches(
        batch_size=1, columns=["problem_id", "tests"]
    ):
        for problem_id, payload in zip(
            batch.column("problem_id").to_pylist(),
            batch.column("tests").to_pylist(),
            strict=True,
        ):
            out_dir = Path(problem_tests_dir(str(tests_root), problem_id))
            if out_dir.is_dir() and not overwrite:
                skipped += 1
                continue
            tests = json.loads(payload or "{}")
            if not isinstance(tests, dict):
                raise ValueError(f"Malformed tests payload for {problem_id!r}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, case in tests.items():
                case = case or {}
                (out_dir / f"{name}.in").write_text(
                    case.get("input", ""), encoding="utf-8"
                )
                (out_dir / f"{name}.out").write_text(
                    case.get("output", ""), encoding="utf-8"
                )
            written += 1
            if verbose:
                print(f"  {problem_id}: {len(tests)} cases")
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-repo",
        default=os.path.join(
            os.environ.get("SIEVAL_DATA_DIR", "data"),
            "LiveOIBench",
            "LiveOIBench_tests",
        ),
        help="Directory holding the staged liveoibench_testcases_v1_*.parquet files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write test directories (default: <tests-repo>/materialized, "
        "which is what the dataset loader looks for).",
    )
    parser.add_argument("--years", nargs="*", default=list(TEST_YEARS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tests_repo = Path(args.tests_repo)
    tests_root = Path(args.out) if args.out else tests_repo / "materialized"

    total_written = total_skipped = 0
    for year in args.years:
        parquet_path = tests_repo / year_parquet_name(year)
        if not parquet_path.exists():
            print(
                f"missing {parquet_path}; "
                "run `sieval dataset download liveoibench` first"
            )
            return 1
        print(f"unpacking {parquet_path.name} -> {tests_root}")
        written, skipped = materialize_year(
            parquet_path, tests_root, args.overwrite, verbose=not args.quiet
        )
        print(f"  {year}: {written} written, {skipped} already present")
        total_written += written
        total_skipped += skipped

    print(f"done: {total_written} problems written, {total_skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
