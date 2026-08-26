# adapted from https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/judges/problem.py
# (Problem.get_prompt) and .../src/process_dataset.py (ParquetReconstructor._write_prompt),
# which build the same string from the same fields — one from a reconstructed
# problem directory, the other from the parquet row.
"""Prompt construction for LiveOIBench.

Divergence from upstream: upstream reads ``statements/statement.md`` and the
``attachments/`` directory off disk, after ``process_dataset.py`` has
materialized them. Here the same values arrive as the parquet fields they were
written from — ``problem_statement`` and the ``starter_code`` bundle (a JSON
map of filename to contents) — so the file-existence tests become key lookups.
The assembled string is unchanged, including the ``time_limit`` /
``memory_limit`` floats rendering as ``1.5`` / ``2048.0``.

Upstream's grader block requires all three of ``grader.cpp`` (or ``stub.cpp``),
``{task}.h`` and ``{task}.cpp``; 17 of the 403 published problems satisfy that,
and the rest take the plain branch.
"""

from typing import Mapping


def build_prompt(
    task: str,
    statement: str,
    time_limit: float | None,
    memory_limit: float | None,
    starter_codes: Mapping[str, str] | None = None,
) -> str:
    """Return the user prompt for one problem.

    *task* is upstream's task token — the ``problem_id`` remainder after
    competition/year/round, which the dataset carries as ``task_name``. It names
    the fence the model is asked to use, so it is also what
    ``CodeExtractor.extract_code`` matches on.
    """
    starter_codes = starter_codes or {}

    prompt = (
        f"Given a competition problem below, write a solution in C++ that solves all the subtasks. "
        f"Make sure to wrap your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
    )
    if statement:
        prompt += "[BEGIN PROBLEM]\n"
        prompt += statement
        prompt += "[END PROBLEM]\n"

    time_text = time_limit if time_limit is not None else "unknown"
    memory_text = memory_limit if memory_limit is not None else "unknown"
    prompt += f"Time limit: {time_text} seconds\n"
    prompt += f"Memory limit: {memory_text} MB\n"

    grader_code = starter_codes.get("grader.cpp")
    if grader_code is None:
        grader_code = starter_codes.get("stub.cpp")
    header_code = starter_codes.get(f"{task}.h")
    solution_code = starter_codes.get(f"{task}.cpp")

    if grader_code is not None and header_code is not None and solution_code is not None:
        prompt += (
            "We are going to grade your solution using the following grader.cpp file and "
            f"{task}.h file. You can write your solution by modifying the {task}.cpp and wrap "
            f"your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
        )
        prompt += "```grader.cpp\n" + grader_code + "```\n\n"
        prompt += f"```{task}.h\n" + header_code + "```\n\n"
        prompt += f"```{task}.cpp\n" + solution_code + "```\n\n"
    else:
        prompt += (
            f"Generate a solution in C++ that solves the task. Make sure to wrap your code in "
            f"'```{task}.cpp' and '```' Markdown delimiters.\n\n"
        )

    return prompt
