"""Unit tests for LiveOIBench prompt construction.

The prompt is the benchmark's input, so these pin the exact strings upstream
assembles -- including the two things a tidy-up would change first: the limits
rendering as the floats the parquet carries, and the grader block appearing only
when all three attachment files are present.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

from sieval.community.liveoibench.prompts import build_prompt

STATEMENT = "# Beech Tree\n\nGiven $N$ nodes...\n"


def test_the_fence_is_named_after_the_task():
    prompt = build_prompt("beechtree", STATEMENT, 1.5, 2048.0)
    assert prompt.startswith(
        "Given a competition problem below, write a solution in C++ that solves "
        "all the subtasks. Make sure to wrap your code in '```beechtree.cpp' and "
        "'```' Markdown delimiters.\n\n"
    )


def test_the_statement_is_delimited_verbatim():
    prompt = build_prompt("t", STATEMENT, 1.0, 256.0)
    assert f"[BEGIN PROBLEM]\n{STATEMENT}[END PROBLEM]\n" in prompt


def test_limits_render_as_the_floats_the_dataset_carries():
    # Upstream interpolates the parquet value straight in, so 2048.0 -- not 2048
    # -- is what every published run showed the model.
    prompt = build_prompt("t", STATEMENT, 1.5, 2048.0)
    assert "Time limit: 1.5 seconds\nMemory limit: 2048.0 MB\n" in prompt


def test_absent_limits_render_as_unknown():
    prompt = build_prompt("t", STATEMENT, None, None)
    assert "Time limit: unknown seconds\nMemory limit: unknown MB\n" in prompt


def test_without_attachments_the_plain_instruction_is_appended():
    prompt = build_prompt("t", STATEMENT, 1.0, 256.0)
    assert prompt.endswith(
        "Generate a solution in C++ that solves the task. Make sure to wrap your "
        "code in '```t.cpp' and '```' Markdown delimiters.\n\n"
    )
    assert "grader.cpp" not in prompt


def test_the_grader_block_needs_all_three_files():
    partial = {"grader.cpp": "int main(){}\n", "t.h": "void f();\n"}  # no t.cpp
    assert "```grader.cpp" not in build_prompt("t", STATEMENT, 1.0, 256.0, partial)

    whole = partial | {"t.cpp": "void f(){}\n"}
    prompt = build_prompt("t", STATEMENT, 1.0, 256.0, whole)
    assert (
        "We are going to grade your solution using the following grader.cpp" in prompt
    )
    assert "```grader.cpp\nint main(){}\n```\n\n" in prompt
    assert "```t.h\nvoid f();\n```\n\n" in prompt
    assert "```t.cpp\nvoid f(){}\n```\n\n" in prompt


def test_stub_cpp_stands_in_for_grader_cpp():
    # Upstream falls back to `stub.cpp` when a problem ships no `grader.cpp`, and
    # still labels the block `grader.cpp`.
    bundle = {
        "stub.cpp": "int main(){}\n",
        "t.h": "void f();\n",
        "t.cpp": "void f(){}\n",
    }
    prompt = build_prompt("t", STATEMENT, 1.0, 256.0, bundle)
    assert "```grader.cpp\nint main(){}\n```\n\n" in prompt


def test_a_task_name_with_a_dash_survives_into_the_fence():
    # `CEOI-2023-contest-grading-server` -> task `grading-server`.
    prompt = build_prompt("grading-server", STATEMENT, 1.0, 256.0)
    assert "'```grading-server.cpp'" in prompt
