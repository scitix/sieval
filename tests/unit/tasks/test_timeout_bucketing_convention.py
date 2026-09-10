"""No task decides a timeout by substring-matching the bare word.

The defect is argued in `sieval/tasks/_code_eval_msg.py`: a code-eval message
interpolates the failing program's own output, so `"timeout" in msg` counts a
`[TimeoutError]`, a compiler diagnostic naming an identifier, and any comparison
failure printing the word — all as the service having stopped a clock.

Six tasks carried that test independently, which is why this is a survey and not
a list. Nothing the six scored was wrong (`correct` comes from the service's
boolean), so no metric test would ever have caught the drift, and the next task
copying an older template back in would re-introduce it silently.

Deliberately narrow: it forbids the BARE literal only. `"[timeouterror]" in msg`
is a different test with a different meaning — an exception class name in the
message tail, the way `memoryerror` is read next to it in `scicode` — and stays
allowed.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import ast
import pathlib

import sieval

#: Resolved from the imported package, so a worktree cannot survey the primary
#: checkout's `sieval/` by accident.
TASKS_DIR = pathlib.Path(sieval.__file__).parent / "tasks"

#: A floor, not a count: it needs raising only when readers are *removed*, and it
#: is what stops a broken scan from passing as "no offenders found".
MIN_EXPECTED_READERS = 6


def _task_sources() -> list[pathlib.Path]:
    return sorted(TASKS_DIR.rglob("*.py"))


def test_no_task_substring_matches_the_bare_word():
    offenders: list[str] = []
    for path in _task_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for op, left in zip(node.ops, [node.left, *node.comparators], strict=False):
                if not isinstance(op, ast.In):
                    continue
                if (
                    isinstance(left, ast.Constant)
                    and isinstance(left.value, str)
                    and left.value.strip().lower() == "timeout"
                ):
                    offenders.append(
                        f"{path.relative_to(TASKS_DIR.parent)}:{node.lineno}"
                    )
    assert offenders == [], (
        "these sites decide a timeout by substring-matching the bare word; use "
        "`_code_eval_msg.is_timeout_message`, which tests the message PREFIX: "
        + ", ".join(offenders)
    )


def test_the_shared_predicate_actually_has_readers():
    """Guards the other direction: a correct helper nobody calls fixes nothing.

    Counts CALL nodes, not occurrences of the name: a file whose only mention is
    the `import` line reads the predicate nowhere, and grepping the text would
    score that as a reader. Verified by reverse-mutation — replacing a call while
    leaving its import behind is exactly the shape that got past the first
    version of this test.
    """
    readers: list[str] = []
    for path in _task_sources():
        if path.name == "_code_eval_msg.py":
            continue
        tree = ast.parse(path.read_text())
        calls = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "is_timeout_message"
        )
        if calls:
            readers.append(f"{path.relative_to(TASKS_DIR.parent).as_posix()} x{calls}")
    assert len(readers) >= MIN_EXPECTED_READERS, (
        f"expected at least {MIN_EXPECTED_READERS} tasks to CALL the shared "
        f"predicate, found {len(readers)}: {readers}"
    )
