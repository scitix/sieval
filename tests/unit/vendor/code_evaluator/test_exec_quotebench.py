"""Tests for the code-evaluator's quotebench source.

These live in-tree rather than upstream: the patch is ours to keep by decision
(see `vendor/code-evaluator/VENDORED.md`), not staged for `scitix/code-evaluator`.

Only `app.exec_quotebench` is imported, never `app.server` -- the latter needs
fastapi, which is the evaluator service's dependency and not sieval's. The
dependency chain here is stdlib plus the vendored quotebench package.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sys
from pathlib import Path

import pytest

# vendor/code-evaluator is a service root, not an installed package — put it on
# sys.path so `app` and `quotebench` resolve the way they do inside the service.
_EVALUATOR_DIR = str(
    Path(__file__).resolve().parents[4] / "vendor" / "code-evaluator"
)
if _EVALUATOR_DIR not in sys.path:
    sys.path.insert(0, _EVALUATOR_DIR)

from app.exec_quotebench import (  # noqa: E402  # type: ignore[unresolved-import]  # vendor/code-evaluator added to sys.path at runtime
    CONTRACT_TRANSPORTS,
    execute_quotebench,
    scenarios_digest,
    transport,
)


def test_raw_transport_is_the_reply_verbatim() -> None:
    reply = "printf '%s' 'a b' > out.txt"
    assert transport("raw", reply) == reply


def test_nested_transport_adds_one_double_quoted_layer() -> None:
    assert transport("nested", "echo hi") == 'bash -c "echo hi"'


def test_nested_transport_does_not_escape_the_reply() -> None:
    """The unescaped interpolation IS the benchmark's subject: a reply carrying
    a double quote is what the added boundary breaks. Escaping here would
    silently repair the very failure being measured."""
    assert transport("nested", 'echo "x"') == 'bash -c "echo "x""'


def test_released_rollout_spelling_is_what_we_accept() -> None:
    """Upstream's own public_cli raises ValueError on `nested`, the spelling its
    released dataset ships, so it cannot score its own release. Ours accepts it
    and does not accept the CLI-only spelling."""
    assert set(CONTRACT_TRANSPORTS) == {"raw", "nested"}


@pytest.mark.parametrize("contract", ["nested-shell", "nested-shell-v2", "native", ""])
def test_unknown_contract_raises(contract: str) -> None:
    with pytest.raises(ValueError, match="unknown contract"):
        transport(contract, "echo hi")


def test_digest_is_a_stable_sha256_over_three_modules() -> None:
    first = scenarios_digest()
    assert first == scenarios_digest()
    assert len(first) == 64
    int(first, 16)  # raises if not hex


def test_digest_matches_a_hand_rolled_hash_of_the_same_bytes() -> None:
    """Pin the module set, not just the stability: adding harness.py would make
    the digest rotate on every execution-side change and defeat the guard."""
    import hashlib

    import quotebench  # type: ignore[unresolved-import]

    root = Path(quotebench.__file__).parent
    expected = hashlib.sha256()
    for name in ("core.py", "scenarios.py", "shellesc.py"):
        expected.update((root / name).read_bytes())
    assert scenarios_digest() == expected.hexdigest()


def test_the_two_vendored_copies_are_byte_identical() -> None:
    """The property the digest echo enforces at runtime, asserted at build time.

    sieval's `community/quotebench/` builds the prompts and this copy grades
    them. Each is pinned to upstream by its own hash test, so equality follows —
    but stating it directly is what fails loudly if one copy is re-vendored
    alone.
    """
    import quotebench  # type: ignore[unresolved-import]

    import sieval.community.quotebench as community

    assert community.__file__ is not None
    evaluator_root = Path(quotebench.__file__).parent
    community_root = Path(community.__file__).parent
    for name in ("core.py", "scenarios.py", "shellesc.py"):
        assert (evaluator_root / name).read_bytes() == (
            community_root / name
        ).read_bytes(), f"{name} differs between the two vendored copies"


def test_unknown_task_id_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        execute_quotebench(
            task_id="no-such/task", contract="raw", reply=":", executor="local"
        )


def test_oracle_passes_and_a_no_op_does_not() -> None:
    """The causal check, on one task: the verdict has to read the command.

    A valid no-op exits 0 and touches nothing, so a metric decoupled from the
    prediction would pass it.
    """
    from quotebench.scenarios import get_task  # type: ignore[unresolved-import]

    task = get_task("write-file/t0-plain")
    passed, _, cls, _, _ = execute_quotebench(
        task_id=task.task_id, contract="raw", reply=task.oracle, executor="local"
    )
    assert (passed, cls) == (True, "pass")

    passed, _, cls, _, _ = execute_quotebench(
        task_id=task.task_id, contract="raw", reply=":", executor="local"
    )
    assert (passed, cls) == (False, "silent-wrong")


def test_all_56_oracles_pass_through_this_entry_point() -> None:
    from quotebench.scenarios import all_tasks  # type: ignore[unresolved-import]

    failures = [
        task.task_id
        for task in all_tasks()
        if not execute_quotebench(
            task_id=task.task_id,
            contract="raw",
            reply=task.oracle,
            executor="local",
        )[0]
    ]
    assert failures == []
