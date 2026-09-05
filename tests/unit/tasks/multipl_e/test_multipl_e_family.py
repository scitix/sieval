"""One contract over the four MultiPL-E leaves.

The leaves are near-clones: each pairs a suite with a protocol and adds
nothing else, so what is worth asserting is that the pairing is wired right in
all four rather than re-testing behaviour per leaf.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.core.tasks.meta import get_task_class, get_task_meta
from sieval.tasks.multipl_e._base import MultiPLEChatTask, MultiPLECompletionTask

CASES = [
    # (task name, suite, eval_source, model_type, protocol base, dataset)
    (
        "multipl_e_humaneval_0shot_base_gen",
        "humaneval",
        "human-eval",
        "gen",
        MultiPLECompletionTask,
        "multipl_e_humaneval",
    ),
    (
        "multipl_e_humaneval_0shot_gen",
        "humaneval",
        "human-eval",
        "chat",
        MultiPLEChatTask,
        "multipl_e_humaneval",
    ),
    (
        "multipl_e_mbpp_0shot_base_gen",
        "mbpp",
        "mbpp",
        "gen",
        MultiPLECompletionTask,
        "multipl_e_mbpp",
    ),
    (
        "multipl_e_mbpp_0shot_gen",
        "mbpp",
        "mbpp",
        "chat",
        MultiPLEChatTask,
        "multipl_e_mbpp",
    ),
]

NAMES = [case[0] for case in CASES]


@pytest.mark.parametrize(
    ("name", "suite", "eval_source", "model_type", "protocol", "dataset"),
    CASES,
    ids=NAMES,
)
def test_leaf_pairs_a_suite_with_a_protocol(
    name, suite, eval_source, model_type, protocol, dataset
):
    # Resolution by NAME, not by import: a nested task that imports fine but is
    # unresolvable shows up as a bare KeyError from `sieval task show` and
    # nothing else.
    cls = get_task_class(name)
    meta = get_task_meta(cls)

    assert issubclass(cls, protocol)
    assert cls.suite == suite
    assert cls.eval_source == eval_source
    assert meta.model_type == model_type
    # The FK is resolved from the first generic arg, through a PEP-695 generic
    # base -- so a leaf that forgot to parameterize would bind the wrong data.
    assert meta.dataset == dataset


@pytest.mark.parametrize("name", NAMES)
def test_leaf_declares_a_procedure_reference_and_experimental_status(name):
    meta = get_task_meta(get_task_class(name))
    # The ground truth is a test suite, so no reference value is recorded.
    assert meta.reference_kind == "procedure"
    # No published-score alignment run has been made yet.
    assert meta.status == "experimental"
    assert meta.n_shot == 0


def _reference(name):
    """The leaf's ReferenceImpl, asserted present rather than assumed.

    Every one of these tasks declares one, so a missing block is a regression
    worth failing on directly instead of an attribute error further down.
    """
    reference = get_task_meta(get_task_class(name)).reference_impl
    assert reference is not None
    return reference


@pytest.mark.parametrize("name", NAMES)
def test_leaf_notes_carry_the_protocol_and_its_traps(name):
    reference = _reference(name)
    notes = reference.notes
    # The upstream repeat protocol differs from this task's default n, so the
    # notes have to say how to match it (.claude/rules/tasks.md).
    assert "args.n: 20" in notes
    assert "temperature" in notes
    # Which languages were actually proven, rather than a bare claim of support.
    assert "cpp / js / sh / pl" in notes
    assert reference.url.startswith("https://github.com/nuprl/MultiPL-E/blob/")
    # Commit-pinned, not a branch link.
    assert "/blob/main/" not in reference.url


@pytest.mark.parametrize("name", NAMES)
def test_chat_leaves_document_the_blank_prompt_rule(name):
    meta = get_task_meta(get_task_class(name))
    if meta.model_type != "chat":
        return
    notes = _reference(name).notes
    # The single most consequential detail of the chat protocol: a port that
    # misses it scores zero across the benchmark.
    assert "blank prompt" in notes
    assert "define the function twice" in notes


MBPP_NAMES = ["multipl_e_mbpp_0shot_base_gen", "multipl_e_mbpp_0shot_gen"]


@pytest.mark.parametrize("name", MBPP_NAMES)
def test_mbpp_leaves_record_the_suites_own_quirks(name):
    notes = _reference(name).notes
    assert "no `mbpp-dart` config" in notes
    # Upstream's translator rewrote "python" inside the docstrings too.
    assert "cppthon" in notes
