"""
Unit tests for TheoremQA k-shot base generative task.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import builtins
import importlib
import math
import time

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response, SamplingParams
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import (
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.context import TaskContext
from tests.conftest import HandlerTransport, ModuleIsolation

_TASK_MODULE = "sieval.tasks.theoremqa_kshot_base_gen"
_DATASET_MODULE = "sieval.datasets.theoremqa"
_LAZY_PACKAGES = ("sieval.tasks", "sieval.datasets")


@pytest.fixture(autouse=True)
def _preserve_registries():
    """Clear the four registries and evict both theoremqa modules, then restore
    the whole set — see `ModuleIsolation` for why the two must move together.

    This file imports the task/dataset on demand (`_task_module()`,
    `_dataset()`) rather than at top level, so evicting them is what makes each
    test's import re-execute the module bodies and re-run the `@sieval_task` /
    `@sieval_dataset` decorators into the registries just cleared. Scope is the
    two exact modules rather than the whole packages.
    """
    from sieval.core.datasets.meta import DATASET_REGISTRY, SAMPLE_TO_DATASET
    from sieval.core.tasks.meta import _TASK_CLASSES, TASK_REGISTRY

    task_snapshot = dict(TASK_REGISTRY)
    task_classes_snapshot = dict(_TASK_CLASSES)
    dataset_snapshot = dict(DATASET_REGISTRY)
    sample_map_snapshot = dict(SAMPLE_TO_DATASET)
    modules = ModuleIsolation((_TASK_MODULE, _DATASET_MODULE), _LAZY_PACKAGES)
    modules.snapshot()

    TASK_REGISTRY.clear()
    _TASK_CLASSES.clear()
    DATASET_REGISTRY.clear()
    SAMPLE_TO_DATASET.clear()
    modules.evict()
    try:
        yield
    finally:
        TASK_REGISTRY.clear()
        TASK_REGISTRY.update(task_snapshot)
        _TASK_CLASSES.clear()
        _TASK_CLASSES.update(task_classes_snapshot)
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(dataset_snapshot)
        SAMPLE_TO_DATASET.clear()
        SAMPLE_TO_DATASET.update(sample_map_snapshot)
        modules.restore()


class _MockGenModel(GenModel):
    def __init__(self):
        self.last_req: Request | None = None
        super().__init__(model="mock-gen", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_completions")

    async def _stub_arun(self, req: Request) -> Response:
        if req.scoring.sampled_logprobs or req.scoring.input_scoring:
            raise NotImplementedError  # the gen task never requests logprobs
        self.last_req = req
        return Response(texts=("The answer is 4",))


def _task_module():
    return importlib.import_module(_TASK_MODULE)


def _theoremqa_examples():
    return _task_module()._THEOREMQA_EXAMPLES


def _dataset():
    sample = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    dataset_module = importlib.import_module(_DATASET_MODULE)
    return dataset_module.TheoremQADataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([sample])})
    )


def _task(n_shot: int | None = None):
    task_module = _task_module()
    return task_module.TheoremQAKShotBaseGenTask(
        _dataset(), _MockGenModel(), n_shot=n_shot
    )


@pytest.mark.anyio
async def test_report_divides_metrics_by_completed_finals_only():
    raw = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    task = _task(n_shot=0)
    judgement = build_judgement_record("4", [build_rollout_judgement(0, True)])
    # report() reads `extracted` off the prediction record for its `empty` count,
    # so the context has to carry both records, not just the judgement.
    final_ctx = (
        TaskContext(sample_id=0, raw_sample=raw)
        .to_preprocessed({"prompt": "p"})
        .to_inferred("i")
        .to_postprocessed(build_prediction_record(["4"]))
        .to_feedback(judgement)
    )
    failed_ctx = TaskContext(sample_id=1, raw_sample=raw)

    report = await task.report([final_ctx], [failed_ctx])

    assert report["score"] == 100.0
    assert report["accuracy"] == 100.0
    assert report["fails"] == 1.0
    assert report["empty"] == 0.0


@pytest.mark.anyio
async def test_infer_only_forwards_prompt_coupled_stop():
    task_module = _task_module()
    model = _MockGenModel()
    task = task_module.TheoremQAKShotBaseGenTask(_dataset(), model, n_shot=0)

    await task.infer(
        {"prompt": "prompt"},
        TaskContext(sample_id=0, raw_sample={"Question": "What is 2+2?"}),
    )

    req = model.last_req
    assert req is not None
    # Only the prompt-coupled stop is forwarded — no other sampling params.
    assert req.sampling == SamplingParams(stop=tuple(task_module._STOP_TOKENS))
    assert req.dialect_options is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("n_shot", "expected_examples"),
    [(None, None), (0, 0), (2, 2)],
)
async def test_preprocess_uses_configured_k(n_shot, expected_examples):
    raw = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    task = _task(n_shot=n_shot)
    examples = _theoremqa_examples()
    if expected_examples is None:
        expected_examples = len(examples)

    prompt = (await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw)))[
        "prompt"
    ]

    assert prompt.count("Problem:\n") == expected_examples + 1
    assert prompt.endswith("Problem:\nWhat is 2+2?\nSolution:\n")
    if expected_examples == 0:
        assert examples[0][0] not in prompt
    if expected_examples == 2:
        assert examples[1][0] in prompt
        assert examples[2][0] not in prompt


@pytest.mark.anyio
async def test_preprocess_preserves_official_runtime_prompt_artifacts():
    raw = {"Question": "What is 2+2?", "Answer": "4", "Answer_type": "integer"}
    task = _task(n_shot=3)

    prompt = (await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw)))[
        "prompt"
    ]

    assert "\u2248 833.33 frames" in prompt
    assert "Bytes/frame is approximately 833.33 frames" not in prompt
    upstream_control_line = (
        "Let's calculate the numerical value of "
        "$\\left[\x0crac{10}{3}, \x0crac{4}{3}\x0dight]_C$ "
        "as [3.33, 1.33]."
    )
    assert upstream_control_line in prompt


def test_constructor_rejects_negative_k():
    with pytest.raises(ValueError, match="n_shot must"):
        _task(n_shot=-1)


def test_constructor_rejects_too_many_examples():
    n_shot = len(_theoremqa_examples()) + 1
    with pytest.raises(ValueError, match="n_shot must"):
        _task(n_shot=n_shot)


@pytest.mark.parametrize("n_shot", [True, 1.5, "2"])
def test_constructor_rejects_non_integer_k(n_shot):
    with pytest.raises(TypeError, match="n_shot must be an int"):
        _task(n_shot=n_shot)


# ---------------------------------------------------------------------------
# safe_eval — two properties that pull against each other, which is why both
# are here: it must not execute model output, and it must still agree with the
# bare `eval` upstream uses on every expression a real answer produces.
# ---------------------------------------------------------------------------

#: Upstream's namespace: number_utils.py's module globals *plus* the real
#: builtins, which is what a bare `eval(num)` sees. Omitting "__builtins__" is
#: what restores the latter — Python injects them.
_UPSTREAM_GLOBALS = {
    "math": math,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "log": math.log,
    "pi": math.pi,
    "factorial": math.factorial,
    "exp": math.exp,
    "e": math.e,
    "E": 2.718,
}

#: The payload from the bug report: a walk from a literal back to a live
#: `open` that a cleared `__builtins__` does not stop.
_ESCAPE = (
    "[c for c in ().__class__.__base__.__subclasses__() "
    "if c.__name__=='catch_warnings'][0]()._module.__builtins__['open']"
    "('{path}','w')"
)


def _safe_eval():
    return _task_module().safe_eval


def _unsafe_expression():
    return _task_module().UnsafeExpression


def test_reported_escape_payload_is_refused(tmp_path):
    """The reported RCE is refused, and writes nothing."""
    target = tmp_path / "pwned"
    with pytest.raises(_unsafe_expression()):
        _safe_eval()(_ESCAPE.format(path=target))
    assert not target.exists()


@pytest.mark.parametrize(
    "entry_point",
    ["number_it", "answer_clean", "compare_answer_with_groundtruth"],
)
def test_escape_is_refused_through_every_grading_entry_point(entry_point, tmp_path):
    """The payload is reachable from an extracted answer, not just from eval."""
    module = _task_module()
    target = tmp_path / entry_point
    payload = _ESCAPE.format(path=target)
    if entry_point == "number_it":
        module.number_it(payload)
    elif entry_point == "answer_clean":
        module.answer_clean(["The answer is"], f"The answer is {payload}")
    else:
        module.compare_answer_with_groundtruth(f"({payload},)", "[1]", [1])
    assert not target.exists()


def test_cleared_builtins_eval_really_was_escapable(tmp_path):
    """The guard this replaced was not a sandbox — the reason for the change.

    Pins the premise rather than the fix: if this ever stops escaping, the
    justification for diverging from upstream has changed and should be re-read.
    """
    target = tmp_path / "pwned"
    builtins.eval(_ESCAPE.format(path=target), {"__builtins__": {}})  # noqa: S307
    assert target.exists()


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__",
        "().__class__.__base__.__subclasses__()",
        "[c for c in (1, 2)][0]",
        "{c for c in (1, 2)}",
        "(c for c in (1, 2))",
        "[1, 2][0]",
        "lambda: 1",
        "__import__('os')",
        "open('x', 'w')",
        "'a string'",
        "f'{1}'",
        "1 if True else 2",
        "1 < 2",
        "True and False",
        "*[1],",
        "{'a': 1}",
        "math.__dict__",
        "math._x",
        "(x := 1)",
    ],
)
def test_execution_shapes_are_refused(expression):
    """Nothing that could reach an object or a statement is evaluable."""
    with pytest.raises((_unsafe_expression(), SyntaxError)):
        _safe_eval()(expression)


def test_unsafe_expression_is_a_value_error():
    """Call sites catch broad `Exception`; this keeps the eval-era contract.

    A refusal has to land in the same `except` that a `SyntaxError` or a
    `NameError` from `eval` landed in, so the answer simply fails to become a
    number instead of failing the sample.
    """
    assert issubclass(_unsafe_expression(), ValueError)


@pytest.mark.parametrize(
    "expression",
    [
        "factorial(2000000)",
        "9**9**9**9",
        "10**10**10",
        "(10**999)**999",
        "1 << 10**10",
        # Same computations spelled through the module. `math` is in scope, so
        # every bound above is reachable by a second name — measured on the
        # unguarded build, `math.factorial(3000000)` ran 33 s and
        # `math.comb(2000000, 1000000)` 14 s, both from five nodes.
        "math.factorial(2000000)",
        "math.comb(2000000, 1000000)",
        "math.perm(200000, 100000)",
        # Repetition, which no *literal* display can express under `_MAX_NODES`:
        # seven nodes for an 800 MB list on the unguarded build.
        "[0] * 100000000",
        "100000000 * [0]",
        "(0,) * 100000000",
    ],
)
def test_unbounded_computation_is_refused_promptly(expression):
    """An unbounded answer holds its worker: `GRADE_TIMEOUT` frees the caller,
    but a pool cannot interrupt a running call, so enough of them empty the pool
    for every task sharing it."""
    started = time.perf_counter()
    with pytest.raises(_unsafe_expression()):
        _safe_eval()(expression)
    assert time.perf_counter() - started < 1.0


def test_node_count_is_bounded():
    module = _task_module()
    with pytest.raises(module.UnsafeExpression, match="too large"):
        module.safe_eval("+".join(["1"] * (module._MAX_NODES + 10)))


def test_factorial_is_bounded_but_usable():
    module = _task_module()
    cap = module._MAX_FACTORIAL
    assert module.safe_eval(f"factorial({cap})") == math.factorial(cap)
    with pytest.raises(module.UnsafeExpression):
        module.safe_eval(f"factorial({cap + 1})")


@pytest.mark.parametrize(
    ("bare", "dotted"),
    [("factorial({n})", "math.factorial({n})")],
)
def test_both_spellings_of_a_bounded_name_agree(bare, dotted):
    """The module is the way *past* a bound wired only to the bare name.

    `_NAMES["factorial"]` is the guarded wrapper, but `math.factorial` is the
    unguarded original unless the attribute allowlist substitutes it — and the
    asymmetry is invisible from the bare spelling alone, which is how it got
    through the first time.
    """
    module = _task_module()
    cap = module._MAX_FACTORIAL
    assert module.safe_eval(bare.format(n=cap)) == module.safe_eval(
        dotted.format(n=cap)
    )
    with pytest.raises(module.UnsafeExpression):
        module.safe_eval(dotted.format(n=cap + 1))


@pytest.mark.parametrize("expression", ["math.comb(1001, 2)", "math.perm(1001, 2)"])
def test_math_integer_callees_share_the_factorial_ceiling(expression):
    """`comb`/`perm` are the same size-driven integer computation as `factorial`."""
    with pytest.raises(_unsafe_expression()):
        _safe_eval()(expression)


def test_sequence_repetition_is_bounded_but_usable():
    module = _task_module()
    cap = module._MAX_SEQUENCE_LENGTH
    assert module.safe_eval(f"[0] * {cap}") == [0] * cap
    with pytest.raises(module.UnsafeExpression, match="sequence repetition"):
        module.safe_eval(f"[0] * {cap + 1}")


def test_math_allowlist_keeps_the_cheap_surface():
    """The bound is on size, not on `math` — narrowing further would be a
    scoring change wearing a safety label."""
    module = _task_module()
    for expression, expected in [
        ("math.prod([2, 3, 4])", 24),
        ("math.gcd(12, 18)", 6),
        ("math.isqrt(17)", 4),
        ("math.log2(8)", 3.0),
        ("math.hypot(3, 4)", 5.0),
    ]:
        assert module.safe_eval(expression) == expected


#: Expressions drawn from the shapes the three call sites actually reach:
#: latex2sympy's `str()` output, and answers a model writes directly.
_FAITHFUL = [
    "1",
    "-1",
    "1.5",
    "-3.25e-7",
    "1 + 2",
    "3 - 4",
    "2*3",
    "7/2",
    "7//2",
    "7%2",
    "2**10",
    "-2**3",
    "+5",
    "(1 + 2)*3",
    "[1, 2, 3]",
    "(1, 2, 3)",
    "[0, 0]",
    "{0}",
    "{1, 2}",
    "sqrt(2)",
    "sin(0)",
    "cos(0)",
    "log(1)",
    "exp(1)",
    "pi",
    "e",
    "E",
    "factorial(5)",
    "math.sqrt(9)",
    "math.pi",
    "3.54*E - 7",
    "E/(-1 + E)",
    # builtins upstream exposed; the cleared-namespace eval lost every one
    "abs(-5)",
    "round(1.6)",
    "pow(2, 10)",
    "min(3, 1)",
    "max(3, 1)",
    "sum([1, 2, 3])",
    "int(2.9)",
    "float(3)",
    "divmod(7, 2)",
    "len([1, 2])",
    # bitwise: kept purely because upstream computes them
    "5 ^ 3",
    "5 & 3",
    "5 | 3",
    "1 << 4",
    "16 >> 2",
    "~5",
    # keywords, which parse as `ast.Constant` rather than as a name lookup
    "True",
    "False",
    "None",
    # under the bounds, where the bounded spellings must still be upstream's
    "math.factorial(5)",
    "math.comb(10, 3)",
    "math.perm(5, 2)",
    "[0] * 5",
    "3 * [1]",
    "(1, 2) * 2",
]


@pytest.mark.parametrize("expression", _FAITHFUL)
def test_matches_upstream_bare_eval(expression):
    """Same value as upstream's `eval(num)` for every evaluable shape."""
    expected = builtins.eval(expression, dict(_UPSTREAM_GLOBALS))  # noqa: S307
    assert _safe_eval()(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["a", "x_2", "oo", "Interval", "Integral", "unknown_name"],
)
def test_unknown_names_fail_as_upstream_does(expression):
    """Symbolic leftovers from latex2sympy: upstream raises NameError, so do we.

    The outcome, not the exception type, is the contract — both readings leave
    the answer un-numeric, which is what the call sites branch on.
    """
    with pytest.raises(_unsafe_expression(), match="unknown name"):
        _safe_eval()(expression)
    with pytest.raises(NameError):
        builtins.eval(expression, dict(_UPSTREAM_GLOBALS))  # noqa: S307


def test_set_display_is_evaluated_not_refused():
    """Fidelity where safety does not object — the set-display defect stands.

    Refusing `{0}` would rescue a list answer latex2sympy folded into a set and
    score +2/800 on the measured run. That is a grader repair, so it belongs to
    a `_fixed` variant and deliberately does not happen under this name.
    """
    assert _safe_eval()("{0}") == {0}


def test_syntax_error_propagates_as_from_eval():
    with pytest.raises(SyntaxError):
        _safe_eval()("1 +")
