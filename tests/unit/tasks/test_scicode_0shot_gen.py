"""Unit tests for the SciCode 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import httpx
import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.scicode import build_test_program, encode_targets
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext, TaskStageOutput
from sieval.datasets.scicode import SciCodeDataset
from sieval.tasks.scicode_0shot_gen import SciCodeZeroShotGenTask

# h5py is an optional (scicode-group) dependency; skip the whole module without it.
h5py = pytest.importorskip("h5py")


class _ScriptedChatModel(ChatModel):
    """Returns queued replies in order, recording each prompt and kwargs."""

    def __init__(self, replies: list[str], model: str = "candidate"):
        super().__init__(model=model, api_key="fake")
        self._replies = list(replies)
        self.prompts: list = []
        self.last_kwargs: dict[str, object] = {}
        self.calls = 0

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.prompts.append(prompt)
        self.last_kwargs = dict(kwargs)
        self.calls += 1
        return ModelOutput(model=self.meta(), texts=[self._replies.pop(0)])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _code_reply(fn: str) -> str:
    return f"```python\ndef {fn}():\n    return 1\n```"


def _dataset() -> SciCodeDataset:
    row = {
        "problem_id": "1",
        "problem_name": "p",
        "required_dependencies": "import numpy as np",
        "sub_steps": [],
    }
    return SciCodeDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list([row])}))


def _substep(number: str, header: str, tests: list[str]) -> dict:
    return {
        "step_number": number,
        "step_description_prompt": f"describe {number}",
        "step_background": f"background {number}",
        "ground_truth_code": "",
        "function_header": header,
        "return_line": "    return 1",
        "test_cases": tests,
    }


def _task(model, **kw) -> SciCodeZeroShotGenTask:
    kw.setdefault("h5_path", "unused")
    return SciCodeZeroShotGenTask(_dataset(), model, **kw)


# --- infer: sequential, single-n, prior code fed forward ---


@pytest.mark.anyio
async def test_infer_is_sequential_and_feeds_prior_code():
    sub_steps = [
        _substep("1.1", "def step_a():", ["assert step_a() == 1"]),
        _substep("1.2", "def step_b():", ["assert step_b() == 1"]),
    ]
    model = _ScriptedChatModel([_code_reply("step_a"), _code_reply("step_b")])
    task = _task(model)
    raw = {
        "problem_id": "1",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert isinstance(boxed, TaskStageOutput)
    assert model.calls == 2
    # Only the scheduling knob is forwarded to the model layer.
    assert model.last_kwargs == {"n": 1}
    # The second prompt must embed the first step's generated function.
    assert "def step_a" in model.prompts[1][0]["content"]

    steps = boxed.value
    assert [s["tested"] for s in steps] == [True, True]
    # code_content = dependencies + accumulated prior funcs + current func.
    assert steps[1]["code_content"].startswith("import numpy as np")
    assert "def step_a" in steps[1]["code_content"]
    assert "def step_b" in steps[1]["code_content"]


@pytest.mark.anyio
async def test_infer_uses_gold_context_for_special_step_without_calling_model():
    # Problem 76 step 3 (index 2) is scientist-authored: never generated/tested,
    # its gold code is fed as context to later steps.
    sub_steps = [
        _substep("76.1", "def step_a():", ["assert step_a() == 1"]),
        _substep("76.2", "def step_b():", ["assert step_b() == 1"]),
        _substep("76.3", "def generate_dna(N, PWM):", ["assert True"]),
        _substep("76.4", "def step_d():", ["assert step_d() == 1"]),
    ]
    model = _ScriptedChatModel(
        [_code_reply("step_a"), _code_reply("step_b"), _code_reply("step_d")]
    )
    task = _task(model)
    raw = {
        "problem_id": "76",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    steps = boxed.value

    # 3 generated steps -> 3 model calls; the special step is skipped.
    assert model.calls == 3
    assert steps[2]["tested"] is False
    assert steps[2]["code_content"] is None
    # The 4th step's prompt must contain the gold generate_dna function.
    assert "def generate_dna" in model.prompts[2][0]["content"]


@pytest.mark.anyio
async def test_infer_embeds_full_gold_class_for_class_special_step():
    # Problem 62 step 1 (index 0) is a gold step whose file defines TWO classes
    # (Block AND EnlargedBlock). Under special_step_mode="verbatim" — the default —
    # the whole gold block is injected, so both classes survive with their `class`
    # keyword: the #59/#49 fix (vs opt-in "extract", which drops them).
    sub_steps = [
        _substep(
            "62.1", "class EnlargedBlock:\n    def __init__(self):", ["assert True"]
        ),
        _substep("62.2", "def uses_block():", ["assert uses_block() == 1"]),
    ]
    model = _ScriptedChatModel([_code_reply("uses_block")])
    task = _task(model)  # default: verbatim
    raw = {
        "problem_id": "62",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))

    # Only the one non-special step calls the model.
    assert model.calls == 1
    # The gold context injected into step 62.2's prompt must contain BOTH classes
    # with their `class` keyword intact (not just an unwrapped __init__).
    gold_ctx = model.prompts[0][0]["content"]
    assert "class Block" in gold_ctx
    assert "class EnlargedBlock" in gold_ctx


@pytest.mark.anyio
async def test_infer_extract_mode_drops_class_for_class_special_step():
    # Opt-in special_step_mode="extract" mirrors upstream get_function_from_code:
    # a `class ...:` header resolves to `__init__`, so the class wrapper is dropped
    # (known upstream bug #59/#49). This documents the behavior reproduced on
    # request for public-leaderboard parity.
    sub_steps = [
        _substep(
            "62.1", "class EnlargedBlock:\n    def __init__(self):", ["assert True"]
        ),
        _substep("62.2", "def uses_block():", ["assert uses_block() == 1"]),
    ]
    model = _ScriptedChatModel([_code_reply("uses_block")])
    task = _task(model, special_step_mode="extract")
    raw = {
        "problem_id": "62",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))

    gold_ctx = model.prompts[0][0]["content"]
    # extract drops the class wrapper -> only a bare __init__ survives.
    assert "class Block" not in gold_ctx
    assert "class EnlargedBlock" not in gold_ctx
    assert "def __init__" in gold_ctx


@pytest.mark.anyio
async def test_infer_flags_empty_extraction_when_model_returns_no_code():
    # A prose-only / truncated response yields no def/class: the step must be
    # flagged so the report can distinguish generation failure from wrong answers.
    sub_steps = [_substep("5.1", "def only_step():", ["assert only_step() == 1"])]
    model = _ScriptedChatModel(["I cannot help with that."])  # no code fence
    task = _task(model)
    raw = {
        "problem_id": "5",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    step = boxed.value[0]
    assert step["empty_extraction"] is True
    assert step["raw_response"] == "I cannot help with that."
    # A normal fenced reply must NOT be flagged (discriminates the check).
    model2 = _ScriptedChatModel([_code_reply("only_step")])
    boxed2 = await _task(model2).infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert boxed2.value[0]["empty_extraction"] is False


@pytest.mark.anyio
async def test_infer_survives_empty_choices_response():
    # An aborted/filtered response can carry an empty `texts` list. infer() must
    # treat it as an empty extraction (raw_response="") rather than IndexError
    # on texts[0], which would fail the whole problem at the pipeline level.
    class _EmptyChoicesModel(_ScriptedChatModel):
        async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
            self.prompts.append(prompt)
            self.calls += 1
            return ModelOutput(model=self.meta(), texts=[])

    sub_steps = [_substep("6.1", "def only_step():", ["assert only_step() == 1"])]
    task = _task(_EmptyChoicesModel([]))
    raw = {
        "problem_id": "6",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    step = boxed.value[0]
    assert step["raw_response"] == ""
    assert step["empty_extraction"] is True


# --- postprocess: h5 targets inlined into a self-contained, runnable program ---


def _write_h5(path, step_number: str, value):
    with h5py.File(path, "w") as f:
        f.create_dataset(f"{step_number}/test1/var1", data=value)


def test_build_test_program_restores_legacy_simps_only_when_imported():
    pytest.importorskip("sympy")
    scipy_integrate = pytest.importorskip("scipy.integrate")
    had_simps = hasattr(scipy_integrate, "simps")
    original_simps = getattr(scipy_integrate, "simps", None)

    code = """from scipy.integrate import simps
import numpy as np

def legacy_even_values():
    x = np.arange(4.0)
    y = x**3
    return (
        simps(y, x, even="first"),
        simps(y, x, even="last"),
        simps(y, x, even="avg"),
    )
"""
    program = build_test_program(
        code,
        encode_targets([None]),
        ["assert np.allclose(legacy_even_values(), (21.5, 20.5, 21.0))"],
    )

    try:
        assert "_install_scicode_simps_compat" in program
        exec(compile(program, "<simps-compat>", "exec"), {})
    finally:
        if had_simps:
            scipy_integrate.simps = original_simps
        elif hasattr(scipy_integrate, "simps"):
            del scipy_integrate.simps

    unaffected = build_test_program(
        "def f():\n    return 1\n", encode_targets([None]), ["assert f() == 1"]
    )
    assert "_install_scicode_simps_compat" not in unaffected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("from scipy.integrate import simps\n", True),
        # simps can arrive in a multi-name import or as an attribute call; an
        # exact-import-line probe would miss both and leave the step ImportError-ing.
        ("from scipy.integrate import quad, simps\n", True),
        ("from scipy.integrate import simps, quad\n", True),
        ("import scipy.integrate\nscipy.integrate.simps(y, x)\n", True),
        ("from scipy import integrate\nintegrate.simps(y, x)\n", True),
        # simpson is a live SciPy name and needs no wrapper — a substring probe
        # for "...import simps" matches it, a word-boundary one must not.
        ("from scipy.integrate import simpson\n", False),
        ("from scipy.integrate import simpson as s\n", False),
        ("import numpy as np\n", False),
    ],
)
def test_build_test_program_injects_simps_compat_by_bare_name(code, expected):
    program = build_test_program(code, encode_targets([None]), ["pass"])
    assert ("_install_scicode_simps_compat" in program) is expected


def test_build_test_program_does_not_leak_comparison_dependencies():
    pytest.importorskip("sympy")
    program = build_test_program(
        "def uses_undeclared_numpy():\n    return np.asarray([1])\n",
        encode_targets([None]),
        [
            "from scicode.compare.cmp import cmp_tuple_or_list\n"
            "assert cmp_tuple_or_list([1], [1])\n"
            'assert "np" not in globals()\n'
            "uses_undeclared_numpy()"
        ],
    )
    namespace: dict[str, object] = {}

    with pytest.raises(NameError, match="np"):
        exec(compile(program, "<isolated-cmp>", "exec"), namespace)
    assert "np" not in namespace
    assert "scipy" not in namespace
    assert "sympy" not in namespace


@pytest.mark.anyio
async def test_postprocess_builds_program_that_executes(tmp_path):
    # The injected comparison shim (vendored cmp) imports sympy; it runs in the
    # code-eval sandbox at runtime, not on the eval side, so sympy is not a
    # declared eval-side dep. Exercise the full program only where it's present.
    pytest.importorskip("sympy")
    h5 = tmp_path / "raw_ground.h5"
    _write_h5(h5, "9.1", 42)
    model = _ScriptedChatModel([])
    task = _task(model, h5_path=str(h5))

    sub_step = _substep("9.1", "def f():", ["assert np.allclose(f(), target)"])
    raw = {"problem_id": "9", "sub_steps": [sub_step]}
    inf = TaskStageOutput(
        value=[
            {
                "step_number": "9.1",
                "tested": True,
                "code_content": "import numpy as np\n\ndef f():\n    return 42\n",
                "raw_response": "```python\ndef f():\n    return 42\n```",
                "empty_extraction": False,
            }
        ]
    )

    programs = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=raw))
    assert len(programs) == 1
    program = programs[0]["program"]

    # The vendored comparison import and the correct target must both resolve at
    # runtime; a matching solution passes without raising.
    exec(compile(program, "<program>", "exec"), {})

    # A wrong solution must fail the injected assertion.
    bad = program.replace("return 42", "return 0")
    with pytest.raises(AssertionError):
        exec(compile(bad, "<program>", "exec"), {})


# --- feedback: transport timeout semantics ---


@pytest.mark.anyio
async def test_feedback_disables_pool_timeout_but_keeps_request_deadline(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": False, "msg": "failed: subprocess timeout: 10.0s"}

    task = _task(_ScriptedChatModel([]), timeout=10.0, max_concurrency=1)
    captured: dict[str, object] = {}

    async def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(task._http_client, "post", fake_post)
    try:
        ok, feedbacks = await task.feedback(
            [
                {
                    "step_number": "1.1",
                    "program": "assert False",
                    "empty_extraction": False,
                }
            ],
            TaskContext(sample_id=0),
        )
    finally:
        await task.shutdown()

    assert ok is True
    assert feedbacks[0]["correct"] is False
    request_timeout = captured["timeout"]
    assert isinstance(request_timeout, httpx.Timeout)
    assert request_timeout.connect == 130.0
    assert request_timeout.read == 130.0
    assert request_timeout.write == 130.0
    assert request_timeout.pool is None


# --- report: sub-problem and main-problem accuracy ---


def _final(sample_id, correct_flags, empty_flags=None, messages=None):
    empty_flags = empty_flags or [False] * len(correct_flags)
    messages = messages or [""] * len(correct_flags)
    return TaskContext(
        sample_id=sample_id,
        feedback_result=[
            {"step_number": f"s{i}", "correct": c, "msg": msg, "empty_extraction": e}
            for i, (c, e, msg) in enumerate(
                zip(correct_flags, empty_flags, messages, strict=True)
            )
        ],
    )


def _fail(sample_id, problem_id, n_steps):
    """A pipeline-failed context, carrying the raw sample its steps come from."""
    return TaskContext(
        sample_id=sample_id,
        raw_sample={
            "problem_id": problem_id,
            "problem_name": "p",
            "required_dependencies": "",
            "sub_steps": [
                _substep(f"{problem_id}.{i + 1}", "def f():", ["assert f() == 1"])
                for i in range(n_steps)
            ],
        },
    )


@pytest.mark.anyio
async def test_report_sub_and_main_accuracy():
    model = _ScriptedChatModel([])
    task = _task(model)
    finals = [
        _final(0, [True, True]),  # fully solved
        _final(1, [True, False, True]),  # partial
    ]
    report = await task.report(finals, fails=[])

    # steps: 4 correct / 5 total = 80%; problems: 1 fully solved / 2 = 50%.
    assert report["correct_steps"] == 4
    assert report["total_steps"] == 5
    assert report["sub_problem_accuracy"] == pytest.approx(80.0)
    assert report["correct_problems"] == 1
    assert report["total_problems"] == 2
    assert report["main_problem_accuracy"] == pytest.approx(50.0)
    assert report["score"] == report["main_problem_accuracy"]


@pytest.mark.anyio
async def test_report_counts_empty_extractions():
    model = _ScriptedChatModel([])
    task = _task(model)
    # Sample 1 has one step whose code failed to extract (also counts as wrong).
    finals = [
        _final(0, [True, True]),
        _final(1, [False, True], empty_flags=[True, False]),
    ]
    report = await task.report(finals, fails=[])
    assert report["empty_extractions"] == 1
    # It still counts as an incorrect step, not a separate bucket.
    assert report["correct_steps"] == 3
    assert report["total_steps"] == 4


@pytest.mark.anyio
async def test_report_surfaces_step_execution_failures():
    task = _task(_ScriptedChatModel([]))
    finals = [
        _final(
            0,
            [False, False, False, False, False],
            messages=[
                "failed: subprocess timeout: 1800.0s",
                "failed: [MemoryError] unable to allocate",
                "failed: [ImportError] cannot import name 'simps'",
                # The evaluator reports the concrete class name, so a package
                # missing from the sandbox image never says "ImportError".
                "failed: [ModuleNotFoundError] No module named 'numba'",
                "failed: [AssertionError]",
            ],
        )
    ]

    report = await task.report(finals, fails=[])

    assert report["timeouts"] == 1
    assert report["memory_errors"] == 1
    assert report["import_errors"] == 2
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_counts_pipeline_fails_as_unsolved_problems():
    model = _ScriptedChatModel([])
    task = _task(model)
    finals = [_final(0, [True, True])]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)

    # The failed problem dilutes main-problem accuracy: 1 solved / 2 = 50%.
    assert report["fails"] == 1
    assert report["total_problems"] == 2
    assert report["main_problem_accuracy"] == pytest.approx(50.0)
    # This context failed before its sample was loaded, so there is no step count
    # to recover and the sub denominator legitimately stays at the evaluated steps.
    assert report["total_steps"] == 2
    assert report["unevaluated_steps"] == 0


@pytest.mark.anyio
async def test_report_keeps_failed_problem_steps_in_sub_denominator():
    task = _task(_ScriptedChatModel([]))
    finals = [_final(0, [True, True])]
    fails = [_fail(1, "1", 3)]
    report = await task.report(finals, fails)

    # The failed problem's 3 tested steps stay in the denominator scoring zero,
    # so removing them cannot inflate sub accuracy: 2/5, not 2/2.
    assert report["correct_steps"] == 2
    assert report["total_steps"] == 5
    assert report["unevaluated_steps"] == 3
    assert report["sub_problem_accuracy"] == pytest.approx(40.0)
    # Both accuracies now agree that the failure is unsolved, rather than one
    # being diluted by it while the other drops it.
    assert report["main_problem_accuracy"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_excludes_special_steps_from_failed_problem_count():
    task = _task(_ScriptedChatModel([]))
    # Problem 13's step 13.6 (zero-based index 5) is scientist-authored: never
    # generated and never tested, so it must stay out of the denominator on the
    # fail path too -- 7 sub-steps, 6 tested.
    report = await task.report([], [_fail(0, "13", 7)])

    assert report["unevaluated_steps"] == 6
    assert report["total_steps"] == 6
    assert report["sub_problem_accuracy"] == pytest.approx(0.0)
    assert report["main_problem_accuracy"] == pytest.approx(0.0)


@pytest.mark.anyio
async def test_report_empty_is_zero():
    model = _ScriptedChatModel([])
    task = _task(model)
    report = await task.report([], [])
    assert report["score"] == 0.0
