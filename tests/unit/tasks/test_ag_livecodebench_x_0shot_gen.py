"""Unit tests for the Ag-LiveCodeBench-X 0-shot chat task.

Three things are pinned here, all of which a reasonable-looking edit would break
silently:

* **The prompt bytes.** They are a frozen ``dspy==3.0.0b2`` ``ChatAdapter``
  rendering, so nothing in the repo derives them and nothing but a hash can tell
  you they still match. The two expectations below that look like typos -- an
  8-space indent inside the system message, a trailing space on one field line
  and not the next -- are the rendering, not this file's mistakes.
* **The response parser**, whose expected values were captured by running
  ``ChatAdapter.parse`` under that same pin. The indented-header case encodes an
  upstream offset bug; "fixing" it would move scores.
* **The code extractor**, whose two surprising branches (no fence at all means
  the whole reply is code; an unterminated fence means nothing is) are what a
  cleanup usually gets wrong.

The verifier half lives in the vendored evaluator, a separate deployable with its
own repo, so only what goes on the wire is asserted here.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import base64
import hashlib
import json
import pickle
import zlib

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.ag_livecodebench_x import AgLiveCodeBenchXDataset
from sieval.tasks.ag_livecodebench_x_0shot_gen import (
    AgLiveCodeBenchXZeroShotGenTask,
    _extract_code_from_markdown,
    _parse_chat_adapter_fields,
    _render_prompt,
)
from tests.conftest import HandlerTransport

# sha256 of `system + NUL + user` for the placeholder inputs below, captured from
# `ChatAdapter().format(ChainOfThought(SolveProblem).predict.signature, [], ...)`
# under upstream's pinned resolution (dspy==3.0.0b2, exclude-newer 2025-08-05).
_PROMPT_DIGEST = "69180174fafdd13d7523628d3d1df3ff9c2ef6e026af5c2529f696744ee3ae22"

_CASES = [{"input": "2\n", "output": "4"}, {"input": "5\n", "output": "10"}]

# What the evaluator reports back as the verifier that scored a rollout.
_IMAGE = "ghcr.io/nuprl/agnostics@sha256:" + "c1" * 32


def _encode(cases: list[dict]) -> str:
    """LiveCodeBench's private-test encoding: JSON -> pickle -> zlib -> base64."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


def _raw() -> dict:
    return {
        "question_id": "1873_A",
        "question_content": "Read an integer and print twice its value.",
        "private_test_cases": _encode(_CASES),
    }


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=("[[ ## solution ## ]]\nx",) * req.sampling.n)


class _Response:
    """Enough of an httpx response for `feedback` -- the submission passed."""

    @staticmethod
    def raise_for_status() -> None:
        return None

    @staticmethod
    def json() -> dict:
        return {
            "status": True,
            "msg": "success",
            "data": {
                "n_cases": 1,
                "n_passed": 1,
                "verifier_image": _IMAGE,
            },
        }


class _CapturingEvaluator:
    """Stands in for the code-eval service, recording what was asked of it."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.deadlines: list[float] = []

    async def post(self, url, *, json, timeout):
        _ = url
        self.bodies.append(json)
        self.deadlines.append(timeout)
        return _Response()

    async def aclose(self) -> None:
        return None


def _task(**kwargs) -> AgLiveCodeBenchXZeroShotGenTask:
    kwargs.setdefault("language", "Lua")
    dataset = AgLiveCodeBenchXDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_raw()])})
    )
    return AgLiveCodeBenchXZeroShotGenTask(dataset, _StubChatModel(), **kwargs)


async def _post_one(prediction: str | None = "print(1)", **kwargs):
    """Run `feedback` for one rollout and hand back what the evaluator saw."""
    task = _task(**kwargs)
    await task._http_client.aclose()  # the real client is never used
    evaluator = _CapturingEvaluator()
    task._http_client = evaluator
    try:
        await task.feedback(
            build_prediction_record([prediction]),
            TaskContext(sample_id=0, raw_sample=_raw()),
        )
    finally:
        await task.shutdown()
    return evaluator


# --------------------------------------------------------------------------- #
# The frozen prompt
# --------------------------------------------------------------------------- #
def test_prompt_bytes_match_the_captured_dspy_rendering():
    system, user = _render_prompt("<<LANGUAGE>>", "<<PROBLEM>>")
    assert system["role"] == "system"
    assert user["role"] == "user"

    digest = hashlib.sha256(
        (system["content"] + "\x00" + user["content"]).encode()
    ).hexdigest()
    # A mismatch means the constants were edited, not that DSPy changed: the
    # rendering is frozen on purpose and there is no runtime DSPy here.
    assert digest == _PROMPT_DIGEST


def test_instructions_keep_the_source_docstrings_indentation():
    # DSPy re-inserts the signature docstring into the system message with the
    # Python source indentation intact, while `Signature.instructions` itself is
    # dedented. Only observable by running it, so it is asserted rather than
    # trusted.
    system, _ = _render_prompt("Lua", "p")
    assert (
        "objective is: \n        Solve the following programming problem"
        in (system["content"])
    )


def test_the_field_list_keeps_its_trailing_space_asymmetry():
    # The list is built with a trailing ": " per line and stripped as a block, so
    # the first line keeps its trailing space and the last one loses it. Looks
    # like a typo; is the wire format.
    system, _ = _render_prompt("Lua", "p")
    assert "1. `programming_language` (str): \n" in system["content"]
    assert "2. `problem_statement` (str):\n" in system["content"]
    assert "1. `reasoning` (str): \n" in system["content"]


def test_the_problem_statement_may_contain_braces():
    # The system message carries literal `{programming_language}` placeholders as
    # wire text, so the user turn is concatenated, never `format`ted.
    _, user = _render_prompt("Lua", "print {x} and {{y}}")
    assert "print {x} and {{y}}" in user["content"]


# --------------------------------------------------------------------------- #
# ChatAdapter response parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        pytest.param(
            "[[ ## reasoning ## ]]\nr\n\n[[ ## solution ## ]]\nprint(1)\n\n"
            "[[ ## completed ## ]]\n",
            {"reasoning": "r", "solution": "print(1)"},
            id="canonical",
        ),
        pytest.param(
            "[[ ## reasoning ## ]]\nr\n\n[[ ## solution ## ]]\nprint(1)",
            {"reasoning": "r", "solution": "print(1)"},
            id="no-completed-marker",
        ),
        pytest.param(
            "[[ ## solution ## ]]\nprint(1)\n\n[[ ## completed ## ]]\n",
            None,
            id="missing-reasoning-is-a-parse-failure",
        ),
        pytest.param(
            "[[ ## reasoning ## ]] r\n[[ ## solution ## ]] print(1)\n",
            {"reasoning": "r", "solution": "print(1)"},
            id="content-inline-with-the-header",
        ),
        pytest.param(
            "   [[ ## reasoning ## ]]\nr\n   [[ ## solution ## ]]\nprint(1)\n",
            {"reasoning": "]]\nr", "solution": "]]\nprint(1)"},
            id="indented-header-leaks-the-closing-brackets",
        ),
        pytest.param(
            "[[ ## reasoning ## ]]\nr\n[[ ## solution ## ]]\nfirst\n"
            "[[ ## solution ## ]]\nsecond\n",
            {"reasoning": "r", "solution": "first"},
            id="first-occurrence-wins",
        ),
        pytest.param(
            "Sure!\n\n[[ ## reasoning ## ]]\nr\n\n[[ ## solution ## ]]\nprint(1)\n",
            {"reasoning": "r", "solution": "print(1)"},
            id="prose-before-the-first-header-is-dropped",
        ),
        pytest.param(
            "[[ ## reasoning ## ]]\nr\n[[ ## thoughts ## ]]\nx\n"
            "[[ ## solution ## ]]\nprint(1)\n",
            {"reasoning": "r", "solution": "print(1)"},
            id="unknown-headers-are-dropped",
        ),
        pytest.param(
            "[[ ## reasoning ## ]]\nr\n\n[[ ## solution ## ]]\n\n"
            "[[ ## completed ## ]]\n",
            {"reasoning": "r", "solution": ""},
            id="blank-solution-parses-as-empty-not-missing",
        ),
        pytest.param(
            "[[ ## reasoning ## ]]\nr and then the tokens ran ou",
            None,
            id="truncated-before-the-solution-marker",
        ),
    ],
)
def test_parse_reproduces_chat_adapter(completion, expected):
    # Every expectation here was produced by running dspy==3.0.0b2's own
    # ChatAdapter.parse on the same input.
    assert _parse_chat_adapter_fields(completion) == expected


# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        pytest.param("```lua\nprint(1)\n```", "print(1)", id="tagged-fence"),
        pytest.param("```\nprint(1)\n```", "print(1)", id="untagged-fence"),
        pytest.param(
            "```lua\nfirst\n```\ntext\n```lua\nsecond\n```",
            "first",
            id="first-block-wins",
        ),
        pytest.param("print(1)", "print(1)", id="no-fence-is-all-code"),
        pytest.param("```lua\nprint(1)", None, id="unterminated-fence-is-nothing"),
        pytest.param(
            "```python\nsolve()\n# Example usage:\nsolve(1)\n```",
            "solve()",
            id="example-usage-truncation",
        ),
        pytest.param("", "", id="empty-in-empty-out"),
        pytest.param(None, None, id="none-in-none-out"),
    ],
)
def test_extract_code_reproduces_upstream(markdown, expected):
    assert _extract_code_from_markdown(markdown) == expected


# --------------------------------------------------------------------------- #
# language / container_lang
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("language", [None, "", "   "])
def test_language_has_no_default(language):
    # The benchmark measures a different thing per language, so a silent default
    # would let two runs of one task name mean different things.
    with pytest.raises(ValueError, match="explicit `language`"):
        _task(language=language)


@pytest.mark.parametrize(
    ("language", "expected_tag"),
    [
        # Tags upstream publishes under the language's own name.
        ("Lua", "lua"),
        ("Lua /nothink", "lua"),
        ("R", "r"),
        ("Python", "python"),
        ("Java", "java"),
        # ...and the ones tagged by FILE EXTENSION instead. Lowercasing the first
        # word -- the obvious derivation -- names no published image for these,
        # so every rollout would come back `infra:exit`. Three of the five
        # languages the paper reports are in this half.
        ("Julia 1.10", "jl"),
        ("OCaml", "ml"),
        ("Fortran", "f90"),
        ("C++", "cpp"),
    ],
)
def test_container_tag_matches_upstreams_published_tags(language, expected_tag):
    # ghcr.io/nuprl/agnostics ships exactly: lua, r, python, jl, java, cpp, ml, f90.
    assert _task(language=language)._container_lang == expected_tag


def test_an_unknown_language_falls_through_to_its_first_word():
    # No table entry is not an error: a private registry may publish any tag, and
    # the first word is the only sensible guess.
    assert _task(language="Zig 0.13")._container_lang == "zig"


def test_container_tag_can_be_overridden():
    # What a private registry or a hand-built container needs; upstream warns the
    # prompt language and the verifier tag are independent.
    task = _task(language="Julia", container_lang="julia-1-10-custom")
    assert task._container_lang == "julia-1-10-custom"


@pytest.mark.anyio
async def test_the_prompt_language_reaches_the_prompt_not_the_container():
    task = _task(language="Lua /nothink", container_lang="lua")
    try:
        record = await task.preprocess(
            _raw(), TaskContext(sample_id=0, raw_sample=_raw())
        )
    finally:
        await task.shutdown()

    user = record["prompt"][1]["content"]
    assert "Lua /nothink" in user
    assert record["extra"]["language"] == "Lua /nothink"


# --------------------------------------------------------------------------- #
# What goes on the wire
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_the_request_carries_the_agnostics_source_and_suite():
    evaluator = await _post_one()

    (body,) = evaluator.bodies
    assert body["source"] == "agnostics"
    assert body["lang"] == "lua"
    assert body["code"] == "print(1)"
    # Decoded from the sample's own base64/zlib/pickle column, split into the
    # parallel lists the evaluator's request model already speaks.
    assert body["test"] == {"inputs": ["2\n", "5\n"], "outputs": ["4", "10"]}
    # Every problem in this release is stdio, so no `fn_name` is ever sent.
    assert "fn_name" not in body["test"]


@pytest.mark.anyio
async def test_the_default_timeout_is_upstreams_fifteen_seconds():
    evaluator = await _post_one()

    (body,) = evaluator.bodies
    assert body["timeout"] == 15.0
    # The HTTP deadline must sit outside the container wall, or a timing-out
    # submission surfaces as a transport error instead of a verdict. The margin
    # is large because writing a decoded suite (tens of MB) is inside it.
    assert evaluator.deadlines == [15.0 + 300.0]


@pytest.mark.anyio
async def test_an_unextractable_answer_is_still_graded():
    # `None` on disk means "could not extract"; on the wire it is "", so the
    # verifier returns a real verdict rather than the rollout being skipped.
    evaluator = await _post_one(prediction=None)

    (body,) = evaluator.bodies
    assert body["code"] == ""


def test_k_may_not_exceed_n():
    with pytest.raises(ValueError, match="pass@2"):
        _task(k=2, n=1)


# --------------------------------------------------------------------------- #
# report()
# --------------------------------------------------------------------------- #
async def _report(
    msgs: list[str],
    corrects: list[bool] | None = None,
    images: list[str | None] | None = None,
):
    task = _task()
    corrects = corrects if corrects is not None else [False] * len(msgs)
    images = images if images is not None else [_IMAGE] * len(msgs)
    try:
        finals = [
            TaskContext(
                sample_id=i,
                raw_sample=_raw(),
                feedback_result=build_judgement_record(
                    None,
                    [
                        build_rollout_judgement(
                            0, ok, extra={"msg": msg, "verifier_image": image}
                        )
                    ],
                ),
            )
            for i, (msg, ok, image) in enumerate(
                zip(msgs, corrects, images, strict=True)
            )
        ]
        return await task.report(finals, [])
    finally:
        await task.shutdown()


@pytest.mark.anyio
async def test_infra_failures_are_counted_separately_from_wrong_answers():
    # Upstream's `run_error_rate` counterpart. It must not touch pass@1: an
    # unrunnable container is still a rollout that did not pass.
    report = await _report(
        ["success", "fail:wrong-output", "infra:timeout", "infra:exit 125: no image"],
        corrects=[True, False, False, False],
    )

    assert report["n_run_errors"] == 2
    assert report["pass@1"] == pytest.approx(25.0)
    assert report["score"] == report["pass@1"]


@pytest.mark.anyio
async def test_a_null_message_does_not_crash_the_counter():
    # An absent `msg` is `None` on disk after serialization drops it.
    report = await _report([None])  # type: ignore[list-item]

    assert report["n_run_errors"] == 0


@pytest.mark.anyio
async def test_the_report_declares_its_column_and_the_language_it_measured():
    report = await _report(["success"], corrects=[True])

    assert report["score_key"] == "pass@1"
    assert report["denominator_policy"] == "requested"
    # The task name cannot carry the language, so the report has to.
    assert report["language"] == "Lua"
    assert report["container_lang"] == "lua"


@pytest.mark.anyio
async def test_the_pinned_verifier_is_recorded_on_the_verdict():
    # A digest-pinned verifier is only as good as the run's record of it: without
    # this, nothing on disk says which grader produced the column.
    evaluator = await _post_one()
    _ = evaluator

    report = await _report(["success"], corrects=[True])
    assert report["verifier_image"] == _IMAGE


@pytest.mark.anyio
async def test_two_verifiers_in_one_run_are_both_reported():
    # An evaluator re-pointed mid-run makes the column a mix of two graders.
    # Picking one to stand for the rest would hide that.
    other = "ghcr.io/nuprl/agnostics@sha256:" + "b4" * 32
    report = await _report(
        ["success", "success"], corrects=[True, True], images=[_IMAGE, other]
    )

    assert report["verifier_image"] == ", ".join(sorted([_IMAGE, other]))


@pytest.mark.anyio
async def test_an_unreported_verifier_is_omitted_not_blanked():
    # `None` is what a command override yields, where the image is unknowable.
    # An empty string would read as "ran unpinned", a different claim.
    report = await _report(["success"], corrects=[True], images=[None])

    assert "verifier_image" not in report


@pytest.mark.anyio
async def test_a_pipeline_failure_counts_against_the_score():
    # `DENOMINATOR_REQUESTED`, matching upstream's `pass1`: it divides by rows
    # written, and a generation that produced no solution is still one of them.
    task = _task()
    try:
        finals = [
            TaskContext(
                sample_id=0,
                raw_sample=_raw(),
                feedback_result=build_judgement_record(
                    None, [build_rollout_judgement(0, True, extra={"msg": "success"})]
                ),
            )
        ]
        report = await task.report(
            finals, [TaskContext(sample_id=1, raw_sample=_raw())]
        )
    finally:
        await task.shutdown()

    assert report["fails"] == 1
    assert report["pass@1"] == pytest.approx(50.0)
