"""
Pattern: a prompt longer than the model's context window is scored, not failed.

A YAML run over real clients and an in-process wire. The candidate's server
refuses one prompt with SGLang's 400; that sample finishes as a wrong, empty
answer flagged ``context_overflow``, counted in ``report.json``'s ``sieval``
block and in the request records, never in ``failed``. A task may declare
``context_overflow = "fail"`` instead; a multi-turn task that already handles a
refusal keeps doing so; and a resume of a directory that recorded the overflow
as a failed sample scores it without asking the server again.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from sieval.cli.session import arun_session
from sieval.core.runners import TaskRunner
from sieval.core.tasks import Task
from sieval.core.tasks.records import (
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.requests import REQUESTS_FILE
from tests.integration.test_api_key_env import CANDIDATE_HOST, _Wire
from tests.integration.test_api_key_env import wire as wire  # noqa: F401 -- fixture

SAMPLES = [
    {"question": "Q1", "answer": "A1"},
    {"question": "Q2", "answer": "A2"},
    {"question": "Q3", "answer": "A3"},
]
#: SGLang's refusal for a prompt plus `max_new_tokens` past a 40,960 window.
REFUSAL = (
    "Requested token count exceeds the model's maximum context length of 40960 "
    "tokens. You requested a total of 49152 tokens: 16384 tokens from the input "
    "messages and 32768 tokens for the completion. Please reduce the number of "
    "tokens in the input messages or the completion to fit within the limit."
)


class AnswerTask(Task):
    """One call; correct when the answer matches exactly."""

    model_type = "chat"
    tags = frozenset({"gen"})

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        text = inf.texts[0].strip() if inf.texts else ""
        return build_prediction_record([text or None])

    async def feedback(self, post, ctx):
        prediction = post["rollouts"][0].get("prediction")
        answer = ctx.raw_sample["answer"]
        return True, build_judgement_record(
            answer, [build_rollout_judgement(0, prediction == answer)]
        )

    async def report(self, finals, fails):
        correct = sum(f.feedback_result["n_correct"] for f in finals)
        return {
            "score": correct / (len(finals) + len(fails)),
            "fails": len(fails),
            "score_key": "score",
            "denominator_policy": "requested",
        }


class FailingAnswerTask(AnswerTask):
    """The same task, opting back into a failed sample."""

    context_overflow = "fail"


class PrecheckTask(AnswerTask):
    """Measures its prompt and records an overflow without sending it."""

    async def infer(self, pre, ctx):
        if pre == "Q2":
            return self.model.context_overflow_output(
                pre, "prompt is 41000 tokens, leaving -40 inside a 40960 window"
            )
        return await self.model.agenerate(pre)


class TwoCallTask(AnswerTask):
    """Answers, then asks a follow-up; nothing caught."""

    async def infer(self, pre, ctx):
        return [
            await self.model.agenerate(pre),
            await self.model.agenerate(pre + "+"),
        ]

    async def postprocess(self, inf, ctx):
        return build_prediction_record([inf[0].texts[0] or None])


class WalkTask(AnswerTask):
    """Two turns; a refused turn ends the walk, as Multi-IF's does."""

    async def infer(self, pre, ctx):
        outputs = [await self.model.agenerate(pre)]
        with contextlib.suppress(Exception):
            outputs.append(await self.model.agenerate(pre + "+"))
        return outputs

    async def postprocess(self, inf, ctx):
        return build_prediction_record(
            [inf[0].texts[0]], extra={"n_answered": len(inf)}
        )


def _refuse(question: str) -> Callable[[httpx.Request], httpx.Response | None]:
    def before(request: httpx.Request) -> httpx.Response | None:
        if request.url.host != CANDIDATE_HOST:
            return None
        if json.loads(request.content)["messages"][-1]["content"] != question:
            return None
        return httpx.Response(400, json={"error": {"message": REFUSAL}})

    return before


def _answer_plus() -> Callable[[httpx.Request], httpx.Response | None]:
    """The wire only knows Q1..Q3; answer a walk's second turn too."""

    from tests.integration.test_api_key_env import _openai_chat

    def before(request: httpx.Request) -> httpx.Response | None:
        question = json.loads(request.content)["messages"][-1]["content"]
        if question == "Q2+":
            return httpx.Response(400, json={"error": {"message": REFUSAL}})
        if question.endswith("+"):
            return _openai_chat("more")
        if question == "Q3":
            return _openai_chat("A3")
        return None

    return before


def _config(tmp_path: Path, task_class: str) -> Path:
    config = tmp_path / "eval.yaml"
    config.write_text(
        f"""\
result_dir: "{tmp_path / "results"}"
models:
  candidate:
    name: org/candidate
    api_base: https://{CANDIDATE_HOST}/v1
    api_key: k
    args:
      max_retries: 2
datasets:
  ds:
    class: tests.conftest.MockDataset
    args:
      samples: {json.dumps(SAMPLES)}
tasks:
  t:
    class: tests.integration.test_context_overflow.{task_class}
    dataset: ds
    model: candidate
    runner_config:
      show_progress: false
""",
        encoding="utf-8",
    )
    return config


def _with_q3(before: Callable[[httpx.Request], httpx.Response | None]):
    from tests.integration.test_api_key_env import _openai_chat

    def both(request: httpx.Request) -> httpx.Response | None:
        early = before(request)
        if early is not None:
            return early
        if json.loads(request.content)["messages"][-1]["content"] == "Q3":
            return _openai_chat("A3")
        return None

    return both


def _read(task_dir: Path, name: str) -> dict:
    return json.loads((task_dir / name).read_text())


def _lines(task_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (task_dir / REQUESTS_FILE).read_text().splitlines()]


class _Asked:
    """Wraps a ``before`` hook, recording every candidate question asked."""

    def __init__(self, wired: _Wire) -> None:
        self.questions: list[str] = []
        self._wire = wired

    def answer_with(self, before: Callable[[httpx.Request], httpx.Response | None]):
        def hook(request: httpx.Request) -> httpx.Response | None:
            if request.url.host == CANDIDATE_HOST:
                content = json.loads(request.content)["messages"][-1]["content"]
                self.questions.append(content)
            return before(request)

        self._wire.before = hook

    def count(self, question: str) -> int:
        return self.questions.count(question)


@pytest.fixture
def asked(wire: _Wire) -> _Asked:  # noqa: F811
    return _Asked(wire)


class TestScored:
    @pytest.mark.anyio
    async def test_an_overflow_is_a_wrong_answer_and_an_anomaly(
        self,
        tmp_path: Path,
        wire: _Wire,  # noqa: F811
    ) -> None:
        wire.before = _with_q3(_refuse("Q2"))
        report = await arun_session(_config(tmp_path, "AnswerTask"))
        task_dir = tmp_path / "results" / "t"

        # Two of three right; the refused prompt counts as wrong, not missing.
        assert report["t"]["score"] == pytest.approx(2 / 3)
        assert report["t"]["fails"] == 0
        saved = _read(task_dir, "report.json")
        counts = saved["sieval"]
        assert {k: counts[k] for k in ("total", "final", "failed")} == {
            "total": 3,
            "final": 3,
            "failed": 0,
        }
        assert counts["anomalies"] == 1
        assert counts["context_overflow"] == 1
        # Not a truncation: the model generated nothing.
        assert saved["n_truncated"] == 0

        anomalies = _read(task_dir, "anomalies.json")["samples"]
        assert anomalies == {"1": {"0": {"context_overflow": [0]}}}
        progress = _read(task_dir, "progress.json")
        assert progress["failed"] == 0
        assert progress["anomaly_details"] == {"context_overflow": 1}

        # The refusal is one attempt -- a 400 is never retried -- labelled as
        # what it was, with the status the server sent.
        refused = [line for line in _lines(task_dir) if line["sample_id"] == 1]
        assert len(refused) == 1
        assert refused[0]["status"] == 400
        assert refused[0]["outcome"] == "context_overflow"
        assert refused[0]["error"] == "exception::BadRequestError"
        # Counted once, under its own outcome: outcomes still sum to attempts.
        summary = _read(task_dir, "profile.json")["requests"]["by_phase"]["infer"]
        candidate = summary["org/candidate"]
        assert candidate["outcomes"] == {"2xx": 2, "context_overflow": 1}
        assert sum(candidate["outcomes"].values()) == candidate["attempts"] == 3
        assert counts["requests"]["infer"]["org/candidate"]["outcomes"] == {
            "2xx": 2,
            "context_overflow": 1,
        }

    @pytest.mark.anyio
    async def test_the_refusal_is_on_the_inference_record(
        self,
        tmp_path: Path,
        wire: _Wire,  # noqa: F811
    ) -> None:
        from sieval.core.tasks.loader import TaskLoader

        wire.before = _with_q3(_refuse("Q2"))
        await arun_session(_config(tmp_path, "AnswerTask"))
        task_dir = tmp_path / "results" / "t"
        loader = TaskLoader(task=_bare_task(), root_dir=task_dir)
        contexts = await loader.load_initial_state()
        await loader.hydrate(contexts, set(), include_stages={contexts[1].stage})
        sample = contexts[1]
        inferred = sample.stage_meta["inferred"][-1]
        assert len(inferred["context_overflow"]) == 1
        assert REFUSAL in inferred["context_overflow"][0]
        assert inferred["model_calls"][0]["finish_reasons"] == ["context_overflow"]
        assert inferred["model_calls"][0]["request_params"]["context_overflow"]


def _bare_task() -> Task:
    from tests.conftest import MockChatModel, MockDataset

    return AnswerTask(dataset=MockDataset(SAMPLES), model=MockChatModel())


class TestPrecheck:
    @pytest.mark.anyio
    async def test_a_task_measured_overflow_is_scored_without_a_request(
        self,
        tmp_path: Path,
        asked: _Asked,
    ) -> None:
        asked.answer_with(_with_q3(lambda _request: None))
        report = await arun_session(_config(tmp_path, "PrecheckTask"))
        task_dir = tmp_path / "results" / "t"
        counts = _read(task_dir, "report.json")["sieval"]
        assert counts["failed"] == 0
        assert counts["context_overflow"] == 1
        assert report["t"]["score"] == pytest.approx(2 / 3)
        assert asked.count("Q2") == 0
        assert all(line["sample_id"] != 1 for line in _lines(task_dir))


class TestFailPolicy:
    @pytest.mark.anyio
    async def test_a_task_can_keep_the_overflow_a_failure(
        self,
        tmp_path: Path,
        asked: _Asked,
    ) -> None:
        asked.answer_with(_with_q3(_refuse("Q2")))
        config = _config(tmp_path, "FailingAnswerTask")
        await arun_session(config)
        task_dir = tmp_path / "results" / "t"
        counts = _read(task_dir, "report.json")["sieval"]
        assert counts["failed"] == 1
        assert counts["context_overflow"] == 1
        assert counts["anomalies"] == 0

        # Non-retriable: a resume asks the server nothing about it.
        before = asked.count("Q2")
        await arun_session(config, resume=True)
        assert asked.count("Q2") == before == 1
        assert _read(task_dir, "report.json")["sieval"]["failed"] == 1


class TestTaskHandled:
    @pytest.mark.anyio
    async def test_a_walk_that_stops_itself_is_unchanged(
        self,
        tmp_path: Path,
        asked: _Asked,
    ) -> None:
        asked.answer_with(_answer_plus())
        await arun_session(_config(tmp_path, "WalkTask"))
        task_dir = tmp_path / "results" / "t"
        counts = _read(task_dir, "report.json")["sieval"]
        # The walk caught the refusal and ended, as it always did: no anomaly.
        assert counts["failed"] == 0
        assert counts["context_overflow"] == 0
        assert counts["anomalies"] == 0
        # And `infer` ran once: nothing was sent twice.
        assert asked.count("Q2") == 1
        assert asked.count("Q2+") == 1


class TestResume:
    @pytest.mark.anyio
    async def test_a_recorded_failure_is_scored_without_asking_again(
        self,
        tmp_path: Path,
        asked: _Asked,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asked.answer_with(_with_q3(_refuse("Q2")))
        config = _config(tmp_path, "AnswerTask")
        task_dir = tmp_path / "results" / "t"

        # Run 1 as 0.10.2 ran it: the refusal fails the sample.
        async def legacy_infer(self, ctx):
            return await self._task.infer(ctx.preprocess_result, ctx), []

        with monkeypatch.context() as patched:
            patched.setattr(TaskRunner, "_infer", legacy_infer)
            await arun_session(config)
        first = _read(task_dir, "report.json")
        assert first["sieval"]["failed"] == 1
        assert first["sieval"]["context_overflow"] == 0
        first_identity = _read(task_dir, "meta.json")["identity"]
        assert asked.count("Q2") == 1

        # Run 2: the resume scores it from the recorded refusal.
        report = await arun_session(config, resume=True)
        assert asked.count("Q2") == 1, "the server was not asked again"
        counts = _read(task_dir, "report.json")["sieval"]
        assert counts["failed"] == 0
        assert counts["final"] == 3
        assert counts["context_overflow"] == 1
        assert counts["anomalies"] == 1
        assert report["t"]["score"] == pytest.approx(2 / 3)

        meta = _read(task_dir, "meta.json")
        # Neither the inference nor the grader identity moved; only what was
        # sent for the formerly failed sample is now on record.
        assert (
            meta["identity"]["inference"]["digest"]
            == (first_identity["inference"]["digest"])
        )
        assert meta["identity"]["grader"] == first_identity["grader"]
        assert meta["runs"][-1]["reran"] == ["inference", "grader"]
        # No request went out, so the resume's request counts have no infer.
        assert "infer" not in (meta["runs"][-1].get("requests") or {})

        # Run 3 finds the report final.
        await arun_session(config, resume=True)
        assert asked.count("Q2") == 1

    @pytest.mark.anyio
    async def test_a_refusal_under_another_inference_identity_is_asked_again(
        self,
        tmp_path: Path,
        asked: _Asked,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Every sample refused, so none keeps an inference and the resume's
        # inference gate lets a changed identity through: only the replay's own
        # identity check stands between it and a refusal it did not record.
        def refuse_all(request: httpx.Request) -> httpx.Response | None:
            if request.url.host != CANDIDATE_HOST:
                return None
            return httpx.Response(400, json={"error": {"message": REFUSAL}})

        asked.answer_with(refuse_all)
        config = _config(tmp_path, "AnswerTask")
        task_dir = tmp_path / "results" / "t"

        async def legacy_infer(self, ctx):
            return await self._task.infer(ctx.preprocess_result, ctx), []

        with monkeypatch.context() as patched:
            patched.setattr(TaskRunner, "_infer", legacy_infer)
            await arun_session(config)
        assert _read(task_dir, "report.json")["sieval"]["failed"] == 3
        first_digest = _read(task_dir, "meta.json")["identity"]["inference"]["digest"]

        config.write_text(
            config.read_text().replace(
                "max_retries: 2", "max_retries: 2\n      temperature: 0.5"
            )
        )
        await arun_session(config, resume=True)
        meta = _read(task_dir, "meta.json")
        assert meta["identity"]["inference"]["digest"] != first_digest
        # Asked again under the new sampling, not scored from the old record.
        assert asked.count("Q2") == 2
        counts = _read(task_dir, "report.json")["sieval"]
        assert counts["failed"] == 0
        assert counts["context_overflow"] == 3

    @pytest.mark.anyio
    async def test_a_refusal_that_cannot_name_its_call_is_inferred_again(
        self,
        tmp_path: Path,
        asked: _Asked,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Q2's follow-up is refused; its first call is answered.
        asked.answer_with(_answer_plus())
        config = _config(tmp_path, "TwoCallTask")
        task_dir = tmp_path / "results" / "t"

        async def legacy_infer(self, ctx):
            return await self._task.infer(ctx.preprocess_result, ctx), []

        with monkeypatch.context() as patched:
            patched.setattr(TaskRunner, "_infer", legacy_infer)
            await arun_session(config)
        assert _read(task_dir, "report.json")["sieval"]["failed"] == 1

        # The recorded refusal may belong to either call, so the resume asks
        # again rather than score the wrong one, and keeps the answered turn.
        report = await arun_session(config, resume=True)
        assert asked.count("Q2") == 2
        assert asked.count("Q2+") == 2
        counts = _read(task_dir, "report.json")["sieval"]
        assert counts["failed"] == 0
        assert counts["context_overflow"] == 1
        # Q2's first turn was answered, and right: kept, not blanked.
        assert report["t"]["score"] == 1.0
