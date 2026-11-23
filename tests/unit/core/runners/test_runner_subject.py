"""Runner identity must describe the subject resolved by task setup.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json

import httpx
import pytest

from sieval.core.models import (
    EvalSubject,
    RecordingAsyncClient,
    subject_call,
    subject_call_key,
)
from sieval.core.runners import StageResumeError, TaskRunner, task_resume_plan
from sieval.core.tasks.task import Task
from tests.conftest import BenchmarkTask, MockChatModel, MockDataset, make_config


class _Subject:
    def __init__(self, revision):
        self.revision = revision

    def subject_meta(self):
        return {"model": "service", "revision": self.revision}

    def call_key(self):
        return subject_call_key(self.subject_meta())

    def request_args(self):
        return {"protocol": "v1"}

    def serving(self):
        return None


class _SetupSubjectTask(Task):
    requires_model = False

    def __init__(self, dataset, model=None, name=None, *, setup_only=False):
        super().__init__(dataset, model, name)
        self._subject = None if setup_only else _Subject("unresolved")
        self.revision = "v1"

    @property
    def subject(self) -> EvalSubject:
        if self._subject is None:
            raise RuntimeError("subject has not been resolved by setup")
        return self._subject

    async def setup(self):
        self._subject = _Subject(self.revision)

    def grader_args(self):
        return {}

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        assert self._subject is not None
        with subject_call(self.subject) as call:
            call.record(request_params={"question": pre})
        return self._subject.revision

    async def postprocess(self, inf, ctx):
        return inf

    async def feedback(self, post, ctx):
        return True, {"prediction": post}

    async def report(self, finals, fails):
        return {"total": len(finals) + len(fails)}


class _DerivedModelTask(BenchmarkTask):
    @property
    def model(self):
        return super().model.with_args(temperature=0.5)


class _OverflowSubjectTask(_SetupSubjectTask):
    async def infer(self, pre, ctx):
        def respond(_request):
            return httpx.Response(400, json={"error": "prompt is too long"})

        with subject_call(self.subject):
            async with RecordingAsyncClient(
                transport=httpx.MockTransport(respond),
                base_url="https://service.invalid",
                max_retries=0,
            ) as client:
                response = await client.post("/evaluate", json={"question": pre})
            raise httpx.HTTPStatusError(
                "Client error '400 Bad Request': prompt is too long",
                request=response.request,
                response=response,
            )


class _ModelWithSubjectTask(_SetupSubjectTask):
    requires_model = True


class _ModelWithOverflowSubjectTask(_OverflowSubjectTask):
    requires_model = True


class _MissingSubjectTask(_SetupSubjectTask):
    subject = Task.subject


class _HandshakeSubjectTask(_SetupSubjectTask):
    services = frozenset({"remote-api"})

    def __init__(self, dataset, *, reachable):
        super().__init__(dataset, setup_only=True)
        self.reachable = reachable
        self.shut = False

    async def setup(self):
        if not self.reachable:
            raise httpx.ConnectError(
                "service not started",
                request=httpx.Request("GET", "http://127.0.0.1:40123/handshake"),
            )
        await super().setup()

    async def shutdown(self):
        self.shut = True


class _NoPublicRuntimePlanModel(MockChatModel):
    @property
    def runtime_plan(self):
        return None


@pytest.mark.anyio
@pytest.mark.parametrize("setup_only", [False, True])
@pytest.mark.parametrize("plan", [False, True])
async def test_subject_resolved_by_setup_is_compared_before_resume(
    tmp_path, setup_only, plan
):
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    task = _SetupSubjectTask(dataset, setup_only=setup_only)
    await TaskRunner(task, make_config(tmp_path)).arun()
    root = tmp_path / "results"
    recorded_meta = (root / "meta.json").read_bytes()
    metadata = json.loads(recorded_meta)
    assert metadata["identity"]["inference"]["requests"]["digest"] is not None
    record = json.loads(next((root / "0" / "inferred").glob("*.jsonl")).read_text())
    assert record["meta_last"]["model_calls"][0]["model"]["revision"] == "v1"

    same = _SetupSubjectTask(dataset, setup_only=setup_only)
    runner = TaskRunner(same, make_config(tmp_path, resume=True))
    reuse = await runner.plan_run()
    assert reuse["start"] == "reuse"

    changed = _SetupSubjectTask(dataset, setup_only=setup_only)
    changed.revision = "v2"
    runner = TaskRunner(changed, make_config(tmp_path, resume=True))
    with pytest.raises(StageResumeError, match="stage `inference` differs"):
        if plan:
            await runner.plan_run()
        else:
            await runner.arun()
    assert (root / "meta.json").read_bytes() == recorded_meta


@pytest.mark.anyio
async def test_overridden_model_property_keeps_the_request_digest(tmp_path):
    model = MockChatModel(default_answer="ok")
    task = _DerivedModelTask(
        MockDataset(samples=[{"question": "q", "answer": "ok"}]), model
    )
    try:
        await TaskRunner(task, make_config(tmp_path)).arun()
        metadata = json.loads((tmp_path / "results" / "meta.json").read_text())
        requests = metadata["identity"]["inference"]["requests"]
        assert requests["digest"] is not None
        assert "unknown" not in requests
    finally:
        await model.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("max_retries", [0, 1])
@pytest.mark.parametrize("with_model", [False, True])
async def test_external_overflow_spends_the_configured_retry_budget(
    tmp_path, max_retries, with_model
):
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    model = MockChatModel() if with_model else None
    try:
        for attempt in range(max_retries + 3):
            task = (
                _ModelWithOverflowSubjectTask(dataset, model)
                if with_model
                else _OverflowSubjectTask(dataset)
            )
            await TaskRunner(
                task,
                make_config(tmp_path, max_retries=max_retries, resume=attempt > 0),
            ).arun()
        requests = (tmp_path / "results" / "requests.jsonl").read_text().splitlines()
        assert len(requests) == 1 + max_retries
    finally:
        if model is not None:
            await model.aclose()


@pytest.mark.anyio
async def test_independent_subject_owns_serving_even_when_a_model_is_bound(tmp_path):
    model = MockChatModel()
    task = _ModelWithSubjectTask(
        MockDataset(samples=[{"question": "q", "answer": "v1"}]), model
    )
    try:
        await TaskRunner(task, make_config(tmp_path)).arun()
        metadata = json.loads((tmp_path / "results" / "meta.json").read_text())
        assert metadata["model"] == "service"
        assert "serving" not in metadata["runs"][-1]
    finally:
        await model.aclose()


@pytest.mark.anyio
async def test_independent_subject_does_not_inherit_helper_deployment_facts(tmp_path):
    model = MockChatModel()
    task = _ModelWithSubjectTask(
        MockDataset(samples=[{"question": "q", "answer": "v1"}]), model
    )
    try:
        await TaskRunner(
            task,
            make_config(tmp_path),
            context_limit=8192,
            weights="helper-w",
            stated_serving={
                "numerics": {"dtype": "float16"},
                "parsers": {"reasoning": "qwen3"},
            },
            checkpoint_numerics={"dtype": "float16"},
        ).arun()
        metadata = json.loads((tmp_path / "results" / "meta.json").read_text())
        assert "weights" not in metadata
        assert "numerics" not in metadata["identity"]["inference"]
        assert "parsers" not in metadata["identity"]["inference"]
    finally:
        await model.aclose()


class _OtherMockChatModel(MockChatModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = "other-mock-chat"


def _helper_runs(model):
    """Two helper bindings of one subject task, alike but for one fact."""
    other = _OtherMockChatModel()
    return {
        "name": ({}, {}, model, other),
        "weights": ({"weights": "w1"}, {"weights": "w2"}, model, model),
        "context_limit": (
            {"context_limit": 8192},
            {"context_limit": 16384},
            model,
            model,
        ),
        "numerics": (
            {"stated_serving": {"numerics": {"dtype": "float16"}}},
            {"stated_serving": {"numerics": {"dtype": "bfloat16"}}},
            model,
            model,
        ),
        "request_args": (
            {},
            {},
            model.with_args(temperature=0.1),
            model.with_args(temperature=0.9),
        ),
    }, other


@pytest.mark.anyio
@pytest.mark.parametrize(
    "changed", ["name", "weights", "context_limit", "numerics", "request_args"]
)
async def test_independent_subject_resume_refuses_a_changed_helper(tmp_path, changed):
    model = MockChatModel()
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    runs, other = _helper_runs(model)
    first, second, first_model, second_model = runs[changed]
    try:
        await TaskRunner(
            _ModelWithSubjectTask(dataset, first_model), make_config(tmp_path), **first
        ).arun()
        same = TaskRunner(
            _ModelWithSubjectTask(dataset, first_model),
            make_config(tmp_path, resume=True),
            **first,
        )
        assert (await same.plan_run())["start"] == "reuse"
        runner = TaskRunner(
            _ModelWithSubjectTask(dataset, second_model),
            make_config(tmp_path, resume=True),
            **second,
        )
        with pytest.raises(StageResumeError, match="stage `inference` differs"):
            await runner.plan_run()
    finally:
        await model.aclose()
        await other.aclose()


@pytest.mark.anyio
async def test_public_resume_plan_compares_helper_request_args(tmp_path):
    model = MockChatModel()
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    try:
        initial = _ModelWithSubjectTask(dataset, model.with_args(temperature=0.1))
        await TaskRunner(initial, make_config(tmp_path)).arun()
        recorded = json.loads((tmp_path / "results" / "meta.json").read_text())
        changed = _ModelWithSubjectTask(dataset, model.with_args(temperature=0.9))
        await changed.setup()
        try:
            config = make_config(tmp_path, resume=True)
            plan = task_resume_plan(
                changed,
                recorded,
                runner_keys={
                    "deterministic": config.deterministic,
                    "max_iterations": config.max_iterations,
                    "max_retries": config.max_retries,
                },
            )
        finally:
            await changed.shutdown()
        assert plan["refused"] == "inference"
    finally:
        await model.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("context_limit", [8192, 16384])
@pytest.mark.parametrize("serving_source", ["none", "recipe", "stated"])
async def test_public_resume_plan_matches_runner_for_independent_subject(
    tmp_path, context_limit, serving_source
):
    model = MockChatModel()
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    recipe = (
        {
            "assignments": [
                {"engine_params": {"dtype": "auto", "reasoning_parser": "qwen3"}}
            ]
        }
        if serving_source == "recipe"
        else None
    )
    stated = (
        {
            "numerics": {"dtype": "float16"},
            "parsers": {"reasoning_parser": "qwen3"},
        }
        if serving_source == "stated"
        else None
    )
    checkpoint = {"dtype": "float16"}
    try:
        await TaskRunner(
            _ModelWithSubjectTask(dataset, model),
            make_config(tmp_path),
            context_limit=8192,
        ).arun()
        recorded = json.loads((tmp_path / "results" / "meta.json").read_text())
        task = _ModelWithSubjectTask(dataset, model)
        config = make_config(tmp_path, resume=True)
        runner = TaskRunner(
            task,
            config,
            context_limit=context_limit,
            inference_recipe=recipe,
            stated_serving=stated,
            checkpoint_numerics=checkpoint,
        )
        same = context_limit == 8192 and serving_source == "none"
        if same:
            planned = await runner.plan_run()
            assert planned["stages"] is not None
            assert planned["stages"]["stages"]["inference"] == "same"
        else:
            with pytest.raises(StageResumeError, match="stage `inference` differs"):
                await runner.plan_run()

        # The public entry point is called after setup, like the runner.
        await task.setup()
        try:
            public = task_resume_plan(
                task,
                recorded,
                runner_keys={
                    "deterministic": config.deterministic,
                    "max_iterations": config.max_iterations,
                    "max_retries": config.max_retries,
                },
                context_limit=context_limit,
                recipe=recipe,
                stated_serving=stated,
                checkpoint_numerics=checkpoint,
            )
        finally:
            await task.shutdown()
        assert public["refused"] == (None if same else "inference")
    finally:
        await model.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("context_limit", [8192, 16384])
async def test_public_resume_plan_still_compares_model_context_limit(
    tmp_path, context_limit
):
    model = MockChatModel(default_answer="v1")
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    try:
        await TaskRunner(
            BenchmarkTask(dataset, model), make_config(tmp_path), context_limit=8192
        ).arun()
        recorded = json.loads((tmp_path / "results" / "meta.json").read_text())
        task = BenchmarkTask(dataset, model)
        config = make_config(tmp_path, resume=True)
        await task.setup()
        try:
            public = task_resume_plan(
                task,
                recorded,
                runner_keys={
                    "deterministic": config.deterministic,
                    "max_iterations": config.max_iterations,
                    "max_retries": config.max_retries,
                },
                context_limit=context_limit,
            )
        finally:
            await task.shutdown()
        runner = TaskRunner(task, config, context_limit=context_limit)
        if context_limit == 8192:
            assert public["refused"] is None
            assert public["stages"]["inference"] == "same"
            assert (await runner.plan_run())["start"] == "reuse"
        else:
            assert public["refused"] == "inference"
            assert public["stages"]["inference"] == "differs"
            with pytest.raises(StageResumeError, match="stage `inference` differs"):
                await runner.plan_run()
    finally:
        await model.aclose()


@pytest.mark.anyio
async def test_fresh_plan_rejects_a_missing_subject_after_successful_setup(tmp_path):
    task = _MissingSubjectTask(
        MockDataset(samples=[{"question": "q", "answer": "v1"}]), setup_only=True
    )
    runner = TaskRunner(task, make_config(tmp_path))
    with pytest.raises(NotImplementedError, match="must provide subject"):
        await runner.plan_run()
    assert task._subject is not None
    assert runner.setup_unreached is None
    assert not (tmp_path / "results" / "meta.json").exists()


@pytest.mark.anyio
async def test_fresh_plan_does_not_require_a_subject_from_an_unstarted_service(
    tmp_path,
):
    task = _HandshakeSubjectTask(
        MockDataset(samples=[{"question": "q", "answer": "v1"}]), reachable=False
    )
    runner = TaskRunner(task, make_config(tmp_path))
    plan = await runner.plan_run(
        unstarted={"remote-api": "http://127.0.0.1:40123/evaluate"}
    )
    assert plan["start"] == "fresh"
    assert plan["to_infer"] == 1
    assert runner.setup_unreached is not None
    assert task.shut
    assert not (tmp_path / "results" / "meta.json").exists()


@pytest.mark.anyio
async def test_resume_plan_still_requires_a_subject_from_setup(tmp_path):
    dataset = MockDataset(samples=[{"question": "q", "answer": "v1"}])
    await TaskRunner(
        _HandshakeSubjectTask(dataset, reachable=True), make_config(tmp_path)
    ).arun()
    root = tmp_path / "results"
    recorded = (root / "meta.json").read_bytes()
    task = _HandshakeSubjectTask(dataset, reachable=False)
    runner = TaskRunner(task, make_config(tmp_path, resume=True))
    with pytest.raises(RuntimeError, match="subject has not been resolved"):
        await runner.plan_run(
            unstarted={"remote-api": "http://127.0.0.1:40123/evaluate"}
        )
    assert task.shut
    assert (root / "meta.json").read_bytes() == recorded


@pytest.mark.anyio
async def test_public_runtime_plan_none_keeps_legacy_inference_identity(
    tmp_path, monkeypatch
):
    model = _NoPublicRuntimePlanModel(default_answer="ok")
    dataset = MockDataset(samples=[{"question": "q", "answer": "ok"}])
    try:
        with monkeypatch.context() as legacy:
            legacy.setattr(
                _NoPublicRuntimePlanModel, "request_args", lambda self: {"params": {}}
            )
            await TaskRunner(
                BenchmarkTask(dataset, model), make_config(tmp_path)
            ).arun()

        assert model.request_args() == {"params": {}}
        runner = TaskRunner(
            BenchmarkTask(dataset, model), make_config(tmp_path, resume=True)
        )
        assert (await runner.plan_run())["start"] == "reuse"
    finally:
        await model.aclose()
