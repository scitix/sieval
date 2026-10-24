"""
TaskRunner end to end: stage identities and run history in ``meta.json``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import orjson
import pytest

from sieval import __version__
from sieval.core.models import ModelOutput
from sieval.core.runners.resume_gate import ResumeVersionError
from sieval.core.runners.runner import (
    ResultDirExistsError,
    TaskRunner,
    run_started,
)
from sieval.core.runners.stage_resume import StageResumeError
from sieval.core.tasks import (
    TaskStageOutput,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.run_meta import (
    RESULT_SCHEMA_VERSION,
    aggregate_digest,
    digest,
    preprocess_identity,
    preprocess_key,
)
from sieval.core.tasks.saver import TaskSaver
from sieval.core.tasks.task import Task
from sieval.core.utils import atomic as saver_module
from sieval.core.utils.meta import build_stage_meta
from tests.conftest import (
    MockChatModel,
    MockCountingChatModel,
    MockDataset,
    MockJudgeModel,
    make_config,
)

SAMPLES = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 2+3?", "answer": "5"},
    {"question": "What is 10-7?", "answer": "3"},
]
ANSWERS = {s["question"]: s["answer"] for s in SAMPLES}


class RecordTask(Task):
    """Builds a `PromptRecord`, so the prompt identity reads `prompt` alone."""

    model_type = "chat"

    def __init__(self, dataset, model, name=None, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n

    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [{"role": "user", "content": raw["question"]}], reference=raw["answer"]
        )

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    async def postprocess(self, inf, ctx):
        return build_prediction_record([t.strip() for t in inf.texts])

    async def feedback(self, post, ctx):
        gold = ctx.raw_sample["answer"]
        rollouts = [
            build_rollout_judgement(r["index"], r.get("prediction") == gold)
            for r in post["rollouts"]
        ]
        return True, build_judgement_record(gold, rollouts)

    async def report(self, finals, fails):
        return {"total": len(finals) + len(fails)}


class JudgedTask(RecordTask):
    """Grades with a judge model held as `self._grader`, like the LLM-judge tasks."""

    def __init__(self, dataset, model, grader, name=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._grader = grader

    def judge_prompts(self):
        return {"grader": "Is {answer} right?"}

    async def feedback(self, post, ctx):
        out = await self._grader.agenerate("Is it right?")
        return True, TaskStageOutput(
            value={"correct": out.texts[0] == "yes"}, meta=build_stage_meta(out)
        )


class VersionedJudge(MockJudgeModel):
    """A judge whose API reports a dated model version."""

    async def _stub_arun(self, req):
        response = await super()._stub_arun(req)
        from dataclasses import replace

        return replace(response, response_model="mock-judge-2026-10-01")


class EchoingModel(MockChatModel):
    """Echoes the request's sampling into `request_params`, as a dialect does."""

    async def _stub_arun(self, req):
        from dataclasses import replace

        response = await super()._stub_arun(req)
        params: dict = {"n": req.sampling.n}
        if req.sampling.stop is not None:
            params["stop"] = list(req.sampling.stop)
        return replace(response, request_params=params)


class StopTask(RecordTask):
    """Passes `stop` at call time, under a name `inference_args` does not read."""

    def __init__(self, dataset, model, name=None, stop=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._stop_words = stop

    def grader_args(self):
        return {}

    async def infer(self, pre, ctx):
        if self._stop_words:
            return await self.model.agenerate(pre["prompt"], stop=self._stop_words)
        return await self.model.agenerate(pre["prompt"])


class OtherModel(MockCountingChatModel):
    """A counting mock under another model name, so its inference differs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = "other-model"


def _meta(root: Path) -> dict:
    return orjson.loads((root / "meta.json").read_bytes())


def _forget_last_sample(root: Path) -> None:
    """Drop the third sample and the report, as an interrupted run leaves them."""
    (root / "report.json").unlink(missing_ok=True)
    for path in root.rglob("*.idx"):
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(ln for ln in lines if not ln.startswith("2\t")))


def _runner(tmp_path, task=None, **cfg):
    task = task or RecordTask(
        dataset=MockDataset(SAMPLES), model=MockChatModel(answers=ANSWERS), name="t"
    )
    return TaskRunner(task, make_config(tmp_path, **cfg))


# ===================================================================
# A fresh run
# ===================================================================
class TestFreshRun:
    @pytest.mark.anyio
    async def test_writes_every_field(self, tmp_path):
        runner = _runner(tmp_path)
        await runner.arun()
        meta = _meta(runner.root_dir)

        assert meta["version"] == __version__
        assert meta["deterministic"] is False
        assert meta["model"] == "mock-chat"
        identity = meta["identity"]
        assert identity["schema"] == 3
        assert identity["data"]["samples"] == 3
        # A test task lives outside the followed packages: no code, and `n`
        # is an inference argument, not a preprocess one.
        assert identity["preprocess"]["modules"] == {}
        assert identity["preprocess"]["args"] == {}
        assert identity["inference"]["digest"].startswith("sha256:")
        # A test task lives outside the followed packages: no modules, no judges.
        for stage in ("postprocess", "feedback", "report"):
            assert identity[stage]["modules"] == {}
        assert identity["feedback"]["judges"] == []
        [run] = meta["runs"]
        assert run["sieval"] == __version__
        assert run["start"] == "fresh"
        assert run["reran"] == [
            "data",
            "preprocess",
            "inference",
            "postprocess",
            "feedback",
            "report",
        ]
        assert run["ended"] >= run["at"]
        assert "interrupted" not in run
        assert "config" not in run
        assert "from" not in run

    @pytest.mark.anyio
    async def test_data_is_the_samples_and_preprocess_the_code(self, tmp_path):
        runner = _runner(tmp_path)
        await runner.arun()
        identity = _meta(runner.root_dir)["identity"]

        expected_data = aggregate_digest(
            (i, None, digest(sample)) for i, sample in enumerate(SAMPLES)
        )
        assert identity["data"] == expected_data
        assert identity["preprocess"] == preprocess_identity(
            {"modules": {}, "judges": [], "services": False, "args": []}, {}
        )

    @pytest.mark.anyio
    async def test_per_sample_digests_land_in_the_preprocess_record(self, tmp_path):
        runner = _runner(tmp_path)
        await runner.arun()

        task = RecordTask(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        contexts = await loader.load_initial_state()
        await loader.hydrate(
            contexts, set(), include_stages={TaskStage.FINAL}, record_each_stage=True
        )
        for sid, ctx in contexts.items():
            sample = SAMPLES[int(sid)]
            [entry] = ctx.stage_meta[TaskStage.PREPROCESSED.value]
            assert entry["identity"] == {
                "data": digest(sample),
                "preprocess": preprocess_key(
                    _meta(runner.root_dir)["identity"]["preprocess"]
                ),
            }

    @pytest.mark.anyio
    async def test_stage_identities_are_written_while_the_run_is_live(self, tmp_path):
        """Inference and grader are on disk before the first model call; data and
        prompt land as the last sample leaves preprocess, without holding any
        sample back (no global stage barrier), so a call made after that point
        already sees them."""
        seen: list[dict] = []

        class Spy(MockChatModel):
            async def _stub_arun(self, req):
                seen.append(_meta(Path(tmp_path / "results")))
                return await super()._stub_arun(req)

        task = RecordTask(
            dataset=MockDataset(SAMPLES), model=Spy(answers=ANSWERS), name="t"
        )
        # Preprocess every sample before the first inference is scheduled.
        runner = TaskRunner(
            task, make_config(tmp_path, concurrency_limits={"infer": 1})
        )
        await runner.arun()
        assert all(m["identity"]["inference"] is not None for m in seen)
        assert all(m["identity"]["feedback"] is not None for m in seen)
        assert all("reran" not in m["runs"][-1] for m in seen)
        assert all("ended" not in m["runs"][-1] for m in seen)
        assert seen[-1]["identity"]["data"] is not None
        assert seen[-1]["identity"]["preprocess"] is not None

    @pytest.mark.anyio
    async def test_failed_preprocess_still_counts_in_data(self, tmp_path):
        class Flaky(RecordTask):
            async def preprocess(self, raw, ctx):
                if raw["question"] == "What is 2+3?":
                    raise ValueError("bad sample")
                return await super().preprocess(raw, ctx)

        task = Flaky(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        identity = _meta(runner.root_dir)["identity"]
        assert identity["data"]["samples"] == 3
        assert identity["data"]["digest"] is not None


class TestInferenceIdentityFromRunner:
    async def _digest(self, tmp_path, name, **cfg):
        runner = _runner(tmp_path / name, **cfg)
        await runner.arun()
        return _meta(runner.root_dir)["identity"]["inference"]["digest"]

    @pytest.mark.anyio
    async def test_deterministic_counts(self, tmp_path):
        """`meta.json.deterministic` was read by nothing; now it is identity."""
        assert await self._digest(tmp_path, "a") != await self._digest(
            tmp_path, "b", deterministic=True
        )

    @pytest.mark.anyio
    async def test_n_counts(self, tmp_path):
        one = RecordTask(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        four = RecordTask(
            dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t", n=4
        )
        a = _runner(tmp_path / "a", task=one)
        b = _runner(tmp_path / "b", task=four)
        await a.arun()
        await b.arun()
        assert (
            _meta(a.root_dir)["identity"]["inference"]
            != _meta(b.root_dir)["identity"]["inference"]
        )

    @pytest.mark.anyio
    async def test_recipe_counts(self, tmp_path):
        task = RecordTask(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        runner = TaskRunner(
            task,
            make_config(tmp_path / "a"),
            inference_recipe={"backend": "vllm"},
        )
        await runner.arun()
        assert (
            _meta(runner.root_dir)["identity"]["inference"]
            != _meta(await self._root(tmp_path / "b"))["identity"]["inference"]
        )

    async def _root(self, tmp_path):
        runner = _runner(tmp_path)
        await runner.arun()
        return runner.root_dir

    @pytest.mark.anyio
    async def test_a_call_time_stop_counts(self, tmp_path):
        """A `stop` the task passes to `agenerate` is in no config the run sees
        up front; the inference identity still moves, through what was sent."""

        async def identity(name, stop):
            task = StopTask(
                dataset=MockDataset(SAMPLES), model=EchoingModel(), name="t", stop=stop
            )
            runner = _runner(tmp_path / name, task=task)
            await runner.arun()
            return _meta(runner.root_dir)["identity"]["inference"]

        plain = await identity("a", stop=None)
        stopped = await identity("b", stop=["Q:"])
        assert plain["digest"] == stopped["digest"]
        assert plain["requests"]["samples"] == 3
        assert plain["requests"]["digest"] is not None
        assert plain["requests"]["digest"] != stopped["requests"]["digest"]

    @pytest.mark.anyio
    async def test_requests_are_unknown_without_model_calls_on_record(self, tmp_path):
        """A resume of a `record_meta=False` run cannot say what was sent."""
        root = tmp_path / "results"
        await _runner(tmp_path, record_meta=False).arun()
        (root / "report.json").unlink()
        await _runner(tmp_path, record_meta=False, resume=True).arun()
        requests = _meta(root)["identity"]["inference"]["requests"]
        assert requests == {"digest": None, "samples": 3, "unknown": 3}

    @pytest.mark.anyio
    async def test_a_sample_never_inferred_has_absent_requests(self, tmp_path):
        class FailingPreprocess(RecordTask):
            async def preprocess(self, raw, ctx):
                raise ValueError("no prompt")

        task = FailingPreprocess(
            dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t"
        )
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        identity = _meta(runner.root_dir)["identity"]
        assert identity["inference"]["requests"]["digest"] is not None
        assert identity["preprocess"]["digest"] is not None


class TestJudgeIdentity:
    @pytest.mark.anyio
    async def test_judge_entry_and_reported_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SIEVAL_JUDGE_MODEL_REF", "harmony://models/judge@7")
        grader = VersionedJudge(temperature=0.0)
        grader._api_base = "http://judge-1234.internal/v1"
        task = JudgedTask(
            dataset=MockDataset(SAMPLES),
            model=MockChatModel(answers=ANSWERS),
            grader=grader,
            name="judged",
        )
        runner = _runner(tmp_path, task=task)
        await runner.arun()

        [judge] = _meta(runner.root_dir)["identity"]["feedback"]["judges"]
        assert judge == {
            "role": "grader",
            "model": "mock-judge",
            "version": "mock-judge-2026-10-01",
            "sampling": {"temperature": 0.0},
            "prompt": digest("Is {answer} right?"),
            "ref": "harmony://models/judge@7",
        }
        assert b"judge-1234" not in (runner.root_dir / "meta.json").read_bytes()

    @pytest.mark.anyio
    async def test_candidate_version_is_not_a_judge_version(self, tmp_path):
        class Reporting(MockChatModel):
            async def _stub_arun(self, req):
                from dataclasses import replace

                return replace(await super()._stub_arun(req), response_model="cand-v2")

        task = JudgedTask(
            dataset=MockDataset(SAMPLES),
            model=Reporting(answers=ANSWERS),
            grader=MockJudgeModel(),
            name="judged",
        )
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        [judge] = _meta(runner.root_dir)["identity"]["feedback"]["judges"]
        assert judge["version"] is None
        assert "ref" not in judge

    @pytest.mark.anyio
    async def test_a_judge_sharing_the_candidates_name_is_told_apart(self, tmp_path):
        """Self-judging: both models are `mock-chat`, configured differently.
        Only the judge's calls set the judge's version."""

        class Reporting(MockChatModel):
            def __init__(self, reported, **kwargs):
                super().__init__(answers=ANSWERS, **kwargs)
                self._reported = reported

            async def _stub_arun(self, req):
                from dataclasses import replace

                response = await super()._stub_arun(req)
                return replace(response, response_model=self._reported)

        task = JudgedTask(
            dataset=MockDataset(SAMPLES),
            model=Reporting("cand-v2", temperature=0.7),
            grader=Reporting("judge-v1", temperature=0.0),
            name="judged",
        )
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        [judge] = _meta(runner.root_dir)["identity"]["feedback"]["judges"]
        assert judge["model"] == "mock-chat"
        assert judge["version"] == "judge-v1"

    @pytest.mark.anyio
    async def test_a_judge_configured_like_the_candidate_has_no_version(self, tmp_path):
        """Identical configuration: the calls cannot be told apart, so the
        version stays unknown rather than borrowing the candidate's."""

        class Reporting(MockChatModel):
            async def _stub_arun(self, req):
                from dataclasses import replace

                return replace(await super()._stub_arun(req), response_model="v")

        task = JudgedTask(
            dataset=MockDataset(SAMPLES),
            model=Reporting(answers=ANSWERS),
            grader=Reporting(answers=ANSWERS),
            name="judged",
        )
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        [judge] = _meta(runner.root_dir)["identity"]["feedback"]["judges"]
        assert judge["version"] is None

    @pytest.mark.anyio
    async def test_a_resume_with_another_judge_drops_the_old_version(self, tmp_path):
        def judged(grader):
            return JudgedTask(
                dataset=MockDataset(SAMPLES),
                model=MockChatModel(answers=ANSWERS),
                grader=grader,
                name="judged",
            )

        first = _runner(tmp_path, task=judged(VersionedJudge(temperature=0.0)))
        await first.arun()
        (first.root_dir / "report.json").unlink()

        # Same name, other sampling: the recorded version describes another judge.
        second = _runner(
            tmp_path, task=judged(MockJudgeModel(temperature=1.0)), resume=True
        )
        await second.arun()
        [judge] = _meta(second.root_dir)["identity"]["feedback"]["judges"]
        assert judge["sampling"] == {"temperature": 1.0}
        assert judge["version"] is None


class TestIdentityUnavailable:
    @pytest.mark.anyio
    async def test_an_undigestable_sample_is_unknown_not_a_failure(self, tmp_path):
        """Unknown is not failure: a sample digest that cannot be taken is
        recorded as unknown and the run carries on."""

        class Opaque:
            pass

        class OpaqueTask(RecordTask):
            async def preprocess(self, raw, ctx):
                record = await super().preprocess(raw, ctx)
                record["prompt"] = [Opaque()]
                return record

            async def infer(self, pre, ctx):
                return await self.model.agenerate("What is 1+1?")

        task = OpaqueTask(
            dataset=MockDataset(SAMPLES[:1]), model=MockChatModel(), name="t"
        )
        runner = _runner(tmp_path, task=task, record_meta=False)
        await runner.arun()
        identity = _meta(runner.root_dir)["identity"]
        # Preprocess is code, so a prompt that will not digest leaves it known.
        assert identity["preprocess"]["digest"].startswith("sha256:")
        assert identity["data"]["digest"] is not None

    @pytest.mark.anyio
    async def test_a_broken_judge_lookup_leaves_the_grader_unknown(self, tmp_path):
        class Broken(RecordTask):
            def judge_models(self):
                raise TypeError("no judges here")

            def inference_args(self):
                raise TypeError("no args here")

        task = Broken(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        identity = _meta(runner.root_dir)["identity"]
        assert all(identity[s] is None for s in ("postprocess", "feedback", "report"))
        assert identity["inference"] is None
        assert identity["data"]["samples"] == 3
        # Identity is bookkeeping: no sample fails over it.
        finals, fails = runner._final_and_failed()
        assert (len(finals), len(fails)) == (3, 0)


# ===================================================================
# Resume
# ===================================================================
class TestResume:
    @pytest.mark.anyio
    async def test_fast_resume_appends_a_run_that_reran_nothing(self, tmp_path):
        first = _runner(tmp_path)
        await first.arun()
        identity = _meta(first.root_dir)["identity"]

        second = _runner(tmp_path, resume=True)
        await second.arun()
        meta = _meta(second.root_dir)
        assert [r["start"] for r in meta["runs"]] == ["fresh", "resume"]
        assert meta["runs"][1]["reran"] == []
        assert "ended" in meta["runs"][1]
        assert meta["identity"] == identity

    @pytest.mark.anyio
    async def test_a_resume_with_another_model_keeps_the_recorded_one(self, tmp_path):
        """Not refused, but not a fast resume either: `model` stays the one
        whose samples are on disk, the invoking model goes in `runs[]`, and what
        was sent is unknown rather than the old model's requests."""
        await _runner(tmp_path).arun()
        root = tmp_path / "results"

        class Other(MockCountingChatModel):
            def __init__(self):
                super().__init__(answers=ANSWERS)
                self._model = "other-model"

            def meta(self):
                return {**super().meta(), "model": "other-model"}

        model = Other()
        reports: list[int] = []
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        original = task.report

        async def _spy(finals, fails):
            reports.append(len(finals))
            return await original(finals, fails)

        setattr(task, "report", _spy)  # noqa: B010
        await _runner(tmp_path, task=task, resume=True).arun()
        meta = _meta(root)
        assert model.call_count == 0
        assert reports == [3]  # not fast-resumed
        assert meta["model"] == "mock-chat"
        assert meta["runs"][-1]["model"] == "other-model"
        assert "model" not in meta["runs"][0]
        assert meta["identity"]["inference"]["requests"]["digest"] is None

    @pytest.mark.anyio
    async def test_a_resume_with_another_model_that_would_infer_is_refused(
        self, tmp_path
    ):
        """Inferring the one sample left would score two models under one name."""
        root = tmp_path / "results"
        await _runner(tmp_path).arun()
        _forget_last_sample(root)
        before = (root / "meta.json").read_bytes()

        class Other(MockCountingChatModel):
            def meta(self):
                return {**super().meta(), "model": "other-model"}

        model = Other(answers=ANSWERS)
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        runner = _runner(tmp_path, task=task, resume=True)
        with pytest.raises(StageResumeError, match="1 sample"):
            await runner.arun()
        assert model.call_count == 0
        assert (root / "meta.json").read_bytes() == before

    @pytest.mark.anyio
    async def test_resume_reruns_only_what_was_left(self, tmp_path):
        """Two samples finished before the interruption; the resume reruns one
        from the start, and the identity covers all three."""
        root = tmp_path / "results"
        fresh = _runner(tmp_path)
        await fresh.arun()
        full = _meta(root)["identity"]

        _forget_last_sample(root)
        model = MockCountingChatModel(answers=ANSWERS)
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        resumed = _runner(tmp_path, task=task, resume=True)
        await resumed.arun()

        meta = _meta(root)
        assert model.call_count == 1
        assert meta["runs"][-1]["start"] == "resume"
        assert meta["runs"][-1]["reran"] == [
            "data",
            "preprocess",
            "inference",
            "postprocess",
            "feedback",
            "report",
        ]
        assert meta["identity"]["data"] == full["data"]
        assert meta["identity"]["preprocess"] == full["preprocess"]

    @pytest.mark.anyio
    async def test_judge_version_survives_a_resume_that_grades_nothing(self, tmp_path):
        def judged():
            return JudgedTask(
                dataset=MockDataset(SAMPLES),
                model=MockChatModel(answers=ANSWERS),
                grader=VersionedJudge(),
                name="judged",
            )

        first = _runner(tmp_path, task=judged())
        await first.arun()
        (first.root_dir / "report.json").unlink()

        second = _runner(tmp_path, task=judged(), resume=True)
        await second.arun()
        meta = _meta(second.root_dir)
        # Only `report()` ran.
        assert meta["runs"][-1]["reran"] == ["report"]
        assert meta["runs"][-1]["reported"] is True
        [judge] = meta["identity"]["feedback"]["judges"]
        assert judge["version"] == "mock-judge-2026-10-01"

    @pytest.mark.anyio
    async def test_config_diff_lands_in_runs(self, tmp_path):
        await _runner(tmp_path).arun()
        task = RecordTask(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        runner = TaskRunner(
            task,
            make_config(tmp_path, resume=True),
            resume_config_diff=["concurrency_limit: 8 → 2"],
        )
        await runner.arun()
        assert _meta(runner.root_dir)["runs"][-1]["config"] == [
            "concurrency_limit: 8 → 2"
        ]

    @pytest.mark.anyio
    async def test_resume_keeps_the_origin_fields(self, tmp_path):
        first = _runner(tmp_path, deterministic=True)
        await first.arun()
        meta = _meta(first.root_dir)
        meta["version"] = __version__  # same series, so the gate admits it
        (first.root_dir / "meta.json").write_bytes(orjson.dumps(meta))

        second = _runner(tmp_path, resume=True, deterministic=True)
        await second.arun()
        assert _meta(second.root_dir)["deterministic"] is True


class TestInterrupt:
    @pytest.mark.anyio
    async def test_interrupted_run_is_ended_without_reran(self, tmp_path, monkeypatch):
        runner = _runner(tmp_path)
        # Interrupt once the run has begun, the way Ctrl+C reaches `arun`.
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(side_effect=KeyboardInterrupt),
        )
        assert await runner.arun() is None
        run = _meta(runner.root_dir)["runs"][-1]
        assert run["interrupted"] is True
        assert "ended" in run
        assert "reran" not in run


class TestRunStartedHere:
    @pytest.mark.anyio
    async def test_killed_before_first_shard_flush_is_not_fresh(self, tmp_path):
        """A run killed after writing `meta.json` but before any shard flush
        has started: the next run must refuse or resume it, not overwrite it."""
        root = tmp_path / "results"
        saver = TaskSaver(root_dir=root)
        await saver.begin_run(resumed=False, model="mock-chat")
        assert not list(root.rglob("*.idx"))
        assert run_started(root)

        with pytest.raises(ResultDirExistsError):
            _runner(tmp_path)

        resumed = _runner(tmp_path, resume=True)
        await resumed.arun()
        meta = _meta(root)
        assert [r["start"] for r in meta["runs"]] == ["fresh", "resume"]
        # The killed run never said it ended.
        assert "reran" not in meta["runs"][0]
        assert "ended" not in meta["runs"][0]
        assert meta["runs"][1]["reran"] == [
            "data",
            "preprocess",
            "inference",
            "postprocess",
            "feedback",
            "report",
        ]

    def test_a_pre_runs_directory_is_started_by_its_meta(self, tmp_path):
        """Before `runs[]`, `meta.json` was written once, when a run started:
        its presence is the signal. An empty `runs[]` is not one."""
        root = tmp_path / "legacy"
        root.mkdir()
        assert not run_started(root)
        (root / "meta.json").write_bytes(
            orjson.dumps(
                {
                    "version": __version__,
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "deterministic": False,
                }
            )
        )
        assert run_started(root)
        (root / "meta.json").write_bytes(
            orjson.dumps(
                {
                    "version": __version__,
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "runs": [],
                }
            )
        )
        assert not run_started(root)

    def test_empty_directory_has_not_started(self, tmp_path):
        assert not run_started(tmp_path)


class TestShardsMeanStarted:
    """Shards count as a started run whatever ``meta.json`` says; a fresh run
    must never adopt them."""

    @staticmethod
    async def _shards_under(tmp_path, meta: str) -> tuple[Path, dict]:
        """A finished run's shards, no report, and ``meta.json`` with empty
        ``runs``, removed, or from before ``runs[]``."""
        first = _runner(tmp_path)
        await first.arun()
        root = first.root_dir
        recorded = _meta(root)
        (root / "report.json").unlink()
        if meta == "runs-empty":
            (root / "meta.json").write_bytes(orjson.dumps({**recorded, "runs": []}))
        elif meta == "missing":
            (root / "meta.json").unlink()
        else:
            (root / "meta.json").write_bytes(
                orjson.dumps(
                    {
                        "version": __version__,
                        "result_schema_version": RESULT_SCHEMA_VERSION,
                        "deterministic": False,
                    }
                )
            )
        assert list(root.rglob("*.idx"))
        return root, recorded

    @staticmethod
    def _other_model_runner(tmp_path, **cfg):
        model = OtherModel(default_answer="zzz")
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        return _runner(tmp_path, task=task, **cfg), model

    @pytest.mark.anyio
    @pytest.mark.parametrize("meta", ["runs-empty", "missing", "pre-runs"])
    async def test_a_fresh_run_refuses_shards(self, tmp_path, meta):
        root, _ = await self._shards_under(tmp_path, meta)
        before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        assert run_started(root)
        with pytest.raises(ResultDirExistsError) as exc:
            self._other_model_runner(tmp_path)
        assert exc.value.path == root
        assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before

    @pytest.mark.anyio
    async def test_runs_empty_with_shards_resumes(self, tmp_path):
        root, recorded = await self._shards_under(tmp_path, "runs-empty")
        model = MockCountingChatModel(answers=ANSWERS)
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        await _runner(tmp_path, task=task, resume=True).arun()
        assert model.call_count == 0
        meta = _meta(root)
        assert [r["start"] for r in meta["runs"]] == ["resume"]
        assert meta["runs"][0]["reran"] == ["report"]
        assert meta["identity"]["inference"] == recorded["identity"]["inference"]

    @pytest.mark.anyio
    async def test_runs_empty_with_shards_keeps_them_the_recorded_models(
        self, tmp_path
    ):
        root, recorded = await self._shards_under(tmp_path, "runs-empty")
        runner, model = self._other_model_runner(tmp_path, resume=True)
        await runner.arun()
        assert model.call_count == 0
        assert _meta(root)["model"] == recorded["model"] == "mock-chat"

    @pytest.mark.anyio
    async def test_runs_empty_with_shards_refuses_to_mix_models(self, tmp_path):
        root, _ = await self._shards_under(tmp_path, "runs-empty")
        _forget_last_sample(root)
        runner, model = self._other_model_runner(tmp_path, resume=True)
        with pytest.raises(StageResumeError, match="'other-model'") as refused:
            await runner.arun()
        assert (refused.value.stage, refused.value.task) == ("inference", "t")
        assert model.call_count == 0
        assert _meta(root)["runs"] == []

    @pytest.mark.anyio
    async def test_missing_meta_with_shards_fails_closed(self, tmp_path):
        root, _ = await self._shards_under(tmp_path, "missing")
        with pytest.raises(ResumeVersionError) as exc:
            self._other_model_runner(tmp_path, resume=True)
        assert exc.value.task == "t"
        message = str(exc.value)
        assert str(root) in message
        assert "another result_dir" in message
        assert not (root / "meta.json").exists()

    @pytest.mark.anyio
    async def test_a_pre_runs_meta_with_shards_resumes(self, tmp_path):
        root, _ = await self._shards_under(tmp_path, "pre-runs")
        runner = _runner(tmp_path, resume=True)
        await runner.arun()
        assert [r["start"] for r in _meta(root)["runs"]] == ["resume"]

    @pytest.mark.anyio
    async def test_a_stray_index_outside_a_stage_dir_is_not_a_run(self, tmp_path):
        """Only SiEval's stage directories hold shards: an ``.idx`` elsewhere
        under ``<iteration>/`` neither starts a run nor is loaded by one."""
        root = tmp_path / "results"
        (root / "0" / "backup").mkdir(parents=True)
        (root / "0" / "backup" / "x.idx").write_text("not\tan\tindex\n")
        assert not run_started(root)
        runner = _runner(tmp_path)
        await runner.arun()
        meta = _meta(root)
        assert [r["start"] for r in meta["runs"]] == ["fresh"]
        assert meta["runs"][0]["reran"] == [
            "data",
            "preprocess",
            "inference",
            "postprocess",
            "feedback",
            "report",
        ]
        assert (root / "0" / "backup" / "x.idx").read_text() == "not\tan\tindex\n"

    @pytest.mark.anyio
    async def test_an_empty_directory_runs_fresh(self, tmp_path):
        root = tmp_path / "results"
        (root / "0" / "final").mkdir(parents=True)
        assert not run_started(root)
        runner = _runner(tmp_path)
        await runner.arun()
        assert [r["start"] for r in _meta(root)["runs"]] == ["fresh"]


class TestBackwardCompatibleRead:
    @pytest.mark.anyio
    async def test_resume_from_a_pre_identity_meta(self, tmp_path):
        """An older run's `meta.json` has neither `identity` nor `runs`; a
        resume reads it, keeps its origin and starts the history there."""
        first = _runner(tmp_path)
        await first.arun()
        root = first.root_dir
        (root / "meta.json").write_bytes(
            orjson.dumps(
                {
                    "version": __version__,
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "deterministic": False,
                }
            )
        )
        (root / "report.json").unlink()

        resumed = _runner(tmp_path, resume=True)
        await resumed.arun()
        meta = _meta(root)
        assert "task" not in meta
        assert [r["start"] for r in meta["runs"]] == ["resume"]
        # Every sample was already final: only `report()` ran; data is
        # digested from the samples, preprocess from the code.
        assert meta["runs"][0]["reran"] == ["report"]
        assert meta["runs"][0]["reported"] is True
        assert meta["identity"]["data"]["samples"] == 3
        assert meta["identity"]["preprocess"]["digest"].startswith("sha256:")


# ===================================================================
# Failed writes fail the task
# ===================================================================
class TestFailedWrites:
    @staticmethod
    def _fail_replace_of(monkeypatch, name, error, *, after=0):
        """Make the atomic replace onto *name* raise *error* after *after* calls."""
        real_replace = saver_module.os.replace
        calls = {"n": 0}

        def _replace(src, dst):
            if Path(dst).name == name:
                calls["n"] += 1
                if calls["n"] > after:
                    raise error
            return real_replace(src, dst)

        monkeypatch.setattr(saver_module.os, "replace", _replace)

    @pytest.mark.anyio
    async def test_meta_write_failure_fails_the_task(self, tmp_path, monkeypatch):
        self._fail_replace_of(monkeypatch, "meta.json", OSError("read-only fs"))
        with pytest.raises(OSError, match="read-only"):
            await _runner(tmp_path).arun()

    @pytest.mark.anyio
    async def test_report_write_failure_fails_the_task(self, tmp_path, monkeypatch):
        self._fail_replace_of(monkeypatch, "report.json", OSError("disk full"))
        runner = _runner(tmp_path)
        with pytest.raises(OSError, match="disk full"):
            await runner.arun()
        # The run never ended: its entry has neither `ended` nor `reran`.
        run = _meta(runner.root_dir)["runs"][-1]
        assert "reran" not in run
        assert "ended" not in run

    @pytest.mark.anyio
    async def test_a_meta_write_failing_mid_run_keeps_finished_samples(
        self, tmp_path, monkeypatch
    ):
        """The data and prompt identity write runs inside a stage worker. When it
        fails the task still fails, but every sample that finished reaches disk,
        whether the storage worker had buffered it or it was still in the
        stream, instead of dying with the storage worker."""
        # Few enough samples that every record fits in the storage stream's
        # buffer, so the stalled storage worker never blocks the dispatcher.
        samples = [{"question": f"q{i}", "answer": str(i)} for i in range(4)]
        last = len(samples) - 1
        # The first write is `begin_run`; the second is the input identity,
        # made once the last sample is past preprocess.
        self._fail_replace_of(monkeypatch, "meta.json", OSError("EIO"), after=1)

        async def _stalled(*_args):
            # A storage worker that has not reached the stream yet: everything
            # sent to storage is still in it when the run stops.
            await anyio.sleep_forever()

        monkeypatch.setattr(TaskSaver, "consume_stream", _stalled)

        class HeldTask(RecordTask):
            async def preprocess(self, raw, ctx):
                # Hold the last sample until every other one has finished, so
                # the failing write lands after exactly those.
                if ctx.sample_id == last:
                    while (
                        sum(c.is_terminal() for c in runner._contexts.values()) < last
                    ):
                        await anyio.sleep(0.001)
                return await super().preprocess(raw, ctx)

        task = HeldTask(dataset=MockDataset(samples), model=MockChatModel(), name="t")
        runner = TaskRunner(task, make_config(tmp_path))
        with pytest.raises(ExceptionGroup) as raised:
            await runner.arun()
        assert raised.group_contains(OSError, match="EIO")

        finished = {s for s, c in runner._contexts.items() if c.is_terminal()}
        assert finished == set(range(last))
        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        on_disk = await loader.load_initial_state()
        assert {s for s, c in on_disk.items() if c.is_terminal()} == finished
        assert run_started(runner.root_dir)

    @pytest.mark.anyio
    async def test_an_interrupt_under_backpressure_keeps_finished_samples(
        self, tmp_path, monkeypatch
    ):
        """With a slow storage worker and one-slot streams, finished samples
        wait in the storage stream, in the compute stream's buffer and in sends
        blocked on either. An interrupt must still leave every one on disk."""
        samples = [{"question": f"q{i}", "answer": str(i)} for i in range(12)]

        async def _slow(self, stream):
            async with stream:
                async for ctx in stream:
                    self._stage_queue.append(ctx)
                    await anyio.sleep(0.01)

        monkeypatch.setattr(TaskSaver, "consume_stream", _slow)
        graded: list[int] = []

        class CountingTask(RecordTask):
            async def feedback(self, post, ctx):
                # No checkpoint between here and the stage's send: a sample in
                # `graded` is finished.
                graded.append(ctx.sample_id)
                return await super().feedback(post, ctx)

        task = CountingTask(
            dataset=MockDataset(samples), model=MockChatModel(), name="t"
        )
        runner = TaskRunner(task, make_config(tmp_path))
        runner._stream_buffer_capacity = 1
        async with anyio.create_task_group() as tg:
            tg.start_soon(runner.arun)
            while len(graded) < 6:
                await anyio.sleep(0.001)
            tg.cancel_scope.cancel()

        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        on_disk = await loader.load_initial_state()
        assert {s for s, c in on_disk.items() if c.is_terminal()} == set(graded)


class TestEndRunFailure:
    """A run whose `report.json` landed but whose `runs[]` entry never ended is
    not complete; the next resume finalizes it again instead of reusing it."""

    @staticmethod
    async def _end_run_fails(runner):
        async def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        runner._saver.end_run = _boom
        with pytest.raises(OSError, match="disk full"):
            await runner.arun()
        assert (runner.root_dir / "report.json").exists()
        assert "ended" not in _meta(runner.root_dir)["runs"][-1]

    async def _assert_refinalized(self, tmp_path):
        reports: list = []
        model = MockCountingChatModel(answers=ANSWERS)
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        original = task.report

        async def _spy(finals, fails):
            reports.append(len(finals))
            return await original(finals, fails)

        # On the instance: a subclass would be another grader, and regrade.
        setattr(task, "report", _spy)  # noqa: B010
        resumed = _runner(tmp_path, task=task, resume=True)
        await resumed.arun()
        meta = _meta(resumed.root_dir)
        assert reports == [3]  # `report()` reran on the recorded samples
        assert model.call_count == 0  # nothing was inferred again
        last = meta["runs"][-1]
        assert last["start"] == "resume"
        assert last["reran"] == ["report"]
        assert last["reported"] is True
        assert meta["identity"]["inference"]["requests"]["samples"] == 3

    @pytest.mark.anyio
    async def test_full_run(self, tmp_path):
        await self._end_run_fails(_runner(tmp_path))
        await self._assert_refinalized(tmp_path)

    @pytest.mark.anyio
    async def test_all_terminal_early_exit(self, tmp_path):
        first = _runner(tmp_path)
        await first.arun()
        (first.root_dir / "report.json").unlink()
        await self._end_run_fails(_runner(tmp_path, resume=True))
        await self._assert_refinalized(tmp_path)

    @pytest.mark.anyio
    async def test_fast_resume(self, tmp_path):
        await _runner(tmp_path).arun()
        await self._end_run_fails(_runner(tmp_path, resume=True))
        await self._assert_refinalized(tmp_path)

    @pytest.mark.anyio
    async def test_refinalizing_writes_the_anomaly_report(self, tmp_path):
        """The rerun that finalizes a failed `end_run` takes the all-terminal
        exit, which writes `anomalies.json` and records its rules hash."""
        await self._end_run_fails(_runner(tmp_path, detect_anomalies=True))
        root = tmp_path / "results"
        (root / "anomalies.json").unlink()
        await _runner(tmp_path, resume=True, detect_anomalies=True).arun()
        anomalies = orjson.loads((root / "anomalies.json").read_bytes())
        assert "summary" not in anomalies
        last = _meta(root)["runs"][-1]
        assert last["rules_hash"] == anomalies["meta"]["rules_hash"]

    @pytest.mark.anyio
    async def test_an_ended_run_is_still_fast_resumed(self, tmp_path):
        await _runner(tmp_path).arun()
        model = MockCountingChatModel(answers=ANSWERS)
        task = RecordTask(dataset=MockDataset(SAMPLES), model=model, name="t")
        reports: list[int] = []
        original = task.report

        async def _spy(finals, fails):
            reports.append(len(finals))
            return await original(finals, fails)

        setattr(task, "report", _spy)  # noqa: B010
        runner = _runner(tmp_path, task=task, resume=True)
        await runner.arun()
        assert reports == []
        assert model.call_count == 0
        assert _meta(runner.root_dir)["runs"][-1]["reran"] == []


class TestReported:
    @pytest.mark.anyio
    async def test_a_report_of_none_is_not_reported(self, tmp_path):
        class NoReport(RecordTask):
            async def report(self, finals, fails):
                return None

        task = NoReport(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        runner = _runner(tmp_path, task=task)
        await runner.arun()
        assert not (runner.root_dir / "report.json").exists()
        run = _meta(runner.root_dir)["runs"][-1]
        assert "ended" in run
        assert "reported" not in run


class TestJudgeModelRef:
    @pytest.mark.anyio
    @pytest.mark.parametrize("bad", ['{"grader": ', '["hm://m/1"]', '{"grader": 1}'])
    async def test_a_malformed_ref_refuses_the_run(self, tmp_path, monkeypatch, bad):
        """A bad value used to erase the grader identity silently, and a JSON
        array passed as one plain reference."""
        monkeypatch.setenv("SIEVAL_JUDGE_MODEL_REF", bad)
        with pytest.raises(ValueError, match="SIEVAL_JUDGE_MODEL_REF"):
            _runner(tmp_path)
        assert not (tmp_path / "results").exists()


def test_model_output_is_imported_for_type_registry():
    # The loader rebuilds ModelOutput records by type; keep the import live.
    assert ModelOutput.__name__ == "ModelOutput"
