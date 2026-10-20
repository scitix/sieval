"""
Per-stage resume: each row of the resume table, resume across tasks, and a
resume from a directory recorded before stage identities.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
import stat
from pathlib import Path

import anyio
import orjson
import pytest
from loguru import logger

from sieval.core.runners.runner import TaskRunner
from sieval.core.runners.stage_resume import (
    StageResumeError,
    copy_recorded_dir,
    recorded_identity,
)
from sieval.core.tasks import (
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.task import Task
from tests.conftest import MockCountingChatModel, MockDataset, make_config

SAMPLES = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 2+3?", "answer": "5"},
    {"question": "What is 10-7?", "answer": "3"},
]
ANSWERS = {s["question"]: s["answer"] for s in SAMPLES}


class QATask(Task):
    """Counts each stage call per instance, so a test can see what reran.

    A resume calls ``preprocess`` once per recorded sample to digest the
    prompt this run would build; that is a comparison, not a rerun, and
    model calls tell the two apart.
    """

    model_type = "chat"

    def __init__(
        self,
        dataset,
        model,
        name=None,
        n: int = 1,
        prefix: str = "",
        threshold: float = 0.5,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._prefix = prefix
        self._threshold = threshold
        self.calls = {"preprocess": 0, "postprocess": 0, "feedback": 0, "report": 0}

    async def preprocess(self, raw, ctx):
        self.calls["preprocess"] += 1
        return build_prompt_record(
            [{"role": "user", "content": self._prefix + raw["question"]}],
            reference=raw["answer"],
        )

    async def infer(self, pre, ctx):
        question = pre["prompt"][0]["content"].removeprefix(self._prefix)
        return await self.model.agenerate(question, n=self._n)

    async def postprocess(self, inf, ctx):
        self.calls["postprocess"] += 1
        return build_prediction_record([t.strip() for t in inf.texts])

    async def feedback(self, post, ctx):
        self.calls["feedback"] += 1
        gold = ctx.raw_sample["answer"]
        rollouts = [
            build_rollout_judgement(r["index"], r.get("prediction") == gold)
            for r in post["rollouts"]
        ]
        return True, build_judgement_record(gold, rollouts)

    async def report(self, finals, fails):
        self.calls["report"] += 1
        return {"total": len(finals) + len(fails), "threshold": self._threshold}


class StrictQATask(QATask):
    """Another task over the same data, prompt and inference: only grading
    differs. Lives in this module, so only the task class tells it apart."""

    async def report(self, finals, fails):
        report = await super().report(finals, fails)
        return {**report, "strict": True}


def _meta(root: Path) -> dict:
    return orjson.loads((root / "meta.json").read_bytes())


def _task(cls=QATask, samples=SAMPLES, **kwargs):
    model = MockCountingChatModel(answers=ANSWERS)
    return cls(dataset=MockDataset(samples), model=model, name="qa", **kwargs), model


async def _run(tmp_path, name, task, **cfg):
    runner = TaskRunner(
        task, make_config(tmp_path, result_dir=str(tmp_path / name), **cfg)
    )
    return runner, await runner.arun()


async def _first(tmp_path, name="first", record_meta=True, **kwargs):
    task, _ = _task(**kwargs)
    runner, _ = await _run(tmp_path, name, task, record_meta=record_meta)
    return runner.root_dir


# ===================================================================
# Same directory: each row of the resume table
# ===================================================================
class TestSameDirectory:
    @pytest.mark.anyio
    async def test_unchanged_finished_run_fast_resumes(self, tmp_path):
        await _first(tmp_path)
        task, model = _task()
        runner, _ = await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 0
        assert task.calls["report"] == 0
        assert _meta(runner.root_dir)["runs"][-1]["reran"] == []

    @pytest.mark.anyio
    async def test_an_argument_resolved_in_setup_still_fast_resumes(self, tmp_path):
        """The fast resume compares the grader after `setup()`, as SciCode
        resolves its data file there."""

        class Resolving(QATask):
            def __init__(self, dataset, model, name=None, mode: str | None = None):
                super().__init__(dataset, model, name=name)
                self._mode = mode

            async def setup(self):
                if self._mode is None:
                    self._mode = "strict"

        task, _ = _task(cls=Resolving)
        await _run(tmp_path, "first", task)
        task, model = _task(cls=Resolving)
        runner, _ = await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 0
        assert task.calls["report"] == 0
        assert _meta(runner.root_dir)["runs"][-1]["reran"] == []

    @pytest.mark.anyio
    async def test_an_excluded_argument_does_not_regrade(self, tmp_path):
        class Workers(QATask):
            GRADER_ARGS_EXCLUDED = QATask.GRADER_ARGS_EXCLUDED | {"workers"}

            def __init__(self, dataset, model, name=None, workers: int = 4):
                super().__init__(dataset, model, name=name)
                self._workers = workers

        task, _ = _task(cls=Workers)
        await _run(tmp_path, "first", task)
        task, _ = _task(cls=Workers, workers=8)
        await _run(tmp_path, "first", task, auto_resume=True)
        assert task.calls["postprocess"] == 0
        assert task.calls["report"] == 0

    @pytest.mark.anyio
    async def test_a_grader_difference_regrades_on_recorded_inference(self, tmp_path):
        """Supported regrading: no shard is deleted by hand."""
        root = await _first(tmp_path)
        task, model = _task(threshold=0.9)
        _, report = await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 0
        assert task.calls["postprocess"] == 3
        assert task.calls["feedback"] == 3
        assert report["threshold"] == 0.9
        meta = _meta(root)
        assert meta["runs"][-1]["reran"] == ["grader"]
        assert meta["identity"]["grader"]["args"]["threshold"] == 0.9

        # The regrade is now the record: a third run fast-resumes.
        again, model = _task(threshold=0.9)
        await _run(tmp_path, "first", again, auto_resume=True)
        assert model.call_count == 0
        assert again.calls["feedback"] == 0
        assert _meta(root)["runs"][-1]["reran"] == []

    @pytest.mark.anyio
    async def test_a_partial_regrade_resumes_without_regrading_twice(self, tmp_path):
        """Samples a regrade reached carry the new grader's stamp; a resume
        retries only the ones it did not finish, from where they stopped."""
        root = await _first(tmp_path)
        task, _ = _task(threshold=0.9)
        original = task.feedback

        async def _feedback(post, ctx):
            if ctx.sample_id != 0:
                raise RuntimeError("judge down")
            return await original(post, ctx)

        # On the instance: a subclass would be another grader.
        setattr(task, "feedback", _feedback)  # noqa: B010
        await _run(tmp_path, "first", task, auto_resume=True)

        resumed, model = _task(threshold=0.9)
        await _run(tmp_path, "first", resumed, auto_resume=True)
        assert model.call_count == 0
        assert resumed.calls["postprocess"] == 0
        assert resumed.calls["feedback"] == 2
        assert _meta(root)["runs"][-1]["reran"] == ["grader"]

    @pytest.mark.anyio
    async def test_an_inference_difference_is_refused_naming_the_stage(self, tmp_path):
        root = await _first(tmp_path)
        before = (root / "meta.json").read_bytes()
        task, model = _task(n=2)
        with pytest.raises(StageResumeError, match="stage `inference`") as raised:
            await _run(tmp_path, "first", task, auto_resume=True)
        assert raised.value.stage == "inference"
        assert model.call_count == 0
        # A refusal leaves the recorded run as it was.
        assert (root / "meta.json").read_bytes() == before

    @pytest.mark.anyio
    async def test_a_prompt_difference_reruns_those_samples_from_the_start(
        self, tmp_path
    ):
        root = await _first(tmp_path)
        task, model = _task(prefix="Q: ")
        await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 3
        assert _meta(root)["runs"][-1]["reran"] == [
            "data",
            "prompt",
            "inference",
            "grader",
        ]

    @pytest.mark.anyio
    async def test_a_data_difference_reruns_only_the_changed_samples(self, tmp_path):
        root = await _first(tmp_path)
        changed = [*SAMPLES[:2], {"question": "What is 1+1?", "answer": "3"}]
        task, model = _task(samples=changed)
        _, report = await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 1
        assert report["total"] == 3
        assert _meta(root)["identity"]["data"]["samples"] == 3

    @pytest.mark.anyio
    async def test_a_sample_subset_drops_the_samples_left_out(self, tmp_path):
        root = await _first(tmp_path)
        task, model = _task(samples=SAMPLES[:2])
        _, report = await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 0
        assert report["total"] == 2
        assert _meta(root)["identity"]["data"]["samples"] == 2

        # Widening again reuses the dropped sample's records: its data,
        # prompt and inference still match.
        widened, model = _task()
        _, report = await _run(tmp_path, "first", widened, auto_resume=True)
        assert model.call_count == 0
        assert report["total"] == 3
        assert _meta(root)["identity"]["data"]["samples"] == 3


# ===================================================================
# Resume across tasks
# ===================================================================
class TestResumeFrom:
    @pytest.mark.anyio
    async def test_another_task_reruns_only_grading(self, tmp_path):
        src = await _first(tmp_path)
        source_meta = _meta(src)
        task, model = _task(cls=StrictQATask)
        runner, report = await _run(
            tmp_path,
            "second",
            task,
            resume_from=str(src),
            profile_stages=True,
        )
        assert model.call_count == 0
        assert task.calls["feedback"] == 3
        assert report["strict"] is True

        meta = _meta(runner.root_dir)
        [run] = meta["runs"]
        assert run["start"] == "from"
        assert run["reran"] == ["grader"]
        assert run["from"] == {"dir": str(src), "identity": source_meta["identity"]}
        assert (
            meta["identity"]["inference"]["digest"]
            == (source_meta["identity"]["inference"]["digest"])
        )
        assert (
            meta["identity"]["grader"]["digest"]
            != (source_meta["identity"]["grader"]["digest"])
        )
        # profile.json aggregates every sample's recorded stage meta, reused
        # ones included, so what reran is told by `reran` and the calls above.
        profile = orjson.loads((runner.root_dir / "profile.json").read_bytes())
        assert profile["stages"]["feedback"]["count"] == 3
        # The source is untouched.
        assert _meta(src) == source_meta

    @pytest.mark.anyio
    async def test_a_read_only_source_is_copied(self, tmp_path):
        src = await _first(tmp_path)
        for dirpath, _dirs, files in os.walk(src):
            for name in files:
                os.chmod(Path(dirpath) / name, stat.S_IRUSR)
        try:
            task, _ = _task(cls=StrictQATask)
            runner, _ = await _run(tmp_path, "second", task, resume_from=str(src))
            assert _meta(runner.root_dir)["runs"][0]["start"] == "from"
        finally:
            for dirpath, _dirs, files in os.walk(src):
                for name in files:
                    os.chmod(Path(dirpath) / name, stat.S_IRUSR | stat.S_IWUSR)

    @pytest.mark.anyio
    async def test_an_inference_difference_is_refused_and_nothing_is_left(
        self, tmp_path
    ):
        src = await _first(tmp_path)
        task, model = _task(cls=StrictQATask, n=2)
        with pytest.raises(StageResumeError, match="stage `inference`"):
            await _run(tmp_path, "second", task, resume_from=str(src))
        assert model.call_count == 0
        dst = tmp_path / "second"
        assert not dst.exists() or not any(dst.iterdir())

    @pytest.mark.anyio
    async def test_a_from_run_is_resumed_in_place(self, tmp_path):
        """Interrupted after the copy: the same config resumes the copy, not
        the source again."""
        src = await _first(tmp_path)
        first, _ = _task(cls=StrictQATask)
        await _run(tmp_path, "second", first, resume_from=str(src))
        again, model = _task(cls=StrictQATask)
        runner, _ = await _run(tmp_path, "second", again, resume_from=str(src))
        assert model.call_count == 0
        assert [r["start"] for r in _meta(runner.root_dir)["runs"]] == [
            "from",
            "resume",
        ]

    def test_the_copy_holds_only_shards(self, tmp_path):
        src = tmp_path / "src"
        (src / "0" / "final").mkdir(parents=True)
        (src / "0" / "final" / "0.jsonl").write_text("{}\n")
        (src / "0" / "final" / "0.idx").write_text(
            "0\t0\tfinal\t0\t3\t\t\t0\t4\n1\t0\tfinal\t0\t3\t\t\t0\t2\n"
            "2\t0\tfinal\t0\t3\t\t\t0\n"
        )
        (src / "meta.json").write_text("{}")
        (src / "report.json").write_text("{}")
        (src / "logs").mkdir()
        dst = tmp_path / "dst"
        copy_recorded_dir(src, dst)
        assert sorted(p.name for p in dst.iterdir()) == ["0"]
        # Shifted under the new run's ordinals, keeping their order; a line
        # without the column is run 0.
        assert (dst / "0" / "final" / "0.idx").read_text() == (
            "0\t0\tfinal\t0\t3\t\t\t0\t0\n1\t0\tfinal\t0\t3\t\t\t0\t-2\n"
            "2\t0\tfinal\t0\t3\t\t\t0\t-4\n"
        )

    @pytest.mark.anyio
    async def test_an_interrupted_regrade_copies_as_the_source_reads(self, tmp_path):
        """A sample the source rewound and failed to regrade stays rewound in
        the copy: its old `final` must not outrank the later records. With
        `record_meta` off, no per-sample grader stamp masks a wrong pick."""
        src = await _first(tmp_path, record_meta=False)
        strict, _ = _task(cls=StrictQATask)
        original = strict.feedback

        async def _feedback(post, ctx):
            if ctx.sample_id != 0:
                raise RuntimeError("judge down")
            return await original(post, ctx)

        setattr(strict, "feedback", _feedback)  # noqa: B010
        await _run(tmp_path, "first", strict, auto_resume=True, record_meta=False)

        task, model = _task(cls=StrictQATask)
        await _run(tmp_path, "second", task, resume_from=str(src), record_meta=False)
        assert model.call_count == 0
        assert task.calls["feedback"] == 2


# ===================================================================
# A directory recorded before stage identities
# ===================================================================
def _strip_identities(root: Path) -> None:
    """Rewrite *root* as a run before #150 left it: no identity in meta.json,
    no per-sample identity in the preprocess records, no run column."""
    (root / "meta.json").write_bytes(
        orjson.dumps({"version": _meta(root)["version"], "deterministic": False})
    )
    for path in root.rglob("*.jsonl"):
        lines = []
        for line in path.read_bytes().splitlines():
            record = orjson.loads(line)
            for history in (record.get("stage_meta") or {}).values():
                for entry in history:
                    entry.pop("identity", None)
            if isinstance(record.get("meta_last"), dict):
                record["meta_last"].pop("identity", None)
            lines.append(orjson.dumps(record))
        path.write_bytes(b"\n".join(lines) + b"\n")
        idx = path.with_suffix(".idx")
        offset, out = 0, []
        old = idx.read_text().splitlines()
        for raw, entry in zip(lines, old, strict=True):
            parts = entry.split("\t")[:8]
            parts[3], parts[4] = str(offset), str(len(raw) + 1)
            out.append("\t".join(parts))
            offset += len(raw) + 1
        idx.write_text("\n".join(out) + "\n")


class TestPreIdentityDirectory:
    @pytest.mark.anyio
    async def test_a_same_directory_resume_trusts_the_recorded_run(self, tmp_path):
        """The contract before #150: the invocation vouched for the config."""
        root = await _first(tmp_path)
        _strip_identities(root)
        (root / "report.json").unlink()
        task, model = _task()
        await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 0
        assert task.calls["feedback"] == 0

    @pytest.mark.anyio
    async def test_a_same_directory_resume_with_another_inference_is_refused(
        self, tmp_path
    ):
        """With a persisted config to compare, the recorded run is no longer
        trusted: a changed `n` would otherwise reuse `n=1` inference."""
        root = await _first(tmp_path)
        _strip_identities(root)
        (root / "report.json").unlink()
        task, model = _task(n=3)
        runner = TaskRunner(
            task,
            make_config(tmp_path, result_dir=str(root), auto_resume=True),
            legacy_match={"data": True, "inference": False},
        )
        with pytest.raises(StageResumeError, match="stage `inference`"):
            await runner.arun()
        assert model.call_count == 0

    @pytest.mark.anyio
    async def test_resume_from_needs_the_persisted_config(self, tmp_path):
        """Without identities or a persisted config the data stage is unknown,
        so the resume is refused; the source is never written to."""
        src = await _first(tmp_path)
        _strip_identities(src)
        before = (src / "meta.json").read_bytes()
        task, model = _task(cls=StrictQATask)
        with pytest.raises(StageResumeError, match="stage `inference`"):
            await _run(tmp_path, "second", task, resume_from=str(src))
        assert model.call_count == 0
        assert (src / "meta.json").read_bytes() == before

    @pytest.mark.anyio
    async def test_resume_from_with_a_matching_config_regrades(self, tmp_path):
        """The caller compared the persisted config: data and inference match.
        The prompt is digested from the recorded preprocess outputs."""
        src = await _first(tmp_path)
        _strip_identities(src)
        before = (src / "meta.json").read_bytes()
        task, model = _task(cls=StrictQATask)
        runner = TaskRunner(
            task,
            make_config(
                tmp_path,
                result_dir=str(tmp_path / "second"),
                resume_from=str(src),
            ),
            legacy_match={"data": True, "inference": True},
        )
        report = await runner.arun()
        assert model.call_count == 0
        assert task.calls["preprocess"] == 3  # the prompt digest, not a rerun
        assert task.calls["feedback"] == 3
        assert report["strict"] is True
        meta = _meta(runner.root_dir)
        [run] = meta["runs"]
        assert run["start"] == "from"
        assert run["reran"] == ["grader"]
        # The source recorded no identity: `from.identity` says so.
        assert run["from"]["identity"]["inference"] is None
        assert meta["identity"]["data"]["digest"] is not None
        # Never written back into the old run.
        assert (src / "meta.json").read_bytes() == before

    @pytest.mark.anyio
    async def test_a_changed_prompt_reruns_from_the_start(self, tmp_path):
        src = await _first(tmp_path)
        _strip_identities(src)
        task, model = _task(cls=StrictQATask, prefix="Q: ")
        runner = TaskRunner(
            task,
            make_config(
                tmp_path,
                result_dir=str(tmp_path / "second"),
                resume_from=str(src),
            ),
            legacy_match={"data": True, "inference": True},
        )
        await runner.arun()
        assert model.call_count == 3

    @pytest.mark.anyio
    async def test_data_unknown_is_refused(self, tmp_path):
        src = await _first(tmp_path)
        _strip_identities(src)
        task, _ = _task(cls=StrictQATask)
        runner = TaskRunner(
            task,
            make_config(
                tmp_path,
                result_dir=str(tmp_path / "second"),
                resume_from=str(src),
            ),
            legacy_match={"inference": True},
        )
        with pytest.raises(StageResumeError, match="stage `data`"):
            await runner.arun()


class TestInterruptedBeforeEveryPreprocess:
    """A fresh run stopped before its data identity was written; rerunning the
    same command resumes it, however often."""

    @staticmethod
    async def _interrupted(tmp_path, *, record_meta, fail_one):
        scope = anyio.CancelScope()

        class Slow(QATask):
            async def preprocess(self, raw, ctx):
                if fail_one and ctx.sample_id == 1:
                    raise ValueError("bad sample")
                if ctx.sample_id == 2:
                    await anyio.sleep_forever()
                return await super().preprocess(raw, ctx)

            async def postprocess(self, inf, ctx):
                if ctx.sample_id == 0:
                    scope.cancel()
                    await anyio.sleep_forever()
                return await super().postprocess(inf, ctx)

        task, _ = _task(cls=Slow)
        with scope:
            await _run(tmp_path, "qa", task, record_meta=record_meta)
        root = tmp_path / "qa"
        assert _meta(root)["identity"]["data"] is None
        return root

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("record_meta", "fail_one"), [(False, False), (True, True)]
    )
    async def test_rerunning_resumes(self, tmp_path, record_meta, fail_one):
        root = await self._interrupted(
            tmp_path, record_meta=record_meta, fail_one=fail_one
        )
        for _ in range(2):
            task, _ = _task()
            report = await _run(
                tmp_path, "qa", task, auto_resume=True, record_meta=record_meta
            )
            assert report[1]["total"] == 3
        runs = _meta(root)["runs"]
        assert [r["start"] for r in runs] == ["fresh", "resume", "resume"]
        assert all("ended" in r for r in runs)

    @pytest.mark.anyio
    @pytest.mark.parametrize(("record_meta", "calls"), [(True, 1), (False, 3)])
    async def test_finished_samples_rerun_only_without_digests(
        self, tmp_path, record_meta, calls
    ):
        """Samples 0 and 1 finished. With `record_meta` off their data cannot be
        checked, so they are inferred again, and the log says why."""
        scope = anyio.CancelScope()
        interrupting = True

        class Slow(QATask):
            async def preprocess(self, raw, ctx):
                if interrupting and ctx.sample_id == 2:
                    await anyio.sleep_forever()
                return await super().preprocess(raw, ctx)

            async def feedback(self, post, ctx):
                result = await super().feedback(post, ctx)
                if interrupting and ctx.sample_id == 1:
                    scope.cancel()
                return result

        task, _ = _task(cls=Slow)
        with scope:
            await _run(tmp_path, "qa", task, record_meta=record_meta)
        interrupting = False

        records: list = []
        sink = logger.add(records.append, format="{message}", level="WARNING")
        try:
            task, model = _task(cls=Slow)
            await _run(tmp_path, "qa", task, auto_resume=True, record_meta=record_meta)
        finally:
            logger.remove(sink)
        assert model.call_count == calls
        warned = any("cannot be checked" in str(r) for r in records)
        assert warned is not record_meta

    @pytest.mark.anyio
    async def test_a_refused_resume_appends_no_run(self, tmp_path):
        """Every refusal comes before `begin_run`, not just the inference one."""
        root = await _first(tmp_path)
        _strip_identities(root)
        (root / "report.json").unlink()
        before = (root / "meta.json").read_bytes()
        task, _ = _task()
        runner = TaskRunner(
            task,
            make_config(tmp_path, result_dir=str(root), auto_resume=True),
            legacy_match={"inference": True},
        )
        with pytest.raises(StageResumeError, match="stage `data`"):
            await runner.arun()
        assert (root / "meta.json").read_bytes() == before


def test_recorded_identity_reads_only_this_schema():
    assert recorded_identity(None) is None
    assert recorded_identity({"identity": {"schema": 99}}) is None
    identity = recorded_identity({"identity": {"schema": 1, "data": {"digest": "x"}}})
    assert identity is not None
    assert identity["data"] == {"digest": "x"}
    assert identity["grader"] is None
