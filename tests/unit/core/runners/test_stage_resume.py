"""
Per-stage resume: each row of the resume table, resume across tasks, and a
resume from a directory recorded before stage identities.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import anyio
import orjson
import pytest
from loguru import logger

from sieval.core.runners import runner as runner_module
from sieval.core.runners import stage_resume
from sieval.core.runners.resume_gate import ResumeVersionError
from sieval.core.runners.runner import TaskRunner, UnfinishedCopyError, run_started
from sieval.core.runners.stage_resume import (
    COPY_MARKER,
    StageResumeError,
    copy_recorded_dir,
    plan_resume,
    recorded_grader,
    recorded_identity,
)
from sieval.core.tasks import (
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.run_meta import GRADING_STAGES, grader_key
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
        runner, _ = await _run(tmp_path, "first", task, resume=True)
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
        runner, _ = await _run(tmp_path, "first", task, resume=True)
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
        await _run(tmp_path, "first", task, resume=True)
        assert task.calls["postprocess"] == 0
        assert task.calls["report"] == 0

    @pytest.mark.anyio
    async def test_a_grader_difference_regrades_on_recorded_inference(self, tmp_path):
        """Supported regrading: no shard is deleted by hand."""
        root = await _first(tmp_path)
        task, model = _task(threshold=0.9)
        _, report = await _run(tmp_path, "first", task, resume=True)
        assert model.call_count == 0
        assert task.calls["postprocess"] == 3
        assert task.calls["feedback"] == 3
        assert report["threshold"] == 0.9
        meta = _meta(root)
        assert meta["runs"][-1]["reran"] == ["postprocess", "feedback", "report"]
        assert meta["runs"][-1]["resume"]["repostprocessed"] == 3
        for stage in ("postprocess", "feedback", "report"):
            assert meta["identity"][stage]["args"]["threshold"] == 0.9

        # The regrade is now the record: a third run fast-resumes.
        again, model = _task(threshold=0.9)
        await _run(tmp_path, "first", again, resume=True)
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
        await _run(tmp_path, "first", task, resume=True)

        resumed, model = _task(threshold=0.9)
        await _run(tmp_path, "first", resumed, resume=True)
        assert model.call_count == 0
        assert resumed.calls["postprocess"] == 0
        assert resumed.calls["feedback"] == 2
        run = _meta(root)["runs"][-1]
        assert run["reran"] == ["feedback", "report"]
        # Graded by this grader already: kept, and the two failures retried.
        assert run["resume"]["kept"] == 3
        assert run["resume"]["regraded"] == 0

    @pytest.mark.anyio
    async def test_an_inference_difference_is_refused_naming_the_stage(self, tmp_path):
        root = await _first(tmp_path)
        before = (root / "meta.json").read_bytes()
        task, model = _task(n=2)
        with pytest.raises(StageResumeError, match="stage `inference`") as raised:
            await _run(tmp_path, "first", task, resume=True)
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
        await _run(tmp_path, "first", task, resume=True)
        assert model.call_count == 3
        assert _meta(root)["runs"][-1]["reran"] == [
            "data",
            "preprocess",
            "inference",
            "postprocess",
            "feedback",
            "report",
        ]

    @pytest.mark.anyio
    async def test_a_data_difference_reruns_only_the_changed_samples(self, tmp_path):
        root = await _first(tmp_path)
        changed = [*SAMPLES[:2], {"question": "What is 1+1?", "answer": "3"}]
        task, model = _task(samples=changed)
        _, report = await _run(tmp_path, "first", task, resume=True)
        assert model.call_count == 1
        assert report["total"] == 3
        assert _meta(root)["identity"]["data"]["samples"] == 3

    @pytest.mark.anyio
    async def test_a_sample_subset_drops_the_samples_left_out(self, tmp_path):
        root = await _first(tmp_path)
        task, model = _task(samples=SAMPLES[:2])
        _, report = await _run(tmp_path, "first", task, resume=True)
        assert model.call_count == 0
        assert report["total"] == 2
        assert _meta(root)["identity"]["data"]["samples"] == 2

        # Widening again reuses the dropped sample's records: its data,
        # prompt and inference still match.
        widened, model = _task()
        _, report = await _run(tmp_path, "first", widened, resume=True)
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
            resume=str(src),
            profile_stages=True,
        )
        assert model.call_count == 0
        assert task.calls["feedback"] == 3
        assert report["strict"] is True

        meta = _meta(runner.root_dir)
        [run] = meta["runs"]
        assert run["start"] == "from"
        # Another task class: every grading stage reruns.
        assert run["reran"] == ["postprocess", "feedback", "report"]
        assert run["from"] == {"dir": str(src), "identity": source_meta["identity"]}
        assert (
            meta["identity"]["inference"]["digest"]
            == (source_meta["identity"]["inference"]["digest"])
        )
        assert (
            meta["identity"]["report"]["digest"]
            != (source_meta["identity"]["report"]["digest"])
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
            runner, _ = await _run(tmp_path, "second", task, resume=str(src))
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
            await _run(tmp_path, "second", task, resume=str(src))
        assert model.call_count == 0
        dst = tmp_path / "second"
        assert not dst.exists() or not any(dst.iterdir())

    @pytest.mark.anyio
    async def test_a_failed_setup_leaves_nothing_and_retries(self, tmp_path):
        src = await _first(tmp_path)
        task, _ = _task(cls=StrictQATask)

        async def _setup():
            raise RuntimeError("sandbox down")

        setattr(task, "setup", _setup)  # noqa: B010
        with pytest.raises(RuntimeError, match="sandbox down"):
            await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        assert not list(dst.rglob("*.idx"))
        again, model = _task(cls=StrictQATask)
        runner, _ = await _run(tmp_path, "second", again, resume=str(src))
        assert model.call_count == 0
        assert _meta(runner.root_dir)["runs"][0]["start"] == "from"

    @pytest.mark.anyio
    async def test_a_runner_never_run_copies_nothing(self, tmp_path):
        """A session builds every runner before running any, so a later task
        that fails to build leaves earlier runners unrun."""
        src = await _first(tmp_path)
        dst = tmp_path / "second"
        task, _ = _task(cls=StrictQATask)
        TaskRunner(task, make_config(tmp_path, result_dir=str(dst), resume=str(src)))
        assert not list(dst.rglob("*.idx"))
        again, model = _task(cls=StrictQATask)
        await _run(tmp_path, "second", again, resume=str(src))
        assert model.call_count == 0

    @pytest.mark.anyio
    async def test_a_source_without_meta_is_refused_without_removing_it(self, tmp_path):
        src = await _first(tmp_path)
        (src / "meta.json").unlink()
        task, _ = _task(cls=StrictQATask)
        with pytest.raises(ResumeVersionError) as exc:
            await _run(tmp_path, "second", task, resume=str(src))
        message = str(exc.value)
        assert str(src) in message
        assert "Remove" not in message
        assert list(src.rglob("*.idx"))

    @pytest.mark.anyio
    async def test_a_from_run_is_resumed_in_place(self, tmp_path):
        """Interrupted after the copy: the same config resumes the copy, not
        the source again."""
        src = await _first(tmp_path)
        first, _ = _task(cls=StrictQATask)
        await _run(tmp_path, "second", first, resume=str(src))
        again, model = _task(cls=StrictQATask)
        runner, _ = await _run(tmp_path, "second", again, resume=str(src))
        assert model.call_count == 0
        assert [r["start"] for r in _meta(runner.root_dir)["runs"]] == [
            "from",
            "resume",
        ]

    @pytest.mark.anyio
    async def test_its_own_directory_resumes_in_place(self, tmp_path):
        """``resume=<result_dir>`` is ``resume=True``: no copy, no refusal."""
        src = await _first(tmp_path)
        task, model = _task(cls=StrictQATask)
        runner, report = await _run(tmp_path, "first", task, resume=str(src))
        assert model.call_count == 0
        assert report["strict"] is True
        [_, run] = _meta(runner.root_dir)["runs"]
        assert run["start"] == "resume"
        assert "from" not in run
        assert not (src / COPY_MARKER).exists()

    @pytest.mark.anyio
    async def test_true_and_its_own_directory_record_alike(self, tmp_path):
        await _first(tmp_path, name="a")
        await _first(tmp_path, name="b")
        task, _ = _task()
        await _run(tmp_path, "a", task, resume=True)
        task, _ = _task()
        await _run(tmp_path, "b", task, resume=str(tmp_path / "b"))
        assert [r["start"] for r in _meta(tmp_path / "a")["runs"]] == [
            r["start"] for r in _meta(tmp_path / "b")["runs"]
        ]

    @pytest.mark.anyio
    async def test_the_deprecated_resume_from_still_copies(self, tmp_path):
        src = await _first(tmp_path)
        task, model = _task(cls=StrictQATask)
        runner, _ = await _run(tmp_path, "second", task, resume_from=str(src))
        assert model.call_count == 0
        assert _meta(runner.root_dir)["runs"][0]["start"] == "from"

    @pytest.mark.anyio
    async def test_the_deprecated_auto_resume_still_resumes(self, tmp_path):
        src = await _first(tmp_path)
        task, model = _task()
        runner, _ = await _run(tmp_path, "first", task, auto_resume=True)
        assert model.call_count == 0
        assert runner.root_dir == src

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
        # The marker stays until the run that copied records itself.
        assert sorted(p.name for p in dst.iterdir()) == [COPY_MARKER, "0"]
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
        await _run(tmp_path, "first", strict, resume=True, record_meta=False)

        task, model = _task(cls=StrictQATask)
        await _run(tmp_path, "second", task, resume=str(src), record_meta=False)
        assert model.call_count == 0
        assert task.calls["feedback"] == 2


# ===================================================================
# A directory recorded before stage identities
# ===================================================================
def _strip_identities(root: Path) -> None:
    """Rewrite *root* as a run before #150 left it: no identity in meta.json,
    no per-sample identity in the preprocess records, no run column."""
    (root / "meta.json").write_bytes(
        orjson.dumps(
            {
                "version": _meta(root)["version"],
                "result_schema_version": _meta(root)["result_schema_version"],
                "deterministic": False,
            }
        )
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
        await _run(tmp_path, "first", task, resume=True)
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
            make_config(tmp_path, result_dir=str(root), resume=True),
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
            await _run(tmp_path, "second", task, resume=str(src))
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
                resume=str(src),
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
        assert run["reran"] == ["postprocess", "feedback", "report"]
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
                resume=str(src),
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
                resume=str(src),
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
                tmp_path, "qa", task, resume=True, record_meta=record_meta
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
            await _run(tmp_path, "qa", task, resume=True, record_meta=record_meta)
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
            make_config(tmp_path, result_dir=str(root), resume=True),
            legacy_match={"inference": True},
        )
        with pytest.raises(StageResumeError, match="stage `data`"):
            await runner.arun()
        assert (root / "meta.json").read_bytes() == before


def test_recorded_identity_reads_schema_1_and_2():
    assert recorded_identity(None) is None
    assert recorded_identity({"identity": {"schema": 99}}) is None
    identity = recorded_identity({"identity": {"schema": 2, "data": {"digest": "x"}}})
    assert identity is not None
    assert identity["data"] == {"digest": "x"}
    assert identity["report"] is None
    old = {
        "schema": 1,
        "prompt": {"digest": "p"},
        "grader": {"digest": "g", "modules": {}, "judges": []},
    }
    identity = recorded_identity({"identity": old})
    assert identity is not None
    assert identity["schema"] == 2
    assert identity["preprocess"] == {"digest": "p"}
    assert identity["postprocess"] is None
    assert recorded_grader({"identity": old}) == old["grader"]
    assert recorded_grader({"identity": {"schema": 2}}) is None


# ===================================================================
# A resume=<dir> copy interrupted before its run began
# ===================================================================
class _Killed(BaseException):
    """Stands in for SIGKILL: nothing after it runs, the rollback included."""


class TestInterruptedCopy:
    """The same ``resume=<dir>`` command resumes a copy wherever a kill
    stopped it. A kill runs no rollback, so each test disables it."""

    @staticmethod
    def _no_rollback(monkeypatch):
        monkeypatch.setattr(TaskRunner, "_abandon_copy", lambda self: None)
        monkeypatch.setattr(stage_resume, "abandon_copy", lambda root, created: None)

    @staticmethod
    async def _rerun(tmp_path, src):
        task, model = _task(cls=StrictQATask)
        runner, report = await _run(tmp_path, "second", task, resume=str(src))
        assert model.call_count == 0
        assert task.calls["feedback"] == 3
        assert report["strict"] is True
        assert not (runner.root_dir / COPY_MARKER).exists()
        return _meta(runner.root_dir)

    @pytest.mark.anyio
    async def test_killed_during_the_copy(self, tmp_path, monkeypatch):
        src = await _first(tmp_path)
        (src / "1" / "final").mkdir(parents=True)
        (src / "1" / "final" / "0.idx").write_text("")
        (src / "1" / "final" / "0.jsonl").write_text("")
        copied = 0
        original = stage_resume._copy_idx

        def _copy_one(*args):
            nonlocal copied
            if copied:
                raise _Killed
            copied += 1
            original(*args)

        with monkeypatch.context() as m:
            self._no_rollback(m)
            m.setattr(stage_resume, "_copy_idx", _copy_one)
            task, _ = _task(cls=StrictQATask)
            with pytest.raises(_Killed):
                await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        assert (dst / COPY_MARKER).read_text() == str(src)
        assert list(dst.rglob("*.idx"))
        assert not (dst / "meta.json").exists()
        assert run_started(dst)

        meta = await self._rerun(tmp_path, src)
        assert [r["start"] for r in meta["runs"]] == ["from"]
        assert meta["runs"][0]["reran"] == ["postprocess", "feedback", "report"]

    @staticmethod
    async def _kill_before_any_index(tmp_path, monkeypatch):
        """Kill at the first ``.idx``: the marker, but no ``.idx`` yet."""
        src = await _first(tmp_path)

        def _kill(*_args):
            raise _Killed

        with monkeypatch.context() as m:
            TestInterruptedCopy._no_rollback(m)
            m.setattr(stage_resume, "_copy_idx", _kill)
            task, _ = _task(cls=StrictQATask)
            with pytest.raises(_Killed):
                await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        assert (dst / COPY_MARKER).exists()
        assert not list(dst.rglob("*.idx"))
        assert not run_started(dst)
        return src, dst

    @pytest.mark.anyio
    async def test_killed_before_any_index(self, tmp_path, monkeypatch):
        src, _ = await self._kill_before_any_index(tmp_path, monkeypatch)
        meta = await self._rerun(tmp_path, src)
        assert [r["start"] for r in meta["runs"]] == ["from"]

    @pytest.mark.anyio
    @pytest.mark.parametrize("auto_resume", [True, False])
    async def test_killed_before_any_index_without_resume_from(
        self, tmp_path, monkeypatch, auto_resume
    ):
        """No ``.idx`` yet, so no run started; the marker alone must refuse."""
        src, dst = await self._kill_before_any_index(tmp_path, monkeypatch)
        task, _ = _task(cls=StrictQATask)
        with pytest.raises(UnfinishedCopyError) as exc:
            TaskRunner(
                task,
                make_config(tmp_path, result_dir=str(dst), resume=auto_resume),
            )
        assert exc.value.source == str(src)

    @pytest.mark.anyio
    async def test_killed_after_the_copy_before_meta(self, tmp_path, monkeypatch):
        """Resume planning (load, hydrate, plan) runs between the copy and
        ``meta.json``."""
        src = await _first(tmp_path)

        async def _plan(_self):
            raise _Killed

        with monkeypatch.context() as m:
            self._no_rollback(m)
            m.setattr(TaskRunner, "_plan_stage_resume", _plan)
            task, _ = _task(cls=StrictQATask)
            with pytest.raises(_Killed):
                await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        assert (dst / COPY_MARKER).exists()
        assert sorted(p.name for p in dst.iterdir()) == [COPY_MARKER, "0"]

        meta = await self._rerun(tmp_path, src)
        assert [r["start"] for r in meta["runs"]] == ["from"]

    @pytest.mark.anyio
    async def test_killed_after_meta(self, tmp_path, monkeypatch):
        """``meta.json`` records the run, so the marker beside it is stale:
        the copy is this directory's run and is resumed, not copied again."""
        src = await _first(tmp_path)

        def _end_copy(_root):
            raise _Killed

        with monkeypatch.context() as m:
            m.setattr(runner_module, "end_copy", _end_copy)
            task, _ = _task(cls=StrictQATask)
            with pytest.raises(_Killed):
                await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        assert (dst / COPY_MARKER).exists()
        assert [r["start"] for r in _meta(dst)["runs"]] == ["from"]

        meta = await self._rerun(tmp_path, src)
        assert [r["start"] for r in meta["runs"]] == ["from", "resume"]

    @pytest.mark.anyio
    async def test_the_reviewer_reproduction(self, tmp_path):
        """A source's shards copied by hand into an empty directory, as a kill
        during the copy leaves them, with the marker the copy writes first."""
        src = await _first(tmp_path)
        dst = tmp_path / "second"
        dst.mkdir()
        (dst / COPY_MARKER).write_text(str(src))
        shutil.copytree(src / "0", dst / "0")
        meta = await self._rerun(tmp_path, src)
        assert [r["start"] for r in meta["runs"]] == ["from"]

    @pytest.mark.anyio
    @pytest.mark.parametrize("auto_resume", [True, False])
    async def test_without_resume_from_names_the_copy(
        self, tmp_path, monkeypatch, auto_resume
    ):
        """A plain ``--resume`` has no source to copy again, and the copy is
        no run to resume; nothing is touched."""
        src = await _first(tmp_path)

        async def _plan(_self):
            raise _Killed

        with monkeypatch.context() as m:
            self._no_rollback(m)
            m.setattr(TaskRunner, "_plan_stage_resume", _plan)
            task, _ = _task(cls=StrictQATask)
            with pytest.raises(_Killed):
                await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        before = {p: p.read_bytes() for p in dst.rglob("*") if p.is_file()}
        task, _ = _task(cls=StrictQATask)
        with pytest.raises(UnfinishedCopyError) as exc:
            TaskRunner(
                task,
                make_config(tmp_path, result_dir=str(dst), resume=auto_resume),
            )
        assert exc.value.path == dst
        assert exc.value.source == str(src)
        assert "same `resume` directory" in str(exc.value)
        assert {p: p.read_bytes() for p in dst.rglob("*") if p.is_file()} == before

    @pytest.mark.anyio
    async def test_a_refused_copy_removes_its_marker(self, tmp_path):
        src = await _first(tmp_path)
        task, _ = _task(cls=StrictQATask, n=2)
        with pytest.raises(StageResumeError):
            await _run(tmp_path, "second", task, resume=str(src))
        dst = tmp_path / "second"
        assert not (dst / COPY_MARKER).exists()
        assert not run_started(dst)

    def test_a_failed_copy_removes_its_marker(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        (src / "0" / "final").mkdir(parents=True)
        (src / "0" / "final" / "0.idx").write_text("0\t0\tfinal\t0\t3\t\t\t0\t1\n")
        (src / "0" / "final" / "0.jsonl").write_text("{}\n")

        def _fail(*_args):
            raise OSError("disk full")

        monkeypatch.setattr(stage_resume, "_copy_idx", _fail)
        dst = tmp_path / "dst"
        with pytest.raises(OSError, match="disk full"):
            copy_recorded_dir(src, dst)
        assert list(dst.iterdir()) == []

    def test_a_shard_dir_left_by_the_rollback_keeps_the_marker(
        self, tmp_path, monkeypatch
    ):
        """A removal that missed a directory leaves it replaceable, not
        unmarked shards that would lock the directory."""
        dst = tmp_path / "dst"
        (dst / "0" / "final").mkdir(parents=True)
        (dst / "0" / "final" / "0.idx").write_text("")
        (dst / COPY_MARKER).write_text("src")
        monkeypatch.setattr(stage_resume, "remove_copied", lambda created: None)
        stage_resume.abandon_copy(dst, [dst / "0"])
        assert (dst / COPY_MARKER).exists()


# ===================================================================
# Per-stage rollback: the earliest grading stage that changed
# ===================================================================
_PLUGIN = "sieval_rollback_plugin"

_PLUGIN_TASK = """
from sieval.core.tasks import (
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.task import Task

from .grade import correct
from .metrics import summarize


class PluginQA(Task):
    model_type = "chat"

    def __init__(self, dataset, model, name=None, judge=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._judge = judge
        self.calls = {"postprocess": 0, "feedback": 0, "report": 0}

    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [{"role": "user", "content": raw["question"]}], reference=raw["answer"]
        )

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"][0]["content"])

    async def postprocess(self, inf, ctx):
        self.calls["postprocess"] += 1
        return build_prediction_record([t.strip() for t in inf.texts])

    async def feedback(self, post, ctx):
        self.calls["feedback"] += 1
        if self._judge is not None:
            await self._judge.agenerate("Is it right?")
        gold = ctx.raw_sample["answer"]
        rollouts = [
            build_rollout_judgement(r["index"], correct(r.get("prediction"), gold))
            for r in post["rollouts"]
        ]
        return True, build_judgement_record(gold, rollouts)

    async def report(self, finals, fails):
        self.calls["report"] += 1
        return summarize(finals, fails)
"""

_PLUGIN_FILES = {
    "__init__.py": "",
    "task.py": _PLUGIN_TASK,
    "grade.py": "def correct(prediction, gold):\n    return prediction == gold\n",
    "metrics.py": (
        "def summarize(finals, fails):\n"
        "    return {'total': len(finals) + len(fails)}\n"
    ),
}


class _CountingJudge(MockCountingChatModel):
    """A judge that counts its calls."""

    def __init__(self):
        super().__init__(answers={})
        self._model = "count-judge"


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """A plugin task on disk; ``load()`` re-imports it after an edit."""
    # An edit can keep a file's size within the second, which a cached
    # bytecode file would not notice.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    pkg = tmp_path / "plugins" / _PLUGIN
    pkg.mkdir(parents=True)
    for name, text in _PLUGIN_FILES.items():
        (pkg / name).write_text(textwrap.dedent(text))
    monkeypatch.syspath_prepend(str(pkg.parent))

    def drop():
        for name in list(sys.modules):
            if name == _PLUGIN or name.startswith(_PLUGIN + "."):
                sys.modules.pop(name, None)

    def load():
        drop()
        return importlib.import_module(f"{_PLUGIN}.task").PluginQA

    yield pkg, load
    drop()


def _edit(pkg, name, old, new):
    path = pkg / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


async def _plugin_run(tmp_path, cls, **cfg):
    model = MockCountingChatModel(answers=ANSWERS)
    judge = _CountingJudge()
    task = cls(dataset=MockDataset(SAMPLES), model=model, name="qa", judge=judge)
    runner = TaskRunner(
        task, make_config(tmp_path, result_dir=str(tmp_path / "plugin"), **cfg)
    )
    report = await runner.arun()
    return runner.root_dir, task, model, judge, report


class TestPerStageRollback:
    @pytest.mark.anyio
    async def test_a_report_change_rereports_without_a_model_or_judge(
        self, tmp_path, plugin
    ):
        pkg, load = plugin
        root, *_ = await _plugin_run(tmp_path, load())
        before = _meta(root)["identity"]
        _edit(pkg, "metrics.py", "'total'", "'count'")

        _, task, model, judge, report = await _plugin_run(tmp_path, load(), resume=True)
        assert report["count"] == 3 and "total" not in report
        assert model.call_count == 0
        assert judge.call_count == 0
        assert task.calls == {"postprocess": 0, "feedback": 0, "report": 1}
        meta = _meta(root)
        run = meta["runs"][-1]
        assert run["reran"] == ["report"]
        assert run["resume"] == {
            "kept": 3,
            "rereported": 3,
            "regraded": 0,
            "repostprocessed": 0,
            "restarted": 0,
            "dropped": 0,
        }
        assert meta["identity"]["postprocess"] == before["postprocess"]
        assert meta["identity"]["feedback"]["digest"] == before["feedback"]["digest"]
        assert meta["identity"]["report"] != before["report"]

        # Re-reported is now the record: the next resume reuses report.json.
        _, again, model, judge, _ = await _plugin_run(tmp_path, load(), resume=True)
        assert again.calls["report"] == 0
        assert _meta(root)["runs"][-1]["reran"] == []

    @pytest.mark.anyio
    async def test_an_unfinished_run_with_a_report_change_keeps_its_verdicts(
        self, tmp_path, plugin
    ):
        """The fast path needs a finished report; without one, the per-sample
        plan keeps every graded sample and only report() runs."""
        pkg, load = plugin
        root, *_ = await _plugin_run(tmp_path, load())
        (root / "report.json").unlink()
        _edit(pkg, "metrics.py", "'total'", "'count'")
        _, task, model, judge, report = await _plugin_run(tmp_path, load(), resume=True)
        assert report["count"] == 3 and "total" not in report
        assert (model.call_count, judge.call_count) == (0, 0)
        assert task.calls == {"postprocess": 0, "feedback": 0, "report": 1}
        assert _meta(root)["runs"][-1]["resume"]["rereported"] == 3

    @pytest.mark.anyio
    async def test_a_feedback_change_refeeds_on_the_recorded_postprocess(
        self, tmp_path, plugin
    ):
        pkg, load = plugin
        root, *_ = await _plugin_run(tmp_path, load())
        _edit(pkg, "grade.py", "prediction == gold", "str(prediction) == str(gold)")

        _, task, model, judge, _ = await _plugin_run(tmp_path, load(), resume=True)
        assert model.call_count == 0
        assert judge.call_count == 3
        assert task.calls == {"postprocess": 0, "feedback": 3, "report": 1}
        run = _meta(root)["runs"][-1]
        assert run["reran"] == ["feedback", "report"]
        assert run["resume"]["regraded"] == 3

    @pytest.mark.anyio
    async def test_a_postprocess_change_reruns_postprocess_onward(
        self, tmp_path, plugin
    ):
        pkg, load = plugin
        root, *_ = await _plugin_run(tmp_path, load())
        _edit(pkg, "task.py", "t.strip() for t", "t.strip().lower() for t")

        _, task, model, judge, _ = await _plugin_run(tmp_path, load(), resume=True)
        assert model.call_count == 0
        assert judge.call_count == 3
        assert task.calls == {"postprocess": 3, "feedback": 3, "report": 1}
        run = _meta(root)["runs"][-1]
        assert run["reran"] == ["postprocess", "feedback", "report"]
        assert run["resume"]["repostprocessed"] == 3

    @pytest.mark.anyio
    async def test_a_report_change_reaches_no_judge_model_even_unfinished(
        self, tmp_path, plugin
    ):
        """A judge whose model changed counts in feedback; report alone stays."""
        _, load = plugin
        root, *_ = await _plugin_run(tmp_path, load())
        identity = _meta(root)["identity"]
        assert [j["role"] for j in identity["feedback"]["judges"]] == ["judge"]
        assert "judges" not in identity["report"]
        assert "judges" not in identity["postprocess"]


class TestSchema1Directory:
    """The fixtures under ``tests/integration/resume`` resume real schema-1
    directories; this covers the plan a dry run reports for one."""

    def test_plan_resume_reads_a_schema_1_block(self):
        meta = {
            "identity": {
                "schema": 1,
                "inference": {"digest": "sha256:i"},
                "grader": {"digest": "g", "modules": {}, "judges": []},
            }
        }
        key = grader_key(meta["identity"]["grader"])
        assert key is not None
        current = {"inference": {"digest": "sha256:i"}}
        kept = plan_resume(meta, current, {key})
        assert kept["legacy_grader"] == "kept"
        assert kept["rerun_from"] is None
        assert kept["refused"] is None
        assert all(kept["stages"][s] == "same" for s in GRADING_STAGES)
        regrade = plan_resume(meta, current, set())
        assert regrade["legacy_grader"] == "regrade"
        assert regrade["rerun_from"] == "postprocess"
        assert regrade["stages"]["postprocess"] == "differs"


def test_plan_resume_rolls_back_to_the_earliest_changed_stage():
    stage = {"digest": "x", "task": "t", "modules": {"m": "1"}, "args": {}}
    changed = {**stage, "modules": {"m": "2"}}
    recorded = {
        "identity": {
            "schema": 2,
            "inference": {"digest": "sha256:i"},
            "postprocess": stage,
            "feedback": stage,
            "report": stage,
        }
    }
    same = {"inference": {"digest": "sha256:i"}, **dict.fromkeys(GRADING_STAGES, stage)}
    assert plan_resume(recorded, same)["rerun_from"] is None
    assert plan_resume(recorded, {**same, "report": changed})["rerun_from"] == "report"
    plan = plan_resume(recorded, {**same, "feedback": changed, "report": changed})
    assert plan["rerun_from"] == "feedback"
    assert plan["stages"]["postprocess"] == "same"
    assert plan["stages"]["data"] == "unknown"
    other = plan_resume(recorded, {**same, "inference": {"digest": "sha256:j"}})
    assert other["refused"] == "inference"
    assert plan_resume(None, same)["identity_schema"] is None
