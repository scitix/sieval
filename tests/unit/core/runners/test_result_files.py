"""
The files a task directory holds, and who writes each: no ``manifest.json``, one
finalize for every exit path, anomalies and the profile over every sample.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.anomaly import get_rules_hash
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.saver import TaskSaver
from tests.conftest import (
    MockChatModel,
    MockCountingChatModel,
    MockDataset,
    make_config,
)
from tests.unit.core.runners.test_runner_run_meta import ANSWERS, SAMPLES, RecordTask


class GenTask(RecordTask):
    """A generation task, so the `gen` anomaly rules apply."""

    tags = frozenset({"gen"})


class TruncatingModel(MockChatModel):
    """Answers, but reports every reply cut short by the token limit."""

    async def _stub_arun(self, req):
        response = await super()._stub_arun(req)
        return replace(response, finish_reasons=("length",) * len(response.texts))


def _read(root: Path, name: str) -> dict:
    return orjson.loads((root / name).read_bytes())


def _config(tmp_path, **overrides):
    return make_config(
        tmp_path,
        detect_anomalies=True,
        dump_progress=True,
        profile_usage=False,
        profile_stages=False,
        **overrides,
    )


async def _run(tmp_path, model=None, task_cls=GenTask, **overrides):
    task = task_cls(
        dataset=MockDataset(SAMPLES),
        model=model or MockCountingChatModel(answers=ANSWERS),
        name="t",
    )
    runner = TaskRunner(task, _config(tmp_path, **overrides))
    return runner, await runner.arun()


def _truncate_idx(root: Path, keep: int) -> None:
    """Keep the first *keep* lines of every final ``.idx``, as a run killed
    between its shard appends and its ``.idx`` appends leaves them."""
    for idx in root.glob("*/final/*.idx"):
        lines = idx.read_text().splitlines(keepends=True)
        idx.write_text("".join(lines[:keep]))


# ===================================================================
# No manifest.json: the shard is the record, `.idx` its index
# ===================================================================
class TestNoManifest:
    @pytest.mark.anyio
    async def test_a_run_writes_no_manifest(self, tmp_path):
        runner, _ = await _run(tmp_path)
        assert not (runner.root_dir / "manifest.json").exists()

    @pytest.mark.anyio
    async def test_a_record_without_its_idx_line_reruns(self, tmp_path):
        """Killed between a shard append and its `.idx` append: the record is
        on disk, no line points at it, so the sample is not done and reruns."""
        runner, _ = await _run(tmp_path, record_each_stage=False)
        root = runner.root_dir
        (root / "report.json").unlink()
        _truncate_idx(root, keep=2)
        shard = next(root.glob("*/final/*.jsonl"))
        assert len(shard.read_bytes().splitlines()) == 3  # the record is there

        model = MockCountingChatModel(answers=ANSWERS)
        _, report = await _run(
            tmp_path, model=model, resume=True, record_each_stage=False
        )
        assert model.call_count == 1
        assert report["total"] == 3

    @pytest.mark.anyio
    async def test_an_idx_line_past_the_shard_end_is_dropped(self, tmp_path):
        runner, _ = await _run(tmp_path, record_each_stage=False)
        root = runner.root_dir
        shard = next(root.glob("*/final/*.jsonl"))
        lines = shard.read_bytes().splitlines(keepends=True)
        shard.write_bytes(b"".join(lines[:2]))  # the third record is gone

        task = GenTask(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t")
        contexts = await TaskLoader(task=task, root_dir=root).load_initial_state()
        assert sorted(contexts) == [0, 1]

    @pytest.mark.anyio
    async def test_fast_resume_is_read_from_the_report_counts(self, tmp_path):
        runner, _ = await _run(tmp_path)
        report = _read(runner.root_dir, "report.json")
        assert report["sieval"] == {"total": 3, "final": 3, "failed": 0, "anomalies": 0}

        model = MockCountingChatModel(answers=ANSWERS)
        _, cached = await _run(tmp_path, model=model, resume=True)
        assert cached == report
        assert model.call_count == 0

    @pytest.mark.anyio
    async def test_a_report_with_failures_is_not_fast_resumed(self, tmp_path):
        runner, _ = await _run(tmp_path)
        root = runner.root_dir
        report = _read(root, "report.json")
        report["sieval"]["failed"] = 1
        (root / "report.json").write_bytes(orjson.dumps(report))
        reports: list[int] = []

        class Spy(GenTask):
            async def report(self, finals, fails):
                reports.append(len(finals))
                return await super().report(finals, fails)

        # Another class is another grader: the full path regrades and reports.
        await _run(tmp_path, task_cls=Spy, resume=True)
        assert reports == [3]


# ===================================================================
# One finalize for every exit path
# ===================================================================
async def _full(tmp_path):
    return await _run(tmp_path, model=TruncatingModel(answers=ANSWERS))


async def _all_terminal(tmp_path):
    runner, _ = await _full(tmp_path)
    (runner.root_dir / "report.json").unlink()
    for name in ("anomalies.json", "progress.json", "profile.json"):
        (runner.root_dir / name).unlink(missing_ok=True)
    return await _run(tmp_path, model=TruncatingModel(answers=ANSWERS), resume=True)


async def _fast_resume(tmp_path):
    runner, _ = await _full(tmp_path)
    for name in ("anomalies.json", "progress.json", "profile.json"):
        (runner.root_dir / name).unlink(missing_ok=True)
    return await _run(tmp_path, model=TruncatingModel(answers=ANSWERS), resume=True)


@pytest.mark.parametrize("exit_path", [_full, _all_terminal, _fast_resume])
class TestOneFinalize:
    @pytest.mark.anyio
    async def test_every_view_is_written(self, tmp_path, exit_path):
        runner, _ = await exit_path(tmp_path)
        root = runner.root_dir
        counts = _read(root, "report.json")["sieval"]
        assert counts == {"total": 3, "final": 3, "failed": 0, "anomalies": 3}

        progress = _read(root, "progress.json")
        assert progress["finished"] is True
        assert progress["completed"] == 3

        anomalies = _read(root, "anomalies.json")
        assert "summary" not in anomalies
        assert set(anomalies["samples"]) == {"0", "1", "2"}

        last = _read(root, "meta.json")["runs"][-1]
        assert last["rules_hash"] == get_rules_hash()
        assert "ended" in last

    @pytest.mark.anyio
    async def test_the_profile_covers_every_sample(self, tmp_path, exit_path):
        """With `record_meta` on, `profile.json` is built from every sample's
        `stage_meta`, resumed samples included, whatever the profile flags."""
        runner, _ = await exit_path(tmp_path)
        profile = _read(runner.root_dir, "profile.json")
        assert profile["stages"]["inferred"]["count"] == 3


class TestAnomalies:
    @pytest.mark.anyio
    async def test_no_timestamped_backup_when_the_rules_change(self, tmp_path):
        runner, _ = await _full(tmp_path)
        root = runner.root_dir
        anomalies = _read(root, "anomalies.json")
        anomalies["meta"]["rules_hash"] = "older-rules"
        (root / "anomalies.json").write_bytes(orjson.dumps(anomalies))
        await _run(tmp_path, model=TruncatingModel(answers=ANSWERS), resume=True)
        assert sorted(p.name for p in root.glob("anomalies*.json")) == [
            "anomalies.json"
        ]
        assert _read(root, "anomalies.json")["meta"]["rules_hash"] == get_rules_hash()

    @pytest.mark.anyio
    async def test_a_fast_resume_keeps_the_report_count_with_anomalies_json(
        self, tmp_path
    ):
        """Re-detection on a fast resume rewrites `anomalies.json`; the reused
        report's `sieval.anomalies` follows it, and nothing else changes."""
        runner, _ = await _full(tmp_path)
        root = runner.root_dir
        report = _read(root, "report.json")
        report["sieval"]["anomalies"] = 0  # as counted under older rules
        (root / "report.json").write_bytes(orjson.dumps(report))
        await _run(tmp_path, model=TruncatingModel(answers=ANSWERS), resume=True)
        refreshed = _read(root, "report.json")
        assert refreshed["sieval"]["anomalies"] == 3
        assert len(_read(root, "anomalies.json")["samples"]) == 3
        report["sieval"]["anomalies"] = 3
        assert refreshed == report

    @pytest.mark.anyio
    async def test_an_interrupted_run_records_its_rules_hash(self, tmp_path):
        runner = TaskRunner(
            GenTask(dataset=MockDataset(SAMPLES), model=MockChatModel(), name="t"),
            _config(tmp_path),
        )

        async def _interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        runner._loader.load_initial_state = _interrupt  # type: ignore[method-assign]
        assert await runner.arun() is None
        last = _read(runner.root_dir, "meta.json")["runs"][-1]
        assert last["interrupted"] is True
        assert last["rules_hash"] == get_rules_hash()


# ===================================================================
# meta.json for readers: the run ID and the config path
# ===================================================================
class TestMetaForReaders:
    @pytest.mark.anyio
    async def test_runs_carry_a_sortable_id(self, tmp_path):
        await _run(tmp_path)
        await _run(tmp_path, resume=True)
        runs = _read(tmp_path / "results", "meta.json")["runs"]
        ids = [run["id"] for run in runs]
        assert len(set(ids)) == 2
        assert ids == sorted(ids)

    @pytest.mark.anyio
    async def test_the_config_path_is_relative_to_the_task_dir(self, tmp_path):
        session = tmp_path / "session"
        (session).mkdir()
        (session / "effective_config.yaml").write_text("tasks: {}\n")
        runner, _ = await _run(
            tmp_path,
            result_dir=str(session / "t"),
            effective_config=str(session / "effective_config.yaml"),
        )
        meta = _read(runner.root_dir, "meta.json")
        assert meta["config"] == "../effective_config.yaml"
        assert (runner.root_dir / meta["config"]).resolve() == (
            session / "effective_config.yaml"
        ).resolve()

    @pytest.mark.anyio
    async def test_without_a_session_config_none_is_recorded(self, tmp_path):
        runner, _ = await _run(tmp_path)
        assert "config" not in _read(runner.root_dir, "meta.json")


def test_saver_writes_only_its_own_files(tmp_path):
    """The saver keeps no per-sample state of its own beside the shards."""
    saver = TaskSaver(root_dir=tmp_path)
    assert not hasattr(saver, "sync_manifest")
