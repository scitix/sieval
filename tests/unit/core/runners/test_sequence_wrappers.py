"""
A task reading ``model.extra["sequence_wrappers"]`` against two models.

``WrapTask`` is the pattern a sequence task follows: in ``preprocess`` it asks
the model for a wrapper by sequence type and, when there is none, renders
exactly what it rendered before. A wrapping model and a plain one then share
``data`` and ``preprocess`` and differ only in ``inference``, whose
per-sample ``requests`` digest the wrapped prompt as sent.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from sieval.core.models import check_extra, sequence_wrapper
from sieval.core.runners import task_resume_plan
from sieval.core.runners.runner import TaskRunner
from sieval.core.runners.stage_resume import StageResumeError, task_preprocess_identity
from sieval.core.tasks import build_prompt_record
from sieval.core.tasks.run_meta import stage_code
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config, prompt_of

SAMPLES = [
    {"sequence": "ACGT", "partner": "TTGA", "answer": "yes"},
    {"sequence": "GGCC", "partner": "AATT", "answer": "no"},
]

WRAPPERS = {"dna": "<dna>{seq}</dna>"}


class WrapTask(Task):
    """A dual-sequence DNA task: each sequence is wrapped on its own."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        wrap = sequence_wrapper(self.model.extra, "dna")
        first, second = raw["sequence"], raw["partner"]
        if wrap:
            first, second = wrap.format(seq=first), wrap.format(seq=second)
        return build_prompt_record(
            [{"role": "user", "content": f"Do {first} and {second} bind?"}],
            reference=raw["answer"],
        )

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        return True, {"correct": post == ctx.raw_sample["answer"]}

    async def report(self, finals, fails):
        return {"total": len(finals) + len(fails)}


class PlainTask(WrapTask):
    """The same prompt with no wrapping code: what the task rendered before."""

    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": f"Do {raw['sequence']} and {raw['partner']} bind?",
                }
            ],
            reference=raw["answer"],
        )


class EchoModel(MockChatModel):
    """Echoes the prompt into ``request_params``, as a dialect records the body."""

    async def _stub_arun(self, req):
        response = await super()._stub_arun(req)
        return replace(response, request_params={"messages": prompt_of(req)})


def _meta(root: Path) -> dict:
    return orjson.loads((root / "meta.json").read_bytes())


def _model(wrappers=None) -> EchoModel:
    model = EchoModel(default_answer="yes")
    return model.with_args(extra={"sequence_wrappers": wrappers}) if wrappers else model


async def _run(tmp_path, name, model, cls=WrapTask, **cfg):
    task = cls(dataset=MockDataset(SAMPLES), model=model, name="seq")
    runner = TaskRunner(
        task, make_config(tmp_path, result_dir=str(tmp_path / name), **cfg)
    )
    await runner.arun()
    return runner.root_dir


def _sent(root: Path) -> list[str]:
    """Every prompt the candidate was sent, from the ``infer`` records."""
    out = []
    for shard in sorted(root.glob("0/*/*.jsonl")):
        for raw in shard.read_bytes().splitlines():
            record = orjson.loads(raw)
            history = (record.get("stage_meta") or {}).get("inferred", [])
            if record.get("stage") == "inferred" and record.get("meta_last"):
                history = [*history, record["meta_last"]]
            for entry in history:
                out += [c["request_params"]["messages"] for c in entry["model_calls"]]
    return sorted(set(out))


class TestWrappingChangesOnlyTheRequest:
    @pytest.mark.anyio
    async def test_data_and_preprocess_are_shared_inference_differs(self, tmp_path):
        plain = _meta(await _run(tmp_path, "plain", _model()))["identity"]
        wrapped = _meta(await _run(tmp_path, "wrapped", _model(WRAPPERS)))["identity"]

        assert wrapped["data"] == plain["data"]
        assert wrapped["preprocess"] == plain["preprocess"]
        assert wrapped["inference"]["digest"] != plain["inference"]["digest"]
        assert (
            wrapped["inference"]["requests"]["digest"]
            != plain["inference"]["requests"]["digest"]
        )

    @pytest.mark.anyio
    async def test_the_request_bodies_show_the_wrapping(self, tmp_path):
        plain = _sent(await _run(tmp_path, "plain", _model()))
        wrapped = _sent(await _run(tmp_path, "wrapped", _model(WRAPPERS)))
        assert plain == ["Do ACGT and TTGA bind?", "Do GGCC and AATT bind?"]
        assert wrapped == [
            "Do <dna>ACGT</dna> and <dna>TTGA</dna> bind?",
            "Do <dna>GGCC</dna> and <dna>AATT</dna> bind?",
        ]

    @pytest.mark.anyio
    async def test_another_type_s_wrapper_leaves_the_prompt(self, tmp_path):
        """A model wrapping only RNA sends a DNA task's prompt unchanged."""
        plain = _sent(await _run(tmp_path, "plain", _model()))
        rna_only = _sent(
            await _run(tmp_path, "rna", _model({"rna": "<rna>{seq}</rna>"}))
        )
        assert rna_only == plain

    @pytest.mark.anyio
    async def test_absent_key_renders_today_s_prompts_byte_for_byte(self, tmp_path):
        """No `sequence_wrappers`: the reading task sends what a task without
        the reading code sends."""
        before = _sent(await _run(tmp_path, "before", _model(), cls=PlainTask))
        after = _sent(await _run(tmp_path, "after", _model()))
        assert [p.encode() for p in after] == [p.encode() for p in before]

    @pytest.mark.anyio
    async def test_a_resume_with_the_wrapping_model_is_refused(self, tmp_path):
        """Per-sample requests and the model's `extra` are inference: a resume
        that would reuse a plain model's inference for a wrapping one is
        refused, as for any inference change."""
        await _run(tmp_path, "run", _model())
        task = WrapTask(
            dataset=MockDataset(SAMPLES), model=_model(WRAPPERS), name="seq"
        )
        runner = TaskRunner(
            task, make_config(tmp_path, result_dir=str(tmp_path / "run"), resume=True)
        )
        with pytest.raises(StageResumeError, match="stage `inference`"):
            await runner.arun()

    @pytest.mark.anyio
    async def test_the_dry_run_plan_shares_preprocess(self, tmp_path):
        """`task_resume_plan`, which a dry run reads, compares a wrapping model
        with a plain run's directory: preprocess is the same, inference is not."""
        root = await _run(tmp_path, "plain", _model())
        recorded = _meta(root)
        task = WrapTask(
            dataset=MockDataset(SAMPLES), model=_model(WRAPPERS), name="seq"
        )
        keys = {"deterministic": False, "max_iterations": 1, "max_retries": 3}
        plan = task_resume_plan(task, recorded, runner_keys=keys)
        assert plan["stages"]["preprocess"] == "same"
        assert plan["stages"]["inference"] == "differs"
        assert plan["refused"] == "inference"
        assert plan["legacy_preprocess"] is False

    @pytest.mark.anyio
    async def test_plan_run_stages_share_preprocess_across_models(self, tmp_path):
        """The dry-run's per-task `stages` (`plan_run`) for the wrapping model
        over a plain model's directory: preprocess is the same. Only the
        preprocess records are kept, so no inference is reused and the plan is
        not refused."""
        root = await _run(tmp_path, "plain", _model())
        (root / "report.json").unlink()
        for stage_dir in root.glob("0/*"):
            if stage_dir.name != "preprocessed":
                for f in stage_dir.iterdir():
                    f.unlink()
        task = WrapTask(
            dataset=MockDataset(SAMPLES), model=_model(WRAPPERS), name="seq"
        )
        runner = TaskRunner(
            task, make_config(tmp_path, result_dir=str(root), resume=True)
        )
        plan = await runner.plan_run()
        stages = plan["stages"]
        assert stages is not None
        assert stages["stages"]["preprocess"] == "same"
        assert stages["stages"]["data"] == "same"
        assert stages["stages"]["inference"] == "differs"
        assert stages["rerun_from"] != "preprocess"
        assert plan["outcomes"] is not None
        assert plan["outcomes"]["restarted"] == 0
        assert plan["outcomes"]["kept"] == 2

    @pytest.mark.anyio
    async def test_plan_run_refuses_the_wrapping_model_as_the_run_does(self, tmp_path):
        await _run(tmp_path, "run", _model())
        task = WrapTask(
            dataset=MockDataset(SAMPLES), model=_model(WRAPPERS), name="seq"
        )
        runner = TaskRunner(
            task, make_config(tmp_path, result_dir=str(tmp_path / "run"), resume=True)
        )
        with pytest.raises(StageResumeError, match="stage `inference`"):
            await runner.plan_run()


class TestValidation:
    def test_a_template_without_seq_is_refused(self):
        with pytest.raises(ValueError, match=r"has no \{seq\}"):
            _model({"dna": "<dna></dna>"})

    @pytest.mark.parametrize(
        ("wrappers", "match"),
        [
            ({"dna": 3}, "must be a format string"),
            ({"genome": "<g>{seq}</g>"}, "unknown sequence type 'genome'"),
            ({"dna": "<{kind}>{seq}"}, r"may only use \{seq\}"),
            ({"dna": "<dna>{seq"}, "is not a format string"),
            (["dna"], "must map a sequence type"),
        ],
    )
    def test_malformed_wrappers_are_refused(self, wrappers, match):
        with pytest.raises(ValueError, match=match):
            check_extra({"sequence_wrappers": wrappers})

    def test_literal_braces_and_every_type_are_accepted(self):
        check_extra(
            {
                "sequence_wrappers": {
                    "dna": "{{dna}}{seq}",
                    "rna": "<rna>{seq}</rna>",
                    "protein": "<protein>{seq}</protein>",
                    "mol": "<mol>{seq}</mol>",
                },
                "anything_else": [1, 2],
            }
        )

    def test_extra_without_wrappers_is_not_checked(self):
        check_extra(None)
        check_extra({"note": "free-form"})

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            ({"sequence_wrappers": WRAPPERS}, "<dna>{seq}</dna>"),
            ({"sequence_wrappers": {"rna": "<rna>{seq}</rna>"}}, None),
            ({"sequence_wrappers": "<dna>{seq}</dna>"}, None),
            ({}, None),
        ],
    )
    def test_sequence_wrapper_reads_one_type(self, extra, expected):
        assert sequence_wrapper(extra, "dna") == expected


class ShotTask(WrapTask):
    """Prefixes each prompt with the ``train`` split's first row: a few-shot
    task, whose ``preprocess`` reads beyond the sample it renders."""

    async def preprocess(self, raw, ctx):
        shot = self.dataset.dataset_dict["train"][0]["sequence"]
        record = await super().preprocess(raw, ctx)
        record["prompt"][0]["content"] = f"{shot}\n" + record["prompt"][0]["content"]
        return record


class NamedShotTask(WrapTask):
    """Draws its exemplar from the split ``fewshot_split`` names."""

    def __init__(self, *args, fewshot_split: str = "dev", **kwargs):
        super().__init__(*args, **kwargs)
        self._fewshot_split = fewshot_split

    async def preprocess(self, raw, ctx):
        split = self.dataset.dataset_dict[self._fewshot_split]
        record = await super().preprocess(raw, ctx)
        record["prompt"][0]["content"] = (
            f"{split[0]['sequence']}\n" + (record["prompt"][0]["content"])
        )
        return record


class TestFewShotSplits:
    @pytest.mark.anyio
    async def test_a_changed_exemplar_split_changes_preprocess(self, tmp_path):
        from datasets import Dataset as HFDataset

        def task(train_seq):
            dataset = MockDataset(SAMPLES)
            dataset.dataset_dict["train"] = HFDataset.from_list(
                [{"sequence": train_seq, "partner": "", "answer": ""}]
            )
            return ShotTask(dataset=dataset, model=_model(), name="seq")

        keys = {"deterministic": False, "max_iterations": 1, "max_retries": 3}
        first = TaskRunner(
            task("AAAA"), make_config(tmp_path, result_dir=str(tmp_path / "r"))
        )
        await first.arun()
        recorded = _meta(first.root_dir)
        assert "train" in recorded["identity"]["preprocess"]["splits"]
        assert "test" not in recorded["identity"]["preprocess"]["splits"]

        same = task_resume_plan(task("AAAA"), recorded, runner_keys=keys)
        assert same["stages"]["preprocess"] == "same"
        other = task_resume_plan(task("CCCC"), recorded, runner_keys=keys)
        assert other["stages"]["preprocess"] == "differs"
        assert other["rerun_from"] == "preprocess"

    def test_only_the_named_fewshot_split_is_digested(self):
        from datasets import Dataset as HFDataset

        def task(validation_seq):
            dataset = MockDataset(SAMPLES)
            for split, seq in (("dev", "AAAA"), ("validation", validation_seq)):
                dataset.dataset_dict[split] = HFDataset.from_list(
                    [{"sequence": seq, "partner": "", "answer": ""}]
                )
            return NamedShotTask(dataset=dataset, model=_model(), name="seq")

        code = stage_code(NamedShotTask, args=["fewshot_split"])["preprocess"]
        one = task_preprocess_identity(task("CCCC"), code)
        assert list(one["splits"]) == ["dev"]
        # An unread split's rows do not move the identity.
        assert task_preprocess_identity(task("GGGG"), code) == one
