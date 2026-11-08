"""Pattern: ``concurrency_limit: auto`` through ``sieval run``, plan to report.

Planning binds each root before its engine exists. The default ``signal:
throughput`` reads no KV pool; ``kv`` and ``both`` read it at the first
request; without ``auto`` nothing is read. The engines are the stubs of
``test_staged_run``; their throughput probe is :class:`Counters`.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from sieval.core.models import ServingFacts
from sieval.core.models.admission import (
    AdaptiveLimit,
    AdaptiveTokenBudget,
    CombinedGate,
    EngineSample,
    TokenBudget,
)
from sieval.core.models.deployment import ConnectionPool
from tests.integration.test_staged_run import (
    JUDGE_FROM,
    SUBJECT_UNTIL,
    Engines,
    _dry,
    _run,
    _write,
    stub_engines,
)

AUTO: dict[str, Any] = {"concurrency_limit": "auto"}
KV: dict[str, Any] = {"concurrency_limit": {"auto": {"signal": "kv"}}}
BOTH: dict[str, Any] = {"concurrency_limit": {"auto": {"signal": "both"}}}


class Counters:
    """The stub engines' ``SglangThroughputProbe``: counters that grow by the
    requests the engine answered."""

    made: list["Counters"] = []

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.reads = 0
        self.closed = False
        Counters.made.append(self)

    async def __call__(self) -> EngineSample:
        self.reads += 1
        return EngineSample(tokens=1000 * self.reads, retracted=0, token_usage=0.1)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def counters() -> Iterator[type[Counters]]:
    Counters.made.clear()
    with patch("sieval.cli.session.SglangThroughputProbe", new=Counters):
        yield Counters


class KvFacts:
    """``kv_facts`` of the stub engines: what an up engine reports, and who asked."""

    def __init__(self, engines: Engines, *, reports: bool = True) -> None:
        self._engines = engines
        self._reports = reports
        self.asked: list[str] = []

    async def __call__(self, backend: str, endpoint: str) -> ServingFacts:
        self.asked.append(backend)
        if not self._reports or not self._engines.up:
            return ServingFacts()
        return ServingFacts(kv_capacity_tokens=500_000, context_tokens=32_768)


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Engines]:
    yield from stub_engines(monkeypatch, tmp_path)


@pytest.fixture
def facts(engines: Engines) -> Iterator[KvFacts]:
    reader = KvFacts(engines)
    with (
        patch("sieval.cli.run.kv_facts", new=reader),
        patch("sieval.cli.session.kv_facts", new=reader),
    ):
        yield reader


def _sglang(config: Path, *, judge: bool = True) -> Path:
    """*config* with the subject (and the judge) served by SGLang."""
    cfg = yaml.safe_load(config.read_text())
    for name in ("subject", "judge") if judge else ("subject",):
        model = cfg["models"][name]
        model["infer"] = {"backend": "sglang", "checkpoint": model.pop("path")}
    config.write_text(yaml.safe_dump(cfg))
    return config


def _budgets() -> list[object]:
    """The ``auto`` gates the pools of this test admitted through."""
    return list(_seen)


_seen: list[object] = []


@pytest.fixture(autouse=True)
def _record_gates(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remember every gate a request was admitted by."""
    _seen.clear()
    real = ConnectionPool.admission

    async def admission(self: ConnectionPool):
        gate = await real(self)
        if gate is not None and gate not in _seen:
            _seen.append(gate)
        return gate

    monkeypatch.setattr(ConnectionPool, "admission", admission)
    yield


def _argv(engines: Engines, model: str) -> list[str]:
    (command,) = engines.commands[model]
    return command.cli_args


def _in_flight(tmp_path: Path, task: str = "judged") -> list[dict[str, Any]]:
    path = tmp_path / "results" / task / "requests.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return [
        line["in_flight"] for line in lines if line["model"]["id"] == "subject-model"
    ]


class TestAutoConcurrency:
    @pytest.mark.anyio
    @pytest.mark.parametrize("limit", [AUTO, KV, BOTH], ids=["auto", "kv", "both"])
    async def test_a_dry_run_accepts_auto_on_a_launched_sglang_engine(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
        limit: dict[str, Any],
    ) -> None:
        result = await _dry(_sglang(_write(tmp_path, subject=limit)))

        assert result["n_errors"] == 0, result
        # Planning launched nothing and read no engine.
        assert engines.events == []
        assert facts.asked == []

    @pytest.mark.anyio
    async def test_a_run_admits_by_throughput_and_reads_no_pool(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        reports = await _run(_sglang(_write(tmp_path, subject=AUTO)))

        assert reports["judged"]["correct"] == 3
        # Only the subject sets `auto`.
        (gate,) = _budgets()
        assert isinstance(gate, AdaptiveLimit)
        # Read once at launch for the `KV:` line; the gate does not use it.
        assert facts.asked == ["sglang"]
        # The engine serves the counters `auto` reads; the judge is untouched.
        assert "--enable-metrics" in _argv(engines, "subject")
        assert "--enable-metrics" not in _argv(engines, "judge")
        (probe,) = Counters.made
        assert probe.closed
        # Records show the limit and the controller's last decision.
        records = _in_flight(tmp_path)
        assert records and all("auto" in r for r in records)
        assert {r["auto"]["total"] for r in records} == {32}
        assert {r["auto"]["decision"] for r in records} == {"start"}
        assert all("kv_tokens" not in r for r in records)

    @pytest.mark.parametrize(
        ("limit", "kind", "keys"),
        [
            (KV, AdaptiveTokenBudget, {"kv_tokens"}),
            (BOTH, CombinedGate, {"auto", "kv_tokens"}),
        ],
        ids=["kv", "both"],
    )
    @pytest.mark.anyio
    async def test_a_kv_signal_reads_the_pool_once_the_engine_is_up(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
        limit: dict[str, Any],
        kind: type,
        keys: set[str],
    ) -> None:
        reports = await _run(_sglang(_write(tmp_path, subject=limit)))

        assert reports["judged"]["correct"] == 3
        (gate,) = _budgets()
        assert isinstance(gate, kind)
        budget = gate.budget if isinstance(gate, CombinedGate) else gate
        assert isinstance(budget, TokenBudget)
        assert budget.total_tokens == 500_000
        assert facts.asked == ["sglang"]
        records = _in_flight(tmp_path)
        assert records and all(keys <= r.keys() for r in records)

    @pytest.mark.anyio
    async def test_a_judge_started_at_a_barrier_gets_auto_after_it_is_up(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        config = _sglang(_write(tmp_path, subject=SUBJECT_UNTIL, judge=JUDGE_FROM | KV))

        reports = await _run(config)

        assert reports["judged"]["correct"] == 3
        assert engines.events.index("start judge") < engines.events.index("ask judge")
        adaptive = [b for b in _budgets() if isinstance(b, AdaptiveTokenBudget)]
        assert [b.total_tokens for b in adaptive] == [500_000]

    @pytest.mark.anyio
    async def test_a_judge_started_at_a_barrier_gets_the_throughput_limit(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        config = _sglang(
            _write(tmp_path, subject=SUBJECT_UNTIL, judge=JUDGE_FROM | AUTO)
        )

        reports = await _run(config)

        assert reports["judged"]["correct"] == 3
        assert [type(g) for g in _budgets()] == [AdaptiveLimit]

    @pytest.mark.anyio
    async def test_throughput_runs_on_an_engine_that_reports_no_pool(
        self,
        tmp_path: Path,
        engines: Engines,
    ) -> None:
        reader = KvFacts(engines, reports=False)
        with (
            patch("sieval.cli.run.kv_facts", new=reader),
            patch("sieval.cli.session.kv_facts", new=reader),
        ):
            reports = await _run(_sglang(_write(tmp_path, subject=AUTO)))

        assert reports["judged"]["correct"] == 3
        assert [type(g) for g in _budgets()] == [AdaptiveLimit]

    @pytest.mark.anyio
    async def test_no_limit_gets_no_kv_admission(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        reports = await _run(_sglang(_write(tmp_path)))

        assert reports["judged"]["correct"] == 3
        assert facts.asked == []
        assert _budgets() == []

    @pytest.mark.anyio
    async def test_a_judge_with_no_limit_started_at_a_barrier_gets_no_kv_admission(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        config = _sglang(_write(tmp_path, subject=SUBJECT_UNTIL, judge=JUDGE_FROM))

        reports = await _run(config)

        assert reports["judged"]["correct"] == 3
        assert facts.asked == []
        assert _budgets() == []

    @pytest.mark.anyio
    async def test_auto_on_an_engine_that_reports_no_pool_is_refused(
        self,
        tmp_path: Path,
        engines: Engines,
    ) -> None:
        reader = KvFacts(engines, reports=False)
        with (
            patch("sieval.cli.run.kv_facts", new=reader),
            patch("sieval.cli.session.kv_facts", new=reader),
            pytest.raises(ValueError, match="reported none at /get_server_info"),
        ):
            await _run(_sglang(_write(tmp_path, subject=KV)))

        # Refused once the engine was up and read, before any request.
        assert engines.asked("subject") == 0

    @pytest.mark.anyio
    async def test_a_deferred_judge_that_reports_no_pool_fails_naming_it(
        self,
        tmp_path: Path,
        engines: Engines,
    ) -> None:
        reader = KvFacts(engines, reports=False)
        config = _sglang(_write(tmp_path, subject=SUBJECT_UNTIL, judge=JUDGE_FROM | KV))

        with (
            patch("sieval.cli.run.kv_facts", new=reader),
            patch("sieval.cli.session.kv_facts", new=reader),
        ):
            reports = await _run(config)

        # The judge is bound before it is up, so its pool is read at its first
        # request; with none, each grading call fails naming the root, and no
        # request reaches the engine.
        assert engines.asked("judge") == 0
        assert reports["judged"].get("correct", 0) == 0

    @pytest.mark.anyio
    async def test_auto_on_vllm_is_refused_by_the_dry_run(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        config = _sglang(_write(tmp_path, subject=AUTO))
        cfg = yaml.safe_load(config.read_text())
        cfg["models"]["subject"]["infer"]["backend"] = "vllm"
        config.write_text(yaml.safe_dump(cfg))

        result = await _dry(config)

        assert result["n_errors"] > 0
        assert "'auto' needs an SGLang engine" in str(result)

    @pytest.mark.anyio
    async def test_auto_on_an_endpoint_is_refused_by_the_dry_run(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        config = _write(tmp_path, subject=AUTO)
        cfg = yaml.safe_load(config.read_text())
        subject = cfg["models"]["subject"]
        subject.pop("path")
        subject["api_base"] = "http://127.0.0.1:9/v1"
        subject["api_key"] = "k"
        config.write_text(yaml.safe_dump(cfg))

        result = await _dry(config)

        assert result["n_errors"] > 0
        assert "'auto' needs an SGLang engine" in str(result)
