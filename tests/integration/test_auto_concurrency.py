"""Pattern: ``concurrency_limit: auto`` through ``sieval run``, plan to report.

Planning binds each root before its engine exists; the KV pool is read at the
first request. Without ``auto`` no pool is read and nothing is admitted by KV.
The engines are the stubs of ``test_staged_run`` and answer
``/get_server_info`` once launched.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from sieval.core.models import ServingFacts
from sieval.core.models.admission import AdaptiveTokenBudget, TokenBudget
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


def _budgets() -> list[TokenBudget]:
    """The KV budgets the pools of this test admitted through."""
    return list(_seen)


_seen: list[TokenBudget] = []


@pytest.fixture(autouse=True)
def _record_budgets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remember every KV budget a request reserved from."""
    _seen.clear()
    real = TokenBudget.reserve

    def reserve(self: TokenBudget, tokens: int):
        if self not in _seen:
            _seen.append(self)
        return real(self, tokens)

    monkeypatch.setattr(TokenBudget, "reserve", reserve)
    yield


class TestAutoConcurrency:
    @pytest.mark.anyio
    async def test_a_dry_run_accepts_auto_on_a_launched_sglang_engine(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        result = await _dry(_sglang(_write(tmp_path, subject=AUTO)))

        assert result["n_errors"] == 0, result
        # Planning launched nothing and read no engine.
        assert engines.events == []
        assert facts.asked == []

    @pytest.mark.anyio
    async def test_a_run_reads_the_pool_once_the_engine_is_up(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        reports = await _run(_sglang(_write(tmp_path, subject=AUTO)))

        assert reports["judged"]["correct"] == 3
        # Only the subject sets `auto`; the judge, with no limit, has no budget.
        (budget,) = _budgets()
        assert isinstance(budget, AdaptiveTokenBudget)
        assert budget.total_tokens == 500_000
        assert facts.asked == ["sglang"]

    @pytest.mark.anyio
    async def test_a_judge_started_at_a_barrier_gets_auto_after_it_is_up(
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
        assert engines.events.index("start judge") < engines.events.index("ask judge")
        adaptive = [b for b in _budgets() if isinstance(b, AdaptiveTokenBudget)]
        assert [b.total_tokens for b in adaptive] == [500_000]

    @pytest.mark.anyio
    async def test_no_limit_gets_no_kv_admission(
        self,
        tmp_path: Path,
        engines: Engines,
        facts: KvFacts,
    ) -> None:
        reports = await _run(_sglang(_write(tmp_path)))

        assert reports["judged"]["correct"] == 3
        # KV admission is `auto`'s: no pool is read, no budget is reserved.
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
            await _run(_sglang(_write(tmp_path, subject=AUTO)))

        # Refused once the engine was up and read, before any request.
        assert engines.asked("subject") == 0

    @pytest.mark.anyio
    async def test_a_deferred_judge_that_reports_no_pool_fails_naming_it(
        self,
        tmp_path: Path,
        engines: Engines,
    ) -> None:
        reader = KvFacts(engines, reports=False)
        config = _sglang(
            _write(tmp_path, subject=SUBJECT_UNTIL, judge=JUDGE_FROM | AUTO)
        )
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
