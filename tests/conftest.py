"""
Top-level shared test infrastructure.

Provides mock classes and factories for all test layers.  These are
exposed both as plain classes (for direct instantiation in test files
that share ``conftest.py`` via pytest's discovery) and as pytest
fixtures for injection into test function signatures.

Pytest automatically makes everything defined here available to any
test under ``tests/``.  Test files should access mock classes through
fixtures when possible, or import directly from this module when they
need the class itself (e.g. for ``isinstance`` checks or YAML-runner
class-path references).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import gc
import random
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import anyio
import psutil
import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from loguru import logger as _loguru_logger

from sieval.core.datasets import Dataset
from sieval.core.models import (
    InputScoringResult,
    ModelOutput,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.dialect import (
    Guarantee,
    OutputContract,
    OutputRule,
    PreparedRequest,
    RequestAudit,
)
from sieval.core.models.dialect_registry import get_dialect_spec
from sieval.core.models.gen_model import GenModel
from sieval.core.models.ir import (
    ChatInput,
    CompletionInput,
    TextPart,
    response_field_contract,
)
from sieval.core.runners.runner import TaskRunnerConfig
from sieval.core.tasks.context import TaskContext, TaskStage
from sieval.core.tasks.saver import TaskSaver
from sieval.core.tasks.task import Task


# ===================================================================
# Registry / module-cache isolation
# ===================================================================
class ModuleIsolation:
    """Snapshot a ``sys.modules`` subtree and put it back exactly as found.

    Pair this with any fixture that clears ``TASK_REGISTRY`` / ``_TASK_CLASSES`` /
    ``DATASET_REGISTRY`` / ``SAMPLE_TO_DATASET``. ``import_all_tasks()`` and
    ``get_task_class()`` only re-run a module's ``@sieval_task`` decorator while
    ``sieval.tasks.{name}`` is absent from ``sys.modules``, so a registry and its
    module cache have to move as a unit — clearing one half alone breaks a test in
    one direction or the other:

    * cleared registry + cached modules — registration silently no-ops, so the
      name stays unregistered and ``get_task_class()`` raises ``KeyError``.
    * restored registry + purged modules — the next ``import_all_tasks()`` re-runs
      the decorators against an already-populated registry and trips the
      duplicate-name guard.

    ``sys.modules`` is not the only view. Parent packages keep their own attribute
    bindings, and a dotted-string target (``monkeypatch.setattr("pkg.mod.attr")``,
    ``mock.patch("pkg.mod.attr")``) is resolved by attribute traversal from the
    root package, never through the cache — so a child attribute left pointing at
    a discarded copy patches a module nobody uses. Both directions therefore keep
    the parent bindings in step: ``evict()`` unbinds the modules it drops, and
    ``restore()`` rebinds parents for the snapshotted modules *and* unbinds
    whatever the test imported on top of them.

    A dropped module left bound on its parent is not a cosmetic leak — it is the
    same failure this class exists to prevent, one scope earlier. ``from pkg
    import submodule`` resolves through ``hasattr`` before it considers an import,
    so the stale attribute wins, the module body never re-executes, and the
    decorator never re-runs against the registry the fixture just cleared.

    ``scope`` and ``exclude`` entries ending in ``"."`` match by prefix; the rest
    match exactly. ``exclude`` wins::

        ModuleIsolation(("sieval.tasks.",))  # submodules only
        ModuleIsolation(("sieval.tasks", "sieval.tasks."))  # package included
        ModuleIsolation(("sieval.datasets.theoremqa",))  # that one module

    Exclude a subtree when something *outside* the scope holds a reference into
    it that a restore cannot repair — see the downloader note in
    ``tests/unit/cli/conftest.py``.

    ``lazy_packages`` names packages whose lazy ``__getattr__`` resolves an export
    once and then caches it in the package's own ``__dict__`` (``sieval.tasks``,
    ``sieval.datasets``). Those caches move with the modules for the same reason
    the parent attributes do: a class cached from a copy this fixture later
    discards no longer matches its own ``SAMPLE_TO_DATASET`` key, and the next
    ``@sieval_task`` FK lookup fails with "No @sieval_dataset found for sample
    type". The package's ``__all__`` is the authoritative name list.
    """

    def __init__(
        self,
        scope: tuple[str, ...],
        lazy_packages: tuple[str, ...] = (),
        exclude: tuple[str, ...] = (),
    ) -> None:
        self._scope = scope
        self._lazy_packages = lazy_packages
        self._exclude = exclude
        self._modules: dict[str, ModuleType] = {}
        self._exports: dict[tuple[str, str], object] = {}

    @staticmethod
    def _matches(name: str, patterns: tuple[str, ...]) -> bool:
        return any(
            name.startswith(pattern) if pattern.endswith(".") else name == pattern
            for pattern in patterns
        )

    def _in_scope(self, name: str) -> bool:
        return self._matches(name, self._scope) and not self._matches(
            name, self._exclude
        )

    def _iter_lazy_caches(self) -> Iterator[tuple[ModuleType, str]]:
        """Yield ``(package, export_name)`` for every declared lazy export in play."""
        for pkg_name in self._lazy_packages:
            pkg = sys.modules.get(pkg_name)
            if pkg is None:
                continue
            for export in getattr(pkg, "__all__", ()):
                yield pkg, export

    def snapshot(self) -> None:
        """Record the module objects and cached lazy exports currently in scope."""
        self._modules = {
            name: module for name, module in sys.modules.items() if self._in_scope(name)
        }
        self._exports = {
            (pkg.__name__, export): pkg.__dict__[export]
            for pkg, export in self._iter_lazy_caches()
            if export in pkg.__dict__
        }

    @staticmethod
    def _unbind_from_parent(name: str, module: ModuleType) -> None:
        """Drop *name*'s attribute on its parent package, if it still points here.

        Identity is checked so this can never clobber an unrelated same-named
        attribute; a parent that is itself gone needs no repair and is skipped.
        """
        parent_name, _, attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and parent.__dict__.get(attr) is module:
            del parent.__dict__[attr]

    def evict(self) -> None:
        """Drop the snapshotted modules, their parent bindings, and lazy exports."""
        # Caches first: a package that is itself in scope takes its ``__dict__``
        # with it, and then there is nothing left to pop from.
        self._drop_lazy_caches()
        for name, module in self._modules.items():
            del sys.modules[name]
            self._unbind_from_parent(name, module)

    def _drop_lazy_caches(self) -> None:
        for pkg, export in list(self._iter_lazy_caches()):
            pkg.__dict__.pop(export, None)

    def restore(self) -> None:
        """Purge whatever is live in scope, then reinstate the snapshot."""
        live = {
            name: module for name, module in sys.modules.items() if self._in_scope(name)
        }
        for name in live:
            del sys.modules[name]
        sys.modules.update(self._modules)
        for name, module in self._modules.items():
            parent_name, _, attr = name.rpartition(".")
            if (parent := sys.modules.get(parent_name)) is not None:
                setattr(parent, attr, module)
        # Unbind the copies the test imported on top of the snapshot.
        for name, module in live.items():
            if name not in self._modules:
                self._unbind_from_parent(name, module)
        # Same purge-then-reinstate as the modules: an export the test resolved
        # on its own would otherwise outlive the copy it came from.
        self._drop_lazy_caches()
        for (pkg_name, export), value in self._exports.items():
            if (pkg := sys.modules.get(pkg_name)) is not None:
                pkg.__dict__[export] = value


# ===================================================================
# Loguru pytest-safe sink (autouse session fixture)
# ===================================================================
@pytest.fixture(scope="session", autouse=True)
def _loguru_pytest_safe_sink():
    """Route loguru through ``sys.__stderr__`` for the test session.

    Without this, background asyncio tasks that emit logs after a test
    body returns can hit ``ValueError: I/O operation on closed file``
    once pytest tears down its captured ``sys.stderr`` — pytest then
    flips the affected test from PASSED to FAILED. ``sys.__stderr__``
    is the original, uncaptured stream, so closure-during-teardown
    cannot race with stragglers. Per-test fixtures that bridge loguru
    into ``caplog`` layer their own sink atop and are unaffected.
    """
    _loguru_logger.remove()
    # ``sys.__stderr__`` is typed ``TextIO | None`` (None only in embedded
    # interpreters with no stderr — never the case under pytest).
    assert sys.__stderr__ is not None, "sys.__stderr__ unavailable"
    sink_id = _loguru_logger.add(sys.__stderr__, catch=True)
    yield
    # Tests that import sieval's CLI trigger ``setup_logging()``, whose
    # first-run path calls ``logger.remove()`` with no args and replaces
    # the active sink. By session teardown our recorded ``sink_id`` may
    # already be gone — that's fine, the safe-sink served its purpose.
    with suppress(ValueError):
        _loguru_logger.remove(sink_id)


# ===================================================================
# Neutralize openai's client finalizer (autouse session fixture)
# ===================================================================
@pytest.fixture(scope="session", autouse=True)
def _neutralize_openai_client_finalizer():
    """Disable ``openai`` ``AsyncHttpxClientWrapper.__del__`` during the session.

    The wrapper's finalizer runs
    ``asyncio.get_running_loop().create_task(self.aclose())``. Tests build many
    short-lived models — each constructs an ``AsyncOpenAI`` client — across
    anyio's per-test event loops without closing them. When such a client is
    garbage-collected during a *later* test, its finalizer schedules ``aclose()``
    on the current loop, which then tears down connections bound to the earlier,
    now-closed loop, raising an intermittent ``RuntimeError: Event loop is
    closed`` on CPython 3.12 that surfaces as an unrelated test failing.

    Production is unaffected: a command runs a single long-lived loop via one
    ``anyio.run`` call, so there is no closed prior loop. The clients here are
    mock-backed with nothing meaningful to close, so dropping the finalizer is
    safe for tests. Not restored on teardown — re-enabling it would just
    reintroduce the race for clients collected during session teardown.
    """
    from openai._base_client import AsyncHttpxClientWrapper

    AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[method-assign]
    yield


# ===================================================================
# Mock Dataset
# ===================================================================
class MockDataset(Dataset):
    """Dataset that returns samples from a provided list."""

    def __init__(self, samples: list[dict] | None = None):
        self._samples = (
            samples
            if samples is not None
            else [
                {"question": "What is 1+1?", "answer": "2"},
                {"question": "What is 2+3?", "answer": "5"},
                {"question": "What is 10-7?", "answer": "3"},
            ]
        )
        super().__init__("dummy")

    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        return HFDatasetDict({"test": HFDataset.from_list(self._samples)})


# ===================================================================
# Mock Models
#
# RFC #25: mocks stub at the Dialect seam. Each mock model overrides
# ``_build_default_transport`` to return a ``HandlerTransport`` bound to
# its ``_stub_arun(Request) -> Response`` handler, so ``agenerate`` /
# ``alogprobs`` exercise the real request builders and Response bridge
# while the wire layer stays canned. Subclass mocks override
# ``_stub_arun`` and chain via ``super()``.
# ===================================================================
class HandlerTransport:
    """Dialect double: forwards ``execute`` to a handler coroutine.

    Records every Request in ``self.requests`` so tests can assert on the
    canonical IR while still exercising Model's real audit and pool path.
    """

    def __init__(self, handler, dialect_id: str):
        spec = get_dialect_spec(dialect_id)
        self._handler = handler
        self.dialect_id = dialect_id
        self.connection_family = spec.connection_family
        self.output_contract = OutputContract(
            {
                name: OutputRule(Guarantee.PRESENT_OR_ERROR)
                for name, (role, _) in response_field_contract().items()
                if role == "channel"
            }
        )
        self.requests: list[Request] = []

    def validate_request(self, req, audit: RequestAudit, plan) -> None:
        del req, audit, plan

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        consumed = frozenset(
            path for path in audit.active if path not in audit.decisions
        )
        for path in consumed:
            audit.consumed(path)
        return PreparedRequest(
            operation="handler",
            body={},
            consumed_paths=consumed,
            passthrough={},
            context=req,
        )

    async def execute(self, prepared: PreparedRequest) -> Response:
        req = prepared.context
        assert isinstance(req, Request)
        self.requests.append(req)
        return await self._handler(req)


def prompt_of(req: Request) -> str:
    """Extract the flat question text from canonical completion/chat input."""
    if isinstance(req.input, CompletionInput):
        return req.input.text
    assert isinstance(req.input, ChatInput)
    if not req.input.messages:
        return ""
    return "".join(
        part.text
        for part in req.input.messages[-1].content
        if isinstance(part, TextPart)
    )


def n_of(req: Request) -> int:
    """Number of samples requested."""
    return req.sampling.n


class MockChatModel(ChatModel):
    """ChatModel that returns deterministic answers without calling any API."""

    def __init__(
        self,
        answers: dict[str, str | list[str]] | None = None,
        default_answer: str = "unknown",
        **kwargs,
    ):
        self._answers = answers or {}
        self._default_answer = default_answer
        super().__init__(model="mock-chat", api_key="fake", **kwargs)

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        q = prompt_of(req)
        answer = self._answers.get(q, self._default_answer)

        n = n_of(req)
        if isinstance(answer, list):
            texts = answer[:n] if len(answer) >= n else answer
        else:
            texts = [answer] * n

        return Response(
            texts=tuple(texts),
            finish_reasons=("stop",) * len(texts),
            usage=UsageStats(input_tokens=10, output_tokens=2, total_tokens=12),
            request_params={"model": "mock-chat", "n": n},
        )


class MockGenModel(GenModel):
    """GenModel that supports alogprobs without calling any API."""

    def __init__(
        self,
        logprob_scores: dict[str, float] | None = None,
        default_answer: str = "unknown",
        **kwargs,
    ):
        self._logprob_scores = logprob_scores or {}
        self._default_answer = default_answer
        super().__init__(model="mock-gen", api_key="fake", **kwargs)

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_completions")

    async def _stub_arun(self, req: Request) -> Response:
        if not (req.scoring.sampled_logprobs or req.scoring.input_scoring):
            return Response(
                texts=(self._default_answer,),
                finish_reasons=("stop",),
                usage=UsageStats(input_tokens=10, output_tokens=2, total_tokens=12),
            )

        # Extract the last character as the option label
        prompt = prompt_of(req)
        option_label = prompt.rstrip()[-1] if prompt.strip() else "A"
        score = self._logprob_scores.get(option_label, -10.0)
        max_tokens = req.sampling.max_tokens

        input_scoring = None
        if req.scoring.input_scoring:
            input_scoring = InputScoringResult(
                (TokenLogprob(token=f" {option_label}", logprob=score),)
            )

        return Response(
            texts=("",),
            finish_reasons=("stop",),
            logprobs=(
                (TokenLogprob(token=f" {option_label}", logprob=score),)
                if req.scoring.sampled_logprobs
                else None
            ),
            top_logprobs=(
                ((TopKEntry(token=f" {option_label}", logprob=score),),)
                if req.scoring.top_logprobs > 0
                else None
            ),
            input_scoring=input_scoring,
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            request_params={"max_tokens": max_tokens},
        )


class MockJudgeModel(ChatModel):
    """ChatModel that acts as a judge, returning configurable verdicts."""

    def __init__(self, verdict: str = "yes", **kwargs):
        self._verdict = verdict
        super().__init__(model="mock-judge", api_key="fake", **kwargs)

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(
            texts=(self._verdict,),
            finish_reasons=("stop",),
            usage=UsageStats(input_tokens=20, output_tokens=1, total_tokens=21),
            request_params={"model": "mock-judge"},
        )


class MockFailingChatModel(ChatModel):
    """ChatModel that fails for specified number of calls, then succeeds."""

    def __init__(self, fail_count: int = 1, success_answer: str = "42", **kwargs):
        self._call_count = 0
        self._fail_count = fail_count
        self._success_answer = success_answer
        super().__init__(model="mock-failing", api_key="fake", **kwargs)

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise TimeoutError(f"Simulated failure #{self._call_count}")
        return Response(
            texts=(self._success_answer,),
            finish_reasons=("stop",),
            usage=UsageStats(input_tokens=5, output_tokens=1, total_tokens=6),
        )


class MockAlwaysFailModel(ChatModel):
    """ChatModel that always raises an exception."""

    def __init__(self, error: type[Exception] = TimeoutError, **kwargs):
        self._error = error
        super().__init__(model="mock-always-fail", api_key="fake", **kwargs)

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        raise self._error("Always fails")


class MockCountingChatModel(MockChatModel):
    """MockChatModel that counts how many times the transport is hit."""

    def __init__(self, **kwargs):
        self.call_count = 0
        super().__init__(**kwargs)

    async def _stub_arun(self, req: Request) -> Response:
        self.call_count += 1
        return await super()._stub_arun(req)


class MockSelectiveFailModel(ChatModel):
    """ChatModel that fails on first call for prompts matching fail_samples."""

    def __init__(
        self,
        fail_samples: set[str] | None = None,
        answers: dict[str, str] | None = None,
        default_answer: str = "42",
        **kwargs,
    ):
        self._fail_samples = fail_samples or set()
        self._answers = answers or {}
        self._default_answer = default_answer
        self._call_counts: dict[str, int] = {}
        super().__init__(model="mock-selective", api_key="fake", **kwargs)

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        q = prompt_of(req)
        self._call_counts[q] = self._call_counts.get(q, 0) + 1

        # Fail on first call if prompt matches any fail pattern
        if self._call_counts[q] <= 1 and q in self._fail_samples:
            raise TimeoutError(f"Simulated first-time failure for: {q}")

        answer = self._answers.get(q, self._default_answer)
        return Response(
            texts=(answer,),
            finish_reasons=("stop",),
            usage=UsageStats(input_tokens=10, output_tokens=2, total_tokens=12),
        )


# ===================================================================
# Factory Functions
# ===================================================================
def make_config(tmp_path, **overrides) -> TaskRunnerConfig:
    """Create a TaskRunnerConfig with test-friendly defaults.

    Pass ``result_dir=None`` to omit the result directory (e.g. when using
    MultiTaskRunner, which manages per-task directories itself).
    """
    defaults: dict = {
        "show_progress": False,
        "detect_anomalies": False,
        "profile_io": False,
        "profile_stages": False,
        "profile_usage": False,
        "dump_progress": False,
    }
    # Only set result_dir if the caller hasn't explicitly overridden it
    if "result_dir" not in overrides:
        defaults["result_dir"] = str(tmp_path / "results")
    defaults.update(overrides)
    # Drop result_dir=None so TaskRunnerConfig uses its own default
    if defaults.get("result_dir") is None:
        defaults.pop("result_dir", None)
    return TaskRunnerConfig(**defaults)


# ===================================================================
# Pytest fixtures
# Tests can request these by name instead of importing directly.
# ===================================================================
@pytest.fixture
def mock_dataset():
    return MockDataset()


@pytest.fixture
def mock_chat_model():
    return MockChatModel()


@pytest.fixture
def mock_gen_model():
    return MockGenModel()


@pytest.fixture
def mock_judge_model():
    return MockJudgeModel()


@pytest.fixture
def mock_always_fail_model():
    return MockAlwaysFailModel()


# ===================================================================
# Performance / Acceptance test infrastructure
# ===================================================================
@dataclass(frozen=True, slots=True)
class IOProfile:
    """Describes a model call I/O pattern for performance benchmarking."""

    name: str
    input_size: int = 100
    output_size: int = 100
    latency_s: float = 0.01
    latency_jitter: float = 0.005
    n: int = 1
    calls_per_sample: int = 1


COMMON_PROFILES = [
    IOProfile(
        "short_in_short_out",
        input_size=100,
        output_size=10,
        latency_s=0.005,
        latency_jitter=0.0,
    ),
    IOProfile(
        "long_in_short_out",
        input_size=4000,
        output_size=50,
        latency_s=0.01,
        latency_jitter=0.0,
    ),
    IOProfile(
        "short_in_long_out",
        input_size=200,
        output_size=2000,
        latency_s=0.03,
        latency_jitter=0.0,
    ),
    IOProfile(
        "balanced",
        input_size=1000,
        output_size=500,
        latency_s=0.015,
        latency_jitter=0.0,
    ),
]


class LatencyMockChatModel(ChatModel):
    """ChatModel with configurable latency and payload size for benchmarks."""

    def __init__(
        self,
        latency_s: float = 0.01,
        latency_jitter: float = 0.005,
        output_size: int = 100,
        default_answer: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model="mock-latency", api_key="fake", **kwargs)
        self._latency_s = latency_s
        self._latency_jitter = latency_jitter
        self._output_size = output_size
        self._output_text = default_answer or ("x" * max(1, output_size))

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        jitter = random.uniform(-self._latency_jitter, self._latency_jitter)
        await anyio.sleep(max(0, self._latency_s + jitter))
        n = n_of(req)
        prompt = prompt_of(req)
        input_tokens = max(1, len(prompt) // 4) if prompt else 10
        output_tokens = max(1, self._output_size)
        return Response(
            texts=(self._output_text,) * n,
            finish_reasons=("stop",) * n,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            request_params={"model": "mock-latency", "n": n},
        )

    @classmethod
    def from_profile(cls, profile: IOProfile, **kwargs: Any) -> "LatencyMockChatModel":
        return cls(
            latency_s=profile.latency_s,
            latency_jitter=profile.latency_jitter,
            output_size=profile.output_size,
            **kwargs,
        )


class PerfMockDataset(Dataset):
    """Dataset from an in-memory list, for performance tests."""

    def __init__(self, samples: list[dict]):
        self._samples = samples
        super().__init__("dummy")

    def load(self, name_or_path: str, **kwargs: Any) -> HFDatasetDict:
        return HFDatasetDict({"test": HFDataset.from_list(self._samples)})


def make_large_dataset(n: int, payload_size: int = 100) -> PerfMockDataset:
    """Generate a dataset with n samples, each with configurable payload."""
    padding = "p" * max(0, payload_size)
    samples = [{"question": f"Q{i} {padding}", "answer": f"A{i}"} for i in range(n)]
    return PerfMockDataset(samples)


def make_profiled_dataset(n: int, profile: IOProfile) -> PerfMockDataset:
    return make_large_dataset(n, payload_size=profile.input_size)


class BenchmarkTask(Task):
    """Standard 4-stage task for performance benchmarks."""

    model_type = "chat"

    def __init__(
        self,
        dataset: Dataset,
        model: ChatModel,
        name: str | None = None,
        output_size: int = 100,
        calls_per_sample: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._output_size = output_size
        self._calls_per_sample = calls_per_sample

    async def preprocess(self, raw: Any, ctx: Any) -> str:
        return raw["question"]

    async def infer(self, pre: str, ctx: Any) -> ModelOutput:
        return await self.model.agenerate(pre)

    async def postprocess(self, inf: ModelOutput, ctx: Any) -> str:
        return inf.texts[0]

    async def feedback(self, post: str, ctx: Any) -> tuple[bool, dict]:
        for _ in range(self._calls_per_sample - 1):
            await self.model.agenerate(post)
        detail = "d" * self._output_size
        correct = post.strip() == str(ctx.raw_sample.get("answer", ""))
        return True, {"correct": correct, "detail": detail}

    async def report(self, finals: list, fails: list) -> dict:
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result.get("correct")
        )
        return {"accuracy": correct / total if total else 0.0, "total": total}

    @classmethod
    def from_profile(
        cls,
        profile: IOProfile,
        model: LatencyMockChatModel | None = None,
        n_samples: int = 100,
        name: str | None = None,
    ) -> tuple["BenchmarkTask", PerfMockDataset]:
        dataset = make_profiled_dataset(n_samples, profile)
        mdl = model or LatencyMockChatModel.from_profile(profile)
        task = cls(
            dataset=dataset,
            model=mdl,
            name=name or f"bench_{profile.name}",
            output_size=profile.output_size,
            calls_per_sample=profile.calls_per_sample,
        )
        return task, dataset


class MultiIterBenchmarkTask(BenchmarkTask):
    """BenchmarkTask that only finalizes after a configurable number of iterations."""

    def __init__(
        self,
        dataset: Dataset,
        model: ChatModel,
        name: str | None = None,
        output_size: int = 100,
        calls_per_sample: int = 1,
        finalize_after: int = 3,
    ):
        super().__init__(
            dataset=dataset,
            model=model,
            name=name,
            output_size=output_size,
            calls_per_sample=calls_per_sample,
        )
        self._finalize_after = finalize_after

    async def feedback(self, post: str, ctx: Any) -> tuple[bool, dict]:
        for _ in range(self._calls_per_sample - 1):
            await self.model.agenerate(post)
        detail = "d" * self._output_size
        finalize = ctx.iteration >= self._finalize_after - 1
        return finalize, {
            "correct": finalize,
            "detail": detail,
            "iteration": ctx.iteration,
        }


class PerfTimer:
    """Context manager for high-resolution timing."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "PerfTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


@contextmanager
def suite_heap_excluded() -> Iterator[None]:
    """Keep the rest of the suite's live heap out of a short timed window.

    A gen2 collection costs in proportion to the whole live heap, which by the
    time a full run reaches the perf gates is thousands of other tests'
    fixtures. After ``tests/unit`` alone this process holds ~833k tracked
    objects and one gen2 pass takes 286 ms -- longer than the 0.20 s window
    ``test_throughput_vs_concurrency`` measures at concurrency 16, and the whole
    difference between 61.7% and 18.9% efficiency for unchanged code.

    ``gc.freeze()``, not ``gc.disable()``: objects the code under test allocates
    are still collected and still charged to the window, so GC pressure the
    pipeline itself creates stays measurable. Only the heap it did not allocate
    stops being rescanned. (``test_serialization``'s ``_gc_paused`` disables
    outright -- right for a tight loop where any collection is pure noise.)

    Whether a gate needs this is whether one 286 ms pause fits inside its own
    margin, not whether its window is short. All were checked. The absolute
    bounds have room to spare, and ``iteration_overhead`` (2.98x/5.0x),
    ``record_each_stage`` (46%/200%), ``multi_task_runner`` (~0.4 s/~1.9 s) and
    ``composite_limiter`` (0.03 ms/1.0 ms) survive a worst-case landing;
    ``dep_loading`` takes the best of five, so a pause must hit every run.
    ``dataset_iteration_overhead`` only looks tighter -- 443.7x against 600x
    leaves 37 ms fresh -- but its plain-list denominator slows 4x on locality
    under 1.03M live objects, dropping the ratio to 104x and raising headroom to
    ~460 ms against a 206 ms pass. Headroom outgrows the pause.

    Wrapped: ``test_throughput_vs_concurrency``, and ``test_benchmark_scenarios``
    via ``_run_scenario`` -- same exposure at no cost, though this does *not*
    close that one's residual gap, whose cause is still open.
    ``test_pipeline_memory_scaling`` is fixed by measurement instead.

    Not reentrant: ``gc.unfreeze()`` is all-or-nothing, so a nested exit would
    drop the outer guard. A nested enter is a no-op -- the outer freeze already
    covers what this window inherited, and anything since belongs to the code
    under test, which must stay charged.
    """
    if gc.get_freeze_count():
        yield
        return

    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


def samples_per_second(n_samples: int, elapsed_s: float) -> float:
    return n_samples / elapsed_s if elapsed_s > 0 else float("inf")


class MemoryTracker:
    """Tracks process-level RSS memory via psutil."""

    def __init__(self, sample_interval_s: float = 0.05) -> None:
        self._proc = psutil.Process()
        self._sample_interval_s = max(0.0, sample_interval_s)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None

        self.baseline_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.final_mb: float = 0.0

    @property
    def delta_mb(self) -> float:
        return self.final_mb - self.baseline_mb

    def _read_rss_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 * 1024)

    def _update_peak(self, current_mb: float) -> None:
        with self._lock:
            if current_mb > self.peak_mb:
                self.peak_mb = current_mb

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self._sample_interval_s):
            self._update_peak(self._read_rss_mb())

    def start(self) -> None:
        gc.collect()
        gc.collect()
        baseline = self._read_rss_mb()
        with self._lock:
            self.baseline_mb = baseline
            self.peak_mb = baseline
            self.final_mb = baseline

        self._stop_event.clear()
        self._sampler_thread = None
        if self._sample_interval_s > 0:
            self._sampler_thread = threading.Thread(
                target=self._sampling_loop,
                name="memory-tracker-sampler",
                daemon=True,
            )
            self._sampler_thread.start()

    def snapshot(self) -> float:
        current = self._read_rss_mb()
        self._update_peak(current)
        return current

    def stop(self) -> None:
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=max(0.5, self._sample_interval_s * 10))
            self._sampler_thread = None

        final = self._read_rss_mb()
        with self._lock:
            self.final_mb = final
        self._update_peak(final)

    def __enter__(self) -> "MemoryTracker":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def make_perf_config(tmp_path: Any, **overrides: Any) -> "TaskRunnerConfig":
    """TaskRunnerConfig for performance measurements."""
    from sieval.core.runners.runner import TaskRunnerConfig

    defaults: dict[str, Any] = {
        "result_dir": str(tmp_path / "perf_results"),
        "show_progress": False,
        "detect_anomalies": False,
        "profile_io": True,
        "profile_stages": True,
        "profile_usage": False,
        "dump_progress": False,
    }
    defaults.update(overrides)
    return TaskRunnerConfig(**defaults)


def _make_bench_ctx(
    sample_id: int,
    stage: TaskStage = TaskStage.FINAL,
    payload_size: int = 100,
    iteration: int = 0,
) -> TaskContext:
    """Build a TaskContext at stage with configurable payload size."""
    padding = "x" * max(0, payload_size)
    raw = {"question": f"Q{sample_id} {padding}", "answer": f"A{sample_id}"}
    ctx: TaskContext = TaskContext(
        sample_id=sample_id, raw_sample=raw, iteration=iteration
    )

    if stage.value in ("initial",):
        return ctx

    ctx = ctx.to_preprocessed(f"pre_{sample_id}_{padding[:50]}")
    if stage == TaskStage.PREPROCESSED:
        return ctx

    infer_text = f"inf_{sample_id}_{padding}"
    infer_result = ModelOutput(
        model={"model": "mock", "api_base": None, "default_params": {}},
        texts=[infer_text],
        finish_reasons=["stop"],
        usage={
            "input_tokens": 10,
            "output_tokens": max(1, payload_size // 4),
            "total_tokens": 10 + max(1, payload_size // 4),
        },
    )
    ctx = ctx.to_inferred(infer_result)
    if stage == TaskStage.INFERRED:
        return ctx

    ctx = ctx.to_postprocessed(infer_text)
    if stage == TaskStage.POSTPROCESSED:
        return ctx

    ctx = ctx.to_feedback({"correct": True, "detail": padding})
    if stage == TaskStage.FEEDBACK:
        return ctx

    ctx = ctx.to_final()
    return ctx


def require_available_memory_gb(min_gb: float) -> None:
    """pytest.skip if available memory is below min_gb."""
    avail_gb = psutil.virtual_memory().available / (1024**3)
    if avail_gb < min_gb:
        pytest.skip(
            f"Insufficient memory: {avail_gb:.1f}GB available, {min_gb}GB required"
        )


async def write_completed_samples(
    root: Path, n_completed: int, shard_samples: int = 256
) -> None:
    """Write n_completed FINAL contexts to disk via TaskSaver.

    Stamps ``meta.json`` first, as a real run does at start, so the partial
    run this fabricates is resumable under the resume version gate.
    """
    saver = TaskSaver(
        root_dir=root,
        shard_samples=shard_samples,
        shard_write_concurrency=8,
        write_buffer_size=max(n_completed + 1, 64),
        write_buffer_flush_interval=9999.0,
    )
    await saver.write_run_meta()
    for i in range(n_completed):
        ctx = _make_bench_ctx(i, TaskStage.FINAL)
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)
    await saver.flush()
