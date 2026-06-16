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

import random
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import psutil
import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from loguru import logger as _loguru_logger

from sieval.core.datasets import Dataset
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.models.gen_model import GenModel
from sieval.core.runners.runner import TaskRunnerConfig
from sieval.core.tasks.context import TaskContext, TaskStage
from sieval.core.tasks.saver import TaskSaver
from sieval.core.tasks.task import Task


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
# ===================================================================
class MockChatModel(ChatModel):
    """ChatModel that returns deterministic answers without calling any API."""

    def __init__(
        self,
        answers: dict[str, str | list[str]] | None = None,
        default_answer: str = "unknown",
        **kwargs,
    ):
        super().__init__(model="mock-chat", api_key="fake", **kwargs)
        self._answers = answers or {}
        self._default_answer = default_answer

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        # Extract question from messages
        if isinstance(prompt, str):
            q = prompt
        else:
            msgs = list(prompt)
            q = msgs[-1]["content"] if msgs else ""

        answer = self._answers.get(q, self._default_answer)

        n = kwargs.get("n", 1)
        if isinstance(answer, list):
            texts = answer[:n] if len(answer) >= n else answer
        else:
            texts = [answer] * n

        return ModelOutput(
            model=self.meta(),
            texts=texts,
            finish_reasons=["stop"] * len(texts),
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            request_params={"model": "mock-chat", "n": n},
        )

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


class MockGenModel(GenModel):
    """GenModel that supports alogprobs without calling any API."""

    def __init__(
        self,
        logprob_scores: dict[str, float] | None = None,
        default_answer: str = "unknown",
        **kwargs,
    ):
        super().__init__(model="mock-gen", api_key="fake", **kwargs)
        self._logprob_scores = logprob_scores or {}
        self._default_answer = default_answer

    async def _agenerate_impl(self, prompt: str, **kwargs) -> ModelOutput:
        return ModelOutput(
            model=self.meta(),
            texts=[self._default_answer],
            finish_reasons=["stop"],
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    async def _alogprobs_impl(self, prompt: str, **kwargs) -> ModelOutput:
        # Extract the last character as the option label
        option_label = prompt.rstrip()[-1] if prompt.strip() else "A"
        score = self._logprob_scores.get(option_label, -10.0)

        return ModelOutput(
            model=self.meta(),
            texts=[""],
            finish_reasons=["stop"],
            logprobs_tokens=[f" {option_label}"],
            logprobs=[score],
            usage={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            request_params={"max_tokens": kwargs.get("max_tokens", 1)},
        )


class MockJudgeModel(ChatModel):
    """ChatModel that acts as a judge, returning configurable verdicts."""

    def __init__(self, verdict: str = "yes", **kwargs):
        super().__init__(model="mock-judge", api_key="fake", **kwargs)
        self._verdict = verdict

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        return ModelOutput(
            model=self.meta(),
            texts=[self._verdict],
            finish_reasons=["stop"],
            usage={"input_tokens": 20, "output_tokens": 1, "total_tokens": 21},
            request_params={"model": "mock-judge"},
        )

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


class MockFailingChatModel(ChatModel):
    """ChatModel that fails for specified number of calls, then succeeds."""

    def __init__(self, fail_count: int = 1, success_answer: str = "42", **kwargs):
        super().__init__(model="mock-failing", api_key="fake", **kwargs)
        self._call_count = 0
        self._fail_count = fail_count
        self._success_answer = success_answer

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise TimeoutError(f"Simulated failure #{self._call_count}")
        return ModelOutput(
            model=self.meta(),
            texts=[self._success_answer],
            finish_reasons=["stop"],
            usage={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        )

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


class MockAlwaysFailModel(ChatModel):
    """ChatModel that always raises an exception."""

    def __init__(self, error: type[Exception] = TimeoutError, **kwargs):
        super().__init__(model="mock-always-fail", api_key="fake", **kwargs)
        self._error = error

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        raise self._error("Always fails")

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


class MockCountingChatModel(MockChatModel):
    """MockChatModel that counts how many times _agenerate_impl is called."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0

    async def _agenerate_impl(self, prompt, **kwargs):
        self.call_count += 1
        return await super()._agenerate_impl(prompt, **kwargs)


class MockSelectiveFailModel(ChatModel):
    """ChatModel that fails on first call for prompts matching fail_samples."""

    def __init__(
        self,
        fail_samples: set[str] | None = None,
        answers: dict[str, str] | None = None,
        default_answer: str = "42",
        **kwargs,
    ):
        super().__init__(model="mock-selective", api_key="fake", **kwargs)
        self._fail_samples = fail_samples or set()
        self._answers = answers or {}
        self._default_answer = default_answer
        self._call_counts: dict[str, int] = {}

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        q = prompt if isinstance(prompt, str) else list(prompt)[-1]["content"]
        self._call_counts[q] = self._call_counts.get(q, 0) + 1

        # Fail on first call if prompt matches any fail pattern
        if self._call_counts[q] <= 1 and q in self._fail_samples:
            raise TimeoutError(f"Simulated first-time failure for: {q}")

        answer = self._answers.get(q, self._default_answer)
        return ModelOutput(
            model=self.meta(),
            texts=[answer],
            finish_reasons=["stop"],
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        raise NotImplementedError


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

    async def _agenerate_impl(self, prompt: Any, **kwargs: Any) -> ModelOutput:
        jitter = random.uniform(-self._latency_jitter, self._latency_jitter)
        await anyio.sleep(max(0, self._latency_s + jitter))
        n = kwargs.get("n", 1)
        texts = [self._output_text] * n
        input_tokens = max(1, len(prompt) // 4) if isinstance(prompt, str) else 10
        output_tokens = max(1, self._output_size)
        return ModelOutput(
            model=self.meta(),
            texts=texts,
            finish_reasons=["stop"] * n,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            request_params={"model": "mock-latency", "n": n},
        )

    async def _alogprobs_impl(self, prompt: Any, **kwargs: Any) -> ModelOutput:
        raise NotImplementedError

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
        import gc as _gc

        _gc.collect()
        _gc.collect()
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
    """Write n_completed FINAL contexts to disk via TaskSaver."""
    saver = TaskSaver(
        root_dir=root,
        shard_samples=shard_samples,
        shard_write_concurrency=8,
        write_buffer_size=max(n_completed + 1, 64),
        write_buffer_flush_interval=9999.0,
    )
    for i in range(n_completed):
        ctx = _make_bench_ctx(i, TaskStage.FINAL)
        saver._update_manifest_entry(ctx)
        saver._stage_queue.append(ctx)
    await saver.flush()
