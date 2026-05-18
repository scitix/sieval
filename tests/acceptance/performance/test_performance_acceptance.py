"""
Benchmark summary / acceptance tests.

Runs 6 realistic scenarios that reflect production usage patterns and
produces a structured summary table + JSON artifact for cross-version
comparison.  Each scenario has a hard acceptance threshold; CI fails if
any threshold is not met.

Scenarios
---------
1. Single-turn eval      — most common MCQ/classification workload
2. Long output generation — code-gen / long-text with 4 KB responses
3. Multi-iteration (x3)  — feedback→iterate loop
4. High concurrency      — saturate concurrency capacity
5. Resume (90 %)         — resume from 90 % pre-written checkpoint
6. Framework overhead    — zero-latency baseline (scheduling + I/O)

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import pytest

from sieval.core.runners.runner import TaskRunner
from tests.conftest import (
    BenchmarkTask,
    LatencyMockChatModel,
    MemoryTracker,
    MockChatModel,
    MultiIterBenchmarkTask,
    PerfTimer,
    make_large_dataset,
    make_perf_config,
    samples_per_second,
    write_completed_samples,
)

# Regression baselines file (co-located with this file)
_BASELINES_PATH = Path(__file__).parent / "baselines.json"
# Tolerance: allow up to 10% degradation from baseline
_REGRESSION_TOLERANCE = 0.90


# ===================================================================
# Scenario definition
# ===================================================================
@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    name: str
    display_name: str
    n_samples: int
    latency_s: float
    output_size: int
    concurrency: int
    iterations: int = 1  # >1 uses MultiIterBenchmarkTask
    min_efficiency: float = 0.0  # acceptance threshold (0 = skip)
    min_sps: float = 0.0  # acceptance threshold (0 = skip)


@dataclass
class BenchmarkResult:
    scenario: BenchmarkScenario
    elapsed_s: float
    sps: float
    efficiency: float  # actual_sps / theoretical_max
    passed: bool
    status_detail: str = ""


# Latency semantics
# -----------------
# ``latency_s`` models the *total wall-clock time* of a single model call
# (TTFT + generation), NOT individual token latencies.  Values are scaled
# down from real LLM latencies to keep CI runtime practical:
#
#   Real-world reference (typical LLM API):
#     TTFT ≈ 200-500 ms, TPOT ≈ 30-50 ms
#     Short answer  (~50 tokens):  ~0.5-1 s total
#     Long output (~4096 tokens):  ~20-160 s total
#
#   Benchmark scaling:
#     0.1 s  ≈ short-answer call  (scaled ~5-10×)
#     0.3 s  ≈ long-output call   (scaled ~50-500×)
#
# The goal is to measure *framework scheduling efficiency* — how well the
# pipeline overlaps concurrent model calls — not actual model speed.
# Efficiency thresholds (60-70%) account for 4-stage pipeline overhead,
# I/O, and scheduling latency.

BENCHMARK_SCENARIOS = [
    BenchmarkScenario(
        name="single_turn",
        display_name="Single-turn eval",
        n_samples=500,
        latency_s=0.1,  # ~short-answer call
        output_size=100,
        concurrency=64,
        min_efficiency=0.70,
    ),
    BenchmarkScenario(
        name="long_output",
        display_name="Long output gen",
        n_samples=200,
        latency_s=0.3,  # ~long-output call (scaled)
        output_size=4096,
        concurrency=64,
        min_efficiency=0.70,
    ),
    BenchmarkScenario(
        name="multi_iteration",
        display_name="Multi-iteration (x3)",
        n_samples=200,
        latency_s=0.1,  # per-iteration call latency
        output_size=100,
        concurrency=64,
        iterations=3,
        min_efficiency=0.60,
    ),
    BenchmarkScenario(
        name="high_concurrency",
        display_name="High concurrency",
        n_samples=500,
        latency_s=0.1,
        output_size=100,
        concurrency=128,
        min_efficiency=0.65,
    ),
    # Scenario 5 (resume) and 6 (framework overhead) handled separately
    BenchmarkScenario(
        name="framework_overhead",
        display_name="Framework overhead",
        n_samples=2000,
        latency_s=0.0,  # zero-latency: pure scheduling + I/O cost
        output_size=100,
        concurrency=128,
        min_sps=500.0,
    ),
]


# ===================================================================
# Scenario runner
# ===================================================================
async def _run_scenario(
    scenario: BenchmarkScenario,
    tmp_path: Path,
) -> BenchmarkResult:
    """Execute one benchmark scenario and return the result."""
    dataset = make_large_dataset(scenario.n_samples, payload_size=scenario.output_size)

    if scenario.latency_s > 0:
        model = LatencyMockChatModel(
            latency_s=scenario.latency_s,
            latency_jitter=0.0,  # deterministic for benchmarks
            output_size=scenario.output_size,
        )
    else:
        model = MockChatModel(default_answer="A0")

    if scenario.iterations > 1:
        task = MultiIterBenchmarkTask(
            dataset=dataset,
            model=model,
            name=scenario.name,
            output_size=scenario.output_size,
            finalize_after=scenario.iterations,
        )
    else:
        task = BenchmarkTask(
            dataset=dataset,
            model=model,
            name=scenario.name,
            output_size=scenario.output_size,
        )

    config = make_perf_config(
        tmp_path / scenario.name,
        concurrency_limit=scenario.concurrency,
        max_iterations=scenario.iterations + 1 if scenario.iterations > 1 else 5,
    )

    runner = TaskRunner(task, config)
    timer = PerfTimer()
    with timer:
        report = await runner.arun()

    assert report is not None, f"Scenario {scenario.name} returned no report"

    sps = samples_per_second(scenario.n_samples, timer.elapsed)

    # Compute efficiency for latency-based scenarios
    if scenario.latency_s > 0:
        # For multi-iteration: each sample does iterations * latency_s total work
        effective_latency = scenario.latency_s * scenario.iterations
        theoretical_max = scenario.concurrency / effective_latency
        efficiency = sps / theoretical_max if theoretical_max > 0 else 0.0
    else:
        efficiency = 0.0

    # Check acceptance criteria
    passed = True
    detail = ""
    if scenario.min_efficiency > 0 and efficiency < scenario.min_efficiency:
        passed = False
        detail = f"efficiency {efficiency:.1%} < {scenario.min_efficiency:.0%}"
    if scenario.min_sps > 0 and sps < scenario.min_sps:
        passed = False
        detail = f"sps {sps:.1f} < {scenario.min_sps:.0f}"

    return BenchmarkResult(
        scenario=scenario,
        elapsed_s=timer.elapsed,
        sps=sps,
        efficiency=efficiency,
        passed=passed,
        status_detail=detail,
    )


# ===================================================================
# Resume scenario (separate logic)
# ===================================================================
async def _run_resume_scenario(tmp_path: Path) -> BenchmarkResult:
    """Run the resume acceptance scenario: 90% pre-written, complete rest."""
    n_total = 1000
    n_completed = 900
    result_dir = tmp_path / "resume_bench"

    # Phase 1: write completed samples
    await write_completed_samples(result_dir, n_completed)

    # Phase 2: resume and complete remaining
    dataset = make_large_dataset(n_total, payload_size=50)
    model = MockChatModel(default_answer="A0")
    task = BenchmarkTask(dataset=dataset, model=model, name="resume_90pct")
    config = make_perf_config(
        tmp_path / "resume_cfg",
        result_dir=str(result_dir),
        auto_resume=True,
        concurrency_limit=64,
    )

    runner = TaskRunner(task, config)
    timer = PerfTimer()
    with timer:
        report = await runner.arun()

    assert report is not None

    scenario = BenchmarkScenario(
        name="resume_90pct",
        display_name="Resume (90%)",
        n_samples=n_total,
        latency_s=0.0,
        output_size=50,
        concurrency=64,
    )

    # Read baseline + tolerance from baselines.json so this pass/fail gate
    # matches _check_regressions() for max_elapsed_s.
    max_elapsed_s = 5.0
    tolerance = _REGRESSION_TOLERANCE
    if _BASELINES_PATH.exists():
        _bl = orjson.loads(_BASELINES_PATH.read_bytes())
        max_elapsed_s = _bl.get("resume_90pct", {}).get("max_elapsed_s", max_elapsed_s)
        raw_tolerance = _bl.get("_tolerance")
        if isinstance(raw_tolerance, int | float) and 0 < raw_tolerance <= 1:
            tolerance = float(raw_tolerance)

    threshold_s = max_elapsed_s / tolerance
    passed = timer.elapsed < threshold_s
    detail = (
        ""
        if passed
        else (
            f"resume took {timer.elapsed:.2f}s > baseline "
            f"{max_elapsed_s:.1f}s / {tolerance:.2f} = {threshold_s:.2f}s"
        )
    )

    return BenchmarkResult(
        scenario=scenario,
        elapsed_s=timer.elapsed,
        sps=0.0,  # not meaningful for resume
        efficiency=0.0,
        passed=passed,
        status_detail=detail,
    )


# ===================================================================
# Summary output
# ===================================================================
def _print_summary_table(
    results: list[BenchmarkResult],
    peak_memory_mb: float,
) -> None:
    """Print a structured ASCII summary table."""
    lines: list[str] = []
    w_name, w_n, w_sps, w_eff, w_status = 23, 9, 11, 12, 13
    sep = f"{'=' * w_name}={'=' * w_n}={'=' * w_sps}={'=' * w_eff}={'=' * w_status}"

    lines.append("")
    lines.append(
        f"{'SiEval Benchmark Summary':^{w_name + w_n + w_sps + w_eff + w_status + 4}}"
    )
    lines.append(sep)
    lines.append(
        f"{'Scenario':<{w_name}} {'Samples':>{w_n}} {'SPS':>{w_sps}} "
        f"{'Efficiency':>{w_eff}} {'Status':>{w_status}}"
    )
    lines.append(sep)

    for r in results:
        name = r.scenario.display_name
        n = str(r.scenario.n_samples)

        if r.scenario.name == "resume_90pct":
            sps_str = "—"
            eff_str = "—"
            status = f"{'PASS' if r.passed else 'FAIL'} {r.elapsed_s:.1f}s"
        elif r.scenario.latency_s == 0:
            sps_str = f"{r.sps:.1f}"
            eff_str = "—"
            status = "PASS" if r.passed else "FAIL"
        else:
            sps_str = f"{r.sps:.1f}"
            eff_str = f"{r.efficiency:.1%}"
            status = "PASS" if r.passed else "FAIL"

        lines.append(
            f"{name:<{w_name}} {n:>{w_n}} {sps_str:>{w_sps}} "
            f"{eff_str:>{w_eff}} {status:>{w_status}}"
        )

    lines.append(sep)
    lines.append(
        f"{'Peak memory (RSS)':<{w_name}} {'':>{w_n}} {'':>{w_sps}} "
        f"{'':>{w_eff}} {peak_memory_mb:>{w_status - 3}.1f} MB"
    )
    lines.append(sep)

    print("\n".join(lines))


def _write_json_artifact(
    results: list[BenchmarkResult],
    peak_memory_mb: float,
    output_dir: Path,
) -> Path:
    """Write benchmark_summary.json for cross-version comparison."""
    data: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "scenarios": {},
        "peak_memory_mb": round(peak_memory_mb, 1),
    }
    for r in results:
        entry: dict[str, Any] = {
            "samples": r.scenario.n_samples,
            "elapsed_s": round(r.elapsed_s, 3),
            "passed": r.passed,
        }
        if r.sps > 0:
            entry["sps"] = round(r.sps, 1)
        if r.efficiency > 0:
            entry["efficiency"] = round(r.efficiency, 4)
        data["scenarios"][r.scenario.name] = entry

    output_path = output_dir / "benchmark_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    return output_path


def _check_regressions(
    results: list[BenchmarkResult],
    tolerance: float | None = None,
) -> list[str]:
    """Compare results against baselines.json; return list of regression messages.

    Returns an empty list if baselines.json does not exist (first run).
    Tolerance of 0.9 means actual must be >= baseline * 0.9 (max 10% degradation).
    If tolerance is None, it is loaded from baselines.json (_tolerance) and
    falls back to _REGRESSION_TOLERANCE.
    """
    if not _BASELINES_PATH.exists():
        return []

    baselines: dict[str, Any] = orjson.loads(_BASELINES_PATH.read_bytes())
    if tolerance is None:
        raw_tolerance = baselines.get("_tolerance")
        if isinstance(raw_tolerance, int | float) and 0 < raw_tolerance <= 1:
            tolerance = float(raw_tolerance)
        else:
            tolerance = _REGRESSION_TOLERANCE
    regressions: list[str] = []

    by_name = {r.scenario.name: r for r in results}
    for name, baseline in baselines.items():
        if name.startswith("_"):
            continue
        result = by_name.get(name)
        if result is None:
            continue

        if "min_sps" in baseline and result.sps > 0:
            threshold = baseline["min_sps"] * tolerance
            if result.sps < threshold:
                regressions.append(
                    f"{name}: sps {result.sps:.1f} < baseline {baseline['min_sps']:.1f}"
                    f" * {tolerance} = {threshold:.1f}"
                )

        if "min_efficiency" in baseline and result.efficiency > 0:
            threshold = baseline["min_efficiency"] * tolerance
            if result.efficiency < threshold:
                regressions.append(
                    f"{name}: efficiency {result.efficiency:.1%} < baseline"
                    f" {baseline['min_efficiency']:.1%} * {tolerance}"
                    f" = {threshold:.1%}"
                )

        if "max_elapsed_s" in baseline and result.elapsed_s > 0:
            # For latency metrics, regression = taking longer (inverse tolerance)
            threshold = baseline["max_elapsed_s"] / tolerance
            if result.elapsed_s > threshold:
                regressions.append(
                    f"{name}: elapsed {result.elapsed_s:.2f}s > baseline"
                    f" {baseline['max_elapsed_s']:.1f}s / {tolerance}"
                    f" = {threshold:.2f}s"
                )

    return regressions


# ===================================================================
# Test class
# ===================================================================
class TestBenchmarkSummary:
    """Acceptance tests: realistic scenarios with structured reporting."""

    @pytest.mark.anyio
    async def test_benchmark_scenarios(self, tmp_path: Path) -> None:
        """Run all benchmark scenarios and produce summary report."""
        results: list[BenchmarkResult] = []
        mem_tracker = MemoryTracker()
        mem_tracker.start()

        # Run throughput/efficiency scenarios
        for scenario in BENCHMARK_SCENARIOS:
            result = await _run_scenario(scenario, tmp_path)
            results.append(result)

        # Run resume scenario
        resume_result = await _run_resume_scenario(tmp_path)
        results.append(resume_result)

        mem_tracker.stop()

        # Print summary table
        _print_summary_table(results, mem_tracker.peak_mb)

        # Write JSON artifact
        artifact_root = Path(
            os.environ.get("SIEVAL_BENCHMARK_ARTIFACT_DIR", str(tmp_path))
        )
        artifact = _write_json_artifact(results, mem_tracker.peak_mb, artifact_root)
        print(f"\nBenchmark artifact: {artifact}")

        # Assert all scenarios passed
        for r in results:
            assert r.passed, (
                f"Scenario '{r.scenario.display_name}' FAILED: {r.status_detail}"
            )

        # Check for performance regressions against baselines
        regressions = _check_regressions(results)
        assert not regressions, (
            "Performance regressions detected "
            "(degradation beyond configured tolerance):\n"
            + "\n".join(f"  - {msg}" for msg in regressions)
        )


class TestRegressionDetection:
    """Unit tests for _check_regressions() logic (no I/O, no baselines.json)."""

    def _make_result(
        self,
        name: str,
        sps: float = 0.0,
        efficiency: float = 0.0,
        elapsed_s: float = 0.0,
    ) -> BenchmarkResult:
        scenario = BenchmarkScenario(
            name=name,
            display_name=name,
            n_samples=100,
            latency_s=0.1,
            output_size=100,
            concurrency=64,
        )
        return BenchmarkResult(
            scenario=scenario,
            elapsed_s=elapsed_s,
            sps=sps,
            efficiency=efficiency,
            passed=True,
        )

    def test_no_regressions_when_baselines_missing(self, tmp_path, monkeypatch):
        """No regressions reported if baselines.json does not exist."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        results = [self._make_result("single_turn", sps=100.0, efficiency=0.5)]
        monkeypatch.setattr(mod, "_BASELINES_PATH", tmp_path / "nonexistent.json")
        assert _check_regressions(results) == []

    def test_sps_regression_detected(self, tmp_path, monkeypatch):
        """sps below baseline * tolerance triggers regression."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {"single_turn": {"min_sps": 400.0}}
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # sps = 300, baseline = 400, threshold = 400 * 0.9 = 360 → regression
        result = self._make_result("single_turn", sps=300.0)
        regressions = _check_regressions([result])

        assert len(regressions) == 1
        assert "single_turn" in regressions[0]
        assert "sps" in regressions[0]

    def test_sps_within_tolerance_no_regression(self, tmp_path, monkeypatch):
        """sps at exactly baseline * tolerance does not trigger regression."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {"single_turn": {"min_sps": 400.0}}
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # sps = 360 = 400 * 0.9, right at the threshold (not below)
        result = self._make_result("single_turn", sps=360.0)
        assert _check_regressions([result]) == []

    def test_efficiency_regression_detected(self, tmp_path, monkeypatch):
        """efficiency below baseline * tolerance triggers regression."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {"single_turn": {"min_efficiency": 0.70}}
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # efficiency = 0.50, threshold = 0.70 * 0.9 = 0.63 → regression
        result = self._make_result("single_turn", efficiency=0.50)
        regressions = _check_regressions([result])

        assert len(regressions) == 1
        assert "efficiency" in regressions[0]

    def test_elapsed_regression_detected(self, tmp_path, monkeypatch):
        """elapsed_s above baseline / tolerance triggers regression."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {"resume_90pct": {"max_elapsed_s": 3.0}}
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # elapsed = 4.0, threshold = 3.0 / 0.9 = 3.33 → regression
        result = self._make_result("resume_90pct", elapsed_s=4.0)
        regressions = _check_regressions([result])

        assert len(regressions) == 1
        assert "elapsed" in regressions[0]

    def test_unknown_scenario_ignored(self, tmp_path, monkeypatch):
        """Baseline entry for a scenario not in results is silently ignored."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {"unknown_scenario": {"min_sps": 999.0}}
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        result = self._make_result("single_turn", sps=500.0)
        assert _check_regressions([result]) == []

    def test_comment_keys_ignored(self, tmp_path, monkeypatch):
        """Keys starting with '_' (comments) are skipped."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {
            "_comment": "this is a comment",
            "_tolerance": 0.9,
            "single_turn": {"min_sps": 400.0},
        }
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # Above threshold → no regression; comment keys must not cause errors
        result = self._make_result("single_turn", sps=450.0)
        assert _check_regressions([result]) == []

    def test_tolerance_loaded_from_baseline_file(self, tmp_path, monkeypatch):
        """If _tolerance exists in baselines, _check_regressions should use it."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {
            "_tolerance": 0.8,
            "single_turn": {"min_sps": 400.0},
        }
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # 330 >= 400 * 0.8 (320), so no regression if file tolerance is used.
        result = self._make_result("single_turn", sps=330.0)
        assert _check_regressions([result]) == []

    def test_explicit_tolerance_overrides_file_tolerance(self, tmp_path, monkeypatch):
        """Explicit tolerance argument should override _tolerance in file."""
        import tests.acceptance.performance.test_performance_acceptance as mod

        baseline_data = {
            "_tolerance": 0.8,
            "single_turn": {"min_sps": 400.0},
        }
        bf = tmp_path / "baselines.json"
        bf.write_bytes(orjson.dumps(baseline_data))
        monkeypatch.setattr(mod, "_BASELINES_PATH", bf)

        # 330 < 400 * 0.95 (380), so explicit tolerance should trigger regression.
        result = self._make_result("single_turn", sps=330.0)
        regressions = _check_regressions([result], tolerance=0.95)
        assert len(regressions) == 1
        assert "sps" in regressions[0]
