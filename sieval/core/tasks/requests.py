"""A task's ``requests.jsonl`` and its per-run summary.

:class:`TaskRequestLog` is the runner's sink for the lines
:mod:`sieval.core.models.request_log` produces. Lines are appended off the
event loop; a resume appends. Nothing reads the file back, so deleting it
changes no result. The summary is folded in as lines arrive, so rotation loses
nothing from it.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict

import anyio
import anyio.to_thread

from sieval.core.models import Model
from sieval.core.types import JSONValue

from .profiler import ProfileTimingStats, compute_timing_stats
from .run_meta import LimiterPeak, RequestCounts, RunRequests, call_key

REQUESTS_FILE = "requests.jsonl"

#: Lines held before a flush is due, and the longest a line waits for one.
FLUSH_LINES = 256
FLUSH_INTERVAL_S = 2.0


class RequestSummary(TypedDict):
    """One phase x model's requests in a run."""

    calls: int
    attempts: int
    retried_calls: int
    max_attempts: int
    outcomes: dict[str, int]
    retry_wait_s_total: float
    in_flight: dict[str, LimiterPeak]
    latency: NotRequired[ProfileTimingStats]
    queue: NotRequired[ProfileTimingStats]
    connect: NotRequired[ProfileTimingStats]


class RequestProfile(TypedDict):
    """``profile.json``'s ``requests``: one run's requests, by phase and model."""

    run: str | None
    by_phase: dict[str, dict[str, RequestSummary]]


@dataclass
class _Totals:
    calls: int = 0
    attempts: int = 0
    retried_calls: int = 0
    max_attempts: int = 0
    outcomes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    retry_wait_s_total: float = 0.0
    in_flight: dict[str, LimiterPeak] = field(default_factory=dict)
    latency: list[float] = field(default_factory=list)
    queue: list[float] = field(default_factory=list)
    connect: list[float] = field(default_factory=list)

    def add(self, line: Mapping[str, JSONValue]) -> None:
        self.attempts += 1
        attempt = line.get("attempt")
        attempts = line.get("attempts")
        if attempt == 1:
            self.calls += 1
            if isinstance(attempts, int):
                if attempts > 1:
                    self.retried_calls += 1
                self.max_attempts = max(self.max_attempts, attempts)
        outcome = line.get("outcome")
        self.outcomes[outcome if isinstance(outcome, str) else "other"] += 1
        wait = line.get("retry_after_s")
        if isinstance(wait, float | int):
            self.retry_wait_s_total += wait
        latency = line.get("latency_s")
        if isinstance(latency, float | int):
            self.latency.append(float(latency))
        queue = line.get("queue_s")
        if isinstance(queue, float | int):
            self.queue.append(float(queue))
        connect = line.get("connect_s")
        if isinstance(connect, float | int):
            self.connect.append(float(connect))
        in_flight = line.get("in_flight")
        if isinstance(in_flight, dict):
            for name, snapshot in in_flight.items():
                if not isinstance(snapshot, dict):
                    continue
                borrowed = snapshot.get("borrowed")
                limit = snapshot.get("total")
                if not isinstance(borrowed, int):
                    continue
                current = self.in_flight.get(name)
                if current is None or borrowed > current["peak"]:
                    self.in_flight[name] = {
                        "peak": borrowed,
                        "limit": limit if isinstance(limit, int) else None,
                    }

    def summary(self) -> RequestSummary:
        result = RequestSummary(
            calls=self.calls,
            attempts=self.attempts,
            retried_calls=self.retried_calls,
            max_attempts=self.max_attempts,
            outcomes=dict(sorted(self.outcomes.items())),
            retry_wait_s_total=round(self.retry_wait_s_total, 6),
            in_flight=dict(sorted(self.in_flight.items())),
        )
        if self.latency:
            result["latency"] = compute_timing_stats(self.latency)
        if self.queue:
            result["queue"] = compute_timing_stats(self.queue)
        if self.connect:
            result["connect"] = compute_timing_stats(self.connect)
        return result


class TaskRequestLog:
    """The runner's :class:`~sieval.core.models.request_log.RequestSink`.

    At *max_bytes* the file rotates to ``requests.<n>.jsonl``; the newest
    *max_files* rotated files are kept.
    """

    def __init__(
        self,
        root_dir: Path,
        *,
        subject: Model | None,
        judges: Mapping[str, Model],
        max_bytes: int,
        max_files: int,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("request_log_max_bytes must be >= 1")
        if max_files < 0:
            raise ValueError("request_log_max_files must be >= 0")
        self._path = root_dir / REQUESTS_FILE
        self._max_bytes = max_bytes
        self._max_files = max_files
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()
        self._lock = anyio.Lock()
        self._totals: dict[tuple[str, str], _Totals] = defaultdict(_Totals)
        self._run: str | None = None
        # By call key, then by name: `with_args` changes a model's call key.
        self._roles: dict[str, str] = {}
        self._roles_by_name: dict[str, str] = {}
        if subject is not None:
            meta = subject.meta()
            self._roles[call_key(meta)] = "subject"
            self._roles_by_name[str(meta.get("model"))] = "subject"
        for role, judge in sorted(judges.items()):
            meta = judge.meta()
            self._roles.setdefault(call_key(meta), role)
            self._roles_by_name.setdefault(str(meta.get("model")), role)
        self._resolved: dict[str, str] = {}

    # ── RequestSink ──────────────────────────────────────────────────────────

    def role_of(self, model_meta: Mapping[str, object]) -> str:
        key = call_key(model_meta)
        role = self._resolved.get(key)
        if role is None:
            role = self._roles.get(key) or self._roles_by_name.get(
                str(model_meta.get("model")), "unknown"
            )
            self._resolved[key] = role
        return role

    def record(self, line: dict[str, JSONValue]) -> None:
        run = line.get("run")
        if isinstance(run, str):
            self._run = run
        model = line.get("model")
        model_id = model.get("id") if isinstance(model, dict) else None
        self._totals[(str(line.get("phase")), str(model_id))].add(line)
        self._buffer.append(json.dumps(line, ensure_ascii=False, sort_keys=True))

    # ── Persistence ─────────────────────────────────────────────────────────

    def flush_due(self) -> bool:
        return bool(self._buffer) and (
            len(self._buffer) >= FLUSH_LINES
            or time.monotonic() - self._last_flush >= FLUSH_INTERVAL_S
        )

    async def maybe_flush(self) -> None:
        """Flush when due, unless a flush is already running."""

        if self.flush_due() and not self._lock.locked():
            await self.flush()

    async def flush(self) -> None:
        """Append every buffered line in a worker thread.

        On a write error the lines are dropped; the summary still counts them.
        """

        async with self._lock:
            self._last_flush = time.monotonic()
            if not self._buffer:
                return
            lines, self._buffer = self._buffer, []
            await anyio.to_thread.run_sync(self._append, lines)

    def _append(self, lines: list[str]) -> None:
        data = "".join(f"{line}\n" for line in lines).encode()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as fh:
            fh.write(data)
            size = fh.tell()
        if size >= self._max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        stem, suffix = self._path.stem, self._path.suffix
        numbers = sorted(
            int(p.name[len(stem) + 1 : -len(suffix)])
            for p in self._path.parent.glob(f"{stem}.*{suffix}")
            if p.name[len(stem) + 1 : -len(suffix)].isdigit()
        )
        if self._max_files == 0:
            self._path.unlink()
            return
        following = (numbers[-1] + 1) if numbers else 1
        os.replace(self._path, self._path.with_name(f"{stem}.{following}{suffix}"))
        numbers.append(following)
        for stale in numbers[: -self._max_files]:
            self._path.with_name(f"{stem}.{stale}{suffix}").unlink(missing_ok=True)

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self) -> RequestProfile | None:
        """This run's requests by phase and model; ``None`` if it made none."""

        if not self._totals:
            return None
        by_phase: dict[str, dict[str, RequestSummary]] = {}
        for (phase, model), totals in sorted(self._totals.items()):
            by_phase.setdefault(phase, {})[model] = totals.summary()
        return RequestProfile(run=self._run, by_phase=by_phase)

    def counts(self) -> RunRequests | None:
        """:meth:`summary`'s counts and peaks, for ``report.json`` and ``runs[]``."""

        if not self._totals:
            return None
        by_phase: RunRequests = {}
        for (phase, model), totals in sorted(self._totals.items()):
            full = totals.summary()
            by_phase.setdefault(phase, {})[model] = RequestCounts(
                calls=full["calls"],
                attempts=full["attempts"],
                outcomes=full["outcomes"],
                in_flight=full["in_flight"],
            )
        return by_phase
