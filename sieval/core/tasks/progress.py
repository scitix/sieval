"""Task progress reporting: tqdm TTY bar, non-TTY loguru log, and JSON dump."""

import contextlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson
from loguru import logger
from tqdm import tqdm


class TaskProgress:
    """Dual-mode progress reporter for task execution.

    * **TTY mode** — renders a ``tqdm`` progress bar on stderr.
    * **Non-TTY mode** — emits periodic loguru messages controlled by time
      and percentage thresholds.

    Optionally writes a ``progress.json`` file for external monitoring tools.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        position: int = 0,
        show_progress: bool = True,
        log_interval: float = 15.0,  # 15s
        log_pct_interval: float = 10.0,  # 10%
        root_dir: Path | None = None,
        dump_progress: bool = False,
        dump_interval: float = 1.0,  # 1s
    ):
        self._total = total
        self._desc = desc
        self._is_tty = sys.stderr.isatty()

        # TTY progress bar
        self._pbar: tqdm[Any] | None = None
        if show_progress and self._is_tty and self._total > 0:
            self._pbar = tqdm(
                total=total,
                desc=desc,
                unit="sample",
                leave=True,
                position=position,
                ncols=120,
                mininterval=0.1,
                dynamic_ncols=True,
                ascii=True,
            )

        # Non-TTY logging fallback
        self._enable_log = show_progress and (not self._is_tty)
        self._last_log_time: float = 0.0
        self._last_log_pct: float = 0.0
        self._log_interval = log_interval
        self._log_pct_interval = log_pct_interval

        # Periodic state dump
        self._progress_file = (
            (root_dir / "progress.json") if (root_dir and dump_progress) else None
        )
        self._last_dump_time: float = 0.0
        self._dump_interval: float = dump_interval

        # Internal state
        self._completed_ids: set[str | int] = set()
        self._failed_ids: set[str | int] = set()
        self._anomaly_ids: set[str | int] = set()
        self._anomaly_details: dict[str, int] = defaultdict(int)
        self._active_counts: dict[str, int] = {}

    def init_state(
        self,
        completed_ids: list[str | int],
        failed_ids: list[str | int] | None = None,
        anomaly_ids: list[str | int] | None = None,
        anomaly_details: dict[str, int] | None = None,
    ) -> None:
        """Set initial completed/failed/anomaly counts (e.g. from a resumed run)."""
        for sid in completed_ids:
            self._completed_ids.add(sid)

        if failed_ids:
            for sid in failed_ids:
                self._failed_ids.add(sid)

        if anomaly_ids:
            for sid in anomaly_ids:
                self._anomaly_ids.add(sid)

        if anomaly_details:
            self._anomaly_details.update(anomaly_details)

        count = len(self._completed_ids)
        if self._pbar:
            self._pbar.update(count)
            self._update_pbar_postfix()

        # Log initial start (Only if logging is enabled)
        if self._enable_log and self._total > 0:
            pct = (count / self._total) * 100
            self._last_log_pct = pct
            self._last_log_time = time.perf_counter()
            logger.info(
                "{} started: {}/{} ({:.1f}%)",
                self._desc,
                count,
                self._total,
                pct,
            )

        self._dump_state(force=True)

    def update(
        self,
        sample_id: str | int,
        current_hydrated_count: int,
        failed: bool = False,
        anomalies: set[str] | None = None,
    ) -> None:
        """Record a newly completed sample and refresh all progress outputs."""
        if sample_id in self._completed_ids:
            return

        self._completed_ids.add(sample_id)
        if failed:
            self._failed_ids.add(sample_id)
        if anomalies:
            self._anomaly_ids.add(sample_id)
            for anomaly_type in anomalies:
                self._anomaly_details[anomaly_type] += 1

        # Update visual progress bar (if enabled)
        if self._pbar:
            self._pbar.update(1)
            self._update_pbar_postfix()

        # Handle Non-TTY logging logic
        self.tick(current_hydrated_count)

        # Try to dump state
        self._dump_state()

    def tick(self, current_hydrated_count: int) -> None:
        """Handle periodic non-TTY logging based on time and percentage thresholds."""
        # If logging is disabled (either by config or TTY), return immediately
        if not self._enable_log or self._total <= 0:
            return

        now = time.perf_counter()
        current_completed = len(self._completed_ids)
        current_pct = (current_completed / self._total) * 100

        # Calculate active tasks: Hydrated (Started) - Completed
        active_count = current_hydrated_count - current_completed

        is_finished = current_completed == self._total
        pct_delta = current_pct - self._last_log_pct
        time_delta = now - self._last_log_time
        if (
            is_finished
            or pct_delta >= self._log_pct_interval
            or time_delta >= self._log_interval
        ):
            self._last_log_pct = current_pct
            self._last_log_time = now

            status = "finished" if is_finished else "progress"
            active_str = f", active={active_count}" if not is_finished else ""

            # Format error stats
            failed_count = len(self._failed_ids)
            anomaly_count = len(self._anomaly_ids)
            stats_str = ""
            if failed_count > 0:
                stats_str += f", failed={failed_count}"
            if anomaly_count > 0:
                stats_str += f", anomalies={anomaly_count}"

            logger.info(
                "{} {}: {}/{} ({:.1f}%{}{})",
                self._desc,
                status,
                current_completed,
                self._total,
                current_pct,
                active_str,
                stats_str,
            )

    def set_status(
        self, active_counts: dict[str, int], current_hydrated_count: int
    ) -> None:
        """Update active per-stage counts and trigger non-TTY heartbeat."""
        self._active_counts = active_counts
        self._update_pbar_postfix()
        self._dump_state()
        self.tick(current_hydrated_count)

    def _update_pbar_postfix(self) -> None:
        if not self._pbar:
            return

        postfix = {}
        failed_count = len(self._failed_ids)
        if failed_count > 0:
            postfix["failed"] = failed_count
        anomaly_count = len(self._anomaly_ids)
        if anomaly_count > 0:
            postfix["anomalies"] = anomaly_count
        for k, v in self._active_counts.items():
            if v > 0:
                postfix[k] = v

        if postfix:
            self._pbar.set_postfix(postfix, refresh=False)
        else:
            if hasattr(self._pbar, "postfix") and self._pbar.postfix:
                self._pbar.set_postfix({}, refresh=False)

    def _dump_state(self, force: bool = False) -> None:
        if not self._progress_file:
            return

        now = time.perf_counter()
        if not force and (now - self._last_dump_time < self._dump_interval):
            return

        self._last_dump_time = now

        completed = len(self._completed_ids)
        pct = (completed / self._total * 100) if self._total > 0 else 0.0

        state = {
            "desc": self._desc,
            "total": self._total,
            "completed": completed,
            "failed": len(self._failed_ids),
            "anomalies": len(self._anomaly_ids),
            "anomaly_details": dict(self._anomaly_details),
            "percent": round(pct, 2),
            "active_details": self._active_counts,
            "timestamp": time.time(),
            "finished": completed == self._total,
        }

        temp_file = self._progress_file.with_suffix(".tmp")
        try:
            with open(temp_file, "wb") as f:
                f.write(orjson.dumps(state))
            temp_file.replace(self._progress_file)
        except Exception as e:
            with contextlib.suppress(OSError):
                temp_file.unlink(missing_ok=True)
            logger.warning("Failed to write progress file: {}", e)

    def close(self) -> None:
        """Close the tqdm bar (if any) and force a final progress state dump."""
        if self._pbar:
            self._pbar.close()
        self._dump_state(force=True)
