import asyncio
import contextlib
from dataclasses import dataclass

import psutil


@dataclass
class ResourceStats:
    cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0


async def monitor_process_resources(
    pid: int, interval: float = 0.1
) -> tuple[ResourceStats, asyncio.Event]:
    stats = ResourceStats()
    stop_event = asyncio.Event()

    cpu_samples = []
    memory_samples = []

    async def _monitor():
        # If we can't monitor, just return zeros
        with contextlib.suppress(Exception):
            process = psutil.Process(pid)
            # Initial CPU measurement (first call returns 0.0)
            process.cpu_percent()

            while not stop_event.is_set():
                try:
                    # Get CPU percentage (averaged over interval)
                    cpu = process.cpu_percent()
                    if cpu > 0:  # Skip initial 0 value
                        cpu_samples.append(cpu)
                        # Track peak CPU
                        if cpu > stats.peak_cpu_percent:
                            stats.peak_cpu_percent = cpu

                    # Get memory usage in MB
                    mem_info = process.memory_info()
                    memory_mb = mem_info.rss / (1024 * 1024)
                    memory_samples.append(memory_mb)

                    # Track peak memory
                    if memory_mb > stats.peak_memory_mb:
                        stats.peak_memory_mb = memory_mb

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process ended or we lost access
                    break

                await asyncio.sleep(interval)

            # Calculate averages
            if cpu_samples:
                stats.cpu_percent = sum(cpu_samples) / len(cpu_samples)
            if memory_samples:
                stats.memory_mb = sum(memory_samples) / len(memory_samples)

    # Start monitoring task
    asyncio.create_task(_monitor())

    return stats, stop_event
