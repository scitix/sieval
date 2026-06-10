"""RULER effective-length aggregation over a multi-length sweep.

RULER reports, per model, the 13-task average at each context length and an
"effective length": the longest length whose average still clears a fixed
threshold. The paper sets that threshold to Llama2-7b's score at 4K (85.6 in the
official table) — a *relative* bar, because absolute scores drift across harnesses
(tokenizer, chat template, sentence splitting). Prefer recomputing it from your
own Llama2-7b @ 4K run rather than hardcoding 85.6.

This module is pure aggregation over the :class:`RunInfo` objects produced by the
leaderboard scanner: it groups task ``score`` fields by the ``_<len>`` suffix that
``scripts/gen_ruler_sweep.py`` puts on every task name (e.g.
``ruler_qa_squad_128k``) and computes per-length averages + the effective length.
The CLI command in :mod:`sieval.cli.leaderboard.commands` wires scanning + output
around it.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import re
from collections import defaultdict

from .scanner import RunInfo

# Default threshold from the RULER paper: Llama2-7b @ 4K. Relative by design —
# pass a reference run to recompute it for your harness instead of trusting this.
DEFAULT_THRESHOLD = 85.6
RULER_TASKS_PER_LENGTH = 13

# Trailing length tag that gen_ruler_sweep.py appends to every task name.
_LEN_SUFFIX = re.compile(r"_(\d+)(k?)$", re.IGNORECASE)


def parse_length(task_name: str) -> int | None:
    """``'ruler_qa_squad_128k'`` → 131072; ``'ruler_vt_4096'`` → 4096; else ``None``."""
    m = _LEN_SUFFIX.search(task_name)
    if not m:
        return None
    n = int(m.group(1))
    return n * 1024 if m.group(2) else n


def len_tag(length: int) -> str:
    """4096 → ``'4k'``; 131072 → ``'128k'``; non-multiples stay raw."""
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


def collect_sweep(runs: list[RunInfo]) -> dict[str, dict[int, list[float]]]:
    """Group run scores into ``{model: {length: [task scores]}}``.

    Runs whose task name has no length suffix, or whose report has no numeric
    ``score``, are skipped.
    """
    by_model: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        length = parse_length(run.task_name)
        score = run.report.get("score")
        if length is None or not isinstance(score, int | float):
            continue
        by_model[run.model_name][length].append(float(score))
    return by_model


def effective_length(per_length_avg: dict[int, float], threshold: float) -> int | None:
    """Longest length whose average clears *threshold* (max passing, not contiguous)."""
    passing = [length for length, avg in per_length_avg.items() if avg >= threshold]
    return max(passing) if passing else None


def reference_threshold(
    ref_by_model: dict[str, dict[int, list[float]]],
) -> tuple[float, int] | None:
    """Return ``(threshold, base_length)`` from a reference sweep's smallest tier.

    The threshold is the average over the smallest evaluated length across all
    models in the reference run (RULER uses Llama2-7b @ 4K). ``None`` if empty.
    """
    avgs: dict[int, list[float]] = defaultdict(list)
    for lengths in ref_by_model.values():
        for length, scores in lengths.items():
            avgs[length].extend(scores)
    if not avgs:
        return None
    base = min(avgs)
    return sum(avgs[base]) / len(avgs[base]), base


def summarize(
    by_model: dict[str, dict[int, list[float]]], threshold: float
) -> dict[str, dict]:
    """Build the JSON-serializable per-model summary consumed by the renderer."""
    out: dict[str, dict] = {}
    for model in sorted(by_model):
        lengths = by_model[model]
        per_length_avg = {
            length: sum(scores) / len(scores) for length, scores in lengths.items()
        }
        eff = effective_length(per_length_avg, threshold)
        rows = [
            {
                "length": length,
                "tag": len_tag(length),
                "avg": per_length_avg[length],
                "n_tasks": len(lengths[length]),
                "complete": len(lengths[length]) == RULER_TASKS_PER_LENGTH,
                "pass": per_length_avg[length] >= threshold,
            }
            for length in sorted(per_length_avg)
        ]
        out[model or "(unnamed)"] = {
            "per_length": rows,
            "avg_all": sum(per_length_avg.values()) / len(per_length_avg),
            "effective_length": eff,
            "effective_length_tag": len_tag(eff) if eff is not None else None,
        }
    return out
