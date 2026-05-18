"""
Leaderboard scan, model resolution, and matrix assembly.

Provides utilities to discover evaluation runs from disk, resolve model names
from inference output, and assemble a structured leaderboard matrix.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import yaml
from loguru import logger

from sieval.cli.leaderboard.annotation import CellAnnotationDict, annotate_cell
from sieval.cli.leaderboard.card import AlignmentCard, load_card

_TIMESTAMP_RE = re.compile(r"\d{14}")


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Metadata for a single evaluation run discovered on disk."""

    task_name: str
    run_id: str
    run_dir: Path
    report: dict
    model_name: str = ""


class LeaderboardResult(TypedDict):
    model: str
    task: str
    run_id: str
    report: dict
    annotation: CellAnnotationDict | None


class LeaderboardMatrix(TypedDict):
    models: list[str]
    tasks: list[str]
    results: list[LeaderboardResult]


def scan_runs(dirs: list[Path]) -> list[RunInfo]:
    """Recursively find ``report.json`` files and return :class:`RunInfo` list.

    Two directory patterns are recognised:

    * **Pattern A** -- ``{root}/{task_name}/{14-digit-timestamp}/report.json``
    * **Pattern B** -- ``{root}/{...}/{task_name}/report.json`` (no timestamp)
    """
    runs: list[RunInfo] = []
    for root in dirs:
        for report_path in root.rglob("report.json"):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping malformed report: {} ({})", report_path, exc)
                continue

            parent = report_path.parent
            if _TIMESTAMP_RE.fullmatch(parent.name):
                # Pattern A: grandparent is task name, parent is timestamp
                task_name = parent.parent.name
                run_id = parent.name
            else:
                # Pattern B: parent is task name
                task_name = parent.name
                run_id = parent.name

            runs.append(
                RunInfo(
                    task_name=task_name,
                    run_id=run_id,
                    run_dir=parent,
                    report=report,
                )
            )
    return runs


def resolve_model_name(run_dir: Path) -> str:
    """Extract model name from inference output, falling back to dir name.

    Looks for ``{run_dir}/{iteration}/final/0.jsonl``, reads the first line,
    and extracts ``infer_result.model.model``.
    """
    fallback = run_dir.name
    candidates = sorted(run_dir.glob("*/final/0.jsonl"))
    if not candidates:
        return fallback

    jsonl_path = candidates[0]
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            return fallback
        record = json.loads(first_line)
        return record["infer_result"]["model"]["model"]
    except (json.JSONDecodeError, KeyError, OSError):
        return fallback


def _find_effective_config(run_dir: Path) -> Path | None:
    """Walk up from ``run_dir`` looking for ``effective_config.yaml``."""
    candidate = run_dir
    while True:
        cfg = candidate / "effective_config.yaml"
        if cfg.is_file():
            return cfg
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def _load_card_from_config(cfg: Path) -> AlignmentCard | None:
    """Parse ``effective_config.yaml`` and load the alignment card it cites.

    Returns ``None`` on any miss (no alignment block, card file missing,
    YAML parse error). Failures log at debug level only — they must never
    break the scan.
    """
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        logger.debug("skipping effective_config at {} ({})", cfg, exc)
        return None

    if not isinstance(data, dict):
        return None
    alignment_block = data.get("alignment")
    if not isinstance(alignment_block, dict):
        return None
    card_rel = alignment_block.get("card")
    if not isinstance(card_rel, str) or not card_rel:
        return None

    card_path = (cfg.parent / card_rel).resolve()
    try:
        return load_card(card_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        logger.debug("skipping alignment card at {} ({})", card_path, exc)
        return None


class _CardResolver:
    """Cache alignment-card lookups across a single :func:`build_matrix` call.

    Two levels of cache: the walk-up from a run directory to its nearest
    ``effective_config.yaml`` (bounded, cheap), and the parse+load of that
    config plus the card it cites (expensive). A 10×50 board shares one
    config and one card, so this collapses 500 parses to 1.
    """

    def __init__(self) -> None:
        self._cfg_cache: dict[Path, Path | None] = {}
        self._card_cache: dict[Path, AlignmentCard | None] = {}

    def card_for(self, run_dir: Path) -> AlignmentCard | None:
        if run_dir not in self._cfg_cache:
            self._cfg_cache[run_dir] = _find_effective_config(run_dir)
        cfg = self._cfg_cache[run_dir]
        if cfg is None:
            return None
        if cfg not in self._card_cache:
            self._card_cache[cfg] = _load_card_from_config(cfg)
        return self._card_cache[cfg]


def _compute_annotation(
    run: RunInfo, card: AlignmentCard | None
) -> CellAnnotationDict | None:
    """Compute the per-cell annotation dict if all three keys line up.

    Returns ``None`` when:

    * the run has no alignment card, or
    * the report has no ``score`` field (or it is non-numeric), or
    * the card has no entry for ``(model_name, task_name)``.

    Score extraction uses ``report["score"]`` — the same key the renderer
    in ``sieval.cli.output`` consumes for cell formatting.
    """
    if card is None:
        return None

    model_refs = card.reference_scores.get(run.model_name)
    if model_refs is None:
        return None
    reference = model_refs.get(run.task_name)
    if reference is None:
        return None

    observed = run.report.get("score")
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return None

    return annotate_cell(
        observed=float(observed),
        reference=float(reference),
        tolerance=card.tolerance,
    ).to_dict()


def build_matrix(runs: list[RunInfo], *, all_runs: bool = False) -> LeaderboardMatrix:
    """Aggregate runs into a :class:`LeaderboardMatrix`.

    When *all_runs* is ``False`` (default), only the latest run per
    ``(model, task)`` pair is kept (lexicographic comparison on *run_id*).
    """
    if not runs:
        return LeaderboardMatrix(models=[], tasks=[], results=[])

    resolver = _CardResolver()

    def _result(r: RunInfo) -> LeaderboardResult:
        return LeaderboardResult(
            model=r.model_name,
            task=r.task_name,
            run_id=r.run_id,
            report=r.report,
            annotation=_compute_annotation(r, resolver.card_for(r.run_dir)),
        )

    if all_runs:
        results: list[LeaderboardResult] = [_result(r) for r in runs]
    else:
        # Dedup: keep latest run_id per (model, task)
        best: dict[tuple[str, str], RunInfo] = {}
        for r in runs:
            key = (r.model_name, r.task_name)
            if key not in best or r.run_id > best[key].run_id:
                best[key] = r
        results = [_result(r) for r in best.values()]

    models = sorted({r["model"] for r in results})
    tasks = sorted({r["task"] for r in results})

    return LeaderboardMatrix(models=models, tasks=tasks, results=results)
