"""
CLI output layer — unified CommandResult + format-aware rendering.

All CLI commands produce a CommandResult. The render() function is the
single output path: text, JSON, or YAML.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import yaml
from loguru import logger

from sieval.cli._readiness import MISSING_KINDS
from sieval.core.runners import ResultDirExistsError
from sieval.core.utils.logging import log_user

if TYPE_CHECKING:
    # Forward-declared to break the output.py ↔ scanner.py import cycle.
    # scanner.py imports annotate_cell + CellAnnotationDict; leaderboard/
    # __init__.py re-exports leaderboard_app from commands.py, which
    # imports CommandResult from this module.
    from sieval.cli.leaderboard.scanner import LeaderboardResult


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Unified result from any CLI command."""

    command: str
    ok: bool
    data: dict | list | None = None
    error: str | None = None
    warnings: list[str] | None = None


# ── error translation ─────────────────────────────────────────────────


def cli_error_message(exc: Exception) -> str:
    """Translate API-framed core exceptions into CLI-flag vocabulary."""
    if isinstance(exc, ResultDirExistsError):
        return (
            f"Result directory '{exc.path}' already exists and contains data.\n"
            "To continue from it, pass --resume.\n"
            "To start fresh, pass --result-dir <path> "
            "or delete the existing directory."
        )
    return str(exc)


# ── render ────────────────────────────────────────────────────────────


def render(result: CommandResult, fmt: OutputFormat) -> None:
    """Single output path for all commands and all formats."""
    if fmt == OutputFormat.TEXT:
        renderer = _TEXT_RENDERERS.get(result.command)
        if renderer is not None:
            renderer(result)
        else:
            # Fallback: dump data as-is for commands without a text renderer
            if result.ok and result.data is not None:
                log_user("{}", result.data)
            elif not result.ok and result.error:
                logger.error("{}", result.error)
        return

    payload: dict[str, object] = {"command": result.command, "ok": result.ok}
    if result.data is not None:
        payload["data"] = result.data
    if result.error is not None:
        payload["error"] = result.error
    if result.warnings:
        payload["warnings"] = result.warnings

    if fmt == OutputFormat.JSON:
        print(json.dumps(payload, indent=2, default=str))
    elif fmt == OutputFormat.YAML:
        print(yaml.dump(payload, default_flow_style=False, sort_keys=False), end="")


# ── text renderers ────────────────────────────────────────────────────


def _render_text_infer_list(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    data = result.data
    if not data:
        log_user("No inference services found.")
        return
    if not isinstance(data, list):
        logger.error("expected list data, got {}", type(data).__name__)
        return
    for row in data:
        log_user(
            "  {:<30s} {:<10s} {}",
            row["model"],
            row["status"],
            row.get("endpoint") or "-",
        )


def _render_text_infer_show(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    d = result.data
    log_user("Model:    {}", d["model"])
    log_user("Status:   {}", d["status"])
    log_user("Backend:  {}", d["backend"])
    log_user("Endpoint: {}", d["endpoint"])
    log_user("PID:      {}", d["handle_id"])
    metadata = d.get("metadata")
    if metadata:
        for key, value in metadata.items():
            log_user("  {}: {}", key, value)
    env = d.get("env")
    if env:
        log_user("")
        _render_env_block(env)


def _render_env_block(env: dict) -> None:
    """Shared helper for rendering env info in text mode."""
    log_user("Framework:      {}", env.get("framework") or "unknown")
    image = env.get("image")
    if image:
        log_user("Image:          {}", image)
    log_user("CUDA Version:   {}", env.get("cuda_version") or "N/A")
    log_user("Driver Version: {}", env.get("driver_version") or "N/A")
    log_user("GPU Model:      {}", env.get("gpu_model") or "N/A")
    log_user("GPU Count:      {}", env.get("gpu_count", 0))
    gpu_topo = env.get("gpu_topo")
    if gpu_topo:
        log_user("GPU Topology:   {}", gpu_topo)
    log_user("Python Version: {}", env.get("python_version") or "N/A")
    extra = env.get("extra")
    if extra:
        for key, value in extra.items():
            log_user("  {}: {}", key, value)


def _render_text_infer_stop(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    d = result.data
    if d["stopped"]:
        log_user("Stopped inference service: {}", d["model"])
    else:
        logger.error(
            "Process {} may still be running (status: {}). Handle preserved.",
            d.get("handle_id"),
            d.get("phase"),
        )


def _render_text_infer_start(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    d = result.data
    log_user("Model:    {}", d["model"])
    log_user("Backend:  {}", d["backend"])
    log_user("Endpoint: {}", d["endpoint"])
    log_user("PID:      {}", d["handle_id"])
    metadata = d.get("metadata")
    if metadata:
        for key, value in metadata.items():
            log_user("  {}: {}", key, value)
    log_user("Handle:   {}", d["handle_path"])


def _render_text_infer_dry_run(result: CommandResult) -> None:
    """Text renderer for infer start --dry-run."""
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    # Dry-run outputs JSON to stdout even in text mode so the result is
    # pipe-friendly (e.g. `sieval infer start --dry-run | jq .command`).
    print(json.dumps(result.data, indent=2, default=str))


def _render_text_task_reports(result: CommandResult) -> None:
    """Shared text renderer for eval and run — prints task reports."""
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    tasks = result.data.get("tasks", {})
    log_user("\n" + "=" * 60)
    log_user("RESULTS")
    log_user("=" * 60)
    for task_name, task_data in tasks.items():
        log_user("\n--- {} ---", task_name)
        log_user("{}", task_data.get("report", ""))


def _render_text_dry_run(result: CommandResult) -> None:
    if not result.ok:
        # Callers always provide data for dry_run, but guard defensively.
        checks = result.data.get("checks", []) if isinstance(result.data, dict) else []
        for check in checks:
            if check["ok"]:
                log_user("✓ {}", check["name"])
            else:
                log_user("✗ {}", check.get("detail", check["name"]))
            for w in check.get("warnings", []):
                log_user("⚠ {}", w)
        d = result.data if isinstance(result.data, dict) else {}
        n_err = d.get("n_errors", 0)
        n_warn = d.get("n_warnings", 0)
        log_user("\nDry-run failed: {} error(s), {} warning(s).", n_err, n_warn)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    checks = result.data.get("checks", [])
    for check in checks:
        if check["ok"]:
            detail = check.get("detail")
            if detail:
                log_user("✓ {}", detail)
            else:
                log_user("✓ {}", check["name"])
        else:
            log_user("✗ {}", check.get("detail", check["name"]))
        for w in check.get("warnings", []):
            log_user("⚠ {}", w)
    n_warn = result.data.get("n_warnings", 0)
    if n_warn:
        log_user("\nDry-run passed with {} warning(s).", n_warn)
    else:
        log_user("\nDry-run passed.")



def _render_text_leaderboard_list(result: CommandResult) -> None:
    """Text renderer for leaderboard.list — NAME / MODELS / TASKS / PATH."""
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return

    rows = result.data.get("leaderboards", [])
    if not rows:
        log_user("No leaderboards found.")
        return

    table_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("error"):
            models_cell = "[malformed]"
            tasks_cell = "[malformed]"
        else:
            models_cell = str(len(row["models"]))
            tasks_cell = str(len(row["tasks"]))
        table_rows.append(
            {
                "name": row["name"],
                "models": models_cell,
                "tasks": tasks_cell,
                "path": row["path"],
            }
        )

    _render_tabular(
        table_rows,
        [
            ("NAME", "name"),
            ("MODELS", "models"),
            ("TASKS", "tasks"),
            ("PATH", "path"),
        ],
    )

    error_rows = [(row["name"], row["error"]) for row in rows if row.get("error")]
    if error_rows:
        log_user("")
        log_user("Errors:")
        for name, err in error_rows:
            log_user("  {}: {}", name, err)


def _format_leaderboard_cell(result: "LeaderboardResult | None") -> str:
    """Format one ``(model, task)`` cell for the text leaderboard table.

    When ``result`` carries an ``annotation`` dict (from an alignment card),
    append ``(Δ<signed> <glyph>)`` and render score + diff at a precision
    chosen from the card's ``tolerance`` — so correlation-scale tolerances
    (0.03) don't collapse to ``Δ-0.0``. Unannotated cells keep the legacy
    ``.1f`` score. Mixed cases (no result, no score, no annotation) degrade
    silently: missing cells render as ``-``.
    """
    if result is None:
        return "-"
    score = result["report"].get("score")
    if score is None:
        return "-"

    ann = result.get("annotation")
    if ann is None:
        return f"{score:.1f}"

    from sieval.cli.leaderboard.annotation import display_precision

    prec = display_precision(ann["tolerance"])
    diff = ann["diff"]
    glyph = "✓" if ann["status"] == "pass" else "✗"  # ✓ / ✗
    if diff == 0:
        return f"{score:.{prec}f} (Δ0 {glyph})"
    sign = "-" if diff < 0 else "+"
    return f"{score:.{prec}f} (Δ{sign}{abs(diff):.{prec}f} {glyph})"


def _render_text_leaderboard_report(result: CommandResult) -> None:
    """Text renderer for leaderboard.report — tabular model × task matrix."""
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return

    models: list[str] = result.data.get("models", [])
    tasks: list[str] = result.data.get("tasks", [])
    results: list[dict] = result.data.get("results", [])

    if not results:
        log_user("No results found.")
        return

    # Build (model, task) → formatted cell lookup
    cell_lookup: dict[tuple[str, str], str] = {}
    for r in results:
        cell_lookup[(r["model"], r["task"])] = _format_leaderboard_cell(r)

    # Column widths — per-task, sized to the widest rendered cell (including
    # any Δ + glyph annotation) so annotated columns don't truncate.
    model_col_w = max(len("Model"), *(len(m) for m in models))
    task_col_ws: list[int] = []
    for t in tasks:
        widest_cell = max(
            (len(cell_lookup.get((m, t), "-")) for m in models), default=0
        )
        task_col_ws.append(max(len(t), widest_cell, 8))

    # Header row
    header = f"{'Model':<{model_col_w}}"
    for t, w in zip(tasks, task_col_ws, strict=True):
        header += f"  {t:>{w}}"
    log_user("{}", header)

    # Separator
    sep = "-" * model_col_w
    for w in task_col_ws:
        sep += f"  {'-' * w}"
    log_user("{}", sep)

    # Data rows
    for model in models:
        row = f"{model:<{model_col_w}}"
        for task, w in zip(tasks, task_col_ws, strict=True):
            cell = cell_lookup.get((model, task), "-")
            row += f"  {cell:>{w}}"
        log_user("{}", row)


def _collapse_constant_columns(
    rows: list[dict],
    cols: list[tuple[str, str]],
    *,
    never_collapse: frozenset[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Partition `cols` into (visible, collapsed) by row-value variance.

    A column collapses when all rows share one value AND the key is not in
    `never_collapse`; collapsed entries carry the shared value so the
    caller can emit a `HEADER: all VALUE` footer. <2 rows skip collapse.
    """
    if len(rows) < 2:
        return list(cols), []

    visible: list[tuple[str, str]] = []
    collapsed: list[tuple[str, str, str]] = []
    for header, key in cols:
        if key in never_collapse:
            visible.append((header, key))
            continue
        values = {str(r[key]) for r in rows}
        if len(values) == 1:
            collapsed.append((header, key, next(iter(values))))
        else:
            visible.append((header, key))
    return visible, collapsed


def _render_tabular(rows: list[dict], cols: list[tuple[str, str]]) -> None:
    """Print `rows` as a fixed-width table; column widths auto-fit content.

    `cols` is `[(header, dict_key), ...]`. The last column is emitted without
    trailing padding so extra-long cells in earlier columns push the tail
    right without producing trailing whitespace.
    """
    widths = [
        max(len(header), *(len(str(r[key])) for r in rows)) for header, key in cols
    ]
    headers = [h for h, _ in cols]

    def _fmt(values: list[str]) -> str:
        padded = [f"{v:<{w}s}" for v, w in zip(values[:-1], widths[:-1], strict=True)]
        padded.append(values[-1])
        return "  " + "  ".join(padded)

    log_user("{}", _fmt(headers))
    for r in rows:
        log_user("{}", _fmt([str(r[key]) for _, key in cols]))


_DATASET_LIST_COLS: list[tuple[str, str]] = [
    ("NAME", "name"),
    ("DOMAIN", "domain"),
    ("DEPS_GROUP", "deps_group"),
    ("LICENSE", "license"),
    ("READY", "ready"),
]
_DATASET_LIST_PINNED: frozenset[str] = frozenset({"name", "ready"})


def _render_text_dataset_list(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    data = result.data
    if not isinstance(data, list):
        logger.error("expected list data, got {}", type(data).__name__)
        return
    if not data:
        log_user("No datasets registered.")
        return
    visible, collapsed = _collapse_constant_columns(
        data, _DATASET_LIST_COLS, never_collapse=_DATASET_LIST_PINNED
    )
    _render_tabular(data, visible)
    for header, _, value in collapsed:
        log_user("{}: all {}", header, value)


_TASK_LIST_COLS: list[tuple[str, str]] = [
    ("NAME", "name"),
    ("DATASET", "dataset"),
    ("EVAL_MODE", "eval_mode"),
    ("N_SHOT", "n_shot"),
    ("DEPS_GROUP", "deps_group"),
    ("STATUS", "status"),
    ("READY", "ready"),
]
_TASK_LIST_PINNED: frozenset[str] = frozenset({"name", "ready"})


def _render_text_task_list(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    data = result.data
    if not isinstance(data, list):
        logger.error("expected list data, got {}", type(data).__name__)
        return
    if not data:
        log_user("No tasks registered.")
        return
    visible, collapsed = _collapse_constant_columns(
        data, _TASK_LIST_COLS, never_collapse=_TASK_LIST_PINNED
    )
    _render_tabular(data, visible)
    for header, _, value in collapsed:
        log_user("{}: all {}", header, value)


def _render_text_task_show(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    d = result.data
    log_user("Name:          {}", d["name"])
    log_user("Display:       {}", d["display_name"])
    log_user("Description:   {}", d["description"])
    cats = d.get("dataset_categories") or []
    dataset_line = d["dataset"]
    if cats:
        dataset_line = f"{d['dataset']} ({', '.join(cats)})"
    log_user("Dataset:       {}", dataset_line)
    log_user("Eval mode:     {}", d["eval_mode"])
    log_user("N-shot:        {}", d["n_shot"])
    log_user("Model type:    {}", d["model_type"] or "-")
    log_user("Deps:          {}", d["deps_group"] or "-")
    log_user("Status:        {}", d["status"])
    ref = d.get("reference_impl")
    if ref:
        log_user("")
        log_user("Reference impl:")
        log_user("  Source:      {}", ref["source"])
        log_user("  URL:         {}", ref["url"])
        if ref.get("notes"):
            log_user("  Notes:       {}", ref["notes"])

    log_user("")
    log_user("Ready:         {}", d["ready"])
    missing = d.get("missing") or []
    if missing:
        log_user("Missing:")
        for entry in missing:
            _render_missing_entry(entry)

    run_hint = d.get("run_hint")
    if run_hint:
        log_user("Run:           {}", run_hint)


# Width = longest kind literal + 1 space, so all kind labels align.
_MISSING_KIND_WIDTH = max(len(k) for k in MISSING_KINDS) + 1


def _render_missing_entry(entry: dict) -> None:
    """One line per source / unmet requirement, indented and kind-aligned."""
    kind = entry["kind"]
    kind_col = f"{kind:<{_MISSING_KIND_WIDTH}s}"
    if kind == "data":
        for src in entry.get("sources", []):
            log_user("  {}{}", kind_col, src)
    else:
        group = entry.get("group")
        suffix = f"   [group: {group}]" if group else ""
        for unmet in entry.get("unmet", []):
            log_user("  {}{}{}", kind_col, unmet, suffix)


def _render_text_dataset_show(result: CommandResult) -> None:
    if not result.ok:
        logger.error("{}", result.error)
        return
    if not isinstance(result.data, dict):
        logger.error("expected dict data, got {}", type(result.data).__name__)
        return
    d = result.data
    log_user("Name:          {}", d["name"])
    log_user("Display:       {}", d["display_name"])
    log_user("Description:   {}", d["description"])
    log_user("Source:        {}", ", ".join(d["source"]))
    log_user("Categories:    {}", d["categories"])
    log_user("Tags:          {}", ", ".join(d["tags"]) if d["tags"] else "-")
    log_user("Deps:          {}", d["deps_group"] or "-")
    log_user("License:       {}", d["license"])
    log_user("")
    log_user("Ready:         {}", d["ready"])
    missing = d.get("missing") or []
    if missing:
        log_user("Missing:")
        for entry in missing:
            _render_missing_entry(entry)
    suggested = d.get("suggested_path")
    if suggested:
        log_user("")
        log_user("YAML path:     {}", suggested)
    log_user("")
    tasks = d.get("tasks") or []
    log_user("Tasks using this dataset ({}):", len(tasks))
    if tasks:
        _render_tabular(
            tasks,
            [
                ("NAME", "name"),
                ("EVAL_MODE", "eval_mode"),
                ("N_SHOT", "n_shot"),
                ("STATUS", "status"),
            ],
        )


# ── registry ──────────────────────────────────────────────────────────

_TEXT_RENDERERS: dict[str, Callable[[CommandResult], None]] = {
    "infer.list": _render_text_infer_list,
    "infer.show": _render_text_infer_show,
    "infer.stop": _render_text_infer_stop,
    "infer.start": _render_text_infer_start,
    "infer.dry_run": _render_text_infer_dry_run,
    "eval": _render_text_task_reports,
    "run": _render_text_task_reports,
    "leaderboard.run": _render_text_task_reports,
    "eval.dry_run": _render_text_dry_run,
    "run.dry_run": _render_text_dry_run,
    "leaderboard.run.dry_run": _render_text_dry_run,
    "leaderboard.report": _render_text_leaderboard_report,
"leaderboard.list": _render_text_leaderboard_list,
    "dataset.list": _render_text_dataset_list,
    "dataset.show": _render_text_dataset_show,
    "task.list": _render_text_task_list,
    "task.show": _render_text_task_show,
}
