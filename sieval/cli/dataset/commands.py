"""
sieval dataset {list, show, download} commands.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from pathlib import Path
from typing import Annotated

import click
import typer
from loguru import logger

from sieval.cli.dataset.render import (
    render_dataset_list,
    render_dataset_show,
)
from sieval.cli.output import CommandResult, OutputFormat, cli_command, render
from sieval.core.datasets.meta import DatasetMeta, Level1Category
from sieval.core.utils.logging import configure_logging, log_user
from sieval.core.utils.paths import resolve_data_dir
from sieval.datasets.downloaders import resolve as resolve_handler
from sieval.datasets.downloaders.local import LocalSourceUnavailable
from sieval.datasets.downloaders.resolver import extras_unsatisfied
from sieval.datasets.downloaders.verify import verify_checksums
from sieval.meta import load_index

dataset_app = typer.Typer(help="Dataset discovery and download.")


@dataset_app.callback()
def _dataset_callback(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
) -> None:
    configure_logging(verbose)


@dataset_app.command("list")
@cli_command
def list_cmd(
    domain: Annotated[
        str | None, typer.Option("--domain", help="Filter by Level1Category.")
    ] = None,
    data_dir: Annotated[
        str | None, typer.Option("--data-dir", help="Override data directory.")
    ] = None,
    output: Annotated[OutputFormat, typer.Option("-o", "--output")] = OutputFormat.TEXT,
) -> None:
    """List registered datasets with domain, deps_group, license, and readiness."""
    datasets, _ = load_index()
    if domain:
        try:
            level1 = Level1Category(domain)
        except ValueError as e:
            valid = [c.value for c in Level1Category]
            raise typer.BadParameter(
                f"Unknown domain {domain!r}. Options: {valid}"
            ) from e
        datasets = [
            m for m in datasets if any(c.level1 is level1 for c in m.categories)
        ]
    resolved_dir = resolve_data_dir(data_dir)
    render(render_dataset_list(datasets, data_dir=resolved_dir), output)


@dataset_app.command("show")
@cli_command
def show_cmd(
    name: Annotated[str, typer.Argument()],
    data_dir: Annotated[
        str | None, typer.Option("--data-dir", help="Override data directory.")
    ] = None,
    output: Annotated[OutputFormat, typer.Option("-o", "--output")] = OutputFormat.TEXT,
) -> None:
    """Show a dataset's full metadata plus the tasks that consume it."""
    datasets, tasks = load_index()
    meta = next((d for d in datasets if d.name == name), None)
    if meta is None:
        render(
            CommandResult(
                command="dataset.show",
                ok=False,
                error=(
                    f"Dataset {name!r} is not registered. "
                    "Run `sieval dataset list` to see available options."
                ),
            ),
            output,
        )
        raise typer.Exit(code=1)
    related = [t for t in tasks if t.dataset == name]
    resolved_dir = resolve_data_dir(data_dir)
    render(render_dataset_show(meta, related, data_dir=resolved_dir), output)


@dataset_app.command("download")
@cli_command
def download_cmd(
    name: Annotated[str | None, typer.Argument()] = None,
    domain: Annotated[str | None, typer.Option("--domain")] = None,
    all_: Annotated[bool, typer.Option("--all/--no-all")] = False,
    data_dir: Annotated[str | None, typer.Option("--data-dir")] = None,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    output: Annotated[OutputFormat, typer.Option("-o", "--output")] = OutputFormat.TEXT,
) -> None:
    """Download dataset sources to local storage."""
    datasets, _ = load_index()

    # Mutually exclusive input mode. `click.UsageError`, not
    # `typer.BadParameter`: no single value is invalid, the *combination* is,
    # and BadParameter's "Invalid value:" framing would misdescribe it. Both
    # exit 2; both reach `@cli_command`'s ClickException branch, so `--output
    # json` gets the structured form while text keeps Click's usage hint.
    provided = sum(bool(x) for x in (name, domain, all_))
    if provided != 1:
        raise click.UsageError(
            "Exactly one of <name>, --domain, or --all must be provided."
        )

    dest_root = resolve_data_dir(data_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    # `task list` / `eval` read from the default, not --data-dir.
    if data_dir is not None and resolve_data_dir(None) != dest_root:
        _warn(
            warnings,
            f"⚠ --data-dir {dest_root} differs from the default "
            f"({resolve_data_dir(None)}). `sieval task list` / `sieval eval` "
            f"will read from the default. "
            f"Set SIEVAL_DATA_DIR={dest_root} to make this override persistent.",
        )

    if name:
        meta = next((d for d in datasets if d.name == name), None)
        if meta is None:
            render(
                CommandResult(
                    command="dataset.download",
                    ok=False,
                    error=f"Dataset {name!r} is not registered.",
                    warnings=warnings or None,
                ),
                output,
            )
            raise typer.Exit(code=1)
        metas = [meta]
    elif domain:
        try:
            level1 = Level1Category(domain)
        except ValueError as e:
            valid = [c.value for c in Level1Category]
            raise typer.BadParameter(
                f"Unknown domain {domain!r}. Options: {valid}"
            ) from e
        metas = [m for m in datasets if any(c.level1 is level1 for c in m.categories)]
        if not metas:
            # Matching nothing is not a failure — exit 0, with the empty
            # summary a machine caller can read as "requested: 0".
            _warn(warnings, f"No datasets matched --domain {domain!r}.")
            render(
                CommandResult(
                    command="dataset.download",
                    ok=True,
                    data=_summary(dest_root, []),
                    warnings=warnings,
                ),
                output,
            )
            return
    else:
        metas = datasets

    batch = len(metas) > 1
    records: list[dict] = []
    for m in metas:
        try:
            record, dataset_warnings = _download_one(m, dest_root, force)
        except Exception as exc:
            if not batch:
                # Fail-fast on single-target: the user asked for exactly one
                # dataset, so there is no partial result worth reporting.
                # `@cli_command` renders the exception in the chosen format.
                raise
            # Batch mode: one bad source shouldn't block the rest. Report it as
            # it happens — a long batch shouldn't stay silent until the end.
            logger.error("[{}] FAILED: {}", m.name, exc)
            records.append({"name": m.name, "ok": False, "error": str(exc)})
        else:
            records.append(record)
            warnings.extend(dataset_warnings)

    failures = [r for r in records if not r["ok"]]
    render(
        CommandResult(
            command="dataset.download",
            ok=not failures,
            data=_summary(dest_root, records),
            error=(
                f"{len(failures)} of {len(records)} dataset(s) failed."
                if failures
                else None
            ),
            warnings=warnings or None,
        ),
        output,
    )
    if failures:
        raise typer.Exit(code=1)


def _warn(sink: list[str], message: str) -> None:
    """Record a warning for the result payload *and* surface it immediately.

    Both, deliberately. ``CommandResult.warnings`` is a machine-payload field —
    no text renderer prints it — and a download runs for as long as its sources
    take, so a warning held back until the final render would never reach a user
    who walks away or hits Ctrl-C. Emitting through ``logger`` also keeps it on
    stderr, where it cannot corrupt ``--output json`` on stdout.
    """
    sink.append(message)
    logger.warning("{}", message)


def _summary(dest_root: Path, records: list[dict]) -> dict:
    """Wire shape for the download result."""
    return {
        "data_dir": str(dest_root),
        "requested": len(records),
        "succeeded": sum(1 for r in records if r["ok"]),
        "failed": sum(1 for r in records if not r["ok"]),
        "datasets": records,
    }


def _download_one(
    m: DatasetMeta, dest_root: Path, force: bool
) -> tuple[dict, list[str]]:
    """Fetch one dataset's sources and verify them.

    Returns its wire record plus any non-fatal warnings; raises on failure.
    Warnings are also emitted as they occur, so the ones raised past — a
    bring-your-own corpus whose instructions precede the raise — still reach
    the user; the exception message carries the same substance for machines.
    """
    warnings: list[str] = []
    missing_local_sources: list[str] = []
    fetched = 0
    already_present = 0
    for src in m.source:
        h = resolve_handler(src)
        if h.is_downloaded(src, dest_root, m.name) and not force:
            # Progress, not result data: `log_user` keeps it on stderr so
            # `--output json` stays parseable on stdout.
            log_user("[{}] already present: {}", m.name, src)
            already_present += 1
            continue
        log_user("[{}] fetching {}", m.name, src)
        try:
            h.download(src, dest_root, m.name, force=force)
        except LocalSourceUnavailable as e:
            # BYO corpus: surface the instructions and track for a summary.
            _warn(warnings, f"[{m.name}] {e}")
            missing_local_sources.append(src)
        else:
            fetched += 1

    mismatches = verify_checksums(m, dest_root)
    if mismatches:
        for mm in mismatches:
            (dest_root / m.name / mm.basename).unlink(missing_ok=True)
        details = "; ".join(
            f"{mm.basename}: expected {mm.expected}, got {mm.actual or 'MISSING'}"
            for mm in mismatches
        )
        raise RuntimeError(
            f"checksum verification failed for {m.name!r} ({details}); "
            f"deleted the mismatched file(s) — re-run "
            f"`sieval dataset download {m.name}` to refetch"
        )

    # BYO corpus: exit non-zero if any local: sources were unavailable. The
    # per-source instructions (including each file's exact target path) were
    # printed above by the handler, so keep this summary dataset-agnostic — it
    # runs for every `local:` source, not just one benchmark's.
    if missing_local_sources:
        raise RuntimeError(
            f"{m.name!r} requires {len(missing_local_sources)} bring-your-own "
            f"corpus file(s) that cannot be fetched automatically: "
            f"{', '.join(missing_local_sources)}. Produce them as instructed "
            f"above, then re-run `sieval dataset download {m.name}`."
        )

    # Post-download hint; print-only, never installs.
    if m.deps_group:
        unmet = extras_unsatisfied(m.deps_group)
        if unmet:
            details = "\n".join(f"    - {u}" for u in unmet)
            _warn(
                warnings,
                f"Dataset {m.name!r} requires extras group {m.deps_group!r}.\n"
                f"  Unsatisfied requirements:\n{details}\n"
                f"  To enable:\n"
                f"    pip install 'sieval[{m.deps_group}]'\n"
                f"  (PDM/Poetry/uv users: use your tool's equivalent.)",
            )

    return (
        {
            "name": m.name,
            "ok": True,
            "fetched": fetched,
            "already_present": already_present,
            "error": None,
        },
        warnings,
    )
