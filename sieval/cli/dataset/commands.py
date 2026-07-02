"""
sieval dataset {list, show, download} commands.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from pathlib import Path
from typing import Annotated

import typer

from sieval.cli.dataset.render import (
    render_dataset_list,
    render_dataset_show,
)
from sieval.cli.output import OutputFormat, render
from sieval.core.datasets.meta import DatasetMeta, Level1Category
from sieval.core.utils.logging import configure_logging
from sieval.core.utils.paths import resolve_data_dir
from sieval.datasets.downloaders import resolve as resolve_handler
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
        typer.secho(
            f"Dataset {name!r} is not registered. "
            "Run `sieval dataset list` to see available options.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    related = [t for t in tasks if t.dataset == name]
    resolved_dir = resolve_data_dir(data_dir)
    render(render_dataset_show(meta, related, data_dir=resolved_dir), output)


@dataset_app.command("download")
def download_cmd(
    name: Annotated[str | None, typer.Argument()] = None,
    domain: Annotated[str | None, typer.Option("--domain")] = None,
    all_: Annotated[bool, typer.Option("--all/--no-all")] = False,
    data_dir: Annotated[str | None, typer.Option("--data-dir")] = None,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
) -> None:
    """Download dataset sources to local storage."""
    datasets, _ = load_index()

    # Mutually exclusive input mode
    provided = sum(bool(x) for x in (name, domain, all_))
    if provided != 1:
        typer.secho(
            "Exactly one of <name>, --domain, or --all must be provided.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    dest_root = resolve_data_dir(data_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    # `task list` / `eval` read from the default, not --data-dir.
    # Fire upfront so Ctrl-C doesn't swallow the warning.
    if data_dir is not None and resolve_data_dir(None) != dest_root:
        typer.secho(
            f"⚠ --data-dir {dest_root} differs from the default "
            f"({resolve_data_dir(None)}). `sieval task list` / `sieval eval` "
            f"will read from the default. "
            f"Set SIEVAL_DATA_DIR={dest_root} to make this override persistent.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    if name:
        meta = next((d for d in datasets if d.name == name), None)
        if meta is None:
            typer.secho(
                f"Dataset {name!r} is not registered.",
                fg=typer.colors.RED,
                err=True,
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
            typer.secho(
                f"No datasets matched --domain {domain!r}.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=0)
    else:
        metas = datasets

    batch = len(metas) > 1
    failures: list[tuple[str, Exception]] = []
    for m in metas:
        try:
            _download_one(m, dest_root, force)
        except Exception as exc:
            if not batch:
                # Fail-fast on single-target to preserve the original traceback.
                raise
            # Batch mode: one bad source shouldn't block the rest.
            typer.secho(
                f"[{m.name}] FAILED: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            failures.append((m.name, exc))

    if failures:
        typer.secho(
            f"\n{len(failures)} of {len(metas)} dataset(s) failed:",
            fg=typer.colors.RED,
            err=True,
        )
        for fname, exc in failures:
            typer.secho(f"  - {fname}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def _download_one(m: DatasetMeta, dest_root: Path, force: bool) -> None:
    for src in m.source:
        h = resolve_handler(src)
        if h.is_downloaded(src, dest_root, m.name) and not force:
            typer.echo(f"[{m.name}] already present: {src}")
            continue
        typer.echo(f"[{m.name}] fetching {src}")
        h.download(src, dest_root, m.name, force=force)

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

    # Post-download hint; print-only, never installs.
    if m.deps_group:
        unmet = extras_unsatisfied(m.deps_group)
        if unmet:
            details = "\n".join(f"    - {u}" for u in unmet)
            typer.secho(
                f"Dataset {m.name!r} requires extras group {m.deps_group!r}.\n"
                f"  Unsatisfied requirements:\n{details}\n"
                f"  To enable:\n"
                f"    pip install 'sieval[{m.deps_group}]'\n"
                f"  (PDM/Poetry/uv users: use your tool's equivalent.)",
                fg=typer.colors.YELLOW,
            )
