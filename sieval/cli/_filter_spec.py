"""Shape of a ``filter`` dataset operation, shared by its two surfaces.

``cli.validation`` shape-checks a config with no dataset in hand and collects
every problem it finds; ``cli.leaderboard.session`` applies the operation and
raises on the first. They ask the same questions of the same keys, so the
questions live here and each caller keeps its own way of reporting the answer —
returning a message rather than raising is what lets one accumulate and the
other raise. ``cli.validation`` already imports ``cli.leaderboard.session``, so
this cannot live in either without pointing that dependency backwards.

Callers prepend their own ``Dataset '<name>': `` to every message returned here.

This module also pins ``values_file``. The accepted values live outside the
config, so the config alone does not say which rows a run selected: two runs
whose ``effective_config.yaml`` compares equal can score different sample sets
if the file changed in between, and ``--resume`` would accept the second. So
the file's digest is recorded *into* the config at load time, where the resume
gate can see it. The path is still stored verbatim, which is what keeps
``effective_config.yaml`` portable across machines.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import hashlib
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path

#: Config key holding the digest of the ``values_file`` a run selected on.
DIGEST_KEY = "values_digest"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def check_by(by: object) -> str | None:
    """The problem with a ``filter`` operation's ``by``, or ``None``.

    Shape only — whether a named column exists and whether a dotted path
    imports are questions for the session, which has the dataset and the
    config's directory.
    """
    if by is None:
        return "'filter' requires 'by'"
    if isinstance(by, str):
        return None
    if isinstance(by, list):
        if not by or not all(isinstance(col, str) for col in by):
            return (
                f"'filter' 'by' as a list must name one or more columns as "
                f"strings; got {by!r}"
            )
        return None
    if isinstance(by, dict):
        # The value is read positionally because `isinstance(x, dict)` narrows
        # an `object` to dict[Never, Never], which rejects a `str` key lookup.
        # The key check to its left is what guarantees there is one to read.
        values = list(by.values())
        if set(by) != {"callable"} or not isinstance(values[0], str):
            return (
                f"'filter' 'by' as a mapping takes exactly one key, "
                f"'callable', naming a dotted path; got {by!r}"
            )
        return None
    return (
        f"'filter' 'by' must be a column name, a list of column names, or "
        f"{{callable: 'pkg.module.fn'}}; got {by!r}"
    )


def check_values_source(op_args: Mapping) -> list[str]:
    """Every problem with where a ``filter`` operation gets its values."""
    problems: list[str] = []

    # Presence, not truthiness: `value: 0` and `value: false` are legitimate
    # column values and must not read as "omitted".
    has_value = "value" in op_args
    values_file = op_args.get("values_file")
    if has_value == (values_file is not None):
        problems.append("'filter' requires exactly one of 'value' or 'values_file'")
    if values_file is not None and not isinstance(values_file, str):
        problems.append(f"'filter' 'values_file' must be a path; got {values_file!r}")

    digest = op_args.get(DIGEST_KEY)
    if digest is not None:
        if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
            problems.append(
                f"'filter' {DIGEST_KEY!r} must look like 'sha256:<64 hex>'; "
                f"got {digest!r}"
            )
        elif values_file is None:
            problems.append(
                f"'filter' {DIGEST_KEY!r} pins a 'values_file', but none is given"
            )

    require_all = op_args.get("require_all")
    if require_all is not None and not isinstance(require_all, bool):
        problems.append(
            f"'filter' 'require_all' must be a boolean; got {require_all!r}"
        )
    return problems


def resolve_values_path(values_file: str, config_dir: Path) -> Path:
    """A ``values_file`` as an absolute path.

    Relative paths resolve against the config that named them — the rule
    ``alignment.card`` already follows.
    """
    path = Path(values_file)
    return path if path.is_absolute() else (config_dir / path).resolve()


def compute_values_digest(data: bytes) -> str:
    """The recorded digest of a values file's contents."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def pin_values_files(cfg: MutableMapping, config_dir: Path) -> None:
    """Record each ``filter`` operation's ``values_file`` digest into *cfg*.

    Called on the reified config before it is both persisted and handed to the
    runtime, so the digest reaches ``effective_config.yaml`` — and therefore
    the ``--resume`` comparison — without the transform ever learning where a
    config lives.

    A digest already present is *verified* rather than overwritten: that is the
    case where a persisted ``effective_config.yaml`` is re-run after its values
    file changed underneath it.

    Malformed operations are left alone. They are ``cli.validation``'s to
    report, and a shape error raised from here would pre-empt a better message.
    """
    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        return
    for name, dataset_cfg in datasets.items():
        if not isinstance(dataset_cfg, dict):
            continue
        operations = dataset_cfg.get("operations")
        if not isinstance(operations, list):
            continue
        for op in operations:
            if not isinstance(op, dict) or len(op) != 1:
                continue
            if next(iter(op)) != "filter":
                continue
            op_args = op["filter"]
            if not isinstance(op_args, dict):
                continue
            values_file = op_args.get("values_file")
            if not isinstance(values_file, str):
                continue

            path = resolve_values_path(values_file, config_dir)
            if not path.is_file():
                raise ValueError(
                    f"Dataset '{name}': 'filter' 'values_file' not found: {path}"
                )
            digest = compute_values_digest(path.read_bytes())
            recorded = op_args.get(DIGEST_KEY)
            if recorded is None:
                op_args[DIGEST_KEY] = digest
            elif recorded != digest:
                raise ValueError(
                    f"Dataset '{name}': 'filter' 'values_file' {values_file} "
                    f"has changed since this config recorded it "
                    f"({DIGEST_KEY} says {recorded}, the file is {digest}). "
                    f"The selection would not be the one this config names. "
                    f"Either:\n"
                    f"  1. Restore {path} to the contents the digest pins\n"
                    f"  2. Drop '{DIGEST_KEY}' to pin the current contents — a "
                    f"different selection, so use a fresh result_dir"
                )
