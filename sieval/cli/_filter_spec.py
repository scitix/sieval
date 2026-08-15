"""Shape of a ``filter`` dataset operation, shared by its two surfaces.

``cli.validation`` shape-checks a config with no dataset in hand and collects
every problem; ``cli.leaderboard.session`` applies the operation and raises on
the first. Same questions, different reporting — which is why these return a
message rather than raising. They live here because ``cli.validation`` already
imports ``cli.leaderboard.session``, so neither file could hold them without
pointing that dependency backwards.

Callers prepend their own ``Dataset '<name>': `` to every message returned here.

This module also pins the two things a ``filter`` names but does not contain:
the ``values_file`` holding its accepted values, and the key function
``by: {callable: ...}`` names. Either can change while the config does not, so
two runs whose ``effective_config.yaml`` compares equal can select different
rows — and ``--resume`` would accept the second. Recording a digest of each
*into* the config puts that difference where the resume gate can see it, while
the path and the dotted path stay verbatim so ``effective_config.yaml`` remains
portable across machines.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import hashlib
import inspect
import re
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from pathlib import Path

from loguru import logger

from .resolution import resolve_key_function

#: Config key holding the digest of the ``values_file`` a run selected on.
VALUES_DIGEST_KEY = "values_digest"

#: Config key holding the digest of the key function a run selected with.
BY_DIGEST_KEY = "by_digest"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

#: Every key a ``filter`` operation accepts. Enumerated so a typo in an
#: *optional* one is caught: a misspelled ``by`` or ``value`` fails the checks
#: below because what they name goes missing, but ``require_all_keys: true``
#: reads as ``require_all`` simply left at its default, and a misspelled
#: ``split`` reads as a split that is not there — both silently.
_FILTER_KEYS = frozenset(
    {
        "by",
        "value",
        "values_file",
        "require_all",
        "split",
        VALUES_DIGEST_KEY,
        BY_DIGEST_KEY,
    }
)


def check_arg_names(op_args: Mapping) -> list[str]:
    """Every problem with a ``filter`` operation's key names, and with *split*.

    *split* is checked here rather than against the dataset: whether the split
    exists is the session's question, but whether it is a *string* is answerable
    with no data in hand, and a non-string one silently matches no split.
    """
    problems: list[str] = []
    unknown = sorted(set(op_args) - _FILTER_KEYS)
    if unknown:
        problems.append(
            f"'filter' has unknown key(s) {unknown}; valid keys are "
            f"{sorted(_FILTER_KEYS)}"
        )
    split = op_args.get("split")
    if split is not None and not isinstance(split, str):
        problems.append(f"'filter' 'split' must be a split name; got {split!r}")
    return problems


def key_function_spec(by: object) -> str | None:
    """The dotted path *by* names, or ``None`` if it is not the callable form.

    The one place that knows how ``by: {callable: 'pkg.module.fn'}`` is spelled,
    so the checker, the pin and the session all read it the same way.
    """
    if not isinstance(by, dict) or set(by) != {"callable"}:
        return None
    # Read positionally: `isinstance(x, dict)` narrows an `object` to
    # dict[Never, Never], which rejects a `str` key lookup. The key check above
    # guarantees there is exactly one value to read.
    spec = next(iter(by.values()))
    return spec if isinstance(spec, str) else None


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
        if key_function_spec(by) is None:
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

    digest = op_args.get(VALUES_DIGEST_KEY)
    if digest is not None:
        if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
            problems.append(
                f"'filter' {VALUES_DIGEST_KEY!r} must look like 'sha256:<64 hex>'; "
                f"got {digest!r}"
            )
        elif values_file is None:
            problems.append(
                f"'filter' {VALUES_DIGEST_KEY!r} pins a 'values_file', but none "
                f"is given"
            )

    require_all = op_args.get("require_all")
    if require_all is not None and not isinstance(require_all, bool):
        problems.append(
            f"'filter' 'require_all' must be a boolean; got {require_all!r}"
        )
    return problems


def check_by_digest(op_args: Mapping) -> list[str]:
    """Every problem with a ``filter`` operation's pinned key-function digest."""
    digest = op_args.get(BY_DIGEST_KEY)
    if digest is None:
        return []
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        return [
            f"'filter' {BY_DIGEST_KEY!r} must look like 'sha256:<64 hex>'; "
            f"got {digest!r}"
        ]
    if key_function_spec(op_args.get("by")) is None:
        return [
            f"'filter' {BY_DIGEST_KEY!r} pins a key function, but 'by' does not "
            f"name one"
        ]
    return []


def resolve_values_path(values_file: str, config_dir: Path) -> Path:
    """A ``values_file`` as an absolute path.

    Relative paths resolve against the config that named them — the rule
    ``alignment.card`` already follows.
    """
    path = Path(values_file)
    return path if path.is_absolute() else (config_dir / path).resolve()


def relative_values_files(cfg: Mapping) -> list[str]:
    """Every ``values_file`` in *cfg* named by a relative path.

    Resolving against the config being run (above) means the persisted
    ``effective_config.yaml`` reproduces the selection only when re-run from
    beside the original config. Its header says so rather than the path being
    rewritten absolute: the persisted copy would then stop comparing equal to
    the source config's relative one, and ``--resume`` would reject its own
    artifact.
    """
    return [
        values_file
        for _, op_args in _filter_operations(cfg)
        if isinstance(values_file := op_args.get("values_file"), str)
        and not Path(values_file).is_absolute()
    ]


def compute_values_digest(data: bytes) -> str:
    """The recorded digest of a values file's contents."""
    return _digest(data)


def compute_key_function_digest(fn: Callable) -> str | None:
    """The recorded digest of a key function's source, or ``None``.

    ``None`` means the source cannot be read — a builtin, a C extension, a
    ``functools.partial``. Those stay unpinned rather than blocking the run.

    The digest covers the ``def`` block and its decorators, not what the body
    calls, so it is a tripwire for the function being edited rather than a proof
    that the selection reproduces — the same reach ``values_digest`` has over a
    file of ids but not over the dataset they index. Reformatting and comment
    edits trip it too: on a resume a false positive is a question, a false
    negative is a wrong number.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return _digest(source.encode("utf-8"))


def pin_filter_digests(cfg: MutableMapping, config_dir: Path) -> None:
    """Record each ``filter`` operation's provenance digests into *cfg*.

    Called on the reified config before it is both persisted and handed to the
    runtime, so the digests reach ``effective_config.yaml`` — and therefore the
    ``--resume`` comparison.

    A digest already present is *verified* rather than overwritten: that is a
    persisted ``effective_config.yaml`` re-run after its values file or its key
    function changed underneath it.

    Malformed operations are left alone — they are ``cli.validation``'s to
    report, and raising here would pre-empt a better message.
    """
    for name, op_args in _filter_operations(cfg):
        _pin_values_file(name, op_args, config_dir)
        _pin_key_function(name, op_args)


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _filter_operations(cfg: Mapping) -> Iterator[tuple[str, MutableMapping]]:
    """Every well-formed ``filter`` operation in *cfg*, with its dataset name."""
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
            if isinstance(op_args, dict):
                yield name, op_args


def _pin_values_file(name: str, op_args: MutableMapping, config_dir: Path) -> None:
    values_file = op_args.get("values_file")
    if not isinstance(values_file, str):
        return

    path = resolve_values_path(values_file, config_dir)
    if not path.is_file():
        raise ValueError(f"Dataset '{name}': 'filter' 'values_file' not found: {path}")
    digest = compute_values_digest(path.read_bytes())
    recorded = op_args.get(VALUES_DIGEST_KEY)
    if recorded is None:
        op_args[VALUES_DIGEST_KEY] = digest
    elif recorded != digest:
        raise ValueError(
            f"Dataset '{name}': 'filter' 'values_file' {values_file} "
            f"has changed since this config recorded it "
            f"({VALUES_DIGEST_KEY} says {recorded}, the file is {digest}). "
            f"The selection would not be the one this config names. "
            f"Either:\n"
            f"  1. Restore {path} to the contents the digest pins\n"
            f"  2. Drop '{VALUES_DIGEST_KEY}' to pin the current contents — a "
            f"different selection, so use a fresh result_dir"
        )


def _pin_key_function(name: str, op_args: MutableMapping) -> None:
    spec = key_function_spec(op_args.get("by"))
    if spec is None:
        return
    try:
        fn = resolve_key_function(spec)
    except (ValueError, ImportError, AttributeError):
        # Unresolvable: the session raises on it a moment later with a message
        # that names the dataset and what could not be imported.
        return

    digest = compute_key_function_digest(fn)
    recorded = op_args.get(BY_DIGEST_KEY)
    if digest is None:
        if recorded is not None:
            raise ValueError(
                f"Dataset '{name}': 'filter' 'by' callable {spec} is pinned by "
                f"{BY_DIGEST_KEY}, but its source can no longer be read, so the "
                f"pin cannot be checked. Drop '{BY_DIGEST_KEY}' to run it "
                f"unpinned — a selection this config can no longer vouch for, "
                f"so use a fresh result_dir"
            )
        logger.warning(
            "Dataset '{}': 'filter' 'by' callable {} has no readable source, so "
            "it is not pinned and --resume cannot tell if it changed. Give the "
            "key a plain 'def' to make it pinnable.",
            name,
            spec,
        )
        return
    if recorded is None:
        op_args[BY_DIGEST_KEY] = digest
    elif recorded != digest:
        raise ValueError(
            f"Dataset '{name}': 'filter' 'by' callable {spec} has changed since "
            f"this config recorded it ({BY_DIGEST_KEY} says {recorded}, its "
            f"source is now {digest}). The selection would not be the one this "
            f"config names. Either:\n"
            f"  1. Restore {spec} to the source the digest pins\n"
            f"  2. Drop '{BY_DIGEST_KEY}' to pin the current source — a "
            f"different selection, so use a fresh result_dir"
        )
