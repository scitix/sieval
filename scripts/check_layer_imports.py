"""
Pre-commit hook: enforce sieval import policy.

Two categories of check:

1. **Layer boundary imports** — each layer has a hard-coded set of sibling
   layers it must not import from. Current map:

       cli/          → orchestration layer, depends on all modules
       infer/        → can depend on core; NOT on tasks/datasets
       tasks/        → depends on core + datasets + community
       datasets/     → depends on core + community
       core/         → zero upper-layer dependencies (independently publishable)
       community/    → third-party evaluation adaptations (used by tasks/datasets)

2. **Private-access discipline** (encodes CLAUDE.md `## Import Policy`):

   * Imports imply public API — a cross-module ``from sieval.x.y import _foo``
     in production code is a smell. Flagged unless relative (same-package)
     or under ``tests/`` (the explicit carve-out).
   * Private modules (``_*.py``) are **protected** — only their own package
     subtree may reach into them. Peer-subpackage access or out-of-subtree
     access is flagged.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import ast
import sys
from pathlib import Path

# layer -> set of sieval sub-packages it must NOT import
FORBIDDEN: dict[str, set[str]] = {
    "core": {"cli", "infer", "tasks", "datasets", "community"},
    "datasets": {"cli", "tasks", "infer"},
    "infer": {"cli", "tasks", "datasets", "community"},
    "tasks": {"cli", "infer"},
}

# Repo-level sibling directories of the `sieval/` package. When the repo
# directory is itself named `sieval`, an absolute path such as
# `/repo/sieval/scripts/foo.py` contains `sieval` as a parent segment but
# the file lives OUTSIDE the Python package. `_sieval_root_index`'s
# "last sieval" heuristic would misattribute it; these names let
# `_get_layer` / `_file_package` reject such misattributions so the file is
# correctly classified as tooling / tests / docs instead of a sieval layer.
_SIEVAL_OUTER_SIBLINGS: frozenset[str] = frozenset(
    {
        "scripts",
        "tests",
        "docs",
        "data",
        "leaderboards",
        "mutants",
        "outputs",
        "vendor",
    }
)

# Guard: if a real sieval layer is ever named the same as an outer-sibling
# dir (e.g. adding `sieval/data/`), `_get_layer` would silently return None
# and skip layer-import checks for that layer. Fail loud at import time
# instead of producing dead enforcement.
_layer_sibling_collision = FORBIDDEN.keys() & _SIEVAL_OUTER_SIBLINGS
if _layer_sibling_collision:
    raise RuntimeError(
        f"sieval layer name(s) collide with outer-sibling directory names: "
        f"{sorted(_layer_sibling_collision)}. Rename the layer or drop the "
        f"entry from _SIEVAL_OUTER_SIBLINGS."
    )


def _sieval_root_index(parts: tuple[str, ...]) -> int | None:
    """Return the index of the package-root ``sieval`` segment in *parts*.

    The project directory is conventionally named ``sieval`` too, so an
    absolute path like ``/home/x/sieval/sieval/tasks/foo.py`` contains the
    token twice. Pick the **last** occurrence — that's always the package
    root, since no sieval submodule is itself named ``sieval``.
    """
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "sieval":
            return i
    return None


def _get_layer(path: Path) -> str | None:
    """Return the layer name if *path* lives under sieval/<layer>/.

    Returns ``None`` for files in repo-level sibling directories of the
    sieval package (``scripts/``, ``tests/``, ``docs/``, …), even when the
    repo dir is itself named ``sieval`` — those files are not inside the
    Python package.
    """
    parts = path.parts
    idx = _sieval_root_index(parts)
    if idx is None or idx + 1 >= len(parts):
        return None
    candidate = parts[idx + 1]
    if candidate in _SIEVAL_OUTER_SIBLINGS:
        return None
    return candidate


def _file_package(path: Path) -> str:
    """Return the dotted package containing *path*.

    For ``sieval/cli/leaderboard/session.py`` return ``sieval.cli.leaderboard``.
    For ``sieval/core/__init__.py`` return ``sieval.core``. Empty string if
    *path* is outside the sieval package — including repo-level sibling
    directories like ``scripts/`` or ``tests/`` under a repo dir named
    ``sieval``.
    """
    parts = path.parts
    idx = _sieval_root_index(parts)
    if idx is None:
        return ""
    if idx + 1 < len(parts) and parts[idx + 1] in _SIEVAL_OUTER_SIBLINGS:
        return ""
    return ".".join(parts[idx:-1])


def _is_within_subtree(file_pkg: str, root: str) -> bool:
    """Return True if *file_pkg* equals *root* or descends from it."""
    if not root:
        return False
    return file_pkg == root or file_pkg.startswith(root + ".")


def _first_private_component(module_parts: list[str]) -> int | None:
    """Return the index of the first leading-underscore component (non-dunder),
    or None if none."""
    for i, part in enumerate(module_parts):
        if part.startswith("_") and not part.startswith("__"):
            return i
    return None


def _subtree_violation(
    path: Path,
    lineno: int,
    full_module: str,
    file_pkg: str,
) -> str | None:
    """If *full_module* traverses a private (leading-underscore) segment,
    ensure *file_pkg* is within the owning subtree. Return an error string
    when the access is out-of-subtree, else None."""
    parts = full_module.split(".")
    priv_idx = _first_private_component(parts)
    if priv_idx is None:
        return None
    owner = ".".join(parts[:priv_idx])
    priv_fq = ".".join(parts[: priv_idx + 1])
    if _is_within_subtree(file_pkg, owner):
        return None
    return (
        f"{path}:{lineno}: "
        f"import from private module {priv_fq!r} outside its subtree "
        f"(importer at {file_pkg!r}, subtree rooted at {owner!r})"
    )


def _check_layer_imports(path: Path, tree: ast.AST) -> list[str]:
    """Layer-boundary check (existing behavior)."""
    forbidden = FORBIDDEN.get(_get_layer(path) or "")
    if not forbidden:
        return []
    errors: list[str] = []
    layer = _get_layer(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "sieval" and parts[1] in forbidden:
                    errors.append(
                        f"{path}:{node.lineno}: "
                        f"{layer}/ must not import {parts[1]}/ "
                        f"({alias.name})"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if len(parts) >= 2 and parts[0] == "sieval" and parts[1] in forbidden:
                errors.append(
                    f"{path}:{node.lineno}: "
                    f"{layer}/ must not import {parts[1]}/ "
                    f"({node.module})"
                )
    return errors


def _check_private_access(path: Path, tree: ast.AST) -> list[str]:
    """Private-name + protected-module subtree check (CLAUDE.md Import Policy).

    Applies to sieval/ package files and to scripts/ tooling. Tests are the
    carve-out: scripts/ may reach into any public API, but may not cross the
    private line any more than production code can.
    """
    in_sieval = _get_layer(path) is not None
    in_scripts = "scripts" in path.parts and not in_sieval
    if not (in_sieval or in_scripts):
        return []
    # Tests carve-out: the common layout is `<repo>/tests/…`, which lacks a
    # `sieval` segment and short-circuits above. This branch handles the
    # absolute-path case where the repo dir itself is named `sieval` — then
    # `_get_layer` returns "tests" and we must still exempt the file.
    if "tests" in path.parts:
        return []

    file_pkg = _file_package(path)
    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Relative imports are same-package by construction; they are the
            # carve-out that makes the `_base.py` sibling pattern legal.
            if node.level > 0:
                continue
            module = node.module or ""
            # Cover both `from sieval import _x` and `from sieval.pkg import …`.
            if module != "sieval" and not module.startswith("sieval."):
                continue

            # Rule: cross-module import of a private NAME is a smell.
            for alias in node.names:
                name = alias.name
                if name.startswith("_") and not name.startswith("__"):
                    errors.append(
                        f"{path}:{node.lineno}: "
                        f"import of private name {name!r} from {module!r} "
                        f"in production code — promote it or redesign the call site"
                    )

            # Rule: protected modules visible only within their own subtree.
            # Check the *module path only*, not module + alias.name composed —
            # alias names are covered by the private-name rule above; we must
            # not conflate a private *variable* name (`import _SOMETHING` from a
            # public module) with a private *module* in the import path.
            err = _subtree_violation(path, node.lineno, module, file_pkg)
            if err:
                errors.append(err)

        elif isinstance(node, ast.Import):
            # `import sieval.x._foo` style
            for alias in node.names:
                if not alias.name.startswith("sieval."):
                    continue
                err = _subtree_violation(path, node.lineno, alias.name, file_pkg)
                if err:
                    errors.append(err)

    return errors


def _check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    errors = _check_layer_imports(path, tree)
    errors.extend(_check_private_access(path, tree))
    return errors


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--stdin" in args:
        files = [line.strip() for line in sys.stdin if line.strip()]
    else:
        files = args
    if not files:
        return 0

    all_errors: list[str] = []
    for f in files:
        p = Path(f)
        if p.suffix == ".py":
            all_errors.extend(_check_file(p))

    for err in all_errors:
        print(err, file=sys.stderr)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
