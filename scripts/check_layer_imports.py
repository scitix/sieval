"""
Pre-commit hook: enforce sieval import policy.

Four categories of check:

1. **Layer boundary imports** — each layer has a hard-coded set of sibling
   layers it must not import from. Current map:

       cli/          → orchestration layer, depends on all modules
       infer/        → can depend on core; NOT on tasks/datasets
       tasks/        → depends on core + datasets + community
       datasets/     → depends on core + community
       core/         → zero upper-layer dependencies (independently publishable)
       community/    → third-party evaluation adaptations (used by tasks/datasets)

1b. **Sub-package boundary imports** — the same rule at dotted granularity.
   Check 1 is layer-granular, so it cannot express an edge *inside* a layer.

2. **Private-access discipline** (encodes CLAUDE.md `## Import Policy`):

   * Imports imply public API — a cross-module ``from sieval.x.y import _foo``
     in production code is a smell. Flagged unless relative (same-package)
     or under ``tests/`` (the explicit carve-out).
   * Private modules (``_*.py``) are **protected** — only their own package
     subtree may reach into them. Peer-subpackage access or out-of-subtree
     access is flagged.

3. **Relative-import scope** (the other half of CLAUDE.md `## Import Policy`:
   "Same package: relative imports. Cross-package: absolute imports."):

   * ``from .sibling import X`` (level 1) is same-package — always fine.
   * ``from ..parent import X`` (level >= 2) escapes the package and is
     therefore a cross-package import written relatively. Flagged; use the
     absolute ``from sieval.a.b import X`` form.

**Relative imports are resolved to absolute (``_absolute_module``) before any
rule runs**, so a violation written relatively is diagnosed as what it is
rather than only as an import-style error. Holes this closed:

* Check 2's carve-out was written for same-package relative imports but
  implemented as ``level > 0``, so ``from ..peer import _foo`` slipped past the
  private-name rule. Narrowing it to ``level == 1`` alone was still too wide —
  a *dotted* level-1 module walks DOWN into a child subpackage, so
  ``from .sub._hidden import X`` escaped as well. ``_check_private_access``
  documents the residual limit.
* Check 1 matched on ``node.module`` alone, so ``from ...tasks import x`` in
  ``core/`` went unreported — as did the absolute ``from sieval import
  tasks``, which names the layer as an alias rather than in the module path.

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

# dotted sub-package -> sub-packages it must NOT import. FORBIDDEN is keyed on a
# single segment, so it cannot express an edge *inside* a layer: `cli/` depends
# on every layer, which says nothing about its own sub-packages reaching into
# each other. `sieval.cli.resolution` exists to remove the edge below.
FORBIDDEN_SUBPACKAGE: dict[str, set[str]] = {
    "sieval.cli.infer": {"sieval.cli.leaderboard"},
}

# Guard: a non-dotted entry can never match a resolved module, so the rule would
# look enforced while being dead. Same failure mode as _layer_sibling_collision.
_malformed_subpackage_rules = {
    pkg
    for pkg in (
        *FORBIDDEN_SUBPACKAGE,
        *(t for targets in FORBIDDEN_SUBPACKAGE.values() for t in targets),
    )
    if not pkg.startswith("sieval.")
}
if _malformed_subpackage_rules:
    raise RuntimeError(
        f"FORBIDDEN_SUBPACKAGE entries must be dotted 'sieval.' sub-packages; "
        f"got {sorted(_malformed_subpackage_rules)}."
    )

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


def _resolve_relative(file_pkg: str, level: int, module: str) -> str | None:
    """Resolve a relative import to its absolute dotted module.

    ``level`` follows :class:`ast.ImportFrom` semantics: 1 is the current
    package, each extra level strips one trailing component. Returns ``None``
    when *file_pkg* is unknown or the level walks above the package root.
    """
    if not file_pkg:
        return None
    parts = file_pkg.split(".")
    strip = level - 1
    if strip >= len(parts):
        return None
    base = parts[: len(parts) - strip] if strip else parts
    return ".".join([*base, module]) if module else ".".join(base)


def _absolute_module(node: ast.ImportFrom, file_pkg: str) -> str:
    """Return the absolute dotted module *node* imports from, or ``""``.

    Absolute imports (level 0) are returned as written; relative ones are
    resolved against *file_pkg* so every rule sees the same absolute form.
    Without this, ``node.module`` is only the bare tail (``"tasks"`` for
    ``from ...tasks import x``) and each rule short-circuits on its ``sieval.``
    prefix test. Empty when the level walks above the package root.
    """
    if node.level == 0:
        return node.module or ""
    return _resolve_relative(file_pkg, node.level, node.module or "") or ""


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
    """Layer-boundary check: flag imports of a layer this file must not reach.

    Matches all three shapes a forbidden layer can be named in: the module path
    (``import sieval.tasks`` / ``from sieval.tasks import X``, relative form
    included via ``_absolute_module``) and the imported alias
    (``from sieval import tasks``).
    """
    layer = _get_layer(path)
    forbidden = FORBIDDEN.get(layer or "")
    if not forbidden:
        return []
    file_pkg = _file_package(path)
    errors: list[str] = []
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
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, file_pkg)
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "sieval" and parts[1] in forbidden:
                errors.append(
                    f"{path}:{node.lineno}: "
                    f"{layer}/ must not import {parts[1]}/ "
                    f"({module})"
                )
            elif module == "sieval":
                # `from sieval import tasks` names the layer as an imported
                # alias, not in the module path — same violation, different
                # shape. Reachable relatively too (`from ... import tasks`),
                # which is what `_check_relative_scope` suggests as the fix.
                for alias in node.names:
                    if alias.name in forbidden:
                        errors.append(
                            f"{path}:{node.lineno}: "
                            f"{layer}/ must not import {alias.name}/ "
                            f"(sieval.{alias.name})"
                        )
    return errors


def _forbidden_subpackages_for(file_pkg: str) -> set[str]:
    """Return every sub-package *file_pkg* must not import.

    Rules inherit down the subtree: one on ``sieval.cli.infer`` also binds
    ``sieval.cli.infer.sub``.
    """
    targets: set[str] = set()
    for root, forbidden in FORBIDDEN_SUBPACKAGE.items():
        if _is_within_subtree(file_pkg, root):
            targets |= forbidden
    return targets


def _check_subpackage_imports(path: Path, tree: ast.AST) -> list[str]:
    """Sub-package boundary check — check 1 at dotted granularity.

    Matches the three shapes ``_check_layer_imports`` does: module path,
    relative form (via ``_absolute_module``), and the imported alias
    (``from sieval.cli import leaderboard``). Tests are exempt, as in
    ``_check_private_access``.
    """
    if "tests" in path.parts:
        return []
    file_pkg = _file_package(path)
    if not file_pkg:
        return []
    targets = _forbidden_subpackages_for(file_pkg)
    if not targets:
        return []

    def _hit(module: str) -> str | None:
        """Return the forbidden root *module* falls under, if any."""
        for target in sorted(targets):
            if _is_within_subtree(module, target):
                return target
        return None

    def _error(lineno: int, target: str, written: str) -> str:
        return (
            f"{path}:{lineno}: "
            f"{file_pkg} must not import {target} ({written}) — "
            f"move the shared name into a leaf both sub-packages can import"
        )

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _hit(alias.name)
                if target:
                    errors.append(_error(node.lineno, target, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, file_pkg)
            if not module:
                continue
            target = _hit(module)
            if target:
                errors.append(_error(node.lineno, target, module))
                continue
            # `from sieval.cli import leaderboard` — the target is the imported
            # name, not the module path. Same violation, different shape.
            for alias in node.names:
                composed = f"{module}.{alias.name}"
                target = _hit(composed)
                if target:
                    errors.append(_error(node.lineno, target, composed))
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
            # Carve-out: a level-1 relative import with an UNDOTTED module is
            # same-package, which is what makes the `_base.py` sibling pattern
            # legal. Both other shapes escape the package and must still reach
            # the rules below:
            #   * level >= 2 walks UP out of the package (`from .._x import _y`);
            #     `_check_relative_scope` also rejects it outright, but the
            #     private rules must see it if that check is ever relaxed.
            #   * a dotted level-1 module walks DOWN into a child subpackage —
            #     `from .sub._hidden import X` resolves to
            #     `sieval.pkg.sub._hidden`, whose owning subtree is
            #     `sieval.pkg.sub`. The importer at `sieval.pkg` is an ancestor,
            #     not a descendant, so that access is out-of-subtree.
            # Residual limit: an undotted level-1 module naming a *subpackage*
            # (`from .sub import _priv`) is cross-package too, but is
            # syntactically identical to a sibling *module* (`from .mod import
            # _priv`) — telling them apart needs a filesystem lookup, which
            # would make the verdict depend on checkout completeness. Left
            # exempt on purpose; the dotted form above is where a private
            # module segment can actually appear mid-path.
            if node.level == 1 and "." not in (node.module or ""):
                continue
            module = _absolute_module(node, file_pkg)
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


def _check_relative_scope(path: Path, tree: ast.AST) -> list[str]:
    """Reject relative imports that escape their own package (level >= 2).

    CLAUDE.md `## Import Policy`: "Same package: relative imports. Cross-package:
    absolute imports." A ``from ..parent import X`` is a cross-package import
    written relatively, so it violates the second half of that rule.

    Deliberately *not* flagged: a dotted level-1 module (``from .sub.mod import
    X``) also crosses into a child package, but reads as a local descent and is
    idiomatic enough that banning it outright buys little. It is still resolved
    to absolute for checks 1 and 2, so the rules that matter — layer boundaries
    and private-module protection — see through it either way. This check is the
    style half only.

    Scoped to the sieval package only: ``scripts/`` files are standalone modules,
    not a package, so a relative import there fails at runtime and needs no
    lint. (The pre-commit hook feeds both trees; this check narrows on purpose.)
    """
    if _get_layer(path) is None:
        return []
    if "tests" in path.parts:
        return []

    file_pkg = _file_package(path)
    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level >= 2:
            written = "." * node.level + (node.module or "")
            absolute = _resolve_relative(file_pkg, node.level, node.module or "")
            fix = f"; use `from {absolute} import ...`" if absolute else ""
            errors.append(
                f"{path}:{node.lineno}: "
                f"cross-package relative import {written!r} — relative imports "
                f"are for the same package only, use absolute across packages"
                f"{fix}"
            )
    return errors


def _check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    errors = _check_layer_imports(path, tree)
    errors.extend(_check_subpackage_imports(path, tree))
    errors.extend(_check_private_access(path, tree))
    errors.extend(_check_relative_scope(path, tree))
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
