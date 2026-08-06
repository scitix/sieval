"""
Preflight checks for sieval project health.

Validates links, dependencies, tasks, datasets, layer boundaries, and version
consistency. Designed to run locally or in CI before merging.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import argparse
import ast
import dataclasses
import importlib
import json
import re
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from sieval.core.datasets.meta import DatasetMeta

# Check THIS checkout, which is also the tree the path-based checks walk.
# `python scripts/x.py` puts scripts/ on sys.path[0] but never the repo root, so
# the registry imports below otherwise resolve through the editable install —
# from a worktree, that validates another branch while reporting on this one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Known import-name → package-name mismatches
_IMPORT_TO_PACKAGE: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "Bio": "biopython",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "Levenshtein": "levenshtein",
    "rouge_score": "rouge-score",
    "bert_score": "bert-score",
    "sentence_transformers": "sentence-transformers",
    "math_verify": "math-verify",
}

_URL_PATTERN = re.compile(r"https?://[^\s\)\]>\"'`{}]+")
_MD_RELATIVE_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_GH_NON_PERMANENT = re.compile(r"github\.com/[^/]+/[^/]+/blob/(main|master|develop)/")

_MAX_DRIFT_DETAILS = 20

_TASK_FILE_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*_(\d+|k)shot_(gen|base_gen|ppl|clp|llmjudge_gen)\.py$"
)
_DATASET_SUFFIX_PATTERN = re.compile(r"(Dataset|DatasetSample|CSVSample)$")

# A parameter name that is itself a word for "number of few-shot examples":
# n_shot, n_shots, num_shots, nshot, fewshot, few_shot, shot_count, fewshot_k.
# Deliberately anchored, so compounds naming a different noun — fewshot_split,
# fewshot_seed, fewshot_as_multiturn — are not shot counts and do not match.
_SHOT_COUNT_PARAM = re.compile(r"^(?:n_?|num_?)?(?:few_?)?shots?(?:_count|_k)?$")


@dataclasses.dataclass
class CheckResult:
    """Outcome of a single preflight check."""

    status: Literal["PASS", "FAIL", "WARN", "SKIP"]
    check: str  # check name, e.g. "check_links"
    message: str  # one-line summary
    details: list[str] = dataclasses.field(default_factory=list)


def format_text(results: list[CheckResult]) -> str:
    """Human-readable text output: ``[STATUS] check_name — message``."""
    lines: list[str] = []
    for r in results:
        lines.append(f"[{r.status}] {r.check} — {r.message}")
        for d in r.details:
            lines.append(f"  {d}")
    return "\n".join(lines)


def format_json(results: list[CheckResult]) -> str:
    """Machine-readable JSON array."""
    return json.dumps(
        [dataclasses.asdict(r) for r in results],
        indent=2,
    )


def _dataset_integrity_violations(metas: "list[DatasetMeta]") -> list[str]:
    """Each hf: source must be revision-pinned; each url: source must have a
    checksum. local: sources are exempt. Returns human-readable violations."""
    from sieval.core.datasets.meta import url_path_basename
    from sieval.datasets.downloaders.hf import parse_hf_source

    violations: list[str] = []
    for meta in metas:
        declared = {basename for basename, _ in meta.checksums}
        for src in meta.source:
            if src.startswith("hf:"):
                # A malformed pin (e.g. trailing '@') is itself a violation,
                # not a reason to abort the whole check with a traceback.
                try:
                    pinned = parse_hf_source(src).revision is not None
                except ValueError:
                    pinned = False
                if not pinned:
                    violations.append(f"{meta.name}: hf source not pinned: {src}")
            elif src.startswith("url:"):
                basename = url_path_basename(src[len("url:") :])
                if basename not in declared:
                    violations.append(
                        f"{meta.name}: url source missing checksum: {src}"
                    )
    return violations


def _is_sieval_task(cls: ast.ClassDef) -> bool:
    """Whether *cls* carries the ``@sieval_task`` decorator."""
    for deco in cls.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Name) and target.id == "sieval_task":
            return True
    return False


def _classes_with_own_init(
    tree: ast.Module,
) -> list[tuple[ast.ClassDef, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every class in a task module declaring its own ``__init__``.

    Deliberately *not* narrowed to ``@sieval_task``-decorated classes. A
    decorated task may inherit its constructor from an undecorated base in the
    same package (the ``arc/_base.py`` layout), and keying on the decorator
    misses both ends of that: the subclass declares no ``__init__``, the base
    carries no decorator. The knob-bearing constructor would then go unchecked
    with only a silently lower count to show for it.
    """
    out: list[tuple[ast.ClassDef, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        init = _init_of(node)
        if init is not None:
            out.append((node, init))
    return out


def _init_of(cls: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the class's own ``__init__``, or None if it does not define one."""
    for item in cls.body:
        if (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "__init__"
        ):
            return item
    return None


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Every declared parameter name, in any position."""
    args = fn.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def _n_shot_sources(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[set[str]]:
    """For each ``self.n_shot = <expr>``, the identifiers ``<expr>`` reads.

    A bare ``n_shot`` contributes ``n_shot``, ``_normalize_n_shot(n_shot)``
    contributes both — enough to tell which knob feeds the persisted count.
    """
    sources: list[set[str]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        assigns_it = any(
            isinstance(t, ast.Attribute)
            and t.attr == "n_shot"
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
            for t in node.targets
        )
        if not assigns_it:
            continue
        names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        names |= {a.attr for a in ast.walk(node.value) if isinstance(a, ast.Attribute)}
        sources.append(names)
    return sources


#: Rollout keys whose absence is a *runtime* outcome, so `[]` on them is a
#: latent KeyError no type checker reports. Only `prediction` qualifies today:
#: extraction failure is not under the task author's control, and the record
#: builder spells that failure as `None`, which never reaches disk.
_GATED_ROLLOUT_KEYS = frozenset({"prediction"})

#: Every *other* `NotRequired` key on a record TypedDict, with the reason it is
#: different in kind from `prediction`. Listed rather than ignored so a key in
#: neither set fails the check. Covers all five record classes, so a key added to
#: `PromptRecord` / `JudgementRecord` forces the same decision.
_UNGATED_RECORD_KEYS = frozenset(
    {
        # Absence is an authoring choice, not a runtime outcome: the task that
        # reads `r["extra"]["grade"]` is the same task that wrote that extra,
        # unconditionally. And every read is *nested*, so the mechanical `.get()`
        # fix would turn a KeyError into `NoneType is not subscriptable` —
        # strictly worse. Needs a per-site pass, not a sweep.
        "extra",
        # Same authoring-choice argument. "Absent is not zero" (RolloutJudgement),
        # so a `.get()` returning None here would silently become a wrong metric
        # rather than a loud failure.
        "score",
        "metrics",
        # Absence is an authoring *property*: None iff the task's ground truth is
        # a procedure (a test suite, a rubric), which is fixed per task. Shares
        # `prediction`'s shape on JudgementRecord, but the one `[]` read (ruler's
        # report) consumes it as an iterable, so `.get()` would swap KeyError for
        # `list(None)` TypeError — `extra`'s trap again. What it needs is a
        # durable signal of *which* absence this is: scitix/sieval#71.
        "reference",
    }
)


#: Builtins that pass a rollout list through to the loop, directly or inside a
#: per-item tuple. Unwrapped so `for i, r in enumerate(post["rollouts"])` -- the
#: natural way to write feedback needing the rollout index -- is not a blind spot.
_ROLLOUT_ITER_WRAPPERS = frozenset({"enumerate", "zip", "reversed", "list", "sorted"})


def _rollout_container(node: ast.expr) -> bool:
    """True for an expression that yields a record's rollout list.

    ``<record>["rollouts"]``, ``<maybe-record>.get("rollouts", ...)``, or either
    of those behind a pass-through builtin (:data:`_ROLLOUT_ITER_WRAPPERS`) or a
    walrus.
    """
    if isinstance(node, ast.NamedExpr):
        return _rollout_container(node.value)
    if isinstance(node, ast.Subscript):
        key = node.slice
        return isinstance(key, ast.Constant) and key.value == "rollouts"
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ROLLOUT_ITER_WRAPPERS:
            return any(_rollout_container(arg) for arg in node.args)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            first = node.args[0]
            return isinstance(first, ast.Constant) and first.value == "rollouts"
    return False


def _iteration_target_names(target: ast.expr) -> set[str]:
    """Names bound by a ``for``/comprehension target: ``r``, or ``i, r``.

    A tuple target binds every ``Name`` element, not just the rollout one. That
    over-approximates ``enumerate``/``zip``, but misfires only if something
    subscripts a non-rollout element with a *gated* key -- and no gated key is
    plausible on an index or a judgement.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {e.id for e in target.elts if isinstance(e, ast.Name)}
    return set()


def _rollout_bound_names(tree: ast.Module) -> set[str]:
    """Names bound by iterating a rollout container (`for r in post["rollouts"]`).

    This is what keeps the check structural rather than name-based. A task-local
    dict that merely happens to carry a ``prediction`` key -- t_eval's
    ``datum["prediction"]``, whose sibling is ``ground_truth`` -- is not a record
    and must not be flagged.

    Not followed, because each needs dataflow rather than a syntactic walk:
    ``rs = post["rollouts"]``, ``r = post["rollouts"][0]``, and a helper taking a
    ``RolloutPrediction`` parameter. Nor ``.pop(<gated key>)`` -- same KeyError,
    but mutating a hydrated record is not a pattern here.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        pairs: list[tuple[ast.expr, ast.expr]] = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            pairs.append((node.target, node.iter))
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            pairs.extend((gen.target, gen.iter) for gen in node.generators)
        for target, iterable in pairs:
            if _rollout_container(iterable):
                names |= _iteration_target_names(target)
    return names


def _gated_rollout_subscripts(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Every ``<rollout>[<gated key>]`` in *tree*, as ``(lineno, key, source)``."""
    bound = _rollout_bound_names(tree)
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if not (isinstance(key, ast.Constant) and key.value in _GATED_ROLLOUT_KEYS):
            continue
        base = node.value
        is_rollout = (isinstance(base, ast.Name) and base.id in bound) or (
            isinstance(base, ast.Subscript) and _rollout_container(base.value)
        )
        if is_rollout:
            out.append((node.lineno, str(key.value), ast.unparse(node)))
    return out


#: The record TypedDicts whose ``NotRequired`` keys must all be classified. A key
#: left out is a key nobody had to think about -- how `prediction` drifted.
_RECORD_CLASSES = (
    "PromptRecord",
    "RolloutPrediction",
    "PredictionRecord",
    "RolloutJudgement",
    "JudgementRecord",
)


def _not_required_record_keys(records_py: Path) -> set[str] | None:
    """``NotRequired`` field names on the record TypedDicts, or None if unreadable."""
    try:
        tree = ast.parse(records_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name in _RECORD_CLASSES):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and ast.unparse(stmt.annotation).startswith("NotRequired[")
            ):
                keys.add(stmt.target.id)
    return keys


def _docstring_constant_ids(cls: ast.ClassDef) -> set[int]:
    """``id()`` of every docstring Constant in *cls* (its own and its methods').

    Prose is not evidence of behaviour. Without this, a class whose docstring
    merely *mentions* ``pass@1`` satisfies :func:`_computes_pass_at_k`, so a
    ``k`` parameter that is really a few-shot count passes rule 3 — the precise
    regression rule 3 exists to catch. Metric *keys* (``f"pass@{k}"`` built in a
    report dict) are real evidence and stay counted.
    """
    ids: set[int] = set()
    for node in ast.walk(cls):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _computes_pass_at_k(cls: ast.ClassDef) -> bool:
    """Whether the class body computes a pass@k metric.

    Docstrings do not count — see :func:`_docstring_constant_ids`.
    """
    docstrings = _docstring_constant_ids(cls)
    for node in ast.walk(cls):
        if isinstance(node, ast.Name) and node.id.endswith("pass_at_k"):
            return True
        if isinstance(node, ast.Attribute) and node.attr.endswith("pass_at_k"):
            return True
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "pass@" in node.value
            and id(node) not in docstrings
        ):
            return True
    return False


@dataclasses.dataclass
class LockDrift:
    """Version changes to already-locked packages, split by justification."""

    justified: list[str] = dataclasses.field(default_factory=list)
    unjustified: list[str] = dataclasses.field(default_factory=list)
    requires_python_changed: bool = False


def _load_toml(text: str) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def _truncated(entries: list[str]) -> list[str]:
    """*entries* capped for console output, with a count of what was dropped."""
    shown = entries[:_MAX_DRIFT_DETAILS]
    hidden = len(entries) - len(shown)
    return shown + ([f"... and {hidden} more"] if hidden else [])


def _parse_lock(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse a ``pdm.lock`` into locked versions and dependency edges, keyed by
    canonical package name.

    One package can hold several entries — one per extras combination, and one
    per target in a multi-target lock. They merge into a single node, and
    versions that disagree join into ``"2.2.0/2.3.0"``: sorting makes the value
    independent of entry order, so a reordered lock reads as unchanged while a
    move in *any* entry still shows up.
    """
    found: dict[str, set[str]] = {}
    edges: dict[str, list[str]] = {}
    for pkg in _load_toml(text).get("package", []):
        raw_name, version = pkg.get("name"), pkg.get("version")
        if not isinstance(raw_name, str) or not isinstance(version, str):
            continue
        name = canonicalize_name(raw_name)
        found.setdefault(name, set()).add(version)
        edges.setdefault(name, []).extend(
            d for d in pkg.get("dependencies", []) if isinstance(d, str)
        )
    return {name: "/".join(sorted(v)) for name, v in found.items()}, edges


def _declared_entries(pyproject: dict) -> list[str]:
    """Every requirement string declared in ``pyproject.toml``, across
    ``project.dependencies``, each optional-dependency group, and each PEP 735
    dependency group (whose members may be ``{include-group = "..."}`` tables
    rather than strings)."""
    project = pyproject.get("project", {})
    entries: list[object] = list(project.get("dependencies", []))
    for deps in project.get("optional-dependencies", {}).values():
        entries.extend(deps)
    for deps in pyproject.get("dependency-groups", {}).values():
        entries.extend(deps)
    return [entry for entry in entries if isinstance(entry, str)]


def _imposed_specifiers(
    edges: dict[str, list[str]], pyproject: dict, trusted: set[str]
) -> dict[str, list[str]]:
    """Canonical package name → every version specifier imposed on it by
    ``pyproject.toml`` and by the lock entries named in *trusted*.

    Only entries that did not themselves move are trustworthy. A bare re-lock
    shifts the whole graph at once, and a package that moved for no reason will
    then happily "require" the new version of everything beneath it — so
    trusting a moved entry lets a drift justify itself. Measured on the one real
    incident: trusting every entry excused 13 of 71 moves; trusting only the
    unmoved ones excused none.

    Markers are ignored: pdm resolves one lock for the whole interpreter range,
    so a marker-gated edge it chose to record is part of that resolution.
    """
    out: dict[str, list[str]] = {}
    groups = [_declared_entries(pyproject)]
    groups.extend(deps for name, deps in edges.items() if name in trusted)
    for group in groups:
        for entry in group:
            try:
                req = Requirement(entry)
            except InvalidRequirement:
                continue
            out.setdefault(canonicalize_name(req.name), []).append(str(req.specifier))
    return out


def _satisfies(version: str, specifiers: list[str]) -> bool:
    """Whether *version* meets every specifier in *specifiers*. A multi-entry
    node (``"2.2.0/2.3.0"``, see `_parse_lock`) must meet them at every version:
    one entry violating a specifier is enough to make the node untenable."""
    try:
        versions = [Version(v) for v in version.split("/")]
    except InvalidVersion:
        return False
    for spec in specifiers:
        try:
            allowed = SpecifierSet(spec, prereleases=True)
        except InvalidSpecifier:
            return False
        if not all(v in allowed for v in versions):
            return False
    return True


def lock_drift(
    base_lock: str, cand_lock: str, base_pyproject: str, cand_pyproject: str
) -> LockDrift:
    """Classify every version change between two ``pdm.lock`` snapshots.

    A change to an already-locked package is *justified* only if the old version
    violates a specifier that something which did **not** move imposes on it —
    that is, only if a ``--update-reuse`` re-lock could not have kept the old
    pin. Everything else is collateral from a bare ``pdm lock``: versions nothing
    asked for, with no review trail.

    Asking whether the old pin *could* have survived, rather than which
    declaration changed, is what stops a new requirement from excusing packages
    it never constrained — being reachable from one is not justification.

    One deliberate gap: a package raised only because something below it needed a
    higher floor has nothing imposing that floor on it, so it reads as
    unjustified. Declaring its floor in ``pyproject.toml`` is both the fix and
    the honest diff.
    """
    base_versions, _ = _parse_lock(base_lock)
    cand_versions, cand_edges = _parse_lock(cand_lock)
    base_proj, cand_proj = _load_toml(base_pyproject), _load_toml(cand_pyproject)

    report = LockDrift(
        requires_python_changed=(
            base_proj.get("project", {}).get("requires-python")
            != cand_proj.get("project", {}).get("requires-python")
        )
    )
    drift = {
        name: (base_versions[name], cand_versions[name])
        for name in base_versions.keys() & cand_versions.keys()
        if base_versions[name] != cand_versions[name]
    }
    if not drift:
        return report

    entries = {name: f"{name} {old} -> {new}" for name, (old, new) in drift.items()}
    if report.requires_python_changed:
        # pdm resolves against the whole interpreter range, so moving *either*
        # bound re-resolves the graph: raising the floor drops releases that
        # only supported the old one, widening the ceiling demands support for a
        # Python no locked version had to cover. No per-package intent to check.
        report.justified = [entries[name] for name in sorted(entries)]
        return report

    imposed = _imposed_specifiers(cand_edges, cand_proj, set(cand_edges) - set(drift))
    for name in sorted(entries):
        forced = not _satisfies(base_versions[name], imposed.get(name, []))
        bucket = report.justified if forced else report.unjustified
        bucket.append(entries[name])
    return report


class PreflightRunner:
    """Orchestrates preflight checks."""

    ALL_CHECKS: list[str] = [
        "check_links",
        "check_deps",
        "check_dep_coverage",
        "check_tasks",
        "check_task_shot_knobs",
        "check_record_key_access",
        "check_datasets",
        "check_imports",
        "check_examples",
        "check_meta_index_sync",
        "check_version",
    ]

    def __init__(self, level: str = "quick", project_root: Path | None = None):
        self.level = level
        self.project_root = project_root or Path(__file__).resolve().parent.parent

    # -- helpers ---------------------------------------------------------------

    def _git_tracked_files(self, *suffixes: str) -> list[Path]:
        """Return tracked files filtered by suffix(es), using git ls-files.

        Falls back to rglob if git is unavailable (e.g. in tmp_path tests).
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0 and result.stdout:
                paths = [
                    self.project_root / p
                    for p in result.stdout.split("\0")
                    if p and any(p.endswith(s) for s in suffixes)
                ]
                return sorted(p for p in paths if p.exists())
        except FileNotFoundError:
            pass
        # Fallback: rglob (for tests using tmp_path without git)
        files: list[Path] = []
        for suffix in suffixes:
            files.extend(self.project_root.rglob(f"*{suffix}"))
        return sorted(files)

    # -- individual checks -----------------------------------------------------

    def _extract_urls_from_md(self, filepath: Path) -> list[tuple[str, int]]:
        """Return (url, line_number) pairs from a markdown file."""
        urls: list[tuple[str, int]] = []
        for i, line in enumerate(filepath.read_text(encoding="utf-8").splitlines(), 1):
            # Skip lines with template placeholders — URLs extracted from
            # such lines are truncated fragments (e.g. "compare/v" from
            # "compare/v{prev}...vX.Y.Z").
            if "{" in line:
                continue
            for match in _URL_PATTERN.finditer(line):
                urls.append((match.group(0).rstrip(".,;:"), i))
        return urls

    def _extract_urls_from_docstrings(self, filepath: Path) -> list[tuple[str, int]]:
        """Return (url, line_number) pairs from Python docstrings."""
        urls: list[tuple[str, int]] = []
        try:
            source = filepath.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(filepath))
        except SyntaxError:
            return urls
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                docstring = ast.get_docstring(node, clean=False)
                if docstring:
                    start_line = node.body[0].lineno if node.body else 1
                    for i, line in enumerate(docstring.splitlines()):
                        for match in _URL_PATTERN.finditer(line):
                            urls.append((match.group(0).rstrip(".,;:"), start_line + i))
        return urls

    def _extract_urls_from_task_registry(
        self,
    ) -> tuple[list[tuple[str, int, Path]], CheckResult | None]:
        """Return (url, line, filepath) for each indexed task's ``reference_impl.url``.

        Reads ``load_index()`` rather than walking the task tree so partial
        dev installs don't WARN on missing task optional deps. ``line=0`` /
        ``filepath=project_root`` are sentinels (the index lacks source
        provenance).
        """
        urls: list[tuple[str, int, Path]] = []
        try:
            from sieval import load_index

            _, tasks = load_index()
        except Exception as exc:
            warn = CheckResult(
                "WARN",
                "check_links",
                f"index load skipped: {exc}",
            )
            return urls, warn
        for meta in tasks:
            if meta.reference_impl is None:
                continue
            urls.append((meta.reference_impl.url, 0, self.project_root))
        return urls, None

    def _extract_relative_links_from_md(
        self, filepath: Path
    ) -> list[tuple[str, str, int]]:
        """Return (link_text, target, line_number) for relative links in markdown."""
        links: list[tuple[str, str, int]] = []
        for i, line in enumerate(filepath.read_text(encoding="utf-8").splitlines(), 1):
            for match in _MD_RELATIVE_LINK.finditer(line):
                target = match.group(2)
                if not target.startswith(("http://", "https://", "#", "mailto:")):
                    links.append((match.group(1), target, i))
        return links

    def check_links(self) -> list[CheckResult]:
        results: list[CheckResult] = []

        # Collect tracked .md and .py files only
        md_files = self._git_tracked_files(".md")
        py_files = [
            f
            for f in self._git_tracked_files(".py")
            if "sieval/" in str(f.relative_to(self.project_root))
        ]

        # 1. Collect all URLs (deduplicated: first occurrence wins)
        all_urls: list[tuple[str, int, Path]] = []
        seen_urls: set[str] = set()

        def _add(url: str, line: int, filepath: Path) -> None:
            if url not in seen_urls:
                all_urls.append((url, line, filepath))
                seen_urls.add(url)

        for md_file in md_files:
            for url, line in self._extract_urls_from_md(md_file):
                _add(url, line, md_file)
        for py_file in py_files:
            for url, line in self._extract_urls_from_docstrings(py_file):
                _add(url, line, py_file)

        registry_urls, registry_warn = self._extract_urls_from_task_registry()
        if registry_warn is not None:
            results.append(registry_warn)
        for url, line, filepath in registry_urls:
            _add(url, line, filepath)

        if not md_files and not py_files and not all_urls:
            results.append(
                CheckResult(
                    "SKIP",
                    "check_links",
                    "no markdown files, Python files, or registry URLs found",
                )
            )
            return results

        # 2. Non-permanent GitHub link detection
        non_permanent: list[str] = []
        for url, line, filepath in all_urls:
            if _GH_NON_PERMANENT.search(url):
                rel = filepath.relative_to(self.project_root)
                non_permanent.append(f"{rel}:{line}: {url}")

        if non_permanent:
            results.append(
                CheckResult(
                    "WARN",
                    "check_links",
                    f"{len(non_permanent)} non-permanent GitHub link(s)"
                    " (use commit SHA instead of branch)",
                    non_permanent,
                )
            )

        # 3. Relative link validation
        broken_links: list[str] = []
        for md_file in md_files:
            for link_text, target, line in self._extract_relative_links_from_md(
                md_file
            ):
                target_path = target.split("#")[0]
                if not target_path:
                    continue  # anchor-only
                resolved = (md_file.parent / target_path).resolve()
                if not resolved.exists():
                    rel = md_file.relative_to(self.project_root)
                    broken_links.append(f"{rel}:{line}: [{link_text}]({target})")

        if broken_links:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_links",
                    f"{len(broken_links)} broken relative link(s)",
                    broken_links,
                )
            )

        # 4. Deep mode: HTTP reachability
        if self.level == "deep":
            results.extend(self._check_links_reachability(all_urls))
        else:
            results.append(
                CheckResult(
                    "SKIP",
                    "check_links",
                    "HTTP reachability check skipped (use --level deep)",
                )
            )

        # Summary if no issues
        if not non_permanent and not broken_links:
            results.insert(
                0,
                CheckResult(
                    "PASS",
                    "check_links",
                    f"scanned {len(all_urls)} URL(s) from"
                    f" {len(md_files)} .md + {len(py_files)} .py files",
                ),
            )

        return results

    def _get_own_repo_url(self) -> str | None:
        """Return the GitHub repo base URL (e.g. github.com/scitix/sieval)."""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                url = result.stdout.strip().removesuffix(".git")
                # git@github.com:org/repo → github.com/org/repo
                if url.startswith("git@"):
                    url = url.replace(":", "/").replace("git@", "https://")
                m = re.search(r"github\.com/([^/]+/[^/]+)", url)
                return m.group(0) if m else None
        except FileNotFoundError:
            pass
        return None

    @staticmethod
    def _should_skip_url(url: str) -> bool:
        """URLs that should not be checked for HTTP reachability."""
        # localhost / loopback
        if re.match(r"https?://(localhost|127\.0\.0\.1)(:\d+)?", url):
            return True
        # Template placeholders
        return "{" in url or "}" in url

    def _check_links_reachability(
        self, urls: list[tuple[str, int, Path]]
    ) -> list[CheckResult]:
        """HTTP HEAD check for URL reachability."""
        try:
            import httpx
        except ImportError:
            return [
                CheckResult(
                    "SKIP",
                    "check_links",
                    "httpx not installed, skipping reachability check",
                )
            ]

        import anyio

        own_repo = self._get_own_repo_url()
        unique_urls = [
            url for url in {u for u, _, _ in urls} if not self._should_skip_url(url)
        ]
        unreachable: list[str] = []
        needs_review: list[str] = []

        async def _check_all() -> None:
            sem = anyio.Semaphore(10)
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:

                async def _check_one(url: str) -> None:
                    async with sem:
                        try:
                            resp = await client.head(url)
                            if resp.status_code == 403:
                                needs_review.append(f"{url} (HTTP 403 Forbidden)")
                            elif (
                                resp.status_code == 404 and own_repo and own_repo in url
                            ):
                                needs_review.append(f"{url} (HTTP 404, private repo?)")
                            elif resp.status_code >= 400:
                                unreachable.append(f"{url} (HTTP {resp.status_code})")
                        except (
                            httpx.TimeoutException,
                            httpx.ConnectError,
                            httpx.HTTPError,
                        ) as e:
                            unreachable.append(f"{url} ({type(e).__name__})")

                async with anyio.create_task_group() as tg:
                    for u in unique_urls:
                        tg.start_soon(_check_one, u)

        anyio.run(_check_all)

        checked = len(unique_urls)
        results: list[CheckResult] = []
        if unreachable:
            results.append(
                CheckResult(
                    "WARN",
                    "check_links",
                    f"{len(unreachable)}/{checked} URL(s) unreachable",
                    unreachable,
                )
            )
        if needs_review:
            results.append(
                CheckResult(
                    "WARN",
                    "check_links",
                    f"{len(needs_review)} URL(s) need manual review"
                    " (403/private-repo 404)",
                    needs_review,
                )
            )
        if checked > 0:
            reachable = checked - len(unreachable) - len(needs_review)
            results.append(
                CheckResult(
                    "PASS",
                    "check_links",
                    f"{reachable}/{checked} URL(s) reachable",
                )
            )
        return results

    def _load_pyproject(self) -> dict | None:
        path = self.project_root / "pyproject.toml"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _git_blob(self, rev: str, path: str) -> str | None:
        """Contents of *path* at *rev*, or None if git or the blob is unavailable
        (no repository, shallow clone that never fetched *rev*, tmp_path tests)."""
        try:
            result = subprocess.run(
                ["git", "show", f"{rev}:{path}"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
        except FileNotFoundError:
            return None
        return result.stdout if result.returncode == 0 else None

    def _git_rev(self, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() or None if result.returncode == 0 else None

    def _lock_baseline_rev(self) -> str | None:
        """Revision whose lock the working tree's lock is judged against.

        ``HEAD`` while a lock change is uncommitted — the pre-commit path, where
        the author can still re-lock. Otherwise the merge base with the default
        branch, so a pushed branch is judged against what it forked from, which
        is what CI sees. None when neither resolves (shallow clone, or the
        default branch itself): nothing to compare against is not a violation.
        """
        head_lock = self._git_blob("HEAD", "pdm.lock")
        if head_lock is None:
            return None
        lockfile = self.project_root / "pdm.lock"
        if lockfile.exists() and lockfile.read_text(encoding="utf-8") != head_lock:
            return "HEAD"
        head = self._git_rev("rev-parse", "HEAD")
        for ref in ("origin/main", "main"):
            merge_base = self._git_rev("merge-base", ref, "HEAD")
            if merge_base is not None and merge_base != head:
                return merge_base
        return None

    def _check_lock_drift(self) -> list[CheckResult]:
        """Enforce `.claude/rules/deps.md`: a lock change may move existing
        package versions only where a changed requirement demands it."""
        rev = self._lock_baseline_rev()
        base_lock = self._git_blob(rev, "pdm.lock") if rev else None
        base_pyproject = self._git_blob(rev, "pyproject.toml") if rev else None
        if rev is None or base_lock is None or base_pyproject is None:
            return [
                CheckResult(
                    "SKIP", "check_deps", "lock drift: no baseline revision to compare"
                )
            ]

        label = "HEAD" if rev == "HEAD" else rev[:8]
        report = lock_drift(
            base_lock,
            (self.project_root / "pdm.lock").read_text(encoding="utf-8"),
            base_pyproject,
            (self.project_root / "pyproject.toml").read_text(encoding="utf-8"),
        )
        # This is the one path that excuses drift wholesale, so it has to show
        # its work: a bare count would let any number of moves through unread.
        if report.requires_python_changed and report.justified:
            return [
                CheckResult(
                    "WARN",
                    "check_deps",
                    f"requires-python changed — {len(report.justified)} version "
                    f"change(s) vs {label} unaudited, review them by hand",
                    _truncated(report.justified),
                )
            ]
        if report.unjustified:
            return [
                CheckResult(
                    "FAIL",
                    "check_deps",
                    f"{len(report.unjustified)} locked package(s) drifted vs "
                    f"{label} with no requirement change asking for it",
                    _truncated(report.unjustified)
                    + [
                        "re-lock with `pdm lock --update-reuse`, or declare the "
                        "bump in pyproject.toml so the intent is reviewable"
                    ],
                )
            ]
        return [
            CheckResult(
                "PASS",
                "check_deps",
                f"no unrequested version drift vs {label} "
                f"({len(report.justified)} justified)",
            )
        ]

    def check_deps(self) -> list[CheckResult]:
        results: list[CheckResult] = []

        pyproject = self._load_pyproject()
        if pyproject is None:
            return [CheckResult("FAIL", "check_deps", "pyproject.toml not found")]

        # Check optional-dependencies exist and are non-empty
        optional = pyproject.get("project", {}).get("optional-dependencies", {})
        if not optional:
            results.append(
                CheckResult("WARN", "check_deps", "no optional-dependencies defined")
            )
        else:
            empty_groups = [name for name, deps in optional.items() if not deps]
            if empty_groups:
                results.append(
                    CheckResult(
                        "FAIL",
                        "check_deps",
                        "empty optional-dependency group(s): "
                        f"{', '.join(empty_groups)}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "PASS",
                        "check_deps",
                        f"{len(optional)} optional-dependency groups all non-empty",
                    )
                )

        # Check pdm.lock exists and is non-empty
        lockfile = self.project_root / "pdm.lock"
        if not lockfile.exists():
            results.append(CheckResult("FAIL", "check_deps", "pdm.lock not found"))
        elif lockfile.stat().st_size == 0:
            results.append(CheckResult("FAIL", "check_deps", "pdm.lock is empty"))
        else:
            results.append(
                CheckResult("PASS", "check_deps", "pdm.lock exists and is non-empty")
            )
            results.extend(self._check_lock_drift())

        # Deep mode: dry-run install per group
        if self.level == "deep" and optional:
            for group in optional:
                dr = subprocess.run(
                    ["pdm", "install", "--dry-run", "-G", group],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )
                if dr.returncode != 0:
                    results.append(
                        CheckResult(
                            "FAIL",
                            "check_deps",
                            f"pdm install --dry-run -G {group} failed",
                            dr.stderr.strip().splitlines(),
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            "PASS",
                            "check_deps",
                            f"pdm dry-run OK for group '{group}'",
                        )
                    )

        return results

    def _extract_top_level_imports(self, filepath: Path) -> set[str]:
        """Extract top-level import package names from a Python file using AST."""
        try:
            source = filepath.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(filepath))
        except SyntaxError:
            return set()
        packages: set[str] = set()
        for node in tree.body:  # only top-level
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                packages.add(node.module.split(".")[0])
        return packages

    def _get_declared_packages(self) -> set[str]:
        """Get package names from deps + optional-deps, normalized."""
        pyproject = self._load_pyproject()
        if pyproject is None:
            return set()
        packages: set[str] = set()
        project = pyproject.get("project", {})
        for dep_str in project.get("dependencies", []):
            name = re.split(r"[>=<!\[;]", dep_str)[0].strip()
            packages.add(name.lower().replace("-", "_"))
        for group_deps in project.get("optional-dependencies", {}).values():
            for dep_str in group_deps:
                name = re.split(r"[>=<!\[;]", dep_str)[0].strip()
                packages.add(name.lower().replace("-", "_"))
        return packages

    def check_dep_coverage(self) -> list[CheckResult]:
        declared = self._get_declared_packages()
        if not declared:
            return [
                CheckResult(
                    "WARN",
                    "check_dep_coverage",
                    "no dependencies found in pyproject.toml",
                )
            ]

        stdlib_names = sys.stdlib_module_names
        uncovered: list[str] = []
        scan_prefixes = ["sieval/tasks/", "sieval/datasets/"]
        scan_files = [
            f
            for f in self._git_tracked_files(".py")
            if any(
                str(f.relative_to(self.project_root)).startswith(p)
                for p in scan_prefixes
            )
            and f.name != "__init__.py"
        ]
        for py_file in scan_files:
            imports = self._extract_top_level_imports(py_file)
            for imp in imports:
                if imp in stdlib_names or imp == "sieval" or imp.startswith("_"):
                    continue
                pkg_name = _IMPORT_TO_PACKAGE.get(imp, imp)
                normalized = pkg_name.lower().replace("-", "_")
                if normalized not in declared:
                    rel = py_file.relative_to(self.project_root)
                    uncovered.append(f"{rel}: {imp} (package: {pkg_name})")

        if uncovered:
            return [
                CheckResult(
                    "WARN",
                    "check_dep_coverage",
                    f"{len(uncovered)} import(s) not covered by declared dependencies",
                    uncovered,
                )
            ]
        return [
            CheckResult(
                "PASS",
                "check_dep_coverage",
                "all task/dataset imports covered by declared dependencies",
            )
        ]

    def check_tasks(self) -> list[CheckResult]:
        results: list[CheckResult] = []

        # Step 1: Load the registry
        try:
            tasks_init = self.project_root / "sieval" / "tasks" / "__init__.py"
            if not tasks_init.exists():
                return [
                    CheckResult(
                        "FAIL",
                        "check_tasks",
                        "sieval/tasks/__init__.py not found",
                    )
                ]

            import sieval.tasks as tasks_mod

            export_map = dict(tasks_mod._EXPORT_TO_MODULE)  # type: ignore[unresolved-attribute]  # dynamic module-level var
            results.append(
                CheckResult(
                    "PASS",
                    "check_tasks",
                    f"task registry loaded: {len(export_map)} exports, no duplicates",
                )
            )
        except RuntimeError as e:
            return [CheckResult("FAIL", "check_tasks", f"task registry error: {e}")]
        except Exception as e:
            return [
                CheckResult("FAIL", "check_tasks", f"failed to load task registry: {e}")
            ]

        # Step 2: Try importing each task module
        import_failures: list[str] = []
        import_warnings: list[str] = []
        imported_classes: dict[str, type] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for class_name, module_name in export_map.items():
                fqn = f"sieval.tasks.{module_name}"
                try:
                    mod = importlib.import_module(fqn)
                    cls = getattr(mod, class_name)
                    imported_classes[class_name] = cls
                except ImportError as e:
                    import_warnings.append(f"{class_name} ({fqn}): {e}")
                except Exception as e:
                    import_failures.append(f"{class_name} ({fqn}): {e}")

        if import_failures:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_tasks",
                    f"{len(import_failures)} task(s) failed to import",
                    import_failures,
                )
            )
        if import_warnings:
            results.append(
                CheckResult(
                    "WARN",
                    "check_tasks",
                    f"{len(import_warnings)} task(s) have missing optional deps",
                    import_warnings,
                )
            )
        if not import_failures and not import_warnings:
            results.append(
                CheckResult(
                    "PASS",
                    "check_tasks",
                    f"all {len(export_map)} task modules imported successfully",
                )
            )

        # Step 3: Check tags on imported classes
        missing_tags: list[str] = []
        for class_name, cls in imported_classes.items():
            tags = getattr(cls, "tags", None)
            if not tags:
                missing_tags.append(class_name)

        if missing_tags:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_tasks",
                    f"{len(missing_tags)} task(s) have empty or missing tags",
                    missing_tags,
                )
            )
        elif imported_classes:
            results.append(
                CheckResult(
                    "PASS",
                    "check_tasks",
                    f"all {len(imported_classes)} imported tasks have non-empty tags",
                )
            )

        # Step 4: File naming convention
        all_task_py = [
            f
            for f in self._git_tracked_files(".py")
            if str(f.relative_to(self.project_root)).startswith("sieval/tasks/")
        ]
        bad_names: list[str] = []
        for py_file in all_task_py:
            if py_file.name.startswith("_"):
                continue
            if not _TASK_FILE_PATTERN.match(py_file.name):
                rel = py_file.relative_to(self.project_root)
                bad_names.append(str(rel))

        if bad_names:
            results.append(
                CheckResult(
                    "WARN",
                    "check_tasks",
                    f"{len(bad_names)} task file(s) don't match naming convention",
                    bad_names,
                )
            )
        else:
            results.append(
                CheckResult(
                    "PASS",
                    "check_tasks",
                    "all task files follow naming convention",
                )
            )

        return results

    def check_task_shot_knobs(self) -> list[CheckResult]:
        """Verify the few-shot knob is spelled ``n_shot`` and reaches meta.json.

        ``meta.json`` records the shot count a run used by reading
        ``Task.n_shot``, which ``@sieval_task`` seeds on the class with the
        declared value. A constructor that takes a shot-count argument and
        stores it anywhere *but* ``self.n_shot`` therefore leaves the class
        value standing, and nothing at runtime can tell — the run directory
        reports a number the run never used. So the wiring is checked here.

        Three rules, AST-only so a task whose optional deps are absent is still
        covered:

        1. a shot-count parameter is spelled ``n_shot``, nothing else;
        2. a constructor accepting ``n_shot`` assigns ``self.n_shot``;
        3. ``k`` is the ``k`` in ``pass@k`` in every task that takes one — the
           metric's parameter, not the sampling budget (that is ``n``, and
           ``k <= n``) — so a task accepting ``k`` must compute a pass@k
           metric, and ``self.n_shot`` may never be fed from it. This is what
           stops ``k`` from re-acquiring a second meaning.

        Rules 1 and 2 bind *every* constructor under ``sieval/tasks/``, not only
        the decorated classes — see :func:`_classes_with_own_init`. Rule 3 reads
        the metric off the class body, so it can only be judged where that body
        is the task's: an undecorated base's ``pass@k`` may well be computed by
        the subclass, so applying it there would be a false positive.
        """
        # Recursive, matching check_tasks' naming sweep: a benchmark with >= 5
        # task files lives in a subdirectory (sieval/tasks/CLAUDE.md). Subpackage
        # __init__.py files are empty by convention, so skipping them by name
        # costs no coverage.
        py_files = [
            f
            for f in self._git_tracked_files(".py")
            if str(f.relative_to(self.project_root)).startswith("sieval/tasks/")
            and f.name != "__init__.py"
        ]
        if not py_files:
            return [CheckResult("SKIP", "check_task_shot_knobs", "no task modules")]

        violations: list[str] = []
        checked = 0
        for py_file in py_files:
            rel = py_file.relative_to(self.project_root)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(
                        py_file.read_text(encoding="utf-8"), filename=str(py_file)
                    )
            except (OSError, SyntaxError) as e:
                violations.append(f"{rel}: could not parse ({e})")
                continue

            for cls, init in _classes_with_own_init(tree):
                checked += 1
                where = f"{rel}:{init.lineno} {cls.name}"
                params = _param_names(init)

                for p in params:
                    if p != "n_shot" and _SHOT_COUNT_PARAM.match(p):
                        violations.append(
                            f"{where}: shot-count parameter is named {p!r}; "
                            "the repo spells it 'n_shot'"
                        )

                sources = _n_shot_sources(init)
                if "n_shot" in params and not sources:
                    violations.append(
                        f"{where}: takes 'n_shot' but never assigns "
                        "self.n_shot, so the class value @sieval_task(n_shot=...) "
                        "seeded would stand and meta.json would report the "
                        "declared default rather than the count this run used"
                    )
                for names in sources:
                    if names & {"k", "_k"}:
                        violations.append(
                            f"{where}: feeds self.n_shot from 'k', which "
                            "is the k in pass@k, not a shot count"
                        )

                if (
                    _is_sieval_task(cls)
                    and "k" in params
                    and not _computes_pass_at_k(cls)
                ):
                    violations.append(
                        f"{where}: takes 'k' but computes no pass@k metric; "
                        "'k' is reserved for the k in pass@k — spell a "
                        "few-shot knob 'n_shot'"
                    )

        if violations:
            return [
                CheckResult(
                    "FAIL",
                    "check_task_shot_knobs",
                    f"{len(violations)} shot-knob violation(s)",
                    violations,
                )
            ]
        return [
            CheckResult(
                "PASS",
                "check_task_shot_knobs",
                f"all {checked} task constructor(s) spell and wire the "
                "shot knob correctly",
            )
        ]

    def check_record_key_access(self) -> list[CheckResult]:
        """Forbid ``[]`` on a rollout key whose absence is a runtime outcome.

        ``build_prediction_record`` spells "could not extract" as
        ``prediction=None``, and ``obj_to_dict`` drops ``None``-valued keys, so
        the key is *absent* on disk -- not null, gone. On resume the loader
        hydrates ``postprocess_result`` from disk and hands it to ``feedback``,
        where ``rollout["prediction"]`` raises ``KeyError`` for exactly the
        samples whose extraction failed. The same line is fine on a fresh run,
        which is why this survives every in-memory test.

        Nothing else catches it. The contract is already stated three times over
        -- ``RolloutPrediction.prediction`` is declared ``NotRequired``, its
        docstring says "absent on disk in that case, so read ``extracted``
        instead", and ``test_records.py::TestSerializationRoundTrip`` pins the
        behaviour -- yet neither ``ty`` nor ``mypy --strict`` reports a ``[]``
        subscript of a ``NotRequired`` key (both accept it by design; the typing
        spec leaves that diagnostic optional). A correctly-declared,
        correctly-documented, unit-tested contract with no enforcement at the
        call site is how 39 modules drifted off it while CI stayed green.

        Structural, not name-based: a key is only flagged when its base provably
        comes from a record's ``rollouts`` -- indexed (``post["rollouts"][0]``) or
        iterated (``for r in post["rollouts"]``, including through
        ``enumerate``/``zip``). A task-local dict that merely carries a
        ``prediction`` key is left alone; :func:`_rollout_bound_names` lists what
        is deliberately not followed.

        Flagging is rollout-scoped (:data:`_GATED_ROLLOUT_KEYS`); *classification*
        covers every ``NotRequired`` key on every record TypedDict
        (:data:`_RECORD_CLASSES`), and a key in neither set fails the check, so
        adding one to ``records.py`` forces the decision. The scopes differ on
        purpose: gating a key not read off a rollout (``reference``) would be inert
        and would read as coverage the check does not have.
        """
        records_py = self.project_root / "sieval" / "core" / "tasks" / "records.py"
        declared = _not_required_record_keys(records_py)
        if declared is None:
            # FAIL, not SKIP: records.py is this check's subject, not an optional
            # input. Renaming it would otherwise narrow the gate to nothing with
            # preflight still green.
            return [
                CheckResult(
                    "FAIL",
                    "check_record_key_access",
                    f"{records_py.relative_to(self.project_root)} not readable",
                    [
                        "the record TypedDicts are this check's subject — if they "
                        "moved, point check_record_key_access at the new path "
                        "rather than leaving the gate unenforced"
                    ],
                )
            ]

        unclassified = sorted(declared - _GATED_ROLLOUT_KEYS - _UNGATED_RECORD_KEYS)
        if unclassified:
            return [
                CheckResult(
                    "FAIL",
                    "check_record_key_access",
                    f"{len(unclassified)} unclassified NotRequired record key(s)",
                    [
                        f"records.py declares {k!r} NotRequired on a record but "
                        "check_preflight lists it in neither _GATED_ROLLOUT_KEYS "
                        "nor _UNGATED_RECORD_KEYS — decide whether `[]` on it is "
                        "a latent KeyError and say so in one of the two"
                        for k in unclassified
                    ],
                )
            ]

        py_files = [
            f
            for f in self._git_tracked_files(".py")
            if str(f.relative_to(self.project_root)).startswith("sieval/")
        ]
        if not py_files:
            return [CheckResult("SKIP", "check_record_key_access", "no sieval modules")]

        violations: list[str] = []
        for py_file in py_files:
            rel = py_file.relative_to(self.project_root)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(
                        py_file.read_text(encoding="utf-8"), filename=str(py_file)
                    )
            except (OSError, SyntaxError) as e:
                violations.append(f"{rel}: could not parse ({e})")
                continue
            for lineno, key, source in _gated_rollout_subscripts(tree):
                violations.append(
                    f"{rel}:{lineno}: {source} — {key!r} is NotRequired and "
                    f"absent on disk when it was None, so this raises KeyError "
                    f"on resume; use .get({key!r}) (same value, and the resumed "
                    "path then matches the fresh one)"
                )

        if violations:
            return [
                CheckResult(
                    "FAIL",
                    "check_record_key_access",
                    f"{len(violations)} unsafe rollout-key access(es)",
                    violations,
                )
            ]
        return [
            CheckResult(
                "PASS",
                "check_record_key_access",
                f"no unsafe `[]` access to {sorted(_GATED_ROLLOUT_KEYS)} "
                f"across {len(py_files)} module(s)",
            )
        ]

    def check_datasets(self) -> list[CheckResult]:
        results: list[CheckResult] = []

        # Step 1: Load the registry
        try:
            datasets_init = self.project_root / "sieval" / "datasets" / "__init__.py"
            if not datasets_init.exists():
                return [
                    CheckResult(
                        "FAIL",
                        "check_datasets",
                        "sieval/datasets/__init__.py not found",
                    )
                ]

            import sieval.datasets as datasets_mod

            export_map = dict(datasets_mod._EXPORT_TO_MODULE)  # type: ignore[unresolved-attribute]  # dynamic module-level var
            results.append(
                CheckResult(
                    "PASS",
                    "check_datasets",
                    f"dataset registry loaded: {len(export_map)} exports,"
                    " no duplicates",
                )
            )
        except RuntimeError as e:
            return [
                CheckResult("FAIL", "check_datasets", f"dataset registry error: {e}")
            ]
        except Exception as e:
            return [
                CheckResult(
                    "FAIL",
                    "check_datasets",
                    f"failed to load dataset registry: {e}",
                )
            ]

        # Step 2: Try importing each dataset module
        import_failures: list[str] = []
        import_warnings: list[str] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for class_name, module_name in export_map.items():
                fqn = f"sieval.datasets.{module_name}"
                try:
                    mod = importlib.import_module(fqn)
                    getattr(mod, class_name)
                except ImportError as e:
                    import_warnings.append(f"{class_name} ({fqn}): {e}")
                except Exception as e:
                    import_failures.append(f"{class_name} ({fqn}): {e}")

        if import_failures:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_datasets",
                    f"{len(import_failures)} dataset(s) failed to import",
                    import_failures,
                )
            )
        if import_warnings:
            results.append(
                CheckResult(
                    "WARN",
                    "check_datasets",
                    f"{len(import_warnings)} dataset(s) have missing optional deps",
                    import_warnings,
                )
            )
        if not import_failures and not import_warnings:
            results.append(
                CheckResult(
                    "PASS",
                    "check_datasets",
                    f"all {len(export_map)} dataset modules imported successfully",
                )
            )

        # Step 3: Naming convention
        bad_names = [
            name for name in export_map if not _DATASET_SUFFIX_PATTERN.search(name)
        ]
        if bad_names:
            results.append(
                CheckResult(
                    "WARN",
                    "check_datasets",
                    f"{len(bad_names)} export(s) don't match naming convention",
                    bad_names,
                )
            )
        else:
            results.append(
                CheckResult(
                    "PASS",
                    "check_datasets",
                    "all dataset exports follow naming convention",
                )
            )

        # Step 4: source-integrity policy (hf pinned, url checksummed)
        from sieval.core.datasets.meta import iter_dataset_metas

        integrity = _dataset_integrity_violations(list(iter_dataset_metas()))
        if integrity:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_datasets",
                    f"{len(integrity)} dataset source(s) not pinned/checksummed",
                    integrity,
                )
            )
        else:
            results.append(
                CheckResult(
                    "PASS",
                    "check_datasets",
                    "all dataset sources pinned (hf) / checksummed (url)",
                )
            )

        return results

    def check_examples(self) -> list[CheckResult]:
        """Every ``class:`` under ``datasets:`` / ``tasks:`` in ``examples/*.yaml``
        must resolve to a registered class — catches silent renames that would
        otherwise only surface when a user copy-pastes the template.
        """
        results: list[CheckResult] = []
        examples_dir = self.project_root / "examples"
        if not examples_dir.exists():
            return [
                CheckResult(
                    "SKIP",
                    "check_examples",
                    "no examples/ directory",
                )
            ]

        try:
            import yaml as _yaml
        except ImportError:
            return [
                CheckResult(
                    "FAIL",
                    "check_examples",
                    "PyYAML not available; cannot parse example configs",
                )
            ]

        try:
            import sieval.datasets as datasets_mod
            import sieval.tasks as tasks_mod

            dataset_exports = set(datasets_mod._EXPORT_TO_MODULE.keys())  # type: ignore[unresolved-attribute]
            task_exports = set(tasks_mod._EXPORT_TO_MODULE.keys())  # type: ignore[unresolved-attribute]
        except Exception as e:
            return [
                CheckResult(
                    "FAIL",
                    "check_examples",
                    f"failed to load dataset/task registries: {e}",
                )
            ]

        yaml_files = sorted(examples_dir.rglob("*.yaml"))
        if not yaml_files:
            return [
                CheckResult(
                    "SKIP",
                    "check_examples",
                    "no YAML files under examples/",
                )
            ]

        unresolved: list[str] = []
        parse_errors: list[str] = []
        total_refs = 0

        for yaml_path in yaml_files:
            rel = yaml_path.relative_to(self.project_root)
            try:
                doc = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except Exception as e:
                parse_errors.append(f"{rel}: {e}")
                continue
            if not isinstance(doc, dict):
                continue

            for section_key, registry, kind in (
                ("datasets", dataset_exports, "Dataset"),
                ("tasks", task_exports, "Task"),
            ):
                section = doc.get(section_key)
                if not isinstance(section, dict):
                    continue
                for entry_name, entry in section.items():
                    if not isinstance(entry, dict):
                        continue
                    class_name = entry.get("class")
                    if not class_name:
                        continue
                    total_refs += 1
                    if class_name not in registry:
                        unresolved.append(
                            f"{rel} [{section_key}.{entry_name}]: "
                            f"{kind} class {class_name!r} is not exported from "
                            f"sieval.{section_key}"
                        )

        if parse_errors:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_examples",
                    f"{len(parse_errors)} example file(s) failed to parse",
                    parse_errors,
                )
            )
        if unresolved:
            results.append(
                CheckResult(
                    "FAIL",
                    "check_examples",
                    f"{len(unresolved)} class reference(s) do not resolve",
                    unresolved,
                )
            )
        if not parse_errors and not unresolved:
            results.append(
                CheckResult(
                    "PASS",
                    "check_examples",
                    f"all {total_refs} class reference(s) across "
                    f"{len(yaml_files)} example file(s) resolve",
                )
            )
        return results

    def check_meta_index_sync(self) -> list[CheckResult]:
        """``sieval/meta/index.json`` must match the live registry.

        Silent divergence hides new datasets/tasks from the discovery verbs
        (``sieval dataset list`` / ``task list``) until the next manual sync.
        """
        script = self.project_root / "scripts" / "sync_meta_index.py"
        if not script.exists():
            return [
                CheckResult(
                    "FAIL",
                    "check_meta_index_sync",
                    f"script not found: {script}",
                )
            ]

        result = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )
        if result.returncode == 0:
            return [
                CheckResult(
                    "PASS",
                    "check_meta_index_sync",
                    "sieval/meta/index.json matches the live registry",
                )
            ]

        # sync_meta_index.py writes its reason to stderr via SystemExit.
        message = (result.stderr or result.stdout).strip().splitlines()
        return [
            CheckResult(
                "FAIL",
                "check_meta_index_sync",
                "sieval/meta/index.json is out of date; "
                "run `python scripts/sync_meta_index.py` to regenerate",
                message,
            )
        ]

    def check_imports(self) -> list[CheckResult]:
        script = self.project_root / "scripts" / "check_layer_imports.py"
        if not script.exists():
            return [CheckResult("FAIL", "check_imports", f"script not found: {script}")]

        # Matches the pre-commit hook's `files:` filter in
        # `.pre-commit-config.yaml` (^(sieval|scripts)/). Narrowing here to
        # `sieval/` only would leave the script's `in_scripts` branch untested
        # by preflight while still running in pre-commit — two enforcement
        # surfaces silently diverging.
        #
        # KNOWN divergence, not parity: pre-commit additionally applies the
        # global `exclude: ^(sieval/community/|vendor/)`, so it skips
        # `sieval/community/` while this wrapper checks it. Inert today (every
        # relative import under `community/` is a bare level-1 `from . import
        # x`), but a future vendored drop using `from ..x import y` would pass
        # pre-commit and fail preflight, and the only offered fix would be to
        # edit code kept byte-identical to upstream. Fixing it is a design call
        # — hoisting the exemption into `_check_file` would also drop the
        # private-access check's coverage of `community/`.
        enforced_py = [
            f
            for f in self._git_tracked_files(".py")
            if str(f.relative_to(self.project_root)).startswith(("sieval/", "scripts/"))
        ]
        py_files = "\n".join(str(p) for p in enforced_py)

        result = subprocess.run(
            [sys.executable, str(script), "--stdin"],
            input=py_files,
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )

        if result.returncode == 0:
            return [CheckResult("PASS", "check_imports", "no import-policy violations")]

        errors = [line for line in result.stderr.strip().splitlines() if line.strip()]
        return [
            CheckResult(
                "FAIL",
                "check_imports",
                f"{len(errors)} import-policy violation(s)",
                errors,
            )
        ]

    def _parse_changelog_version(self) -> str | None:
        """Extract the first version from CHANGELOG.md (Keep a Changelog format)."""
        changelog = self.project_root / "CHANGELOG.md"
        if not changelog.is_file():
            return None
        text = changelog.read_text()
        m = re.search(r"^## \[(\d+\.\d+\.\d+)]", text, re.MULTILINE)
        return m.group(1) if m else None

    def _parse_dockerfile_version(self) -> str | None:
        """Extract sieval wheel version from Dockerfile."""
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.is_file():
            return None
        text = dockerfile.read_text()
        m = re.search(r"sieval-(\d+\.\d+\.\d+)-py3-none-any\.whl", text)
        return m.group(1) if m else None

    def _get_latest_git_tag(self) -> str | None:
        """Get the latest git tag, stripping any ``v`` prefix."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root,
            )
            tag = result.stdout.strip()
            return tag.removeprefix("v") if tag else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def check_version(self) -> list[CheckResult]:
        """Check CHANGELOG / git tag / Dockerfile version alignment."""
        results: list[CheckResult] = []
        check = "check_version"

        # 1. Parse CHANGELOG version
        cl_version = self._parse_changelog_version()
        if cl_version is None:
            results.append(
                CheckResult(
                    "FAIL",
                    check,
                    "CHANGELOG.md missing or has no version heading",
                )
            )
            return results

        results.append(
            CheckResult(
                "PASS",
                check,
                f"CHANGELOG version: {cl_version}",
            )
        )

        # 2. Compare CHANGELOG vs git tag
        git_version = self._get_latest_git_tag()
        if git_version is None:
            results.append(
                CheckResult(
                    "WARN",
                    check,
                    "no git tag found — cannot compare with CHANGELOG",
                )
            )
        elif git_version == cl_version:
            results.append(
                CheckResult(
                    "PASS",
                    check,
                    f"git tag matches CHANGELOG ({cl_version})",
                )
            )
        else:
            results.append(
                CheckResult(
                    "FAIL",
                    check,
                    f"git tag ({git_version}) != CHANGELOG ({cl_version})",
                )
            )

        # 3. Compare CHANGELOG vs Dockerfile
        df_version = self._parse_dockerfile_version()
        if df_version is None:
            results.append(
                CheckResult(
                    "WARN",
                    check,
                    "Dockerfile missing or has no wheel version",
                )
            )
        elif df_version == cl_version:
            results.append(
                CheckResult(
                    "PASS",
                    check,
                    f"Dockerfile matches CHANGELOG ({cl_version})",
                )
            )
        else:
            results.append(
                CheckResult(
                    "FAIL",
                    check,
                    f"Dockerfile ({df_version}) != CHANGELOG ({cl_version})",
                )
            )

        # 4. Check CHANGELOG has compare link for the version
        changelog_text = (self.project_root / "CHANGELOG.md").read_text()
        compare_pattern = re.compile(
            rf"^\[{re.escape(cl_version)}]:\s*https://",
            re.MULTILINE,
        )
        if compare_pattern.search(changelog_text):
            results.append(
                CheckResult(
                    "PASS",
                    check,
                    f"CHANGELOG compare link found for {cl_version}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "WARN",
                    check,
                    f"CHANGELOG compare link missing for {cl_version}",
                )
            )

        return results

    # -- orchestration --------------------------------------------------------

    def run(self, only: str | None = None) -> list[CheckResult]:
        """Run all checks, or a single named check."""
        if only is not None:
            if only not in self.ALL_CHECKS:
                raise ValueError(f"Unknown check: {only!r}")
            return getattr(self, only)()

        results: list[CheckResult] = []
        for name in self.ALL_CHECKS:
            results.extend(getattr(self, name)())
        return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run sieval preflight checks.",
    )
    parser.add_argument(
        "--level",
        choices=["quick", "deep"],
        default="quick",
        help="Check depth (default: quick)",
    )
    parser.add_argument(
        "--check",
        metavar="NAME",
        help="Run only the named check",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="fmt",
        help="Output format (default: text)",
    )
    args = parser.parse_args(argv)

    runner = PreflightRunner(level=args.level)
    results = runner.run(only=args.check)

    if args.fmt == "json":
        print(format_json(results))
    else:
        print(format_text(results))

    has_failure = any(r.status == "FAIL" for r in results)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
