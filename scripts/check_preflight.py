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

if TYPE_CHECKING:
    from sieval.core.datasets.meta import DatasetMeta

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

_TASK_FILE_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*_(\d+|k)shot_(gen|base_gen|ppl|clp|llmjudge_gen)\.py$"
)
_DATASET_SUFFIX_PATTERN = re.compile(r"(Dataset|DatasetSample|CSVSample)$")


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


class PreflightRunner:
    """Orchestrates preflight checks."""

    ALL_CHECKS: list[str] = [
        "check_links",
        "check_deps",
        "check_dep_coverage",
        "check_tasks",
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

        # Must match the pre-commit hook's `files:` filter in
        # `.pre-commit-config.yaml` (^(sieval|scripts)/). Narrowing here to
        # `sieval/` only would leave the script's `in_scripts` branch untested
        # by preflight while still running in pre-commit — two enforcement
        # surfaces silently diverging.
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
