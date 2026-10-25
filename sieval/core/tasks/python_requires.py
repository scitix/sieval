"""Third-party Python packages a task needs, checked before a run starts.

A task declares them as ``Task.python_requires = {import name: requirement}``,
where the requirement is a PEP 508 string naming the distribution
(``{"sklearn": "scikit-learn>=1.3"}``). A missing package is an environment
failure: the CLI refuses the run before any engine starts or dataset loads.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata

from packaging.requirements import InvalidRequirement, Requirement


@dataclass(frozen=True, slots=True)
class UnmetPythonRequirement:
    """One declared package the running interpreter does not satisfy."""

    import_name: str
    requirement: str
    reason: str

    def __str__(self) -> str:
        return f"{self.requirement} [import {self.import_name}]: {self.reason}"


class MissingPythonRequirementsError(Exception):
    """Tasks declare Python packages the running interpreter does not satisfy.

    ``unmet`` maps each task name to what it lacks, in configuration order.
    """

    def __init__(self, unmet: Mapping[str, Sequence[UnmetPythonRequirement]]):
        self.unmet: dict[str, tuple[UnmetPythonRequirement, ...]] = {
            task: tuple(entries) for task, entries in unmet.items() if entries
        }
        if not self.unmet:
            raise ValueError("MissingPythonRequirementsError needs an unmet entry")
        lines = [
            f"- {task}: {entry}"
            for task, entries in self.unmet.items()
            for entry in entries
        ]
        installs = sorted(
            {entry.requirement for entries in self.unmet.values() for entry in entries}
        )
        super().__init__(
            "Missing Python packages the tasks declare in python_requires:\n"
            + "\n".join(lines)
            + "\nInstall them, e.g. pip install "
            + " ".join(f"'{req}'" for req in installs)
        )

    @property
    def task(self) -> str | None:
        """The one task refused, ``None`` when several are."""
        return next(iter(self.unmet)) if len(self.unmet) == 1 else None


def validate_python_requires(declared: object, owner: str) -> dict[str, Requirement]:
    """Parse a ``python_requires`` declaration, or raise naming *owner*.

    Raises:
        TypeError: *declared* is not a mapping of strings to strings.
        ValueError: an import name is not a dotted identifier, or a
            requirement is not PEP 508 or names extras (which go unchecked).
    """
    if not isinstance(declared, Mapping):
        raise TypeError(
            f"{owner} python_requires must be a mapping of import name to "
            f"requirement, got {type(declared).__name__}"
        )
    parsed: dict[str, Requirement] = {}
    for import_name, requirement in declared.items():
        if not isinstance(import_name, str) or not isinstance(requirement, str):
            raise TypeError(
                f"{owner} python_requires keys and values must be strings, "
                f"got {import_name!r}: {requirement!r}"
            )
        if not all(part.isidentifier() for part in import_name.split(".")):
            raise ValueError(
                f"{owner} python_requires key {import_name!r} is not an import name"
            )
        try:
            req = Requirement(requirement)
        except InvalidRequirement as exc:
            raise ValueError(
                f"{owner} python_requires[{import_name!r}] = {requirement!r} is "
                f"not a PEP 508 requirement: {exc}"
            ) from exc
        if req.extras:
            raise ValueError(
                f"{owner} python_requires[{import_name!r}] = {requirement!r} names "
                "extras, which are not checked; declare each extra's import instead"
            )
        parsed[import_name] = req
    return parsed


def _not_importable(import_name: str) -> str | None:
    """Why *import_name* cannot be found, or ``None`` when it can."""
    # find_spec does not import the module, but does import a dotted name's
    # parents, and a broken parent may raise anything.
    try:
        found = importlib.util.find_spec(import_name) is not None
    except ModuleNotFoundError as exc:
        if exc.name is not None and f"{import_name}.".startswith(f"{exc.name}."):
            return "not installed"  # a parent is itself missing
        return f"parent import failed: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return f"parent import failed: {type(exc).__name__}: {exc}"
    return None if found else "not installed"


def unmet_python_requires(
    declared: Mapping[str, str], owner: str
) -> tuple[UnmetPythonRequirement, ...]:
    """What in *declared* the running interpreter does not satisfy.

    Met when the import name is found and, for a pinned requirement, the
    installed distribution satisfies the pin; an unpinned requirement's
    distribution is not looked up. A marker excluding this interpreter is met.
    The module itself is not imported, so one that fails on import passes.
    """
    unmet: list[UnmetPythonRequirement] = []
    for import_name, req in validate_python_requires(declared, owner).items():
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        text = str(req)
        reason = _not_importable(import_name)
        if reason is not None:
            unmet.append(UnmetPythonRequirement(import_name, text, reason))
            continue
        if not req.specifier:
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            unmet.append(
                UnmetPythonRequirement(
                    import_name, text, f"no installed distribution {req.name!r}"
                )
            )
            continue
        if not req.specifier.contains(installed, prereleases=True):
            unmet.append(
                UnmetPythonRequirement(import_name, text, f"installed {installed}")
            )
    return tuple(unmet)
