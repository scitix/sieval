"""Authoritative extras satisfaction checks for dataset deps_group hints.

AI-Generated Code - Claude Haiku 4.5 (Anthropic)
"""

import re
from functools import lru_cache
from importlib import metadata

from packaging.requirements import Requirement

# `extra == "foo"` literal inside a PEP 508 marker expression.
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


def _extras_requirements_for(
    dist: metadata.Distribution,
) -> dict[str, list[Requirement]]:
    """Return ``extras_group -> [Requirement, ...]`` parsed from *dist*."""
    result: dict[str, list[Requirement]] = {}
    for req_str in dist.requires or []:
        try:
            req = Requirement(req_str)
        except Exception:
            continue
        if req.marker is None:
            continue
        for match in _EXTRA_MARKER.finditer(str(req.marker)):
            result.setdefault(match.group(1), []).append(req)
    return result


@lru_cache(maxsize=1)
def _extras_requirements() -> dict[str, list[Requirement]]:
    try:
        dist = metadata.distribution("sieval")
    except metadata.PackageNotFoundError:
        return {}
    return _extras_requirements_for(dist)


def extras_unsatisfied(group: str) -> list[str]:
    """Return unmet requirements for *group*, or ``[]``.

    A requirement is satisfied iff the distribution is importable AND its
    version satisfies the declared specifier — a transitive install that
    pins compatibly is fine. Unknown *group* or non-installed checkout
    returns ``[]`` (nothing authoritative to report).
    """
    reqs = _extras_requirements().get(group, [])
    unmet: list[str] = []
    for req in reqs:
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            unmet.append(str(req))
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            unmet.append(f"{req} (installed: {installed})")
    return unmet
