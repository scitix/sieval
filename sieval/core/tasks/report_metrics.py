"""The metric names a task's ``report.json`` may write, known before it runs.

``names`` are written as spelled; ``patterns`` such as ``score_{category}`` stand
for keys built from the data. Either may be absent from one run; a name neither
declares is one the task never writes. Only numeric keys are metrics.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Literal

#: Keys the runner adds to a generative task's report.
RUNNER_GEN_METRICS = frozenset({"n_truncated", "n_scored_rollouts"})

#: What :func:`~sieval.core.tasks.metrics.sampling_report` can write; the vote
#: keys only with votes.
SAMPLING_METRICS = frozenset(
    {"pass@1", "avg@n", "pass@k", "pass^k", "n", "k", "n_short", "n_problems"}
)
SAMPLING_VOTE_METRICS = frozenset({"maj@k", "self_consistency"})

#: What :func:`~sieval.core.tasks.metrics.health_metrics` writes.
HEALTH_METRICS = frozenset({"n_unextracted"})

_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


@cache
def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """Each placeholder matches one or more characters; the rest is literal."""
    parts = _PLACEHOLDER.split(pattern)
    # Without a literal part a pattern matches every name and checks nothing.
    if len(parts) < 2 or not any(parts):
        raise ValueError(
            f"report metric pattern {pattern!r} needs a literal part and at least "
            "one {placeholder}, such as 'score_{category}'"
        )
    return re.compile(".+".join(re.escape(part) for part in parts))


@dataclass(frozen=True)
class ReportMetrics:
    """The numeric keys one task's report may write."""

    names: frozenset[str]
    patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", frozenset(self.names))
        object.__setattr__(self, "patterns", tuple(dict.fromkeys(self.patterns)))
        if bad := sorted(n for n in self.names if not isinstance(n, str) or not n):
            raise ValueError(f"report metric names are non-empty strings: {bad!r}")
        for pattern in self.patterns:
            _pattern_regex(pattern)

    def match(self, name: str) -> Literal["name", "pattern"] | None:
        """``name`` when declared, ``pattern`` when only a pattern matches."""
        if name in self.names:
            return "name"
        if any(_pattern_regex(p).fullmatch(name) for p in self.patterns):
            return "pattern"
        return None

    def __or__(self, other: "ReportMetrics | Iterable[str]") -> "ReportMetrics":
        if isinstance(other, ReportMetrics):
            return ReportMetrics(
                self.names | other.names, (*self.patterns, *other.patterns)
            )
        if isinstance(other, str):
            # Iterating it would add one name per character.
            raise TypeError(f"ReportMetrics | str: wrap the name, ({other!r},)")
        return ReportMetrics(self.names | frozenset(other), self.patterns)

    def to_json(self) -> dict[str, list[str]]:
        """``{"names": [...], "patterns": [...]}``, names sorted."""
        return {"names": sorted(self.names), "patterns": list(self.patterns)}


def report_metrics(*names: str, patterns: Iterable[str] = ()) -> ReportMetrics:
    """:class:`ReportMetrics` from names written out."""
    return ReportMetrics(frozenset(names), tuple(patterns))


def _patched(owner: type) -> bool:
    """Whether *owner*'s ``report`` is defined in another module than *owner*.

    That is how a patch (``setattr(Task, "report", report)``) looks, and also a
    shared function bound in the class body: at runtime the two are the same.
    """
    report = vars(owner)["report"]
    function = getattr(report, "__func__", report)
    return getattr(function, "__module__", owner.__module__) != owner.__module__


def class_path(cls: type) -> str:
    """``module.QualName``: how a table names a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


def declared_report_metrics(
    task_cls: type, table: Mapping[str, ReportMetrics] | None = None
) -> ReportMetrics | None:
    """The declaration describing *task_cls*'s report, or ``None`` when unknown.

    The nearest class in the MRO with a ``report_metrics`` attribute, or a
    *table* entry under its :func:`class_path`, answers. Unknown when a nearer
    class defines ``report`` (its keys are its own), when ``report`` comes from
    another module (a patch, or a shared function bound in the class body: wrap
    it in a ``def``), or when the attribute is ``None``. A generative task also
    gets :data:`RUNNER_GEN_METRICS`.

    Raises:
        TypeError: if a ``report_metrics`` attribute is not a
            :class:`ReportMetrics` or ``None``.
    """
    table = table or {}
    mro = task_cls.__mro__
    for klass in mro:
        found = vars(klass).get("report_metrics")
        if found is not None and not isinstance(found, ReportMetrics):
            raise TypeError(
                f"{class_path(klass)}.report_metrics is a ReportMetrics or None, "
                f"not {type(found).__name__}"
            )
    report_at = next((i for i, k in enumerate(mro) if "report" in vars(k)), None)
    for depth, klass in enumerate(mro):
        if "report_metrics" in vars(klass):
            found = vars(klass)["report_metrics"]
        elif class_path(klass) in table:
            found = table[class_path(klass)]
        else:
            continue
        if found is None or report_at is None or report_at < depth:
            return None
        if _patched(mro[report_at]):
            return None
        if "gen" in getattr(task_cls, "tags", ()):
            found = found | RUNNER_GEN_METRICS
        return found
    return None
