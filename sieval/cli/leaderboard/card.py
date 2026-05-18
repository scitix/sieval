"""Alignment card loader for leaderboard YAML references.

Cards carry the reference claim in YAML frontmatter::

    ---
    reference:
      kind: tr                                # tr | user-defined
      source: "arXiv:<id>"                    # free-form identifier
      title: <human-readable title>
    tolerance: <absolute numeric threshold>
    reference_scores: {<model>: {<task>: <score>}}
    ---

Prose body is ignored. The loader is path-agnostic; by convention callers
resolve to ``leaderboards/alignment/<tr-slug>/<stage>.md``.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

ReferenceKind = Literal["tr", "user-defined"]
_VALID_KINDS: frozenset[ReferenceKind] = frozenset(("tr", "user-defined"))


@dataclass(frozen=True, slots=True)
class AlignmentCard:
    """Parsed frontmatter of an alignment card.

    Attributes
    ----------
    path : Path
        Resolved filesystem path to the card, for traceability.
    kind : Literal["tr", "user-defined"]
        ``"tr"`` external paper; ``"user-defined"`` local baseline.
    source : str
        Free-form identifier (``arXiv:...``, ``DOI:...``, URL, ``git:<sha>``,
        or any project-specific identifier).
    title : str
        Human-readable title.
    tolerance : float
        Absolute tolerance in the metric's native unit.
    reference_scores : dict[str, dict[str, float]]
        Nested mapping: model key -> task key -> score.
    """

    path: Path
    kind: ReferenceKind
    source: str
    title: str
    tolerance: float
    reference_scores: dict[str, dict[str, float]]


def load_card(path: Path) -> AlignmentCard:
    """Load and validate a card from a markdown-with-YAML-frontmatter file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has no frontmatter block, or frontmatter is missing
        required fields (``reference.kind``, ``reference.source``,
        ``reference.title``, ``tolerance``, ``reference_scores``), has
        empty ``reference_scores``, or ``reference.kind`` has an invalid
        value.
    """
    if not path.exists():
        raise FileNotFoundError(f"Alignment card does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            f"Alignment card {path} missing YAML frontmatter block (expected `---` "
            f"delimiters at top of file)"
        )

    meta = yaml.safe_load(match.group(1))
    if not isinstance(meta, dict):
        raise ValueError(f"Alignment card {path} frontmatter must be a YAML mapping")

    reference = meta.get("reference")
    if not isinstance(reference, dict):
        raise ValueError(f"Alignment card {path} missing required field: reference")

    kind = reference.get("kind")
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Alignment card {path} reference.kind must be one of "
            f"{sorted(_VALID_KINDS)!r} (got {kind!r})"
        )

    source = reference.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(
            f"Alignment card {path} missing required field: reference.source"
        )

    title = reference.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError(
            f"Alignment card {path} missing required field: reference.title"
        )

    tolerance = meta.get("tolerance")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or tolerance <= 0
    ):
        raise ValueError(
            f"Alignment card {path} missing or invalid required field: tolerance "
            f"(must be a positive number)"
        )

    raw_refs = meta.get("reference_scores")
    if not isinstance(raw_refs, dict):
        raise ValueError(
            f"Alignment card {path} missing required field: reference_scores"
        )
    if not raw_refs:
        raise ValueError(f"Alignment card {path} reference_scores is empty")

    # Normalize to dict[str, dict[str, float]]; validate shape.
    refs: dict[str, dict[str, float]] = {}
    for model_key, task_scores in raw_refs.items():
        if not isinstance(model_key, str) or not model_key:
            raise ValueError(
                f"Alignment card {path} reference_scores has non-string "
                f"model key: {model_key!r}"
            )
        if not isinstance(task_scores, dict):
            raise ValueError(
                f"Alignment card {path} reference_scores[{model_key!r}] "
                f"must be a mapping"
            )
        inner: dict[str, float] = {}
        for task_key, score in task_scores.items():
            if not isinstance(task_key, str) or not task_key:
                raise ValueError(
                    f"Alignment card {path} reference_scores[{model_key!r}] "
                    f"has non-string task key: {task_key!r}"
                )
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(
                    f"Alignment card {path} "
                    f"reference_scores[{model_key!r}][{task_key!r}] "
                    f"score must be numeric (got {type(score).__name__})"
                )
            inner[task_key] = float(score)
        refs[model_key] = inner

    return AlignmentCard(
        path=path,
        kind=kind,
        source=source,
        title=title,
        tolerance=float(tolerance),
        reference_scores=refs,
    )
