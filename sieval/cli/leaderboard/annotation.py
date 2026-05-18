"""Per-cell annotation helper for `sieval leaderboard report`.

Pure computation: given an observed score, a reference (from a card), and
a tolerance (also from the card), produce a :class:`CellAnnotation` with
absolute diff and pass/fail status.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

CellStatus = Literal["pass", "fail"]

# Floating-point slack added to the inclusive boundary ``abs(diff) <= tolerance``.
# Small tolerances (e.g. 0.03 for correlation) combined with IEEE-754 subtraction
# can place the gap a few ULPs past the strict inequality; widen by `rel_tol * tol`
# plus an absolute floor so the boundary case stays inclusive without loosening
# the real contract. Same shape as ``math.isclose`` defaults.
_BOUNDARY_REL_TOL = 1e-9
_BOUNDARY_ABS_TOL = 1e-12


class CellAnnotationDict(TypedDict):
    """Serialized shape of :class:`CellAnnotation` carried in matrix rows."""

    observed: float
    reference: float
    diff: float
    tolerance: float
    status: CellStatus


@dataclass(frozen=True, slots=True)
class CellAnnotation:
    """Annotation attached to one (model, task) cell when reference is known.

    Attributes
    ----------
    observed : float
        Score measured in the current run.
    reference : float
        Score claimed by the reference card.
    diff : float
        Signed absolute difference: ``observed - reference``.
    tolerance : float
        Absolute tolerance from the card; retained so the renderer can
        pick a display precision that actually shows the gap.
    status : Literal["pass", "fail"]
        ``"pass"`` iff ``abs(diff) <= tolerance`` (inclusive, float-noise aware).
    """

    observed: float
    reference: float
    diff: float
    tolerance: float
    status: CellStatus

    def to_dict(self) -> CellAnnotationDict:
        return CellAnnotationDict(
            observed=self.observed,
            reference=self.reference,
            diff=self.diff,
            tolerance=self.tolerance,
            status=self.status,
        )


def annotate_cell(
    observed: float, reference: float, tolerance: float
) -> CellAnnotation:
    """Compute the cell annotation.

    Returns a :class:`CellAnnotation` unconditionally. The caller handles the
    "no reference" case (returns ``None`` itself). Tolerance is expected to
    be positive (enforced upstream by :func:`sieval.cli.leaderboard.card.load_card`).
    The boundary ``abs(diff) == tolerance`` passes, with float-noise slack so
    that e.g. ``abs(0.82 - 0.85) = 0.0300…027`` against ``tolerance = 0.03``
    still counts as a pass.
    """
    diff = float(observed) - float(reference)
    slack = max(_BOUNDARY_REL_TOL * tolerance, _BOUNDARY_ABS_TOL)
    status: CellStatus = "pass" if abs(diff) <= tolerance + slack else "fail"
    return CellAnnotation(
        observed=float(observed),
        reference=float(reference),
        diff=diff,
        tolerance=float(tolerance),
        status=status,
    )


def display_precision(tolerance: float) -> int:
    """Choose a decimal precision that makes ``tolerance`` visible.

    Targets "tolerance shows in the rendered digit": a tolerance of 3.0
    wants 1 decimal, 0.3 wants 2, 0.03 wants 3, and so on. Clamped to
    ``[0, 6]`` so wildly small tolerances don't produce unreadable cells.
    Non-positive tolerance falls back to 1 (defensive; :func:`annotate_cell`
    rejects it upstream).
    """
    if not tolerance > 0:
        return 1
    # tolerance=3   → log10=0.48  → ceil(−0.48)=0  → max(1, 0+1)=1
    # tolerance=0.3 → log10=−0.52 → ceil(0.52)=1   → max(1, 1+1)=2
    # tolerance=0.03 → log10=−1.52 → ceil(1.52)=2  → max(1, 2+1)=3
    return max(1, min(6, math.ceil(-math.log10(tolerance)) + 1))
