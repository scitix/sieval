"""``Model.extra``: what a model offers the tasks that evaluate it.

A dict on the model, never sent to the API. A task asks for a capability by
key and does nothing different when it is absent. Core passes the dict through
and records it in the inference identity; it only checks the shape of the
keys documented here, so a typo is refused before a run.

======================  ======================================================
``sequence_wrappers``   ``{<seq_type>: "<format with {seq}>"}``: the special
                        tokens a model was trained with around biological
                        sequences. Read with :func:`sequence_wrapper`.
======================  ======================================================

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import string
from collections.abc import Mapping
from typing import cast

from sieval.core.types import JSONValue

#: ``extra`` key of the sequence wrappers a model needs.
SEQUENCE_WRAPPERS = "sequence_wrappers"

#: Sequence types a task reads from ``sequence_wrappers``: DNA, RNA (mRNA and
#: oligonucleotides included), protein, and molecules as SMILES.
SEQUENCE_TYPES: tuple[str, ...] = ("dna", "rna", "protein", "mol")


def sequence_wrapper(extra: Mapping[str, JSONValue], seq_type: str) -> str | None:
    """The model's template for *seq_type* in ``Model.extra``, or ``None``.

    ``wrap.format(seq=...)`` wraps a sequence; with ``None`` a task renders
    what it renders for a plain model.
    """
    wrappers = extra.get(SEQUENCE_WRAPPERS)
    if not isinstance(wrappers, Mapping):
        return None
    template = cast(Mapping[str, object], wrappers).get(seq_type)
    return template if isinstance(template, str) else None


def check_extra(extra: object, where: str = "extra") -> None:
    """Raise :class:`ValueError` when a key documented here has the wrong shape;
    other keys are not checked."""
    if extra is None:
        return
    if not isinstance(extra, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(extra).__name__}")
    keyed = cast(Mapping[str, object], extra)
    if SEQUENCE_WRAPPERS in keyed:
        wrappers = keyed[SEQUENCE_WRAPPERS]
        check_sequence_wrappers(wrappers, f"{where}.{SEQUENCE_WRAPPERS}")


def check_sequence_wrappers(wrappers: object, where: str = SEQUENCE_WRAPPERS) -> None:
    """Refuse ``sequence_wrappers`` that are not ``{type: template}``: each type
    in :data:`SEQUENCE_TYPES`, each template a format string whose only field
    is ``{seq}``."""
    if not isinstance(wrappers, Mapping):
        raise ValueError(
            f"{where} must map a sequence type to a template, "
            f"got {type(wrappers).__name__}"
        )
    for seq_type, template in cast(Mapping[object, object], wrappers).items():
        key = f"{where}.{seq_type}"
        if seq_type not in SEQUENCE_TYPES:
            raise ValueError(
                f"{key}: unknown sequence type {seq_type!r}; "
                f"expected one of {', '.join(SEQUENCE_TYPES)}"
            )
        if not isinstance(template, str):
            raise ValueError(
                f"{key} must be a format string, got {type(template).__name__}"
            )
        try:
            fields = [f for _, f, _, _ in string.Formatter().parse(template)]
        except ValueError as exc:
            raise ValueError(
                f"{key}: {template!r} is not a format string ({exc})"
            ) from None
        named = [f for f in fields if f is not None]
        if "seq" not in named:
            raise ValueError(
                f"{key}: {template!r} has no {{seq}}, so the sequence would be "
                "dropped; write e.g. '<dna>{seq}</dna>'"
            )
        others = sorted({f for f in named if f != "seq"})
        if others:
            raise ValueError(
                f"{key}: {template!r} may only use {{seq}}, not "
                + ", ".join(f"{{{f}}}" for f in others)
            )
