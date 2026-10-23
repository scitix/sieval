"""``Model.extra``: what a model offers the tasks that evaluate it.

``extra`` is a dict on the model, set through ``infer_args.extra`` or the
model config's ``extra``, and never sent to the API. It is a contract between
tasks and models: a task asks the model for a capability by key, and when the
key is absent the task does nothing different. The framework passes the dict
through untouched, records it in the inference identity, and never changes a
request because of it; it only checks the shape of the keys listed here, so a
mistyped value is refused before a run rather than silently ignored.

======================  ======================================================
``sequence_wrappers``   ``{<seq_type>: "<format with {seq}>"}``. A model
                        trained with special tokens around biological
                        sequences needs them at inference; a task wraps each
                        sequence it puts in a prompt with
                        ``template.format(seq=...)``. Types are
                        :data:`SEQUENCE_TYPES`.
======================  ======================================================

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import string
from collections.abc import Mapping
from typing import cast

#: ``extra`` key of the sequence wrappers a model needs.
SEQUENCE_WRAPPERS = "sequence_wrappers"

#: Sequence types a task reads from ``sequence_wrappers``: DNA, RNA (mRNA and
#: oligonucleotides included), protein, and molecules as SMILES.
SEQUENCE_TYPES: tuple[str, ...] = ("dna", "rna", "protein", "mol")


def check_extra(extra: object, where: str = "extra") -> None:
    """Refuse an ``extra`` whose documented keys have the wrong shape.

    Keys not documented in this module are not checked. Raises
    :class:`ValueError` naming *where* and the offending key.
    """
    if extra is None:
        return
    if not isinstance(extra, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(extra).__name__}")
    keyed = cast(Mapping[str, object], extra)
    if SEQUENCE_WRAPPERS in keyed:
        wrappers = keyed[SEQUENCE_WRAPPERS]
        check_sequence_wrappers(wrappers, f"{where}.{SEQUENCE_WRAPPERS}")


def check_sequence_wrappers(wrappers: object, where: str = SEQUENCE_WRAPPERS) -> None:
    """Refuse ``sequence_wrappers`` that are not ``{type: template}``.

    Each type must be one of :data:`SEQUENCE_TYPES` and each template a
    string whose only replacement field is ``{seq}``, present at least once;
    literal braces are written ``{{`` and ``}}``, as in :meth:`str.format`.
    """
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
