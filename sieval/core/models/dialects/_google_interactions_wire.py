"""Versioned routes for the Gemini Interactions REST API.

``api_base`` follows the same convention as the ``google_genai`` dialect: an
endpoint root selects ``v1beta``, an explicit trailing ``/v1`` or ``/v1beta``
selects that version, and a gateway prefix before it is preserved.  Only the
terminal segment selects the version, because an earlier version-like segment
may belong to a gateway's own routing scheme.

AI-Generated Code - Claude Sonnet 5.5 (Anthropic)
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import httpx

type InteractionsVersion = Literal["v1", "v1beta"]

_VERSIONS: Mapping[str, InteractionsVersion] = MappingProxyType(
    {"v1": "v1", "v1beta": "v1beta"}
)


def resolve_interactions_endpoint(
    base_url: httpx.URL,
) -> tuple[InteractionsVersion, httpx.URL]:
    """Split a root or versioned gateway URL into ``(version, endpoint_root)``.

    The returned root ends in ``/`` and carries no version, so an operation is
    ``endpoint_root.join(f"{version}/interactions")``.  ``base_url`` is not
    mutated.
    """

    if base_url.query or base_url.fragment:
        raise ValueError("Google api_base must not contain query data or fragments")
    segments = base_url.path.rstrip("/").split("/")
    if "interactions" in segments or (
        segments[-1] not in _VERSIONS
        and re.match(r"v[0-9]", segments[-1], re.IGNORECASE)
    ):
        raise ValueError(
            "Google api_base must be an endpoint root or end in /v1 or /v1beta"
        )
    version: InteractionsVersion = "v1beta"
    if segments[-1] in _VERSIONS:
        version = _VERSIONS[segments.pop()]
    return version, base_url.copy_with(path=f"{'/'.join(segments)}/")
