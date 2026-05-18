"""Key normalization helpers for engine_params dicts.

`engine_params` keys are CLI-style flags that accept either dash or underscore
form (e.g. `tensor-parallel-size` / `tensor_parallel_size`). This module
canonicalizes them to underscore form so downstream code and the IR
(`RoleAssignment.engine_params`) carry a single, unambiguous form.

Design decision — asymmetric tolerance:
    We normalize at *user-facing* entry points (YAML ``overrides:`` field,
    ``sieval infer start -- --flag=value`` passthrough, ``auto_resolve_plan``
    overrides) so users can paste dash-form flags from framework docs
    without surprise. Project-
    owned recipe YAMLs are expected to follow one style (underscore, by
    convention) and are *not* pre-normalized — if a recipe author mixes
    dash and underscore forms within the same profile, the IR-layer
    collision check in ``RoleAssignment.__post_init__`` surfaces it as a
    bug rather than silently picking a winner. Rationale: user overrides
    are ad-hoc and should be forgiving; recipe YAMLs are curated project
    config and should be consistent. Revisit if recipe authorship ever
    moves out-of-tree.

AI-Generated Code - Claude Opus 4.7 (Anthropic)
"""

from collections.abc import Mapping

from sieval.infer.config import ParamValue


def normalize_param_key(key: str) -> str:
    """Canonicalize a CLI-style engine-param key to underscore form."""
    return key.replace("-", "_")


def merge_params(
    *sources: Mapping[str, ParamValue],
) -> dict[str, ParamValue]:
    """Merge engine-param dicts with key normalization.

    - Each source is normalized before merging.
    - Across sources: later sources win (preserves existing
      `params.update(overrides)` user-wins semantics).
    - Within a single source: if both dash and underscore forms of the same
      normalized key are present, raises ``ValueError`` naming both keys.
      This indicates a caller bug (e.g. hand-written dict literal with
      inconsistent key style, or an un-migrated naive merge); callers
      performing a multi-source merge should use this function rather than
      squashing sources into a single dict literal first.
    """
    result: dict[str, ParamValue] = {}
    for source in sources:
        # First pass: detect intra-source collisions before writing anything (atomic).
        # Dict keys are unique, so a repeat `normalized` always comes from a
        # *different* original key (e.g. `foo-bar` and `foo_bar`).
        seen: dict[str, str] = {}  # normalized_key -> original_key
        for original_key in source:
            normalized = normalize_param_key(original_key)
            if normalized in seen:
                raise ValueError(
                    f"ambiguous engine_params key: {seen[normalized]!r} and "
                    f"{original_key!r} both normalize to {normalized!r}"
                )
            seen[normalized] = original_key
        # Second pass: commit values now that the source is known collision-free.
        for original_key, value in source.items():
            result[normalize_param_key(original_key)] = value
    return result
