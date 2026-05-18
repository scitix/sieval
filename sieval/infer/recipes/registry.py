"""
Recipe registry: load, match, and resolve infer recipes.

Supports two-level matching:
  1. Family match: model architecture → recipe YAML file (e.g. qwen3 → qwen.yaml)
  2. Size match: parameter count → size bucket within that file (e.g. ~4B → qwen3-4b)

Profile resolution uses fuzzy GPU matching to look up a complete, self-contained
parameter set from ``profiles[hardware][precision][framework]``.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from sieval.infer.config import ParamValue

_RECIPE_DIR = Path(__file__).parent


def _coerce_param(value: object) -> ParamValue:
    """Coerce a YAML scalar to a typed ParamValue.

    yaml.safe_load returns Python-native types for scalars (str, int,
    float, bool), but the static type is ``object``.  This validates
    at runtime and satisfies the type checker without ``# type: ignore``.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    # Fallback: stringify anything unexpected (e.g. None from YAML null)
    return str(value)


@dataclass
class Recipe:
    """Typed representation of a model infer recipe.

    Fields:
        name: Recipe name (e.g. "qwen3-8b")
        size_range: [min_b, max_b) parameter count range in billions
        profiles: Per-hardware, per-precision, per-framework params, e.g.
            {"H100-80G": {"bf16": {"vllm": {"dtype": "bfloat16", ...}}}}
        known_issues: Human-readable issue descriptions
        tested_versions: Per-framework version specifiers
    """

    name: str = ""
    size_range: tuple[float, float] = (0.0, float("inf"))
    profiles: dict[str, dict[str, dict[str, dict[str, ParamValue]]]] = field(
        default_factory=dict,
    )
    known_issues: list[str] = field(default_factory=list)
    tested_versions: dict[str, list[str]] = field(default_factory=dict)


def _parse_recipe(name: str, raw: dict[str, object]) -> Recipe:
    """Parse a raw YAML dict into a typed Recipe.

    Old fields (``frameworks``, ``hardware_overrides``, ``precision_overrides``)
    are silently ignored if present — only ``profiles`` is parsed.
    """
    # size_range (optional)
    raw_range = raw.get("size_range")
    if isinstance(raw_range, list) and len(raw_range) == 2:
        lo, hi = raw_range
        size_range = (
            float(lo) if isinstance(lo, (int, float, str)) else 0.0,
            float(hi) if isinstance(hi, (int, float, str)) else float("inf"),
        )
    else:
        size_range = (0.0, float("inf"))

    # profiles (optional) — hw_key → prec_key → framework → params
    raw_profiles = raw.get("profiles", {})
    profiles: dict[str, dict[str, dict[str, dict[str, ParamValue]]]] = {}
    if isinstance(raw_profiles, dict):
        for hw_key, prec_map in raw_profiles.items():
            if isinstance(prec_map, dict):
                prec_dict: dict[str, dict[str, dict[str, ParamValue]]] = {}
                for prec_key, fw_map in prec_map.items():
                    if isinstance(fw_map, dict):
                        fw_dict: dict[str, dict[str, ParamValue]] = {}
                        for fw_name, fw_params in fw_map.items():
                            if isinstance(fw_params, dict):
                                typed: dict[str, ParamValue] = {}
                                for k, v in fw_params.items():
                                    typed[str(k)] = _coerce_param(v)
                                fw_dict[str(fw_name)] = typed
                        prec_dict[str(prec_key)] = fw_dict
                profiles[str(hw_key)] = prec_dict

    # known_issues (optional)
    raw_issues = raw.get("known_issues", [])
    known_issues: list[str] = []
    if isinstance(raw_issues, list):
        known_issues = [str(issue) for issue in raw_issues]

    # tested_versions (optional) — top-level, NOT inside frameworks
    raw_tv = raw.get("tested_versions", {})
    tested_versions: dict[str, list[str]] = {}
    if isinstance(raw_tv, dict):
        for fw_name, specs in raw_tv.items():
            if isinstance(specs, list):
                tested_versions[str(fw_name)] = [str(s) for s in specs]
            elif isinstance(specs, str):
                tested_versions[str(fw_name)] = [specs]

    return Recipe(
        name=name,
        size_range=size_range,
        profiles=profiles,
        known_issues=known_issues,
        tested_versions=tested_versions,
    )


def _emit_known_issues(recipe: Recipe) -> None:
    """Log recipe.known_issues as warnings. Call only from lookup paths."""
    for issue in recipe.known_issues:
        logger.warning("Recipe {!r} known issue: {}", recipe.name, issue)


def check_tested_versions(
    framework: str,
    installed_version: str,
    specifiers: list[str],
) -> bool:
    """Check if the installed framework version satisfies the tested specifiers.

    Each item in *specifiers* is an independent version constraint (OR
    semantics across items).  A single item may itself contain comma-
    separated sub-specifiers which are ANDed together per PEP 440 (e.g.
    ``">=0.4.0,<1.0"``).

    This allows recipes to declare both an official release range **and**
    a specific dev build, e.g.::

        tested_versions:
          sglang: [">=0.4.6.post1", "==0.0.0.dev1"]

    Logs a warning if the version matches none of the specifiers.

    Args:
        framework: Framework name (e.g. "vllm", "sglang")
        installed_version: Installed version string (e.g. "0.8.3")
        specifiers: List of PEP 440 version specifiers — OR across items

    Returns:
        True if the version satisfies **any** specifier, False otherwise.
    """
    try:
        ver = Version(installed_version)
    except InvalidVersion:
        logger.warning(
            "Cannot parse {} version {!r} — skipping version check",
            framework,
            installed_version,
        )
        return False

    for raw_spec in specifiers:
        try:
            spec = SpecifierSet(raw_spec)
        except InvalidSpecifier:
            logger.warning(
                "Invalid version specifier {!r} for {} — skipping",
                raw_spec,
                framework,
            )
            continue
        if ver in spec:
            return True

    logger.warning(
        "{} version {} does not satisfy any tested specifier {} — "
        "results may differ from validated configurations",
        framework,
        installed_version,
        specifiers,
    )
    return False


def load_family_recipes(family: str) -> list[Recipe]:
    """Load all recipes for a given family from YAML files.

    Scans all .yaml files for a `_family` metadata field matching the
    requested family. Returns all recipe entries (excluding `_family`).
    """
    recipes: list[Recipe] = []
    for yaml_file in sorted(_RECIPE_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        file_family = data.get("_family", "")
        if file_family != family:
            continue
        for key, raw in data.items():
            if key.startswith("_") or not isinstance(raw, dict):
                continue
            recipes.append(_parse_recipe(key, raw))
    return recipes


def match_recipe(family: str, param_billions: float) -> Recipe | None:
    """Two-level recipe matching: family → size bucket.

    Args:
        family: Model family name (e.g. "qwen3")
        param_billions: Approximate parameter count in billions

    Returns:
        Matching Recipe, or None if no match found.
        Returns None (instead of guessing) when param_billions falls
        outside all defined size ranges — fail early, fail loudly.
    """
    candidates = load_family_recipes(family)
    if not candidates:
        logger.warning("No recipes found for family {!r}", family)
        return None

    # Find the recipe whose size_range contains param_billions
    for recipe in candidates:
        lo, hi = recipe.size_range
        if lo <= param_billions < hi:
            logger.info(
                "Matched recipe {!r} for {:.1f}B params (range [{:.0f}, {:.0f}))",
                recipe.name,
                param_billions,
                lo,
                hi,
            )
            _emit_known_issues(recipe)
            return recipe

    # No exact match — report available ranges and return None (fail early)
    ranges = ", ".join(
        f"{r.name} [{r.size_range[0]:.0f}, {r.size_range[1]:.0f})" for r in candidates
    )
    logger.warning(
        "No size match for {:.1f}B params in family {!r}. Available: {}",
        param_billions,
        family,
        ranges,
    )
    return None


def resolve_profile(
    recipe: Recipe,
    gpu_model: str | None,
    precision: str | None,
    framework: str,
) -> dict[str, ParamValue] | None:
    """Look up a complete parameter set from the recipe's profiles.

    Uses fuzzy GPU matching: splits the profile hardware key on ``[-_\\s]+``
    and checks that every resulting token appears in ``gpu_model`` (case-
    insensitive).  For example, key ``"H100-80G"`` matches GPU string
    ``"NVIDIA H100-SXM5-80GB"``.

    Args:
        recipe: Typed Recipe with a ``profiles`` field.
        gpu_model: Detected GPU name (e.g. ``"NVIDIA A100-SXM4-80GB"``).
            ``None`` → return ``None`` immediately.
        precision: Precision key (e.g. ``"bf16"``, ``"fp8"``).
            ``None`` → defaults to ``"bf16"``.
        framework: Framework name (e.g. ``"vllm"``, ``"sglang"``).

    Returns:
        A shallow copy of the matched params dict, or ``None`` when any
        level of the lookup fails (GPU, precision, or framework).
    """
    if gpu_model is None:
        return None

    if precision is None:
        precision = "bf16"

    if not recipe.profiles:
        return None

    gpu_lower = gpu_model.lower()
    for hw_key, prec_map in recipe.profiles.items():
        key_tokens = re.split(r"[-_\s]+", hw_key.lower())
        if all(token in gpu_lower for token in key_tokens):
            logger.info("GPU {!r} matched profile hardware key {!r}", gpu_model, hw_key)
            if precision not in prec_map:
                logger.warning(
                    "Precision {!r} not in profile for hw {!r} (available: {})",
                    precision,
                    hw_key,
                    ", ".join(prec_map),
                )
                return None
            fw_map = prec_map[precision]
            if framework not in fw_map:
                logger.info(
                    "Framework {!r} not in profile {}[{}] (available: {})",
                    framework,
                    hw_key,
                    precision,
                    ", ".join(fw_map),
                )
                return None
            return dict(fw_map[framework])

    logger.info(
        "GPU {!r} did not match any profile hardware key "
        "in recipe {!r} (available: {})",
        gpu_model,
        recipe.name,
        ", ".join(recipe.profiles),
    )
    return None


def load_recipe(name: str) -> Recipe:
    """Load a recipe by exact name from the registry.

    Searches for recipe files (.yaml) in the recipes directory.
    Recipe names are top-level keys in YAML files. A recipe named
    "qwen3-8b" could be in any .yaml file (e.g., qwen.yaml).

    Raises KeyError if recipe not found or name starts with '_'.
    """
    if name.startswith("_"):
        raise KeyError(
            f"Recipe name {name!r} must not start with '_' (reserved for metadata)"
        )

    for yaml_file in _RECIPE_DIR.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and name in data:
            raw = data[name]
            if not isinstance(raw, dict):
                raise KeyError(f"Recipe {name!r} is not a dict")
            recipe = _parse_recipe(name, raw)
            _emit_known_issues(recipe)
            return recipe

    available = list_recipes()
    raise KeyError(
        f"Recipe {name!r} not found. Available: {', '.join(available) or '(none)'}"
    )


def list_recipes() -> list[str]:
    """List all available recipe names across all YAML files."""
    names: list[str] = []
    for yaml_file in sorted(_RECIPE_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            names.extend(k for k in data if not k.startswith("_"))
    return sorted(set(names))
