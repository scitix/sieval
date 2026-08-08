"""Regenerate the recipe-resolution golden fixture from the pre-split recipes.

The fixture pins what every ``(recipe, hardware, precision, framework)`` triple
resolved to **before** ``profiles`` was split into ``hardware`` +
``capabilities``, with parameter **order** preserved — ``infer_plans.yaml`` is
compared byte-for-byte under ``--resume``, so ordering is part of the contract.

It is captured from a git ref rather than from the working tree, because the
pre-split schema no longer exists in the code: pre-split ``resolve_profile``
returned the YAML leaf dict verbatim, so the leaf order *is* the resolved order.

The committed fixture — not this script — is the durable artifact. What it pins
is a historical fact that cannot change, so nothing in the test suite needs to
regenerate it; the script exists to show where those numbers came from and to
let a reviewer reproduce them. That also bounds the cost of the git dependency:
if ``_PRE_SPLIT_REF`` ever becomes unreachable (a shallow clone, or a re-cut
history), this script stops working while the tests keep passing. It fails with
the ref named rather than writing an empty fixture.

Usage: pdm run python tests/unit/infer/recipes/_capture_golden.py [git-ref]

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

_OUT = Path(__file__).parent / "golden_recipe_profiles.json"
_RECIPES = ("gpt_oss.yaml", "qwen2_5.yaml", "qwen3.yaml")

# Last commit on main still carrying the pre-split ``profiles`` key — this
# branch's base. Not ``origin/main``: once the split merges, that ref no longer
# has the schema this script reads, and the capture would come back empty.
# Full SHA, not abbreviated: an abbreviation is only unambiguous against the
# history that existed when it was written.
_PRE_SPLIT_REF = "1c15c00c499b640808db03a64f73591ac9733c06"


def _load_at_ref(ref: str, path: str) -> dict:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Cannot read {path!r} at {ref!r}: {proc.stderr.strip()}\n"
            "The pre-split ref is unreachable here (shallow clone, or the "
            "history no longer contains it). The committed fixture remains "
            "valid — it pins a fact that cannot change — so the test suite is "
            "unaffected; only regeneration is. Pass a reachable ref that still "
            "carries the `profiles` schema if you need to re-derive it."
        )
    return yaml.safe_load(proc.stdout) or {}


def capture(ref: str) -> list[dict]:
    """Record every pre-split profile leaf, preserving parameter order."""
    records: list[dict] = []
    for name in _RECIPES:
        data = _load_at_ref(ref, f"sieval/infer/recipes/{name}")
        for entry, raw in data.items():
            if entry.startswith("_") or not isinstance(raw, dict):
                continue
            for hw_key, prec_map in (raw.get("profiles") or {}).items():
                for precision, fw_map in prec_map.items():
                    for framework, params in fw_map.items():
                        records.append(
                            {
                                "recipe": entry,
                                "hardware_key": hw_key,
                                "precision": precision,
                                "framework": framework,
                                # List of pairs, not a dict: order is asserted.
                                "params": [[k, v] for k, v in params.items()],
                            }
                        )
    records.sort(
        key=lambda r: (
            r["recipe"],
            r["hardware_key"],
            r["precision"],
            r["framework"],
        )
    )
    return records


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else _PRE_SPLIT_REF
    records = capture(ref)
    if not records:
        raise SystemExit(f"No pre-split recipes found at {ref!r}")
    _OUT.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(records)} records from {ref} to {_OUT}")
