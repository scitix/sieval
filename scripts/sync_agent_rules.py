"""Regenerate the agent rule map inside AGENTS.md.

Claude Code auto-loads `sieval/*/CLAUDE.md` (on demand) and `.claude/rules/*.md`
(by `paths:` glob). opencode and Codex do neither: opencode only walks upward
from the working directory, and Codex reads only AGENTS.md files on the
root-to-cwd path. Both therefore need an explicit index.

Hand-writing that index would duplicate the `paths:` frontmatter that already
declares each rule's scope, and would go stale with no signal. Generating it
keeps the copy honest, the same way sync_meta_index.py does for the registry.

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BEGIN = "<!-- BEGIN generated: rule-map -->"
END = "<!-- END generated: rule-map -->"

# Codex's project_doc_max_bytes default. Past this, it truncates silently:
# instructions near the end of the file simply stop applying, with no warning.
MAX_BYTES = 32768


def _parse_paths_frontmatter(text: str) -> list[str]:
    """Return the `paths:` globs from a rule file's YAML frontmatter.

    Hand-rolled rather than via PyYAML: this script runs from pre-commit and
    preflight, where an import-light dependency footprint matters, and the
    frontmatter shape here is fixed (a `paths:` key over a list of strings).
    """
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    if end == -1:
        return []
    block = text[3:end]

    globs: list[str] = []
    in_paths = False
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.strip() == "paths:":
            in_paths = True
            continue
        if in_paths:
            stripped = line.strip()
            if stripped.startswith("- "):
                globs.append(stripped[2:].strip().strip('"').strip("'"))
                continue
            # A non-list line at the same or lower indent ends the paths block.
            if not line.startswith(" "):
                in_paths = False
    return globs


def render_block(root: Path) -> str:
    """Render the rule map from layer files and rule frontmatter."""
    rows: list[tuple[str, str]] = []

    for layer_file in sorted((root / "sieval").glob("*/CLAUDE.md")):
        layer = layer_file.parent.name
        rows.append((f"`sieval/{layer}/**`", f"`sieval/{layer}/CLAUDE.md`"))

    rules_dir = root / ".claude" / "rules"
    for rule_file in sorted(rules_dir.glob("*.md")) if rules_dir.is_dir() else []:
        globs = _parse_paths_frontmatter(rule_file.read_text(encoding="utf-8"))
        rel = rule_file.relative_to(root).as_posix()
        scope = ", ".join(f"`{g}`" for g in globs) if globs else "_(always)_"
        rows.append((scope, f"`{rel}`"))

    lines = [
        BEGIN,
        "",
        "Read the matching file before editing a path it covers. Claude Code",
        "loads these automatically; opencode and Codex need this map.",
        "",
        "| When editing | Read first |",
        "| --- | --- |",
    ]
    lines += [f"| {scope} | {target} |" for scope, target in rows]
    lines += ["", END]
    return "\n".join(lines)


def render_document(root: Path) -> str:
    """Return AGENTS.md with a freshly generated block."""
    agents_path = root / "AGENTS.md"
    if not agents_path.exists():
        raise SystemExit(f"not found: {agents_path}")

    current = agents_path.read_text(encoding="utf-8")
    start = current.find(BEGIN)
    stop = current.find(END)
    if start == -1 or stop == -1:
        raise SystemExit(
            f"AGENTS.md is missing the generated markers ({BEGIN} / {END}). "
            "Add them before running this script."
        )

    return current[:start] + render_block(root) + current[stop + len(END) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the agent rule map inside AGENTS.md."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether the committed map is up-to-date without writing.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (defaults to this checkout).",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    agents_path = root / "AGENTS.md"
    rendered = render_document(root)

    size = len(rendered.encode("utf-8"))
    if size > MAX_BYTES:
        raise SystemExit(
            f"AGENTS.md would be {size} bytes, over the {MAX_BYTES}-byte budget. "
            "Codex truncates project docs past this point silently — trim the "
            "file rather than raising the cap."
        )

    current = agents_path.read_text(encoding="utf-8")
    if args.check:
        if current == rendered:
            return 0
        raise SystemExit(
            "AGENTS.md rule map is out of date. "
            "Run `python scripts/sync_agent_rules.py` to regenerate."
        )

    if current != rendered:
        agents_path.write_text(rendered, encoding="utf-8")
        print(f"Updated rule map in {agents_path.relative_to(root)}")
    else:
        print("Rule map already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
