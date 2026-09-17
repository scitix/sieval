"""Regenerate the agent rule map inside AGENTS.md.

Claude Code auto-loads `sieval/*/CLAUDE.md` (on demand) and `.claude/rules/*.md`
(by `paths:` glob). No other harness does: opencode only walks upward from the
working directory, and Codex reads AGENTS.md files on the root-to-cwd path.
They need an explicit index.

Hand-writing it would duplicate the `paths:` frontmatter that already declares
each rule's scope, and would go stale with no signal — so it is generated, the
way sync_meta_index.py handles the registry.

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


def _parse_paths_frontmatter(text: str, source: str = "<frontmatter>") -> list[str]:
    """Return the `paths:` globs from a rule file's YAML frontmatter.

    Hand-rolled rather than via PyYAML: this runs from pre-commit and preflight,
    where an import-light footprint matters, and the shape is fixed (a `paths:`
    key over a list of strings).

    A file with no `paths:` key is scoped to everything, which is a real choice.
    A `paths:` key this parser cannot read is not: returning ``[]`` there would
    drop the rule from the map with nothing to notice it, since the check
    compares the generated map against the same empty parse. So it raises.
    """
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    if end == -1:
        return []
    block = text[3:end]

    globs: list[str] = []
    in_paths = False
    seen_paths = False
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        stripped = line.strip()
        if stripped.startswith("paths:"):
            # Only a block list is supported; `paths: [...]` on one line is not.
            if stripped != "paths:":
                raise SystemExit(
                    f'{source}: `paths:` must be a block list, one `- "glob"` '
                    f"per line, not {stripped[len('paths:') :].strip()!r}."
                )
            in_paths = True
            seen_paths = True
            continue
        if in_paths:
            if stripped.startswith("- "):
                globs.append(stripped[2:].strip().strip('"').strip("'"))
                continue
            # A non-list line at the same or lower indent ends the paths block.
            if not line.startswith(" "):
                in_paths = False

    if seen_paths and not globs:
        raise SystemExit(f"{source}: `paths:` is present but lists no globs.")
    return globs


def render_block(root: Path) -> str:
    """Render the rule map from layer files and rule frontmatter."""
    rows: list[tuple[str, str]] = []

    for layer_file in sorted((root / "sieval").glob("*/CLAUDE.md")):
        layer = layer_file.parent.name
        rows.append((f"`sieval/{layer}/**`", f"`sieval/{layer}/CLAUDE.md`"))

    rules_dir = root / ".claude" / "rules"
    for rule_file in sorted(rules_dir.glob("*.md")) if rules_dir.is_dir() else []:
        rel = rule_file.relative_to(root).as_posix()
        globs = _parse_paths_frontmatter(rule_file.read_text(encoding="utf-8"), rel)
        scope = ", ".join(f"`{g}`" for g in globs) if globs else "_(always)_"
        rows.append((scope, f"`{rel}`"))

    lines = [
        BEGIN,
        "",
        "Read the matching file before editing a path it covers. Claude Code",
        "loads these automatically; every other harness needs this map.",
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
