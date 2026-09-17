"""Tests for scripts/sync_agent_rules.py.

The generator reads `paths:` frontmatter from .claude/rules/*.md and the
locations of sieval/*/CLAUDE.md, then rewrites a marker-delimited block in
AGENTS.md. Tests drive it against a synthetic tree via --root so they never
depend on the real repository's rule inventory.

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sync_agent_rules.py"


def _load():
    spec = importlib.util.spec_from_file_location("sync_agent_rules", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_agent_rules"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A synthetic repo: one layer file, two rules, one AGENTS.md."""
    (tmp_path / "sieval" / "core").mkdir(parents=True)
    (tmp_path / "sieval" / "core" / "CLAUDE.md").write_text("# Core\n")

    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "tests.md").write_text(
        '---\npaths:\n  - "tests/**/*.py"\n---\n\n# Test Rules\n'
    )
    (rules / "deps.md").write_text(
        '---\npaths:\n  - "pyproject.toml"\n  - "pdm.lock"\n---\n\n# Deps\n'
    )

    (tmp_path / "AGENTS.md").write_text(
        "# Guidelines\n\n"
        "<!-- BEGIN generated: rule-map -->\n"
        "<!-- END generated: rule-map -->\n"
    )
    return tmp_path


def test_writes_layer_file_with_derived_scope(tree: Path):
    mod = _load()
    assert mod.main(["--root", str(tree)]) == 0
    text = (tree / "AGENTS.md").read_text()
    assert "`sieval/core/**`" in text
    assert "`sieval/core/CLAUDE.md`" in text


def test_reads_paths_frontmatter_verbatim(tree: Path):
    mod = _load()
    mod.main(["--root", str(tree)])
    text = (tree / "AGENTS.md").read_text()
    assert "`tests/**/*.py`" in text
    assert "`pyproject.toml`" in text
    assert "`pdm.lock`" in text


def test_rule_without_paths_is_scoped_to_everything(tree: Path):
    """No `paths:` key is a real choice, not a parse failure."""
    mod = _load()
    (tree / ".claude" / "rules" / "always.md").write_text(
        "---\ndescription: applies everywhere\n---\n\n# Always\n"
    )
    assert mod.main(["--root", str(tree)]) == 0
    assert "_(always)_" in (tree / "AGENTS.md").read_text()


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ('---\npaths: ["a/**", "b/**"]\n---\n\n# Flow\n', "block list"),
        ("---\npaths:\n---\n\n# Empty\n", "lists no globs"),
        ('---\npaths: "a/**"\n---\n\n# Scalar\n', "block list"),
    ],
)
def test_unreadable_paths_key_is_an_error(tree: Path, frontmatter: str, expected: str):
    """A `paths:` key the parser cannot read must not silently drop the rule.

    Returning [] would render it as `_(always)_` and still satisfy --check,
    since the check re-parses the same frontmatter the same wrong way.
    """
    mod = _load()
    (tree / ".claude" / "rules" / "broken.md").write_text(frontmatter)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--root", str(tree)])
    assert expected in str(exc.value)
    assert "broken.md" in str(exc.value)


def test_paths_are_backticked_not_markdown_links(tree: Path):
    """check_links validates Markdown links in tracked files; backticks keep
    the generated block off that check's surface."""
    mod = _load()
    mod.main(["--root", str(tree)])
    block = (tree / "AGENTS.md").read_text()
    assert "](" not in block


def test_generation_is_idempotent(tree: Path):
    mod = _load()
    mod.main(["--root", str(tree)])
    once = (tree / "AGENTS.md").read_text()
    mod.main(["--root", str(tree)])
    assert (tree / "AGENTS.md").read_text() == once


def test_check_passes_when_current(tree: Path):
    mod = _load()
    mod.main(["--root", str(tree)])
    assert mod.main(["--root", str(tree), "--check"]) == 0


def test_check_fails_when_stale(tree: Path):
    mod = _load()
    mod.main(["--root", str(tree)])
    rules = tree / ".claude" / "rules"
    (rules / "newrule.md").write_text('---\npaths:\n  - "src/**"\n---\n\n# New\n')
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tree), "--check"])


def test_check_fails_when_block_hand_edited(tree: Path):
    mod = _load()
    mod.main(["--root", str(tree)])
    agents = tree / "AGENTS.md"
    agents.write_text(agents.read_text().replace("sieval/core/**", "sieval/bogus/**"))
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tree), "--check"])


def test_rejects_output_over_size_budget(tree: Path):
    """Codex truncates project docs past 32 KiB silently."""
    mod = _load()
    agents = tree / "AGENTS.md"
    agents.write_text("x" * 33000 + "\n" + agents.read_text())
    with pytest.raises(SystemExit) as exc:
        mod.main(["--root", str(tree)])
    assert "32768" in str(exc.value)


def test_missing_markers_is_an_error(tree: Path):
    mod = _load()
    (tree / "AGENTS.md").write_text("# Guidelines\n")
    with pytest.raises(SystemExit):
        mod.main(["--root", str(tree)])
