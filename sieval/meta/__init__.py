"""Read-only loader for the release-authored ``index.json`` — lets CLI
discovery verbs browse metadata without importing per-dataset optional
deps (e.g. ifeval → nltk + langdetect, math → math-verify).

``scripts/sync_meta_index.py`` writes the file from the live registry at
release time.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from sieval.meta.loader import load_index

__all__ = ["load_index"]
