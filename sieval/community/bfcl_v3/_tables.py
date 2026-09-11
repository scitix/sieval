"""BFCL v3 category tables, shared by the dataset loader and the tasks.

Held here rather than in either caller because both need them and neither may
import the other's private module. Keeping them out of the vendored files is
also what lets those stay byte-identical.

Row counts are ours, measured at the pinned revision; the category membership is
upstream's `TEST_COLLECTION_MAPPING` at v1.3.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

NON_LIVE_COUNTS: dict[str, int] = {
    "simple": 400,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
    "java": 100,
    "javascript": 50,
    "irrelevance": 240,
}

LIVE_COUNTS: dict[str, int] = {
    "live_simple": 258,
    "live_multiple": 1053,
    "live_parallel": 16,
    "live_parallel_multiple": 24,
    "live_irrelevance": 882,
    "live_relevance": 18,
}

#: Upstream ships no `possible_answer/` file for these: the correct outcome is
#: about whether a call was produced, not about which call it was.
GOLDLESS_CATEGORIES = frozenset(
    {"irrelevance", "live_irrelevance", "live_relevance"}
)

#: Of the goldless three, the one where producing a call is CORRECT.
CALL_EXPECTED = frozenset({"live_relevance"})

#: Only these two are not Python. The same string selects both the
#: type converter inside `ast_checker` and the source-text parser inside
#: `ast_parse` -- these categories' model outputs are Java/JavaScript source.
LANGUAGE_BY_CATEGORY = {"java": "Java", "javascript": "JavaScript"}
