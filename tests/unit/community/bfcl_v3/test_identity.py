"""Pin the vendored BFCL v3 files, and keep their provenance resolvable.

A sha256 taken over our own copy pins whatever was copied -- it detects drift
after vendoring and proves nothing about what was vendored. The non-circular
anchor is the `Upstream blob:` line in each vendored module's docstring: a git
blob id, resolvable in upstream's repository by anyone, independent of this
checkout. All twelve were resolved against `ShishirPatil/gorilla` at
`ea13468e`, and all twelve matched.

So the two tests here divide the work deliberately:

* :func:`test_vendored_file_has_not_drifted` is the drift detector. Its digests
  are read off the copy, on purpose -- that is the only thing they can be.
* :func:`test_vendored_file_records_resolvable_provenance` keeps the real anchor
  present. It cannot resolve the blob (that needs upstream's repository, which
  no test may reach for), so it checks the claim is *recorded* in a shape a
  reader can resolve. Without it, a re-vendoring that drops the docstring leaves
  the first test passing over bytes with no stated origin.

`Upstream path:` is relative to `berkeley-function-call-leaderboard/`, the
package root inside upstream's monorepo -- not to the repository root, where
none of these paths exist.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import hashlib
import re
from pathlib import Path

import pytest

import sieval.community.bfcl_v3 as pkg

#: sha256 of each vendored file AS COMMITTED -- upstream's bytes plus the
#: documented import rewrites, provenance docstring, and for `parser.py` the
#: annotated safety deviation. Updated only alongside a change to that file's
#: docstring deviation list, in the same commit, so the reason a digest moved is
#: always readable next to the new digest.
VENDORED_SHA256 = {
    "ast_checker.py": (
        "66ba103f5dd17d9e2b4ee597ebfdb013486add739412366e65c4cb6691acb936"
    ),
    "type_mappings.py": (
        "cebc6f9a1b28e277483dd13b2a3fd3ceb95ad8a4631daf50933ad1af3015c698"
    ),
    "type_convertor/java_type_converter.py": (
        "4b450f96bfe528823de681e9d6b0b7fb748ca51a534b2d7a59d76d1047dfebf2"
    ),
    "type_convertor/js_type_converter.py": (
        "ef9562e296bcd7aeeff8b2b88f0fba1c996951b91b229fc91a03bf0f855fcfe2"
    ),
    "source_parser/java_parser.py": (
        "024dfc247dbb7890c4ed969ab48a9fab3212b85bdb74402528a5ce7b8abab5fe"
    ),
    "source_parser/js_parser.py": (
        "3a096d914cc54514009e11915e8d4cc4c35af26b5a1c9f21ced640cd896dc737"
    ),
    "parser.py": ("bed1fd6c50b4164dd6f4277ef54b1eae908491f5476e7830f742f614fe7266f9"),
    "tool_convert.py": (
        "11b758572a09fca738479243e4cc3005387def672786b3231c868e6d73ad386e"
    ),
    "output_checks.py": (
        "683a43d71e3e55d52b5edc7afd94def3ce449261642c3dbde543ae82ceab5268"
    ),
    "aggregate.py": (
        "b7abb648a15492db0101d7a0628b8cc1fdfa2d1542992e0819a44911729c9472"
    ),
    "prompts.py": ("fae6ed78b354c3e1a355795569a78ba18f2b0826f52bfa1dca35a2454655b78f"),
    "preprocess.py": (
        "ce638e4bee853d12644528e728475025810b719287631f0ea68e9197dfde763f"
    ),
}

#: Upstream revision every `Upstream blob:` line above was resolved against.
UPSTREAM_REVISION = "ea13468e"


@pytest.mark.parametrize("relative_path", sorted(VENDORED_SHA256))
def test_vendored_file_has_not_drifted(relative_path: str):
    root = Path(pkg.__file__).parent
    digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
    assert digest == VENDORED_SHA256[relative_path], (
        f"{relative_path} changed since vendoring. A vendored file tracks "
        "upstream; if the change is deliberate, say what it is in the module "
        "docstring's deviation list and update this hash in the same commit."
    )


@pytest.mark.parametrize("relative_path", sorted(VENDORED_SHA256))
def test_vendored_file_records_resolvable_provenance(relative_path: str):
    """Every vendored file states where it came from, resolvably.

    Three files legitimately share one blob (`parser.py`, `tool_convert.py` and
    `preprocess.py` are all partial extractions from upstream's
    `model_handler/utils.py`), so the blob ids are not asserted unique.
    """
    text = (Path(pkg.__file__).parent / relative_path).read_text()
    assert re.search(r"^Upstream path: \S+\.py$", text, re.M), (
        f"{relative_path} records no `Upstream path:`. The sha256 above pins "
        "these bytes but says nothing about their origin; this line is what a "
        "reader follows to check the copy against upstream."
    )
    assert re.search(r"^Upstream blob: [0-9a-f]{40}$", text, re.M), (
        f"{relative_path} records no `Upstream blob:` git object id. That id is "
        f"the only anchor outside this checkout -- resolvable at "
        f"{UPSTREAM_REVISION} as "
        f"`berkeley-function-call-leaderboard/<Upstream path>`."
    )


def test_category_tables_total_the_pinned_row_counts():
    """The per-category tables are constants; these are the totals they must sum to.

    Named for what it does: it reads no snapshot. The check that the tables
    match the downloaded data is the acceptance replay, which is skipped
    wherever the snapshot is absent -- so this one has to stand on its own as a
    guard against an edited table, not be mistaken for the one that walks rows.
    """
    from sieval.community.bfcl_v3 import LIVE_COUNTS, NON_LIVE_COUNTS

    assert sum(NON_LIVE_COUNTS.values()) == 1390
    assert sum(LIVE_COUNTS.values()) == 2251
    assert set(NON_LIVE_COUNTS) & set(LIVE_COUNTS) == set()
