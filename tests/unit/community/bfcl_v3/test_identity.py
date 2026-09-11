import hashlib
from pathlib import Path

import pytest

import sieval.community.bfcl_v3 as pkg

#: sha256 of each vendored file AS COMMITTED (upstream's bytes plus the
#: documented import rewrite and provenance docstring). Fill these in from the
#: failing run's output, once, after reviewing the diff against upstream.
VENDORED_SHA256 = {
    "ast_checker.py": (
        "66ba103f5dd17d9e2b4ee597ebfdb013486add739412366e65c4cb6691acb936"
    ),
    "type_mappings.py": (
        "cebc6f9a1b28e277483dd13b2a3fd3ceb95ad8a4631daf50933ad1af3015c698"
    ),
    "type_convertor/java_type_converter.py": (
        "d33be3f7e65f0302ef729b0ae328e4d7395758b9b92d8d4562365f97dbc84942"
    ),
    "type_convertor/js_type_converter.py": (
        "dfb0373b58b51c45e63346e1a893c055168dd6ae39e27307436fefa45981e6cc"
    ),
    "parser.py": ("4bba883c1f537a3ae0ad0c85260974861d056ffe25ab50a90bd07d7bb8dec2a3"),
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
}


@pytest.mark.parametrize("relative_path", sorted(VENDORED_SHA256))
def test_vendored_file_has_not_drifted(relative_path: str):
    root = Path(pkg.__file__).parent
    digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
    assert digest == VENDORED_SHA256[relative_path], (
        f"{relative_path} changed since vendoring. A vendored file tracks "
        "upstream; if the change is deliberate, say what it is in the module "
        "docstring's deviation list and update this hash in the same commit."
    )


def test_category_tables_match_the_pinned_snapshot():
    from sieval.community.bfcl_v3 import LIVE_COUNTS, NON_LIVE_COUNTS

    assert sum(NON_LIVE_COUNTS.values()) == 1390
    assert sum(LIVE_COUNTS.values()) == 2251
    assert set(NON_LIVE_COUNTS) & set(LIVE_COUNTS) == set()
