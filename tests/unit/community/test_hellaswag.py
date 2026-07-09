"""Pinning tests for the HellaSwag lm-eval preprocessing adaptation.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.community.hellaswag import preprocess, process_doc


def test_preprocess_strip_and_title_marker():
    # " [title]" -> ". "  (and surrounding whitespace stripped)
    assert preprocess("  Climbing [title] the wall  ") == "Climbing. the wall"


def test_preprocess_removes_bracketed_artifacts():
    assert preprocess("clean [substeps] this") == "clean this"
    # bracket at the very start: strip() runs BEFORE removal, so the resulting
    # leading space is NOT re-stripped (faithful to lm-eval; no "  " to collapse).
    assert preprocess("[header] lead") == " lead"


def test_preprocess_double_space_collapse_is_single_pass():
    # lm-eval applies ONE replace("  ", " "); 4 spaces collapse to 2, not 1.
    assert preprocess("a    b") == "a  b"


def test_process_doc_builds_query_choices_gold():
    doc = {
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he starts pulling up roofing on a roof.",
        "endings": [
            "is ripping [title] up the shingles.",
            "holds a stick.",
        ],
        "label": "1",
    }
    out = process_doc(doc)
    assert out["query"] == (
        "Roof shingle removal: A man is sitting on a roof. "
        "He starts pulling up roofing on a roof."
    )
    assert out["choices"] == ["is ripping. up the shingles.", "holds a stick."]
    assert out["gold"] == 1


def test_process_doc_capitalize_lowercases_rest_of_ctx_b():
    # str.capitalize() upper-cases the first char and LOWER-cases the rest.
    doc = {
        "activity_label": "X",
        "ctx_a": "A.",
        "ctx_b": "iPhone IS here",
        "endings": ["e"],
        "label": "0",
    }
    assert process_doc(doc)["query"] == "X: A. Iphone is here"
