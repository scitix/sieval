"""Tests for the RULER common-words-extraction (CWE) synthetic dataset."""

from sieval.datasets.ruler.ruler_cwe import RulerCweDataset, _get_example


def test_get_example_common_words_repeat_more():
    """Common words repeat ``common_repeats`` times; answer = the common slice."""
    words = [f"word{i}" for i in range(30)]
    context, common = _get_example(
        num_words=20,
        words=words,
        common_repeats=5,
        uncommon_repeats=1,
        common_nums=3,
        random_seed=42,
    )
    assert len(common) == 3
    # Each common word appears 5×; each uncommon (17 of them) appears 1×.
    for cw in common:
        assert context.count(f" {cw}") >= 5


def test_load_emits_prompt_answer_rows():
    ds = RulerCweDataset(name_or_path=".", max_seq_length=512, num_samples=4)
    rows = ds.test_set
    assert len(rows) == 4
    for r in rows:
        assert r["prompt"]
        assert len(r["answer"]) == 10  # default num_cw
        # Answer words are present in the numbered list within the prompt.
        for w in r["answer"]:
            assert w in r["prompt"]


def test_load_is_deterministic_for_fixed_seed():
    kw = {"name_or_path": ".", "max_seq_length": 512, "num_samples": 3}
    first = RulerCweDataset(**kw).test_set[0]
    second = RulerCweDataset(**kw).test_set[0]
    assert first["prompt"] == second["prompt"]
    assert first["answer"] == second["answer"]
