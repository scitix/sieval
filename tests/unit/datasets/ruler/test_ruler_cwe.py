"""Tests for the RULER common-words-extraction (CWE) synthetic dataset."""

from sieval.datasets.ruler.ruler_cwe import RulerCweDataset, _get_example


def test_get_example_common_words_repeat_more():
    """Common words repeat ``common_repeats`` times; answer = the common slice."""
    words = [f"word{i}" for i in range(30)]
    context, common = _get_example(
        num_words=20,
        words=words,
        randle_words=[],
        common_repeats=5,
        uncommon_repeats=1,
        common_nums=3,
        random_seed=42,
    )
    assert len(common) == 3
    # Each common word appears 5×; each uncommon (17 of them) appears 1×.
    for cw in common:
        assert context.count(f" {cw}") >= 5


def test_load_emits_rows_with_ruler_schema():
    ds = RulerCweDataset(name_or_path=".", max_seq_length=512, num_samples=4)
    rows = ds.test_set
    assert rows is not None and len(rows) == 4
    for r in rows:
        assert set(r) == {"index", "input", "outputs", "length", "answer_prefix"}
        assert r["input"]
        assert len(r["outputs"]) == 10  # default num_cw
        # The answer cue is split off the tail into answer_prefix.
        assert r["answer_prefix"].startswith(" Answer: The top 10 words")
        # Answer words are present in the numbered list within the prompt body.
        for w in r["outputs"]:
            assert w in r["input"]


def test_load_is_deterministic_for_fixed_seed():
    first = RulerCweDataset(
        name_or_path=".", max_seq_length=512, num_samples=3
    ).test_set
    second = RulerCweDataset(
        name_or_path=".", max_seq_length=512, num_samples=3
    ).test_set
    assert first is not None and second is not None
    assert first[0]["input"] == second[0]["input"]
    assert first[0]["outputs"] == second[0]["outputs"]
