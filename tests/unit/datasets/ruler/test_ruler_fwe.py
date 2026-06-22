"""Tests for the RULER frequent-words-extraction (FWE) synthetic dataset."""

from sieval.community.ruler.scripts.tokenizer import select_tokenizer
from sieval.datasets.ruler.ruler_fwe import RulerFweDataset, _generate_input_output


def test_generate_input_output_returns_top3_excluding_noise():
    tokenizer = select_tokenizer("openai", "cl100k_base")
    text, answer, num_words = _generate_input_output(
        512,
        tokenizer=tokenizer,
        coded_wordlen=6,
        vocab_size=50,
        incremental=16,
        alpha=2.0,
        random_seed=42,
    )
    assert "coded text" in text
    assert len(answer) == 3  # top-3 frequent coded words
    assert "..." not in answer  # the noise entry is excluded
    assert num_words > 0
    # The prompt ends with the answer cue (RULER bakes it into the template).
    assert "the three most frequently appeared words are:" in text


def test_load_emits_rows_with_ruler_schema():
    ds = RulerFweDataset(name_or_path=".", max_seq_length=512, num_samples=4)
    rows = ds.test_set
    assert rows is not None and len(rows) == 4
    for r in rows:
        assert set(r) == {"index", "input", "outputs", "length", "answer_prefix"}
        assert r["input"]
        assert len(r["outputs"]) == 3
        # The answer cue is split off the tail into answer_prefix.
        assert r["answer_prefix"].startswith(" Answer: According to the coded text")
        # Every reported coded word appears in the prompt body.
        for w in r["outputs"]:
            assert w in r["input"]


def test_load_is_deterministic_for_fixed_seed():
    first = RulerFweDataset(
        name_or_path=".", max_seq_length=512, num_samples=3
    ).test_set
    second = RulerFweDataset(
        name_or_path=".", max_seq_length=512, num_samples=3
    ).test_set
    assert first is not None and second is not None
    assert first[0]["input"] == second[0]["input"]
    assert first[0]["outputs"] == second[0]["outputs"]
