"""Tests for the RULER frequent-words-extraction (FWE) synthetic dataset."""

from sieval.datasets.ruler._common import build_tokenizer
from sieval.datasets.ruler.ruler_fwe import RulerFweDataset


def test_generate_input_output_returns_top3_excluding_noise():
    from sieval.datasets.ruler.ruler_fwe import _generate_input_output

    tokenizer = build_tokenizer("gpt-4")
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


def test_load_emits_prompt_answer_rows():
    ds = RulerFweDataset(name_or_path=".", max_seq_length=512, num_samples=4)
    rows = ds.test_set
    assert len(rows) == 4
    for r in rows:
        assert r["prompt"]
        assert len(r["answer"]) == 3
        for w in r["answer"]:
            assert w in r["prompt"]


def test_load_is_deterministic_for_fixed_seed():
    kw = {"name_or_path": ".", "max_seq_length": 512, "num_samples": 3}
    first = RulerFweDataset(**kw).test_set[0]
    second = RulerFweDataset(**kw).test_set[0]
    assert first["prompt"] == second["prompt"]
    assert first["answer"] == second["answer"]
