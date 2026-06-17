"""Tests for the RULER variable-tracking (VT) synthetic dataset."""

from sieval.datasets.ruler._common import _NOISE_HAYSTACK
from sieval.datasets.ruler.ruler_vt import (
    RulerVtDataset,
    _generate_chains,
    _generate_input_output,
    _randomize_icl,
)


def test_generate_chains_shape():
    """Each chain has num_hops+1 distinct variables and num_hops+1 assignments."""
    variables, chains = _generate_chains(num_chains=2, num_hops=3)
    assert len(variables) == 2
    assert len(chains) == 2
    for v, c in zip(variables, chains, strict=True):
        assert len(v) == 4  # num_hops + 1
        assert len(set(v)) == 4  # distinct names
        assert len(c) == 4  # one root assignment + num_hops hops
        assert c[0].startswith(f"VAR {v[0]} = ")  # root binds a literal


def test_icl_chains_use_literal_seed_value():
    """is_icl chains use 3-char names and the fixed root value 12345 (RULER)."""
    variables, chains = _generate_chains(num_chains=1, num_hops=2, is_icl=True)
    assert all(len(name) == 3 for name in variables[0])
    assert chains[0][0] == f"VAR {variables[0][0]} = 12345"


def test_randomize_icl_replaces_value_and_answer_vars():
    """randomize_icl swaps the literal 12345 and the trailing answer variables."""
    icl = "VAR ABC = 12345 ... they are: ABC\n"
    out = _randomize_icl(icl, num_hops=0)
    assert "12345" not in out  # root value refreshed
    assert "ABC" not in out  # answer var (last token) refreshed everywhere


def test_generate_input_output_answer_is_chain_vars():
    """The prompt embeds the chain; the answer is the chain's variable names."""
    prompt, answer = _generate_input_output(
        num_noises=20,
        num_chains=1,
        num_hops=4,
        type_haystack="noise",
        haystack=_NOISE_HAYSTACK,
    )
    assert "Memorize and track" in prompt
    assert len(answer) == 5  # num_hops + 1 variables in the single chain
    # Every answer variable name must appear somewhere in the prompt body.
    for var in answer:
        assert var in prompt


def test_generate_input_output_rejects_unknown_haystack():
    """Only essay/noise haystacks are supported."""
    import pytest

    with pytest.raises(NotImplementedError):
        _generate_input_output(
            num_noises=5,
            num_chains=1,
            num_hops=2,
            type_haystack="bogus",
            haystack=_NOISE_HAYSTACK,
        )


def test_load_emits_rows_with_ruler_schema():
    """Loaded rows carry the RULER VT schema (input/outputs/answer_prefix split)."""
    ds = RulerVtDataset(
        name_or_path=".", max_seq_length=512, num_samples=4, num_hops=2
    )
    rows = ds.test_set
    assert rows is not None and len(rows) == 4
    for r in rows:
        assert set(r) == {"index", "input", "outputs", "length", "answer_prefix"}
        assert r["input"]
        assert isinstance(r["outputs"], list) and r["outputs"]
        # The real answer_prefix is split off the tail (input ends at the body);
        # only the embedded ICL copy's prefix remains inside input.
        assert r["answer_prefix"].startswith(" Answer: According to the chain(s)")
        assert r["input"].rstrip().endswith("text above.")
        assert r["input"].count("they are:") == 1
        # Every answer variable resolves to a name present in the prompt body.
        for var in r["outputs"]:
            assert var in r["input"]


def test_load_prepends_one_shot_icl():
    """RULER bakes a 1-shot worked example before the real prompt, so the template
    head appears twice (ICL copy + real body) across input + answer_prefix."""
    ds = RulerVtDataset(
        name_or_path=".", max_seq_length=1024, num_samples=2, num_hops=2
    )
    row = ds.test_set[0]
    full = row["input"] + row["answer_prefix"]
    assert full.count("Memorize and track the chain(s)") == 2
    assert full.count("they are:") == 2


def test_load_is_deterministic_for_fixed_seed():
    kw = {"name_or_path": ".", "max_seq_length": 512, "num_samples": 3, "num_hops": 2}
    first = RulerVtDataset(**kw).test_set[0]
    second = RulerVtDataset(**kw).test_set[0]
    assert first["input"] == second["input"]
    assert first["outputs"] == second["outputs"]
