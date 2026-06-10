"""Tests for the RULER variable-tracking (VT) synthetic dataset."""

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
    prompt, answer = _generate_input_output(num_noises=20, num_chains=1, num_hops=4)
    assert "Memorize and track" in prompt
    assert answer  # the variables of the (single) chain
    # Every answer variable name must appear somewhere in the prompt body.
    for var in answer:
        assert var in prompt


def test_load_emits_prompt_answer_rows():
    ds = RulerVtDataset(name_or_path=".", max_seq_length=512, num_samples=4, num_hops=2)
    rows = ds.test_set
    assert len(rows) == 4
    for r in rows:
        assert r["prompt"]
        assert r["answer"]
        assert isinstance(r["answer"], list)


def test_load_prepends_one_shot_icl():
    """RULER bakes a 1-shot worked example before the real prompt, so the
    template head + answer cue each appear twice."""
    ds = RulerVtDataset(
        name_or_path=".", max_seq_length=1024, num_samples=2, num_hops=2
    )
    prompt = ds.test_set[0]["prompt"]
    assert prompt.count("Memorize and track the chain(s)") == 2
    assert prompt.count("they are:") == 2


def test_load_is_deterministic_for_fixed_seed():
    kw = {"name_or_path": ".", "max_seq_length": 512, "num_samples": 3, "num_hops": 2}
    first = RulerVtDataset(**kw).test_set[0]
    second = RulerVtDataset(**kw).test_set[0]
    assert first["prompt"] == second["prompt"]
    assert first["answer"] == second["answer"]
