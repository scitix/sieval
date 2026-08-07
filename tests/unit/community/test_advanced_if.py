"""Unit tests for the AdvancedIF judge assets and scoring kernel.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json

import pytest

from sieval.community import advanced_if
from sieval.community.advanced_if import (
    RELEASED_SYSTEM_STEER_BENCHMARK,
    SRC_ENV_VAR,
    SYSTEM_STEER_BENCHMARK,
    UPSTREAM_JUDGE_SHA256,
    JudgePrompts,
    aggregate_metrics,
    compose_judge_prompt,
    count_all_checks,
    count_in_range_passes,
    format_conversation_history,
    is_system_steer,
    last_user_turn,
    load_judge_prompts,
    parse_conversation,
    parse_judgement,
    parse_rubrics,
    rubric_level_pass_rate,
    system_prompt_of,
)

# --- dataset field decoding ---


def test_parse_conversation_from_json_string_keeps_role_and_content():
    raw = json.dumps(
        [
            {"role": "user", "content": "hi", "extra_field": "dropped"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert parse_conversation(raw) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_parse_conversation_accepts_decoded_list():
    assert parse_conversation([{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": "hi"}
    ]


def test_parse_conversation_rejects_non_list():
    with pytest.raises(ValueError, match="must decode to a list"):
        parse_conversation(json.dumps({"role": "user"}))


def test_parse_rubrics_handles_double_encoded_rubrics():
    """Upstream decodes ``rubrics`` twice because it is sometimes a JSON string."""
    as_list = json.dumps({"rubrics": ["a?", "b?"]})
    as_string = json.dumps({"rubrics": json.dumps(["a?", "b?"])})
    assert parse_rubrics(as_list) == ["a?", "b?"]
    assert parse_rubrics(as_string) == ["a?", "b?"]
    assert parse_rubrics({"rubrics": ["a?"]}) == ["a?"]


def test_parse_rubrics_requires_the_key():
    with pytest.raises(ValueError, match="Rubrics not found"):
        parse_rubrics(json.dumps({"something_else": []}))


# --- conversation rendering (feeds the judge prompt verbatim) ---


def test_format_conversation_history_drops_last_turn_and_numbers_by_assistant():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    # The trailing user turn is excluded -- it is passed to the prompt
    # separately -- and the turn counter advances only past an assistant reply.
    assert format_conversation_history(messages) == "user [1]: u1\nassistant [1]: a1"


def test_last_user_turn_and_system_prompt():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    assert last_user_turn(messages) == "u2"
    assert system_prompt_of(messages) == "sys"
    assert system_prompt_of(messages[1:]) == ""
    assert last_user_turn([{"role": "assistant", "content": "a"}]) == ""


# --- judge routing: upstream's defect, reproduced on purpose ---


def test_system_steer_routing_never_fires_on_released_data():
    """Pin the reproduced defect so a "cleanup" cannot silently change scores.

    Upstream compares against ``if_system_steerability_oss``; the released
    dataset ships ``system_steerability_v2``, so every row -- including all 507
    system-prompt ones -- goes to the user-instruction judge.
    """
    assert is_system_steer(SYSTEM_STEER_BENCHMARK)
    assert not is_system_steer(RELEASED_SYSTEM_STEER_BENCHMARK)
    assert not is_system_steer("complex_if_single_turn_v5")
    assert not is_system_steer("carried_context_multi_turn_eval_v5")


# --- grader reply parsing ---


def _reply(checks: dict, satisfied: str) -> str:
    return json.dumps(
        {"rubrics_check": checks, "SATISFIED_ALL_REQUIREMENTS": satisfied}
    )


def test_parse_judgement_reads_checks_and_declaration():
    judgement = parse_judgement(_reply({"question_1": "Yes"}, "YES"))
    assert judgement is not None
    assert judgement.rubrics_check == {"question_1": "Yes"}
    assert judgement.satisfied_all


@pytest.mark.parametrize("declared", ["Yes", "yes", "YES", " yes "])
def test_parse_judgement_declaration_is_case_insensitive(declared):
    """Upstream's own few-shot examples answer "Yes"/"No", not "YES"/"NO"."""
    judgement = parse_judgement(_reply({}, declared))
    assert judgement is not None
    assert judgement.satisfied_all


def test_parse_judgement_defaults_missing_declaration_to_not_satisfied():
    judgement = parse_judgement(json.dumps({"rubrics_check": {"question_1": "Yes"}}))
    assert judgement is not None
    assert not judgement.satisfied_all


def test_parse_judgement_recovers_fenced_json():
    """sieval cannot force response_format=json_object on every endpoint."""
    fenced = f"Here you go:\n```json\n{_reply({'question_1': 'Yes'}, 'Yes')}\n```"
    judgement = parse_judgement(fenced)
    assert judgement is not None
    assert judgement.rubrics_check == {"question_1": "Yes"}


def test_parse_judgement_returns_none_without_json():
    assert parse_judgement("I could not evaluate this.") is None
    assert parse_judgement("") is None


def test_parse_judgement_stringifies_non_string_answers():
    judgement = parse_judgement(_reply({"question_1": ["Yes"]}, "No"))
    assert judgement is not None
    assert judgement.rubrics_check == {"question_1": "['Yes']"}


# --- counting: the two rates do not share a denominator ---


def test_count_in_range_passes_skips_out_of_range_and_malformed_keys():
    checks = {
        "question_1": "Yes",
        "question_2": "No",
        "question_3": "Yes",  # indexes past a 2-rubric sample
        "notaquestion": "Yes",  # unparseable key
        "question_x": "Yes",  # unparseable index
    }
    assert count_in_range_passes(checks, ["r1", "r2"]) == 1


def test_count_in_range_passes_matches_justified_answers_by_substring():
    """Rubric answers routinely carry a justification before the verdict."""
    checks = {"question_1": "The intro is four sentences. No", "question_2": "Yes"}
    assert count_in_range_passes(checks, ["r1", "r2"]) == 1


def test_rubric_level_pass_rate_divides_by_the_data_rubric_count():
    checks = {"question_1": "Yes"}
    # Two rubrics in the data, one answered -> 0.5, not 1.0.
    assert rubric_level_pass_rate(checks, ["r1", "r2"]) == 0.5
    assert rubric_level_pass_rate({}, []) == 0.0


def test_count_all_checks_ignores_the_rubric_count():
    """The pooled micro rate counts grader-emitted keys, with no range filter."""
    checks = {"question_1": "Yes", "question_9": "Yes", "question_2": "No"}
    assert count_all_checks(checks) == (3, 2)


def test_the_two_denominators_disagree_when_the_grader_under_answers():
    checks = {"question_1": "Yes"}
    rubrics = ["r1", "r2", "r3", "r4"]
    # Per-sample rate is over the data's 4 rubrics ...
    assert rubric_level_pass_rate(checks, rubrics) == 0.25
    # ... while the pooled micro rate is over the single key the grader emitted.
    assert count_all_checks(checks) == (1, 1)


# --- aggregation ---


def _verdict(satisfied: bool, n_checks: int, n_passed: int) -> dict:
    return {
        "satisfied_all": satisfied,
        "n_checks": n_checks,
        "n_checks_passed": n_passed,
    }


def test_aggregate_metrics_pools_both_rates():
    metrics = aggregate_metrics(
        [_verdict(True, 4, 4), _verdict(False, 4, 2), _verdict(False, 2, 0)]
    )
    assert metrics["overall_pass_rate"] == pytest.approx(100 / 3)
    assert metrics["micro_pass_rate"] == pytest.approx(600 / 10)
    assert metrics["n_samples"] == 3.0
    assert metrics["n_rubric_checks"] == 10.0


def test_aggregate_metrics_counts_ungradeable_rollouts_against_the_pass_rate_only():
    """Upstream's failed-row path: in the pass-rate denominator, out of micro."""
    metrics = aggregate_metrics([_verdict(True, 2, 2), _verdict(False, 0, 0)])
    assert metrics["overall_pass_rate"] == pytest.approx(50.0)
    # The failed row contributes no rubrics, so micro stays 100%.
    assert metrics["micro_pass_rate"] == pytest.approx(100.0)


def test_aggregate_metrics_handles_an_empty_set():
    metrics = aggregate_metrics([])
    assert metrics["overall_pass_rate"] == 0.0
    assert metrics["micro_pass_rate"] == 0.0


# --- loading the upstream prompts (never vendored: CC-BY-NC-4.0) ---


def test_load_judge_prompts_requires_the_env_var(monkeypatch):
    monkeypatch.delenv(SRC_ENV_VAR, raising=False)
    load_judge_prompts.cache_clear()
    with pytest.raises(RuntimeError, match=SRC_ENV_VAR):
        load_judge_prompts()


def test_load_judge_prompts_reports_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv(SRC_ENV_VAR, str(tmp_path))
    load_judge_prompts.cache_clear()
    with pytest.raises(RuntimeError, match="does not contain judge.py"):
        load_judge_prompts()


def test_load_judge_prompts_rejects_a_drifted_revision(monkeypatch, tmp_path):
    """The prompts are the benchmark; grading against a drifted copy is fatal."""
    (tmp_path / "judge.py").write_text("JUDGE_PROMPT = 'not upstream'\n")
    monkeypatch.setenv(SRC_ENV_VAR, str(tmp_path))
    load_judge_prompts.cache_clear()
    with pytest.raises(RuntimeError, match=UPSTREAM_JUDGE_SHA256):
        load_judge_prompts()


# --- prompt assembly (upstream prompts stubbed: they are not redistributable) ---


@pytest.fixture
def stub_prompts(monkeypatch):
    prompts = JudgePrompts(
        judge_prompt=(
            "IF|{full_conversation}|{user_prompt_last_turn}|"
            "{response_text}|{rubrics_text}"
        ),
        system_steer_judge_prompt=(
            "STEER|{few_shot_examples}|{system_prompt}|"
            "{user_prompt_last_turn}|{response_text}|{rubrics_text}"
        ),
        steer_few_shot_examples="SHOTS",
    )
    monkeypatch.setattr(advanced_if, "load_judge_prompts", lambda: prompts)
    return prompts


@pytest.mark.usefixtures("stub_prompts")
def test_compose_judge_prompt_fills_the_if_judge_slots():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    composed = compose_judge_prompt(
        "complex_if_single_turn_v5", messages, "the answer", ["r1"]
    )
    kind, conversation, last_turn, response, rubrics_text = composed.split("|")
    assert kind == "IF"
    assert conversation == "user [1]: u1\nassistant [1]: a1"
    assert last_turn == "u2"
    assert response == "the answer"
    # Upstream renders the rubric block with indent=4.
    assert rubrics_text == json.dumps(["r1"], indent=4)


@pytest.mark.usefixtures("stub_prompts")
def test_compose_judge_prompt_uses_the_steer_judge_for_upstreams_literal():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
    ]
    composed = compose_judge_prompt(
        SYSTEM_STEER_BENCHMARK, messages, "the answer", ["r1"]
    )
    kind, shots, system_prompt, last_turn, response, _ = composed.split("|")
    assert kind == "STEER"
    assert shots == "SHOTS"
    assert system_prompt == "sys"
    assert last_turn == "u1"
    assert response == "the answer"


@pytest.mark.usefixtures("stub_prompts")
def test_released_system_steer_rows_compose_the_if_prompt():
    """The reproduced routing defect, seen end to end at the prompt level."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
    ]
    composed = compose_judge_prompt(
        RELEASED_SYSTEM_STEER_BENCHMARK, messages, "answer", ["r1"]
    )
    assert composed.startswith("IF|")
