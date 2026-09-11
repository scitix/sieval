"""Cover the BFCL v3 shared grading entry point and the two protocol mixins.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json

import pytest

from sieval.core.models import FunctionToolCall, ModelMeta, ModelOutput
from sieval.tasks.bfcl_v3._base import (
    BfclV3FCMixin,
    BfclV3PromptMixin,
    grade_single_turn,
)
from sieval.tasks.bfcl_v3.bfcl_v3_non_live_0shot_gen import BfclV3NonLiveZeroShotGenTask
from sieval.tasks.bfcl_v3.bfcl_v3_non_live_0shot_gen_fc import (
    BfclV3NonLiveZeroShotGenFCTask,
)
from tests.conftest import MockChatModel, MockDataset

_META: ModelMeta = {"model": "mock-chat", "api_base": None, "default_params": {}}


def _output(text: str = "", tool_calls=None) -> ModelOutput:
    return ModelOutput(model=_META, texts=[text], tool_calls=tool_calls)


def test_a_correct_call_grades_true(simple_row):
    decoded = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    assert (
        grade_single_turn(
            simple_row["function"],
            decoded,
            simple_row["ground_truth"],
            "Python",
            "simple",
            False,
        )
        is True
    )


def test_a_wrong_argument_grades_false(simple_row):
    decoded = [{"calculate_triangle_area": {"base": 10, "height": 99}}]
    assert (
        grade_single_turn(
            simple_row["function"],
            decoded,
            simple_row["ground_truth"],
            "Python",
            "simple",
            False,
        )
        is False
    )


def test_a_failed_decode_grades_false_for_an_ast_category(simple_row):
    assert (
        grade_single_turn(
            simple_row["function"],
            None,
            simple_row["ground_truth"],
            "Python",
            "simple",
            False,
        )
        is False
    )


# Both irrelevance sets, because they are separate keys in the polarity table
# and a typo in one of them is invisible while the other is tested.
@pytest.mark.parametrize("category", ["irrelevance", "live_irrelevance"])
def test_irrelevance_is_correct_when_no_call_is_produced(irrelevance_row, category):
    assert (
        grade_single_turn(
            irrelevance_row["function"], None, None, "Python", category, False
        )
        is True
    )
    assert (
        grade_single_turn(
            irrelevance_row["function"], [], None, "Python", category, False
        )
        is True
    )


@pytest.mark.parametrize("category", ["irrelevance", "live_irrelevance"])
def test_irrelevance_is_wrong_when_a_call_is_produced(irrelevance_row, category):
    decoded = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    assert (
        grade_single_turn(
            irrelevance_row["function"], decoded, None, "Python", category, False
        )
        is False
    )


def test_live_relevance_inverts_irrelevance(irrelevance_row):
    decoded = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    assert (
        grade_single_turn(
            irrelevance_row["function"],
            decoded,
            None,
            "Python",
            "live_relevance",
            False,
        )
        is True
    )
    assert (
        grade_single_turn(
            irrelevance_row["function"], None, None, "Python", "live_relevance", False
        )
        is False
    )


def test_underscore_to_dot_reconciles_the_fc_tool_name(dotted_row):
    """FC sent `geometry_triangle_area`; gold says `geometry.triangle_area`."""
    fc_decoded = [{"geometry_triangle_area": {"base": 10, "height": 5}}]
    prompt_decoded = [{"geometry.triangle_area": {"base": 10, "height": 5}}]

    assert (
        grade_single_turn(
            dotted_row["function"],
            fc_decoded,
            dotted_row["ground_truth"],
            "Python",
            "simple",
            True,
        )
        is True
    )
    assert (
        grade_single_turn(
            dotted_row["function"],
            prompt_decoded,
            dotted_row["ground_truth"],
            "Python",
            "simple",
            False,
        )
        is True
    )
    # And the flag is load-bearing in both directions, which is what makes
    # passing it through the process boundary necessary rather than tidy.
    assert (
        grade_single_turn(
            dotted_row["function"],
            fc_decoded,
            dotted_row["ground_truth"],
            "Python",
            "simple",
            False,
        )
        is False
    )


def test_a_missing_gold_raises_rather_than_scoring_the_row(simple_row):
    """A value-reference row with no gold fails; it is never graded as wrong.

    The distinction is the whole point: scoring it `False` would charge the
    model for a dataset fault, and under `DENOMINATOR_REQUESTED` the two read
    the same in the headline. Reached only past the produced-call guard, so the
    call passed here has to be a valid one.
    """
    with pytest.raises(ValueError, match="not goldless"):
        grade_single_turn(
            simple_row["function"],
            [{"calculate_triangle_area": {"base": 10, "height": 5}}],
            None,
            "Python",
            "simple",
            False,
        )


# --------------------------------------------------------------------------
# The protocol axis. Both mixins are stateless, so they are exercised directly
# rather than through a whole Task.
# --------------------------------------------------------------------------

#: Valid JavaScript that is ALSO valid Python, and decodes differently under
#: each. Deliberately not a snippet only one parser accepts: this one pins the
#: failure mode a hardcoded language actually has on real rows.
_JS_CALL = "[obj.method(opts={a: 1, b: 2})]"


def test_the_prompt_mixin_parses_under_the_row_language():
    """A JavaScript row must reach the JavaScript parser, not Python's `ast`.

    The failure this pins is the quiet one: `_JS_CALL` parses under BOTH
    languages, and Python's parser resolves the object literal into a real
    dict where JavaScript's keeps it as the source string. Hardcoding
    `"Python"` therefore yields a plausible wrong value rather than a raise --
    a wrong score with nothing in the record to show for it.
    """
    decoded = BfclV3PromptMixin()._decode(_output(_JS_CALL), "JavaScript")
    assert decoded == [{"obj.method": {"opts": "{a: 1, b: 2}"}}]
    # The same text under Python -- what a hardcoded language would return.
    assert BfclV3PromptMixin()._decode(_output(_JS_CALL), "Python") == [
        {"obj.method": {"opts": {"a": 1, "b": 2}}}
    ]


def test_the_prompt_mixin_puts_the_schemas_in_a_system_turn_and_sends_no_tools(
    dotted_row,
):
    messages, tools = BfclV3PromptMixin()._build_messages(dotted_row)
    assert tools is None
    assert messages[0]["role"] == "system"
    # Verbatim, dot included: the Prompt protocol never renames anything, which
    # is why its `UNDERSCORE_TO_DOT` is False.
    assert "geometry.triangle_area" in messages[0]["content"]
    assert messages[1:] == dotted_row["question"]


def test_the_prompt_mixin_merges_into_a_row_that_already_has_a_system_turn(
    system_turn_row,
):
    """Upstream merges; prepending a second system turn is a different payload.

    92 live rows open with their own system turn. What makes this worth a test
    of its own is that the wrong shape is not visibly wrong: two system turns
    is a conversation every provider accepts and answers, so the run completes
    and the score is merely somewhat off.
    """
    messages, _ = BfclV3PromptMixin()._build_messages(system_turn_row)
    assert [m["role"] for m in messages] == ["system", "user"]
    content = messages[0]["content"]
    assert "calculate_triangle_area" in content
    # The row's own instruction survives, after the schema block rather than
    # in a turn of its own.
    assert content.endswith("You are a geometry tutor.")


def test_building_messages_leaves_the_stored_row_untouched(system_turn_row, java_row):
    """Both vendored helpers rewrite their argument in place.

    Reached without a copy, they would edit the sample the runner holds, and
    the damage is invisible on a first pass -- it surfaces only on the second
    (a resume, a retry, `max_iterations > 1`), as a schema block appended twice
    or a type already flattened. Idempotence is the discriminating check: one
    call looks correct either way.
    """
    for mixin in (BfclV3PromptMixin(), BfclV3FCMixin()):
        for row in (system_turn_row, java_row):
            before = json.dumps(row, sort_keys=True)
            first = mixin._build_messages(row)
            second = mixin._build_messages(row)
            assert first == second
            assert json.dumps(row, sort_keys=True) == before


def test_both_protocols_restate_java_parameters_as_strings(java_row):
    """Upstream's language preprocessing, which both columns share.

    It is shared on purpose: the Prompt and FC numbers are two readings of one
    benchmark only while the schemas they carry agree. So this asserts the same
    property through both protocols rather than trusting one to stand for both.
    """
    messages, _ = BfclV3PromptMixin()._build_messages(java_row)
    prompt_text = messages[0]["content"]
    _, tools = BfclV3FCMixin()._build_messages(java_row)
    assert tools is not None
    fc_properties = tools[0]["function"]["parameters"]["properties"]

    assert [v["type"] for v in fc_properties.values()] == ["string", "string", "string"]
    assert "'type': 'string'" in prompt_text
    # The Java 8 hint, and the element type carried over from the deleted
    # `items` -- both protocols, because both call the same helper.
    for rendered in (prompt_text, json.dumps(tools)):
        assert "Java 8 SDK syntax" in rendered
        assert "The list elements are of type String" in rendered


@pytest.mark.anyio
@pytest.mark.parametrize(
    "task_cls", [BfclV3NonLiveZeroShotGenTask, BfclV3NonLiveZeroShotGenFCTask]
)
async def test_preprocess_stores_the_unprocessed_schema_for_grading(task_cls, java_row):
    """Preprocessing is model-facing only; the grader must see the real types.

    Upstream's `ast_file_runner` reads `function` back out of the dataset and
    hands it to `ast_checker` unprocessed, so a port that stored the flattened
    copy would compare a Java row's real `ArrayList` gold against a schema
    claiming every parameter is a `string`.
    """
    task = task_cls(MockDataset(), MockChatModel())
    record = await task.preprocess(java_row, None)
    assert record["extra"]["function"] == java_row["function"]
    assert json.loads(record["extra"]["function"])[0]["parameters"]["properties"][
        "tags"
    ] == {"type": "ArrayList", "description": "tags", "items": {"type": "String"}}


class _Ctx:
    preprocess_result = {"extra": {"language": "Python", "category": "simple"}}


@pytest.mark.anyio
async def test_an_undecodable_reply_is_scored_not_raised():
    """Upstream's `ast_decoder:decoder_failed` -- the model's outcome."""
    task = BfclV3NonLiveZeroShotGenTask(MockDataset(), MockChatModel())
    reply = _output("this is not a function call at all")
    record = await task.postprocess(reply, _Ctx())
    (rollout,) = record["rollouts"]
    assert rollout.get("prediction") is None
    assert rollout["extracted"] is False
    assert "decode_error" in rollout["extra"]


@pytest.mark.anyio
async def test_a_missing_parser_dependency_propagates_instead_of_scoring(monkeypatch):
    """An absent optional extra is a broken environment, not a bad answer.

    `_decode` defers its import of the vendored parser (tree-sitter lives
    behind the `bfcl-v3` extra), so the `ModuleNotFoundError` is raised from
    inside the same `try` that turns a decode failure into a score. Swallowed,
    it grades every row of the run wrong with `fails` at 0 -- indistinguishable
    from a model that cannot call a function, and the logs go with the run.
    """
    task = BfclV3NonLiveZeroShotGenTask(MockDataset(), MockChatModel())

    def no_tree_sitter(_self, _output, _language):
        raise ModuleNotFoundError("No module named 'tree_sitter'")

    monkeypatch.setattr(BfclV3PromptMixin, "_decode", no_tree_sitter)
    with pytest.raises(ModuleNotFoundError, match="tree_sitter"):
        await task.postprocess(_output("[f(x=1)]"), _Ctx())


def test_the_fc_mixin_sends_tools_and_renames_the_dotted_function(dotted_row):
    """Upstream's own rewrite -- the one `UNDERSCORE_TO_DOT` later reverses."""
    messages, tools = BfclV3FCMixin()._build_messages(dotted_row)
    assert messages == dotted_row["question"]
    assert tools is not None
    assert [tool["function"]["name"] for tool in tools] == ["geometry_triangle_area"]


def test_the_fc_mixin_ignores_the_language():
    """No source text is parsed, so the row's language changes nothing."""
    output = _output(
        tool_calls=(FunctionToolCall("c0", "calculate_triangle_area", '{"base": 10}'),)
    )
    expected = [{"calculate_triangle_area": {"base": 10}}]
    assert BfclV3FCMixin()._decode(output, "Java") == expected
    assert BfclV3FCMixin()._decode(output, "Python") == expected


def test_the_fc_mixin_reads_no_calls_as_an_empty_list():
    """`[]` is "decoded fine, called nothing" -- correct for irrelevance.

    Distinct from a raise, which means undecodable. Both score wrong for an AST
    category, but they are different upstream outcomes.
    """
    assert BfclV3FCMixin()._decode(_output("no thanks"), "Python") == []


@pytest.mark.parametrize("arguments", ["", "   ", {}])
def test_the_fc_mixin_reads_an_absent_argument_payload_as_no_arguments(arguments):
    """A zero-parameter tool is a real shape, and `json.loads("")` raises."""
    output = _output(tool_calls=(FunctionToolCall("c0", "list_all", arguments),))
    assert BfclV3FCMixin()._decode(output, "Python") == [{"list_all": {}}]


def test_the_two_protocols_disagree_on_a_dotted_name_and_each_flag_reconciles_it(
    dotted_row,
):
    """The end-to-end reason the flag exists, across both mixins at once."""
    prompt_decoded = BfclV3PromptMixin()._decode(
        _output("[geometry.triangle_area(base=10, height=5)]"), "Python"
    )
    fc_decoded = BfclV3FCMixin()._decode(
        _output(
            tool_calls=(
                FunctionToolCall(
                    "c0",
                    "geometry_triangle_area",
                    json.dumps({"base": 10, "height": 5}),
                ),
            )
        ),
        "Python",
    )
    assert prompt_decoded != fc_decoded

    for decoded, mixin in (
        (prompt_decoded, BfclV3PromptMixin),
        (fc_decoded, BfclV3FCMixin),
    ):
        assert (
            grade_single_turn(
                dotted_row["function"],
                decoded,
                dotted_row["ground_truth"],
                "Python",
                "simple",
                mixin.UNDERSCORE_TO_DOT,
            )
            is True
        )
