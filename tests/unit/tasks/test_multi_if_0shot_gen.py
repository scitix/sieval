"""Unit tests for the Multi-IF task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import asyncio
import subprocess
import sys

import pytest

from sieval.tasks import multi_if_0shot_gen as module
from sieval.tasks.multi_if_0shot_gen import MultiIFZeroShotGenTask


def test_import_does_not_pull_evaluation_lib():
    # evaluation_lib pulls langdetect/nltk/emoji and the 3.5k-line checker
    # fork; registration must not import it.
    code = (
        "import sys\n"
        "import sieval.tasks.multi_if_0shot_gen\n"
        "assert 'sieval.community.multi_if.evaluation_lib' not in sys.modules, "
        "'evaluation_lib must be lazy-imported'\n"
        # `_ensure_punkt_tab` imports nltk when called, not at module scope, so
        # registration must not drag it in either.
        "assert 'nltk' not in sys.modules, 'nltk must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules
    # don't mask the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _task() -> MultiIFZeroShotGenTask:
    return MultiIFZeroShotGenTask.__new__(MultiIFZeroShotGenTask)


class _Ctx:
    def __init__(self, raw):
        self.raw_sample = raw


class _Final:
    def __init__(self, judgement):
        self.feedback_result = judgement


# Every fixture below grades against langdetect-free checkers on purpose.
# `change_case:*` and the length constraints route through `langdetect.detect`,
# which upstream leaves unseeded (documented on the task), so a fixture built on
# those can flip a verdict between runs on text this short.
_NO_COMMA = "punctuation:no_comma"
_END = "startend:end_checker"
_KEYWORD = "keywords:existence"


def _sample(n_turns: int = 3) -> dict:
    # Turn t's constraints extend turn t-1's, as they do in the real data.
    turns = [
        {
            "prompt": "answer without commas",
            "instruction_id_list": [_NO_COMMA],
            "kwargs": ["{}"],
        },
        {
            "prompt": "now end with the exact phrase",
            "instruction_id_list": [_NO_COMMA, _END],
            "kwargs": ["{}", '{"end_phrase": "the end."}'],
        },
        {
            "prompt": "and mention alpha",
            "instruction_id_list": [_NO_COMMA, _END, _KEYWORD],
            "kwargs": [
                "{}",
                '{"end_phrase": "the end."}',
                '{"keywords": ["alpha"]}',
            ],
        },
    ]
    return {"key": "k:1:en", "language": "English", "turns": turns[:n_turns]}


def _wide_sample(n_turns: int = 2) -> dict:
    """A conversation whose *first* turn already carries two constraints.

    Pooling only differs from averaging per-sample rates when a cell mixes
    conversations with unequal constraint counts, so a report test needs a second
    shape -- `_sample()`'s turn 1 always has exactly one.
    """
    turns = [
        {
            "prompt": "no commas, and end with the phrase",
            "instruction_id_list": [_NO_COMMA, _END],
            "kwargs": ["{}", '{"end_phrase": "the end."}'],
        },
        {
            "prompt": "and mention alpha",
            "instruction_id_list": [_NO_COMMA, _END, _KEYWORD],
            "kwargs": [
                "{}",
                '{"end_phrase": "the end."}',
                '{"keywords": ["alpha"]}',
            ],
        },
    ]
    return {"key": "k:2:en", "language": "English", "turns": turns[:n_turns]}


def _judge(
    responses: list[str],
    n_turns: int = 3,
    raw: dict | None = None,
    n_answered: int | None = None,
) -> dict:
    task = _task()
    post: dict = {
        "rollouts": [
            {
                "index": 0,
                "extracted": True,
                "prediction": [
                    {"turn": i, "response": r} for i, r in enumerate(responses, start=1)
                ],
            }
        ]
    }
    if n_answered is not None:
        post["extra"] = {"n_answered": n_answered}
    _final, judgement = asyncio.run(
        task.feedback(post, _Ctx(raw if raw is not None else _sample(n_turns)))
    )
    return judgement


class _StubOutput:
    def __init__(self, text, finish_reasons=("stop",)):
        self.texts = [text] if text is not None else []
        self.finish_reasons = list(finish_reasons) if finish_reasons else None


class _StubModel:
    """Records the conversation it is handed on each call."""

    def __init__(self, replies, fail_at: int | None = None):
        self._replies = list(replies)
        self._fail_at = fail_at
        self.seen: list[list[dict]] = []

    async def agenerate(self, messages, **kwargs):
        # Copy: the task keeps mutating the same list across turns.
        self.seen.append([dict(m) for m in messages])
        if self._fail_at is not None and len(self.seen) == self._fail_at:
            raise RuntimeError("upstream returned 503")
        return _StubOutput(self._replies[len(self.seen) - 1])


def _run_infer(replies, n_turns=3, fail_at=None):
    task = _task()
    model = _StubModel(replies, fail_at=fail_at)
    task._model = model
    raw = _sample(n_turns)
    ctx = _Ctx(raw)
    pre = asyncio.run(task.preprocess(raw, ctx))
    outputs = asyncio.run(task.infer(pre, ctx))
    return model, outputs


def test_infer_feeds_each_turn_the_preceding_conversation():
    model, outputs = _run_infer(["reply one", "reply two", "reply three"])
    assert len(outputs) == 3
    assert len(model.seen) == 3

    # Turn 1: just the opening user message.
    assert model.seen[0] == [{"role": "user", "content": "answer without commas"}]
    # Turn 2: turn 1 plus the model's own reply plus the new user turn.
    assert model.seen[1] == [
        {"role": "user", "content": "answer without commas"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "now end with the exact phrase"},
    ]
    # Turn 3: the whole history, in order, ending on the third user turn.
    assert [m["role"] for m in model.seen[2]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert model.seen[2][3] == {"role": "assistant", "content": "reply two"}
    assert model.seen[2][4]["content"] == "and mention alpha"


def test_infer_stops_at_the_last_turn_of_a_two_turn_conversation():
    model, outputs = _run_infer(["reply one", "reply two"], n_turns=2)
    assert len(outputs) == 2
    assert len(model.seen) == 2


def test_infer_continues_after_a_choiceless_response():
    # An aborted turn has no choices; the conversation must carry on with an
    # empty assistant turn rather than raising.
    model, outputs = _run_infer([None, "reply two", "reply three"])
    assert len(outputs) == 3
    assert model.seen[1][1] == {"role": "assistant", "content": ""}


def test_infer_keeps_the_answered_turns_when_a_later_turn_fails():
    # Turn 3's request dies. The two turns that did answer are real results and
    # must survive -- failing the sample would delete them from turn 1 and 2's
    # denominators as well.
    model, outputs = _run_infer(["reply one", "reply two", "reply three"], fail_at=3)
    assert len(model.seen) == 3, "the third turn must still have been attempted"
    assert len(outputs) == 2
    assert [o.texts[0] for o in outputs] == ["reply one", "reply two"]


def test_infer_propagates_a_failure_on_the_very_first_turn():
    # Nothing answered means nothing to salvage: the sample fails outright.
    with pytest.raises(RuntimeError, match="503"):
        _run_infer(["reply one", "reply two", "reply three"], fail_at=1)


def test_preprocess_records_only_the_first_turn():
    task = _task()
    raw = _sample()
    record = asyncio.run(task.preprocess(raw, _Ctx(raw)))
    assert record["prompt"] == [{"role": "user", "content": "answer without commas"}]
    assert record["extra"]["n_turns"] == 3
    # The later user turns are recorded even though they are not sent yet.
    assert len(record["extra"]["later_turn_prompts"]) == 2
    # Reference is the per-turn cumulative constraint list.
    assert [len(ids) for ids in record["reference"]] == [1, 2, 3]


def test_feedback_grades_each_turn_against_its_cumulative_constraints():
    # Turn 1 satisfies its only constraint; turn 2 misses the end phrase;
    # turn 3 satisfies all three.
    judgement = _judge(
        [
            "no commas here.",
            "still no commas but the wrong ending",
            "alpha appears without commas. the end.",
        ]
    )
    m = judgement["metrics"]
    assert m["turn_1_strict_follow_all"] is True
    assert m["turn_2_strict_follow_all"] is False
    assert m["turn_3_strict_follow_all"] is True
    # Turn 2 kept 1 of its 2 constraints.
    assert m["turn_2_strict_instruction_level"] == pytest.approx(0.5)
    # A single failed turn makes the whole conversation incorrect.
    assert judgement["rollouts"][0]["correct"] is False


def test_feedback_handles_a_two_turn_conversation():
    judgement = _judge(["no commas.", "no commas. the end."], n_turns=2)
    assert judgement["extra"]["n_turns"] == 2
    assert judgement["extra"]["n_answered"] == 2
    assert "turn_3" not in judgement["extra"]
    assert judgement["rollouts"][0]["correct"] is True


def test_feedback_grades_only_the_turns_that_were_answered():
    # A three-turn conversation whose walk stopped after turn 2: the unreached
    # turn must be absent, not a zero, or an infrastructure failure would read as
    # a model that stopped following instructions.
    judgement = _judge(["no commas.", "no commas. the end."], n_turns=3, n_answered=2)
    m = judgement["metrics"]
    assert "turn_1_strict_follow_all" in m
    assert "turn_2_strict_follow_all" in m
    assert "turn_3_strict_follow_all" not in m
    assert "turn_3" not in judgement["extra"]
    assert judgement["extra"]["n_turns"] == 3
    assert judgement["extra"]["n_answered"] == 2
    # Both graded turns passed, so score stays 1.0 -- averaged over the turns
    # that were graded, not over the three the dataset ships.
    assert judgement["score"] == pytest.approx(1.0)
    # ...but the conversation never reached its last turn, so it cannot claim the
    # strictest reading.
    assert judgement["rollouts"][0]["correct"] is False


def test_missing_turn_response_scores_zero_rather_than_raising():
    # An aborted turn arrives as "": every constraint fails, nothing raises.
    judgement = _judge(["", "", ""])
    m = judgement["metrics"]
    assert m["turn_1_strict_follow_all"] is False
    assert m["turn_1_strict_instruction_level"] == 0.0


def test_postprocess_reports_no_extraction_only_when_every_turn_is_blank():
    task = _task()
    _Out = _StubOutput

    blank = asyncio.run(task.postprocess([_Out(""), _Out("  ")], _Ctx(_sample())))
    assert blank["rollouts"][0]["extracted"] is False
    # An all-blank conversation still *answered* two turns; `prediction` collapses
    # to None, so only `extra` can still say so.
    assert blank["extra"]["n_answered"] == 2

    # A partly-blank conversation is still a real answer that scores badly.
    partial = asyncio.run(task.postprocess([_Out(""), _Out("hi")], _Ctx(_sample())))
    assert partial["rollouts"][0]["extracted"] is True
    assert partial["rollouts"][0]["prediction"][1] == {"turn": 2, "response": "hi"}
    assert partial["extra"]["n_answered"] == 2


def test_postprocess_records_each_turn_finish_reason():
    # `detect_truncated_output` flags the rollout but not which turn caused it --
    # its indices are rollout positions. This record is where the turn is visible.
    task = _task()
    outputs = [
        _StubOutput("fine", finish_reasons=("stop",)),
        _StubOutput("cut off here", finish_reasons=("length",)),
        # A backend that reports nothing leaves the field None, not [].
        _StubOutput("no reason given", finish_reasons=None),
    ]
    record = asyncio.run(task.postprocess(outputs, _Ctx(_sample())))
    assert record["extra"]["finish_reasons"] == [["stop"], ["length"], []]


def test_report_pools_instruction_counts_rather_than_averaging_rates():
    # Turn 1 mixes conversations with unequal constraint counts, which is the only
    # situation where the two aggregations differ:
    #   `_sample()`      turn 1 -> 1 constraint,  1 followed -> [True]
    #   `_wide_sample()` turn 1 -> 2 constraints, 1 followed -> [True, False]
    # pooled instruction-level = (1 + 1) / (1 + 2) = 2/3
    # averaged per-sample rate = (1/1 + 1/2) / 2   = 3/4
    # prompt-level             = (1 + 0) / 2       = 1/2
    # Strict and loose agree here (stripping '*' or an edge line cannot conjure
    # the missing end phrase), so overall = mean(1/2, 2/3, 1/2, 2/3) = 7/12.
    narrow = _judge(["no commas here."], n_turns=1)
    wide = _judge(["no commas but no ending"], raw=_wide_sample(n_turns=1))

    assert narrow["extra"]["turn_1"]["strict"]["follow_instruction_list"] == [True]
    assert wide["extra"]["turn_1"]["strict"]["follow_instruction_list"] == [True, False]

    report = asyncio.run(_task().report([_Final(narrow), _Final(wide)], []))
    assert report["turn_1_strict_instruction_level_accuracy"] == pytest.approx(
        2 / 3 * 100
    )
    assert report["turn_1_strict_prompt_level_accuracy"] == pytest.approx(50.0)
    assert report["turn_1_all_languages_overall"] == pytest.approx(7 / 12 * 100)
    # The averaged-rates reading would land on 62.5 instead; asserting it is *not*
    # that value is what gives this test teeth.
    assert report["turn_1_all_languages_overall"] != pytest.approx(62.5)


def test_report_skips_absent_turns():
    # One three-turn conversation and one two-turn conversation.
    three = _judge(
        ["no commas.", "no commas. the end.", "alpha without commas. the end."]
    )
    two = _judge(["no commas.", "no commas. the end."], n_turns=2)
    report = asyncio.run(_task().report([_Final(three), _Final(two)], []))

    # Turn 3 exists for only one of the two conversations.
    assert report["turn_1_prompts_number"] == 2
    assert report["turn_3_prompts_number"] == 1
    # Both conversations pass everything, so every cell is 100.
    assert report["turn_1_all_languages_overall"] == pytest.approx(100.0)
    assert report["turn_1_English_overall"] == pytest.approx(100.0)
    assert report["score"] == pytest.approx(100.0)
    assert report["fails"] == 0


def test_report_counts_only_the_conversations_that_reached_a_turn():
    # Same three-turn shape twice, but one walk stopped after turn 2. Turn 3's
    # denominator must drop to 1 rather than scoring the unreached turn zero.
    full = _judge(
        ["no commas.", "no commas. the end.", "alpha without commas. the end."]
    )
    cut = _judge(["no commas.", "no commas. the end."], n_turns=3, n_answered=2)
    report = asyncio.run(_task().report([_Final(full), _Final(cut)], []))

    assert report["turn_1_prompts_number"] == 2
    assert report["turn_2_prompts_number"] == 2
    assert report["turn_3_prompts_number"] == 1
    # Every answered turn passed, so the truncation costs no accuracy -- it only
    # narrows what turn 3's number is an average over.
    assert report["turn_3_all_languages_overall"] == pytest.approx(100.0)


def test_report_turn_range_follows_the_data():
    # Nothing reached turn 3, so no turn-3 key exists at all -- the range is read
    # off the run rather than hardcoded to (1, 2, 3).
    two = _judge(["no commas.", "no commas. the end."], n_turns=2)
    report = asyncio.run(_task().report([_Final(two)], []))
    assert report["turn_2_prompts_number"] == 1
    assert not [key for key in report if key.startswith("turn_3")]


def test_report_survives_a_run_where_every_sample_failed():
    report = asyncio.run(_task().report([], [object(), object()]))
    assert report["fails"] == 2
    assert report["score"] == 0.0
    # Even with nothing to pool, the headline still says which column it is and
    # which population it averaged over -- a bare report on a wholly failed run
    # is exactly where a reader most needs both.
    assert report["score_key"] == "all_turns_all_languages_overall"
    assert report["all_turns_all_languages_overall"] == 0.0
    assert report["denominator_policy"] == "judged"


def test_report_declares_the_headline_column_and_its_population():
    # `score` is the mean of the per-turn overalls, so unlike its IFEval and
    # IFBench siblings it is not a copy of one of four co-equal rates -- it is a
    # cell of its own, and `score_key` names it rather than crowning a turn.
    three = _judge(
        ["no commas.", "no commas. the end.", "alpha without commas. the end."]
    )
    two = _judge(["no commas.", "no commas. the end."], n_turns=2)
    report = asyncio.run(_task().report([_Final(three), _Final(two)], [object()]))

    assert report["score_key"] == "all_turns_all_languages_overall"
    # The named cell and the headline are the same number, not two computations
    # that could drift.
    assert report["score"] == report["all_turns_all_languages_overall"]
    # And it really is the mean over the turns present, not turn 3 or turn 1.
    turns = [
        report[f"turn_{turn}_all_languages_overall"]
        for turn in (1, 2, 3)
        if f"turn_{turn}_all_languages_overall" in report
    ]
    assert report["score"] == pytest.approx(sum(turns) / len(turns))

    # `judged`: the failure went to `fails`, and did not enter any turn's
    # denominator as a conversation that followed nothing.
    assert report["denominator_policy"] == "judged"
    assert report["fails"] == 1
    assert report["turn_1_prompts_number"] == 2


@pytest.fixture
def fresh_punkt_cache():
    module._download_punkt_tab_once.cache_clear()
    yield
    module._download_punkt_tab_once.cache_clear()


@pytest.mark.usefixtures("fresh_punkt_cache")
def test_ensure_punkt_tab_skips_the_download_when_already_staged(monkeypatch):
    import nltk

    downloads: list[str] = []
    monkeypatch.setattr(nltk.data, "find", lambda path: path)
    monkeypatch.setattr(nltk, "download", lambda name, **_kw: downloads.append(name))
    module._ensure_punkt_tab()
    assert downloads == []


@pytest.mark.usefixtures("fresh_punkt_cache")
def test_ensure_punkt_tab_downloads_punkt_tab_when_missing(monkeypatch):
    import nltk

    staged = {"ok": False}
    downloads: list[str] = []

    def fake_find(path):
        if not staged["ok"]:
            raise LookupError(path)
        return path

    def fake_download(name, **_kwargs):
        downloads.append(name)
        staged["ok"] = True
        return True

    monkeypatch.setattr(nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk, "download", fake_download)
    module._ensure_punkt_tab()
    # `punkt_tab`, not the legacy `punkt`: on nltk >= 3.9 staging that one
    # satisfies neither of the two NLTK call sites the checkers reach.
    assert downloads == ["punkt_tab"]


@pytest.mark.usefixtures("fresh_punkt_cache")
def test_ensure_punkt_tab_raises_when_the_resource_stays_missing(monkeypatch):
    import nltk

    downloads: list[str] = []

    def fake_find(path):
        raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk, "download", lambda name, **_kw: downloads.append(name))

    # An offline run must stop here naming the resource, not surface later as one
    # LookupError per graded sample.
    with pytest.raises(LookupError):
        module._ensure_punkt_tab()
    # And it must only have tried to fetch it once.
    assert downloads == ["punkt_tab"]
