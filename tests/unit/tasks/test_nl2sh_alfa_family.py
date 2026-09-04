"""Unit tests for both NL2SH-ALFA readings.

No container and no embedding service: the shell-eval call is a capturing double
and the embedding client returns vectors the test chose, which is enough to cover
everything the port decides -- the branch order of part 3, the strictness of the
threshold, what reaches the wire, and which failures must not be scored as wrong
answers. What is left for a Docker host is what the five images actually print.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from types import SimpleNamespace

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.intercode_alfa import PART_CREDIT, TIMEOUT_DURATION
from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.nl2sh_alfa import NL2SHAlfaDataset
from sieval.tasks.nl2sh_alfa_0shot_gen import (
    BASELINE_SYSTEM_PROMPT,
    NL2SHAlfaZeroShotGenTask,
)
from sieval.tasks.nl2sh_alfa_0shot_gen_parse import (
    PARSE_SYSTEM_PROMPT,
    NL2SHAlfaZeroShotGenParseTask,
)
from tests.conftest import HandlerTransport

_GOLD = "ls -al"
_RAW = {
    "nl": "list files in the current directory",
    "bash": _GOLD,
    "bash2": "ls -l",
    "difficulty": 0,
    "query": "list files in the current directory",
    "gold": _GOLD,
    "gold2": "ls -l",
    "fs_id": 3,
}


class _StubChatModel(ChatModel):
    def __init__(self, reply: str = _GOLD):
        super().__init__(model="mock-chat", api_key="fake")
        self._reply = reply

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=(self._reply,) * req.sampling.n)


def _facts(**overrides) -> dict:
    """Execution facts for a sample where both sides did the same nothing."""
    return {
        "gold_output": "total 0\n",
        "model_output": "total 0\n",
        "gold_status": "",
        "model_status": "",
        "gold_hashes": {},
        "model_hashes": {},
        "gold_exit_ok": True,
        "model_exit_ok": True,
        "gold_timed_out": False,
        "model_timed_out": False,
    } | overrides


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _CapturingShell:
    """Stands in for the code-eval service's shell route."""

    def __init__(
        self, facts: dict | None = None, *, status: bool = True, msg: str = ""
    ):
        self.bodies: list[dict] = []
        self.deadlines: list[float] = []
        self._payload = {
            "status": status,
            "msg": msg,
            "data": facts if facts is not None else _facts(),
        }

    async def post(self, url, *, json, timeout):
        _ = url
        self.bodies.append(json)
        self.deadlines.append(timeout)
        return _Response(self._payload)

    async def aclose(self) -> None:
        return None


class _CapturingEmbeddings:
    def __init__(self, vectors: list[list[float]]):
        self.inputs: list[str] = []
        self._vectors = list(vectors)

    async def create(self, *, input, model):
        _ = model
        self.inputs.append(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self._vectors.pop(0))])


class _CapturingEmbedClient:
    def __init__(self, vectors: list[list[float]]):
        self.embeddings = _CapturingEmbeddings(vectors)

    async def close(self) -> None:
        return None


def _task(cls=NL2SHAlfaZeroShotGenTask, reply: str = _GOLD, **kwargs):
    dataset = NL2SHAlfaDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_RAW])})
    )
    return cls(dataset, _StubChatModel(reply), **kwargs)


async def _judge(
    *,
    cls=NL2SHAlfaZeroShotGenTask,
    command: str = _GOLD,
    facts: dict | None = None,
    vectors: list[list[float]] | None = None,
    shell: _CapturingShell | None = None,
    **kwargs,
):
    """Run `feedback` for one rollout and hand back the verdict plus doubles."""
    task = _task(cls, **kwargs)
    await task._http_client.aclose()  # the real client is never used
    shell = shell if shell is not None else _CapturingShell(facts)
    task._http_client = shell
    embed = _CapturingEmbedClient(vectors or [])
    task._embed_client = embed
    try:
        _, judgement = await task.feedback(
            build_prediction_record([command]),
            TaskContext(sample_id=0, raw_sample=_RAW),
        )
    finally:
        await task.shutdown()
    return judgement["rollouts"][0], shell, embed


@pytest.fixture(autouse=True)
def _embed_credential(monkeypatch):
    monkeypatch.setenv("SIEVAL_EMBED_API_KEY", "fake-embed-key")


# --------------------------------------------------------------------------- #
# The two readings
# --------------------------------------------------------------------------- #
def test_only_the_base_prompt_forbids_markdown():
    # The two sentences are the whole textual difference between the readings,
    # and the reason the Base column can be lower for a chatty model.
    assert "will not output markdown" in BASELINE_SYSTEM_PROMPT
    assert "will not output markdown" not in PARSE_SYSTEM_PROMPT
    # Everything before them is shared, verbatim.
    assert BASELINE_SYSTEM_PROMPT.startswith(PARSE_SYSTEM_PROMPT)


@pytest.mark.anyio
async def test_the_prompt_is_a_system_turn_plus_the_bare_instruction():
    for cls, expected in (
        (NL2SHAlfaZeroShotGenTask, BASELINE_SYSTEM_PROMPT),
        (NL2SHAlfaZeroShotGenParseTask, PARSE_SYSTEM_PROMPT),
    ):
        task = _task(cls)
        try:
            record = await task.preprocess(
                _RAW, TaskContext(sample_id=0, raw_sample=_RAW)
            )
        finally:
            await task.shutdown()
        assert record["prompt"] == [
            {"role": "system", "content": expected},
            {"role": "user", "content": _RAW["nl"]},
        ]
        # The reference is the GRADED gold, not the Hub's `bash` column.
        assert record["reference"] == _RAW["gold"]


@pytest.mark.anyio
async def test_only_the_parse_reading_strips_a_fence():
    fenced = "```bash\nls -al\n```"
    for cls, expected in (
        (NL2SHAlfaZeroShotGenTask, fenced),
        (NL2SHAlfaZeroShotGenParseTask, "ls -al"),
    ):
        task = _task(cls, reply=fenced)
        try:
            inferred = await task.infer(
                await task.preprocess(_RAW, TaskContext(sample_id=0, raw_sample=_RAW)),
                TaskContext(sample_id=0, raw_sample=_RAW),
            )
            record = await task.postprocess(
                inferred, TaskContext(sample_id=0, raw_sample=_RAW)
            )
        finally:
            await task.shutdown()
        assert record["rollouts"][0]["prediction"] == expected


@pytest.mark.anyio
async def test_a_blank_reply_normalizes_to_no_prediction():
    task = _task(reply="   \n ")
    try:
        record = await task.postprocess(
            await task.infer(
                await task.preprocess(_RAW, TaskContext(sample_id=0, raw_sample=_RAW)),
                TaskContext(sample_id=0, raw_sample=_RAW),
            ),
            TaskContext(sample_id=0, raw_sample=_RAW),
        )
    finally:
        await task.shutdown()
    assert record["rollouts"][0].get("prediction") is None


# --------------------------------------------------------------------------- #
# What reaches the shell-eval service
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_the_request_carries_the_filesystem_the_sample_belongs_to():
    _, shell, _ = await _judge()
    (body,) = shell.bodies
    # A misrouted fs_id would score against a different filesystem and lose the
    # run silently, so it travels with every request rather than being config.
    assert body["fs_id"] == _RAW["fs_id"]
    assert body["gold"] == _GOLD
    assert body["command"] == _GOLD
    assert body["timeout"] == float(TIMEOUT_DURATION) == 10.0
    # The HTTP deadline must outlast two commands under that wall.
    assert shell.deadlines[0] > body["timeout"] * 2


@pytest.mark.anyio
async def test_an_unextracted_prediction_still_gets_executed():
    # "" is a real verdict (an empty command), not a skipped rollout -- the same
    # convention human_eval and livecodebench use.
    _, shell, _ = await _judge(command="")
    assert shell.bodies[0]["command"] == ""


@pytest.mark.anyio
async def test_a_refusing_service_raises_instead_of_scoring_zero():
    # A broken grader must not be indistinguishable from a wrong answer. Under
    # DENOMINATOR_REQUESTED the sample is charged as wrong either way, so
    # raising costs nothing and keeps the cause visible.
    with pytest.raises(RuntimeError, match="refused sample"):
        await _judge(shell=_CapturingShell(status=False, msg="fs_id mismatch"))


# --------------------------------------------------------------------------- #
# Part 3's branch order
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_identical_commands_short_circuit_before_any_output_compare():
    rollout, _, embed = await _judge(
        command=_GOLD, facts=_facts(model_output="anything else")
    )
    assert rollout["extra"]["p3_branch"] == "command_match"
    assert rollout["extra"]["reward_parts"]["output"] == PART_CREDIT
    assert rollout["correct"]
    # No embedding call was made -- the FEH never saw this sample.
    assert embed.embeddings.inputs == []


@pytest.mark.anyio
async def test_identical_outputs_score_without_the_heuristic():
    rollout, _, embed = await _judge(command="ls -la")
    assert rollout["extra"]["p3_branch"] == "output_match"
    assert rollout["correct"]
    assert embed.embeddings.inputs == []


@pytest.mark.anyio
async def test_an_empty_output_on_either_side_scores_zero_before_embedding():
    for facts in (
        _facts(model_output="", gold_output="x"),
        _facts(model_output="x", gold_output=""),
    ):
        rollout, _, embed = await _judge(command="ls -la", facts=facts)
        assert rollout["extra"]["p3_branch"] == "empty_output"
        assert rollout["extra"]["reward_parts"]["output"] == 0.0
        assert not rollout["correct"]
        # The FEH is never handed a pair one side of which is empty.
        assert embed.embeddings.inputs == []


@pytest.mark.anyio
async def test_differing_outputs_reach_the_embedding_and_pass_above_threshold():
    rollout, _, embed = await _judge(
        command="ls -la",
        facts=_facts(model_output="a\n", gold_output="b\n"),
        vectors=[[1.0, 0.0], [1.0, 0.0]],
    )
    assert rollout["extra"]["p3_branch"] == "embed"
    assert rollout["extra"]["similarity"] == pytest.approx(1.0)
    assert rollout["correct"]
    # Two separate calls, gold first, each with its own output -- upstream makes
    # them one at a time rather than batching a pair.
    assert embed.embeddings.inputs == ["b\n", "a\n"]


@pytest.mark.anyio
async def test_a_dissimilar_pair_fails():
    rollout, _, _ = await _judge(
        command="ls -la",
        facts=_facts(model_output="a\n", gold_output="b\n"),
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )
    assert rollout["extra"]["similarity"] == pytest.approx(0.0)
    assert not rollout["correct"]


@pytest.mark.anyio
async def test_the_threshold_comparison_is_strict():
    # `similarity > eval_param`, not >=. Tested by pushing the threshold to the
    # maximum a cosine can reach, which avoids asserting a float boundary.
    rollout, _, _ = await _judge(
        command="ls -la",
        facts=_facts(model_output="a\n", gold_output="b\n"),
        vectors=[[1.0, 0.0], [1.0, 0.0]],
        embed_threshold=1.0,
    )
    assert rollout["extra"]["similarity"] == pytest.approx(1.0)
    assert not rollout["correct"]


@pytest.mark.anyio
async def test_the_embedded_text_is_truncated_at_a_thousand_characters():
    rollout, _, embed = await _judge(
        command="ls -la",
        facts=_facts(model_output="a" * 4000, gold_output="b" * 4000),
        vectors=[[1.0, 0.0], [1.0, 0.0]],
    )
    assert rollout["extra"]["p3_branch"] == "embed"
    assert [len(text) for text in embed.embeddings.inputs] == [1000, 1000]


# --------------------------------------------------------------------------- #
# Parts 1 and 2
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_a_filesystem_divergence_fails_a_matching_output():
    # The metric is a conjunction: identical output does not rescue a sample
    # whose filesystem effect differed.
    rollout, _, _ = await _judge(
        command="ls -la", facts=_facts(model_status="?? workspace/extra.txt")
    )
    assert rollout["extra"]["reward_parts"]["file_diff"] == 0.05
    assert rollout["extra"]["reward_parts"]["output"] == PART_CREDIT
    assert not rollout["correct"]
    assert rollout["extra"]["diff_extra"] == [("workspace/extra.txt", "??")]


@pytest.mark.anyio
async def test_a_shared_change_with_a_different_hash_fails():
    rollout, _, _ = await _judge(
        command="ls -la",
        facts=_facts(
            model_status="?? workspace/a.txt",
            gold_status="?? workspace/a.txt",
            model_hashes={"workspace/a.txt": "aaa  workspace/a.txt"},
            gold_hashes={"workspace/a.txt": "bbb  workspace/a.txt"},
        ),
    )
    assert rollout["extra"]["reward_parts"]["file_diff"] == PART_CREDIT
    assert rollout["extra"]["reward_parts"]["file_changes"] == 0.0
    assert not rollout["correct"]


@pytest.mark.anyio
async def test_a_shared_change_with_the_same_hash_passes():
    rollout, _, _ = await _judge(
        command="ls -la",
        facts=_facts(
            model_status="?? workspace/a.txt",
            gold_status="?? workspace/a.txt",
            model_hashes={"workspace/a.txt": "aaa  workspace/a.txt"},
            gold_hashes={"workspace/a.txt": "aaa  workspace/a.txt"},
        ),
    )
    assert rollout["correct"]
    assert rollout["extra"]["n_hashes_matched"] == 1


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_a_missing_embedding_credential_fails_at_construction(monkeypatch):
    # The FEH is not an optional axis here, so a run without a credential is
    # misconfigured rather than reduced -- and finding that out after paying for
    # 300 completions is the failure this guard exists to prevent.
    monkeypatch.delenv("SIEVAL_EMBED_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SIEVAL_EMBED_API_KEY"):
        _task()


# --------------------------------------------------------------------------- #
# report()
# --------------------------------------------------------------------------- #
def _final(correct: bool, **extra) -> TaskContext:
    detail = {
        "reward": 1.0 if correct else 0.67,
        "reward_parts": {"file_diff": 0.33, "file_changes": 0.33, "output": 0.33},
        "p3_branch": "embed",
        "similarity": 0.9,
        "gold_timed_out": False,
        "model_timed_out": False,
        "model_exit_ok": True,
    } | extra
    return TaskContext(
        sample_id=0,
        raw_sample=_RAW,
        feedback_result=build_judgement_record(
            _GOLD, [build_rollout_judgement(0, correct, extra=detail)]
        ),
    )


@pytest.mark.anyio
async def test_report_declares_its_headline_and_denominator():
    task = _task()
    try:
        report = await task.report([_final(True), _final(False)], [])
    finally:
        await task.shutdown()
    assert report["score"] == report["accuracy"] == 0.5
    assert report["score_key"] == "accuracy"
    assert report["denominator_policy"] == "requested"
    # The interval and its population arrive together, and the alias shares it.
    assert report["n_problems"] == 2
    assert "score_ci95" in report and "accuracy_ci95" in report
    # `ci95_units` is keyed by the metric, not by the interval key, and the
    # alias declares the same population -- one number, one interval.
    assert report["ci95_units"] == {"score": "n_problems", "accuracy": "n_problems"}


@pytest.mark.anyio
async def test_a_pipeline_failure_is_charged_as_wrong():
    task = _task()
    try:
        report = await task.report(
            [_final(True)], [TaskContext(sample_id=1, raw_sample=_RAW)]
        )
    finally:
        await task.shutdown()
    assert report["fails"] == 1
    assert report["score"] == 0.5


@pytest.mark.anyio
async def test_report_publishes_the_branch_mix_and_gold_timeouts():
    task = _task()
    try:
        report = await task.report(
            [
                _final(True, p3_branch="command_match"),
                _final(True, p3_branch="output_match"),
                _final(False, p3_branch="empty_output"),
                _final(False, p3_branch="embed", gold_timed_out=True),
            ],
            [],
        )
    finally:
        await task.shutdown()
    # The two short-circuit branches score full credit without consulting the
    # heuristic, so their share bounds how much of the headline it decided.
    assert report["n_p3_command_match"] == 1
    assert report["n_p3_output_match"] == 1
    assert report["n_p3_empty_output"] == 1
    assert report["n_p3_embed"] == 1
    # A gold that hits the wall this port adds is a fidelity problem, not a
    # model problem, so it is published rather than folded into the score.
    assert report["n_gold_timeouts"] == 1
    assert report["mean_reward"] == pytest.approx((1.0 + 1.0 + 0.67 + 0.67) / 4)
