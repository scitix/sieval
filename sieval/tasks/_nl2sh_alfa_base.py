"""Shared 0-shot protocol for NL2SH-ALFA's two published readings.

Both readings score through one grader -- InterCode-ALFA's, at ``icalfa`` 0.3.6 --
and differ only in the prompt and in whether the reply is stripped of a markdown
fence. So everything but those two lives here, and each leaf module is metadata
plus a system prompt.

**The metric.** ``submit_command`` returns 1 iff ``reward == 1``, where
``reward = 0.01 + p1 + p2 + p3`` and each part is capped at 0.33 -- an exact float
comparison, safe because those four addends sum to 1.0 with no representation
error, and the reason the score is a strict conjunction:

* ``p1`` -- the two ``git status --short`` states must agree exactly; one
  divergent entry already drops the part to 0.05.
* ``p2`` -- for paths both sides added/untracked/copied, the ``md5sum`` output
  must match. Full credit when there is no such path.
* ``p3`` -- functional equivalence of the outputs, decided in four branches:
  identical command strings, then identical outputs, then a zero for an empty
  output on either side, and only then the FEH.

**The FEH is exec + mxbai-embed at threshold 0.75**, which is what upstream's
own reproduction scripts pass (``eval_mode="embed", eval_param=0.75``) and what
the paper states Table 5 was produced with. It is *not* ``submit_command``'s
default (``eval_mode="openai"``, ``eval_param="gpt-4-0613"``): that mode scores
higher as a heuristic (0.95 vs 0.90 F1 on upstream's own FEH benchmark) but no
published model number was produced with it, and the model it names is retired.

**Execution happens in the code-eval service**, over ``POST /shell-evaluations``,
because the grader needs a specific Linux image with a git-committed baseline
filesystem and it runs model-authored shell commands. The service returns raw
execution facts -- both outputs, both status listings, per-path hash output --
and the arithmetic above happens here, so the embedding call, its similarity and
the per-part breakdown all land in the judgement record. ``fs_id`` selects the
image and travels with the sample; the service rejects a request whose ``fs_id``
is not the one it hosts, because a misrouted sample would score zero silently.

**Divergences from upstream**, each also in ``reference_impl.notes``:

* One container per image instead of two peers. Upstream runs the gold in a
  second container so the model can never perturb it; the service resets the
  tracked tree before each of the two commands, which preserves that within a
  sample. Cross-sample residue in gitignored paths survives -- as it does
  upstream, whose containers persist across all 300 samples.
* The 10 s wall (``utils.TIMEOUT_DURATION``) is applied to the gold command too.
  Upstream bounds only the model's; an unbounded gold would hang the service.
  This is an execution-safety bound, so it owes evidence that it never binds --
  ``n_gold_timeouts`` is published for exactly that.
* The embedding is served over an OpenAI-compatible endpoint rather than Ollama's
  ``/api/embeddings``, so the weights are the Hub's rather than Ollama's f16 GGUF
  conversion of them. Both calls stay separate, as upstream's are.

**Two upstream rewrites are reproduced, not repaired.** ``clean_cmd`` wraps a
command in double quotes and docker-py ``shlex.split``s the result, which
rewrites 37 of the 300 golds and truncates one (index 230) into a syntax error;
and ``exec_action`` resolves the argument of any model reply starting with ``cd``
against the working directory, so ``cd ~`` reaches bash as ``cd /~``. Both are
text transforms applied before anything runs, both change the verdict on replies
that carry them, and both live in the service -- see
``sieval.community.intercode_alfa.command_argv`` for the first and the service's
``model_action`` for the second. The ``cd`` rewrite reaches only the model's
command: upstream runs the gold by a path that never touches ``exec_action``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
import time
from typing import ClassVar, override

import httpx
from loguru import logger
from openai import AsyncOpenAI

from sieval.community.intercode_alfa import (
    DEFAULT_EMBED_THRESHOLD,
    ICALFA_VERSION,
    PART_CREDIT,
    TIMEOUT_DURATION,
    file_change_score,
    file_diff_score,
    is_correct,
    output_similarity,
    parse_bash,
    parse_status,
    shared_changes,
    total_reward,
    truncate_for_embedding,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    RolloutJudgement,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
    interval_metrics,
)
from sieval.datasets import NL2SHAlfaDatasetSample

#: Upstream's harness URL, cited by both leaves' `reference_impl`.
NL2SH_ALFA_HARNESS_URL = (
    "https://github.com/westenfelder/InterCode-ALFA/blob/"
    "2d3a69473a68569828ab0b4859073ef4a0ae482c/src/icalfa/envs/bash/bash_env.py"
)
#: The paper, cited by both leaves.
NL2SH_ALFA_PAPER_URL = "https://arxiv.org/abs/2502.06858"

#: What every `reference_impl.notes` for this benchmark has to say, whichever
#: reading it is. The reading-specific half is prepended by each leaf.
NL2SH_ALFA_SHARED_NOTES = (
    f"Graded by InterCode-ALFA {ICALFA_VERSION} (the version the paper states "
    "Table 5 was produced with), vendored as sieval.community.intercode_alfa: "
    "reward = 0.01 + git-status agreement (0.33) + md5 agreement on commonly "
    "added paths (0.33) + output equivalence (0.33), correct iff the sum is "
    "exactly 1.0. FEH = exec + mxbai-embed-large at cosine > 0.75, upstream's "
    "own reproduction setting, NOT submit_command's retired gpt-4-0613 default. "
    "300 test rows (the card's '600 pairs' counts two verified commands per "
    "instruction). GRADED GOLD comes from the harness's own vendored table, not "
    "from the Hub's `bash` column: the two differ on 2 of 300 rows (index 38 "
    "echo -n vs echo, index 100 'length < 20' vs '< 40'), and 3 more rows have "
    "reworded instructions. Upstream's clean_cmd + docker-py shlex.split "
    "rewrites 37 golds and truncates index 230 into a syntax error; reproduced "
    "deliberately, as is exec_action's rewrite of any model reply starting with "
    "'cd' (its argument is resolved against the tree root, so 'cd ~' executes "
    "as 'cd /~' and fails). DIVERGENCES: one container per image with a reset "
    "before each command instead of two peer containers; the 10s command wall "
    "also applied to the gold (n_gold_timeouts publishes that it does not "
    "bind); a reply starting with 'cd' but carrying no 'cd ' is reported as a "
    "command that did not execute, where upstream raises out of exec_action and "
    "scores the sample 0 outright; hashing on the second side is restricted to "
    "the first side's changed paths, which leaves the scored set (diff_same) "
    "identical; the "
    "embedding served over an OpenAI-compatible endpoint rather than Ollama, so "
    "the weights are the Hub's rather than its f16 GGUF conversion. Requires a "
    "shell-eval service per filesystem image (SIEVAL_SHELL_EVAL_API) and an "
    "embedding endpoint (SIEVAL_EMBED_API / SIEVAL_EMBED_API_KEY)."
)

#: Default endpoint of the code-eval service's shell route, overridable so the
#: five per-image deployments can be addressed one run at a time.
_DEFAULT_SHELL_EVAL_API = "http://localhost:11451/shell-evaluations"
#: The embedding endpoint, sharing `t_eval_before_calling_0shot_gen`'s two
#: variables so one credential covers every embedding-scored task.
_DEFAULT_EMBED_API = "https://console.siflow.cn/model-api"
#: Upstream's model name for the FEH, as Ollama spells it.
_DEFAULT_EMBED_MODEL = "mxbai-embed-large"

#: Which p3 branch decided a rollout. Published as counts, because the branch mix
#: is how a reader tells a real equivalence rate from a string-match rate.
_P3_BRANCHES = ("command_match", "output_match", "empty_output", "embed")


class NL2SHAlfaSharedZeroShotGenTask(
    Task[
        NL2SHAlfaDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key` and `denominator_policy`,
        # which name a column and a population rather than measuring one;
        # `list[float]` carries the interval and `dict[str, str]` its unit map.
        dict[str, float | str | list[float] | dict[str, str]],
    ]
):
    """The shared protocol; leaves supply the prompt and the parsing decision."""

    #: The system turn. Both readings put the instruction in a bare user turn.
    SYSTEM_PROMPT: ClassVar[str]
    #: Whether the reply passes through upstream's markdown-fence stripper.
    PARSE_MARKDOWN: ClassVar[bool]

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
        timeout: float = float(TIMEOUT_DURATION),
        max_concurrency: int = 4,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._embed_model = embed_model
        self._embed_threshold = embed_threshold
        self._timeout = timeout
        self._shell_eval_api = os.getenv(
            "SIEVAL_SHELL_EVAL_API", _DEFAULT_SHELL_EVAL_API
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )
        # Checked here rather than at the first call: the FEH is not an optional
        # axis for this benchmark -- it is how part 3 is decided whenever the two
        # outputs differ -- so a missing credential is a misconfigured run, not a
        # reduced one. An explicit "" would skip the OpenAI client's own
        # OPENAI_API_KEY fallback and then name that variable in the error, which
        # would not help: the endpoint is an embedding service, not OpenAI's.
        api_key = os.getenv("SIEVAL_EMBED_API_KEY")
        if not api_key:
            raise ValueError(
                "NL2SH-ALFA's functional-equivalence heuristic embeds both "
                "command outputs, so it needs a credential for the embedding "
                "endpoint: set SIEVAL_EMBED_API_KEY (and SIEVAL_EMBED_API if the "
                f"endpoint is not {_DEFAULT_EMBED_API}). The model defaults to "
                f"{_DEFAULT_EMBED_MODEL!r}, which is what upstream scores with."
            )
        self._embed_client = AsyncOpenAI(
            base_url=os.getenv("SIEVAL_EMBED_API", _DEFAULT_EMBED_API), api_key=api_key
        )

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": raw["nl"]},
            ],
            # The graded gold, which is the harness's copy rather than the Hub's
            # `bash` column -- the module docstring says why they differ.
            reference=raw["gold"],
            extra={"fs_id": raw["fs_id"], "difficulty": raw["difficulty"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        # n=1: upstream translates once per instruction, at temperature 0.
        text = inf.texts[0]
        command = parse_bash(text) if self.PARSE_MARKDOWN else text
        # Empty extraction normalizes to None so `extracted` reports the miss;
        # the command still reaches the shell as "" below, which is a real
        # verdict rather than a skipped rollout.
        return build_prediction_record([command if command.strip() else None])

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        gold: str = raw["gold"]
        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            index = rollout["index"]
            command = rollout.get("prediction") or ""
            facts = await self._execute(index, raw["fs_id"], command, gold)
            verdict, detail = await self._score(command, gold, facts)
            rollouts.append(build_rollout_judgement(index, verdict, extra=detail))
        return True, build_judgement_record(gold, rollouts)

    async def _execute(self, index: int, fs_id: int, command: str, gold: str) -> dict:
        """Run both commands in the image `fs_id` names, returning raw facts.

        Every failure propagates. A shell-eval service that is unreachable, or
        that refuses the request, is a broken grader -- not a model that answered
        wrongly -- and the two must not look alike in a report.
        """
        try:
            resp = await self._http_client.post(
                self._shell_eval_api,
                json={
                    "uuid": f"{index}-{time.perf_counter_ns()}",
                    "fs_id": fs_id,
                    "command": command,
                    "gold": gold,
                    "timeout": self._timeout,
                },
                # Two commands, each with its own wall, plus reset and hashing.
                timeout=self._timeout * 2 + 10,
            )
            resp.raise_for_status()
            res = resp.json()
            if not res["status"]:
                raise RuntimeError(
                    f"shell-eval service refused sample {index}: {res['msg']}"
                )
            return res["data"]
        except Exception as exc:
            logger.warning(
                "Shell evaluation error for sample {} (fs {}): [{}] {}",
                index,
                fs_id,
                type(exc).__name__,
                exc,
            )
            raise

    async def _score(self, command: str, gold: str, facts: dict) -> tuple[bool, dict]:
        """Upstream's three parts over one sample's execution facts."""
        model_status = parse_status(facts["model_status"])
        gold_status = parse_status(facts["gold_status"])
        p1, diff_miss, diff_extra = file_diff_score(model_status, gold_status)
        diff_same = shared_changes(model_status, gold_status)
        p2, n_same = file_change_score(
            diff_same, facts["model_hashes"], facts["gold_hashes"]
        )
        p3, branch, similarity = await self._output_score(
            command, gold, facts["model_output"], facts["gold_output"]
        )
        reward = total_reward(p1, p2, p3)
        return is_correct(reward), {
            "reward": reward,
            "reward_parts": {"file_diff": p1, "file_changes": p2, "output": p3},
            "p3_branch": branch,
            "similarity": similarity,
            "command": command,
            "gold_output": facts["gold_output"],
            "model_output": facts["model_output"],
            "diff_miss": diff_miss,
            "diff_extra": diff_extra,
            "diff_same": diff_same,
            "n_hashes_matched": n_same,
            "model_exit_ok": facts["model_exit_ok"],
            "gold_exit_ok": facts["gold_exit_ok"],
            "model_timed_out": facts["model_timed_out"],
            "gold_timed_out": facts["gold_timed_out"],
        }

    async def _output_score(
        self, command: str, gold: str, model_output: str, gold_output: str
    ) -> tuple[float, str, float | None]:
        """Part 3, in upstream's branch order.

        The order is the whole content of this function: a string match is
        checked before an output match, and an empty output on either side
        scores zero *before* the embedding is reached, so the FEH never sees a
        pair one side of which is empty. Reordering any of it changes the score
        without changing any single branch.
        """
        if gold == command:
            return PART_CREDIT, "command_match", None
        if gold_output == model_output:
            return PART_CREDIT, "output_match", None
        if gold_output == "" or model_output == "":
            return 0.0, "empty_output", None
        gold_embedding = await self._embed(gold_output)
        model_embedding = await self._embed(model_output)
        similarity = float(output_similarity(gold_embedding, model_embedding))
        score = PART_CREDIT if similarity > self._embed_threshold else 0.0
        return score, "embed", similarity

    async def _embed(self, output: str) -> list[float]:
        """One embedding, of one truncated output.

        Separate calls per output, as upstream's ``get_embedding`` makes them:
        batching would be one request, and whether a server's pooling is
        batch-invariant is not a property worth assuming inside a metric.
        """
        resp = await self._embed_client.embeddings.create(
            input=truncate_for_embedding(output), model=self._embed_model
        )
        return resp.data[0].embedding

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        rewards = [
            float(f.feedback_result["rollouts"][0]["extra"]["reward"]) for f in finals
        ]
        correct = [
            1.0 if f.feedback_result["rollouts"][0]["correct"] else 0.0 for f in finals
        ]
        accuracy = sum(correct) / total if total else 0.0
        branches = [
            str(f.feedback_result["rollouts"][0]["extra"]["p3_branch"]) for f in finals
        ]
        metrics: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": accuracy,
            "accuracy": accuracy,
            "fails": len(fails),
            # The mean reward is not the metric -- the metric is the strict
            # conjunction -- but it says how *near* the misses came, which one
            # all-or-nothing rate cannot.
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
            # Execution health. `n_gold_timeouts` is the evidence the 10s wall
            # this port adds to the gold command never binds; a nonzero here is
            # a fidelity problem, not a model problem.
            "n_gold_timeouts": sum(
                1
                for f in finals
                if f.feedback_result["rollouts"][0]["extra"]["gold_timed_out"]
            ),
            "n_model_timeouts": sum(
                1
                for f in finals
                if f.feedback_result["rollouts"][0]["extra"]["model_timed_out"]
            ),
            "n_model_nonzero_exit": sum(
                1
                for f in finals
                if not f.feedback_result["rollouts"][0]["extra"]["model_exit_ok"]
            ),
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # How each rollout's part 3 was decided. Published because the two
        # short-circuit branches score full credit without consulting the FEH at
        # all, so their share bounds how much of the headline the heuristic set.
        for branch in _P3_BRANCHES:
            metrics[f"n_p3_{branch}"] = sum(1 for b in branches if b == branch)
        # Merged with `|` rather than folded with `merge_metrics`, because
        # exactly one fragment here declares units: `interval_metrics` carries
        # the only `ci95_units` map, and `health_metrics` returns plain floats.
        # `merge_metrics` earns its keep when two interval-bearing fragments meet
        # and a plain merge would drop one's declarations; with one, it would only
        # widen this dict's value type to exclude the two string declarations.
        metrics |= interval_metrics(correct, denominator=total, aliases=("accuracy",))
        return metrics | health_metrics(finals)

    @override
    async def shutdown(self):
        await self._http_client.aclose()
        await self._embed_client.close()
