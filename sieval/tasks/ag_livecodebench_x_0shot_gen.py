"""Ag-LiveCodeBench-X — 0-shot competitive programming in an arbitrary language.

Port of `nuprl/Ag-LiveCodeBench-X <https://github.com/nuprl/Ag-LiveCodeBench-X>`_,
the multi-language LiveCodeBench prepared by Agnostics (Boruch-Gruszecki et al.,
ICLR 2026; ``arXiv:2508.04865``). The 499 problems are LiveCodeBench v5's
stdin/stdout subset, so the *same* problem and the *same* test cases score a
solution in any language: which language is asked for is an argument, not a
property of the data.

Two knobs, deliberately independent, because upstream keeps them independent:

* ``language`` is a free-form string spliced into the prompt (``"Lua"``,
  ``"Julia 1.10"``, ``"Lua /nothink"`` — upstream's own README suggests appending
  a version or a thinking-disable directive there).
* ``container_lang`` selects the verifier the evaluator runs. Upstream warns that
  nothing checks the two against each other; the default here derives the tag
  from ``language``'s first word, lowercased, which is a convenience and not a
  guarantee. Override it whenever the derivation is wrong (``"C++"`` is not a
  container tag).

Both are recorded in ``report()``, because the task name cannot hold them: one
registered task measures a different quantity per language, and a leaderboard
column that does not say which language it ran is not comparable with anything.

**The prompt is a frozen DSPy rendering, not a string upstream wrote.** Upstream
declares a ``dspy.Signature`` and wraps it in ``dspy.ChainOfThought``; the bytes
that reach the model are whatever ``dspy==3.0.0b2``'s default ``ChatAdapter``
renders from that declaration, pinned by upstream's ``exclude-newer`` of
2025-08-05. Reproducing it by reading the signature is not possible — two of its
properties are only observable by running it:

* the instructions are re-inserted into the system message **still carrying the
  source docstring's 8-space indentation**, while ``Signature.instructions``
  itself is dedented;
* the field list is assembled with a trailing ``": "`` per line and then
  stripped as a block, so the *first* field line keeps its trailing space and the
  *last* one loses it.

So the rendering was captured once from ``dspy==3.0.0b2`` and frozen below.
Freezing is the point rather than a shortcut: a runtime DSPy dependency would
let a patch release silently re-word every prompt in the fleet, which is the
opposite of what a pinned benchmark is for.
``tests/unit/tasks/test_ag_livecodebench_x_0shot_gen.py`` pins the rendered
bytes by hash.

**Reimplemented rather than vendored, because upstream ships no license.** The
Hub card declares none and the scripts repo has no ``LICENSE``, so nothing lands
in ``sieval/community/``: the response parser and the code extractor below are
written from observed behaviour and pinned by tests, in the shape
``sieval/community/`` is reserved for when a license permits a mirror.

Divergences from upstream, each with what it costs:

* **No JSONAdapter retry.** ``ChatAdapter.__call__`` catches *any* exception from
  its own parse and re-issues the whole call through ``JSONAdapter`` — a second,
  differently shaped prompt with a JSON response format. This port makes one
  call. A reply that omits either marker therefore scores 0 here where upstream
  gets one more chance, so this number is a **lower bound**, and the size of the
  gap is exactly ``n_unextracted``. Read it before comparing against a published
  row; upstream's own README warns that truncation at its default
  ``max_tokens=5000`` is common enough to mention, and truncation is precisely
  what strips the closing marker.
* **The container is run by the evaluator, not by this task.** Upstream shells
  out to ``podman run ... ghcr.io/nuprl/agnostics:<lang>`` (or ``apptainer`` for
  a ``.sif``) from the harness process. Here the task POSTs to sieval's
  code-evaluator, which runs that same container server-side; the invocation is
  the evaluator's deployment config, not a per-request field, so a client cannot
  name an arbitrary image to execute. Same verifier, same protocol, different
  process boundary.
* **No smoke call.** Upstream sends ``lm("Say this is a test!")`` before the run;
  it measures nothing and is not reproduced.
* **No DSPy cache.** Upstream defaults it off (``--enable-dspy-cache`` is opt-in
  and forces concurrency 1), so this matches the default rather than diverging.

Sampling protocol: upstream's ``--num-completions`` defaults to **1**, and
``pass1`` divides successes by rows written — a generation that produced no
solution is still executed and still counted, so the denominator is
``DENOMINATOR_REQUESTED``. Upstream's decoding defaults are ``temperature=0.6``,
``top_p=0.95``, ``max_tokens=5000``, with the README recommending ``0.2`` /
``2048`` plus ``/nothink`` for Qwen 3. Those are model-layer settings in sieval
(``models:`` / ``infer_args``), never this task's.

``status="experimental"``: the port is faithful on the axes above, but no run has
been aligned against a published number yet. The anchor when one is attempted is
the paper's Ag-LiveCodeBench-X table (Qwen-3 4B/8B, DeepSeek-Coder-6.7B-Instruct,
Phi-4-Mini over Lua, Julia, R, OCaml and Fortran); promoting this task to
``stable`` wants one of those cells reproduced within its noise, at the paper's
own decoding settings and language.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import base64
import json
import os
import pickle
import re
import time
import zlib
from typing import override

import httpx
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RolloutJudgement,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
    sampling_report,
)
from sieval.datasets import AgLiveCodeBenchXDatasetSample

# --- The frozen DSPy 3.0.0b2 ChatAdapter rendering -------------------------
#
# Captured from `ChatAdapter().format(ChainOfThought(SolveProblem).predict.signature,
# [], inputs)` under upstream's pinned resolution. Written as adjacent literals
# with explicit `\n` so that the two lines which legitimately END IN A SPACE
# survive an editor and a linter -- as a triple-quoted block they would not.
# Do not reflow: every byte here is on the wire.
_SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `programming_language` (str): \n"
    "2. `problem_statement` (str):\n"
    "Your output fields are:\n"
    "1. `reasoning` (str): \n"
    "2. `solution` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n"
    "\n"
    "[[ ## programming_language ## ]]\n"
    "{programming_language}\n"
    "\n"
    "[[ ## problem_statement ## ]]\n"
    "{problem_statement}\n"
    "\n"
    "[[ ## reasoning ## ]]\n"
    "{reasoning}\n"
    "\n"
    "[[ ## solution ## ]]\n"
    "{solution}\n"
    "\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Solve the following programming problem using the programming "
    "language that I have specified. Use ONLY the\n"
    "        programming language given below."
)

# The user turn is assembled by concatenation rather than `format`: the system
# message above contains literal `{...}` placeholders that are part of the wire
# text, and a problem statement is free to contain braces of its own.
_USER_PREFIX = "[[ ## programming_language ## ]]\n"
_USER_INFIX = "\n\n[[ ## problem_statement ## ]]\n"
_USER_SUFFIX = (
    "\n\nRespond with the corresponding output fields, starting with the field "
    "`[[ ## reasoning ## ]]`, then `[[ ## solution ## ]]`, and then ending with "
    "the marker for `[[ ## completed ## ]]`."
)

# `dspy.adapters.chat_adapter.field_header_pattern`, verbatim.
_FIELD_HEADER_PATTERN = re.compile(r"\[\[ ## (\w+) ## \]\]")

# `ChainOfThought` prepends `reasoning` to the signature's own outputs, and
# ChatAdapter requires *every* output field to be present or it raises.
_OUTPUT_FIELDS = frozenset({"reasoning", "solution"})

# The evaluator answers with the Agnostics `result` value verbatim when the
# container ran and spoke the protocol, and with an `infra:`-prefixed code when
# it did not. Upstream's own `pass1` reports the same split as `run_error_rate`,
# reading it off a stderr suffix; the prefix is a server-side category instead,
# because a client-side classifier over free text decays silently.
# Whether a `result` counts as a pass is the evaluator's call, not read back from
# this string here -- the task reads the response's `status`.
_INFRA_PREFIX = "infra:"


def _render_prompt(
    language: str, question_content: str
) -> tuple[dict[str, str], dict[str, str]]:
    """The two turns DSPy would have sent, system first.

    A tuple rather than a list so the type stays precise: a ``list[dict[str, str]]``
    is not a ``list[JSONValue]`` (``list`` is invariant), and widening the element
    type here would cost every caller the ability to index a message.
    """
    return (
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": _USER_PREFIX
            + language
            + _USER_INFIX
            + question_content
            + _USER_SUFFIX,
        },
    )


def _parse_chat_adapter_fields(completion: str) -> dict[str, str] | None:
    """Reproduce ``ChatAdapter.parse`` for this signature; ``None`` on its raise.

    Upstream wraps the DSPy call in a bare ``except`` that turns any parse
    failure into ``solution = None``, so a raise and a miss are the same outcome
    there. Faithfully reproduced quirks:

    * the **first** occurrence of a field wins; later ones are ignored;
    * content on the same line as a header is kept as that field's first line;
    * unknown headers (``completed`` included) and any prose before the first
      header are dropped;
    * an **indented** header still matches, but the inline-content slice is taken
      with an offset measured on the *stripped* line and applied to the
      *unstripped* one -- so ``"   [[ ## solution ## ]]"`` yields a field whose
      first line is the leftover ``"]]"``. That is upstream's behaviour, bug and
      all, and changing it here would silently move scores.

    ``parse_value`` is not reproduced because it cannot matter: both fields are
    annotated ``str``, for which it is ``str(value)``.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in completion.splitlines():
        match = _FIELD_HEADER_PATTERN.match(line.strip())
        if match:
            header = match.group(1)
            remaining_content = line[match.end() :].strip()
            sections.append((header, [remaining_content] if remaining_content else []))
        else:
            sections[-1][1].append(line)

    fields: dict[str, str] = {}
    for name, lines in sections:
        if name is not None and name not in fields and name in _OUTPUT_FIELDS:
            fields[name] = "\n".join(lines).strip()

    # `ChatAdapter` compares the parsed key set against the signature's and
    # raises when they differ -- a reply carrying `solution` but no `reasoning`
    # is a parse failure, not a partial success.
    if fields.keys() != _OUTPUT_FIELDS:
        return None
    return fields


def _extract_code_from_markdown(markdown: str | None) -> str | None:
    """Reproduce upstream's ``extract_code_from_markdown``.

    First fenced block wins. Note the two behaviours that are easy to
    "improve" by accident: **no** fence at all means the whole reply is treated
    as code, and an **unterminated** fence returns ``None``. The
    ``# Example usage:`` truncation is a Python-shaped heuristic that upstream
    applies to every language, so it applies here too.
    """
    if markdown is None:
        return None
    code_block_start = markdown.find("```")
    if code_block_start == -1:
        # Assume that the whole string is code.
        return markdown

    code_start = code_block_start + 3
    code_block_end = markdown.find("```", code_start)
    if code_block_end == -1:
        return None

    code = markdown[code_start:code_block_end].strip()

    if "# Example usage:" in code:
        code = code.split("# Example usage:")[0]

    # Drop the fence's language tag, which `strip()` above left on line one.
    first_newline = code.find("\n")
    if first_newline > 0:
        code = code[first_newline + 1 :]

    return code.strip()


def _decode_private_test_cases(text: str) -> list[dict[str, str]]:
    """base64 -> zlib -> pickle -> JSON, as LiveCodeBench encodes and upstream
    decodes. No plain-JSON fallback: unlike ``livecodebench/code_generation_lite``,
    every row of this release is encoded, and a fallback would turn a corrupt
    column into a silently empty suite."""
    cases: list[dict[str, str]] = json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
    )
    return cases


@sieval_task(
    name="ag_livecodebench_x_0shot_gen",
    display_name="Ag-LiveCodeBench-X (0-shot)",
    description=(
        "Competitive programming in an arbitrary language, graded by the "
        "Agnostics verifier."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "code-exec", "multi-language"),
    model_type="chat",
    status="experimental",
    reference_kind="procedure",
    reference_impl=ReferenceImpl(
        source="nuprl/Ag-LiveCodeBench-X",
        url=(
            "https://github.com/nuprl/Ag-LiveCodeBench-X/blob/"
            "b7b273ef5710db814d6eff8bca3c4f432b60ad4b/livecodebench_x.py"
        ),
        notes=(
            "Prompt is the frozen dspy==3.0.0b2 ChatAdapter rendering of "
            "upstream's ChainOfThought(SolveProblem) -- captured, not read off "
            "the signature, because the 8-space instruction indent and the "
            "trailing-space asymmetry in the field list are only observable at "
            "runtime; pinned by hash in the task's tests. Response parsing and "
            "code extraction are REIMPLEMENTED, not vendored: upstream ships no "
            "license (no Hub license field, no LICENSE file). Divergences: (1) "
            "upstream's ChatAdapter retries a failed parse through JSONAdapter "
            "with a differently shaped prompt, which this port does not -- so a "
            "reply missing either marker scores 0 here and the gap is exactly "
            "n_unextracted, making this a lower bound; (2) the Agnostics "
            "verifier container is run by sieval's code-evaluator rather than "
            "from the harness process, same protocol and same image, different "
            "process boundary; (3) upstream's pre-run 'Say this is a test!' "
            "smoke call is not reproduced. Upstream defaults: n=1 completion, "
            "temperature 0.6 / top_p 0.95 / max_tokens 5000, container timeout "
            "supplied per run (README uses 15s, and sends that one number as "
            "BOTH the container's timeout_s and the outer process wall). No "
            "sieval run has been aligned against a published number yet, which "
            "is what status='experimental' records."
        ),
    ),
)
class AgLiveCodeBenchXZeroShotGenTask(
    Task[
        AgLiveCodeBenchXDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report names its own score column and the language
        # it measured, neither of which is a measurement.
        dict[str, float | str],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        language: str | None = None,
        container_lang: str | None = None,
        k: int = 1,
        n: int = 1,
        max_concurrency: int = 4,
        timeout: float = 15.0,
    ):
        """*language* is the string spliced into the prompt and has no default:
        this benchmark measures a different thing per language, so picking one
        silently would let two runs of the same task name mean different things.

        *container_lang* names the Agnostics verifier the evaluator should run;
        when omitted it is ``language``'s first word, lowercased. *timeout* is
        the container budget in seconds, sent as upstream sends it -- one number
        serving as both the container's own ``timeout_s`` and the wall the
        evaluator holds the process to.
        """
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        if not language or not language.strip():
            raise ValueError(
                "Ag-LiveCodeBench-X needs an explicit `language` (the string "
                "spliced into the prompt, e.g. 'Lua' or 'Julia 1.10'); it is "
                "the measurement, not a default. Set `container_lang` too when "
                "the verifier tag is not `language`'s first word lowercased."
            )
        self._language = language
        self._container_lang = container_lang or language.split()[0].lower()
        self._k = k
        self._n = n
        self._timeout = timeout
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def preprocess(self, raw, ctx):
        system, user = _render_prompt(self._language, raw["question_content"])
        return build_prompt_record(
            [system, user],
            # No `reference`: the ground truth is a test suite, not a value.
            extra={"question_id": raw["question_id"], "language": self._language},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        predictions: list[str | None] = []
        for choice in inf.texts:
            fields = _parse_chat_adapter_fields(choice)
            code = (
                None
                if fields is None
                else _extract_code_from_markdown(fields["solution"])
            )
            # Empty normalizes to None so `extracted` reports it as a miss, the
            # same rule the LiveCodeBench sibling follows. It is still sent to
            # the evaluator as "" below, which returns a real verdict rather
            # than skipping the rollout.
            predictions.append(code or None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        cases = _decode_private_test_cases(ctx.raw_sample["private_test_cases"])
        inputs = [case["input"] for case in cases]
        outputs = [case["output"] for case in cases]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            idx = rollout["index"]
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "agnostics",
                        "code": rollout.get("prediction") or "",
                        "lang": self._container_lang,
                        "test": {"inputs": inputs, "outputs": outputs},
                        "timeout": self._timeout,
                    },
                    # Beyond the container wall, for network latency and for the
                    # test suite itself: a decoded suite runs to tens of MB, and
                    # writing it is inside this deadline.
                    timeout=self._timeout + 300.0,
                )
                resp.raise_for_status()
                res = resp.json()
                # should raise error if no `status` & `msg` field
                correct, msg = res["status"], res["msg"]
                data = res["data"] or {}
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {}: [{}] {}",
                    idx,
                    type(e).__name__,
                    e,
                )
                raise e

            rollouts.append(
                build_rollout_judgement(
                    idx,
                    correct,
                    extra={
                        # The Agnostics `result` verbatim ("success",
                        # "fail:wrong-output", "fail:timeout", ...) or an
                        # `infra:` code. Stored raw; the taxonomy is the
                        # evaluator's to name, not this client's to guess.
                        "msg": msg,
                        "n_cases": data.get("n_cases"),
                        "n_passed": data.get("n_passed"),
                        "resources": {
                            key: value
                            for key, value in data.items()
                            if key not in ("n_cases", "n_passed")
                        },
                    },
                )
            )

        return True, build_judgement_record(
            None,  # the reference is the test suite below, not a value
            rollouts,
            extra={
                "question_id": ctx.raw_sample["question_id"],
                "n_test_cases": len(cases),
                # Every case is private and every problem is stdio: this release
                # carries no public split and no `fn_name`, so there is nothing
                # to report per problem beyond the count.
                "io_mode": "stdio",
                "container_lang": self._container_lang,
            },
        )

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        # Upstream's `run_error_rate` counterpart: rollouts whose verdict came
        # from the harness failing to run the container rather than from the
        # submitted program. Kept as a count, not a rate, so it reads correctly
        # against whatever denominator the consumer picks.
        n_run_errors = sum(
            1
            for f in finals
            for r in f.feedback_result["rollouts"]
            if (r["extra"].get("msg") or "").startswith(_INFRA_PREFIX)
        )
        # `votes=False`: two correct programs are not one answer, so there is
        # nothing well-defined to take a majority over (RFC #74).
        rolled = sampling_report(
            finals, n=self._n, k=self._k, denominator=total, votes=False
        )
        pass_at_1 = rolled["pass@1"]
        metrics: dict[str, float | str] = {
            "score": pass_at_1,
            "fails": len(fails),
            "n_run_errors": n_run_errors,
            "pass@1": pass_at_1,
            # Which language this column measured. Not a metric, but the column
            # is meaningless without it and the task name cannot carry it.
            "language": self._language,
            "container_lang": self._container_lang,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`.
            metrics.update(rolled)
        # Outside the gate: extraction health is a fact about the parser, not
        # about the draw, and it is what bounds the JSONAdapter divergence.
        return metrics | health_metrics(finals)

    @override
    async def shutdown(self):
        await self._http_client.aclose()
