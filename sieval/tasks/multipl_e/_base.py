"""Shared protocol, grading and reporting for the MultiPL-E task family.

MultiPL-E is HumanEval and MBPP translated into other programming languages.
A problem is a partial program that stops at the function's opening; the model
continues it; the graded program is the prompt, the continuation and the
suite's own assertions, compiled and run in the target language. Nothing is
parsed or compared -- the exit code is the verdict.

Four tasks come out of two independent choices, and this module holds
everything both axes share.

**Suite** (``humaneval`` / ``mbpp``) differs only in which dataset the task
binds to. Upstream's two suites have different language sets -- HumanEval was
translated to Dart and MBPP was not.

**Protocol** is the axis that actually changes behaviour, and upstream
documents two:

* **Completion** (``_base_gen``) is upstream's primary path, the one the README
  leads with. The prompt is the partial program verbatim, the continuation is
  truncated at the language's own ``stop_tokens``, and the graded program is
  ``prompt + completion``.
* **Chat** (``_gen``) follows ``dataset_builder/chat_completions.py``, which
  exists because "the original MultiPL-E prompts are sub-optimal for all chat
  models, and are not compatible with chat-only models". It asks for the WHOLE
  program back -- prefix repeated verbatim plus the completion -- and then
  grades the model's own restatement of the prefix, **discarding the dataset
  prompt entirely**. Upstream's comment calls this "Federico's hack" and gives
  the reason: models paraphrase docstrings and parameter names, and failing a
  program over that would be "a spurious error".

That second point is the one thing a MultiPL-E port cannot get away with
guessing. Prepending the dataset prompt to a chat reply that already repeats it
duplicates the function definition, which does not compile in most of these
languages -- a whole-benchmark zero that looks like a very bad model. The two
protocols therefore differ in exactly one place, :meth:`prompt_prefix`, and
that place is why they are separate tasks rather than a flag.

Program assembly is upstream's, separator included
(``evaluation/src/main.py``)::

    program = problem["prompt"] + completion + "\\n" + problem["tests"]

Execution goes to the SiEval code-eval API rather than to upstream's own
container: sieval already routes every code-exec task through that service, and
a second execution mechanism for one benchmark would be a second thing to
secure. What the service can run is asked at setup, not assumed -- see
:meth:`MultiPLETask.setup`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar, cast, override

import httpx
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    RolloutJudgement,
    Task,
    TaskContext,
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
    sampling_report,
    ungated_intervals,
)

# Commit-pinned: `main` moves, and a divergence is argued against the code that
# was actually read. `main.py` is where the graded program is assembled.
MULTIPL_E_COMMIT = "3025a531af7450e7df8b96fe0440e9804480bbad"
MULTIPL_E_UPSTREAM_URL = (
    f"https://github.com/nuprl/MultiPL-E/blob/{MULTIPL_E_COMMIT}/evaluation/src/main.py"
)
CHAT_UPSTREAM_URL = (
    f"https://github.com/nuprl/MultiPL-E/blob/{MULTIPL_E_COMMIT}/"
    "dataset_builder/chat_completions.py"
)

# Client-side walls for one grade request. Both exist so that no grade can wait
# forever: grading is synchronous on one shared event loop, so an unbounded wait
# stalls the session rather than the sample. Sized to sit above the evaluator's
# own server-side budget, whose largest row today is c++'s 15s run plus a
# separate 60s compile.
GRADE_WALL_HEADROOM = 90.0
DEFAULT_GRADE_WALL = 180.0

# `reference_impl.notes`, factored the way the four tasks are: one note per
# protocol, one per suite that has something of its own to say. Held here rather
# than imported leaf-to-leaf, so a shared sentence has one home.
_SHARED_NOTES = (
    "A program passes iff it builds and exits 0; nothing is parsed or compared. "
    "Per-language build and run commands follow "
    "`evaluation/src/eval_<lang>.py` and live in the code-eval service, not "
    "here. DIVERGENCES: (1) execution runs through the SiEval code-eval API "
    "rather than upstream's own container, so only languages that service "
    "deploys can be graded — validated end-to-end for cpp / js / sh / pl; the "
    "task refuses at setup, before spending inference, for any requested "
    "language it cannot run. (2) Upstream's four-way failure taxonomy "
    "(SyntaxError / AssertionError / ReferenceError / Exception) is reduced to "
    "the build-vs-run split the service's one boolean can carry, reported as "
    "`n_build_errors` / `n_execution_errors` / `timeouts`; scoring is "
    "unaffected. (3) upstream's perl rule that an otherwise-passing run fails "
    "when its output contains `ERROR` is implemented in the service, and is "
    "unreachable through the shipped test templates — 0 of 161 `humaneval-pl` "
    "rows mention `ERROR` and all 161 signal failure with `exit 1` — so it "
    "fires only on model stdout. (4) the per-step wall clocks are the service's "
    "rather than upstream's: every `safe_subprocess.run` call upstream takes its "
    "flat 15s default, the build included, while here an interpreted row (bash, "
    "perl) gets 3s to run and a c++ build gets 60s. Both directions are "
    "reachable — a program needing between 3s and 15s fails here and passes "
    "upstream; a compile upstream abandons at 15s completes here — though the "
    "run wall has measured headroom: a pure-bash prime sieve to n=20000 "
    "finishes in 0.6s. "
    "UNMEASURED: no published-score alignment run has been made, so the score "
    "is not yet anchored to a MultiPL-E table — hence `experimental`. "
    "Upstream's per-language row counts differ (161 c++, 158 bash), so the "
    "per-language `pass@1_<tag>` keys are the paper-comparable ones; the "
    "headline is a size-weighted pool over whatever languages ran, with "
    "`pass@1_macro` beside it."
)

COMPLETION_NOTES = (
    "Follows MultiPL-E's base-model path: the prompt is upstream's partial "
    "program verbatim, the continuation is truncated at the row's own "
    "`stop_tokens` (per language) with upstream's `stop_at_stop_token`, and "
    'the graded program is `prompt + completion + "\\n" + tests` — '
    "`evaluation/src/main.py`'s own assembly, separator included. "
    + _SHARED_NOTES
    + " PROTOCOL: upstream publishes pass@1 from 20 completions at temperature "
    "0.2 (`--completion-limit 20`), and pass@10/pass@100 need temperature 0.8 "
    "and more samples. This task defaults to n=1, so match upstream with "
    "`args.n: 20` plus `infer_args.temperature: 0.2`."
)

CHAT_NOTES = (
    "Follows MultiPL-E's chat path (`dataset_builder/chat_completions.py`). "
    "The instruction — repeat the prefix verbatim, complete only the "
    "incomplete function, add no tests or usage examples — is upstream's "
    "`Completion` signature text carried verbatim, and `postprocess_completion` "
    "is reproduced exactly: a reply starting with a fence loses its first and "
    "last lines, then the reply is split at `len(prompt)` characters and only "
    "the tail is scanned for `stop_tokens`. Critically, the graded program uses "
    'upstream\'s blank prompt — `"" + completion + "\\n" + tests` — so the '
    "MODEL's restatement of the prefix is what compiles. Upstream's comment "
    "gives the reason: models paraphrase docstrings and parameter names, and "
    "failing a program over that would be a spurious error. Prepending the "
    "dataset prompt instead would define the function twice and fail to compile "
    "in most of these languages. A further divergence: upstream drives this "
    "through DSPy `ChainOfThought`, so its rendered prompt carries DSPy's field "
    "markers and a reasoning field; reproducing that would pin the port to a "
    "DSPy version rather than to MultiPL-E, so the instruction text and field "
    "descriptions are carried in a plain chat prompt and the scaffolding is "
    "not. " + _SHARED_NOTES + " PROTOCOL: upstream's chat script takes "
    "`--max-completions` with `--temperature 0.2` for pass@1 (20 in its own "
    "example). This task defaults to n=1; match upstream with `args.n: 20` plus "
    "`infer_args.temperature: 0.2`. No `stop` is sent, deliberately — the reply "
    "is asked to contain the prefix, whose text holds this language's stop "
    "tokens."
)

MBPP_SUITE_NOTES = (
    " SUITE: MBPP, 23 languages — upstream translated HumanEval to Dart and "
    "MBPP not, so there is no `mbpp-dart` config. Upstream's translator also "
    "rewrote the word `python` inside MBPP's docstrings along with the code, so "
    'prompts read "Write a cppthon function" and similar: 132/397 `mbpp-cpp` '
    "rows, 131/382 `mbpp-sh`, 131/396 `mbpp-pl`, 132/397 `mbpp-js` — about a "
    "third of each language. Carried as-is: the unqualified task tracks "
    "upstream, and a prompt fix would be a `datasets/` concern rather than a "
    "task one."
)

# MultiPL-E's registry tag -> the `lang` the code-eval API knows the language
# by. A pure vocabulary map, and complete on purpose: what the DEPLOYED
# evaluator can run is a different question, answered by its own
# `GET /languages` at setup. Keeping the two apart means a language lights up by
# adding a toolchain to the evaluator, with no edit here -- and that a language
# it cannot run is named as such instead of scoring zero.
#
# Names on the right follow the evaluator's existing spelling (`javascript`,
# `typescript`, `python` -- English names, not tags), which is this service's
# vocabulary and only ever this service's: nothing here is passed to upstream's
# `containerized_eval`, whose `EVALUATORS` keys are the tags. Left column
# upstream's, right column the evaluator's, and the map is the seam.
EVALUATOR_LANG_BY_TAG: dict[str, str] = {
    "adb": "ada",
    "clj": "clojure",
    "cpp": "cpp",
    "cs": "csharp",
    "d": "d",
    "dart": "dart",
    "elixir": "elixir",
    "go": "go",
    "hs": "haskell",
    "java": "java",
    "jl": "julia",
    "js": "javascript",
    "lua": "lua",
    "ml": "ocaml",
    "php": "php",
    "pl": "perl",
    "r": "r",
    "rb": "ruby",
    "rkt": "racket",
    "rs": "rust",
    "scala": "scala",
    "sh": "bash",
    "swift": "swift",
    "ts": "typescript",
}

# Upstream's `Completion` signature docstring and field descriptions, from
# `dataset_builder/chat_completions.py`. Verbatim, because the instruction to
# repeat the prefix is what makes the blank-prompt grading path work: a prompt
# that fails to ask for the prefix produces replies that grade as a bare
# function body with no signature.
CHAT_INSTRUCTION = (
    "I will give you the prefix of an incomplete program that you must "
    "complete.\n"
    "In your response, do not alter the prefix but repeat it exactly, and when\n"
    "you complete the program just complete the incomplete function. Do\n"
    "not write additional functions, tests, or usage examples.\n\n"
    "You MUST return a full program with the prefix verbatim and the completion."
)


def stop_at_stop_token(decoded_string: str, stop_tokens: Sequence[str]) -> str:
    """Truncate at the first stop token, upstream's function verbatim.

    ``decoded_string`` must NOT include the prompt -- upstream carries that as a
    shouted warning, and the reason is that the prompt contains stop tokens of
    its own (``\\n}`` for c++, ``\\nfunction `` for js), so scanning it would
    truncate the completion to nothing at position zero.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


class MultiPLETask[TSample](
    Task[
        TSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one; `list[float]` carries an interval, and
        # `dict[str, str]` the `ci95_units` map naming each interval's unit.
        dict[str, float | str | list[float] | dict[str, str]],
    ]
):
    """Grading, reporting and the setup probe. Protocol lives in the subclasses.

    Leaves supply the sample type; subclasses supply :attr:`suite`,
    :attr:`eval_source` and the protocol methods.
    """

    # `humaneval` / `mbpp` -- only used in messages; the dataset decides the rows.
    suite: ClassVar[str] = ""
    # The code-eval API's direct-run alias for this suite. Both aliases reach
    # the same code path; sending the matching one keeps the service's log
    # readable rather than attributing every MBPP run to human-eval.
    eval_source: ClassVar[str] = ""

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        k: int = 1,
        n: int = 1,
        max_concurrency: int = 4,
        # Upstream's own per-program wall (`libeval.run_without_exn` uses 5s,
        # `safe_subprocess.run` 15s). Left at the evaluator's per-language
        # default when None, which is where a compiled language's larger budget
        # is declared -- overriding it here with one number would charge c++ the
        # same wall as bash.
        timeout: float | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}. "
                "Raise the task arg `n` (tasks.<name>.args.n) to at least k — "
                "setting `n` on the model is silently overridden call-time."
            )
        self._k = k
        self._n = n
        self._timeout = timeout
        # Filled on first use, off the live test set -- see `_language_column`.
        self._languages_by_sample_id: list[str] | None = None
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    # ---- protocol hooks ----------------------------------------------------

    def prompt_prefix(self, raw) -> str:
        """What the graded program starts with, before the model's text.

        The whole difference between the two protocols. Completion returns the
        dataset prompt; chat returns ``""``, because the model was asked to
        repeat the prefix itself and upstream grades that copy.
        """
        raise NotImplementedError

    # ---- shared lifecycle --------------------------------------------------

    @override
    async def setup(self) -> None:
        """Refuse the run if the evaluator cannot execute a requested language.

        Checked BEFORE any inference, because the alternative is silent and
        expensive: an unsupported language returns a clean failed verdict per
        sample, so the run completes, costs a full generation budget, and
        reports ``pass@1 = 0`` with no errors — indistinguishable from a model
        that simply cannot write Racket. Naming the gap up front is the only
        point at which it is cheap.
        """
        requested = self._requested_languages()
        unmapped = sorted(set(requested) - set(EVALUATOR_LANG_BY_TAG))
        if unmapped:
            raise ValueError(
                f"MultiPL-E {self.suite}: no code-eval language is mapped for "
                f"tag(s) {', '.join(unmapped)}. Add them to "
                f"EVALUATOR_LANG_BY_TAG."
            )

        advertised = await self._advertised_languages()
        missing = sorted(
            {tag for tag in requested if EVALUATOR_LANG_BY_TAG[tag] not in advertised}
        )
        if missing:
            runnable = sorted(
                tag for tag, lang in EVALUATOR_LANG_BY_TAG.items() if lang in advertised
            )
            raise ValueError(
                f"MultiPL-E {self.suite}: the code-eval service at "
                f"{self._code_eval_api} cannot run "
                f"{', '.join(missing)} "
                f"(needs {', '.join(EVALUATOR_LANG_BY_TAG[t] for t in missing)}). "
                f"Deploy those toolchains, or restrict the run with the dataset "
                f"arg `languages`. Runnable there now: "
                f"{', '.join(runnable) or '(none)'}."
            )

    async def _advertised_languages(self) -> frozenset[str]:
        """The `lang` values the deployed evaluator says it accepts.

        A service without the endpoint is refused rather than assumed capable:
        it predates table-driven languages, so it cannot run any of the ones
        this benchmark needs beyond js/ts, and treating "cannot ask" as "can
        run" is what turns a deployment gap into a run of zeros.
        """
        # `rstrip` first: a trailing slash would otherwise make `rsplit` drop the
        # empty last segment instead of `evaluations`, probing
        # `.../evaluations/languages` -- a 404, reported as "this deployment
        # predates table-driven languages" when the deployment is in fact fine.
        url = self._code_eval_api.rstrip("/").rsplit("/", 1)[0] + "/languages"
        try:
            resp = await self._http_client.get(url, timeout=10.0)
        except Exception as e:
            raise RuntimeError(
                f"MultiPL-E {self.suite}: cannot reach the code-eval service at "
                f"{url} ([{type(e).__name__}] {e}). It must be running before a "
                f"run starts — grading a MultiPL-E program needs its language's "
                f"toolchain, so there is nothing to fall back to."
            ) from e
        if resp.status_code == 404:
            raise RuntimeError(
                f"MultiPL-E {self.suite}: the code-eval service at {url} has no "
                f"`/languages` endpoint, so it predates table-driven language "
                f"support and cannot run this benchmark. Redeploy from "
                f"`vendor/code-evaluator` (see docker/Dockerfile.multipl-e)."
            )
        resp.raise_for_status()
        payload = resp.json()
        return frozenset(payload.get("data") or ())

    def _language_column(self) -> list[str]:
        """The whole ``language`` column, read once and cached.

        One column read rather than a row read per sample: an Arrow row access
        pulls the row group its column chunk lives in, so placing every sample
        one at a time would re-pay that for each of them, twice over in
        ``_per_language_metrics``. The test set is fixed for a run, so caching
        it cannot go stale.
        """
        if self._languages_by_sample_id is None:
            test_set = self._dataset.test_set
            if not test_set or "language" not in test_set.column_names:
                raise ValueError(
                    f"MultiPL-E {self.suite}: the test set carries no `language` "
                    f"column; grading cannot pick a toolchain per problem."
                )
            self._languages_by_sample_id = [
                str(value) for value in test_set["language"]
            ]
        return self._languages_by_sample_id

    def _requested_languages(self) -> tuple[str, ...]:
        """Language tags present in the test set, in first-seen order."""
        seen: dict[str, None] = {}
        for value in self._language_column():
            seen.setdefault(value, None)
        return tuple(seen)

    def _language_of(self, ctx: TaskContext) -> str:
        """The language of *ctx*'s problem, read off the live dataset.

        Read from the dataset rather than from the persisted judgement, for the
        reason :meth:`Task.problem_groups` gives: a resume rebuilds every
        context through ``make_context`` off the same index, so this resolves
        identically fresh and resumed — and it is the only route that also works
        for a FAILED sample, which has no judgement to read a language out of.
        Without that, a failure could not be attributed to a language and the
        per-language denominators would silently disagree with the headline's.
        """
        column = self._language_column()
        sample_id = ctx.sample_id
        if isinstance(sample_id, int) and 0 <= sample_id < len(column):
            return column[sample_id]
        raise ValueError(
            f"MultiPL-E {self.suite}: sample {sample_id!r} cannot be placed in "
            f"the test set, so its language is unknown. A per-language rate "
            f"computed over an unplaceable sample would be wrong in a way the "
            f"report cannot show."
        )

    def _grade_wall(self) -> float:
        """The client-side wall for one grade request. Never unbounded.

        The evaluator enforces its own budget server-side, so this only has to
        sit ABOVE it — a client wall below it would report a network timeout for
        a program the service was still compiling. But it must EXIST: grading is
        synchronous on one shared event loop, so a service that hangs or a
        connection that stalls would take the whole session down rather than one
        sample.

        :data:`GRADE_WALL_HEADROOM` is added to a caller-supplied budget;
        :data:`DEFAULT_GRADE_WALL` covers the case where the budget is the
        evaluator's own per-language default, which this side does not know.
        """
        if self._timeout is None:
            return DEFAULT_GRADE_WALL
        return self._timeout + GRADE_WALL_HEADROOM

    @override
    async def shutdown(self) -> None:
        await self._http_client.aclose()

    # ---- grading -----------------------------------------------------------

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        language = str(raw["language"])
        lang = EVALUATOR_LANG_BY_TAG[language]
        prefix = self.prompt_prefix(raw)

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            idx = rollout["index"]
            # An unextractable completion is None here but "" on the wire, so
            # the evaluator still runs the program and reports a real build
            # failure rather than the run skipping a verdict.
            program = prefix + (rollout.get("prediction") or "") + "\n" + raw["tests"]
            payload = {
                "uuid": f"{idx}-{time.perf_counter_ns()}",
                "source": self.eval_source,
                "lang": lang,
                "code": program,
            }
            if self._timeout is not None:
                payload["timeout"] = self._timeout
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json=payload,
                    timeout=self._grade_wall(),
                )
                resp.raise_for_status()
                res = resp.json()
                data = res["data"] or {}
                rollouts.append(
                    build_rollout_judgement(
                        idx,
                        res["status"],
                        extra={
                            "msg": res["msg"],
                            # Persisted so a per-language rate can be recomputed
                            # from the shard files alone, without the dataset.
                            "language": language,
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
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {} ({}): [{}] {}",
                    idx,
                    language,
                    type(e).__name__,
                    e,
                )
                raise e

        return True, build_judgement_record(
            None,  # the reference is the test suite, not a value
            rollouts,
            extra={"name": str(raw["name"]), "language": language},
        )

    # ---- reporting ---------------------------------------------------------

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            # The declarations belong on this path too, so an empty report is
            # not less readable than a full one. No interval pair: there is
            # nothing to estimate, and a zeroed population would read as
            # measured.
            return {
                "score": 0.0,
                "pass@1": 0.0,
                "fails": 0.0,
                "n_languages": 0.0,
                SCORE_KEY_FIELD: "pass@1",
                DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            }

        buckets = self._failure_buckets(finals)
        # `votes=False`: two correct programs are not one answer, so there is
        # nothing well-defined to take a majority over (RFC #74).
        rolled = sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=total,
            votes=False,
            score_key="pass@1",
            grouping=self.problem_groups(finals),
        )
        # Read back out of the shared block, so `score` cannot drift from it.
        pass_at_1 = rolled["pass@1"]
        metrics: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": pass_at_1,
            "fails": float(len(fails)),
            "pass@1": pass_at_1,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        metrics |= buckets
        metrics |= self._per_language_metrics(finals, fails)
        # Outside the n>1 gate, because the metrics they bracket are: `pass@1`
        # is published at every budget, and so is the headline copied from it.
        metrics |= ungated_intervals(rolled, metrics=("score", "pass@1"))
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`.
            metrics.update(rolled)
        # Outside the gate: extraction health is a fact about the parser, not
        # about the draw, and n=1 is where a stopped extractor hides longest.
        return metrics | health_metrics(finals)

    def _failure_buckets(self, finals) -> dict[str, float]:
        """Split failed rollouts by the stage that failed.

        Upstream sorts a failure four ways (``SyntaxError`` / ``AssertionError``
        / ``ReferenceError`` / ``Exception``); the code-eval API answers with one
        boolean and a message, so what survives is the split a caller cannot
        reconstruct from the verdict — build versus run. Read off the message
        prefixes the service documents, the same way the HumanEval tasks read
        theirs for `timeouts`.

        ``n_execution_errors`` keeps the name it has in every other task that
        executes a prediction; the build bucket is new because no other task has
        a compile step to lose a program in.

        Matched on the message's PREFIX, not by substring. The evaluator's
        messages are a small closed vocabulary it documents, but their tails
        carry the compiler's and the program's own output — so a build error
        whose diagnostic happens to contain the word "timeout" (``error: no
        member named 'timeout'``) reads as a timed-out run under a substring
        test, in the one bucket that suggests the model's program was slow
        rather than broken. A build that exceeds ITS wall lands in the build
        bucket rather than under ``timeouts``: the run never started, and
        build-versus-run is the split these three keys exist to carry.
        """
        timeouts = build_errors = execution_errors = 0
        for final in finals:
            feedback = final.feedback_result
            if feedback is None:
                continue
            for rollout in feedback["rollouts"]:
                if rollout["correct"]:
                    continue
                # A null msg from the evaluator is absent on disk -- default it.
                msg = (rollout["extra"].get("msg") or "").lower()
                if msg.startswith(("failed: build timeout", "failed [build exit")):
                    build_errors += 1
                elif msg.startswith("failed: timeout"):
                    timeouts += 1
                else:
                    execution_errors += 1
        return {
            "timeouts": float(timeouts),
            "n_build_errors": float(build_errors),
            "n_execution_errors": float(execution_errors),
        }

    def _per_language_metrics(self, finals, fails) -> dict[str, float]:
        """``pass@1`` per language, plus the unweighted mean over languages.

        MultiPL-E publishes one column per language and no single number, so the
        per-language keys are the comparable ones and the headline is a POOLED
        micro-average over whatever languages the run covered. That pooling is
        size-weighted, and upstream's row counts differ by language (161 for
        c++, 158 for bash), so ``pass@1_macro`` is published beside it rather
        than left for a reader to reconstruct — they are different numbers and
        only the per-language columns are comparable with a paper.

        Each language's rate comes from :func:`sampling_report` over that
        language's own samples, not from a hand-rolled mean: the estimator is
        then the headline's by construction, so a macro cannot drift from the
        pooled number for reasons other than weighting.

        No intervals on these keys. Each language is its own axis with its own
        population, and one ``n_problems`` cannot carry all of them — the same
        reason ``mmmlu_kshot_clp`` leaves its per-locale breakdown bare.
        """
        by_language: dict[str, list] = defaultdict(list)
        for final in finals:
            by_language[self._language_of(final)].append(final)
        requested: dict[str, int] = defaultdict(int)
        for ctx in (*finals, *fails):
            requested[self._language_of(ctx)] += 1

        metrics: dict[str, float] = {"n_languages": float(len(requested))}
        rates: list[float] = []
        for language in sorted(requested):
            language_finals = by_language.get(language, [])
            metrics[f"n_problems_{language}"] = float(requested[language])
            if not language_finals:
                # Every sample of this language failed the pipeline. Under
                # DENOMINATOR_REQUESTED that is a real 0, not a missing cell.
                metrics[f"pass@1_{language}"] = 0.0
                rates.append(0.0)
                continue
            block = sampling_report(
                language_finals,
                n=self._n,
                k=self._k,
                denominator=requested[language],
                votes=False,
                grouping=self.problem_groups(language_finals),
            )
            # `sampling_report`'s value type spans intervals and the units map,
            # but `pass@1` is documented as always present and always a bare
            # rate -- which is why it is the key a headline is read back from.
            rate = cast(float, block["pass@1"])
            metrics[f"pass@1_{language}"] = rate
            rates.append(rate)

        metrics["pass@1_macro"] = sum(rates) / len(rates) if rates else 0.0
        return metrics


class MultiPLECompletionTask[TSample](MultiPLETask[TSample]):
    """Upstream's primary protocol: the model continues a partial program.

    The prompt is the partial program verbatim -- no instruction, no wrapper --
    so this needs a base (completion) model. Decoding params come from the
    model's configured args or per-task ``infer_args``; the task owns only the
    per-sample stop tokens and ``n``.
    """

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            raw["prompt"],
            # No `reference`: the ground truth is a test suite, not a value. It
            # is described at judgement time instead.
            extra={"name": str(raw["name"]), "language": str(raw["language"])},
        )

    @override
    async def infer(self, pre, ctx):
        # Stop tokens are per LANGUAGE, so they come off the sample rather than
        # from a task constant: `\n}` ends a c++ function and would never fire
        # in Python, and `\nfunction ` is js-only.
        return await self.model.agenerate(
            pre["prompt"],
            n=self._n,
            stop=list(ctx.raw_sample["stop_tokens"]),
        )

    @override
    async def postprocess(self, inf, ctx):
        stop_tokens = list(ctx.raw_sample["stop_tokens"])
        predictions: list[str | None] = []
        for text in inf.texts:
            # Truncated here as well as server-side: upstream truncates its own
            # decoded output, and a server that returns the stop sequence (or
            # honours none) would otherwise leave a second function definition
            # in the program.
            completion = stop_at_stop_token(text, stop_tokens)
            # A blank completion normalizes to None so `extracted` stays a real
            # signal -- the program is still built and still graded.
            predictions.append(completion or None)
        return build_prediction_record(predictions)

    @override
    def prompt_prefix(self, raw) -> str:
        return str(raw["prompt"])


class MultiPLEChatTask[TSample](MultiPLETask[TSample]):
    """Upstream's chat protocol, including the blank-prompt grading path.

    The model is asked for the whole program -- prefix repeated verbatim plus
    the completion -- and the graded program is built from the model's own copy
    of the prefix, with the dataset prompt discarded. See the module docstring
    for why that is upstream's rule rather than a shortcut.

    One documented divergence: upstream drives this through DSPy's
    ``ChainOfThought``, whose rendered prompt carries DSPy's own field markers
    and a reasoning field. Reproducing that byte-for-byte would pin the port to
    a DSPy version rather than to MultiPL-E, so the instruction text and field
    descriptions are carried verbatim in a plain chat prompt and the scaffolding
    is not. The instruction is the load-bearing half -- it is what makes the
    reply contain a repeated prefix for the blank-prompt path to grade.
    """

    @override
    async def preprocess(self, raw, ctx):
        language = str(raw["language"])
        return build_prompt_record(
            [
                {"role": "user", "content": CHAT_INSTRUCTION},
                {
                    "role": "user",
                    "content": (
                        f"The programming language of the program to complete: "
                        f"{language}\n\n"
                        f"The prefix of the program to complete:\n{raw['prompt']}"
                    ),
                },
            ],
            # No `reference`: the ground truth is a test suite, not a value.
            extra={"name": str(raw["name"]), "language": language},
        )

    @override
    async def infer(self, pre, ctx):
        # No `stop`: the reply is asked to contain the prefix, whose own text
        # holds this language's stop tokens, so passing them would cut the reply
        # off inside the part it was told to repeat. Truncation happens in
        # postprocess, past the prefix, exactly as upstream does it.
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        prompt = str(ctx.raw_sample["prompt"])
        stop_tokens = list(ctx.raw_sample["stop_tokens"])
        predictions: list[str | None] = []
        for text in inf.texts:
            predictions.append(
                _postprocess_chat_completion(text, prompt, stop_tokens) or None
            )
        return build_prediction_record(predictions)

    @override
    def prompt_prefix(self, raw) -> str:
        # Blank, and this is the whole point: the reply already carries a copy
        # of the prefix, and upstream grades that copy rather than the dataset's
        # -- so prepending the dataset prompt here would define the function
        # twice.
        return ""


def _postprocess_chat_completion(
    completion: str, prompt: str, stop_tokens: Sequence[str]
) -> str:
    """``chat_completions.postprocess_completion``, upstream's logic verbatim.

    Two steps, and the second is stranger than it looks:

    1. A reply that STARTS with a fence is assumed to be one whole Markdown
       block, and its first and last lines are dropped. Upstream does not
       search for a fence anywhere else, and does not check that the last line
       closes one -- so a reply that opens with a fence and then keeps talking
       loses its final line of code. Kept as-is; it is upstream's rule.
    2. The reply is split at ``len(prompt)`` CHARACTERS and only the tail is
       scanned for stop tokens. The split point is a heuristic about the
       *reply*, not a match against the prompt, and the two halves are
       immediately re-joined -- so the only thing it decides is where stop-token
       scanning may begin. That is what keeps a stop token belonging to the
       repeated prefix from truncating the whole program to nothing.
    """
    if completion.startswith("```"):
        lines = completion.split("\n")
        lines = lines[1:-1]
        completion = "\n".join(lines)

    head = completion[: len(prompt)]
    tail = stop_at_stop_token(completion[len(prompt) :], stop_tokens)
    return head + tail


__all__ = [
    "CHAT_INSTRUCTION",
    "CHAT_NOTES",
    "CHAT_UPSTREAM_URL",
    "COMPLETION_NOTES",
    "DEFAULT_GRADE_WALL",
    "EVALUATOR_LANG_BY_TAG",
    "GRADE_WALL_HEADROOM",
    "MBPP_SUITE_NOTES",
    "MULTIPL_E_COMMIT",
    "MULTIPL_E_UPSTREAM_URL",
    "MultiPLEChatTask",
    "MultiPLECompletionTask",
    "MultiPLETask",
    "stop_at_stop_token",
]
