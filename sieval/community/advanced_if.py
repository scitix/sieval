# Upstream: https://github.com/facebookresearch/AdvancedIF/blob/f9d30137c4139d4d9af260ae28108b5afae828c0/judge.py
#
# The judge prompts are deliberately NOT vendored: every file upstream is
# CC-BY-NC-4.0, which cannot be redistributed inside sieval's Apache-2.0 tree.
# The operator stages their own checkout and points SIEVAL_ADVANCED_IF_SRC at it;
# this module loads the prompts from there and contributes only sieval-authored
# scoring code. The benchmark data carries the same terms and is likewise only
# referenced, so running AdvancedIF accepts the upstream terms either way.
"""AdvancedIF rubric-judge assets and scoring kernel.

AdvancedIF (Meta, arXiv:2511.10507) scores a response against expert-written
rubrics: a grader LLM answers every rubric question yes/no and declares whether
the response satisfied all of them.

**The two published rates do not share a denominator** -- upstream computes them
in different places, so they disagree whenever the grader answers a different
number of questions than there are rubrics:

* per-sample ``rubric_level_pass_rate`` (``judge._calc_rubric_level_pass_rate``)
  divides in-range passes by ``len(rubrics)``, the count the *data* carries;
* aggregate ``micro_pass_rate`` (``processor._calculate_stats``) pools every key
  the *grader* emitted, with no range check.

:func:`count_in_range_passes` and :func:`count_all_checks` keep them separate so
each pooled metric matches its own upstream definition.

**Upstream defect, reproduced deliberately.** Upstream routes to the
system-steerability judge on ``benchmark_name == "if_system_steerability_oss"``,
a value the released dataset never contains (it ships
``system_steerability_v2``), so all 507 system-prompt rows go to the plain
user-instruction judge and the CLI's ``--task`` choices match zero rows --
``processor.process_file``'s own docstring uses the released spelling, so the
``if_*_oss`` literals are what went stale. :func:`is_system_steer` keeps the
comparison verbatim: the unqualified task name tracks upstream, defects included.
Correcting it is **measured at +0.00** on the SS pass rate (507 rows, same
responses and grader) while flipping 13.8% of individual verdicts, so no
``_fixed`` variant ships -- it would matter only for per-sample analysis.

Deviations from upstream @ f9d3013, both in grader-reply handling:

* **Reply parsing.** Upstream guarantees JSON with
  ``response_format={"type": "json_object"}``, which sieval cannot force on an
  arbitrary ``ChatModel`` endpoint, so :func:`parse_judgement` falls back to the
  first JSON object in the reply carrying a schema key -- otherwise a grader that
  fences its JSON scores zero everywhere, a harness artifact rather than a
  property of the model under test. The same scan also recovers a verdict wrapped
  in a JSON *array*, which upstream fails the row on (its ``.get`` runs against
  the list). That recovery is the *only* leniency: a reply that decodes but
  carries no usable ``rubrics_check`` still fails the row. What the scan costs is
  documented on :func:`_recover_json_object`.
* **Non-string rubric answers** are stringified rather than raising, which keeps
  the row gradeable where upstream's bare ``.lower()`` aborts it into ``except
  Exception``. A *non-mapping* ``rubrics_check`` is not forgiven the same way --
  it carries no per-question verdicts to preserve, so the row fails as upstream's
  does.

  What this moves depends on the declaration: with NO, ``micro_pass_rate`` alone
  (the stringified answer joins the pooled denominator, where upstream's failed
  row contributes nothing); with **YES it also moves the headline**, since
  upstream fails the whole row while this port honours the declaration and scores
  a pass. Differentially tested against upstream on all 1,645 rows -- 0
  divergences on every other reply shape, and this one never fired in 1,644 live
  gradings, a grader emitting JSON answering with strings.

  A non-string *declaration* parts ways too, but only from a crash: upstream's
  ``evaluate`` succeeds and ``_calculate_stats`` then raises on ``bool.lower()``,
  losing the whole aggregation, where ``str(declared).lower()`` scores the row
  not-satisfied.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

UPSTREAM_COMMIT = "f9d30137c4139d4d9af260ae28108b5afae828c0"
"""Upstream revision the prompts and scoring rules are pinned to."""

UPSTREAM_JUDGE_SHA256 = (
    "415164e9c3cb1e267321fa0561a8d61b81b3ac134a7d018764ae53a6e5a84955"
)
"""sha256 of ``judge.py`` at :data:`UPSTREAM_COMMIT`."""

SRC_ENV_VAR = "SIEVAL_ADVANCED_IF_SRC"
"""Environment variable pointing at the operator's upstream checkout."""

UPSTREAM_MODULE_NAME = "advanced_if_upstream_judge"
""":mod:`sys.modules` name for the operator's ``judge.py``, outside ``sieval.*``."""

# Upstream's literal, verbatim. The released dataset spells the same aspect
# `system_steerability_v2`, so this never matches it -- see the module docstring.
SYSTEM_STEER_BENCHMARK = "if_system_steerability_oss"

RELEASED_SYSTEM_STEER_BENCHMARK = "system_steerability_v2"
"""What the released dataset calls the aspect :data:`SYSTEM_STEER_BENCHMARK` misses."""

# Upstream's reply-schema keys, verbatim. `_recover_json_object` sniffs for them
# and `parse_judgement` reads them, so the two must agree on the spelling.
_CHECKS_KEY = "rubrics_check"
_DECLARATION_KEY = "SATISFIED_ALL_REQUIREMENTS"

_MISSING_SOURCE_HINT = (
    f"AdvancedIF's judge prompts are CC-BY-NC-4.0 and are not redistributed "
    f"with sieval. Clone the upstream harness and point {SRC_ENV_VAR} at it:\n"
    f"  git clone https://github.com/facebookresearch/AdvancedIF\n"
    f"  git -C AdvancedIF checkout {UPSTREAM_COMMIT}\n"
    f"  export {SRC_ENV_VAR}=$PWD/AdvancedIF"
)


@dataclass(frozen=True)
class JudgePrompts:
    """The three prompt templates upstream ``judge.py`` defines."""

    judge_prompt: str
    system_steer_judge_prompt: str
    steer_few_shot_examples: str


@dataclass(frozen=True)
class Judgement:
    """A parsed grader verdict.

    Attributes:
        rubrics_check: Answer keyed by ``question_<n>``, as the grader emitted
            it -- keys are neither renumbered nor range-filtered here, because
            the two pooled rates disagree about which of them count.
        satisfied_all: The grader's all-rubrics-passed declaration.
    """

    rubrics_check: dict[str, str]
    satisfied_all: bool


def is_system_steer(benchmark_name: str) -> bool:
    """Whether *benchmark_name* routes to the system-steerability judge.

    Upstream's comparison, kept verbatim -- which means this returns ``False``
    for every row of the released dataset, including all 507 system-prompt
    ones. That is upstream's behaviour, not an oversight here; see the module
    docstring.
    """
    return benchmark_name == SYSTEM_STEER_BENCHMARK


def _judge_source_path() -> Path:
    raw = os.environ.get(SRC_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(f"{SRC_ENV_VAR} is not set.\n\n{_MISSING_SOURCE_HINT}")
    root = Path(raw).expanduser()
    path = root / "judge.py" if root.is_dir() else root
    if not path.is_file():
        raise RuntimeError(
            f"{SRC_ENV_VAR}={raw} does not contain judge.py (looked at {path})."
            f"\n\n{_MISSING_SOURCE_HINT}"
        )
    return path


@cache
def load_judge_prompts() -> JudgePrompts:
    """Load the upstream prompt templates from the operator's checkout.

    Digest-checked against :data:`UPSTREAM_JUDGE_SHA256` *before* the import, so
    only the pinned bytes ever execute and a score is always attributable to a
    known prompt revision. A mismatch is fatal by design -- these prompts *are*
    the benchmark. Recovery is to check out :data:`UPSTREAM_COMMIT`.
    """
    path = _judge_source_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_JUDGE_SHA256:
        raise RuntimeError(
            f"{path} does not match the pinned AdvancedIF revision.\n"
            f"  expected sha256 {UPSTREAM_JUDGE_SHA256} (commit {UPSTREAM_COMMIT})\n"
            f"  found    sha256 {digest}\n"
            f"Check out the pinned commit:\n"
            f"  git -C {path.parent} checkout {UPSTREAM_COMMIT}"
        )

    # Deliberately not a `sieval.*` name: this is upstream's file, and importing
    # it must not make it addressable as part of this package. Registering it in
    # `sys.modules` at all is what `exec_module` documents.
    spec = importlib.util.spec_from_file_location(UPSTREAM_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AdvancedIF judge module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return JudgePrompts(
        judge_prompt=module.JUDGE_PROMPT,
        system_steer_judge_prompt=module.SYSTEM_STEER_JUDGE_PROMPT,
        steer_few_shot_examples=module.STEER_FEW_SHOT_EXAMPLES,
    )


def parse_conversation(conversation_history: str | list) -> list:
    """Decode the dataset's ``conversation_history`` into role/content dicts.

    Only the two keys the judge prompt reads are kept, so the record persisted
    downstream is the message list actually sent to the model. The return type
    is left bare because this list becomes a record's ``prompt`` (a
    ``JSONValue``), and ``list`` is invariant.
    """
    messages = (
        json.loads(conversation_history)
        if isinstance(conversation_history, str)
        else conversation_history
    )
    if not isinstance(messages, list):
        raise ValueError(
            f"conversation_history must decode to a list, got {type(messages).__name__}"
        )
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


def parse_rubrics(prompt_metadata: str | dict) -> list[str]:
    """Extract the rubric list from the dataset's ``prompt_metadata``.

    ``rubrics`` is itself sometimes a JSON-encoded string rather than a list,
    which is why upstream decodes it a second time.
    """
    metadata = (
        json.loads(prompt_metadata)
        if isinstance(prompt_metadata, str)
        else prompt_metadata
    )
    if "rubrics" not in metadata:
        raise ValueError("Rubrics not found in prompt_metadata")
    rubrics = metadata["rubrics"]
    if isinstance(rubrics, str):
        rubrics = json.loads(rubrics)
    return [str(rubric) for rubric in rubrics]


def format_conversation_history(messages: list[dict]) -> str:
    """Render prior turns as upstream's ``role [turn]: content`` block.

    The final message is dropped: the dataset's ``conversation_history`` ends on
    the user prompt being answered, and that turn is passed to the prompt
    separately. The turn counter advances on assistant messages, so a
    user/assistant pair shares one number.
    """
    formatted = []
    turn = 1
    for message in messages[:-1]:
        formatted.append(f"{message['role']} [{turn}]: {message['content']}")
        if message["role"] == "assistant":
            turn += 1
    return "\n".join(formatted)


def last_user_turn(messages: list[dict]) -> str:
    """The most recent user message, or ``""`` when there is none."""
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def system_prompt_of(messages: list[dict]) -> str:
    """The leading system message, or ``""`` when the turn list has none."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"]
    return ""


def compose_judge_prompt(
    benchmark_name: str,
    messages: list[dict],
    response_text: str,
    rubrics: list[str],
) -> str:
    """Assemble the grader prompt for one sample.

    Routing between the user-instruction and system-steerability judges follows
    :func:`is_system_steer`; the rubric block is JSON with upstream's
    ``indent=4``, which the grader's ``question_<n>`` keys are positional over.
    """
    prompts = load_judge_prompts()
    rubrics_text = json.dumps(rubrics, indent=4)

    if is_system_steer(benchmark_name):
        return prompts.system_steer_judge_prompt.format(
            few_shot_examples=prompts.steer_few_shot_examples,
            system_prompt=system_prompt_of(messages),
            user_prompt_last_turn=last_user_turn(messages),
            response_text=response_text,
            rubrics_text=rubrics_text,
        )
    return prompts.judge_prompt.format(
        full_conversation=format_conversation_history(messages),
        user_prompt_last_turn=last_user_turn(messages),
        response_text=response_text,
        rubrics_text=rubrics_text,
    )


def _loads_json_object(reply: str) -> dict | None:
    try:
        parsed = json.loads(reply)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _iter_json_objects(text: str) -> Iterator[dict]:
    """Decode a JSON object at each ``{`` in *text*, skipping the ones that fail.

    ``raw_decode`` stops at the end of the value it parsed, so trailing prose
    (or a closing code fence) does not matter and no brace counting is needed.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except ValueError:
            pass
        else:
            if isinstance(parsed, dict):
                yield parsed
        start = text.find("{", start + 1)


def _recover_json_object(reply: str) -> dict | None:
    """The first JSON object in *reply* that looks like a grader verdict.

    Every ``{`` is tried in turn, so prose that itself contains braces ("I
    checked {each} rubric") no longer swallows the real object the way a greedy
    ``{.*}`` match did. A candidate carrying one of the schema keys wins
    outright; otherwise the first decodable object stands, which is what a
    grader that answered with a bare object gives.

    Two known costs of scanning, both reachable in a run:

    * **A judge prompt can parse as a reply here**, so a grader that echoes its
      prompt is scored on the verdict its *prompt* carried instead of failing the
      row, and those answers pool into ``micro_pass_rate``. Two sources, and the
      live one is the plain IF prompt every released row composes: the candidate's
      response is interpolated into it verbatim, so any response quoting a verdict
      object puts a complete one there. The system-steer prompt carries four of
      its own in its few-shot block, dormant only because :func:`is_system_steer`
      never fires on the released data -- a routing ``_fixed`` variant has to
      harden this first, but it is not what gates the hazard. The echoing grader
      is the whole precondition; the released data is not a guard.
    * The *first* schema-carrying object wins, so a grader that drafts a verdict
      and then revises it is scored on the draft. Preferring the last one is the
      better reading, but it would move stored runs whose replies went through
      this fallback, so it owes its own measured delta rather than a quiet flip.
    """
    fallback: dict | None = None
    for candidate in _iter_json_objects(reply):
        if _CHECKS_KEY in candidate or _DECLARATION_KEY in candidate:
            return candidate
        if fallback is None:
            fallback = candidate
    return fallback


def parse_judgement(reply: str) -> Judgement | None:
    """Parse a grader reply, or ``None`` when it yields no usable verdict.

    ``None`` is the analogue of upstream's ``JudgeResult(success=False)``: the
    sample counts against the overall pass rate but contributes no rubrics to
    the pooled micro rate.
    """
    parsed = _loads_json_object(reply)
    if parsed is None:
        # Fenced or prose-wrapped JSON -- see the parsing deviation above.
        parsed = _recover_json_object(reply)
    if parsed is None:
        return None

    raw_checks = parsed.get(_CHECKS_KEY, {})
    if not isinstance(raw_checks, dict):
        # Upstream iterates `.items()` unguarded, so a non-mapping raises and the
        # row lands in its `except Exception` path -- a *failed* row, not an empty
        # one. Emptying it here would let the declaration alone score a pass. A
        # *missing* key defaults to `{}` on both sides.
        return None
    # Upstream defaults a missing declaration to "NO" and compares
    # `.lower() == "yes"`: case-insensitive (its few-shot examples answer
    # "Yes"/"No") but neither stripped nor substring-matched, so neither is this.
    declared = parsed.get(_DECLARATION_KEY, "NO")
    return Judgement(
        rubrics_check={str(k): str(v) for k, v in raw_checks.items()},
        satisfied_all=str(declared).lower() == "yes",
    )


def _is_pass(answer: str) -> bool:
    # Upstream's substring test, not equality: rubric answers routinely carry a
    # justification ("The intro is four sentences. No").
    return "yes" in answer.lower()


def count_in_range_passes(rubrics_check: dict[str, str], rubrics: list[str]) -> int:
    """Passes among answers whose ``question_<n>`` indexes a real rubric.

    Numerator of the per-sample ``rubric_level_pass_rate``; keys that are
    unparseable or index past the rubric list are skipped, as upstream does.
    """
    passes = 0
    for key, answer in rubrics_check.items():
        try:
            index = int(key.split("_")[1]) - 1
        except (IndexError, ValueError):
            continue
        if index >= len(rubrics):
            continue
        if _is_pass(answer):
            passes += 1
    return passes


def rubric_level_pass_rate(rubrics_check: dict[str, str], rubrics: list[str]) -> float:
    """Per-sample rubric pass rate, over the rubric count the data carries."""
    return count_in_range_passes(rubrics_check, rubrics) / max(len(rubrics), 1)


def count_all_checks(rubrics_check: dict[str, str]) -> tuple[int, int]:
    """``(answers emitted, answers passed)`` with no range filtering.

    The pooled ``micro_pass_rate`` counts exactly these, so a grader that
    answers fewer questions than there are rubrics shrinks its own denominator.
    """
    return (
        len(rubrics_check),
        sum(1 for answer in rubrics_check.values() if _is_pass(answer)),
    )


def aggregate_metrics(verdicts: list[dict]) -> dict[str, float]:
    """Pool per-rollout verdicts into the two published rates, plus the macro.

    Each entry carries ``satisfied_all``, ``n_checks``, ``n_checks_passed``,
    ``rubric_pass_rate`` and ``judge_unparsed``; an ungradeable rollout reaches
    only the pass-rate denominator, matching upstream.

    ``macro_pass_rate`` is sieval's, not published: it averages the per-sample
    rate, weighing every sample equally where micro weighs every rubric equally,
    and is the one of the two a per-sample reader can reconstruct.

    ``n_judge_unparsed`` is a diagnostic, not a metric: every rate above scores an
    unparseable grader reply exactly as it scores a response that missed every
    rubric, so it is the only number that separates a broken grader from a
    failing model.
    """
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v["satisfied_all"])
    checks = sum(v["n_checks"] for v in verdicts)
    checks_passed = sum(v["n_checks_passed"] for v in verdicts)
    macro = sum(v["rubric_pass_rate"] for v in verdicts)
    unparsed = sum(1 for v in verdicts if v["judge_unparsed"])
    return {
        "overall_pass_rate": passed / total * 100 if total else 0.0,
        "micro_pass_rate": checks_passed / checks * 100 if checks else 0.0,
        "macro_pass_rate": macro / total * 100 if total else 0.0,
        "n_samples": float(total),
        "n_rubric_checks": float(checks),
        "n_judge_unparsed": float(unparsed),
    }
