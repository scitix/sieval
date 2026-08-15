"""IFBench 0-shot generative task, corrected — four repaired constraint checkers.

Identical to :mod:`sieval.tasks.ifbench_0shot_gen` in every respect but one: four
checkers whose code grades something other than what their own instruction text
asks for are looked up from a repaired overlay
(:mod:`sieval.tasks._ifbench_fixed_checkers`, where each defect is stated in
full). Prompts, dataset, decoding, pooling and the reported metric are the base
task's, inherited rather than restated. No item's text is touched — a defect
licenses repairing the verifier, not rewriting the question.

**Measured delta.** A/B over the full official 300-prompt IFBench test set at 8
rollouts (Qwen3-30B-A3B, thinking enabled), replaying stored responses through
each task's own ``feedback()`` and ``report()``:

===============================  ========  ========  =======
metric                           upstream  repaired  delta
===============================  ========  ========  =======
loose prompt-level (``score``)     39.2917   39.3333  +0.0417
strict prompt-level                30.7917   31.2500  +0.4583
loose instruction-level            42.2602   42.2965  +0.0363
strict instruction-level           34.0480   34.4477  +0.3997
===============================  ========  ========  =======

**The delta is small, and that is the finding.** The four ids appear on 28 of the
300 prompts (224 of 2,752 constraint slots); 12 of the 5,504 gradings flip, every
one FAIL→PASS, none PASS→FAIL. A second run over the same 300 prompts at 2
rollouts moves no metric at all. This is the same benchmark with a bounded
correction, not a different one — and without the bound the residual is a guess.

Only two of the four repairs flip anything on these responses:

- ``format:line_indent`` — 11 flips (strict), essentially the whole delta.
- ``words:words_position`` — 1 flip (loose).
- ``ratio:sentence_type`` — **0 flips** across 48 gradings. Quoted-declarative
  recovery fired twice without changing a verdict; the vacuous ``0 == 0`` false
  pass was reachable but never fired, since all 5 gradings with no interrogative
  also had a nonzero declarative count — upstream was False by luck.
- ``words:vowel`` — **0 flips** across 80 gradings. The line-count and
  paragraph-count gates agreed on all 70 non-blank responses (38 single, 32
  multi): this model never produced the soft-wrapped single paragraph that
  separates them.

Both are kept — a checker wrong only on inputs one model happens not to produce
is still wrong, and the next model is not bound by its formatting habits.

Use the unqualified ``ifbench_0shot_gen`` to compare against published IFBench
numbers, which were produced by upstream's code, defects included; use this task
when the grading needs to match what the items actually ask.

**Not a sibling of the IFEval-family repairs.** ``ifeval_0shot_gen_fixed`` and
``multi_if_0shot_gen_fixed`` share one mixin module because their two vendored
copies of google-research's checkers are logic-identical. IFBench is allenai's
own checker set, sharing none of those three defects, so its repairs live in a
module of their own.

**Status.** ``stable``. Its parent stays ``experimental`` because of its
*reproduction* — upstream's temperature=0 protocol does not reproduce, so its
published-number claim is unverified. That is a property of the anchor, which
this variant neither changes nor inherits: it claims none, and ``notes`` forbids
the comparison outright. What a ``_fixed`` owes instead is a quantified delta,
measured above.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import override

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.tasks.ifbench_0shot_gen import IFBenchZeroShotGenTask


@sieval_task(
    name="ifbench_0shot_gen_fixed",
    display_name="IFBench (0-shot, generative, corrected)",
    description="IFBench with four repaired checkers; grading otherwise upstream's.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="ifbench",
    model_type="chat",
    # The parent stays `experimental` because its published anchor is not
    # reproducible. That reason does not carry over: this variant grades
    # differently on purpose and claims no anchor, so what it owes instead is
    # the quantified delta in `notes`.
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="allenai/IFBench",
        url="https://github.com/allenai/IFBench/blob/1091c4c3de6c1f6ed12c012ed68f11ea450b0117/evaluation_lib.py",
        notes=(
            "Deliberate fork of ifbench_0shot_gen, which stays byte-faithful to "
            "upstream. Four vendored checkers grade something other than what "
            "their instruction says and are replaced by the repaired subclasses "
            "in sieval.tasks._ifbench_fixed_checkers. "
            "DIVERGENCES (all four, exhaustive): "
            "(1) format:line_indent drops blank "
            "lines with `for line in lines: if not line.strip(): "
            "lines.remove(line)`, which mutates the list under its own iterator "
            "and so skips one element per removal -- a surviving blank line has "
            "indent 0 and breaks the increasing-indent chain, making the verdict "
            "depend on how many blank lines the response happened to contain. "
            "(2) ratio:sentence_type tests the raw final character on both "
            "counts, endswith('.') and endswith('?'), so any sentence closing "
            "on a quote or bracket goes uncounted; the repair reads the "
            "terminal mark past those closers on both sides, recovering quoted "
            "interrogatives as well as quoted declaratives. It also returns "
            "declarative == 2 * interrogative, so a response with neither "
            "passes vacuously at 0==0. "
            "(3) words:words_position indexes tokens, handling exactly one "
            "trailing punctuation token, so a response ending in two shifts "
            "every position by one. (4) words:vowel counts paragraphs with "
            "split('\\n'), rejecting any soft-wrapped single paragraph on line "
            "count. Nothing else differs: the task overrides one method "
            "(_instruction_dict) of ifbench_0shot_gen and inherits prompting, "
            "decoding, grading, records and report; no item text is changed. "
            "SCORE IMPACT, measured against the unqualified task over the "
            "full official 300-prompt test set at 8 rollouts (Qwen3-30B-A3B, "
            "thinking on), replaying stored responses: loose prompt-level (the "
            "headline) 39.2917→39.3333 (+0.0417), strict prompt-level "
            "30.7917→31.2500 (+0.4583), loose instruction-level "
            "42.2602→42.2965, strict instruction-level 34.0480→34.4477. "
            "The four ids cover 28 of 300 prompts; 12 of 5,504 gradings flip, "
            "all FAIL->PASS, none PASS->FAIL. Only format:line_indent (11 "
            "flips) and words:words_position (1) flip anything on these "
            "responses; ratio:sentence_type and words:vowel flip nothing here "
            "and are kept because the defect is unexercised by this model, not "
            "absent. Do not compare these numbers to published IFBench results, "
            "which were produced by upstream's code. "
            "REPRODUCTION NOTE: 16 of IFBench's 58 checkers fall back to "
            "random.randint/random.choice in build_description when a row omits "
            "the kwarg, so seed `random` in both arms before attributing a flip "
            "to a repair -- an unseeded A/B can move checkers this task does not "
            "touch. None of the four repaired here draws, so the repairs consume "
            "no draw and leave that shared stream unperturbed."
        ),
    ),
)
class IFBenchZeroShotGenFixedTask(IFBenchZeroShotGenTask):
    @override
    def _instruction_dict(self) -> dict[str, type] | None:
        # Imported here, not at module scope, for the reason the base task
        # lazy-imports `evaluation_lib`: registration imports every task module,
        # and must not pay for the vendored checker fork and NLTK.
        from sieval.tasks._ifbench_fixed_checkers import fixed_ifbench_registry

        # A fresh dict per call, never the vendored global: samples grade
        # concurrently, and mutating the shared registry would change how the
        # unqualified task grades in the same session.
        return fixed_ifbench_registry()
