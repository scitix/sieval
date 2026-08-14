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
rollouts moves no metric at all. So this variant is not a materially different
benchmark — it is the same benchmark with a bounded, known correction, and the
bound is the point: without it the residual is an unmeasured guess.

Only two of the four repairs flip anything on real responses, and saying which is
part of the measurement:

- ``format:line_indent`` — 11 flips (strict). The whole delta, essentially.
- ``words:words_position`` — 1 flip (loose).
- ``ratio:sentence_type`` — **0 flips** across 48 gradings. Quoted-declarative
  recovery fired twice without changing a verdict, and the vacuous ``0 == 0``
  false pass was *reachable but never fired*: 5 gradings had no interrogative,
  and all 5 also had a nonzero declarative count, so upstream's test was False
  for the right reason by luck rather than by construction.
- ``words:vowel`` — **0 flips** across 80 gradings. The line-count gate and the
  paragraph-count gate agreed on all 70 non-blank responses (38 single, 32
  multi): this model never produced the soft-wrapped single paragraph that
  separates them. The defect is real and unexercised, not absent.

Both are kept. A checker that is wrong only on inputs a particular model happens
not to produce is still wrong, and the next model is not bound by this one's
formatting habits — but a reader deserves to know the number came from two
repairs, not four.

Use the unqualified ``ifbench_0shot_gen`` to compare against published IFBench
numbers; those were produced by upstream's code, defects included. Use this task
when the grading needs to match what the items actually ask.

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
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="allenai/IFBench",
        url="https://github.com/allenai/IFBench/blob/1091c4c3de6c1f6ed12c012ed68f11ea450b0117/evaluation_lib.py",
        notes=(
            "Deliberate fork of ifbench_0shot_gen, which stays byte-faithful to "
            "upstream. Four vendored checkers grade something other than what "
            "their instruction says and are repaired here; nothing else differs, "
            "and no item text is changed. (1) format:line_indent drops blank "
            "lines with `for line in lines: if not line.strip(): "
            "lines.remove(line)`, which mutates the list under its own iterator "
            "and so skips one element per removal -- a surviving blank line has "
            "indent 0 and breaks the increasing-indent chain, making the verdict "
            "depend on how many blank lines the response happened to contain. "
            "(2) ratio:sentence_type counts declaratives with endswith('.'), "
            "missing quoted declaratives, and returns declarative == 2 * "
            "interrogative, so a response with neither passes vacuously at 0==0. "
            "(3) words:words_position indexes tokens, handling exactly one "
            "trailing punctuation token, so a response ending in two shifts "
            "every position by one. (4) words:vowel counts paragraphs with "
            "split('\\n'), rejecting any soft-wrapped single paragraph on line "
            "count. SCORE IMPACT, measured against the unqualified task over the "
            "full official 300-prompt test set at 8 rollouts (Qwen3-30B-A3B, "
            "thinking on), replaying stored responses: loose prompt-level (the "
            "headline) 39.2917→39.3333 (+0.0417), strict prompt-level "
            "30.7917→31.2500 (+0.4583), loose instruction-level "
            "42.2602→42.2965, strict instruction-level 34.0480→34.4477. "
            "The four ids cover 28 "
            "of 300 prompts; 12 of 5,504 gradings flip, all FAIL->PASS, none "
            "PASS->FAIL. Only format:line_indent (11 flips) and "
            "words:words_position (1) flip anything on these responses; "
            "ratio:sentence_type and words:vowel flip nothing here and are kept "
            "because the defect is unexercised by this model, not absent. So "
            "this variant is not a different benchmark, it is the same one with "
            "a bounded correction -- do not compare its numbers to published "
            "IFBench results, which were produced by upstream's code."
        ),
    ),
)
class IFBenchZeroShotGenFixedTask(IFBenchZeroShotGenTask):
    """IFBench graded through the repaired checkers.

    Overrides exactly one method. Everything that decides what a sample is asked,
    how it is decoded, and how verdicts are pooled into the headline is inherited,
    so the two tasks cannot drift apart anywhere except the registry lookup.
    """

    @override
    def _instruction_dict(self) -> dict[str, type] | None:
        # Imported here, not at module scope, for the reason the base task
        # lazy-imports `evaluation_lib`: the repair module subclasses the
        # vendored 2.3k-line checker fork and reaches NLTK through it, and
        # importing this task -- which `sieval task show` and the registry
        # scan do -- must not pay for either.
        from sieval.tasks._ifbench_fixed_checkers import fixed_ifbench_registry

        # A fresh dict per call, never the vendored global: samples grade
        # concurrently, and mutating the shared registry would change how the
        # unqualified task grades in the same session.
        return fixed_ifbench_registry()
