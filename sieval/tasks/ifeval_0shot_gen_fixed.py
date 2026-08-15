"""IFEval 0-shot generative task, corrected — three repaired constraint checkers.

``ifeval_0shot_gen`` keeps grading through the vendored google-research
registry, defects included, because that is what reproduces a published number.
This variant substitutes three repaired checkers and is therefore *not* a
reproduction: it measures whether a response obeyed the constraint its own
prompt states.

**One method differs.** This class overrides
:meth:`~sieval.tasks.ifeval_0shot_gen.IFEvalZeroShotGenTask._instruction_dict`
and nothing else — the prompt, the strict/loose graders, the per-sample record
and the pooled report are all inherited, so the two tasks cannot diverge
anywhere except at the registry the checkers are looked up in. That is the
strongest available statement that the delta is the repair and not a second
change riding along with it.

The three defects, and why each fix is the narrowest one, are documented on the
mixins in :mod:`sieval.community.instruction_following_eval_fixed`; the
divergences and the measured deltas are enumerated in ``notes`` below. Three
things those numbers do not say on their own:

* **Coverage.** The three ids hold 70 of the pinned set's 834 constraint slots,
  over 68 of 541 prompts. Every ``nth_paragraph_first_word`` slot carries a
  single-token ``first_word``, so the multi-token half of that repair is inert
  here and is exercised only by the Multi-IF sibling.
* **What fires the paragraph defect.** ``re.split(r"\\n\\n", ...)`` yields an
  empty chunk only where two newlines meet, so the trigger is a response whose
  first chunk is *blank*, not merely one that opens with a newline. Whole runs
  do this uniformly and by different routes — 540 of 541 Intern-S2 responses
  open with a space then a blank line, all 100 Qwen3 responses open with
  ``\\n\\n`` outright, a second full-541 run does it in 0.2% — which is why the
  per-set deltas span an order of magnitude rather than reading as noise. On a
  run whose responses never open blank, this repair moves ``english_capital``
  and nothing else.
* **Seed ``langdetect`` before reproducing any of it.** ``langdetect.detect`` is
  randomized and SiEval does not set ``DetectorFactory.seed``, so an unseeded
  A/B flips *unrepaired* checkers too — ``change_case:english_lowercase``
  flipped between arms during this measurement and looked for a while like the
  repair leaking through a shared RNG. That same unseededness is why
  ``english_capital`` is worth repairing rather than merely worth noting.

**Status.** ``stable``. The divergence is carried by the name, so ``status`` is
not gated on reproducing a published number — by construction this variant
cannot, since it grades differently on purpose. What a ``_fixed`` variant owes
instead is a quantified delta, measured here on three stored response sets, with
each repair reducing to upstream's own expression on the inputs upstream already
handled (asserted in the tests). ``experimental`` is for a faithful port whose
published anchor is not reachable, which is not a claim this task makes.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import override

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.tasks.ifeval_0shot_gen import IFEvalZeroShotGenTask


@sieval_task(
    name="ifeval_0shot_gen_fixed",
    display_name="IFEval (0-shot, generative, corrected)",
    description="IFEval with three repaired checkers; grading otherwise upstream's.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="ifeval",
    model_type="chat",
    # The divergence is carried by the name, not by `status`: this task grades
    # differently on purpose, so it claims no published number to be gated on.
    # What a `_fixed` variant owes instead is the quantified delta in `notes`.
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="google-research/instruction_following_eval",
        url="https://github.com/google-research/google-research/blob/f97f6adab57bd3065b24169bcfc559dc34d0db84/instruction_following_eval/evaluation_lib.py",
        notes=(
            "Same vendored evaluation_lib + instructions_registry as "
            "ifeval_0shot_gen, with three checkers replaced by the repaired "
            "subclasses in sieval.community.instruction_following_eval_fixed. "
            "DIVERGENCES (all three, exhaustive): (1) "
            "length_constraints:nth_paragraph_first_word — upstream decrements a "
            "paragraph count for each blank '\\n\\n' chunk but indexes the "
            "unfiltered list, so a blank chunk at or before the target index "
            "checks the wrong paragraph (or a blank one); the fix filters once "
            "and indexes what it counted. It also "
            "compares as many tokens as the constraint's own value spans, which is "
            "inert on IFEval (all 12 slots are single-token) and matters for "
            "Multi-IF. (2) keywords:letter_frequency — upstream replaces a letter "
            "outside [a-z] with a fresh random.choice per call; the fix keeps the "
            "character the item names (keys 1122 '#', 1129 '!'), and consumes no "
            "draw, so the global RNG stream other checkers share is unperturbed. "
            "(3) change_case:english_capital — upstream runs langdetect on ALL-CAPS "
            "text, which every profile is off-distribution for; the fix detects on "
            "a case-folded copy and leaves isupper() to decide the capitals "
            "requirement. Nothing else differs: the task overrides one method "
            "(_instruction_dict) of ifeval_0shot_gen and inherits prompting, "
            "grading, records and report. "
            "SCORE IMPACT, measured by running this task's own feedback/report "
            "and the unqualified task's over the same stored responses, "
            "langdetect seeded identically per arm — strict prompt-level "
            "92.79→95.38 (+2.59, instruction-level 95.20→96.88) on a full-541 "
            "Intern-S2-Preview run, 94.82→95.19 (+0.37) on a second full-541 run, "
            "65.00→67.00 (+2.00) on a 100-prompt Qwen3-30B-A3B thinking-on subset; "
            "loose prompt-level +0.55/+0.55/+2.00. Flips are bidirectional: the "
            "second run's loose reading moves english_capital 2 FAIL→PASS and 1 "
            "PASS→FAIL. Coverage 70 of 834 slots over 68 of 541 prompts (12.6%). "
            "It also removes nondeterminism: over 5 langdetect seeds the repaired "
            "task is seed-invariant on both full-541 sets (spread 0.00, no slot "
            "changing verdict) where upstream spans 0.18 and 0.37. The 100-prompt "
            "subset keeps a 2.00 spread at two slots that are not this task's to "
            "fix — change_case:english_lowercase (detects on already-lowercase "
            "text, deliberately unrepaired) and one english_capital whose response "
            "is a 94k-character repetition loop. "
            "REPRODUCTION NOTE: langdetect.detect is randomized and sieval never "
            "sets DetectorFactory.seed, so an unseeded A/B also flips checkers "
            "this task does not touch; pin the seed in both arms."
        ),
    ),
)
class IFEvalZeroShotGenFixedTask(IFEvalZeroShotGenTask):
    @override
    def _instruction_dict(self) -> dict[str, type] | None:
        # Imported here, not at module scope, for the reason the base task
        # lazy-imports `evaluation_lib`: registration imports every task module,
        # and must not pay for the vendored checkers and langdetect.
        from sieval.community.instruction_following_eval_fixed import (
            fixed_ifeval_registry,
        )

        # A fresh dict per call, never the vendored global: samples grade
        # concurrently, and mutating the shared registry would change how the
        # unqualified task grades in the same session.
        return fixed_ifeval_registry()
