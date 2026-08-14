"""IFEval 0-shot generative task, corrected — three repaired constraint checkers.

The ``_fixed`` variant. ``ifeval_0shot_gen`` keeps grading through the vendored
google-research registry, defects included, because that is what reproduces a
published number. This one substitutes three repaired checkers and is therefore
*not* a reproduction: it measures whether a response obeyed the constraint its
own prompt states.

**One method differs.** This class overrides
:meth:`~sieval.tasks.ifeval_0shot_gen.IFEvalZeroShotGenTask._instruction_dict`
and nothing else — the prompt, the strict/loose graders, the per-sample record
and the pooled report are all inherited, so the two tasks cannot diverge
anywhere except at the registry the checkers are looked up in. That is the
strongest available statement that the delta below is the repair and not a
second change riding along with it.

The three repairs, their defects, and why each is the narrowest fix are
documented on the mixins in
:mod:`sieval.community.instruction_following_eval_fixed`. In one line each:

* ``length_constraints:nth_paragraph_first_word`` — upstream counts paragraphs
  on a blank-filtered list and then indexes the *unfiltered* one, so an empty
  ``\\n\\n`` chunk at or before the nth paragraph checks the wrong paragraph
  against a total computed the other way.
* ``keywords:letter_frequency`` — a letter outside ``[a-z]`` is silently
  replaced by ``random.choice(string.ascii_letters)``, freshly drawn per call,
  so the item is graded against a letter nobody asked for and not even the same
  one twice.
* ``change_case:english_capital`` — ``langdetect`` is called on ALL-CAPS text,
  which is off-distribution for every profile it ships; the repair detects on a
  case-folded copy, leaving ``isupper()`` (upstream's, and still first) to
  decide the capitals requirement.

**Coverage on the pinned 541.** The three ids account for 70 of 834 constraint
slots (``english_capital`` 25, ``letter_frequency`` 33,
``nth_paragraph_first_word`` 12), spread over 68 prompts — 12.6% of the set. Two
``letter_frequency`` slots carry a non-``[a-z]`` letter (keys 1122 ``#`` and 1129
``!``). Every ``nth_paragraph_first_word`` slot carries a single-token
``first_word``, so the multi-token half of that repair is inert here and is
exercised only by the Multi-IF sibling.

**Score impact, measured on stored responses.** Three stored response sets were
re-graded by running *this task's own* ``feedback()`` and ``report()`` against
them and the unqualified task's over the same records — so these are the numbers
a run reports, not a grader-level approximation — with ``langdetect``'s factory
seeded identically per arm (see the caveat below):

==============================================  ==============  ==============
set                                             strict prompt   loose prompt
==============================================  ==============  ==============
Intern-S2-Preview, full 541                     92.79 → 95.38   95.19 → 95.75
a second stored full-541 run                    94.82 → 95.19   95.93 → 96.49
Qwen3-30B-A3B thinking-on, 100-prompt subset    65.00 → 67.00   71.00 → 73.00
==============================================  ==============  ==============

So **+0.37 to +2.59 strict prompt-level**; strict instruction-level moves
95.20 → 96.88, 96.52 → 96.76 and 79.70 → 80.69. Per-id flips, strict: set 1
``nth_paragraph_first_word`` 12 (every slot in the set) and ``english_capital``
2; set 2 ``english_capital`` 2; set 3 one each of ``english_capital`` and
``nth_paragraph_first_word``. The direction is not uniformly upward, which is
the point — set 2's loose reading moves ``english_capital`` 2 slots FAIL→PASS
*and 1 PASS→FAIL*: a well-posed detector removes false passes as well as false
failures.

**It also removes a source of run-to-run nondeterminism, and that is
measurable.** Re-grading each set under five different ``langdetect`` seeds,
strict prompt-level spans:

======================  =======================  =====================
set                     upstream                 fixed
======================  =======================  =====================
Intern-S2-Preview 541   92.61–92.79 (0.18)       95.38–95.38 (**0.00**)
second full-541 run     94.45–94.82 (0.37)       95.19–95.19 (**0.00**)
Qwen3 100-prompt        63.00–65.00 (2.00)       65.00–67.00 (2.00)
======================  =======================  =====================

On both full sets the repaired task is *seed-invariant*: not one of the 834
slots changes verdict across all five seeds, against upstream's 1–2 unstable
slots (``letter_frequency``, whose random draw the repair removes outright, and
``english_capital``). The 100-prompt subset keeps a 2.00 spread, and the
attribution is exact — its two unstable slots are one
``change_case:english_lowercase`` (deliberately not repaired: it detects on
already-lowercase text) and one ``english_capital`` whose response is a
94,152-character, 18,544-word repetition loop that reads as Welsh on one seed in
five. **The repair makes detection well-posed; it does not make ``langdetect``
deterministic**, and on a degenerate response nothing would.

**What fires the paragraph defect, and how often.** ``re.split(r"\\n\\n", ...)``
yields an empty chunk only where two newlines meet, so the trigger is a response
whose first ``\\n\\n``-separated chunk is *blank* — not merely one that opens
with a newline. Whole runs do this uniformly, and by different routes: 540 of
541 Intern-S2 responses open with a space and then a blank line, all 100 Qwen3
responses open with ``\\n\\n`` outright, and the second full-541 run does it in
0.2%. The per-set flips track that rate exactly — 12 of 12
``nth_paragraph_first_word`` slots have their target index moved in set 1 and
0 of 12 in set 2, and set 2's flips at this id are correspondingly zero, which
is why its delta is the smallest of the three. So the defect is not specific to
reasoning models, but neither is its blast radius uniform: on a run whose
responses never open with a blank line, this repair moves ``english_capital``
and nothing else.

**Caveat on reproducing these deltas: seed langdetect.** ``langdetect.detect``
is randomized and SiEval does not set ``DetectorFactory.seed``, so an unseeded
A/B shows flips at *unrepaired* checkers too — ``change_case:english_lowercase``
flipped between arms during this measurement before its factory was pinned, and
looked for a while like the repair leaking through a shared RNG. Pin
``langdetect.DetectorFactory.seed`` identically in both arms before attributing
any flip to a fix. The same unseededness is why ``english_capital`` is worth
repairing at all rather than merely worth noting.

**Status.** ``experimental``: the divergence from upstream is measured three
ways above and the repairs each reduce to upstream's own expression on the
inputs upstream already handled (asserted in the tests), but the variant has not
yet been run end to end against a published anchor — and by construction it
cannot reproduce one, since it grades differently on purpose. Promotion wants a
full live run of this name, not a replay of the unqualified one's responses.

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
    # Not a reproduction of a published number by construction -- it grades
    # differently on purpose -- and not yet run end to end under this name. The
    # divergence is quantified (module docstring, and `notes` below), which is
    # what `_fixed` owes; a live run is what promotion owes.
    status="experimental",
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
        # lazy-imports `evaluation_lib`: the repair module reaches the vendored
        # checkers and langdetect, and registration -- which imports every task
        # module to build the registry -- must not pay for them.
        from sieval.community.instruction_following_eval_fixed import (
            fixed_ifeval_registry,
        )

        # A fresh dict per call, never the vendored global: samples grade
        # concurrently, and mutating the shared registry would change how the
        # unqualified task grades in the same session.
        return fixed_ifeval_registry()
