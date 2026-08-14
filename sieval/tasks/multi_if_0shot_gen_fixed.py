"""Multi-IF 0-shot generative task, corrected — three repaired constraint checkers.

The ``_fixed`` variant. ``multi_if_0shot_gen`` keeps grading through Meta's
vendored fork of the IFEval checkers, defects included; its module docstring
names two of them and says they are tracked rather than repaired, "per the
unqualified-name rule; fixing either needs a ``_fixed`` variant with a measured
delta". This is that variant.

**One method differs.** This class overrides
:meth:`~sieval.tasks.multi_if_0shot_gen.MultiIFZeroShotGenTask._instruction_dict`
and nothing else. Conversation walking, per-turn grading, the cumulative
constraint lists, the per-language and per-turn pooling — all inherited, so the
two tasks cannot diverge anywhere except at the registry the checkers are looked
up in.

The repairs are shared with ``ifeval_0shot_gen_fixed`` and documented on the
mixins in :mod:`sieval.community.instruction_following_eval_fixed`; Multi-IF's
vendored copies of these three checkers are logic-identical to google-research's
(the only differences between the two copies are one import path and one logging
call), which is why one mixin serves both registries.

**What is Multi-IF-specific.** Two of the three defects bite harder here, and
one of them does not exist in IFEval at all:

* ``length_constraints:nth_paragraph_first_word`` carries a ``first_word`` that
  is *not a single whitespace-delimited token* in 35 of its 749 slots — 32
  multi-token (Hindi 12, Spanish 9, French 3, Portuguese 3, Russian 3, Italian
  2, English 0) and 3 blank, all three in one conversation (``2215:14:zh``).
  Upstream compares against ``paragraph.split()[0]``, one token, so a
  multi-token slot **returns FAIL for every possible response** — a check that
  cannot pass measures nothing. The repair reads how many tokens the
  constraint's own value spans and compares that many, each normalised by
  upstream's own routine; it does not relax the comparison, and it reduces to
  upstream's expression token for token whenever the value is one token, which
  is every IFEval slot and 714 of these. The 3 blank slots stay ungradeable —
  a constraint with no value states nothing to check, and inventing a rule for
  it would be worse than upstream's — so this repair is not a backfill.
* ``keywords:letter_frequency`` with a non-``[a-z]`` letter appears in one
  conversation, ``1122:18:en``, at 3 turn-cells (the constraint is cumulative,
  so turn 1's slot recurs in turns 2 and 3). That is the row the unqualified
  task's notes call out as one it "cannot grade reproducibly itself".

**Coverage on the pinned 4,501.** The three ids account for 1,121 constraint
slots (``nth_paragraph_first_word`` 749, ``english_capital`` 210,
``letter_frequency`` 162) across 515 conversations — 11.4% of the set.

**Score impact, measured on stored responses.** 160 stored conversations
(Qwen3-30B-A3B, thinking on; 1,218 constraint slots) re-graded by running *this
task's own* ``feedback()`` and ``report()`` against them and the unqualified
task's over the same records — so these are the numbers a run reports, not a
grader-level approximation — with ``langdetect``'s factory seeded identically
per arm:

======  =========================  ==================================
turn    ``overall`` (upstream's)   strict instruction-level
======  =========================  ==================================
1       75.45 → 77.51  (+2.06)     78.14 → 80.97  (+2.83)
2       69.09 → 70.74  (+1.65)     79.46 → 81.44  (+1.98)
3       62.67 → 63.30  (+0.63)     78.75 → 79.64  (+0.89)
======  =========================  ==================================

``score`` (the mean of the three turns' all-language overalls) moves
**69.07 → 70.52, +1.45**. Flips are 17 ``nth_paragraph_first_word`` and 3
``english_capital`` under strict, 1 and 3 under loose, all FAIL→PASS.

The strict/loose asymmetry is mechanical rather than surprising: the loose
reading already re-tries each response with its first and last lines stripped,
which happens to undo some instances of the paragraph-index defect. That is also
why loose is the weaker evidence of the two — it was accidentally masking the
defect, not immune to it.

**Caveat on reproducing these deltas: seed langdetect.** Identical to the
IFEval sibling's, and it binds harder here — ``langdetect`` also selects the
word- and sentence-counting algorithm behind every length constraint, in a set
that is multilingual by design. Unlike the IFEval sibling, which is
*seed-invariant* after repair on both full sets it was measured on, this task
keeps a 0.18 ``score`` spread over five seeds — the same 0.18 upstream has,
because that residue is language routing on genuine multilingual text, which no
repair here touches and none should.

Pin ``langdetect.DetectorFactory.seed`` in both arms before attributing a flip
to a repair.

**Status.** ``experimental``, for both of its parent's reasons and one of its
own: no first-party published Multi-IF number is reproducible under any single
reduction (see ``multi_if_0shot_gen``'s notes — that is a property of the
anchor and this variant does not change it), and this name has not been run end
to end. Repairing three checkers does not make an unreproducible anchor
reproducible; if anything it moves further from one, deliberately.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from typing import override

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.tasks.multi_if_0shot_gen import MultiIFZeroShotGenTask


@sieval_task(
    name="multi_if_0shot_gen_fixed",
    display_name="Multi-IF (0-shot, generative, corrected)",
    description="Multi-IF with three repaired checkers; grading otherwise upstream's.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("multilingual", "multi-turn", "open-ended"),
    deps_group="multi-if",
    model_type="chat",
    # Inherits its parent's unreproducible-anchor problem and adds a deliberate
    # divergence on top, so `stable` is not reachable by reproduction here. The
    # divergence is quantified, which is what `_fixed` owes.
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="facebookresearch/Multi-IF",
        url="https://github.com/facebookresearch/Multi-IF/blob/1cdb53ed18499ad729e0766e5d3099dd5344406f/metrics.py",
        notes=(
            "Same vendored graders and multilingual checker fork as "
            "multi_if_0shot_gen, with three checkers replaced by the repaired "
            "subclasses in sieval.community.instruction_following_eval_fixed "
            "(shared with ifeval_0shot_gen_fixed; the two vendored copies of these "
            "three classes are logic-identical). DIVERGENCES (exhaustive): (1) "
            "length_constraints:nth_paragraph_first_word — paragraphs are counted "
            "and indexed on the same blank-filtered list, and the comparison spans "
            "as many tokens as the constraint's own value; 35 of 749 slots carry a "
            "first_word that is not one token (32 multi-token across 6 languages, "
            "3 blank in conversation 2215:14:zh) and upstream FAILs every response "
            "on them unconditionally. The 3 blank slots stay ungradeable by "
            "design. (2) keywords:letter_frequency — the item's own character is "
            "kept instead of a fresh random.choice per call; 1 conversation, "
            "1122:18:en, 3 turn-cells. (3) change_case:english_capital — langdetect "
            "runs on a case-folded copy, since ALL-CAPS text is off-distribution "
            "for every profile it ships. Nothing else differs: one method "
            "(_instruction_dict) is overridden and everything else inherited. "
            "SCORE IMPACT, measured by running this task's own feedback/report "
            "and the unqualified task's over 160 stored Qwen3-30B-A3B thinking-on "
            "conversations (1,218 slots), langdetect seeded identically per arm — "
            "score 69.07→70.52 (+1.45); per-turn overall 75.45→77.51 / "
            "69.09→70.74 / 62.67→63.30; strict instruction-level 78.14→80.97 / "
            "79.46→81.44 / 78.75→79.64. Flips 17 nth_paragraph_first_word + 3 "
            "english_capital (strict), 1 + 3 (loose), all FAIL→PASS. Loose moves "
            "less because its line-stripping retries already undo some instances "
            "of the paragraph defect. Coverage 1,121 slots over 515 of 4,501 "
            "conversations (11.4%). Unlike the IFEval sibling, which is "
            "seed-invariant after repair, score here still spans 0.18 over 5 "
            "langdetect seeds — the same 0.18 upstream spans, since that residue "
            "is language routing on genuinely multilingual text. "
            "REPRODUCTION NOTE: langdetect.detect is randomized and sieval never "
            "sets DetectorFactory.seed; pin it in both arms, since it also selects "
            "the counting algorithm behind every length constraint."
        ),
    ),
)
class MultiIFZeroShotGenFixedTask(MultiIFZeroShotGenTask):
    @override
    def _instruction_dict(self) -> dict[str, type] | None:
        # Imported here, not at module scope, for the reason the base task
        # lazy-imports `evaluation_lib`: the repair module reaches the vendored
        # 3.5k-line checker fork and langdetect, and registration -- which
        # imports every task module to build the registry -- must not pay for
        # them.
        from sieval.community.instruction_following_eval_fixed import (
            fixed_multi_if_registry,
        )

        # A fresh dict per call, never the vendored global: samples grade
        # concurrently, and mutating the shared registry would change how the
        # unqualified task grades in the same session.
        return fixed_multi_if_registry()
