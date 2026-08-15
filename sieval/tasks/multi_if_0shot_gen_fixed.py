"""Multi-IF 0-shot generative task, corrected — three repaired constraint checkers.

``multi_if_0shot_gen`` keeps grading through Meta's vendored fork of the IFEval
checkers, defects included; its module docstring names two of them and says they
are tracked rather than repaired, "per the unqualified-name rule; fixing either
needs a ``_fixed`` variant with a measured delta". This is that variant.

**One method differs.** This class overrides
:meth:`~sieval.tasks.multi_if_0shot_gen.MultiIFZeroShotGenTask._instruction_dict`
and nothing else. Conversation walking, per-turn grading, the cumulative
constraint lists, the per-language and per-turn pooling — all inherited, so the
two tasks cannot diverge anywhere except at the registry the checkers are looked
up in.

The repairs are shared with ``ifeval_0shot_gen_fixed`` and documented on the
mixins in :mod:`sieval.community.instruction_following_eval_fixed`; Multi-IF's
vendored copies of these three checkers are logic-identical to google-research's
(the two copies differ only in one import path and one logging call), which is
why one mixin serves both registries. The divergences and the measured deltas
are enumerated in ``notes`` below. Three things those numbers do not say on
their own, two of them specific to this port:

* ``length_constraints:nth_paragraph_first_word`` carries a ``first_word`` that
  is *not a single whitespace-delimited token* in 35 of its 749 slots — 32
  multi-token (Hindi 12, Spanish 9, French 3, Portuguese 3, Russian 3, Italian
  2, English 0) and 3 blank, all three in one conversation (``2215:14:zh``).
  Upstream compares against ``paragraph.split()[0]``, one token, so a
  multi-token slot **returns FAIL for every possible response** — a check that
  cannot pass measures nothing. The repair spans as many tokens as the
  constraint's own value, each normalised by upstream's own routine, so it does
  not relax the comparison and reduces to upstream's expression token for token
  whenever the value is one token — every IFEval slot, and 714 of these. The 3
  blank slots stay ungradeable, since a constraint with no value states nothing
  to check; this repair is not a backfill.
* ``keywords:letter_frequency`` with a non-``[a-z]`` letter appears in one
  conversation, ``1122:18:en``, at 3 turn-cells (the constraint is cumulative,
  so turn 1's slot recurs in turns 2 and 3). That is the row the unqualified
  task's notes call out as one it "cannot grade reproducibly itself".
* The strict/loose asymmetry in the deltas is mechanical: the loose reading
  already re-tries each response with its first and last lines stripped, which
  happens to undo some instances of the paragraph-index defect. That also makes
  loose the weaker evidence of the two — it was accidentally masking the defect,
  not immune to it.

**Caveat on reproducing the deltas: seed langdetect.** Identical to the IFEval
sibling's, and it binds harder here — ``langdetect`` also selects the word- and
sentence-counting algorithm behind every length constraint, in a set that is
multilingual by design. Unlike that sibling, which is *seed-invariant* after
repair on both full sets it was measured on, this task keeps a 0.18 ``score``
spread over five seeds — the same 0.18 upstream has, because that residue is
language routing on genuine multilingual text, which no repair here touches and
none should. Pin ``langdetect.DetectorFactory.seed`` in both arms before
attributing a flip to a repair.

**Status.** ``stable``. Its parent stays ``experimental`` because it is a
faithful port whose first-party published number is not reproducible under any
single reduction (see ``multi_if_0shot_gen``'s notes). That is a property of the
anchor, which this variant neither changes nor inherits: it claims no anchor at
all. Repairing three checkers moves further from that number, deliberately; what
it owes instead is a quantified delta, which ``notes`` carries.

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
    # The parent stays `experimental` because it is a faithful port whose
    # published anchor is not reachable. That reason does not carry over: this
    # variant grades differently on purpose and claims no anchor, so what it
    # owes instead is the quantified delta in `notes`.
    status="stable",
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
        # lazy-imports `evaluation_lib`: registration imports every task module,
        # and must not pay for the 3.5k-line checker fork and langdetect.
        from sieval.community.instruction_following_eval_fixed import (
            fixed_multi_if_registry,
        )

        # A fresh dict per call, never the vendored global: samples grade
        # concurrently, and mutating the shared registry would change how the
        # unqualified task grades in the same session.
        return fixed_multi_if_registry()
