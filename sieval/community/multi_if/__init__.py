"""Meta Multi-IF evaluation adaptation.

Source: https://github.com/facebookresearch/Multi-IF
Revision: 1cdb53ed18499ad729e0766e5d3099dd5344406f (Apache-2.0, archived)

Multi-IF ships its *own* multilingual fork of Google's IFEval checkers. It
carries the same 25 instruction ids as
``sieval.community.instruction_following_eval``, but routes word counting,
sentence counting and casing through ``langdetect``, so the two are not
interchangeable and both are vendored.

Local adaptations (each marked "Local adaptation:" at its site):

- ``ifeval.py``: ``pythainlp`` is imported on first use rather than at module
  scope. Thai is not one of the eight languages in the released Multi-IF CSV,
  so the Thai branches are unreachable for that data and the dependency stays
  optional.
- ``evaluation_lib.py``: only upstream ``metrics.py``'s two per-response
  graders are taken; conversation assembly and aggregation live in the task.

Deliberately *not* adapted: upstream leaves ``langdetect`` unseeded, so grading
is not reproducible run to run (~2-3% of ``detect()`` calls flip on short or
mixed-script text). Seeding it would change the grader, which belongs in a
``_fixed`` task variant with a measured score delta rather than under the
faithful name. The import site carries the full note.

Infra: scoring needs NLTK's **punkt_tab** data, and reaches it from two places --
``count_sentences`` (behind ``length_constraints:number_sentences``) via
``nltk.data.load("nltk:tokenizers/punkt/english.pickle")``, and
``nltk.word_tokenize`` (behind ``change_case:capital_word_frequency``). The first
path is named after the legacy ``punkt`` package but nltk >= 3.9 resolves it
through ``punkt_tab`` too, so staging ``punkt`` alone satisfies neither call and
staging ``punkt_tab`` alone satisfies both. Nothing here downloads it -- the task
in ``sieval/tasks/multi_if_0shot_gen.py`` ensures it once before grading starts,
which keeps these vendored files free of anything upstream does not have.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""
