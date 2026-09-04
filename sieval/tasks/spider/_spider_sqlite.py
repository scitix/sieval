"""Spider 1.0's measured bounds on a database read.

Three consumers, one pair of numbers: the prompt builder reads a schema and
sample rows, and the two graders each execute model-generated SQL. *How* a
database is opened and a statement bounded is the shared safety contract in
``sieval.tasks._sqlite_exec``; what belongs here is the part that is measured
against Spider 1.0's own corpus and would be a guess anywhere else.

Kept out of ``_spider_exec`` for the same reason the mechanism is kept out of
this subpackage: ``_spider_exec`` imports the vendored parser, which reaches
nltk at module scope, so a prompt path importing *it* for a bound would make
registering the task pay for a grader it may never run.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

#: Per-statement wall-clock budget. Measured against every dev gold on the
#: pinned data: the slowest completes in 0.486 s on the shipped databases and
#: 0.359 s across the distilled ones, so this is ~10x headroom and does not
#: bind. Short enough that a pathological prediction cannot occupy a pool worker.
DEFAULT_DEADLINE_S = 5.0
#: Row cap. A cap that truncates a real comparison is a scoring change wearing a
#: safety label, so it is measured rather than guessed, against every gold on
#: both paths: the largest result is 20,662 rows on the shipped databases and
#: 92,450 across the distilled suite, and only four dev golds exceed 10,000 rows
#: at all. 500,000 is ~5.4x the worst of those. Time is bounded separately by the
#: deadline above, so this is a memory bound and not a second time one.
#:
#: Raised from 100,000, where a real gold sat 7.5% under the cap — a bound that
#: close is not evidence that no bound binds, and on the gold side binding means
#: a *raise*, failing a sample rather than mis-scoring it. The raise cannot move
#: a verdict either metric previously reported: both compare row-for-row, so a
#: prediction returning between the old cap and the new one differs in length
#: from every gold here and was already wrong.
#:
#: **What the raise costs, measured** (2026-09-03). The cap is also the ceiling:
#: ``run_bounded`` fetches ``max_rows + 1`` in one call, so a capped result is
#: fully materialised before the length check rejects it. A three-way cross join
#: on ``car_1`` moves peak RSS by ~450 MB for one statement (~90 MB at the old
#: cap), and with ``result_eq`` copying both sides across up to 8 pool workers
#: the worst case is a multi-GB transient. Kept anyway: the exposure is not
#: reachable by the data — no dev gold comes within 5x, and no prediction in
#: either full dev pass hit it — while the old cap's failure mode was a *failed
#: sample*. Should it ever need lowering, the fix is a chunked fetch that rejects
#: at the threshold rather than a smaller number, which decouples the ceiling
#: from the cap.
DEFAULT_MAX_ROWS = 500_000
