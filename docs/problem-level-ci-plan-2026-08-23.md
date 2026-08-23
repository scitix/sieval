# Problem-level CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report a deterministic, closed-form 95% confidence interval on every task's
headline score, clustered at the problem rather than the sample.

**Architecture:** A pure estimator in `sieval/core/tasks/metrics.py` (Wilson on an
effective sample size — no RNG, no seed). Clustering of `Dataset.repeat`
pseudo-duplicates comes from a new `repeat_group` column stamped at `repeat()` time
and read back off the *live dataset* by an overridable `Task.problem_groups`, never
off a `TaskContext` — `repeat_index` does not survive a resume, so a context-keyed
grouping would silently disable itself. Tasks opt in explicitly by naming which
metric the interval belongs to, which is what keeps the clustering axis honest.

**Tech Stack:** Python 3.12, `pdm`, `pytest`, `ruff`, `ty`. Standard library only —
the estimator needs no `scipy`.

**Spec:** `docs/problem-level-ci-design-2026-08-23.md`

## Global Constraints

- Python ≥ 3.12. No `from __future__ import annotations`.
- AI-generated source files carry `AI-Generated Code - Claude Opus 5 (Anthropic)`
  as the **last line** of the module docstring. Only for files *created* here;
  do not add it to files that already exist.
- `sieval/core/` must not import from `sieval.infer`, `sieval.tasks`,
  `sieval.datasets`, or `sieval.cli`.
- `sieval/core` coverage ≥ 95%, gated in CI. Locally use
  `python -m coverage run --source=sieval -m pytest <tests>` then `coverage report -m`
  (plain `pytest --cov` dies on a pyarrow double-registration in some environments).
- A report key is **omitted, never zeroed**, when it cannot be computed.
- Report keys never spell a value of `k`. Counts take the `n_` prefix.
- No private references in shipped comments: no `docs/…` paths, no `/volume/…`,
  no "spec §X". Refer to RFC #94 by number only, as `metrics.py` already does for #74.
- Run `pdm run ruff format . && pdm run ruff check . && pdm run ty check` before each
  commit. `ty` must be run bare (a relative `python = "./.venv"` aborts in a worktree).
- Do not touch `--resume` behaviour: no changes to `loader.py`, and no change to the
  `if not ctx.is_terminal()` condition in `runner.py`.

---

### Task 1: The Wilson estimator

**Files:**

- Modify: `sieval/core/tasks/metrics.py` (append after `aggregate`, ~line 344)
- Test: `tests/unit/core/tasks/test_metrics.py`

**Interfaces:**

- Consumes: nothing.
- Produces: `wilson_interval(values: Sequence[float], denominator: int, *, z: float = 1.96, scale: float = 100.0) -> tuple[float, float] | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/tasks/test_metrics.py`:

```python
def _plain_wilson(k: int, m: int, z: float = 1.96) -> tuple[float, float]:
    """Textbook Wilson score interval, as the reduction target."""
    p = k / m
    centre = (p + z * z / (2 * m)) / (1 + z * z / m)
    half = (
        z
        / (1 + z * z / m)
        * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    )
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


@pytest.mark.parametrize(
    "correct,m", [(1, 30), (5, 30), (15, 30), (29, 30), (391, 500), (101, 198), (3, 7)]
)
def test_wilson_interval_reduces_to_plain_wilson_on_booleans(correct, m):
    # The population SD divisor is what makes this EXACT rather than off by the
    # m/(m-1) factor -- a sample divisor puts m_eff at m-1 and misses by 0.34pp
    # at 1/30.
    values = [1.0] * correct + [0.0] * (m - correct)
    got = wilson_interval(values, m)
    assert got is not None
    want = _plain_wilson(correct, m)
    assert got[0] == pytest.approx(want[0], abs=1e-9)
    assert got[1] == pytest.approx(want[1], abs=1e-9)


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes():
    # A Wald half-width would put the lower bound at -3.09 for 1/30. The bound
    # is exactly where saturated and very hard sets live, so it must hold.
    lo, hi = wilson_interval([1.0] + [0.0] * 29, 30)
    assert lo > 0.0
    assert hi < 100.0


def test_wilson_interval_uses_clopper_pearson_when_nothing_was_correct():
    # p=0 leaves no dispersion to estimate, and m_eff is undefined -- but a 0.0
    # headline is exactly when a reader needs the upper bound.
    lo, hi = wilson_interval([0.0] * 30, 30)
    assert lo == 0.0
    assert hi == pytest.approx(100 * (1 - 0.025 ** (1 / 30)), abs=1e-9)


def test_wilson_interval_uses_clopper_pearson_when_everything_was_correct():
    lo, hi = wilson_interval([1.0] * 30, 30)
    assert lo == pytest.approx(100 * 0.025 ** (1 / 30), abs=1e-9)
    assert hi == 100.0


def test_wilson_interval_narrows_when_failures_pad_the_denominator():
    # Failed samples are FIXED ZEROS carrying no variance, so the estimator's
    # variance is m*s^2/D^2, not s^2/m. Using s^2/m would overstate the width
    # by 67% at D=50, m=30.
    values = [1.0] * 15 + [0.0] * 15
    tight = wilson_interval(values, 50)
    loose = wilson_interval(values, 30)
    assert tight[1] - tight[0] < loose[1] - loose[0]


def test_wilson_interval_omitted_below_two_problems():
    assert wilson_interval([1.0], 1) is None
    assert wilson_interval([], 0) is None


def test_wilson_interval_omitted_when_every_problem_scored_alike():
    # Zero observed dispersion at m >= 2 is a real signal, but not a variance
    # estimate; a zero-width interval would claim certainty the run lacks.
    assert wilson_interval([0.5] * 8, 8) is None


def test_wilson_interval_is_order_independent():
    a = wilson_interval([1.0, 0.0, 0.5, 0.25], 4)
    b = wilson_interval([0.25, 0.5, 0.0, 1.0], 4)
    assert a == b
```

Add `import math` to the test module's imports and `wilson_interval` to the
`from sieval.core.tasks.metrics import (...)` block.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pdm run pytest tests/unit/core/tasks/test_metrics.py -k wilson -v`
Expected: FAIL — `ImportError: cannot import name 'wilson_interval'`

- [ ] **Step 3: Write the implementation**

Add `import math` to `metrics.py`'s imports (it currently imports only
`collections` and `collections.abc`). Append after `aggregate`:

```python
def wilson_interval(
    values: Sequence[float],
    denominator: int,
    *,
    z: float = 1.96,
    scale: float = 100.0,
) -> tuple[float, float] | None:
    """A 95% interval on ``sum(values) / denominator``, clustered on *values*.

    The resampling unit is one element of *values* -- one PROBLEM. Pooling the
    rollouts of one problem as independent trials understates the width, and
    understates it more the more the model varies per problem.

    Wilson on an effective sample size, ``m_eff = p(1-p)/Var``, rather than a Wald
    half-width: the half-width puts the lower bound below zero exactly where
    saturated and very hard sets live (a real 1/30 run reads ``3.33 +/- 6.42``),
    and the asymmetry near a bound is the part worth reporting. With boolean
    *values* and ``denominator == len(values)`` this reduces EXACTLY to the
    textbook Wilson interval -- which is why the variance below uses the
    population divisor, not ``m - 1``.

    *denominator* is the population the headline is averaged over, which is not
    ``len(values)`` whenever failed samples count as wrong. Those are DETERMINISTIC
    zeros: they enter the mean but contribute no variance, so the variance of
    ``sum/D`` over ``m`` random terms is ``m*s**2/D**2`` -- smaller than ``s**2/m``,
    while the mean is pulled down by the same zeros. Spelling it ``s**2/m`` would
    overstate the width on any run with failures (67% at ``D=50, m=30``).

    Returns ``None`` -- omitted, never zeroed -- when there is nothing to estimate:
    fewer than two problems, or no dispersion between them. At ``p`` exactly 0 or 1
    there is no dispersion either, but that is when a reader most needs the bound,
    so those fall back to the exact one-sided Clopper-Pearson limit over problems.

    No randomness, so two readers of the same values compute the same interval
    (RFC #74 D refused a seed in this layer). Order-independent, which matters
    because a resumed run rebuilds its finals in manifest order.
    """
    m = len(values)
    if m < 2 or denominator <= 0:
        return None
    total = sum(values)
    p = total / denominator
    if p <= 0.0:
        return 0.0, scale * (1.0 - 0.025 ** (1 / m))
    if p >= 1.0:
        return scale * 0.025 ** (1 / m), scale
    mean = total / m
    # Population divisor: with the sample divisor `m_eff` lands on `m - 1` and the
    # reduction to plain Wilson is off by 0.34pp at 1/30.
    spread = sum((v - mean) ** 2 for v in values) / m
    variance = m * spread / (denominator * denominator)
    if variance <= 0.0:
        return None
    m_eff = p * (1.0 - p) / variance
    centre = (p + z * z / (2 * m_eff)) / (1 + z * z / m_eff)
    half = (
        z
        / (1 + z * z / m_eff)
        * math.sqrt(p * (1 - p) / m_eff + z * z / (4 * m_eff * m_eff))
    )
    return scale * max(0.0, centre - half), scale * min(1.0, centre + half)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pdm run pytest tests/unit/core/tasks/test_metrics.py -k wilson -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Lint, type-check, commit**

```bash
pdm run ruff format sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
pdm run ruff check sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
pdm run ty check
git add sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
git commit -m "feat(core): add a clustered Wilson interval estimator"
```

---

### Task 2: The `repeat_group` column

**Files:**

- Modify: `sieval/core/datasets/dataset.py:23-48` (constants + reader),
  `sieval/core/datasets/dataset.py:100-153` (`repeat`)
- Modify: `sieval/core/datasets/__init__.py:1,16` (exports)
- Test: `tests/unit/core/test_datasets.py` (class `TestRepeat`, ~line 143)

**Interfaces:**

- Consumes: nothing.
- Produces: `REPEAT_GROUP_COLUMN = "repeat_group"`,
  `repeat_group_of(raw: object) -> int | None`

- [ ] **Step 1: Write the failing tests**

Add to `TestRepeat` in `tests/unit/core/test_datasets.py`:

```python
    def test_repeat_stamps_the_original_row_index(self):
        # Row-major within a copy, so copy c's row j sits at c*n_rows + j and
        # carries group j. This is the ONLY key that groups copies of one
        # problem: content cannot (gpqa permutes choices per copy) and position
        # cannot (shuffle breaks the arithmetic).
        ds = _make(3)
        rows = list(ds.repeat(2).test_set)
        assert [r[REPEAT_GROUP_COLUMN] for r in rows] == [0, 1, 2, 0, 1, 2]

    def test_repeat_group_pairs_with_the_copy_index(self):
        ds = _make(3)
        rows = list(ds.repeat(2).test_set)
        assert [(r[REPEAT_INDEX_COLUMN], r[REPEAT_GROUP_COLUMN]) for r in rows] == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ]

    def test_repeat_once_still_stamps_the_group(self):
        ds = _make(2)
        rows = list(ds.repeat(1).test_set)
        assert [r[REPEAT_GROUP_COLUMN] for r in rows] == [0, 1]

    def test_repeat_group_survives_shuffle(self):
        # The whole point of a column over position arithmetic.
        ds = _make(4)
        shuffled = ds.repeat(2).shuffle(seed=7)
        pairs = [
            (r[REPEAT_INDEX_COLUMN], r[REPEAT_GROUP_COLUMN])
            for r in shuffled.test_set
        ]
        assert sorted(pairs) == [(c, j) for c in range(2) for j in range(4)]

    def test_repeat_refuses_a_split_already_carrying_the_group_column(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict(
                {"test": HFDataset.from_list([{REPEAT_GROUP_COLUMN: 0}])}
            )
        )
        with pytest.raises(ValueError, match=REPEAT_GROUP_COLUMN):
            ds.repeat(2)
```

Add a new class after `TestRepeatIndexOf`:

```python
class TestRepeatGroupOf:
    def test_reads_the_column_repeat_stamped(self):
        ds = _make(2)
        assert repeat_group_of(ds.repeat(2).test_set[3]) == 1

    def test_absent_column_is_not_grouped(self):
        assert repeat_group_of({"id": 1}) is None

    @pytest.mark.parametrize("bad", [None, "0", 1.5, True, [0]])
    def test_non_integer_reads_as_not_grouped(self, bad):
        assert repeat_group_of({REPEAT_GROUP_COLUMN: bad}) is None
```

Add `REPEAT_GROUP_COLUMN` and `repeat_group_of` to the module's imports from
`sieval.core.datasets`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pdm run pytest tests/unit/core/test_datasets.py -k "group" -v`
Expected: FAIL — `ImportError: cannot import name 'REPEAT_GROUP_COLUMN'`

- [ ] **Step 3: Write the implementation**

In `dataset.py`, after `REPEAT_INDEX_COLUMN` (line 27):

```python
#: Column :meth:`Dataset.repeat` stamps on every row, naming which ORIGINAL row the
#: copy came from. Separate from :data:`REPEAT_INDEX_COLUMN`, which says *which copy*
#: a row is: grouping the copies of one problem needs *which problem*, and nothing
#: downstream can recover it. Content cannot -- a task may legitimately vary the
#: prompt per copy (gpqa_diamond permutes the answer choices) -- and position cannot,
#: for the same reason the copy number is stamped rather than derived.
REPEAT_GROUP_COLUMN = "repeat_group"
```

After `repeat_index_of` (line 48):

```python
def repeat_group_of(raw: object) -> int | None:
    """Read a repeated row's ORIGINAL row index, or ``None`` if it carries none.

    Tolerant on the same terms as :func:`repeat_index_of`, and for the same reason:
    a raw sample is whatever the dataset yields. A bool is rejected explicitly.
    """
    if not isinstance(raw, Mapping):
        return None
    value = cast("Mapping[str, object]", raw).get(REPEAT_GROUP_COLUMN)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
```

Replace the guard and the stamp in `repeat` (lines 137-152):

```python
        for column in (REPEAT_INDEX_COLUMN, REPEAT_GROUP_COLUMN):
            if column in original.column_names:
                raise ValueError(
                    f"split {split!r} already has a {column!r} column; "
                    f"repeating twice needs a composite index one column cannot "
                    f"carry. Repeat once: if the task repeats this split itself "
                    f"(those taking an 'n_repeats' argument), drop the repeat "
                    f"around it or set that to 1; otherwise rename the column, or "
                    f"drop an earlier repeat()'s."
                )
        n_rows = len(original)
        # Copy-major, matching what HuggingFace's own `repeat` concatenates: copy 0
        # in full, then copy 1. Built from `times`/`n_rows` rather than read back off
        # the result so the stamp cannot agree with a reordering that already
        # happened.
        new_dict[split] = (
            original.repeat(times)
            .add_column(
                REPEAT_INDEX_COLUMN, [i for i in range(times) for _ in range(n_rows)]
            )
            .add_column(
                REPEAT_GROUP_COLUMN, [j for _ in range(times) for j in range(n_rows)]
            )
        )
```

Then update the `repeat` docstring's `Raises:` clause to name both columns, and
export the two new names in `sieval/core/datasets/__init__.py` — add to the
`from .dataset import ...` line and to `__all__`, keeping both alphabetical.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pdm run pytest tests/unit/core/test_datasets.py -v`
Expected: PASS — the new tests plus every pre-existing `TestRepeat` test, since
the guard message still contains `repeat_index` for the old assertion.

- [ ] **Step 5: Lint, type-check, commit**

```bash
pdm run ruff format sieval/core/datasets/ tests/unit/core/test_datasets.py
pdm run ruff check sieval/core/datasets/ tests/unit/core/test_datasets.py
pdm run ty check
git add sieval/core/datasets/ tests/unit/core/test_datasets.py
git commit -m "feat(datasets): stamp the original row index alongside the copy number"
```

---

### Task 3: `Task.problem_groups`

**Files:**

- Modify: `sieval/core/tasks/task.py` (imports; new method after `make_context`, ~line 371)
- Test: `tests/unit/core/tasks/test_task.py`

**Interfaces:**

- Consumes: `REPEAT_GROUP_COLUMN`, `repeat_group_of` (Task 2).
- Produces: `ProblemGrouping` (frozen dataclass, fields `keys: list[Hashable]`,
  `n_problems: int`) and `Task.problem_groups(finals) -> ProblemGrouping | None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/core/tasks/test_task.py`:

```python
def test_problem_groups_returns_none_on_an_unrepeated_split(_dummy_task_factory):
    task = _dummy_task_factory(rows=[{"id": i} for i in range(4)])
    finals = [task.make_context(i) for i in range(4)]
    assert task.problem_groups(finals) is None


def test_problem_groups_collapses_repeat_copies(_dummy_task_factory):
    task = _dummy_task_factory(rows=[{"id": i} for i in range(3)], repeat=2)
    finals = [task.make_context(i) for i in range(6)]
    grouping = task.problem_groups(finals)
    assert grouping.keys == [0, 1, 2, 0, 1, 2]
    assert grouping.n_problems == 3


def test_problem_groups_counts_problems_over_the_whole_split(_dummy_task_factory):
    # n_problems spans the REQUESTED population, so a problem whose every copy
    # failed still occupies a slot -- the denominator must not shrink with run
    # health.
    task = _dummy_task_factory(rows=[{"id": i} for i in range(3)], repeat=2)
    finals = [task.make_context(i) for i in (0, 1, 3)]
    grouping = task.problem_groups(finals)
    assert grouping.keys == [0, 1, 0]
    assert grouping.n_problems == 3


def test_problem_groups_survives_a_shuffle(_dummy_task_factory):
    task = _dummy_task_factory(rows=[{"id": i} for i in range(4)], repeat=2, shuffle=11)
    finals = [task.make_context(i) for i in range(8)]
    grouping = task.problem_groups(finals)
    assert sorted(grouping.keys) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert grouping.n_problems == 4


def test_problem_groups_raises_when_a_final_cannot_be_placed(_dummy_task_factory):
    # Loud, not silent: an unplaceable sample on a repeated split would narrow
    # every interval by sqrt(times) with nothing in the report to say so.
    task = _dummy_task_factory(rows=[{"id": i} for i in range(3)], repeat=2)
    finals = [task.make_context("not-an-index")]
    with pytest.raises(ValueError, match="cannot be grouped"):
        task.problem_groups(finals)
```

Write `_dummy_task_factory` as a fixture in that module, building a minimal
concrete `Task` over an in-memory `HFDatasetDict`, applying `.repeat(...)` and
`.shuffle(seed=...)` when asked. Follow the existing `Task` subclass already used
in `test_task.py` for the abstract-method stubs.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pdm run pytest tests/unit/core/tasks/test_task.py -k problem_groups -v`
Expected: FAIL — `AttributeError: 'Task' object has no attribute 'problem_groups'`

- [ ] **Step 3: Write the implementation**

In `task.py`, add to the imports: `from collections.abc import Hashable, Sequence`,
`from dataclasses import dataclass`, and extend the existing
`from sieval.core.datasets import Dataset, repeat_index_of` to also bring
`REPEAT_GROUP_COLUMN, repeat_group_of`.

Above the `Task` class:

```python
@dataclass(frozen=True)
class ProblemGrouping:
    """Which problem each judged sample belongs to, and how many there are.

    The two travel together because neither is usable alone: the keys say how to
    collapse the samples, and *n_problems* says what to divide by afterwards --
    read off the whole split, so a problem whose every copy failed still occupies
    a slot and the population does not shrink with run health.
    """

    keys: list[Hashable]
    n_problems: int
```

After `make_context`:

```python
    def problem_groups(self, finals: Sequence[object]) -> ProblemGrouping | None:
        """Which problem each of *finals* belongs to, or ``None`` if each is its own.

        The clustering unit for an interval on this task's headline. ``None`` means
        no collapsing is needed -- the ordinary case, where one sample is one
        problem.

        Read off the LIVE DATASET rather than off the contexts, and that is the
        load-bearing choice: ``repeat_index`` is written by
        ``TaskContext.serialize`` but never read back, and the runner's resume
        backfill covers only non-terminal samples -- so on a resumed run the
        samples a report aggregates carry no copy number at all. A grouping keyed
        on the context would disable itself on every resume and narrow every
        interval by ``sqrt(times)`` with nothing to say it had. ``sample_id``
        indexes the post-transform test set, the same relation the runner's backfill
        relies on, so this resolves identically fresh and resumed.

        Override this in a task whose clustering comes from somewhere other than
        ``Dataset.repeat``. There is exactly one grouping per task and the task owns
        it, so core and a task cannot both collapse the same samples. A task
        publishing metrics on two different axes -- a per-version rate beside a
        per-problem one -- must say per metric which axis its interval belongs to.

        Raises:
            ValueError: if the split is repeated but a sample cannot be placed in
                it. Silence here is the failure that cannot be seen in the report.
        """
        test_set = self._dataset.test_set
        if not test_set or REPEAT_GROUP_COLUMN not in test_set.column_names:
            return None
        keys: list[Hashable] = []
        for final in finals:
            sample_id = getattr(final, "sample_id", None)
            group = (
                repeat_group_of(test_set[sample_id])
                if isinstance(sample_id, int) and 0 <= sample_id < len(test_set)
                else None
            )
            if group is None:
                raise ValueError(
                    f"sample {sample_id!r} cannot be grouped: this split carries a "
                    f"{REPEAT_GROUP_COLUMN!r} column, so every judged sample must "
                    f"resolve to one of its rows. Leaving it ungrouped would treat "
                    f"the copies of one problem as independent problems and narrow "
                    f"every interval."
                )
            keys.append(group)
        return ProblemGrouping(keys, len(set(test_set[REPEAT_GROUP_COLUMN])))
```

Export `ProblemGrouping` from `sieval/core/tasks/__init__.py` alongside `Task`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pdm run pytest tests/unit/core/tasks/test_task.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
pdm run ruff format sieval/core/tasks/ tests/unit/core/tasks/test_task.py
pdm run ruff check sieval/core/tasks/ tests/unit/core/tasks/test_task.py
pdm run ty check
git add sieval/core/tasks/ tests/unit/core/tasks/test_task.py
git commit -m "feat(core): read a task's problem grouping off the live dataset"
```

---

### Task 4: `interval_metrics` — collapse, then estimate

**Files:**

- Modify: `sieval/core/tasks/metrics.py` (append after `wilson_interval`)
- Test: `tests/unit/core/tasks/test_metrics.py`

**Interfaces:**

- Consumes: `wilson_interval` (Task 1), `ProblemGrouping` — passed structurally as
  two arguments, so `metrics.py` stays free of the `task` import.
- Produces: `interval_metrics(values: Sequence[float], *, denominator: int, group_keys: Sequence[Hashable] | None = None, n_problems: int | None = None) -> dict[str, float | list[float]]`
  returning `{"score_ci95": [lo, hi], "n_problems": float(G)}`, either key omitted
  only with the other.

The arithmetic, which is what keeps this additive: with `G` problems, declared
denominator `D`, and `v_g` the sum of group `g`'s per-sample values, the per-problem
unit is `u_g = v_g * G / D`. Then `mean(u) = sum(values) / D` — bit-for-bit today's
score — so no stored score moves. Unrepeated, `G == D` and `u_g == values[g]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_interval_metrics_reports_the_problem_count_and_a_pair():
    got = interval_metrics([1.0] * 10 + [0.0] * 10, denominator=20)
    assert got["n_problems"] == 20.0
    assert isinstance(got["score_ci95"], list)
    assert len(got["score_ci95"]) == 2


def test_interval_metrics_collapsing_does_not_move_the_mean():
    # The whole change is additive: collapsing must leave `score` alone.
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    keys = [0, 1, 2, 0, 1, 2]
    flat = interval_metrics(values, denominator=6)
    grouped = interval_metrics(values, denominator=6, group_keys=keys, n_problems=3)
    assert grouped["n_problems"] == 3.0
    # Same headline, wider interval -- three problems, not six.
    assert (grouped["score_ci95"][1] - grouped["score_ci95"][0]) > (
        flat["score_ci95"][1] - flat["score_ci95"][0]
    )


def test_interval_metrics_widens_by_root_times_on_a_pure_repeat():
    # A 4x repeat of the same 50/50 split: the honest interval is ~2x wider.
    per_problem = [1.0, 0.0] * 25
    flat = interval_metrics(per_problem * 4, denominator=200)
    keys = [g for _ in range(4) for g in range(50)]
    grouped = interval_metrics(
        per_problem * 4, denominator=200, group_keys=keys, n_problems=50
    )
    ratio = (grouped["score_ci95"][1] - grouped["score_ci95"][0]) / (
        flat["score_ci95"][1] - flat["score_ci95"][0]
    )
    assert 1.8 < ratio < 2.2


def test_interval_metrics_omits_both_keys_together_when_it_cannot_estimate():
    assert interval_metrics([0.5] * 4, denominator=4) == {}
    assert interval_metrics([1.0], denominator=1) == {}


def test_interval_metrics_rejects_a_grouping_that_does_not_align():
    with pytest.raises(ValueError, match="one key per value"):
        interval_metrics([1.0, 0.0], denominator=2, group_keys=[0], n_problems=1)


def test_interval_metrics_needs_the_problem_count_with_the_keys():
    with pytest.raises(ValueError, match="n_problems"):
        interval_metrics([1.0, 0.0], denominator=2, group_keys=[0, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pdm run pytest tests/unit/core/tasks/test_metrics.py -k interval_metrics -v`
Expected: FAIL — `ImportError: cannot import name 'interval_metrics'`

- [ ] **Step 3: Write the implementation**

```python
#: Report key carrying the interval on the headline, as ``[lo, hi]``.
SCORE_CI_FIELD = "score_ci95"

#: Report key carrying the population the interval was clustered over.
PROBLEM_COUNT_FIELD = "n_problems"


def interval_metrics(
    values: Sequence[float],
    *,
    denominator: int,
    group_keys: Sequence[Hashable] | None = None,
    n_problems: int | None = None,
) -> dict[str, float | list[float]]:
    """The headline's interval and the population it was measured over.

    *values* are the PER-SAMPLE contributions to the headline, in the caller's own
    units -- whichever quantity that task's ``score`` is a mean of. Passed in rather
    than picked here, because only the task knows which of its metrics the interval
    belongs to, and a task publishing rates on two different axes must not have one
    guessed for it.

    *group_keys* collapses samples that are not independent problems -- the copies
    ``Dataset.repeat`` makes. With ``G`` problems and declared denominator ``D``, a
    group's summed value ``v`` becomes the per-problem unit ``v * G / D``, so the
    mean is ``sum(values) / D`` either way: collapsing widens the interval and
    leaves ``score`` bit-for-bit unchanged. Unrepeated, ``G == D`` and each unit is
    its own value.

    The two keys are emitted as a **pair or not at all**: an interval whose
    population is unknown cannot be read, and a population with no interval beside
    it is a count nothing asked for.

    Raises:
        ValueError: if *group_keys* is given without *n_problems*, or does not carry
            one key per value. Both would silently mis-scale the interval.
    """
    if group_keys is not None:
        if n_problems is None:
            raise ValueError(
                "interval_metrics: group_keys needs n_problems beside it -- the "
                "collapsed values are scaled by it, so guessing would mis-scale "
                "the interval."
            )
        if len(group_keys) != len(values):
            raise ValueError(
                f"interval_metrics: group_keys must carry one key per value; got "
                f"{len(group_keys)} keys for {len(values)} values."
            )
        sums: dict[Hashable, float] = collections.defaultdict(float)
        for key, value in zip(group_keys, values, strict=True):
            sums[key] += value
        scale = n_problems / denominator if denominator else 0.0
        units = [total * scale for total in sums.values()]
        population = n_problems
    else:
        units = list(values)
        population = denominator
    interval = wilson_interval(units, population)
    if interval is None:
        return {}
    return {
        SCORE_CI_FIELD: [interval[0], interval[1]],
        PROBLEM_COUNT_FIELD: float(population),
    }
```

Add `Hashable` to the `collections.abc` import at the top of `metrics.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pdm run pytest tests/unit/core/tasks/test_metrics.py -v`
Expected: PASS, whole module.

- [ ] **Step 5: Lint, type-check, commit**

```bash
pdm run ruff format sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
pdm run ruff check sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
pdm run ty check
git add sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
git commit -m "feat(core): collapse repeat copies before estimating the interval"
```

---

### Task 5: Wire it through `sampling_report`

**Files:**

- Modify: `sieval/core/tasks/metrics.py` (`sampling_report`, ~line 365-412)
- Test: `tests/unit/core/tasks/test_metrics.py`

**Interfaces:**

- Consumes: `interval_metrics` (Task 4), `ProblemGrouping` (Task 3, structurally).
- Produces: `sampling_report(..., score_key: str | None = None, grouping=None)`
  merging `interval_metrics`' keys when *score_key* names a key it computed.

- [ ] **Step 1: Write the failing tests**

```python
def test_sampling_report_adds_no_interval_without_a_score_key(_finals_factory):
    got = sampling_report(_finals_factory([[True], [False]]), n=1, k=1, denominator=2)
    assert "score_ci95" not in got
    assert "n_problems" not in got


def test_sampling_report_intervals_the_named_key(_finals_factory):
    finals = _finals_factory([[True], [False], [True], [False], [True]])
    got = sampling_report(finals, n=1, k=1, denominator=5, score_key="pass@1")
    assert got["n_problems"] == 5.0
    lo, hi = got["score_ci95"]
    assert lo < got["pass@1"] < hi


def test_sampling_report_refuses_a_score_key_it_did_not_compute(_finals_factory):
    with pytest.raises(ValueError, match="does not compute"):
        sampling_report(
            _finals_factory([[True], [False]]), n=1, k=1, denominator=2,
            score_key="maj@k",
        )


def test_sampling_report_interval_is_reported_at_n_equals_one(_finals_factory):
    # Not gated on n > 1: the interval is WIDEST at n=1, which is where a reader
    # most needs it -- the same argument health_metrics already makes.
    finals = _finals_factory([[True], [False], [True], [False]])
    got = sampling_report(finals, n=1, k=1, denominator=4, score_key="pass@1")
    assert "score_ci95" in got
```

`_finals_factory` builds judged finals from lists of per-rollout verdicts, reusing
the `build_judgement_record` / `build_rollout_judgement` helpers already imported
in this module.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pdm run pytest tests/unit/core/tasks/test_metrics.py -k sampling_report -v`
Expected: FAIL — `TypeError: sampling_report() got an unexpected keyword argument 'score_key'`

- [ ] **Step 3: Write the implementation**

Add the two parameters to `sampling_report`'s signature (keyword-only, after
`unit`), and after `rolled` is built:

```python
    if score_key is None:
        return rolled | budget_metrics(observed, n=n, k=k, unit=unit)
    if score_key not in rolled:
        raise ValueError(
            f"sampling_report: score_key {score_key!r} names a column this block "
            f"does not compute; got {sorted(rolled)}. A headline pointing at a "
            f"missing column would report an interval on a different metric."
        )
    values = [metrics.get(score_key, 0.0) for metrics in per_problem]
    return (
        rolled
        | budget_metrics(observed, n=n, k=k, unit=unit)
        | interval_metrics(
            values,
            denominator=denominator,
            group_keys=None if grouping is None else grouping.keys,
            n_problems=None if grouping is None else grouping.n_problems,
        )
    )
```

Extend the docstring with a paragraph on *score_key* naming the axis, and on the
interval being reported at every budget.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pdm run pytest tests/unit/core/tasks/ -v`
Expected: PASS.

- [ ] **Step 5: Coverage gate, then commit**

```bash
python -m coverage run --source=sieval -m pytest tests/unit/core/
python -m coverage report -m --include="sieval/core/tasks/metrics.py,sieval/core/datasets/dataset.py,sieval/core/tasks/task.py"
```

Expected: ≥ 95% on each of the three. Add tests for any uncovered branch before
committing.

```bash
git add sieval/core/tasks/metrics.py tests/unit/core/tasks/test_metrics.py
git commit -m "feat(core): report a problem-clustered interval on the named headline"
```

---

### Task 6: Adopt it in the sampling tasks

**Files:**

- Modify: the 24 modules calling `sampling_report` — `aime_2024_0shot_gen.py`,
  `aime_2025_0shot_gen.py`, `aime_2026_0shot_gen.py`, `apex_2025_0shot_gen.py`,
  `apex_shortlist_2025_0shot_gen.py`, `brumo_2025_0shot_gen.py`,
  `cmimc_2025_0shot_gen.py`, `gsm8k_0shot_gen.py`, `gsm8k_kshot_base_gen.py`,
  `hendrycks_math_kshot_base_gen.py`, `hmmt_feb_2025_0shot_gen.py`,
  `hmmt_feb_2026_0shot_gen.py`, `hmmt_nov_2025_0shot_gen.py`,
  `human_eval_0shot_base_gen.py`, `human_eval_0shot_gen.py`,
  `imo_answer_bench_0shot_gen.py`, `inverse_ifeval_0shot_gen.py`,
  `livecodebench_code_generation_0shot_gen.py`,
  `livecodebench_code_generation_kshot_base_gen.py`, `math_500_0shot_gen.py`,
  `mbpp_kshot_base_gen.py`, `platinum_bench/_base.py`, `smt_2025_0shot_gen.py`,
  `theoremqa_kshot_base_gen.py` — all under `sieval/tasks/`
- Test: the matching `tests/unit/tasks/test_*.py`

**Interfaces:**

- Consumes: `sampling_report(..., score_key=, grouping=)` (Task 5).
- Produces: `score_ci95` and `n_problems` in each task's `report()`.

- [ ] **Step 1: Write the failing test for one task first**

In `tests/unit/tasks/test_aime_2025_0shot_gen.py`:

```python
@pytest.mark.anyio
async def test_report_carries_an_interval_around_the_headline(...):
    report = await task.report(finals, [])
    lo, hi = report["score_ci95"]
    assert lo < report["score"] < hi
    assert report["n_problems"] == len(finals)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pdm run pytest tests/unit/tasks/test_aime_2025_0shot_gen.py -k interval -v`
Expected: FAIL — `KeyError: 'score_ci95'`

- [ ] **Step 3: Change that one task**

In `aime_2025_0shot_gen.py`'s `report()`, the `sampling_report` call becomes:

```python
        rolled = sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=total,
            normalize=normalize_vote,
            score_key="pass@1",
            grouping=self.problem_groups(finals),
        )
```

and the two new keys must reach the returned dict. They arrive inside `rolled`,
which is merged only under `if self._n > 1:` — so lift them out of that gate:

```python
        metrics: dict[str, float | str | list[float]] = {
            "score": pass_at_1,
            "fails": len(fails),
            "pass@1": pass_at_1,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        for field in (SCORE_CI_FIELD, PROBLEM_COUNT_FIELD):
            if field in rolled:
                metrics[field] = rolled[field]
        if self._n > 1:
            metrics.update(rolled)
```

Import `PROBLEM_COUNT_FIELD` and `SCORE_CI_FIELD` beside the existing
`SCORE_KEY_FIELD`.

- [ ] **Step 4: Run it to verify it passes**

Run: `pdm run pytest tests/unit/tasks/test_aime_2025_0shot_gen.py -v`
Expected: PASS.

- [ ] **Step 5: Commit the pattern**

```bash
git add sieval/tasks/aime_2025_0shot_gen.py tests/unit/tasks/test_aime_2025_0shot_gen.py
git commit -m "feat(tasks): aime_2025 reports a problem-clustered interval"
```

- [ ] **Step 6: Repeat steps 1-4 for the remaining 23 modules**

The `score_key` is **not** uniform — pass the value each module already gives
`SCORE_KEY_FIELD`. The single-draw family (`gsm8k_*`,
`hendrycks_math_kshot_base_gen`, `theoremqa_kshot_base_gen`, `platinum_bench/_base`)
publishes a first-rollout headline that `sampling_report` does not compute, so
those pass `score_key=None` and call `interval_metrics` directly with their own
per-problem values:

```python
        first = [
            1.0 if ((f.feedback_result or {}).get("rollouts") or [{}])[0].get("correct")
            else 0.0
            for f in finals
        ]
        grouping = self.problem_groups(finals)
        interval = interval_metrics(
            first,
            denominator=total,
            group_keys=None if grouping is None else grouping.keys,
            n_problems=None if grouping is None else grouping.n_problems,
        )
```

merged into the returned dict. This is the axis rule in force: the interval is
over the same quantity the headline is a mean of, never a neighbouring one.

- [ ] **Step 7: Commit in batches of four, then run the whole task suite**

```bash
pdm run pytest tests/unit/tasks/ -n 4
```

`-n 4`, not `-n auto`: `auto` ignores taskset/cgroup limits and the resulting
oversubscription trips the 30s grade timeout, which makes starved graders score
*wrong* rather than error.

---

### Task 7: Guide, preflight, and the gpqa regression

**Files:**

- Modify: `docs/guide/metrics.md`
- Modify: `scripts/check_preflight.py` (`check_report_declarations`, ~line 1835)
- Test: `tests/unit/tasks/test_gpqa_diamond_0shot_gen.py`

**Interfaces:**

- Consumes: everything above.
- Produces: no new code interface.

- [ ] **Step 1: Write the failing regression test**

`gpqa_diamond_0shot_gen` calls `dataset.repeat(n_repeats)` with `n_repeats=4` by
default, so it is the task where the silent failure mode is live. In
`tests/unit/tasks/test_gpqa_diamond_0shot_gen.py`:

```python
@pytest.mark.anyio
async def test_report_counts_questions_not_copies(...):
    # 4 copies of each question: n_problems is the QUESTION count. Reading 4x
    # here would narrow the interval by 2x and nothing in the report would say so.
    task = GpqaDiamond0shotGen(dataset, model, n_repeats=4)
    report = await task.report(finals, [])
    assert report["n_problems"] == len(dataset.test_set) / 4
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pdm run pytest tests/unit/tasks/test_gpqa_diamond_0shot_gen.py -k n_problems -v`
Expected: FAIL — the MCQ four take no `n`/`k` and do not call `sampling_report`,
so they need the `interval_metrics` route from Task 6 Step 6.

- [ ] **Step 3: Add the interval to the four MCQ tasks**

`gpqa_diamond_0shot_gen`, `mmlu_0shot_gen`, `mmlu_pro_0shot_gen`,
`openbookqa_kshot_gen` — same `interval_metrics` call as Task 6 Step 6, over
their first-rollout verdicts. Their `score_<category>` keys get **no** interval
here; that is the first follow-up.

- [ ] **Step 4: Run it to verify it passes**

Run: `pdm run pytest tests/unit/tasks/test_gpqa_diamond_0shot_gen.py -v`

- [ ] **Step 5: Write the guide section**

In `docs/guide/metrics.md`, add to the "Always present" table:

| Key | Meaning |
| --- | --- |
| `score_ci95` | 95% interval on `score`, as `[lo, hi]`, clustered on problems |
| `n_problems` | Distinct problems the headline was averaged over |

Then a section covering, each in a short paragraph: the three estimands and which
one this is; that width is **not** comparable across budgets, because the interval
carries rollout noise at `n = 1` and sheds it as `n` rises, converging to a floor
set by `n_problems`; that an interval must be read with its `denominator_policy`
and its `n_problems`; that a paired delta must never be inferred from two per-run
intervals; that a task publishing rates on two axes says which axis each interval
belongs to; and that `hle_0shot_gen`'s `confidence_interval` is a different
estimator — a Wald half-width pooled over attempts, not clustered over problems —
kept because it is HLE's published convention, exactly as `accuracy` sits beside
`pass@1`.

- [ ] **Step 6: Extend the preflight check**

In `check_report_declarations`, add a fifth rule: a report writing `score_ci95`
must also write `n_problems`, and vice versa. Mirror the existing `_merged_sources`
tracing so a key merged in from `metrics.py` counts.

- [ ] **Step 7: Run preflight and the full suite**

```bash
pdm run python scripts/check_preflight.py
pdm run pytest tests/ -n 4
```

- [ ] **Step 8: Commit**

```bash
git add docs/guide/metrics.md scripts/check_preflight.py sieval/tasks/ tests/
git commit -m "docs(metrics): document the headline interval and its clustering"
```

---

## Self-Review

**Spec coverage.** §1 scope → Tasks 1-7. §2 estimands → Task 7 Step 5. §3 estimator,
both divisors → Task 1. §4 `repeat_group` → Task 2; read-from-dataset → Task 3;
task-owned override → Task 6 Step 6 and Task 8 below. §5 report keys, pair rule,
no `n > 1` gate → Tasks 4-6. §6 omission and Clopper-Pearson → Task 1. §7 rollout
interval excluded → nothing to build. §8 guide and preflight → Task 7. §9 tests →
distributed.

**Gap found:** the spec's UGMathBench override has no task. Added below as Task 8.

**Type consistency.** `ProblemGrouping.keys` / `.n_problems` are used under those
names in Tasks 4, 5, 6 and 8. `score_key` is a `str | None` throughout.
`interval_metrics` takes `group_keys` / `n_problems` (not a `ProblemGrouping`) in
every call site, keeping `metrics.py` free of a `task` import.

---

### Task 8: The UGMathBench override

**Files:**

- Modify: `sieval/tasks/ugmathbench_0shot_gen_fixed.py` (`report`, ~line 429)
- Test: `tests/unit/tasks/test_ugmathbench_0shot_gen_fixed.py`

**Interfaces:**

- Consumes: `ProblemGrouping` (Task 3), `interval_metrics` (Task 4).
- Produces: an override whose keys are `problem_id`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.anyio
async def test_interval_clusters_on_problems_not_versions(...):
    # One sample is one (problem, version) pair, three versions per problem. The
    # headline is EAcc, a per-PROBLEM statistic, so the interval's population is
    # the problem count. Using the version count is the same sqrt(times)
    # narrowing as an uncollapsed repeat, under a different name.
    report = await task.report(finals, [])
    assert report["n_problems"] == n_distinct_problem_ids
    assert report["n_problems"] < len(finals)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pdm run pytest tests/unit/tasks/test_ugmathbench_0shot_gen_fixed.py -k clusters -v`
Expected: FAIL — `KeyError: 'n_problems'`

- [ ] **Step 3: Write the override**

The task already extracts `problem_id` from each judgement's `extra`, with an
`_identify(final)` fallback for a version that cannot name its problem. Reuse
exactly that, so the interval's grouping cannot disagree with the `by_problem`
grouping `report()` already builds:

```python
    @override
    def problem_groups(self, finals) -> ProblemGrouping | None:
        """Cluster on the PROBLEM, which is EAcc's unit and the headline's.

        UGMathBench's clustering is inherent to the data -- one sample is one
        (problem, version) pair, three versions per problem -- so it never routes
        through ``Dataset.repeat`` and the base implementation would find no
        column. Grouping on ``problem_id`` also absorbs a repeat for free: copies
        of a version share their problem, so the two keys never need combining.

        AAcc is per VERSION and deliberately gets no interval from this: reusing
        the problem count for it would report a per-problem width on a per-version
        rate.
        """
        keys = []
        for final in finals:
            problem_id = ((final.feedback_result or {}).get("extra") or {}).get(
                "problem_id"
            )
            if problem_id is None:
                problem_id, _ = _identify(final)
            keys.append(problem_id)
        return ProblemGrouping(keys, len(set(keys)))
```

Then pass it to `interval_metrics` over the per-problem EAcc values the existing
`by_problem` loop already produces.

- [ ] **Step 4: Run it to verify it passes**

Run: `pdm run pytest tests/unit/tasks/test_ugmathbench_0shot_gen_fixed.py -v`

- [ ] **Step 5: Commit**

```bash
git add sieval/tasks/ugmathbench_0shot_gen_fixed.py tests/unit/tasks/test_ugmathbench_0shot_gen_fixed.py
git commit -m "feat(tasks): ugmathbench clusters its interval on problems, not versions"
```

Note on `n_problems` here: unlike the base implementation it counts the problems
*observed*, not the whole split, because a UGMathBench problem is not a repeat of
a row and the split has no column to count. A problem whose every version failed
is therefore absent from the population — a known, narrower reading, worth a line
in the task's `reference_impl.notes`.
