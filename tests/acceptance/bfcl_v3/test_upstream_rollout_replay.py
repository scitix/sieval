"""Anchor BFCL v3 grading on upstream's own released rollouts and verdicts.

BFCL publishes an evaluation archive beside the leaderboard: per model, per
protocol, per category, both the model's recorded replies and its own per-row
verdicts. That is per-prediction ground truth, so the port pins exactly with no
model spend.

Stronger than `test_gold_replay.py`, which feeds gold back and so only ever
exercises the *correct* path -- never a wrong answer, a decode failure, or
either protocol's decoder. This replays real output, 9% of it wrong, through the
production `_decode` of both mixins. `java` at 64% is the point: a third of
those rows are wrong and we have to agree about which third.

It does NOT cover prompt construction -- the replies are upstream's. That rests
on the two model-facing helpers being byte-identical to upstream (pinned by
`test_identity.py`) and running on a row-for-row verified snapshot: a
construction argument, not a measurement.

No file hashes are pinned, and they would be redundant: a swapped archive fails
the :data:`PUBLISHED` header assertion, and a doctored `result` file fails the
comparison it feeds.

Staging is opt-in, like the gold replay -- ~6MB, so not vendored:

    R=https://raw.githubusercontent.com/HuanzhiMao/BFCL-Result/main/2025-06-14
    D="$SIEVAL_DATA_DIR/HuanzhiMao/BFCL-Result/2025-06-14"
    for m in gpt-4.1-2025-04-14 gpt-4.1-2025-04-14-FC; do
      for k in result score; do
        mkdir -p "$D/$k/$m"
        for c in simple multiple parallel parallel_multiple java javascript \
                 irrelevance live_simple live_multiple live_parallel \
                 live_parallel_multiple live_irrelevance live_relevance; do
          curl -sSf -o "$D/$k/$m/BFCL_v3_${c}_${k}.json" \
            "$R/$k/$m/BFCL_v3_${c}_${k}.json"
        done
      done
    done

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import collections
import json
import pathlib

import pytest

from sieval.core.models import FunctionToolCall, GenModel, ModelOutput
from sieval.core.utils.hf import maybe_resolve_hf_path
from sieval.datasets import BfclV3LiveDataset, BfclV3NonLiveDataset
from sieval.tasks.bfcl_v3._base import (
    BfclV3FCMixin,
    BfclV3LiveTask,
    BfclV3NonLiveTask,
    BfclV3PromptMixin,
    grade_single_turn,
)

#: The archive snapshot these counts were read from. It predates the pinned
#: harness revision (v1.3, 2025-07-17) by a month, and agrees anyway -- which is
#: itself evidence the checker did not move for these categories in between.
ARCHIVE = "HuanzhiMao/BFCL-Result"
SNAPSHOT = "2025-06-14"

DATA_SNAPSHOT = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

NON_LIVE = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "irrelevance",
)
LIVE = (
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)

#: `(correct_count, total_count)` off each score file's header line -- upstream's
#: published cell, as integers rather than the rounded percentage the
#: leaderboard renders. Asserted both against the staged file (archive
#: integrity) and against our own replay (the actual anchor).
PUBLISHED = {
    "gpt-4.1-2025-04-14": {
        "simple": (382, 400),
        "multiple": (188, 200),
        "parallel": (186, 200),
        "parallel_multiple": (175, 200),
        "java": (64, 100),
        "javascript": (41, 50),
        "irrelevance": (213, 240),
        "live_simple": (221, 258),
        "live_multiple": (806, 1053),
        "live_parallel": (15, 16),
        "live_parallel_multiple": (18, 24),
        "live_irrelevance": (687, 882),
        "live_relevance": (16, 18),
    },
    "gpt-4.1-2025-04-14-FC": {
        "simple": (374, 400),
        "multiple": (181, 200),
        "parallel": (182, 200),
        "parallel_multiple": (172, 200),
        "java": (61, 100),
        "javascript": (34, 50),
        "irrelevance": (215, 240),
        "live_simple": (207, 258),
        "live_multiple": (825, 1053),
        "live_parallel": (11, 16),
        "live_parallel_multiple": (16, 24),
        "live_irrelevance": (726, 882),
        "live_relevance": (14, 18),
    },
}

#: Protocol under test: archive model dir, and the mixin whose production
#: `_decode` and `UNDERSCORE_TO_DOT` are exercised.
PROTOCOLS = {
    "prompt": ("gpt-4.1-2025-04-14", BfclV3PromptMixin),
    "fc": ("gpt-4.1-2025-04-14-FC", BfclV3FCMixin),
}

_ROOT = pathlib.Path(maybe_resolve_hf_path(ARCHIVE)) / SNAPSHOT

requires_archive = pytest.mark.skipif(
    not _ROOT.is_dir(),
    reason=f"needs the staged BFCL result archive ({ARCHIVE} @ {SNAPSHOT})",
)


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def _model_output(protocol: str, recorded) -> ModelOutput:
    """Upstream's recorded reply, in the shape the mixin's `_decode` reads.

    A real `ModelOutput`, not a stand-in with the two attributes `_decode`
    touches. The FC archive stores each call as `{name: arguments-JSON-string}`,
    which is what the OpenAI dialects put on `FunctionToolCall.arguments`, so
    the calls are rebuilt rather than pre-parsed and `_call_arguments` is part
    of what this anchors.
    """
    meta = GenModel(model="archive-replay", api_key="none").meta()
    if protocol == "prompt":
        text = recorded if isinstance(recorded, str) else ""
        return ModelOutput(model=meta, texts=[text])
    calls = []
    for item in recorded if isinstance(recorded, list) else []:
        if not isinstance(item, dict):
            continue
        for index, (name, arguments) in enumerate(item.items()):
            calls.append(
                FunctionToolCall(
                    f"call_{len(calls)}_{index}",
                    name,
                    arguments if isinstance(arguments, str) else json.dumps(arguments),
                )
            )
    return ModelOutput(model=meta, texts=[""], tool_calls=tuple(calls))


def _decode_or_fail(decoder, protocol: str, recorded, language):
    """The mixin's production `_decode`, under `postprocess`'s failure semantic.

    `postprocess` wraps the call in `except Exception -> [None]`: an undecodable
    reply is upstream's `ast_decoder:decoder_failed`, a SCORE rather than a
    fault. Mirrored here or a real reply takes the test out -- a model that
    declines in prose raises `SyntaxError` straight out of `ast.parse`. Only
    those two lines are mirrored; the decoder is the shipped one, and
    `postprocess`'s narrower `ImportError` re-raise is pinned in the unit tests.
    """
    try:
        return decoder._decode(_model_output(protocol, recorded), language)
    except Exception:  # noqa: BLE001 -- a decode failure is a score
        return None


def _replay(protocol: str, categories, dataset) -> dict[str, tuple[int, int]]:
    """Grade every recorded reply; return `{category: (ours, total)}`.

    Asserts row-for-row agreement with upstream's verdicts on the way, so a
    disagreement names the row rather than only moving a total.
    """
    model_dir, mixin = PROTOCOLS[protocol]
    decoder = mixin()

    # `live_relevance` ships one id twice (18 rows, 17 distinct ids), so rows and
    # recorded replies are both consumed from per-id queues. Keying either by id
    # would drop that row and quietly score 17.
    rows = collections.defaultdict(collections.deque)
    for row in dataset.test_set:
        rows[row["id"]].append(row)

    out: dict[str, tuple[int, int]] = {}
    for category in categories:
        base = _ROOT / "{kind}" / model_dir / f"BFCL_v3_{category}_{{kind}}.json"
        results = _jsonl(pathlib.Path(str(base).format(kind="result")))
        scored = _jsonl(pathlib.Path(str(base).format(kind="score")))

        header = scored[0]
        expected = PUBLISHED[model_dir][category]
        assert (header["correct_count"], header["total_count"]) == expected, (
            f"{model_dir}/{category}: staged archive publishes "
            f"{header['correct_count']}/{header['total_count']}, pinned "
            f"{expected[0]}/{expected[1]} -- the archive moved, so every number "
            "below was measured against a different corpus."
        )
        wrong = collections.Counter(r["id"] for r in scored[1:] if "id" in r)

        pool = {k: collections.deque(v) for k, v in rows.items()}
        correct = walked = 0
        disagreed = []
        for record in results:
            queue = pool.get(record["id"])
            if not queue:
                continue
            row = queue.popleft()
            walked += 1
            ours = grade_single_turn(
                row["function"],
                _decode_or_fail(decoder, protocol, record["result"], row["language"]),
                row["ground_truth"],
                row["language"],
                category,
                mixin.UNDERSCORE_TO_DOT,
            )
            correct += int(ours)
            if ours != (wrong[record["id"]] == 0):
                disagreed.append(record["id"])

        assert walked == expected[1], (
            f"{model_dir}/{category}: replayed {walked} of {expected[1]} rows. "
            "A short walk scores the missing rows as absent, not as wrong."
        )
        assert not disagreed, (
            f"{model_dir}/{category}: our verdict differs from upstream's on "
            f"{len(disagreed)} row(s), e.g. {disagreed[:5]}"
        )
        out[category] = (correct, expected[1])
    return out


@requires_archive
@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
@pytest.mark.parametrize(
    ("group", "categories"), [("non_live", NON_LIVE), ("live", LIVE)]
)
def test_replaying_upstream_rollouts_reproduces_every_published_cell(
    protocol: str, group: str, categories
):
    dataset = (
        BfclV3NonLiveDataset(DATA_SNAPSHOT)
        if group == "non_live"
        else BfclV3LiveDataset(DATA_SNAPSHOT)
    )
    model_dir, _ = PROTOCOLS[protocol]
    replayed = _replay(protocol, categories, dataset)
    assert replayed == {c: PUBLISHED[model_dir][c] for c in categories}


@requires_archive
@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_the_live_rollup_is_the_pooled_rate_over_every_live_row(protocol: str):
    """The weighted rollup's own claim, checked rather than asserted in prose.

    `BfclV3LiveTask` documents its weighted mean as algebraically the pooled
    rate over the union of the six categories' rows -- which is what makes the
    headline a genuine per-sample rate, and so interval-bearing.
    """
    model_dir, _ = PROTOCOLS[protocol]
    cells = {c: PUBLISHED[model_dir][c] for c in LIVE}

    def cell(category: str) -> dict:
        correct, total = cells[category]
        accuracy = 100.0 * correct / total
        return {
            "accuracy": accuracy,
            "total_count": total,
            "display_accuracy": accuracy,
        }

    # Built without `__init__` (which would demand a live model binding); the
    # rollup reads only ClassVars. Runs the PRODUCTION aggregation.
    task = object.__new__(BfclV3LiveTask)
    rollup = task._aggregate(cell)  # noqa: SLF001 -- tests are the carve-out
    pooled = (
        100.0 * sum(c for c, _ in cells.values()) / sum(t for _, t in cells.values())
    )
    assert rollup["live_overall_acc"] == pytest.approx(pooled, abs=1e-9)


@requires_archive
@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_the_non_live_rollup_is_the_documented_mean_of_means(protocol: str):
    """Non-live nests `simple_ast` inside a five-way unweighted mean.

    Deliberately NOT the pooled rate: javascript's 50 rows weigh as much as
    simple's 400, which is why the task publishes no interval on this headline.
    Pinned here so a "fix" toward pooling fails loudly.
    """
    model_dir, _ = PROTOCOLS[protocol]
    cells = {c: PUBLISHED[model_dir][c] for c in NON_LIVE}

    def rate(category: str) -> float:
        correct, total = cells[category]
        return 100.0 * correct / total

    def cell(category: str) -> dict:
        return {
            "accuracy": rate(category),
            "total_count": cells[category][1],
            "display_accuracy": rate(category),
        }

    task = object.__new__(BfclV3NonLiveTask)
    rollup = task._aggregate(cell)  # noqa: SLF001 -- tests are the carve-out
    simple_ast = sum(rate(c) for c in ("simple", "java", "javascript")) / 3
    expected = (
        simple_ast
        + sum(
            rate(c)
            for c in ("multiple", "parallel", "parallel_multiple", "irrelevance")
        )
    ) / 5
    assert rollup["simple_ast"] == pytest.approx(simple_ast, abs=1e-9)
    assert rollup["non_live_overall_acc"] == pytest.approx(expected, abs=1e-9)
    pooled = (
        100.0 * sum(c for c, _ in cells.values()) / sum(t for _, t in cells.values())
    )
    assert rollup["non_live_overall_acc"] != pytest.approx(pooled, abs=1e-6)
