"""A blank prediction must survive the round trip to disk and back.

`obj_to_dict` **drops None-valued keys** and `PredictionRecord.prediction` is
`NotRequired[JSONValue | None]`, so a rollout whose generation was empty persists as
``{"index", "extracted"}`` with no ``prediction`` at all. In-process the key is present
and a subscript works; the failure only appears when a lane is **re-graded from disk**
(`sieval run --resume` after clearing its terminal records), which is the path a harness
bump makes people take.

Found for real: re-grading a 90-rollout LiveCodeBench lane raised
``exception::KeyError 'prediction'`` on exactly the 4 truncated rollouts
(``finish_reason: length``, 0 chars of ``texts``).

This is a family, not an incident — most of `sieval/tasks/` subscripts `prediction`
directly. The test is written against the record contract so it can be extended to the
other tasks one at a time.
"""

import json

import pytest

from sieval.core.tasks import build_prediction_record
from sieval.core.utils.serialization import obj_to_dict


def _round_trip(record):
    """Exactly what the saver/loader do: serialize, write, read back.

    ``add_type=False`` matches a run without type metadata; the None-dropping that
    matters here is unconditional (``serialization.py``: ``if v is not None``).
    """
    return json.loads(json.dumps(obj_to_dict(record, False)))


def test_a_blank_prediction_loses_its_key_on_disk():
    """The premise. If this ever fails, the bug below is gone at the source."""
    on_disk = _round_trip(build_prediction_record([None, "code"]))
    assert "prediction" not in on_disk["rollouts"][0], (
        "expected obj_to_dict to drop the None-valued key; if it now keeps it, the "
        "consumers below no longer need .get and this test should be retired"
    )
    assert on_disk["rollouts"][0]["extracted"] is False
    assert on_disk["rollouts"][1]["prediction"] == "code"


def test_livecodebench_reads_a_reloaded_blank_prediction():
    """The regression: read the field the way `feedback` does, from a reloaded record.

    A subscript raises `KeyError`; `.get(...) or ""` yields the empty string the task's
    own comment says it wants ("an unextractable answer is None here but '' on the
    wire").
    """
    reloaded = _round_trip(build_prediction_record([None]))
    rollout = reloaded["rollouts"][0]
    assert rollout.get("prediction") or "" == ""

    with pytest.raises(KeyError):
        rollout["prediction"]  # the shape that shipped, kept as the counter-example


@pytest.mark.parametrize("predictions", [[None], [None, "x"], ["x", None]])
def test_every_rollout_is_addressable_after_a_reload(predictions):
    """Whatever the mix, a consumer using `.get` sees one entry per rollout."""
    reloaded = _round_trip(build_prediction_record(predictions))
    codes = [r.get("prediction") or "" for r in reloaded["rollouts"]]
    assert codes == [p or "" for p in predictions]
