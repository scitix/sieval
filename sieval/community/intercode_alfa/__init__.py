"""InterCode-ALFA scoring: upstream's arithmetic, without its container plumbing.

Vendored from ``westenfelder/InterCode-ALFA`` at commit ``2d3a6947``, which is
``icalfa`` 0.3.6 -- the version the NAACL 2025 paper (arXiv:2502.06858) states
Table 5 was produced with. The gold tables under ``assets/`` were last changed
2024-11-23 ("Updated dataset to nl2bash with 300 test cases"); every commit after
that is documentation, so these bytes are the ones behind the published numbers.

``parse_bash`` comes from a *different* upstream repo -- ``westenfelder/NL2SH``,
``code/model_comparison.ipynb`` and ``code/example.ipynb`` -- because the harness
does not parse model output; the paper's translation methods do. Same authors,
same benchmark, both MIT.

**What is verbatim here**: ``index_to_img``'s split table, ``parse_status``,
``get_hash_cmd``, ``clean_cmd``, ``parse_bash``, the three reward parts with their
``math.erf`` / ``round(..., 2)`` shapes and the ``0.01`` base, the ``A``/``??``/``C``
status filter, the 1000-character embedding truncation, the empty-output rule and
the strict ``similarity > threshold`` comparison.

**What is left out, and why**: everything that talks to Docker or to a model.
``BashEnv``/``IntercodeEnv`` drive two containers over ``docker-py`` and call
OpenAI or Ollama inline; in sieval those two halves belong to different layers --
execution to the code-eval service (which is the sandbox), the embedding call to
the task (which owns model credentials and records). So this module is the
arithmetic between them: it takes execution facts and a similarity, and returns
upstream's reward. Nothing here performs I/O beyond reading the gold tables.

Three upstream behaviours are preserved deliberately even though they look like
defects, because reproducing the published number requires them:

* ``get_hash_cmd`` picks ``md5deep -r`` for any path without a ``.`` in it, and
  no image installs ``md5deep``. Both sides therefore get the same failure text
  and compare equal, so a directory-only change always scores full credit on
  part 2.
* the part-2 status filter is ``["A", "??", "C"]`` -- ``M`` is absent, so a
  *modified* file's content is never compared, despite upstream's comment saying
  "added or modified".
* ``parse_status`` splits on whitespace and re-pairs tokens two at a time, so a
  path containing a space, or a rename's ``->`` arrow, shifts every later pair.

See ``docs/designs/specs/2026-09-04-nl2sh-alfa-design.md`` for the measurements
behind each.
"""

import json
import math
import pathlib
import re
import shlex
from typing import TypedDict

#: Upstream commit these bytes and this arithmetic come from.
ICALFA_COMMIT = "2d3a69473a68569828ab0b4859073ef4a0ae482c"
ICALFA_VERSION = "0.3.6"

#: `westenfelder/NL2SH` commit `parse_bash` and the two prompts come from.
NL2SH_COMMIT = "b405201aeac7630353e7b46ebe7d069b70b4506c"

#: `BashEnv.get_reward`'s base, added before any part scores. Load-bearing:
#: `0.01 + 0.33 + 0.33 + 0.33 == 1.0` exactly in IEEE double, and
#: `submit_command` returns 1 only on `reward == 1`.
REWARD_BASE = 0.01
#: Each part's full credit. Three of them plus the base make exactly 1.0.
PART_CREDIT = 0.33
#: `get_reward`'s part-2 filter: which `git status --short` codes are content-
#: compared. `M` is deliberately absent -- see the module docstring.
HASHED_STATUS_CODES = ("A", "??", "C")
#: `BashEnv.get_reward`'s embed branch truncates both outputs to this many
#: characters *before* embedding them.
EMBED_TRUNCATION = 1000
#: The `eval_param` upstream's own reproduction scripts pass for `eval_mode="embed"`.
DEFAULT_EMBED_THRESHOLD = 0.75
#: `utils.TIMEOUT_DURATION`, the wall `exec_action` puts on the model's command.
TIMEOUT_DURATION = 10
#: `exec_action`'s observation when that wall fires. Compared as an output, so
#: the exact string matters.
TIMEOUT_OBSERVATION = "Command timed out"

#: `main.index_to_img`'s split table: how many of the 300 test rows each of the
#: five filesystem images carries, in order.
IMAGE_SPLITS = (153, 49, 57, 23, 18)

#: `bash_env.IMAGE_TO_SETTINGS`, keyed by filesystem id rather than image name.
#: Image 5 is Alpine, whose shell is `/bin/sh`.
FS_ENTRYPOINT = {1: "/bin/bash", 2: "/bin/bash", 3: "/bin/bash", 4: "/bin/bash", 5: "/bin/sh"}

#: `bash_env.GIT_RESET_SCRIPT` / `GIT_STATUS_SCRIPT`.
GIT_RESET_SCRIPT = "git reset --hard; git clean -fd;"
GIT_STATUS_SCRIPT = "git status --short;"

_ASSETS = pathlib.Path(__file__).parent / "assets"


class GoldRow(TypedDict):
    """One row of upstream's vendored gold table.

    ``query`` is the harness's own copy of the instruction, which differs from
    the Hub's ``nl`` on three rows; ``gold`` is what execution is graded against
    and differs from the Hub's ``bash`` on two. ``gold2`` is never read by the
    harness.
    """

    query: str
    gold: str
    gold2: str
    difficulty: int


def index_to_img(index: int) -> tuple[int, int]:
    """`main.index_to_img`, verbatim: test index -> (index within image, image).

    Returns upstream's 0-based image number; callers wanting a filesystem id add
    one. Raises for ``index >= 300``, which is what upstream's
    ``submit_command`` swallows into a score of 0.
    """
    index = int(index)
    cumsum = 0

    for img_num, count in enumerate(IMAGE_SPLITS, start=0):
        if index < cumsum + count:
            idx = index - cumsum
            return idx, img_num
        cumsum += count

    raise ValueError("Index out of allowable range")


def fs_id_for_index(index: int) -> int:
    """1-based filesystem id for a test index, as this port numbers the images."""
    return index_to_img(index)[1] + 1


def gold_table() -> tuple[GoldRow, ...]:
    """The five asset files concatenated in image order -- 300 rows.

    Position is the join key: ``index_to_img`` is positional, and upstream's own
    loop pairs the Hub's row *i* with asset row *i*.
    """
    rows: list[GoldRow] = []
    for i in range(1, len(IMAGE_SPLITS) + 1):
        with (_ASSETS / f"nl2bash_fs_{i}.json").open(encoding="utf-8") as handle:
            rows.extend(json.load(handle))
    return tuple(rows)


def parse_status(status: str) -> list[tuple[str, str]]:
    """`BashEnv.parse_status`, verbatim: `git status --short` -> (path, code).

    Splits on whitespace and re-pairs two tokens at a time, so a path with a
    space in it shifts every later pair. Preserved -- it is what upstream scores.
    """
    status_lst = status.split()
    changes = []
    for i in range(0, len(status_lst), 2):
        changes.append((status_lst[i + 1], status_lst[i]))
    return changes


def hash_command(path: str) -> str:
    """`BashEnv.get_reward`'s `get_hash_cmd`, verbatim.

    ``md5deep`` is not installed in any of the five images, so the second branch
    always fails -- identically on both sides, which is why it still compares
    equal. Run without a shell wrapper upstream (``exec_run(hash_cmd)``), so the
    caller must not wrap it in ``clean_cmd``.
    """
    return f"md5sum {path}" if "." in path else f"md5deep -r {path}"


def clean_cmd(entrypoint: str, action: str) -> str:
    """`BashEnv.clean_cmd`, verbatim -- note the *double* quotes."""
    return f'{entrypoint} -c "{action.strip()}"'


def command_argv(entrypoint: str, action: str) -> list[str]:
    """What the container actually executes, quoting quirk included.

    ``clean_cmd`` wraps the command in double quotes and hands the result to
    ``container.exec_run`` as a **string**; docker-py normalizes a string command
    through ``utils.split_command``, which is ``shlex.split``. So the text bash
    receives is not the command -- on the 300 pinned golds, 37 are rewritten and
    one (index 230) is truncated into a syntax error.

    Reproducing that is mandatory, not optional: passing
    ``[entrypoint, "-c", action]`` is the obviously-correct thing to write and
    disagrees with upstream on those 37 samples.
    """
    return shlex.split(clean_cmd(entrypoint, action))


def parse_bash(text: str) -> str:
    """`parse_bash` from ``westenfelder/NL2SH``'s notebooks, verbatim.

    Strips one markdown fence, trying ```` ```bash ````, then a bare ```` ``` ````,
    then a single backtick span; returns the text unchanged when none matches.
    Used by the paper's Parse / CD / IWL methods, not by the Base reading.
    """
    patterns = [
        r"```bash\s*(.*?)\s*```",
        r"```(.*?)```",
        r"`(.*?)`",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text


def file_diff_score(
    diff_agent: list[tuple[str, str]], diff_eval: list[tuple[str, str]]
) -> tuple[float, list[tuple[str, str]], list[tuple[str, str]]]:
    """Part 1, verbatim: how far the two filesystem states diverge.

    ``round(0.33 * (1 - erf(n)), 2)`` -- 0.33 at zero divergence, 0.05 at one,
    0.0 from two on. Returns the score and upstream's ``diff_miss`` /
    ``diff_extra`` lists, whose *lengths* are all the score reads.
    """
    diff_miss = list(set(diff_eval) - set(diff_agent))
    diff_extra = list(set(diff_agent) - set(diff_eval))
    score = round(
        PART_CREDIT * (1 - math.erf(len(diff_miss) + len(diff_extra))), 2
    )
    return score, diff_miss, diff_extra


def shared_changes(
    diff_agent: list[tuple[str, str]], diff_eval: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Part 2's ``diff_same``: changes both sides made, filtered to A / ?? / C.

    Intersects (path, code) *pairs*, so a path both sides touched under
    different codes is absent -- upstream's behaviour, and the reason filtering
    per side before intersecting is equivalent to filtering after.
    """
    return [
        change
        for change in list(set(diff_agent) & set(diff_eval))
        if change[1] in HASHED_STATUS_CODES
    ]


def file_change_score(
    diff_same: list[tuple[str, str]],
    agent_hashes: dict[str, str],
    eval_hashes: dict[str, str],
) -> tuple[float, int]:
    """Part 2, verbatim: of the paths both sides changed, how many match.

    Full credit when the two sides changed no path in common -- upstream's
    ``p2_score = 0.33`` default, reached before any hashing. The hash values are
    the *raw stdout* of ``hash_command``, compared as strings, because that is
    what upstream compares.

    Raises ``KeyError`` on a path missing from either map rather than treating it
    as a mismatch: the maps are produced by the same parse that produced
    ``diff_same``, so an absence means the two disagreed, which is a broken
    grader and not a wrong answer.
    """
    if not diff_same:
        return PART_CREDIT, 0
    same_changes = 0
    for path, _code in diff_same:
        same_changes += 1 if agent_hashes[path] == eval_hashes[path] else 0
    return round(PART_CREDIT * (same_changes / len(diff_same)), 2), same_changes


def output_similarity(gold_embedding, model_embedding) -> float:
    """`1 - scipy.spatial.distance.cosine(...)`, upstream's exact expression.

    scipy rather than a hand-rolled dot product, and imported here rather than at
    module scope: only the embed FEH needs it, and this package is imported by
    tasks whose dependency group does not carry scipy.
    """
    from scipy.spatial.distance import cosine

    return 1 - cosine(gold_embedding, model_embedding)


def truncate_for_embedding(output: str) -> str:
    """Upstream's ``output[:1000]``, applied before the embedding call."""
    return output[:EMBED_TRUNCATION]


def total_reward(p1: float, p2: float, p3: float) -> float:
    """`0.01 + p1 + p2 + p3`, summed in upstream's order."""
    reward = REWARD_BASE
    for part in (p1, p2, p3):
        reward += part
    return reward


def is_correct(reward: float) -> bool:
    """`submit_command`'s verdict: 1 iff the reward is *exactly* 1.

    An exact float comparison, deliberately. It is safe -- the four addends sum
    to 1.0 with no representation error -- and it is what makes the metric a
    strict conjunction of the three parts.
    """
    return reward == 1


__all__ = [
    "DEFAULT_EMBED_THRESHOLD",
    "EMBED_TRUNCATION",
    "FS_ENTRYPOINT",
    "GIT_RESET_SCRIPT",
    "GIT_STATUS_SCRIPT",
    "HASHED_STATUS_CODES",
    "ICALFA_COMMIT",
    "ICALFA_VERSION",
    "IMAGE_SPLITS",
    "NL2SH_COMMIT",
    "PART_CREDIT",
    "REWARD_BASE",
    "TIMEOUT_DURATION",
    "TIMEOUT_OBSERVATION",
    "GoldRow",
    "clean_cmd",
    "command_argv",
    "file_change_score",
    "file_diff_score",
    "fs_id_for_index",
    "gold_table",
    "hash_command",
    "index_to_img",
    "is_correct",
    "output_similarity",
    "parse_bash",
    "parse_status",
    "shared_changes",
    "total_reward",
    "truncate_for_embedding",
]
