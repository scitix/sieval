"""Unit tests for the vendored InterCode-ALFA arithmetic.

Everything here is checkable without a container, which is the point: the reward
parts, the branch filters and the command-quoting rewrite are pure functions over
the pinned gold table, so a port defect in any of them is caught before an image
exists. What is NOT covered here is what the five images do when a command runs
in them -- that needs Docker, and it is rung 2 of the validation ladder.

Several assertions pin upstream behaviour that reads like a bug. They are
deliberate: reproducing the published 0.74 requires them, and a future reader
"fixing" one silently moves every score. Each says so.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import math

import pytest

from sieval.community.intercode_alfa import (
    DEFAULT_EMBED_THRESHOLD,
    EMBED_TRUNCATION,
    FS_ENTRYPOINT,
    IMAGE_SPLITS,
    PART_CREDIT,
    REWARD_BASE,
    clean_cmd,
    command_argv,
    file_change_score,
    file_diff_score,
    fs_id_for_index,
    gold_table,
    hash_command,
    index_to_img,
    is_correct,
    parse_bash,
    parse_status,
    shared_changes,
    total_reward,
    truncate_for_embedding,
)

_TOTAL_ROWS = 300


def test_gold_table_is_300_rows_matching_the_split_table():
    table = gold_table()
    assert len(table) == _TOTAL_ROWS == sum(IMAGE_SPLITS)
    # Every row carries all four upstream columns; a missing `gold` would be
    # scored as an empty command rather than raising, so check the schema.
    assert all({"query", "gold", "gold2", "difficulty"} <= set(row) for row in table)


def test_difficulty_is_a_hundred_of_each_tier():
    tiers = [row["difficulty"] for row in gold_table()]
    assert [tiers.count(t) for t in (0, 1, 2)] == [100, 100, 100]


def test_index_to_img_walks_the_split_boundaries():
    # First and last index of each image, derived from the split table rather
    # than hardcoded, so a changed table fails here instead of silently
    # re-routing samples to the wrong filesystem.
    starts = [sum(IMAGE_SPLITS[:i]) for i in range(len(IMAGE_SPLITS))]
    for image, (start, count) in enumerate(zip(starts, IMAGE_SPLITS, strict=True)):
        assert index_to_img(start) == (0, image)
        assert index_to_img(start + count - 1) == (count - 1, image)
    assert [fs_id_for_index(i) for i in starts] == [1, 2, 3, 4, 5]


def test_index_past_the_end_raises():
    # `submit_command` swallows this into a score of 0 after a bare print, which
    # is how a caller who believes the card's "600 pairs" loses half a run
    # silently. The raise is the only signal, so it must stay a raise.
    with pytest.raises(ValueError, match="Index out of allowable range"):
        index_to_img(_TOTAL_ROWS)


def test_every_filesystem_has_an_entrypoint_and_only_five_do():
    assert set(FS_ENTRYPOINT) == {1, 2, 3, 4, 5}
    # Image 5 is Alpine: /bin/sh, not bash. Getting this wrong changes what 18
    # samples execute without changing anything else.
    assert FS_ENTRYPOINT[5] == "/bin/sh"
    assert {FS_ENTRYPOINT[i] for i in (1, 2, 3, 4)} == {"/bin/bash"}


def test_file_diff_score_collapses_after_one_divergence():
    assert file_diff_score([], [])[0] == PART_CREDIT
    one = file_diff_score([("testbed/a", "??")], [])
    assert one[0] == round(PART_CREDIT * (1 - math.erf(1)), 2) == 0.05
    two = file_diff_score([("testbed/a", "??"), ("testbed/b", "??")], [])
    assert two[0] == 0.0
    # The parts are reported as well as scored: their lengths are the score.
    assert file_diff_score([("a", "??")], [("b", "??")])[1] == [("b", "??")]
    assert file_diff_score([("a", "??")], [("b", "??")])[2] == [("a", "??")]


def test_file_change_score_is_full_credit_when_nothing_is_shared():
    # Upstream's `p2_score = 0.33` default, reached before any hashing -- so a
    # read-only command pair scores this part without touching a file.
    assert file_change_score([], {}, {}) == (PART_CREDIT, 0)


def test_file_change_score_is_the_matching_fraction():
    same = [("testbed/a.txt", "??"), ("testbed/b.txt", "??")]
    agent = {"testbed/a.txt": "aa  a", "testbed/b.txt": "bb  b"}
    matched = file_change_score(same, agent, dict(agent))
    assert matched == (PART_CREDIT, 2)
    half = file_change_score(same, agent, {**agent, "testbed/b.txt": "cc  b"})
    assert half == (round(PART_CREDIT * 0.5, 2), 1)


def test_file_change_score_raises_on_a_path_it_was_not_given():
    # The hash maps come from the same parse that produced `diff_same`, so an
    # absence means the two sides disagreed -- a broken grader, not a wrong
    # answer, and the two must not look alike in a report.
    with pytest.raises(KeyError):
        file_change_score([("testbed/a.txt", "??")], {}, {})


def test_shared_changes_intersects_pairs_and_drops_modified():
    agent = [("a.txt", "??"), ("b.txt", "A"), ("c.txt", "M"), ("d.txt", "??")]
    evaluation = [("a.txt", "??"), ("b.txt", "A"), ("c.txt", "M")]
    shared = shared_changes(agent, evaluation)
    # `M` is absent from upstream's filter even though its comment says "added
    # or modified" -- so a modified file's CONTENT is never compared. Preserved.
    assert sorted(shared) == [("a.txt", "??"), ("b.txt", "A")]
    # A path both sides touched under different codes is not shared at all,
    # because the intersection is over (path, code) pairs.
    assert shared_changes([("e.txt", "A")], [("e.txt", "??")]) == []


def test_parse_status_pairs_tokens():
    assert parse_status(" M testbed/a\n?? testbed/b\n") == [
        ("testbed/a", "M"),
        ("testbed/b", "??"),
    ]
    assert parse_status("") == []


def test_parse_status_desynchronizes_on_a_path_with_a_space():
    # Whitespace splitting with two-at-a-time re-pairing: one extra token
    # shifts every later pair. Two space-bearing paths keep the token count
    # even, so it silently mis-pairs rather than raising -- a status code ends
    # up as a path and vice versa.
    assert parse_status("?? a b\n?? c d") == [
        ("a", "??"),
        ("??", "b"),
        ("d", "c"),
    ]


def test_parse_status_raises_on_an_odd_token_count():
    # ...and with an odd count it walks off the end. So a model command that
    # creates a file whose name contains a space does not score badly upstream,
    # it raises IndexError inside get_reward -- which submit_command swallows
    # into a score of 0 after a print.
    #
    # This port lets it propagate instead (`.claude/rules/tasks.md`: a grading
    # call site catches TimeoutError and nothing else), so the sample lands in
    # `fails` with a recorded reason. Under DENOMINATOR_REQUESTED a fail is
    # already charged as wrong, so the headline agrees with upstream's 0 while
    # the cause stays visible instead of being indistinguishable from a model
    # that answered incorrectly.
    with pytest.raises(IndexError):
        parse_status("?? testbed/two words")


def test_hash_command_reaches_md5deep_for_a_path_without_a_dot():
    assert hash_command("testbed/a.txt") == "md5sum testbed/a.txt"
    # No image installs md5deep, so this branch always fails -- identically on
    # both sides, which is why a directory-only change still compares equal and
    # scores full credit on part 2.
    assert hash_command("testbed/dir") == "md5deep -r testbed/dir"


def test_clean_cmd_wraps_in_double_quotes():
    assert clean_cmd("/bin/bash", "  ls -al  ") == '/bin/bash -c "ls -al"'


def test_command_argv_reproduces_the_shlex_rewrite():
    # docker-py normalizes a string command through shlex.split, so the double
    # quotes clean_cmd added are consumed along with any the command carried.
    assert command_argv("/bin/bash", "ls -al") == ["/bin/bash", "-c", "ls -al"]
    assert command_argv("/bin/bash", 'echo "True"') == [
        "/bin/bash",
        "-c",
        "echo True",
    ]


def test_thirty_seven_golds_are_rewritten_and_index_230_is_truncated():
    table = gold_table()
    rewritten = [
        i
        for i, row in enumerate(table)
        if command_argv("/bin/bash", row["gold"])[2:] != [row["gold"].strip()]
    ]
    # Measured over the pinned table. This count is the size of the divergence a
    # `[entrypoint, "-c", command]` implementation would introduce -- which is
    # why the count, not just the mechanism, is pinned.
    assert len(rewritten) == 37
    assert 230 in rewritten

    argv = command_argv("/bin/bash", table[230]["gold"])
    # The `" "` inside a single-quoted awk program reads as a quote delimiter to
    # shlex, so the script is cut there and the remainder becomes extra argv
    # words -- bash gets a syntactically incomplete program. Its gold "answer"
    # is an awk error, and that is what a model is scored against.
    assert len(argv) == 4
    assert argv[2].endswith("awk '{print $2 ")
    # The space that was inside the quoted region survives into the next word,
    # so what bash sees as $0 is " $1}'" -- pinned exactly, because a port that
    # normalized it would be a different command.
    assert argv[3] == " $1}'"


def test_the_two_golds_that_disagree_with_the_hub():
    table = gold_table()
    # The graded side of the two-source divergence. The Hub's column reads
    # `echo -n 'hello' | base64` (aGVsbG8=) and `awk 'length < 20' ...`; the
    # harness grades against these. Only the vendored side is asserted here --
    # the Hub side needs the network and is covered by the dataset loader.
    assert table[38]["gold"] == "echo 'hello' | base64"
    assert table[100]["gold"] == "awk 'length < 40' setup_nl2b_fs_1.sh"


def test_parse_bash_strips_one_fence_in_pattern_order():
    assert parse_bash("```bash\nls -al\n```") == "ls -al"
    assert parse_bash("```\nls -al\n```") == "ls -al"
    assert parse_bash("`ls -al`") == "ls -al"
    # No fence: the reply is the command, unchanged and unstripped.
    assert parse_bash("ls -al") == "ls -al"
    # A fenced block wins over a later backtick span, because ```bash is tried
    # first -- the order is the behaviour.
    assert parse_bash("```bash\nls\n``` and `pwd`") == "ls"


def test_full_credit_sums_to_exactly_one():
    reward = total_reward(PART_CREDIT, PART_CREDIT, PART_CREDIT)
    # An exact float comparison decides the verdict upstream. It is safe, and
    # this test is what says so: 0.01 + 0.33 * 3 has no representation error.
    assert reward == 1.0
    assert is_correct(reward)


def test_any_shortfall_in_any_part_fails():
    assert not is_correct(total_reward(0.05, PART_CREDIT, PART_CREDIT))
    assert not is_correct(total_reward(PART_CREDIT, PART_CREDIT, 0.0))
    # Even the base alone is not zero, so a total of 0.01 is a complete miss.
    assert total_reward(0.0, 0.0, 0.0) == REWARD_BASE


def test_embedding_inputs_are_truncated_at_a_thousand_characters():
    assert len(truncate_for_embedding("x" * 5000)) == EMBED_TRUNCATION == 1000
    assert truncate_for_embedding("short") == "short"


def test_threshold_is_upstreams_reproduction_setting():
    assert DEFAULT_EMBED_THRESHOLD == 0.75
