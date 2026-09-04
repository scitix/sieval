"""Tests for the NL2SH shell backend, run against a temporary git tree.

No container: ``NL2SH_FS_ROOT`` points the protocol at a throwaway repo built to
the same shape as upstream's images -- a git repo at the root, a ``.gitignore``
excluding the FHS directories, and a prepared subtree committed as the baseline.
That covers everything the backend decides (reset, status, hashing, the quoting
rewrite, the timeout, the fs_id refusal); what it cannot cover is what upstream's
own five images print, which needs Docker.

These live with the vendored service rather than under sieval's ``tests/``,
which mirrors ``sieval/``. Run them from ``vendor/code-evaluator``:

    python -m pytest tests/ -q

They belong upstream in ``scitix/code-evaluator`` along with ``app/exec_sh.py``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.exec_sh import (  # noqa: E402
    HASHED_STATUS_CODES,
    TIMEOUT_OBSERVATION,
    _argv,
    execute_shell,
    fs_root,
    hosted_fs_id,
)

# Upstream's docker.gitignore: every standard root directory, so `git status`
# at the root reports only the prepared tree.
_GITIGNORE = "\n".join(
    [
        "bin",
        "boot",
        "dev",
        "etc",
        "home",
        "lib",
        "media",
        "opt",
        "proc",
        "root",
        "run",
        "sbin",
        "srv",
        "sys",
        "usr",
        "var",
        ".dockerenv",
    ]
)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway stand-in for an upstream image's baseline tree."""
    root = tmp_path / "fsroot"
    (root / "workspace" / "dir1").mkdir(parents=True)
    (root / "workspace" / "textfile1.txt").write_text("Hello, World!\n")
    (root / "workspace" / "dir1" / "textfile2.txt").write_text("second\n")
    (root / ".gitignore").write_text(_GITIGNORE + "\n")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "intercode",
        "GIT_AUTHOR_EMAIL": "intercode@pnlp.org",
        "GIT_COMMITTER_NAME": "intercode",
        "GIT_COMMITTER_EMAIL": "intercode@pnlp.org",
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial commit"], cwd=root, check=True, env=env
    )
    monkeypatch.setenv("NL2SH_FS_ROOT", str(root))
    monkeypatch.setenv("NL2SH_FS_ID", "3")
    return root


def _run(command: str, gold: str, **kwargs):
    """Drive one evaluation on a fresh event loop.

    ``asyncio.run`` per call rather than an async test plugin: this suite has to
    stay runnable in the vendored service's own environment, which declares no
    test dependencies beyond pytest.
    """
    ok, msg, data = asyncio.run(
        execute_shell(fs_id=3, command=command, gold=gold, **kwargs)
    )
    assert ok, msg
    assert data is not None
    return data


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_hosted_fs_id_reads_the_environment(monkeypatch):
    monkeypatch.delenv("NL2SH_FS_ID", raising=False)
    assert hosted_fs_id() is None
    monkeypatch.setenv("NL2SH_FS_ID", "5")
    assert hosted_fs_id() == 5
    # Neither a non-integer nor an out-of-range id may fall back to a default:
    # serving filesystem 1 for a request that asked for 9 is the silent-zero
    # failure this route exists to refuse.
    monkeypatch.setenv("NL2SH_FS_ID", "banana")
    assert hosted_fs_id() is None
    monkeypatch.setenv("NL2SH_FS_ID", "9")
    assert hosted_fs_id() is None


def test_a_request_for_another_filesystem_is_refused(tree):
    ok, msg, data = asyncio.run(execute_shell(fs_id=1, command="ls", gold="ls"))
    assert not ok
    assert "fs_id mismatch" in msg
    assert data is None


def test_an_unconfigured_instance_refuses_the_route(tree, monkeypatch):
    monkeypatch.delenv("NL2SH_FS_ID")
    ok, msg, data = asyncio.run(execute_shell(fs_id=3, command="ls", gold="ls"))
    assert not ok
    assert "hosts no NL2SH filesystem" in msg
    assert data is None


def test_fs_root_defaults_to_the_image_root(monkeypatch):
    monkeypatch.delenv("NL2SH_FS_ROOT", raising=False)
    assert fs_root() == "/"


# --------------------------------------------------------------------------- #
# The quoting rewrite
# --------------------------------------------------------------------------- #
def test_argv_reproduces_docker_pys_shlex_split():
    assert _argv("/bin/bash", "ls -al") == ["/bin/bash", "-c", "ls -al"]
    # The quotes a command carried are consumed along with the ones clean_cmd
    # added. This is upstream's behaviour and 37 of its 300 golds depend on it.
    assert _argv("/bin/bash", 'echo "hi"') == ["/bin/bash", "-c", "echo hi"]


def test_the_rewrite_reaches_the_shell(tree):
    # A single-quoted awk program containing a double-quoted space is cut at
    # that space, so the shell receives an unterminated program -- upstream's
    # gold at index 230 is exactly this shape, and its "answer" is the error.
    data = _run("""awk '{print $1 " " $2}' workspace/textfile1.txt""", "true")
    assert data["model_exit_ok"] is False
    assert data["model_output"] != ""


# --------------------------------------------------------------------------- #
# Outputs, status and hashes
# --------------------------------------------------------------------------- #
def test_both_commands_run_and_their_outputs_come_back(tree):
    data = _run("cat workspace/textfile1.txt", "echo 'Hello, World!'")
    assert data["model_output"] == "Hello, World!\n"
    assert data["gold_output"] == "Hello, World!\n"
    assert data["model_exit_ok"] and data["gold_exit_ok"]
    # A read-only pair changes nothing, so both status listings are empty and
    # part 2 will take its full-credit default.
    assert data["model_status"] == data["gold_status"] == ""
    assert data["model_hashes"] == data["gold_hashes"] == {}


def test_stderr_is_merged_into_the_output(tree):
    # docker's exec_run attaches both streams and upstream decodes the one it
    # gets, so a command's error text IS its output for comparison purposes.
    data = _run("cat workspace/nope.txt", "true")
    assert "No such file" in data["model_output"]
    assert data["model_exit_ok"] is False


def test_a_created_file_is_reported_and_hashed(tree):
    data = _run("echo new > workspace/created.txt", "true")
    assert data["model_status"].split() == ["??", "workspace/created.txt"]
    # Raw `md5sum` stdout, not a digest: upstream compares these strings.
    assert data["model_hashes"]["workspace/created.txt"].endswith(
        "  workspace/created.txt\n"
    )
    assert data["gold_status"] == ""


def test_a_directory_change_hashes_through_the_missing_md5deep(tree):
    # git reports the directory, whose path carries no dot, so upstream's
    # get_hash_cmd picks `md5deep -r` -- a binary no image installs.
    made = "mkdir workspace/newdir && touch workspace/newdir/x"
    data = _run(made, made)
    (path,) = data["model_hashes"]
    assert path == "workspace/newdir/"
    # The failure text differs from docker's ("executable file not found in
    # $PATH") because the transport differs, and that is not what the metric
    # reads: upstream compares the two sides' strings to each other. Identical
    # on both sides is the property, so a directory-only change scores full
    # credit here exactly as it does upstream.
    assert data["model_hashes"][path] == data["gold_hashes"][path]
    assert "md5deep" in data["model_hashes"][path]


def test_only_added_untracked_copied_paths_are_hashed(tree):
    # A modification is reported by git but never hashed: `M` is absent from
    # upstream's filter, so a modified file's content is not compared.
    data = _run("echo changed > workspace/textfile1.txt", "true")
    assert data["model_status"].split() == ["M", "workspace/textfile1.txt"]
    assert data["model_hashes"] == {}
    assert "M" not in HASHED_STATUS_CODES


# --------------------------------------------------------------------------- #
# Isolation between the two commands
# --------------------------------------------------------------------------- #
def test_the_gold_does_not_see_the_models_changes(tree):
    # The property upstream buys with a second container: the tree is restored
    # before each command, so a model that deletes the world cannot change what
    # the gold prints.
    data = _run("rm -rf workspace", "cat workspace/textfile1.txt")
    assert data["gold_output"] == "Hello, World!\n"
    assert data["gold_exit_ok"]


def test_the_tree_is_restored_after_a_destructive_command(tree):
    _run("rm -rf workspace && echo litter > litter.txt", "true")
    # And restored again on the way out, so the next sample starts clean.
    assert (tree / "workspace" / "textfile1.txt").read_text() == "Hello, World!\n"
    assert not (tree / "litter.txt").exists()


def test_consecutive_samples_do_not_leak_into_each_other(tree):
    first = _run("echo a > workspace/a.txt", "true")
    second = _run("true", "true")
    assert first["model_status"] != ""
    assert second["model_status"] == ""


# --------------------------------------------------------------------------- #
# The wall
# --------------------------------------------------------------------------- #
def test_a_hanging_model_command_hits_the_wall(tree):
    data = _run("sleep 30", "true", timeout=1.0)
    assert data["model_timed_out"] is True
    # The exact observation string upstream reports, because it is compared as
    # an output rather than inspected.
    assert data["model_output"] == TIMEOUT_OBSERVATION
    assert data["model_exit_ok"] is False
    assert data["gold_timed_out"] is False


def test_a_hanging_gold_is_reported_rather_than_hidden(tree):
    # Bounding the gold is this service's divergence from upstream, so it has
    # to be visible: the caller publishes n_gold_timeouts to show it never bound.
    data = _run("true", "sleep 30", timeout=1.0)
    assert data["gold_timed_out"] is True
    assert data["gold_output"] == TIMEOUT_OBSERVATION


def test_a_timeout_kills_the_process_group(tree):
    # Upstream abandons the command inside its container; a survivor here would
    # keep writing to the tree the next sample is scored against.
    data = _run(
        "sh -c 'sleep 30 && echo late > workspace/late.txt' & sleep 30",
        "true",
        timeout=1.0,
    )
    assert data["model_timed_out"] is True
    assert not (tree / "workspace" / "late.txt").exists()
