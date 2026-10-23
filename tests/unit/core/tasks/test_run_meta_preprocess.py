"""
Tests for the code-based ``preprocess`` identity in ``run_meta``.

Each test writes a throwaway ``sieval_*`` plugin package, so the walk reads
real source files, and edits it to see whether ``preprocess`` changes.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import sys
import textwrap

import pytest

from sieval.core.tasks.run_meta import (
    is_code_identity,
    preprocess_identity,
    preprocess_key,
    split_digests,
    stage_code,
)

PACKAGE = "sieval_preprocess_plugin"

TASK = """
from sieval.core.tasks.task import Task

from .render import render

HEADER = "Classify the sequence."


class SeqTask(Task):
    def __init__(self, dataset, model, name=None, label="promoter", k=1):
        super().__init__(dataset, model, name)
        self._label = label
        self._k = k

    async def preprocess(self, raw, ctx):
        return render(HEADER, self._label, raw["sequence"])

    async def infer(self, pre, ctx):
        return pre

    async def postprocess(self, inf, ctx):
        return inf.strip()

    async def feedback(self, post, ctx):
        return True, post == "yes"

    async def report(self, finals, fails):
        return {"k": self._k}
"""

FILES = {
    "__init__.py": "",
    "task.py": TASK,
    "render.py": (
        "def render(header, label, seq):\n"
        "    return f'{header} Is it a {label}?\\n{seq}'\n"
    ),
}

ARGS = {"label": "promoter", "k": 1}


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Write the plugin package; ``load()`` imports it afresh, after edits."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    root = tmp_path / "plugins"
    pkg = root / PACKAGE
    pkg.mkdir(parents=True)
    for name, text in FILES.items():
        (pkg / name).write_text(textwrap.dedent(text))
    monkeypatch.syspath_prepend(str(root))

    def drop():
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + "."):
                sys.modules.pop(name, None)

    def load():
        drop()
        return importlib.import_module(f"{PACKAGE}.task").SeqTask

    yield pkg, load
    drop()


def _edit(pkg, name, old, new):
    path = pkg / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def _identity(load, args=ARGS):
    code = stage_code(load(), args=list(args))["preprocess"]
    return preprocess_identity(code, args)


class TestCodeChanges:
    def test_preprocess_holds_its_code_and_the_args_it_reads(self, plugin):
        _, load = plugin
        identity = _identity(load)
        assert is_code_identity(identity)
        assert "./task.py:SeqTask.preprocess" in identity["modules"]
        assert "./render.py" in identity["modules"]
        # `k` is read only by `report`.
        assert identity["args"] == {"label": "promoter"}
        assert "fallback" not in identity

    def test_an_edit_to_preprocess_code_changes_it(self, plugin):
        pkg, load = plugin
        before = _identity(load)
        _edit(pkg, "task.py", 'raw["sequence"])', 'raw["sequence"].upper())')
        assert preprocess_key(_identity(load)) != preprocess_key(before)

    def test_an_edit_to_a_helper_it_calls_changes_it(self, plugin):
        pkg, load = plugin
        before = _identity(load)
        _edit(pkg, "render.py", "Is it a", "Is this a")
        assert preprocess_key(_identity(load)) != preprocess_key(before)

    def test_a_module_constant_changes_it(self, plugin):
        pkg, load = plugin
        before = _identity(load)
        _edit(pkg, "task.py", "Classify the sequence.", "Classify it.")
        assert preprocess_key(_identity(load)) != preprocess_key(before)

    def test_an_edit_to_an_unrelated_method_does_not(self, plugin):
        pkg, load = plugin
        before = _identity(load)
        _edit(pkg, "task.py", 'return {"k": self._k}', 'return {"k": self._k + 1}')
        _edit(pkg, "task.py", 'post == "yes"', 'post == "no"')
        assert _identity(load) == before

    def test_an_arg_preprocess_reads_changes_it(self, plugin):
        _, load = plugin
        before = _identity(load)
        after = _identity(load, {**ARGS, "label": "enhancer"})
        assert preprocess_key(after) != preprocess_key(before)

    def test_an_arg_preprocess_does_not_read_leaves_it(self, plugin):
        _, load = plugin
        assert _identity(load, {**ARGS, "k": 5}) == _identity(load)

    def test_an_arg_kept_under_another_name_counts_with_an_override(self, plugin):
        """A ``grader_args`` override may return ``label`` while ``preprocess``
        reads ``self._kind``: every argument it returns counts."""
        pkg, load = plugin
        _edit(pkg, "task.py", "self._label = label", "self._kind = label")
        _edit(pkg, "task.py", "self._label, raw", "self._kind, raw")
        _edit(
            pkg,
            "task.py",
            "    async def infer(",
            "    def grader_args(self):\n"
            '        return {"label": self._kind, "k": self._k}\n\n'
            "    async def infer(",
        )
        identity = _identity(load)
        assert identity["args"] == ARGS
        after = _identity(load, {**ARGS, "label": "enhancer"})
        assert preprocess_key(after) != preprocess_key(identity)

    def test_the_key_ignores_why_it_fell_back(self, plugin):
        _, load = plugin
        identity = _identity(load)
        explained = {**identity, "fallback": "a reason"}
        assert preprocess_key(explained) == preprocess_key(identity)


class TestDynamicCode:
    def test_a_file_read_at_run_time_counts_the_package_and_its_data(self, plugin):
        """A template opened by path is a file no import names: the stage
        counts all of the task's code and the data files beside it."""
        pkg, load = plugin
        (pkg / "template.txt").write_text("Is it a {label}?")
        _edit(
            pkg,
            "task.py",
            '        return render(HEADER, self._label, raw["sequence"])\n',
            "        here = __file__.rpartition('/')[0]\n"
            "        with open(here + '/template.txt') as f:\n"
            "            template = f.read()\n"
            '        return template.format(label=self._label) + raw["sequence"]\n',
        )
        identity = _identity(load)
        assert "read a file at run time" in identity["fallback"]
        assert "./template.txt" in identity["modules"]
        # Under the fallback every argument counts.
        assert identity["args"] == ARGS

        before = preprocess_key(identity)
        (pkg / "template.txt").write_text("Is this a {label}?")
        assert preprocess_key(_identity(load)) != before

    def test_a_markdown_template_is_data_but_agent_notes_are_not(self, plugin):
        pkg, load = plugin
        (pkg / "prompt.md").write_text("Is it a {label}?")
        (pkg / "CLAUDE.md").write_text("notes")
        _edit(
            pkg,
            "task.py",
            '        return render(HEADER, self._label, raw["sequence"])\n',
            "        here = __file__.rpartition('/')[0]\n"
            "        with open(here + '/prompt.md') as f:\n"
            '            return f.read() + raw["sequence"]\n',
        )
        modules = _identity(load)["modules"]
        assert "./prompt.md" in modules
        assert "./CLAUDE.md" not in modules

    def test_a_run_time_name_falls_back(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            '        return render(HEADER, self._label, raw["sequence"])\n',
            '        return getattr(self, "_" + "render")(raw)\n',
        )
        identity = _identity(load)
        assert "run time" in identity["fallback"]
        assert identity["args"] == ARGS

    def test_a_dataset_split_read_counts_every_split(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            '        return render(HEADER, self._label, raw["sequence"])\n',
            "        shots = self.dataset.dataset_dict['train']\n"
            '        return str(list(shots)) + raw["sequence"]\n',
        )
        code = stage_code(load(), args=list(ARGS))["preprocess"]
        assert code["splits"] is True
        one = preprocess_identity(code, ARGS, split_digests({"train": [{"s": "A"}]}))
        two = preprocess_identity(code, ARGS, split_digests({"train": [{"s": "C"}]}))
        assert preprocess_key(one) != preprocess_key(two)


def test_a_recorded_digest_of_prompts_is_not_a_code_identity():
    assert not is_code_identity({"digest": "sha256:p", "samples": 3})
    assert not is_code_identity(None)
    assert preprocess_key({"digest": "sha256:p", "samples": 3}) is None


def test_split_digests_is_unknown_for_an_undigestable_row():
    assert split_digests({"train": [object()]}) is None
