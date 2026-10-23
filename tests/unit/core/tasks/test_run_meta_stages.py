"""
Tests for the per-stage code walk and grading identities in ``run_meta``.

Every test builds a throwaway ``sieval_*`` plugin package, so the walk reads
real source files, and edits one function to see which stages change.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import sys
import textwrap

import pytest

from sieval.core.tasks.run_meta import (
    GRADING_STAGES,
    grader_identity,
    grader_key,
    grader_modules,
    grader_modules_v0_10_0,
    grader_task,
    grader_task_v0_10_0,
    grading_identities,
    judge_identities,
    legacy_grader_keys,
    stage_code,
    stage_key,
)
from sieval.core.tasks.services import ServiceIdentity
from tests.conftest import MockJudgeModel

PACKAGE = "sieval_stagewalk_plugin"

TASK = """
import importlib

from sieval.core.tasks.task import Task

from .extract import extract
from .grade import grade
from . import metrics
from .shared import normalize

LIMIT = 3


class Base(Task):
    def __init__(self, dataset, model, name=None, grader=None):
        super().__init__(dataset, model, name)
        self._grader = grader

    async def preprocess(self, raw, ctx):
        return raw

    async def infer(self, pre, ctx):
        return pre


class StageTask(Base):
    async def postprocess(self, inf, ctx):
        return extract(normalize(inf))

    async def feedback(self, post, ctx):
        return True, grade(normalize(post), LIMIT)

    async def report(self, finals, fails):
        return {"score": metrics.mean(finals)}
"""

FILES = {
    "__init__.py": "",
    "task.py": TASK,
    "extract.py": "def extract(text):\n    return text.strip()\n",
    "grade.py": "def grade(answer, limit):\n    return answer[:limit] == 'yes'\n",
    "metrics.py": "def mean(finals):\n    return len(finals)\n",
    "shared.py": "def normalize(text):\n    return text.lower()\n",
}


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
        return importlib.import_module(f"{PACKAGE}.task").StageTask

    yield pkg, load
    drop()


def _edit(pkg, name, old, new):
    path = pkg / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def _changed(before, after):
    return [
        stage
        for stage in GRADING_STAGES
        if before[stage]["modules"] != after[stage]["modules"]
    ]


class TestStageCode:
    def test_each_stage_reaches_its_own_helpers(self, plugin):
        _, load = plugin
        code = stage_code(load())
        keys = {stage: set(code[stage]["modules"]) for stage in GRADING_STAGES}
        assert "./extract.py" in keys["postprocess"]
        assert "./extract.py" not in keys["feedback"] | keys["report"]
        assert "./grade.py" in keys["feedback"]
        assert "./grade.py" not in keys["postprocess"] | keys["report"]
        assert "./metrics.py" in keys["report"]
        assert "./metrics.py" not in keys["postprocess"] | keys["feedback"]
        # The method itself, keyed by its qualname, and the shared state.
        assert "./task.py:StageTask.report" in keys["report"]
        assert all("./task.py:Base.__init__" in k for k in keys.values())
        assert not any("fallback" in code[s] for s in GRADING_STAGES)

    def test_a_report_change_changes_only_report(self, plugin):
        pkg, load = plugin
        before = stage_code(load())
        _edit(pkg, "task.py", '{"score": metrics', '{"total": metrics')
        assert _changed(before, stage_code(load())) == ["report"]

    def test_a_metrics_helper_change_changes_only_report(self, plugin):
        pkg, load = plugin
        before = stage_code(load())
        _edit(pkg, "metrics.py", "len(finals)", "len(finals) * 2")
        assert _changed(before, stage_code(load())) == ["report"]

    def test_a_grader_change_changes_only_feedback(self, plugin):
        pkg, load = plugin
        before = stage_code(load())
        _edit(pkg, "grade.py", "== 'yes'", "== 'no'")
        assert _changed(before, stage_code(load())) == ["feedback"]

    def test_code_two_stages_reach_counts_in_both(self, plugin):
        pkg, load = plugin
        before = stage_code(load())
        _edit(pkg, "shared.py", "lower()", "upper()")
        assert _changed(before, stage_code(load())) == ["postprocess", "feedback"]

    def test_a_module_constant_counts_in_every_stage(self, plugin):
        """The module's shell (imports, constants, signatures) runs at import."""
        pkg, load = plugin
        before = stage_code(load())
        _edit(pkg, "task.py", "LIMIT = 3", "LIMIT = 4")
        assert _changed(before, stage_code(load())) == list(GRADING_STAGES)

    def test_state_the_constructor_sets_counts_in_every_stage(self, plugin):
        pkg, load = plugin
        before = stage_code(load())
        _edit(pkg, "task.py", "self._grader = grader", "self._grader = grader or 1")
        assert _changed(before, stage_code(load())) == list(GRADING_STAGES)

    def test_an_override_counts_with_what_it_overrides(self, plugin):
        """A stage reaches a method by name, so every definition of that name in
        the MRO counts: a `super()` call needs no special case."""
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "    async def infer(self, pre, ctx):\n        return pre\n",
            "    async def infer(self, pre, ctx):\n        return pre\n\n"
            "    async def report(self, finals, fails):\n        return {}\n",
        )
        code = stage_code(load())
        assert "./task.py:Base.report" in code["report"]["modules"]
        assert "./task.py:StageTask.report" in code["report"]["modules"]

    def test_a_method_reached_by_another_counts(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            '        return {"score": self._score(finals)}\n\n'
            "    def _score(self, finals):\n        return metrics.mean(finals)\n",
        )
        code = stage_code(load())
        assert "./task.py:StageTask._score" in code["report"]["modules"]
        assert "./task.py:StageTask._score" not in code["feedback"]["modules"]
        assert "./metrics.py" in code["report"]["modules"]

    def test_a_patched_report_counts_in_report_with_what_it_replaced(
        self, plugin, tmp_path, monkeypatch
    ):
        _, load = plugin
        root = tmp_path / "patch"
        (root / "stagewalk_patch").mkdir(parents=True)
        (root / "stagewalk_patch" / "__init__.py").write_text(
            "async def report(self, finals, fails):\n    return {'patched': 1}\n"
        )
        monkeypatch.syspath_prepend(str(root))
        task_cls = load()
        try:
            patch = importlib.import_module("stagewalk_patch")
            task_cls.report = patch.report
            code = stage_code(task_cls)
        finally:
            sys.modules.pop("stagewalk_patch", None)
        report = code["report"]["modules"]
        assert "stagewalk_patch/__init__.py:report" in report
        assert "./task.py:StageTask.report" in report
        assert "stagewalk_patch/__init__.py:report" not in code["feedback"]["modules"]

    def test_a_computed_getattr_on_the_task_falls_back_from_that_stage_on(self, plugin):
        """Unsound to split: that stage and every one after it count all the
        grading code, never less."""
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "        return True, grade(normalize(post), LIMIT)\n",
            "        how = getattr(self, 'mode_' + post)\n"
            "        return True, grade(normalize(how()), LIMIT)\n",
        )
        task_cls = load()
        code = stage_code(task_cls)
        whole = grader_modules(task_cls)
        assert "fallback" not in code["postprocess"]
        for stage in ("feedback", "report"):
            assert code[stage]["modules"] == whole
            assert "getattr(self, <expr>)" in code[stage]["fallback"]

    def test_importlib_falls_back(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            "        mod = importlib.import_module('x')\n        return {}\n",
        )
        code = stage_code(load())
        assert "import_module" in code["report"]["fallback"]
        assert "fallback" not in code["feedback"]

    def test_a_judge_counts_where_its_attribute_is_read(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "        return extract(normalize(inf))\n",
            "        return extract(normalize(self._grader(inf)))\n",
        )
        code = stage_code(load(), ["grader"])
        assert code["postprocess"]["judges"] == ["grader"]
        assert code["report"]["judges"] == []


class TestGradingIdentities:
    def _identities(self, task_cls, judges=()):
        return grading_identities(
            stage_code(task_cls, [j["role"] for j in judges]),
            list(judges),
            [{"name": "code-eval", "source": "external"}],
            {"k": 1},
            grader_task(task_cls),
        )

    def test_feedback_holds_judges_and_services(self, plugin):
        _, load = plugin
        judges = judge_identities({"grader": MockJudgeModel()}, {})
        ids = self._identities(load(), judges)
        assert ids["feedback"]["judges"] == judges
        assert ids["feedback"]["services"] == [
            {"name": "code-eval", "source": "external"}
        ]
        assert "services" not in ids["report"]
        assert "judges" not in ids["report"]
        assert all(ids[s]["args"] == {"k": 1} for s in GRADING_STAGES)
        assert all(ids[s]["task"].endswith(":StageTask") for s in GRADING_STAGES)

    def test_a_reported_judge_version_is_not_a_change(self, plugin):
        _, load = plugin
        task_cls = load()
        unreported = judge_identities({"grader": MockJudgeModel()}, {})
        reported = judge_identities(
            {"grader": MockJudgeModel()}, {}, {"grader": "judge-2026"}
        )
        before = self._identities(task_cls, unreported)["feedback"]
        after = self._identities(task_cls, reported)["feedback"]
        assert before["digest"] != after["digest"]
        assert stage_key(before) == stage_key(after)

    def test_stage_key_is_none_for_unknown(self):
        assert stage_key(None) is None
        assert stage_key({"modules": "x"}) is None
        assert stage_key({"modules": {}, "judges": "x"}) is None

    def test_a_fresh_identity_is_deterministic(self, plugin):
        _, load = plugin
        assert self._identities(load()) == self._identities(load())


class TestLegacyGraderKeys:
    def test_both_old_walks_match_their_own_blocks(self, plugin):
        _, load = plugin
        task_cls = load()
        keys = legacy_grader_keys(task_cls, [], [], {"k": 1})
        by_path = grader_identity(
            grader_modules(task_cls), [], args={"k": 1}, task=grader_task(task_cls)
        )
        by_name = grader_identity(
            grader_modules_v0_10_0(task_cls),
            [],
            args={"k": 1},
            task=grader_task_v0_10_0(task_cls),
        )
        assert keys == {grader_key(by_path), grader_key(by_name)}
        assert f"{PACKAGE}.task" in by_name["modules"]
        assert "./task.py" in by_path["modules"]

    def test_a_changed_report_no_longer_matches(self, plugin):
        pkg, load = plugin
        recorded = legacy_grader_keys(load(), [], [], {})
        _edit(pkg, "metrics.py", "len(finals)", "len(finals) * 2")
        assert recorded.isdisjoint(legacy_grader_keys(load(), [], [], {}))


class TestJudgesAndServices:
    def test_a_judge_stored_under_another_name_counts_where_it_is_read(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "        self._grader = grader\n",
            "        self._grader = grader\n        self._client = self._grader\n",
        )
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            '        return {"score": self._client}\n',
        )
        code = stage_code(load(), ["grader"])
        assert code["report"]["judges"] == ["grader"]
        # `__init__` reads it too, and every stage reaches `__init__`: counted
        # in each, as shared state is.
        assert code["postprocess"]["judges"] == ["grader"]

    def test_a_judge_only_one_stage_reads_counts_only_there(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "        self._grader = grader\n",
            "        self._judge = grader\n",
        )
        _edit(
            pkg,
            "task.py",
            "        return True, grade(normalize(post), LIMIT)\n",
            "        return True, grade(normalize(self._verdict(post)), LIMIT)\n\n"
            "    def _verdict(self, post):\n        return self.judge_call(post)\n",
        )
        code = stage_code(load(), ["judge_call"])
        assert code["feedback"]["judges"] == ["judge_call"]
        assert code["report"]["judges"] == []

    def test_a_task_overriding_judge_models_reads_every_judge_everywhere(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "    async def infer(self, pre, ctx):\n        return pre\n",
            "    async def infer(self, pre, ctx):\n        return pre\n\n"
            "    def judge_models(self):\n        return {}\n",
        )
        code = stage_code(load(), ["grader"])
        assert all(code[s]["judges"] == ["grader"] for s in GRADING_STAGES)

    def test_a_service_counts_where_its_endpoint_is_read(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "        self._grader = grader\n",
            "        self._grader = grader\n"
            "        self._api = os.getenv('SIEVAL_CODE_EVAL_API')\n",
        )
        _edit(pkg, "task.py", "import importlib\n", "import importlib\nimport os\n")
        _edit(
            pkg,
            "task.py",
            "        return True, grade(normalize(post), LIMIT)\n",
            "        return True, grade(normalize(self._api), LIMIT)\n",
        )
        task_cls = load()
        code = stage_code(task_cls, [], ["code-eval"])
        assert code["feedback"]["services"] is True
        assert code["postprocess"]["services"] is False
        services: list[ServiceIdentity] = [{"name": "code-eval", "source": "external"}]
        ids = grading_identities(code, [], services, {}, grader_task(task_cls))
        assert ids["feedback"]["services"] == services
        assert "services" not in ids["postprocess"]

    def test_feedback_holds_a_judge_its_code_does_not_name(self, plugin):
        _, load = plugin
        task_cls = load()
        judges = judge_identities({"hidden": MockJudgeModel()}, {})
        code = stage_code(task_cls, ["hidden"])
        assert all(code[s]["judges"] == [] for s in GRADING_STAGES)
        ids = grading_identities(code, judges, [], {}, grader_task(task_cls))
        assert ids["feedback"]["judges"] == judges
        assert "judges" not in ids["report"]


class TestWalkConstructs:
    def test_a_helper_class_counts_with_all_its_methods(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "LIMIT = 3\n",
            "LIMIT = 3\n\n\nclass Scorer:\n    def score(self, x):\n"
            "        return x\n\n    def other(self):\n        return 0\n",
        )
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            '        return {"score": Scorer().score(finals)}\n',
        )
        code = stage_code(load())
        assert "./task.py:Scorer.score" in code["report"]["modules"]
        assert "./task.py:Scorer.other" in code["report"]["modules"]
        assert "./task.py:Scorer.score" not in code["feedback"]["modules"]

    def test_a_star_import_counts_in_every_stage(self, plugin):
        pkg, load = plugin
        _edit(
            pkg, "task.py", "LIMIT = 3\n", "from .grade import *  # noqa\nLIMIT = 3\n"
        )
        before = stage_code(load())
        _edit(pkg, "grade.py", "== 'yes'", "== 'no'")
        assert set(_changed(before, stage_code(load()))) == set(GRADING_STAGES)

    def test_an_aliased_import_is_followed(self, plugin):
        pkg, load = plugin
        _edit(pkg, "task.py", "from . import metrics\n", "from . import metrics as m\n")
        _edit(pkg, "task.py", "metrics.mean(finals)", "m.mean(finals)")
        before = stage_code(load())
        _edit(pkg, "metrics.py", "len(finals)", "len(finals) + 1")
        assert _changed(before, stage_code(load())) == ["report"]

    def test_a_helper_importing_back_from_the_task_module(self, plugin):
        """A whole module that imports a function of a split module reaches
        that function, so a change to it reruns that stage."""
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "LIMIT = 3\n",
            "LIMIT = 3\n\n\ndef weight():\n    return 1\n",
        )
        (pkg / "metrics.py").write_text(
            "def mean(finals):\n"
            "    from .task import weight\n"
            "    return len(finals) * weight()\n"
        )
        before = stage_code(load())
        _edit(pkg, "task.py", "    return 1\n", "    return 2\n")
        assert "report" in _changed(before, stage_code(load()))

    def test_a_plugin_helper_loading_modules_by_name_falls_back(self, plugin):
        pkg, load = plugin
        (pkg / "metrics.py").write_text(
            "import importlib\n\n"
            "def mean(finals):\n"
            "    return importlib.import_module('x').f(finals)\n"
        )
        code = stage_code(load())
        assert "loads modules by name" in code["report"]["fallback"]
        assert "fallback" not in code["postprocess"]

    def test_a_constant_getattr_is_a_plain_reference(self, plugin):
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            "        return {'score': getattr(self, '_score')(finals)}\n\n"
            "    def _score(self, finals):\n        return metrics.mean(finals)\n",
        )
        code = stage_code(load())
        assert "fallback" not in code["report"]
        assert "./task.py:StageTask._score" in code["report"]["modules"]

    def test_getattr_on_a_module_outside_the_walk_is_allowed(self, plugin):
        pkg, load = plugin
        _edit(pkg, "task.py", "import importlib\n", "import importlib\nimport math\n")
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            "        return {'score': getattr(math, 'f' + 'loor')(1.5)}\n",
        )
        assert "fallback" not in stage_code(load())["report"]

    def test_a_method_without_source_falls_back_everywhere(self, plugin):
        _, load = plugin
        task_cls = load()
        task_cls.report = eval("lambda self, finals, fails: {}")  # noqa: S307
        task_cls.report.__module__ = task_cls.__module__
        code = stage_code(task_cls)
        whole = grader_modules(task_cls)
        for stage in GRADING_STAGES:
            assert code[stage]["modules"] == whole
            assert "source" in code[stage]["fallback"]

    def test_an_absolute_import_of_the_own_package_is_walked(self, plugin):
        """The own package's name is left out of the digest, not out of the
        parse: ``from <package>.x`` is not Python."""
        pkg, load = plugin
        _edit(
            pkg,
            "task.py",
            "from .extract import extract\n",
            f"from {PACKAGE}.extract import extract\n",
        )
        _edit(pkg, "task.py", "from . import metrics\n", f"import {PACKAGE}.metrics\n")
        _edit(pkg, "task.py", "metrics.mean(finals)", f"{PACKAGE}.metrics.mean(finals)")
        before = stage_code(load())
        assert not any("fallback" in before[s] for s in GRADING_STAGES)
        _edit(pkg, "extract.py", "text.strip()", "text.strip().lower()")
        assert _changed(before, stage_code(load())) == ["postprocess"]

    def test_a_re_exported_function_counts_where_it_is_called(self, plugin):
        """A module holding a class of the MRO is read a function at a time;
        a name it only imports is followed to where it is defined."""
        pkg, load = plugin
        (pkg / "base.py").write_text(
            "from .extract import extract  # noqa: F401\n\n\nclass Mixin:\n    pass\n"
        )
        _edit(
            pkg,
            "task.py",
            "from .extract import extract\n",
            "from .base import Mixin, extract\n",
        )
        _edit(pkg, "task.py", "class StageTask(Base):", "class StageTask(Base, Mixin):")
        before = stage_code(load())
        _edit(pkg, "extract.py", "text.strip()", "text.strip().lower()")
        assert _changed(before, stage_code(load())) == ["postprocess"]

    def test_a_name_imported_in_two_methods_resolves_per_method(self, plugin):
        pkg, load = plugin
        _edit(pkg, "task.py", "from .extract import extract\n", "")
        _edit(pkg, "task.py", "from . import metrics\n", "")
        _edit(
            pkg,
            "task.py",
            "        return extract(normalize(inf))\n",
            "        from .extract import extract as run\n"
            "        return run(normalize(inf))\n",
        )
        _edit(
            pkg,
            "task.py",
            '        return {"score": metrics.mean(finals)}\n',
            "        from .metrics import mean as run\n"
            '        return {"score": run(finals)}\n',
        )
        before = stage_code(load())
        _edit(pkg, "metrics.py", "len(finals)", "len(finals) * 2")
        assert _changed(before, stage_code(load())) == ["report"]
