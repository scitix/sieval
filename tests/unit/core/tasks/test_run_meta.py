"""
Tests for the stage identities and run history recorded in ``meta.json``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from sieval.core.models import (
    Deployment,
    DeploymentPlanProjection,
    Engine,
    ServingFacts,
)
from sieval.core.models.capabilities import RequestDefaults
from sieval.core.tasks.run_meta import (
    ABSENT,
    GRADER_PACKAGES,
    GRADING_STAGES,
    IDENTITY_SCHEMA,
    JUDGE_MODEL_REF_ENV,
    STAGES,
    JudgeIdentity,
    _file_key,
    _without_own_name,
    aggregate_digest,
    call_key,
    canonical_json,
    carried_identity,
    carried_runs,
    digest,
    empty_identity,
    grader_identity,
    grader_key,
    grader_modules,
    grader_task,
    inference_identity,
    inference_matches,
    is_code_identity,
    judge_identities,
    judge_model_id,
    judge_model_refs,
    legacy_inference_digest,
    model_request_args,
    new_run_record,
    ordered_stages,
    prompt_of,
    read_identity,
    request_definition,
    request_digest,
    serving_fingerprint,
    validate_judge_model_ref,
    warn_unmatched_judge_refs,
)
from tests.conftest import MockChatModel, MockJudgeModel


# ===================================================================
# Canonical digests
# ===================================================================
class TestDigest:
    def test_is_sha256_over_sorted_compact_json(self):
        value = {"b": [1, 2], "a": "x"}
        expected = hashlib.sha256(b'{"a":"x","b":[1,2]}').hexdigest()
        assert digest(value) == f"sha256:{expected}"

    def test_key_order_does_not_matter(self):
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_different_content_differs(self):
        assert digest({"a": 1}) != digest({"a": 2})

    def test_encodes_values_a_raw_sample_may_hold(self):
        raw = {
            "ids": (1, 2),
            "tags": {"y", "x"},
            "blob": b"\x00\x01",
            "score": np.float32(0.5),
            "vec": np.array([1, 2]),
            "when": datetime(2026, 1, 1, tzinfo=UTC),
        }
        decoded = json.loads(canonical_json(raw))
        assert decoded["ids"] == [1, 2]
        assert decoded["tags"] == ['"x"', '"y"']
        assert decoded["blob"] == {"__bytes__": hashlib.sha256(b"\x00\x01").hexdigest()}
        assert decoded["score"] == 0.5
        assert decoded["vec"] == [1, 2]
        assert decoded["when"].startswith("2026-01-01")

    def test_an_image_is_digested_by_its_bytes(self):
        class _Image:
            def __init__(self, data):
                self._data = data

            def tobytes(self):
                return self._data

        assert digest({"img": _Image(b"a")}) != digest({"img": _Image(b"b")})

    def test_undigestable_value_raises(self):
        with pytest.raises(TypeError, match="cannot digest"):
            digest({"f": object()})

    def test_a_missing_float_is_digested_and_tells_samples_apart(self):
        """NaN is encoded, not refused: a refused sample could not tell two
        datasets apart."""
        a = digest({"q": "What is 2+2?", "score": float("nan")})
        b = digest({"q": "Totally different", "score": float("nan")})
        assert a != b
        assert digest({"x": np.float64("nan")}) == digest({"x": float("nan")})

    def test_nan_is_not_the_string_nan(self):
        assert digest({"x": float("nan")}) != digest({"x": "NaN"})
        assert digest({"x": float("inf")}) != digest({"x": float("-inf")})


class TestPromptOf:
    def test_prompt_record_contributes_only_its_prompt(self):
        a = {"prompt": [{"role": "user", "content": "q"}], "reference": "1"}
        b = {"prompt": [{"role": "user", "content": "q"}], "reference": "2"}
        assert prompt_of(a) == prompt_of(b) == [{"role": "user", "content": "q"}]

    def test_bare_prompt_is_used_whole(self):
        assert prompt_of("What is 1+1?") == "What is 1+1?"


class TestAggregateDigest:
    def test_counts_samples_and_ignores_arrival_order(self):
        rows = [(2, None, "sha256:c"), (0, None, "sha256:a"), ("x", None, "sha256:b")]
        forward = aggregate_digest(rows)
        backward = aggregate_digest(reversed(rows))
        assert forward == backward
        assert forward["samples"] == 3

    def test_repeat_index_is_part_of_the_key(self):
        """A split repeated four times is other data than the same split once."""
        once = aggregate_digest([(0, None, "sha256:a")])
        repeated = aggregate_digest([(0, 0, "sha256:a")])
        assert once["digest"] != repeated["digest"]

    def test_an_unknown_sample_makes_the_stage_unknown(self):
        """Two runs whose only difference is an undigestable sample must not
        share a digest, so the stage claims none."""
        stage = aggregate_digest([(0, None, "sha256:a"), (1, None, None)])
        assert stage == {"digest": None, "samples": 2, "unknown": 1}

    def test_absent_is_a_fact_and_keeps_a_digest(self):
        """A failed preprocess has no prompt; that is known, so it digests."""
        absent = aggregate_digest([(0, None, ABSENT)])
        present = aggregate_digest([(0, None, "sha256:a")])
        assert absent["digest"] is not None
        assert absent["digest"] != present["digest"]
        assert "unknown" not in absent


class TestCallKey:
    def test_endpoint_and_provenance_are_left_out(self):
        base = {"model": "m", "api_base": "http://a/v1", "default_params": {}}
        moved = base | {"api_base": "http://b/v1", "provenance": {"engine_id": "x"}}
        assert call_key(base) == call_key(moved)

    def test_sampling_tells_two_configurations_of_one_name_apart(self):
        a = {"model": "m", "api_base": None, "default_params": {"temperature": 0}}
        b = {"model": "m", "api_base": None, "default_params": {"temperature": 1}}
        assert call_key(a) != call_key(b)


class TestRequestDigest:
    CANDIDATE = {"model": "m", "api_base": None, "default_params": {}}
    JUDGE = {"model": "j", "api_base": None, "default_params": {}}

    def _call(self, model, **params):
        return {"model": model, "request_params": params}

    def test_a_call_time_argument_counts(self):
        """`stop` passed by a task at call time is not in the model's config,
        and is still what was sent."""
        key = call_key(self.CANDIDATE)
        plain = request_digest([self._call(self.CANDIDATE, n=1)], key)
        stopped = request_digest([self._call(self.CANDIDATE, n=1, stop=["Q:"])], key)
        assert plain is not None
        assert plain != stopped

    def test_a_judge_call_in_the_same_stage_is_left_out(self):
        key = call_key(self.CANDIDATE)
        alone = request_digest([self._call(self.CANDIDATE, n=1)], key)
        judged = request_digest(
            [self._call(self.CANDIDATE, n=1), self._call(self.JUDGE, temperature=0)],
            key,
        )
        assert alone == judged

    def test_unknown_when_what_was_sent_is_not_on_record(self):
        key = call_key(self.CANDIDATE)
        assert request_digest([{"model": self.CANDIDATE}], key) is None
        assert request_digest([self._call(self.JUDGE, n=1)], key) is None
        # A call that names no model is no one's, so it is not the candidate's.
        assert request_digest([{"request_params": {"n": 1}}], key) is None

    def test_unknown_when_what_was_sent_will_not_digest(self):
        key = call_key(self.CANDIDATE)
        assert request_digest([self._call(self.CANDIDATE, x=object())], key) is None


class TestRequestDefinition:
    """Only the body and a stream flag define a request, for every dialect."""

    KEY = call_key({"model": "m", "api_base": None, "default_params": {}})

    def _digest(self, params):
        return request_digest(
            [{"model": {"model": "m", "default_params": {}}, "request_params": params}],
            self.KEY,
        )

    @staticmethod
    def _google(model, prefix="/v1beta", stream=False, temperature=0.2):
        method = "streamGenerateContent" if stream else "generateContent"
        params = {
            "envelope": "http",
            "method": "POST",
            "path": f"{prefix}/models/{model}:{method}",
            "body": {"generationConfig": {"temperature": temperature}},
        }
        if stream:
            params["query"] = {"alt": "sse"}
        return params

    def test_google_path_model_and_gateway_are_left_out(self):
        a = self._digest(self._google("gemini-2.5-flash"))
        assert a == self._digest(self._google("gemini-2.5-pro"))
        assert a == self._digest(self._google("gemini-2.5-flash", prefix="/gw/v1"))
        assert a != self._digest(self._google("gemini-2.5-flash", temperature=0.3))

    def test_google_streaming_counts_once_normalized(self):
        streamed = self._google("gemini-2.5-flash", stream=True)
        assert request_definition(streamed) == {
            "body": {"generationConfig": {"temperature": 0.2}},
            "stream": True,
        }
        assert self._digest(streamed) != self._digest(self._google("gemini-2.5-flash"))

    def test_flat_bodies_drop_model_and_stream_options(self):
        chat = {
            "model": "a",
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert request_definition(chat) == {"body": {"temperature": 0}, "stream": True}
        assert self._digest(chat) == self._digest(chat | {"model": "b"})
        assert self._digest(chat) != self._digest(chat | {"stream": False})

    def test_a_flat_and_an_envelope_spelling_agree(self):
        flat = {"temperature": 0.2, "stream": False}
        envelope = {
            "envelope": "http",
            "method": "POST",
            "path": "/x",
            "body": {"temperature": 0.2},
        }
        assert request_definition(flat) == request_definition(envelope)

    def test_a_flat_body_with_path_and_body_fields_stays_whole(self):
        """Only the dialect's own marker makes an envelope: a flat body whose
        request happens to carry `path` or `method` keeps them."""
        flat = {"path": "/srv/a", "method": "beam", "body": {"x": 1}, "query": "q"}
        assert request_definition(flat) == {"body": flat, "stream": False}
        assert self._digest(flat) != self._digest(flat | {"path": "/srv/b"})


# ===================================================================
# Inference
# ===================================================================
class TestInferenceIdentity:
    @staticmethod
    def _identity(model, **overrides):
        args: dict[str, Any] = {
            "model": model,
            "task_args": {"n": 1},
            "runner_keys": {
                "deterministic": False,
                "max_iterations": None,
                "max_retries": None,
            },
            "context_limit": None,
        }
        args.update(overrides)
        return inference_identity(**args)["digest"]

    def test_model_under_test_is_left_out(self):
        """The model is the result's row, so swapping it keeps the identity."""
        a = MockChatModel(temperature=0.0)
        b = MockChatModel(temperature=0.0)
        b._model = "another-model"
        assert self._identity(a) == self._identity(b)

    def test_endpoint_is_left_out(self):
        a = MockChatModel(temperature=0.0)
        b = MockChatModel(temperature=0.0)
        b._api_base = "http://elsewhere:9000/v1"
        assert self._identity(a) == self._identity(b)

    def test_how_the_model_is_served_is_left_out(self):
        """An engine the run launched and the same model served elsewhere ask
        the same: the engine is the serving fingerprint's, not this digest's."""
        external = self._identity(MockChatModel(temperature=0.0))
        with patch.object(
            MockChatModel,
            "deployment",
            new_callable=PropertyMock,
            return_value=_launched_deployment(),
        ):
            launched = MockChatModel(temperature=0.0)
            assert launched.deployment.engine.engine_id == "sglang"
            assert self._identity(launched) == external

    def test_request_arguments_count(self):
        assert self._identity(MockChatModel(temperature=0.0)) != self._identity(
            MockChatModel(temperature=0.7)
        )

    @pytest.mark.parametrize(
        ("key", "value"),
        [("deterministic", True), ("max_iterations", 3), ("max_retries", 2)],
    )
    def test_result_changing_runner_keys_count(self, key, value):
        model = MockChatModel()
        base = {"deterministic": False, "max_iterations": None, "max_retries": None}
        changed = base | {key: value}
        assert self._identity(model, runner_keys=base) != self._identity(
            model, runner_keys=changed
        )

    def test_n_counts(self):
        model = MockChatModel()
        assert self._identity(model, task_args={"n": 1}) != self._identity(
            model, task_args={"n": 4}
        )

    def test_context_limit_counts(self):
        """It decides which prompts overflow, so it changes results."""
        model = MockChatModel()
        assert self._identity(model, context_limit=4096) != self._identity(
            model, context_limit=32768
        )
        assert self._identity(model, context_limit=4096) != self._identity(model)


def _launched_deployment() -> Deployment:
    return Deployment(
        deployment_id="dep-1",
        plan=DeploymentPlanProjection(fingerprint="sha256:p", engine_id="sglang"),
        engine=Engine("sglang"),
        engine_source="deployment",
        api_base="http://127.0.0.1:30000/v1",
        endpoints={},
        topology=None,
        metrics_url=None,
        facts=ServingFacts(engine_version="0.5.4"),
    )


class TestLegacyInferenceDigest:
    """The digest identity schemas 1 to 3 wrote, kept so their directories
    still resume."""

    KEYS = {"deterministic": False, "max_iterations": None, "max_retries": None}

    def _digest(self, model, recipe=None):
        return legacy_inference_digest(
            model=model, task_args={"n": 1}, runner_keys=self.KEYS, recipe=recipe
        )

    def test_matches_the_formula_released_directories_were_written_with(self):
        model = MockChatModel(temperature=0.0)
        payload = {
            "model_args": model_request_args(model),
            "engine": "unknown",
            "task_args": {"n": 1},
            "runner": self.KEYS,
            "recipe": {"backend": "vllm"},
        }
        assert self._digest(model, {"backend": "vllm"}) == digest(payload)

    def test_recipe_and_engine_count(self):
        model = MockChatModel()
        assert self._digest(model, {"backend": "vllm"}) != self._digest(
            model, {"backend": "sglang"}
        )
        external = self._digest(model)
        with patch.object(
            MockChatModel,
            "deployment",
            new_callable=PropertyMock,
            return_value=_launched_deployment(),
        ):
            assert self._digest(MockChatModel()) != external


class TestInferenceMatches:
    def test_this_schema_compares_the_request_identity(self):
        block = {"schema": IDENTITY_SCHEMA, "inference": {"digest": "sha256:a"}}
        assert inference_matches(block, "sha256:a", "sha256:old") is True
        assert inference_matches(block, "sha256:b", "sha256:a") is False

    @pytest.mark.parametrize("schema", [1, 2, 3])
    def test_an_older_schema_compares_the_digest_it_was_written_with(self, schema):
        block = {"schema": schema, "inference": {"digest": "sha256:old"}}
        assert inference_matches(block, "sha256:new", "sha256:old") is True
        assert inference_matches(block, "sha256:old", "sha256:other") is False
        assert inference_matches(block, "sha256:old", None) is None

    def test_an_unknown_side_is_unknown(self):
        assert inference_matches(None, "sha256:a") is None
        assert inference_matches({"schema": IDENTITY_SCHEMA}, "sha256:a") is None
        block = {"schema": IDENTITY_SCHEMA, "inference": {"digest": None}}
        assert inference_matches(block, "sha256:a") is None
        block = {"schema": IDENTITY_SCHEMA, "inference": {"digest": "sha256:a"}}
        assert inference_matches(block, None) is None


class TestServingFingerprint:
    def test_an_external_endpoint(self):
        assert serving_fingerprint(MockChatModel(), None) == {
            "engine": "unknown",
            "engine_version": None,
            "launch": "external",
            "recipe": None,
        }

    def test_an_engine_the_run_launched(self):
        model = MockChatModel()
        recipe = {"backend": "sglang", "assignments": [{"role": "full"}]}
        with patch.object(
            type(model),
            "deployment",
            new_callable=PropertyMock,
            return_value=_launched_deployment(),
        ):
            fingerprint = serving_fingerprint(model, recipe)
        assert fingerprint == {
            "engine": "sglang",
            "engine_version": "0.5.4",
            "launch": "in_run",
            "recipe": recipe,
        }
        assert fingerprint["recipe"] is not recipe

    def test_what_a_caller_states_about_an_endpoint_is_recorded_as_given(self):
        stated = {
            "engine_version": "0.5.4",
            "topology": {"prefill": 1, "decode": 3},
            "hardware": "8x H200",
        }
        fingerprint = serving_fingerprint(MockChatModel(), None, stated)
        assert fingerprint["stated"] == stated
        assert fingerprint["stated"] is not stated
        # The deployment reported no version, so the stated one stands in.
        assert fingerprint["engine_version"] == "0.5.4"
        assert "stated" not in serving_fingerprint(MockChatModel(), None, {})

    def test_a_reported_engine_version_wins_over_a_stated_one(self):
        model = MockChatModel()
        with patch.object(
            type(model),
            "deployment",
            new_callable=PropertyMock,
            return_value=_launched_deployment(),
        ):
            fingerprint = serving_fingerprint(model, None, {"engine_version": "x"})
        assert fingerprint["engine_version"] == "0.5.4"
        assert fingerprint["stated"] == {"engine_version": "x"}


# ===================================================================
# Grader
# ===================================================================
@pytest.fixture
def plugin_tree(tmp_path, monkeypatch):
    """A throwaway ``sieval_plugins`` package on ``sys.path``."""
    root = tmp_path / "plugins"
    pkg = root / "sieval_plugins"
    (pkg / "graders").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "graders" / "__init__.py").write_text("from .exact import grade\n")
    (pkg / "graders" / "exact.py").write_text(
        "from sieval.tasks import _math_verify\n\ndef grade(a, b):\n    return a == b\n"
    )
    (pkg / "unused.py").write_text("X = 1\n")
    (pkg / "task.py").write_text(
        textwrap.dedent(
            """
            import json

            from sieval.core.tasks import Task
            from sieval_plugins.graders import grade
            from . import helpers


            class PluginTask(Task):
                async def preprocess(self, raw, ctx):
                    return raw

                async def infer(self, pre, ctx):
                    return pre

                async def postprocess(self, inf, ctx):
                    return inf

                async def feedback(self, post, ctx):
                    return True, grade(post, post)

                async def report(self, finals, fails):
                    return {}
            """
        )
    )
    (pkg / "helpers.py").write_text("def helper():\n    return 1\n")
    monkeypatch.syspath_prepend(str(root))
    for name in list(sys.modules):
        if name == "sieval_plugins" or name.startswith("sieval_plugins."):
            monkeypatch.delitem(sys.modules, name)
    import importlib

    module = importlib.import_module("sieval_plugins.task")
    yield pkg, module.PluginTask
    for name in list(sys.modules):
        if name == "sieval_plugins" or name.startswith("sieval_plugins."):
            sys.modules.pop(name, None)


class TestGraderModules:
    def test_gsm8k_closure_leaves_out_core_and_datasets(self):
        from sieval.tasks import GSM8KZeroShotGenTask

        modules = grader_modules(GSM8KZeroShotGenTask)
        assert set(modules) == {
            "sieval/tasks/gsm8k_0shot_gen.py",
            "sieval/tasks/_math_verify.py",
            "sieval/community/math.py",
            "sieval/community/deepseek_math.py",
            "sieval/community/_sympy_guards.py",
        }
        assert all(v.startswith("sha256:") for v in modules.values())

    def test_digests_its_own_package_whole_and_follows_sieval(self, plugin_tree):
        """The task's own package is keyed by path under `.`, unimported
        files included; `sieval` only as far as imports reach."""
        pkg, task_cls = plugin_tree
        modules = grader_modules(task_cls)
        assert set(modules) == {
            "./__init__.py",
            "./task.py",
            "./helpers.py",
            "./unused.py",
            "./graders/__init__.py",
            "./graders/exact.py",
            "sieval/tasks/_math_verify.py",
            "sieval/community/math.py",
        }
        source = (pkg / "helpers.py").read_bytes()
        assert modules["./helpers.py"] == (
            "sha256:" + hashlib.sha256(source).hexdigest()
        )

    def test_a_changed_helper_changes_the_digest(self, plugin_tree):
        pkg, task_cls = plugin_tree
        before = grader_modules(task_cls)
        (pkg / "helpers.py").write_text("def helper():\n    return 2\n")
        after = grader_modules(task_cls)
        assert before["./helpers.py"] != after["./helpers.py"]
        assert before["./task.py"] == after["./task.py"]

    def test_walks_the_mro(self, plugin_tree):
        _, task_cls = plugin_tree

        class Sub(task_cls):
            pass

        Sub.__module__ = "tests.not_followed"
        assert "./task.py" in grader_modules(Sub)

    def test_a_task_outside_the_packages_has_no_modules(self):
        from tests.conftest import MockDataset  # noqa: F401 -- only for its module

        class Local:
            pass

        assert grader_modules(Local) == {}

    def test_shared_package_is_followed(self, tmp_path, monkeypatch):
        """The shared package is digested whole like any plugin package;
        parent-relative imports resolve."""
        assert "sieval_shared" in GRADER_PACKAGES
        root = tmp_path / "shared"
        (root / "sieval_shared" / "judging").mkdir(parents=True)
        (root / "sieval_shared" / "__init__.py").write_text("VERSION = 1\n")
        (root / "sieval_shared" / "judging" / "__init__.py").write_text("")
        (root / "sieval_shared" / "judging" / "prompts.py").write_text(
            "from .. import VERSION\nfrom ..missing import nothing\n"
        )
        monkeypatch.syspath_prepend(str(root))

        class Local:
            pass

        (root / "sieval_shared" / "judging" / "unused.py").write_text("X = 1\n")
        Local.__module__ = "sieval_shared.judging.prompts"
        assert set(grader_modules(Local)) == {
            "./__init__.py",
            "./judging/__init__.py",
            "./judging/prompts.py",
            "./judging/unused.py",
        }

    def test_an_uninstalled_package_is_skipped(self):
        class Local:
            pass

        Local.__module__ = "sieval_plugins_never_installed_here.task"
        assert grader_modules(Local) == {}


class TestGraderPluginPackages:
    """A plugin package named `sieval_<x>_plugin` is followed like `sieval_plugins`."""

    @pytest.fixture
    def plugin(self, tmp_path, monkeypatch):
        root = tmp_path / "plug"
        pkg = root / "sieval_sysbench_plugin"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "_scoring.py").write_text("X = 1\n")
        (pkg / "task.py").write_text(
            "from sieval.core.tasks.task import Task\n"
            "from sieval.tasks import _math_verify\n"
            "from ._scoring import X\n"
            "class PTask(Task):\n    pass\n"
        )
        monkeypatch.syspath_prepend(str(root))
        import importlib

        yield pkg, importlib.import_module("sieval_sysbench_plugin.task").PTask
        for name in list(sys.modules):
            if name.startswith("sieval_sysbench_plugin"):
                sys.modules.pop(name, None)

    def test_its_own_package_is_walked(self, plugin):
        _, task_cls = plugin
        modules = grader_modules(task_cls)
        assert "./task.py" in modules
        assert "./_scoring.py" in modules
        assert "sieval/tasks/_math_verify.py" in modules
        excluded = ("sieval/core/", "sieval/datasets/")
        assert not any(m.startswith(excluded) for m in modules)

    def test_a_changed_plugin_helper_changes_the_digest(self, plugin):
        pkg, task_cls = plugin
        before = grader_modules(task_cls)
        (pkg / "_scoring.py").write_text("X = 2\n")
        assert grader_modules(task_cls) != before

    def test_a_non_plugin_package_is_not_followed(self):
        class Local:
            pass

        Local.__module__ = "tests.conftest"
        assert grader_modules(Local) == {}


class TestGraderKeysByPath:
    """Issue #170: the same plugin code under another package name, as a
    runner merging several variants into one process places it, digests the
    same."""

    TASK = (
        "from sieval.core.tasks.task import Task\n"
        "from sieval_shared import RUBRIC\n"
        "from .scoring import grade\n"
        "class VTask(Task):\n    pass\n"
    )

    @pytest.fixture
    def load(self, tmp_path, monkeypatch):
        import importlib

        shared = tmp_path / "shared" / "sieval_shared"
        shared.mkdir(parents=True)
        (shared / "__init__.py").write_text("RUBRIC = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path / "shared"))
        loaded: list[str] = []

        def load(name: str, task: str = self.TASK, init: bool = True) -> type:
            root = tmp_path / name
            pkg = root / name
            pkg.mkdir(parents=True)
            if init:
                (pkg / "__init__.py").write_text("")
            (pkg / "scoring.py").write_text("def grade():\n    return 1\n")
            (pkg / "_dynamic.py").write_text("Y = 1\n")
            (pkg / "task.py").write_text(task)
            monkeypatch.syspath_prepend(str(root))
            loaded.append(name)
            return importlib.import_module(f"{name}.task").VTask

        yield load
        for name in list(sys.modules):
            if name.partition(".")[0] in (*loaded, "sieval_shared"):
                sys.modules.pop(name, None)

    def test_a_renamed_copy_digests_the_same(self, load):
        local = load("sieval_plugins")
        merged = load("sieval_plugins__ugmathbench__math1k_ex")
        assert grader_modules(local) == grader_modules(merged)
        assert grader_task(local) == grader_task(merged) == "./task.py:VTask"
        assert "./_dynamic.py" in grader_modules(local)
        assert "sieval_shared/__init__.py" in grader_modules(local)

    def test_a_renamed_copy_with_a_changed_file_differs(self, load):
        local = load("sieval_plugins")
        merged = load(
            "sieval_plugins__ugmathbench__math1k_ex", self.TASK + "# edited\n"
        )
        assert grader_modules(local) != grader_modules(merged)

    def test_a_renamed_namespace_package_digests_the_same(self, load):
        local = load("sieval_plugins", init=False)
        merged = load("sieval_plugins__ugmathbench__math1k_ex", init=False)
        modules = grader_modules(local)
        assert {"./task.py", "./scoring.py", "./_dynamic.py"} <= set(modules)
        assert modules == grader_modules(merged)

    def test_a_file_outside_its_package_is_refused(self):
        with pytest.raises(ValueError, match="outside package"):
            _file_key("sieval.tasks.x", "/elsewhere/x.py", None)

    def test_a_task_without_source_keeps_its_import_path(self):
        class Local:
            pass

        Local.__module__ = "sieval_plugins_never_installed_here.task"
        assert grader_task(Local) == (
            f"sieval_plugins_never_installed_here.task.{Local.__qualname__}"
        )

    def test_a_builtin_task_is_keyed_under_sieval(self):
        from sieval.tasks import GSM8KZeroShotGenTask

        assert grader_task(GSM8KZeroShotGenTask) == (
            "sieval/tasks/gsm8k_0shot_gen.py:GSM8KZeroShotGenTask"
        )


#: Path to sieval-plugins' ``tools/variants/rewrite.py``; unset skips that test.
REWRITE_ENV = "SIEVAL_TEST_VARIANT_REWRITE"


class TestGraderOwnPackageName:
    """A plugin package renamed by sieval-plugins' variant rewrite (#170)
    digests the same as the original."""

    OWN = "sieval_plugins"
    RENAMED = "sieval_plugins__ugmathbench__math1k_ex"

    #: Every absolute self-import form, plus non-import uses of the name.
    TASK = (
        '"""Grades with {pkg}.scoring."""\n'
        "import json\n"
        "import {pkg}\n"
        "import {pkg}.scoring\n"
        "import {pkg}.scoring as sc\n"
        "import os, {pkg}.scoring as sc2\n"
        "from {pkg} import scoring  # trailing comment kept\n"
        "from {pkg}.scoring import (\n"
        "    grade,\n"
        ")\n"
        "from sieval.core.tasks.task import Task\n"
        "from sieval_shared import RUBRIC\n"
        "from . import scoring as rel\n"
        "NAME = 'see {pkg} docs'  # {pkg}\n"
        "class VTask(Task):\n"
        "    def score(self):\n"
        "        return {pkg}.scoring.grade() + sc.grade() + json.loads('1')\n"
    )

    #: ``TASK`` as the variant rewrite writes it under package *pkg*.
    REWRITTEN = (
        '"""Grades with sieval_plugins.scoring."""\n'
        "import json\n"
        "import {pkg} as sieval_plugins\n"
        "import {pkg}.scoring, {pkg} as sieval_plugins\n"
        "import {pkg}.scoring as sc\n"
        "import os, {pkg}.scoring as sc2\n"
        "from {pkg} import scoring  # trailing comment kept\n"
        "from {pkg}.scoring import grade\n"
        "from sieval.core.tasks.task import Task\n"
        "from sieval_shared import RUBRIC\n"
        "from . import scoring as rel\n"
        "NAME = 'see sieval_plugins docs'  # sieval_plugins\n"
        "class VTask(Task):\n"
        "    def score(self):\n"
        "        return sieval_plugins.scoring.grade() + sc.grade() + "
        "json.loads('1')\n"
    )

    #: Imported by nothing, so only the whole-package walk digests it.
    SELFTEST = "from {pkg}.scoring import grade\nassert grade() == 1\n"

    @staticmethod
    def write(root, name: str, task: str, selftest: str) -> None:
        pkg = root / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "scoring.py").write_text("def grade():\n    return 1\n")
        (pkg / "_selftest.py").write_text(selftest)
        (pkg / "task.py").write_text(task)

    @pytest.fixture
    def load(self, tmp_path, monkeypatch):
        import importlib

        shared = tmp_path / "shared" / "sieval_shared"
        shared.mkdir(parents=True)
        (shared / "__init__.py").write_text("RUBRIC = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path / "shared"))
        loaded: list[str] = []

        def load(root, name: str) -> type:
            monkeypatch.syspath_prepend(str(root))
            loaded.append(name)
            return importlib.import_module(f"{name}.task").VTask

        yield load
        for name in list(sys.modules):
            if name.partition(".")[0] in (*loaded, "sieval_shared"):
                sys.modules.pop(name, None)

    def original(self, tmp_path, load) -> type:
        root = tmp_path / "original"
        own = self.OWN
        self.write(root, own, self.TASK.format(pkg=own), self.SELFTEST.format(pkg=own))
        return load(root, own)

    def test_a_rewritten_copy_digests_the_same(self, tmp_path, load):
        original = self.original(tmp_path, load)
        root = tmp_path / "rewritten"
        name = self.RENAMED
        self.write(
            root,
            name,
            self.REWRITTEN.format(pkg=name),
            self.SELFTEST.format(pkg=name),
        )
        renamed = load(root, name)
        assert grader_modules(original) == grader_modules(renamed)

    def test_the_name_outside_an_import_still_counts(self, tmp_path, load):
        """Only import statements are normalised."""
        original = self.original(tmp_path, load)
        root = tmp_path / "copy"
        name = self.RENAMED
        task = self.REWRITTEN.replace("return sieval_plugins.", "return {pkg}.")
        task = task.replace(" as sieval_plugins\n", " as {pkg}\n")
        self.write(root, name, task.format(pkg=name), self.SELFTEST.format(pkg=name))
        renamed = grader_modules(load(root, name))
        assert renamed["./task.py"] != grader_modules(original)["./task.py"]
        assert renamed["./_selftest.py"] == grader_modules(original)["./_selftest.py"]

    def test_an_unimported_file_with_a_self_import_counts(self, tmp_path, load):
        original = self.original(tmp_path, load)
        before = grader_modules(original)
        assert "./_selftest.py" in before
        selftest = tmp_path / "original" / self.OWN / "_selftest.py"
        selftest.write_text(self.SELFTEST.format(pkg=self.OWN) + "# edited\n")
        after = grader_modules(original)
        assert after["./_selftest.py"] != before["./_selftest.py"]

    def test_a_changed_import_target_still_counts(self, tmp_path, load):
        """Only the package's own name is left out, not what it imports."""
        original = self.original(tmp_path, load)
        before = grader_modules(original)
        task = tmp_path / "original" / self.OWN / "task.py"
        edited = task.read_text().replace("    grade,\n", "    grade as g,\n")
        assert edited != task.read_text()
        task.write_text(edited)
        assert grader_modules(original)["./task.py"] != before["./task.py"]

    @pytest.mark.parametrize(
        "line",
        [
            "from sieval_shared import RUBRIC\n",
            "from sieval_plugins__other__v1.scoring import grade\n",
            "import sieval_plugins__other__v1.scoring\n",
            "import sieval\n",
        ],
    )
    def test_another_packages_name_still_counts(self, line):
        """A cross-package name is the dependency's identity; it stays."""
        renamed = line.replace("sieval_shared", "sieval_shared2").replace(
            "__other__v1", "__other__v2"
        )
        renamed = renamed.replace("import sieval\n", "import sieval2\n")
        own = self.OWN
        assert _without_own_name(line.encode(), own) == line.encode()
        assert _without_own_name(line.encode(), own) != _without_own_name(
            renamed.encode(), own
        )

    def test_a_file_without_a_self_import_hashes_its_raw_bytes(self, tmp_path, load):
        """Unchanged from 81ad90c: only files with a self-import rotate."""
        original = self.original(tmp_path, load)
        pkg = tmp_path / "original" / self.OWN
        modules = grader_modules(original)
        for rel in ("__init__.py", "scoring.py"):
            raw = hashlib.sha256((pkg / rel).read_bytes()).hexdigest()
            assert modules[f"./{rel}"] == f"sha256:{raw}"
        shared = (tmp_path / "shared" / "sieval_shared" / "__init__.py").read_bytes()
        assert modules["sieval_shared/__init__.py"] == (
            f"sha256:{hashlib.sha256(shared).hexdigest()}"
        )
        task = (pkg / "task.py").read_bytes()
        assert modules["./task.py"] != f"sha256:{hashlib.sha256(task).hexdigest()}"

    def test_only_the_import_spans_change(self):
        source = self.TASK.format(pkg=self.OWN).encode()
        out = _without_own_name(source, self.OWN).decode()
        assert "import <package> as sieval_plugins\n" in out
        assert "import <package>.scoring, <package> as sieval_plugins\n" in out
        assert "from <package> import scoring  # trailing comment kept\n" in out
        assert "from <package>.scoring import grade\n" in out
        assert "'see sieval_plugins docs'  # sieval_plugins\n" in out
        assert '"""Grades with sieval_plugins.scoring."""' in out
        assert "return sieval_plugins.scoring.grade()" in out
        assert "from . import scoring as rel\n" in out

    def test_an_explicit_own_alias_is_not_normalised(self):
        """``import own.x, own as n`` reads as the rewrite's form, which
        appends a second binding, so it still digests per name."""
        own, name = self.OWN, self.RENAMED
        line = f"import {own}.x, {own} as sp\n"
        rewritten = f"import {name}.x, {name} as {own}, {name} as sp\n"
        assert _without_own_name(line.encode(), own) == (
            b"import <package>.x, <package> as sp\n"
        )
        assert _without_own_name(line.encode(), own) != _without_own_name(
            rewritten.encode(), name
        )

    def test_a_missing_init_digests_as_the_empty_one_the_rewrite_adds(
        self, tmp_path, load
    ):
        def build(root, name: str, init: bool) -> type:
            self.write(
                root,
                name,
                self.TASK.format(pkg=name)
                if name == self.OWN
                else self.REWRITTEN.format(pkg=name),
                self.SELFTEST.format(pkg=name),
            )
            prompts = root / name / "prompts"
            prompts.mkdir()
            (prompts / "a.txt").write_text("prompt\n")
            if init:
                (prompts / "__init__.py").write_text("")
            return load(root, name)

        original = grader_modules(build(tmp_path / "original", self.OWN, False))
        renamed = grader_modules(build(tmp_path / "copy", self.RENAMED, True))
        assert original == renamed
        assert original["./prompts/__init__.py"] == (
            f"sha256:{hashlib.sha256(b'').hexdigest()}"
        )

    def test_an_empty_init_on_disk_digests_as_a_missing_one(self, tmp_path, load):
        original = self.original(tmp_path, load)
        init = tmp_path / "original" / self.OWN / "__init__.py"
        before = grader_modules(original)
        init.unlink()
        assert grader_modules(original) == before

    @pytest.mark.parametrize(
        "source",
        [
            b"from sieval_plugins.scoring import grade\ndef broken(:\n",
            b"# -*- coding: latin-1 -*-\nimport sieval_plugins.x\nS = '\xe9'\n",
            b"import sieval_plugins.x\nS = '\xff'\n",
        ],
        ids=["syntax-error", "latin-1", "not-utf-8"],
    )
    def test_an_unreadable_file_keeps_its_raw_bytes(self, source):
        assert _without_own_name(source, self.OWN) == source

    @pytest.mark.parametrize(
        "prefix",
        [b"\xef\xbb\xbf", b"# -*- coding: latin-1 -*-\n"],
        ids=["byte-order-mark", "latin-1-cookie"],
    )
    def test_offsets_index_the_utf_8_bytes(self, prefix):
        """A multibyte character before the import does not shift the span."""
        line = "S = '\u00e9'; from sieval_plugins import x  # \u00e9\n"
        out = _without_own_name(prefix + line.encode(), self.OWN)
        expected = line.replace("sieval_plugins", "<package>")
        assert out == prefix + expected.encode()

    @pytest.mark.skipif(
        not os.environ.get(REWRITE_ENV), reason=f"{REWRITE_ENV} is not set"
    )
    def test_sieval_plugins_variant_rewrite_digests_the_same(self, tmp_path, load):
        """Against the real tool."""
        original = self.original(tmp_path, load)
        target = tmp_path / "merged"
        script = (
            "import importlib.util, pathlib, sys\n"
            "spec = importlib.util.spec_from_file_location('rewrite', sys.argv[1])\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['rewrite'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "copied = mod.copy_package(\n"
            "    pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]),"
            " 'ugmathbench/math1k-ex'\n"
            ")\n"
            "print(copied.package)\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                script,
                os.environ[REWRITE_ENV],
                str(tmp_path / "original"),
                str(target),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        name = result.stdout.strip().splitlines()[-1]
        assert name == self.RENAMED
        assert (target / name / "task.py").read_text() == self.REWRITTEN.format(
            pkg=name
        )
        assert grader_modules(original) == grader_modules(load(target, name))


class TestGraderPatchesAndDynamicImports:
    """What no import statement shows: a patch that replaced a task method, and
    a plugin package that loads its graders by name."""

    @pytest.fixture
    def patch_pkg(self, tmp_path, monkeypatch):
        root = tmp_path / "patch"
        pkg = root / "arck_demo_patch"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("from .patch import apply\n")
        (pkg / "patch.py").write_text(
            "from ._helpers import SCALE\n"
            "async def report(self, finals, fails):\n"
            "    return {'score': SCALE}\n"
            "def apply(cls):\n"
            "    cls.report = report\n"
        )
        (pkg / "_helpers.py").write_text("SCALE = 1\n")
        monkeypatch.syspath_prepend(str(root))
        import importlib

        yield pkg, importlib.import_module("arck_demo_patch")
        for name in list(sys.modules):
            if name.startswith("arck_demo_patch"):
                sys.modules.pop(name, None)

    def test_a_patched_method_brings_its_package_in(self, patch_pkg):
        from sieval.core.tasks.task import Task

        pkg, module = patch_pkg

        class Local(Task):
            pass

        Local.__module__ = "sieval.tasks.gsm8k_0shot_gen"
        before = grader_modules(Local)
        module.apply(Local)
        after = grader_modules(Local)
        assert not any(m.startswith("arck_demo_patch/") for m in before)
        assert {
            "arck_demo_patch/__init__.py",
            "arck_demo_patch/patch.py",
            "arck_demo_patch/_helpers.py",
        } <= set(after)
        (pkg / "_helpers.py").write_text("SCALE = 2\n")
        assert grader_modules(Local) != after

    def test_stdlib_and_installed_functions_are_not_patches(self):
        import json

        class Local:
            loads = staticmethod(json.loads)
            check = staticmethod(pytest.approx)

        Local.__module__ = "tests.not_followed"
        assert grader_modules(Local) == {}

    def test_a_package_loading_graders_by_name_is_digested_whole(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "dyn"
        pkg = root / "sieval_dyn_plugin"
        (pkg / "graders").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "task.py").write_text(
            "from sieval.core.tasks.task import Task\n"
            "from .graders import score\n"
            "class DTask(Task):\n    pass\n"
        )
        (pkg / "graders" / "__init__.py").write_text(
            "import importlib\n"
            "def score(slug):\n"
            "    return importlib.import_module(f'{__name__}.{slug}').grade\n"
        )
        (pkg / "graders" / "venus.py").write_text("def grade():\n    return 1\n")
        (pkg / "graders" / "tape.py").write_text("def grade():\n    return 2\n")
        monkeypatch.syspath_prepend(str(root))
        import importlib

        try:
            task_cls = importlib.import_module("sieval_dyn_plugin.task").DTask
            modules = grader_modules(task_cls)
        finally:
            for name in list(sys.modules):
                if name.startswith("sieval_dyn_plugin"):
                    sys.modules.pop(name, None)
        assert {
            "./__init__.py",
            "./task.py",
            "./graders/__init__.py",
            "./graders/venus.py",
            "./graders/tape.py",
        } <= set(modules)


class TestJudges:
    def test_judge_is_recorded_without_its_endpoint(self):
        judge = MockJudgeModel(temperature=0.0)
        judge._api_base = "http://judge.internal:8000/v1"
        [entry] = judge_identities({"grader": judge}, {"grader": "Grade: {x}"}, {})
        assert entry == {
            "role": "grader",
            "model": "mock-judge",
            "version": None,
            "sampling": {"temperature": 0.0},
            "prompt": digest("Grade: {x}"),
        }
        assert "judge.internal" not in canonical_json(entry)

    @pytest.mark.parametrize(
        ("defaults", "expected"),
        [
            (
                {"reasoning.budget_tokens": 512},
                {
                    "temperature": 0.0,
                    "request_defaults": {"reasoning.budget_tokens": 512},
                },
            ),
            ({}, {"temperature": 0.0}),
        ],
    )
    def test_capability_request_defaults_land_in_judge_sampling(
        self, defaults, expected
    ):
        judge = MockJudgeModel(temperature=0.0)
        plan = SimpleNamespace(request_defaults=RequestDefaults(defaults))
        with patch.object(
            type(judge), "runtime_plan", new_callable=PropertyMock, return_value=plan
        ):
            [entry] = judge_identities({"grader": judge}, {}, {})
        assert entry["sampling"] == expected

    def test_reported_version_and_missing_prompt(self):
        judges = {"grader": MockJudgeModel()}
        [entry] = judge_identities(judges, {}, {"grader": "mock-judge-2026-01"})
        assert entry["version"] == "mock-judge-2026-01"
        assert entry["prompt"] is None

    def test_env_ref_plain_applies_to_every_role(self):
        refs = judge_model_refs(
            ["grader", "extractor"], {JUDGE_MODEL_REF_ENV: "harmony://m/judge@3"}
        )
        assert refs == {
            "grader": "harmony://m/judge@3",
            "extractor": "harmony://m/judge@3",
        }

    def test_env_ref_json_maps_roles(self):
        env = {JUDGE_MODEL_REF_ENV: '{"grader": "harmony://m/a@1", "other": "x"}'}
        assert judge_model_refs(["grader", "extractor"], env) == {
            "grader": "harmony://m/a@1"
        }

    def test_env_ref_absent_or_blank(self):
        assert judge_model_refs(["grader"], {}) == {}
        assert judge_model_refs(["grader"], {JUDGE_MODEL_REF_ENV: "  "}) == {}

    @pytest.mark.parametrize(
        "bad", ['{"grader": 1}', '{"grader": ', '["hm://m/1"]', "[", '{"1": ["x"]}']
    )
    def test_env_ref_bad_json_raises(self, bad):
        """A value that starts like JSON must be a JSON object of strings; a
        JSON array is no longer taken as one plain reference."""
        with pytest.raises(ValueError, match=JUDGE_MODEL_REF_ENV):
            judge_model_refs(["grader"], {JUDGE_MODEL_REF_ENV: bad})
        with pytest.raises(ValueError, match=JUDGE_MODEL_REF_ENV):
            validate_judge_model_ref({JUDGE_MODEL_REF_ENV: bad})

    @pytest.mark.parametrize("good", ["", "hm://m/1", '{"grader": "hm://m/1"}'])
    def test_env_ref_valid_values_pass(self, good):
        validate_judge_model_ref({JUDGE_MODEL_REF_ENV: good})

    def test_judge_model_id_strips_googles_resource_prefix(self):
        assert judge_model_id("models/gemini-2.5-pro") == "gemini-2.5-pro"
        assert judge_model_id("tunedModels/my-judge") == "my-judge"
        assert judge_model_id("gemini-2.5-pro") == "gemini-2.5-pro"
        assert judge_model_id("org/models/x") == "org/models/x"

    def test_env_ref_roles_no_task_has_are_warned(self):
        from loguru import logger

        messages: list[str] = []
        sink = logger.add(
            lambda m: messages.append(m.record["message"]), level="WARNING"
        )
        try:
            env = {JUDGE_MODEL_REF_ENV: '{"grader": "hm://a", "graderr": "hm://b"}'}
            unmatched = warn_unmatched_judge_refs({"grader", "extractor"}, env)
            plain = warn_unmatched_judge_refs({"grader"}, {JUDGE_MODEL_REF_ENV: "hm"})
        finally:
            logger.remove(sink)
        assert unmatched == ["graderr"]
        assert plain == []
        assert len(messages) == 1
        assert "graderr" in messages[0]

    def test_a_models_prefixed_judge_is_one_model(self):
        prefixed = MockJudgeModel()
        prefixed._model = "models/gemini-x"
        plain = MockJudgeModel()
        plain._model = "gemini-x"
        assert judge_identities({"g": prefixed}, {}) == judge_identities(
            {"g": plain}, {}
        )

    def test_env_ref_lands_in_the_judge_entry(self):
        env = {JUDGE_MODEL_REF_ENV: "harmony://m/judge@3"}
        [entry] = judge_identities({"grader": MockJudgeModel()}, {}, {}, environ=env)
        assert entry["ref"] == "harmony://m/judge@3"

    def test_grader_digest_covers_modules_and_judges(self):
        judges = judge_identities({"grader": MockJudgeModel()}, {}, {})
        base = grader_identity({"m": "sha256:1"}, judges)
        other_module = grader_identity({"m": "sha256:2"}, judges)
        no_judge = grader_identity({"m": "sha256:1"}, [])
        assert len({base["digest"], other_module["digest"], no_judge["digest"]}) == 3
        assert base["modules"] == {"m": "sha256:1"}


# ===================================================================
# Document helpers
# ===================================================================
class TestDocument:
    def test_carried_identity_tolerates_old_files(self):
        assert carried_identity(None) == empty_identity()
        assert carried_identity({"version": "0.6.0"}) == empty_identity()

    def test_carried_identity_keeps_known_stages(self):
        data = {"digest": "sha256:d", "samples": 1}
        carried = carried_identity({"identity": {"schema": 2, "data": data}})
        assert carried["data"] == data
        assert carried["report"] is None

    def test_a_schema_1_block_reads_as_this_schema(self):
        data = {"digest": "sha256:d", "samples": 1}
        prompt = {"digest": "sha256:p", "samples": 1}
        grader = {"digest": "sha256:g", "modules": {}, "judges": []}
        block = {
            "schema": 1,
            "data": data,
            "prompt": prompt,
            "inference": {"digest": "sha256:i"},
            "grader": grader,
        }
        read = read_identity(block)
        assert read is not None
        identity, legacy = read
        assert identity["schema"] == IDENTITY_SCHEMA == 4
        assert identity["data"] == data
        assert identity["preprocess"] == prompt
        assert identity["inference"] == {"digest": "sha256:i"}
        assert all(identity.get(s) is None for s in GRADING_STAGES)
        assert legacy == grader
        # Carried, the old inference digest is dropped: schema 4 digests it
        # without serving, so the old one would read as differing.
        assert carried_identity({"identity": block}) == {**identity, "inference": None}

    def test_a_schema_2_block_is_read_as_written(self):
        prompts = {"digest": "sha256:p", "samples": 1}
        read = read_identity({"schema": 2, "preprocess": prompts})
        assert read is not None
        identity, legacy = read
        assert identity["schema"] == IDENTITY_SCHEMA
        assert identity["preprocess"] == prompts
        assert not is_code_identity(identity["preprocess"])
        assert legacy is None

    def test_another_schema_is_not_read(self):
        assert read_identity({"schema": 5, "data": {"digest": "x"}}) is None
        assert read_identity("nope") is None
        assert carried_identity({"identity": {"schema": 5}}) == empty_identity()

    def test_carried_runs_tolerates_old_and_malformed(self):
        assert carried_runs(None) == []
        assert carried_runs({"runs": "nope"}) == []
        assert carried_runs({"runs": [1, {"start": "fresh"}]}) == [{"start": "fresh"}]

    def test_new_run_record(self):
        when = datetime(2026, 10, 20, tzinfo=UTC)
        record = new_run_record("0.9.0", "fresh", now=when)
        run_id = record.pop("id")
        assert record == {
            "sieval": "0.9.0",
            "at": "2026-10-20T00:00:00+00:00",
            "start": "fresh",
        }
        assert run_id.startswith("20261020T000000.000000Z-")
        assert run_id != new_run_record("0.9.0", "fresh", now=when)["id"]
        assert new_run_record("0.9.0", "resume", ["a: 1 → 2"], when)["config"] == [
            "a: 1 → 2"
        ]

    def test_ordered_stages(self):
        assert ordered_stages({"report", "data", "feedback", "grader"}) == [
            "data",
            "feedback",
            "report",
        ]
        assert STAGES == (
            "data",
            "preprocess",
            "inference",
            "postprocess",
            "feedback",
            "report",
        )


class TestGraderKey:
    JUDGE: JudgeIdentity = {
        "role": "grader",
        "model": "j",
        "version": "j-2026",
        "sampling": {},
        "prompt": None,
        "ref": "hm://j@1",
    }

    def test_args_and_task_change_the_grader(self):
        base = grader_identity({"m": "sha256:1"}, [], args={"t": 0.5}, task="a.T")
        assert base["args"] == {"t": 0.5}
        assert base["task"] == "a.T"
        other_args = grader_identity({"m": "sha256:1"}, [], args={"t": 0.9}, task="a.T")
        other_task = grader_identity({"m": "sha256:1"}, [], args={"t": 0.5}, task="a.U")
        keys = {grader_key(g) for g in (base, other_args, other_task)}
        assert len(keys) == 3

    def test_a_reported_version_or_ref_is_not_a_regrade(self):
        """Only known after the judge is called, or where it was served."""
        unreported = self.JUDGE.copy()
        unreported["version"] = None
        before = grader_identity({}, [unreported])
        after = grader_identity({}, [self.JUDGE])
        assert before["digest"] != after["digest"]
        assert grader_key(before) == grader_key(after)

    def test_unknown_or_malformed_is_none(self):
        assert grader_key(None) is None
        assert grader_key({"modules": "x", "judges": []}) is None
