"""
Tests for session pure functions: class resolution, submodule guessing,
dataset operations, and runner config building.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sieval.cli.leaderboard.session import (
    _NONMATCH_RUNNER_KEYS,
    _STRICT_RUNNER_KEYS,
    _THROUGHPUT_RUNNER_KEYS,
    DETERMINISTIC_DEFAULT_SEED,
    EvalSession,
    _append_resume_note,
    _apply_endpoint_injection,
    _brief_diff,
    _diff_dicts,
    _diff_lines,
    _format_comment_header,
    _guess_submodule_names,
    _reify_cli_overrides,
    _split_header,
    _strip_header,
    _strip_noncomparable_fields,
    arun_session,
    load_class_from_name,
    load_class_from_path,
    resolve_class,
    resolve_deterministic,
    run_session,
    unwrap_proxies,
)
from sieval.core.models.model import Model
from sieval.core.runners import TaskRunnerConfig
from sieval.core.runners.multi_runner import MultiTaskRunner
from tests.conftest import MockChatModel


def _write_yaml_config(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def _prepare_eval_session(
    config_path: Path,
    *,
    models: dict[str, Model[Any]] | None = None,
    resume: bool = False,
) -> MultiTaskRunner:
    runner = EvalSession(config_path=str(config_path), resume=resume)
    if models is not None:
        runner.models = models
    runner._init_runner()
    runner._setup_datasets()
    runner._setup_tasks()
    assert runner.runner is not None
    return runner.runner


# ===================================================================
# load_class_from_path
# ===================================================================
class TestLoadClassFromPath:
    def test_valid_path(self):
        cls = load_class_from_path("sieval.core.models.model.ModelOutput")
        from sieval.core.models.model import ModelOutput

        assert cls is ModelOutput

    # (class_path, expected_exception, expected_error_pattern)
    @pytest.mark.parametrize(
        "class_path,error_type,error_match",
        [
            ("ModelOutput", ValueError, "Invalid class path"),
            ("nonexistent.module.Foo", ImportError, "Could not import"),
            (
                "sieval.core.models.model.NonExistentClass",
                AttributeError,
                "has no class",
            ),
        ],
    )
    def test_invalid_path_raises(self, class_path, error_type, error_match):
        with pytest.raises(error_type, match=error_match):
            load_class_from_path(class_path)


# ===================================================================
# load_class_from_name
# ===================================================================
class TestLoadClassFromName:
    # (class_name, search_modules, expected_fully_qualified_class_path)
    @pytest.mark.parametrize(
        "class_name,search_modules,expected_path",
        [
            (
                "ModelOutput",
                ["sieval.core.models.model"],
                "sieval.core.models.model.ModelOutput",
            ),
            (
                "ChatModel",
                ["sieval.core.models.model", "sieval.core.models.chat_model"],
                "sieval.core.models.chat_model.ChatModel",
            ),
            (
                "ModelOutput",
                ["nonexistent.module", "sieval.core.models.model"],
                "sieval.core.models.model.ModelOutput",
            ),
        ],
    )
    def test_find_class_in_search_modules(
        self, class_name, search_modules, expected_path
    ):
        cls = load_class_from_name(class_name, search_modules)
        assert cls is load_class_from_path(expected_path)

    def test_not_found_raises(self):
        with pytest.raises(ImportError, match="Could not find class"):
            load_class_from_name("FakeClass", ["sieval.core.models.model"])


# ===================================================================
# resolve_class
# ===================================================================
class TestResolveClass:
    # (class_spec, search_modules, expected_fully_qualified_class_path)
    @pytest.mark.parametrize(
        "class_spec,search_modules,expected_path",
        [
            (
                "sieval.core.models.model.ModelOutput",
                [],
                "sieval.core.models.model.ModelOutput",
            ),
            (
                "ModelOutput",
                ["sieval.core.models.model"],
                "sieval.core.models.model.ModelOutput",
            ),
        ],
    )
    def test_resolve_class(self, class_spec, search_modules, expected_path):
        cls = resolve_class(class_spec, search_modules)
        assert cls is load_class_from_path(expected_path)

    def test_leading_dot_raises_value_error(self):
        # Relative import syntax is explicitly rejected.
        with pytest.raises(ValueError, match="Relative import syntax"):
            resolve_class(".ModelOutput", ["sieval.core.models.model"])


# ===================================================================
# _guess_submodule_names
# ===================================================================
class TestGuessSubmoduleNames:
    # (class_name, expected_fragment_in_candidates)
    @pytest.mark.parametrize(
        "class_name,expected_fragment",
        [
            ("MathTask", "math"),
            ("AIME2024Task", "aime_2024"),
            ("GPQADiamondTask", "gpqa"),
        ],
    )
    def test_expected_candidates_present(self, class_name, expected_fragment):
        names = _guess_submodule_names(class_name)
        assert any(expected_fragment in n for n in names)

    def test_task_suffix_removed(self):
        names = _guess_submodule_names("MathTask")
        # Should not keep trailing "_task" in any candidate.
        for n in names:
            assert not n.endswith("_task")

    # (class_name, expected_fragments_in_candidates)
    @pytest.mark.parametrize(
        "class_name,expected_fragments",
        [
            ("AIME2024ZeroShotGenTask", ["0shot", "zero_shot"]),
            ("MathFewShotTask", ["kshot", "few_shot"]),
        ],
    )
    def test_shot_aliases_present(self, class_name, expected_fragments):
        names = _guess_submodule_names(class_name)
        for fragment in expected_fragments:
            assert any(fragment in n for n in names)


# ===================================================================
# Dataset operations (via _apply_dataset_operations)
# ===================================================================
class TestDatasetOperations:
    """Test _apply_dataset_operations logic using mock datasets."""

    def _make_runner(self):
        """Create a minimal EvalSession-like object for testing operations."""

        # We can't instantiate EvalSession without a real file, so we test
        # the method logic directly by creating a mock
        runner = object.__new__(EvalSession)
        return runner

    # (op_name, op_args, dataset_method, method_args, method_kwargs)
    @pytest.mark.parametrize(
        "op,op_args,method_name,method_args,method_kwargs",
        [
            (
                "slice",
                {"num": 10},
                "slice",
                (10,),
                {"split": "test"},
            ),
            (
                "shuffle",
                {"seed": 42},
                "shuffle",
                (),
                {"seed": 42, "split": "test"},
            ),
            (
                "repeat",
                {"times": 3},
                "repeat",
                (3,),
                {"split": "test"},
            ),
        ],
    )
    def test_basic_operations(
        self, op, op_args, method_name, method_args, method_kwargs
    ):
        runner = self._make_runner()
        ds = MagicMock()
        getattr(ds, method_name).return_value = ds

        runner._apply_dataset_operations(ds, [{op: op_args}], "test_ds")
        getattr(ds, method_name).assert_called_once_with(*method_args, **method_kwargs)

    # (operations_yaml, expected_error_pattern)
    @pytest.mark.parametrize(
        "operations,error_match",
        [
            ([{"foobar": {}}], "Unknown operation"),
            ([{"a": 1, "b": 2}], "Invalid operation format"),
            ([{"shuffle": 1}], "args must be a dictionary"),
        ],
    )
    def test_invalid_operation_definitions_raise(self, operations, error_match):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match=error_match):
            runner._apply_dataset_operations(ds, operations, "test_ds")

    def test_renamed_operation_raises_migration_hint(self):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match="'select' was renamed to 'slice'"):
            runner._apply_dataset_operations(ds, [{"select": {"num": 5}}], "test_ds")

    def test_never_shipped_operation_raises_unknown_not_renamed(self):
        # 'stratified_select' never shipped, so it must hit the generic unknown
        # branch — no migration hint for a name users never saw.
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match="Unknown operation 'stratified_select'"):
            runner._apply_dataset_operations(
                ds, [{"stratified_select": {"num": 5}}], "test_ds"
            )

    # (op_name, missing_args, expected_error_pattern)
    @pytest.mark.parametrize(
        "op,missing_args,error_match",
        [
            ("slice", {}, "'slice' requires 'num'"),
            ("repeat", {}, "'repeat' requires 'times'"),
        ],
    )
    def test_operation_required_args(self, op, missing_args, error_match):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match=error_match):
            runner._apply_dataset_operations(ds, [{op: missing_args}], "test_ds")

    def test_chained_operations(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.shuffle.return_value = ds
        ds.slice.return_value = ds

        _result = runner._apply_dataset_operations(
            ds,
            [{"shuffle": {"seed": 0}}, {"slice": {"num": 5}}],
            "test_ds",
        )
        ds.shuffle.assert_called_once()
        ds.slice.assert_called_once()

    def test_operation_argument_variants(self):
        """Validate alias, custom split, and None-arg default behaviors."""
        runner = self._make_runner()

        ds_alias = MagicMock()
        ds_alias.slice.return_value = ds_alias
        runner._apply_dataset_operations(ds_alias, [{"slice": {"n": 7}}], "test_ds")
        ds_alias.slice.assert_called_once_with(7, split="test")

        ds_custom_split = MagicMock()
        ds_custom_split.slice.return_value = ds_custom_split
        runner._apply_dataset_operations(
            ds_custom_split, [{"slice": {"num": 5, "split": "train"}}], "test_ds"
        )
        ds_custom_split.slice.assert_called_once_with(5, split="train")

        ds_none_args = MagicMock()
        ds_none_args.shuffle.return_value = ds_none_args
        runner._apply_dataset_operations(ds_none_args, [{"shuffle": None}], "test_ds")
        ds_none_args.shuffle.assert_called_once_with(seed=0, split="test")

    def test_stratified_sample_dispatch(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.stratified_sample.return_value = ds
        runner._apply_dataset_operations(
            ds,
            [
                {
                    "stratified_sample": {
                        "by": "Subject",
                        "num": 800,
                        "min_per_group": 5,
                        "seed": 42,
                    }
                }
            ],
            "test_ds",
        )
        ds.stratified_sample.assert_called_once_with(
            "Subject", num=800, per_group=None, min_per_group=5, seed=42, split="test"
        )

    def test_stratified_sample_defaults(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.stratified_sample.return_value = ds
        runner._apply_dataset_operations(
            ds, [{"stratified_sample": {"by": "category", "num": 600}}], "test_ds"
        )
        ds.stratified_sample.assert_called_once_with(
            "category",
            num=600,
            per_group=None,
            min_per_group=None,
            seed=0,
            split="test",
        )

    def test_stratified_sample_requires_by(self):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match="requires 'by'"):
            runner._apply_dataset_operations(
                ds, [{"stratified_sample": {"num": 5}}], "test_ds"
            )
        with pytest.raises(ValueError, match="requires 'by'"):
            runner._apply_dataset_operations(ds, [{"stratified_sample": {}}], "test_ds")

    def test_stratified_sample_requires_exactly_one_budget(self):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match="exactly one of 'num' or 'per_group'"):
            runner._apply_dataset_operations(
                ds, [{"stratified_sample": {"by": "Subject"}}], "test_ds"
            )
        with pytest.raises(ValueError, match="exactly one of 'num' or 'per_group'"):
            runner._apply_dataset_operations(
                ds,
                [{"stratified_sample": {"by": "Subject", "num": 5, "per_group": 2}}],
                "test_ds",
            )

    def test_stratified_sample_min_per_group_excludes_per_group(self):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match="cannot be combined with 'per_group'"):
            runner._apply_dataset_operations(
                ds,
                [
                    {
                        "stratified_sample": {
                            "by": "Subject",
                            "per_group": 5,
                            "min_per_group": 1,
                        }
                    }
                ],
                "test_ds",
            )

    def test_stratified_sample_per_group_dispatch(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.stratified_sample.return_value = ds
        runner._apply_dataset_operations(
            ds,
            [
                {
                    "stratified_sample": {
                        "by": ["locale", "subject"],
                        "per_group": 20,
                        "seed": 42,
                    }
                }
            ],
            "test_ds",
        )
        ds.stratified_sample.assert_called_once_with(
            ["locale", "subject"],
            num=None,
            per_group=20,
            min_per_group=None,
            seed=42,
            split="test",
        )


# ===================================================================
# Model type inference
# ===================================================================
class TestInferModelType:
    def _make_runner(self, tasks_cfg=None):
        runner = object.__new__(EvalSession)
        runner.config = {"tasks": tasks_cfg or {}}
        return runner

    def test_explicit_type_and_default(self):
        runner = self._make_runner()
        assert runner._infer_model_type("m", "gen") == "gen"
        assert runner._infer_model_type("m", "chat") == "chat"
        assert runner._infer_model_type("m", None) == "chat"

    def test_inferred_from_task(self):
        """When a task class has model_type attribute, it's used."""

        class FakeTask:
            model_type = "gen"

        tasks_cfg = {
            "t1": {"model": "m", "class": "fake.FakeTask"},
        }
        runner = self._make_runner(tasks_cfg)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=FakeTask,
        ):
            assert runner._infer_model_type("m", None) == "gen"

    def test_conflicting_types_raises(self):
        class GenTask:
            model_type = "gen"

        class ChatTask:
            model_type = "chat"

        tasks_cfg = {
            "t1": {"model": "m", "class": "fake.GenTask"},
            "t2": {"model": "m", "class": "fake.ChatTask"},
        }
        runner = self._make_runner(tasks_cfg)

        call_count = 0

        def mock_resolve(_spec):
            nonlocal call_count
            call_count += 1
            return GenTask if call_count == 1 else ChatTask

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                side_effect=mock_resolve,
            ),
            pytest.raises(ValueError, match="different types"),
        ):
            runner._infer_model_type("m", None)


class TestSetupModelsEngine:
    """`engine` field dispatches a gen model to GenModel vs SglangGenModel."""

    def _make_runner(self, models_cfg):
        runner = object.__new__(EvalSession)
        runner.config = {"models": models_cfg, "tasks": {}}
        runner.models = {}
        runner.deterministic = False
        runner.model_override = None
        return runner

    def test_default_engine_is_gen_model(self):
        from sieval.core.models import GenModel, SglangGenModel

        runner = self._make_runner(
            {"m": {"name": "x", "type": "gen", "api_key": "local"}}
        )
        runner._setup_models()
        assert isinstance(runner.models["m"], GenModel)
        assert not isinstance(runner.models["m"], SglangGenModel)

    def test_sglang_engine_is_sglang_gen_model(self):
        from sieval.core.models import SglangGenModel

        runner = self._make_runner(
            {"m": {"name": "x", "type": "gen", "engine": "sglang", "api_key": "local"}}
        )
        runner._setup_models()
        assert isinstance(runner.models["m"], SglangGenModel)

    def test_explicit_vllm_engine_is_gen_model(self):
        from sieval.core.models import GenModel, SglangGenModel

        runner = self._make_runner(
            {"m": {"name": "x", "type": "gen", "engine": "vllm", "api_key": "local"}}
        )
        runner._setup_models()
        assert isinstance(runner.models["m"], GenModel)
        assert not isinstance(runner.models["m"], SglangGenModel)

    def test_invalid_engine_raises(self):
        runner = self._make_runner(
            {"m": {"name": "x", "type": "gen", "engine": "bogus", "api_key": "local"}}
        )
        with pytest.raises(ValueError, match="invalid engine"):
            runner._setup_models()


# ===================================================================
# Resolve task model / dataset helpers
# ===================================================================
class TestResolveTaskModel:
    def _make_runner(self, models=None):
        runner = object.__new__(EvalSession)
        runner.models = models or {}
        return runner

    def test_explicit_model_ref(self):
        m = MagicMock()
        runner = self._make_runner({"my_model": m})
        result = runner._resolve_task_model({"model": "my_model"}, "t1")
        assert result is m

    def test_single_model_default(self):
        m = MagicMock()
        runner = self._make_runner({"only": m})
        result = runner._resolve_task_model({}, "t1")
        assert result is m

    def test_error_cases(self):
        # (models_map, task_cfg, expected_error_pattern)
        cases = [
            # Explicit model reference does not exist.
            (
                {"my_model": MagicMock()},
                {"model": "bad_ref"},
                "unknown model",
            ),
            # No model available and no reference provided.
            (
                {},
                {},
                "no models defined",
            ),
            # Multiple models available but task doesn't choose one.
            (
                {"a": MagicMock(), "b": MagicMock()},
                {},
                "'model' required",
            ),
        ]
        for models, task_cfg, error_match in cases:
            runner = self._make_runner(models)
            with pytest.raises(ValueError, match=error_match):
                runner._resolve_task_model(task_cfg, "t1")

    def test_model_ref_must_be_string(self):
        runner = self._make_runner({"my_model": MagicMock()})
        with pytest.raises(ValueError, match="'model' must be a string reference"):
            runner._resolve_task_model({"model": 123}, "t1")


class TestResolveTaskDataset:
    def _make_runner(self, datasets=None):
        runner = object.__new__(EvalSession)
        runner.datasets = datasets or {}
        return runner

    def test_success_paths(self):
        ds = MagicMock()
        runner = self._make_runner({"my_ds": ds})
        result = runner._resolve_task_dataset({"dataset": "my_ds"}, "t1")
        assert result is ds

        runner = self._make_runner()
        fake_ds_class = MagicMock()
        fake_ds_instance = MagicMock()
        fake_ds_class.return_value = fake_ds_instance

        with patch(
            "sieval.cli.leaderboard.session.resolve_dataset_class",
            return_value=fake_ds_class,
        ):
            result = runner._resolve_task_dataset(
                {"dataset": {"class": "FakeDS", "path": "/data"}}, "t1"
            )
        fake_ds_class.assert_called_once_with("/data")
        assert result is fake_ds_instance

    def test_error_cases(self):
        # (datasets_map, task_cfg, expected_error_pattern)
        cases = [
            # Explicit dataset reference does not exist.
            (
                {"my_ds": MagicMock()},
                {"dataset": "bad_ref"},
                "unknown dataset",
            ),
            # Inline dataset config must include class.
            (
                {},
                {"dataset": {"path": "/data"}},
                "requires 'class' field",
            ),
            # Dataset field must be str ref or inline dict.
            (
                {},
                {"dataset": 42},
                "string reference or inline",
            ),
        ]
        for datasets, task_cfg, error_match in cases:
            runner = self._make_runner(datasets)
            with pytest.raises(ValueError, match=error_match):
                runner._resolve_task_dataset(task_cfg, "t1")


# ===================================================================
# End-to-end: YAML config → EvalSession → report
# ===================================================================
class TestEvalSessionE2E:
    """Full pipeline test: write a YAML config to disk, run EvalSession.arun()."""

    @pytest.mark.anyio
    async def test_single_task_yaml_e2e(self, tmp_path):
        """A minimal YAML config with one task should produce a correct report."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}

tasks:
  math_eval:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""".format(result_dir=str(tmp_path / "yaml_results"))

        config_path = _write_yaml_config(tmp_path, "test_config.yaml", yaml_content)
        task_runner = _prepare_eval_session(
            config_path,
            models={"mock_model": MockChatModel()},
        )
        results = await task_runner.arun()

        assert "math_eval" in results
        report = results["math_eval"]
        assert report is not None
        assert report["total"] == 3
        # MockChatModel default_answer="unknown" → answers won't match → accuracy=0.0
        assert report["accuracy"] == 0.0

    @pytest.mark.anyio
    async def test_yaml_resume_override(self, tmp_path):
        """The resume CLI flag should set auto_resume on all tasks."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}

tasks:
  resume_eval:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""".format(result_dir=str(tmp_path / "yaml_resume"))

        config_path = _write_yaml_config(tmp_path, "resume_config.yaml", yaml_content)

        # First run
        task_runner1 = _prepare_eval_session(
            config_path,
            models={"mock_model": MockChatModel()},
        )
        results1 = await task_runner1.arun()
        assert results1["resume_eval"]["total"] == 3

        # Second run with resume=True
        task_runner2 = _prepare_eval_session(
            config_path,
            models={"mock_model": MockChatModel()},
            resume=True,
        )
        results2 = await task_runner2.arun()

        assert results2["resume_eval"] == results1["resume_eval"]

    @pytest.mark.anyio
    async def test_yaml_model_derivation(self, tmp_path):
        """Derived models with concurrency_limit should work via YAML."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  base_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"
      concurrency_limit: 128
  child_model:
    base: base_model
    args:
      concurrency_limit: 32

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}

tasks:
  derived_eval:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: child_model
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""".format(result_dir=str(tmp_path / "yaml_derived"))

        config_path = _write_yaml_config(tmp_path, "derived_config.yaml", yaml_content)

        # Patch _setup_models to use MockChatModel instead of real ChatModel
        mock_base = MockChatModel(concurrency_limit=128)
        mock_child = mock_base.with_args(concurrency_limit=32)
        task_runner = _prepare_eval_session(
            config_path,
            models={"base_model": mock_base, "child_model": mock_child},
        )
        results = await task_runner.arun()

        assert "derived_eval" in results
        assert results["derived_eval"]["total"] == 3
        # Verify the child model has the expected concurrency structure
        assert mock_child._parent_limiter is mock_base._limiter

    @pytest.mark.anyio
    async def test_yaml_dataset_operations(self, tmp_path):
        """Dataset operations (shuffle, slice) should be applied from YAML."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}
    operations:
      - shuffle: {{seed: 42}}
      - slice: {{num: 2}}

tasks:
  ops_eval:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""".format(result_dir=str(tmp_path / "yaml_ops"))

        config_path = _write_yaml_config(tmp_path, "ops_config.yaml", yaml_content)
        task_runner = _prepare_eval_session(
            config_path,
            models={"mock_model": MockChatModel()},
        )
        results = await task_runner.arun()

        assert "ops_eval" in results
        # slice num=2 should reduce dataset to 2 samples
        assert results["ops_eval"]["total"] == 2

    @pytest.mark.anyio
    async def test_yaml_multi_task(self, tmp_path):
        """Multiple tasks in one YAML config should all run."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}

runner_config:
  show_progress: false
  detect_anomalies: false
  profile_io: false
  profile_stages: false
  profile_usage: false
  dump_progress: false

tasks:
  eval_a:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
  eval_b:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
""".format(result_dir=str(tmp_path / "yaml_multi"))

        config_path = _write_yaml_config(tmp_path, "multi_config.yaml", yaml_content)
        task_runner = _prepare_eval_session(
            config_path,
            models={"mock_model": MockChatModel()},
        )
        results = await task_runner.arun()

        assert "eval_a" in results
        assert "eval_b" in results
        assert results["eval_a"]["total"] == 3
        assert results["eval_b"]["total"] == 3


# ===================================================================
# EvalSession.arun(): full _prepare_execution chain
# ===================================================================
class TestEvalSessionArun:
    """Test that EvalSession.arun() walks the full _prepare_execution pipeline."""

    @pytest.mark.anyio
    async def test_arun_full_chain(self, tmp_path):
        """EvalSession.arun() should load config, set up all components, and run."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}

tasks:
  chain_eval:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""".format(result_dir=str(tmp_path / "arun_results"))

        config_path = tmp_path / "arun_config.yaml"
        config_path.write_text(yaml_content)

        # Use arun() — goes through _prepare_execution → _init_runner → setup*
        from sieval.cli.leaderboard.session import arun_session

        results = await arun_session(config_path)

        assert "chain_eval" in results
        assert results["chain_eval"]["total"] == 3


class TestEvalSessionConfigLoading:
    def test_init_rejects_non_dict_top_level_yaml(self, tmp_path):
        config_path = tmp_path / "bad_root.yaml"
        config_path.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(
            ValueError, match="Top-level YAML config must be a dictionary"
        ):
            EvalSession(config_path=str(config_path))

    def test_init_treats_null_top_level_as_empty_dict(self, tmp_path):
        config_path = tmp_path / "null_root.yaml"
        config_path.write_text("null\n", encoding="utf-8")

        runner = EvalSession(config_path=str(config_path))

        assert runner.config == {}
        assert runner.runner is None

    def test_init_runner_forwards_result_dir_and_limits(self, tmp_path):
        runner = object.__new__(EvalSession)
        runner.config = {
            "result_dir": str(tmp_path / "from_config"),
            "concurrency_limit": 7,
            "concurrency_limits": {"infer": 3},
        }
        runner.result_dir_override = str(tmp_path / "from_override")
        runner.deterministic = False
        runner.runner = None

        with patch(
            "sieval.cli.leaderboard.session.MultiTaskRunner"
        ) as multi_runner_cls:
            runner._init_runner()

        multi_runner_cls.assert_called_once_with(
            result_dir=str(tmp_path / "from_override"),
            concurrency_limit=7,
            concurrency_limits={"infer": 3},
            deterministic=False,
        )

    def test_init_runner_forwards_deterministic(self, tmp_path):
        runner = object.__new__(EvalSession)
        runner.config = {"result_dir": str(tmp_path / "out")}
        runner.result_dir_override = None
        runner.deterministic = True
        runner.runner = None

        with patch(
            "sieval.cli.leaderboard.session.MultiTaskRunner"
        ) as multi_runner_cls:
            runner._init_runner()

        assert multi_runner_cls.call_args.kwargs["deterministic"] is True


# ===================================================================
# _setup_models: base model creation, derived model creation,
# type conversion, api_key/api_base injection, missing name error
# ===================================================================
class TestSetupModels:
    """Test _setup_models for base model creation, derivation, type conversion."""

    def _make_runner(self, config: dict) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = config
        runner.model_override = None
        runner.resume_override = False
        runner.deterministic = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = None
        return runner

    def test_models_not_dict_raises(self):
        runner = self._make_runner({"models": None})
        with pytest.raises(ValueError, match="must be a dictionary"):
            runner._setup_models()

    def test_model_item_not_dict_raises(self):
        runner = self._make_runner({"models": {"m1": "not-a-dict"}})
        with pytest.raises(
            ValueError, match="'models.m1' configuration must be a dictionary"
        ):
            runner._setup_models()

    def test_base_model_validation_errors(self):
        """Base model setup should reject missing name and invalid type."""
        # (yaml_config, expected_error_pattern)
        cases = [
            # Base model requires name when no override is given.
            (
                {"models": {"m1": {"args": {}}}},
                "requires 'name'",
            ),
            # Base model type must be one of the supported families.
            (
                {"models": {"m1": {"name": "mock-model", "type": "unknown_type"}}},
                "invalid type",
            ),
        ]
        for config, error_match in cases:
            runner = self._make_runner(config)
            with pytest.raises(ValueError, match=error_match):
                runner._setup_models()

    def test_base_model_args_type_error_mentions_model_name(self):
        runner = self._make_runner(
            {"models": {"m1": {"name": "mock-model", "type": "chat", "args": 123}}}
        )
        with pytest.raises(ValueError, match="Model 'm1' args must be a dictionary"):
            runner._setup_models()

    def test_model_override_replaces_name(self):
        """model_override should replace the 'name' field for base models."""
        runner = self._make_runner(
            {"models": {"m1": {"type": "chat", "args": {"api_key": "fake"}}}}
        )
        runner.model_override = "override-model-name"

        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()
            # The first positional/keyword argument must be the overridden name
            call_kwargs = MockCls.call_args
            assert call_kwargs is not None
            # model kwarg should use override
            assert call_kwargs.kwargs.get("model") == "override-model-name" or (
                call_kwargs.args and call_kwargs.args[0] == "override-model-name"
            )

    def test_base_model_infer_uses_model_key_and_forwards_model_name(self):
        runner = self._make_runner(
            {"models": {"m1": {"name": "mock-gen", "args": {"temperature": 0.1}}}}
        )
        created_model = MagicMock()

        with (
            patch.object(runner, "_infer_model_type", return_value="gen") as infer_mock,
            patch(
                "sieval.cli.leaderboard.session.GenModel",
                return_value=created_model,
            ) as gen_cls,
        ):
            runner._setup_models()

        infer_mock.assert_called_once_with("m1", None)
        gen_cls.assert_called_once_with(model="mock-gen", temperature=0.1)
        assert runner.models["m1"] is created_model

    def test_derived_model_no_type_conversion(self):
        """A derived model without 'type' should call with_args (no type conversion)."""
        base = MockChatModel(concurrency_limit=64)
        runner = self._make_runner(
            {
                "models": {
                    "base": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"api_key": "fake"},
                    },
                    "child": {"base": "base", "args": {"concurrency_limit": 32}},
                }
            }
        )
        # Inject pre-built base model to avoid real ChatModel creation
        runner.models["base"] = base

        # Patch the first-pass loop to skip re-creating 'base'
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=base,
        ):
            runner._setup_models()

        child = runner.models.get("child")
        assert child is not None
        # child should share parent_limiter = base._limiter
        assert child._parent_limiter is base._limiter

    def test_derived_model_with_type_conversion(self):
        """A derived model with 'type' should call as_type on the base."""
        base = MockChatModel(concurrency_limit=64)
        runner = self._make_runner(
            {
                "models": {
                    "base": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"api_key": "fake"},
                    },
                    "child": {"base": "base", "type": "chat", "args": {}},
                }
            }
        )
        runner.models["base"] = base

        mock_converted = MagicMock()
        mock_converted.with_args.return_value = mock_converted

        with (
            patch.object(base, "as_type", return_value=mock_converted) as mock_as_type,
            patch(
                "sieval.cli.leaderboard.session.ChatModel",
                return_value=base,
            ) as mock_chat_cls,
        ):
            runner._setup_models()

        mock_as_type.assert_called_once_with(mock_chat_cls)
        assert runner.models["child"] is mock_converted

    def test_derived_model_with_gen_type_conversion(self):
        """Derived model with type='gen' should request GenModel conversion."""
        base = MockChatModel(concurrency_limit=64)
        runner = self._make_runner(
            {
                "models": {
                    "base": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"api_key": "fake"},
                    },
                    "child": {"base": "base", "type": "gen", "args": {}},
                }
            }
        )
        runner.models["base"] = base

        mock_converted = MagicMock()
        mock_converted.with_args.return_value = mock_converted

        with (
            patch.object(base, "as_type", return_value=mock_converted) as mock_as_type,
            patch(
                "sieval.cli.leaderboard.session.ChatModel",
                return_value=base,
            ),
        ):
            runner._setup_models()

        from sieval.core.models.gen_model import GenModel

        mock_as_type.assert_called_once_with(GenModel)
        assert runner.models["child"] is mock_converted

    def test_derived_model_validation_errors(self):
        """Derived model should reject unknown base and invalid type."""
        runner = self._make_runner(
            {"models": {"child": {"base": "non_existent_base", "args": {}}}}
        )
        with pytest.raises(ValueError, match="unknown base model"):
            runner._setup_models()

        base = MockChatModel()
        runner = self._make_runner(
            {
                "models": {
                    "base": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"api_key": "fake"},
                    },
                    "child": {"base": "base", "type": "bad_type", "args": {}},
                }
            }
        )
        runner.models["base"] = base

        with (
            patch(
                "sieval.cli.leaderboard.session.ChatModel",
                return_value=base,
            ),
            pytest.raises(ValueError, match="invalid type"),
        ):
            runner._setup_models()

    def test_derived_model_non_string_base_raises_even_if_key_exists(self):
        """Non-string base should fail validation before base lookup."""
        base = MockChatModel()
        runner = self._make_runner(
            {"models": {"child": {"base": 123, "args": {"temperature": 0.2}}}}
        )
        runner.models[123] = base  # type: ignore[index]

        with pytest.raises(ValueError, match="invalid 'base' value"):
            runner._setup_models()

    def test_derived_model_out_of_order_definition(self):
        """Derived models should resolve even if YAML order is child-before-parent."""
        base = MockChatModel(concurrency_limit=128)
        runner = self._make_runner(
            {
                "models": {
                    "child": {"base": "mid", "args": {"temperature": 0.2}},
                    "mid": {"base": "base", "args": {"concurrency_limit": 32}},
                    "base": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"api_key": "fake", "concurrency_limit": 128},
                    },
                }
            }
        )

        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=base,
        ):
            runner._setup_models()

        assert runner.models["mid"]._parent_limiter is base._limiter
        assert runner.models["child"]._limiter is runner.models["mid"]._limiter
        assert runner.models["child"]._kwargs["temperature"] == 0.2

    def test_derived_model_cycle_raises(self):
        runner = self._make_runner(
            {
                "models": {
                    "model_a": {"base": "model_b", "args": {}},
                    "model_b": {"base": "model_a", "args": {}},
                }
            }
        )

        with pytest.raises(ValueError, match="cyclic dependencies"):
            runner._setup_models()

    def test_base_model_created_by_type(self):
        """Base model should instantiate expected concrete model type."""
        from sieval.core.models.chat_model import ChatModel as RealChatModel
        from sieval.core.models.gen_model import GenModel as RealGenModel

        # (model_type_literal, model_name, expected_concrete_class)
        cases = [
            ("chat", "mock-chat", RealChatModel),
            ("gen", "mock-gen", RealGenModel),
        ]
        for model_type, model_name, expected_cls in cases:
            runner = self._make_runner(
                {
                    "models": {
                        "m1": {
                            "name": model_name,
                            "type": model_type,
                            "args": {"api_key": "fake"},
                        }
                    }
                }
            )
            runner._setup_models()
            assert "m1" in runner.models
            assert isinstance(runner.models["m1"], expected_cls)


class TestSetupModelsApiKeyInjection:
    """Test top-level api_key/api_base handling in model setup."""

    def _make_runner(self, config: dict) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = config
        runner.model_override = None
        runner.resume_override = False
        runner.deterministic = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = None
        return runner

    def test_top_level_api_key_forwarded_for_base_model(self):
        """Top-level api_key/api_base should be forwarded for base models."""
        runner = self._make_runner(
            {
                "models": {
                    "m1": {
                        "name": "mock-chat",
                        "type": "chat",
                        "api_key": "top-level-key",
                        "api_base": "https://custom.endpoint/v1",
                        "args": {},
                    }
                }
            }
        )

        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()

        call_kwargs = MockCls.call_args.kwargs
        assert call_kwargs.get("api_key") == "top-level-key"
        assert call_kwargs.get("api_base") == "https://custom.endpoint/v1"

    @pytest.mark.parametrize(
        "override_field,override_value",
        [
            ("api_key", "child-key"),
            ("api_base", "https://child.endpoint/v1"),
        ],
    )
    def test_derived_model_api_override_rejected(
        self, override_field: str, override_value: str
    ):
        """Derived models cannot override api_key/api_base from base model."""
        base = MockChatModel()
        runner = self._make_runner(
            {
                "models": {
                    "child": {
                        "base": "base",
                        override_field: override_value,
                        "args": {"temperature": 0.3},
                    },
                }
            }
        )
        runner.models["base"] = base

        with pytest.raises(ValueError, match="cannot override api_key/api_base"):
            runner._setup_models()


# ===================================================================
# _check_over_subscription: warns when child quotas exceed base quota
# ===================================================================
class TestCheckOverSubscription:
    """Test _check_over_subscription warns when children exceed base quota."""

    def _make_runner(self) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = {}
        runner.model_override = None
        runner.resume_override = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = None
        return runner

    def test_no_warning_paths(self):
        """No warning when not oversubscribed or when base has no limiter."""
        runner = self._make_runner()

        # No children
        base_no_children = MockChatModel(concurrency_limit=100)
        runner.models = {"base": base_no_children}
        runner._check_over_subscription()  # must not raise

        # Children within quota
        base = MockChatModel(concurrency_limit=100)
        child = base.with_args(concurrency_limit=50)
        runner.models = {"base": base, "child": child}
        with patch("sieval.cli.leaderboard.session.logger") as mock_logger:
            runner._check_over_subscription()
            mock_logger.warning.assert_not_called()
        base = MockChatModel()  # no concurrency_limit -> _limiter is None
        runner.models = {"base": base}
        runner._check_over_subscription()

    def test_warning_when_children_exceed_quota(self):
        """A warning should be logged when total child quotas exceed base quota."""
        runner = self._make_runner()
        base = MockChatModel(concurrency_limit=50)
        child1 = base.with_args(concurrency_limit=40)
        child2 = base.with_args(concurrency_limit=30)
        runner.models = {"base": base, "child1": child1, "child2": child2}

        with patch("sieval.cli.leaderboard.session.logger") as mock_logger:
            runner._check_over_subscription()
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args.args
            # Lazy formatting: template is args[0], values follow
            template = call_args[0]
            assert "Over-subscription" in template
            # base_name is the second arg
            assert call_args[1] == "base"
            # child_info is the 4th arg (base_quota, total_reserved, child_info)
            child_info = call_args[4]
            assert "child1=40" in child_info
            assert "child2=30" in child_info


# ===================================================================
# _build_runner_config: resume_override, defaults merge, field filter
# ===================================================================
class TestBuildRunnerConfigFull:
    """Test _build_runner_config: resume_override, defaults merge, field filter."""

    def _make_runner(self, resume_override: bool = False) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = {}
        runner.config_path = Path("test.yaml")
        runner.model_override = None
        runner.resume_override = resume_override
        runner.deterministic = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = None
        return runner

    # (resume_override, defaults, task_cfg, expected_auto_resume)
    @pytest.mark.parametrize(
        "resume_override,defaults,task_cfg,expected_auto_resume",
        [
            (True, {}, {}, True),
            (False, {}, {}, False),
        ],
    )
    def test_auto_resume_resolution(
        self, resume_override, defaults, task_cfg, expected_auto_resume
    ):
        runner = self._make_runner(resume_override=resume_override)
        cfg = runner._build_runner_config(task_cfg, defaults)
        assert cfg.auto_resume is expected_auto_resume

    # (defaults, task_cfg, expected_concurrency_limit, expected_show_progress)
    @pytest.mark.parametrize(
        "defaults,task_cfg,expected_limit,expected_show_progress",
        [
            (
                {"concurrency_limit": 42, "show_progress": False},
                {},
                42,
                False,
            ),
            (
                {"concurrency_limit": 10, "show_progress": True},
                {"runner_config": {"concurrency_limit": 99}},
                99,
                True,
            ),
        ],
    )
    def test_defaults_and_task_override_merge(
        self, defaults, task_cfg, expected_limit, expected_show_progress
    ):
        runner = self._make_runner()
        cfg = runner._build_runner_config(task_cfg, defaults)
        assert cfg.concurrency_limit == expected_limit
        assert cfg.show_progress is expected_show_progress

    def test_field_filter_and_empty_defaults(self):
        """Unknown fields are dropped; empty inputs still use TaskRunner defaults."""
        runner = self._make_runner()
        defaults = {"nonexistent_field": "should_be_dropped", "show_progress": False}
        cfg = runner._build_runner_config({}, defaults)
        assert not hasattr(cfg, "nonexistent_field")
        assert cfg.show_progress is False

        from sieval.core.runners.runner import TaskRunnerConfig

        cfg = self._make_runner()._build_runner_config({}, {})
        expected = TaskRunnerConfig()
        assert cfg.record_each_stage == expected.record_each_stage
        assert cfg.max_iterations == expected.max_iterations


# ===================================================================
# _setup_tasks: errors for missing class/task fields and non-dict tasks
# ===================================================================
class TestSetupTasksErrors:
    """Test _setup_tasks raises for missing class/task fields and non-dict tasks."""

    def _make_runner(self, tasks_cfg, models=None, datasets=None) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = {
            "tasks": tasks_cfg,
            "runner_config": {
                "show_progress": False,
                "detect_anomalies": False,
                "profile_io": False,
                "profile_stages": False,
                "profile_usage": False,
                "dump_progress": False,
            },
        }
        runner.config_path = Path("test.yaml")
        runner.model_override = None
        runner.resume_override = False
        runner.deterministic = False
        runner.models = models or {}
        runner.datasets = datasets or {}
        runner.runner = MultiTaskRunner()
        return runner

    def test_tasks_not_dict_raises(self):
        """When 'tasks' is a list instead of dict, ValueError should be raised."""
        runner = self._make_runner([{"class": "some.Task"}])
        with pytest.raises(ValueError, match="must be a dictionary"):
            runner._setup_tasks()

    def test_task_item_not_dict_raises(self):
        runner = self._make_runner({"bad_task": "not-a-dict"})
        with pytest.raises(
            ValueError, match="'tasks.bad_task' configuration must be a dictionary"
        ):
            runner._setup_tasks()

    def test_runner_config_not_dict_raises(self):
        runner = self._make_runner({})
        object.__setattr__(runner, "config", {"tasks": {}, "runner_config": []})
        with pytest.raises(
            ValueError, match="'runner_config' configuration must be a dictionary"
        ):
            runner._setup_tasks()

    def test_task_missing_class_field_raises(self):
        """A task without 'class' should raise ValueError."""
        runner = self._make_runner({"my_task": {"dataset": "ds", "model": "m"}})
        with pytest.raises(ValueError, match="requires 'class' field"):
            runner._setup_tasks()

    def test_tasks_is_none_treated_as_empty(self):
        """When 'tasks' key is absent, no iteration occurs and runner is untouched."""
        runner = object.__new__(EvalSession)
        runner.config = {"runner_config": {}}
        runner.config_path = Path("test.yaml")
        runner.model_override = None
        runner.resume_override = False
        runner.deterministic = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = MultiTaskRunner()
        add_task_mock = MagicMock()
        with patch.object(runner.runner, "add_task", add_task_mock):
            # Missing "tasks" should behave like an empty task mapping.
            runner._setup_tasks()
        add_task_mock.assert_not_called()


# ===================================================================
# _setup_datasets: errors for missing class field
# ===================================================================
class TestSetupDatasetsErrors:
    """Test _setup_datasets raises for missing class field."""

    def _make_runner(self, datasets_cfg: object) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = {"datasets": datasets_cfg}
        runner.model_override = None
        runner.resume_override = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = None
        return runner

    def test_datasets_not_dict_raises(self):
        runner = self._make_runner(None)
        with pytest.raises(ValueError, match="must be a dictionary"):
            runner._setup_datasets()

    def test_dataset_item_not_dict_raises(self):
        runner = self._make_runner({"my_ds": "not-a-dict"})
        with pytest.raises(
            ValueError, match="'datasets.my_ds' configuration must be a dictionary"
        ):
            runner._setup_datasets()

    # dataset config variants that should be rejected as missing/invalid class
    @pytest.mark.parametrize(
        "dataset_cfg",
        [
            {"my_ds": {"path": "/some/path"}},
            {"my_ds": {"class": None}},
        ],
    )
    def test_missing_or_empty_class_field_raises(self, dataset_cfg):
        """A dataset without a valid 'class' key should raise ValueError."""
        runner = self._make_runner(dataset_cfg)
        with pytest.raises(ValueError, match="requires 'class' field"):
            runner._setup_datasets()

    def test_valid_class_field_resolves(self):
        """A dataset with a valid 'class' path should be instantiated correctly."""
        runner = self._make_runner(
            {"mock_ds": {"class": "tests.conftest.MockDataset", "args": {}}}
        )
        runner._setup_datasets()
        assert "mock_ds" in runner.datasets


# ===================================================================
# _infer_model_type: skips unresolvable task classes
# ===================================================================
class TestInferModelTypeSkipsUnresolvable:
    """Test that ImportError/AttributeError during task class resolution is skipped."""

    def _make_runner(self, tasks_cfg=None) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = {"tasks": tasks_cfg or {}}
        return runner

    # (task_spec, resolution_error_to_simulate)
    @pytest.mark.parametrize(
        "task_spec,error",
        [
            (
                "nonexistent.module.SomeTask",
                ImportError("module not found"),
            ),
            (
                "sieval.core.models.model.NoSuchClass",
                AttributeError("class not found"),
            ),
        ],
    )
    def test_unresolvable_task_class_is_skipped(self, task_spec, error):
        """Import/attribute errors in class resolution should default to chat."""
        tasks_cfg = {"t1": {"model": "m", "class": task_spec}}
        runner = self._make_runner(tasks_cfg)
        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            side_effect=error,
        ):
            result = runner._infer_model_type("m", None)
        assert result == "chat"

    def test_unresolvable_task_does_not_block_resolvable_task(self):
        """
        When one task is unresolvable and another resolves to 'gen', 'gen' is returned.
        """

        class GenTask:
            model_type = "gen"

        tasks_cfg = {
            "bad_task": {"model": "m", "class": "bad.module.BadTask"},
            "good_task": {"model": "m", "class": "good.module.GenTask"},
        }
        runner = self._make_runner(tasks_cfg)

        call_count = 0

        def mock_resolve(_spec):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ImportError("bad module")
            return GenTask

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            side_effect=mock_resolve,
        ):
            result = runner._infer_model_type("m", None)

        assert result == "gen"


class TestEvalSessionWrappers:
    @pytest.mark.anyio
    async def test_arun_session_delegates_to_runner(self):
        fake_arun = AsyncMock(return_value={"task_a": {"ok": True}})
        fake_runner = types.SimpleNamespace(arun=fake_arun)

        with patch(
            "sieval.cli.leaderboard.session.EvalSession",
            return_value=fake_runner,
        ) as eval_session_cls:
            result = await arun_session(
                "cfg.yaml",
                model="model-1",
                resume=True,
                result_dir="out_dir",
            )

        eval_session_cls.assert_called_once_with(
            config_path="cfg.yaml",
            model_override="model-1",
            resume=True,
            result_dir_override="out_dir",
            deterministic_override=None,
            endpoint_map=None,
            infer_plans=None,
            invocation=None,
            self_managed_endpoints=frozenset(),
        )
        assert result == {"task_a": {"ok": True}}

    @pytest.mark.anyio
    async def test_arun_session_passes_deterministic(self):
        fake_arun = AsyncMock(return_value={"task_a": {"ok": True}})
        fake_runner = types.SimpleNamespace(arun=fake_arun)

        with patch(
            "sieval.cli.leaderboard.session.EvalSession",
            return_value=fake_runner,
        ) as eval_session_cls:
            result = await arun_session(
                "cfg.yaml",
                model="model-1",
                resume=True,
                result_dir="out_dir",
                deterministic=True,
            )

        eval_session_cls.assert_called_once_with(
            config_path="cfg.yaml",
            model_override="model-1",
            resume=True,
            result_dir_override="out_dir",
            deterministic_override=True,
            endpoint_map=None,
            infer_plans=None,
            invocation=None,
            self_managed_endpoints=frozenset(),
        )
        assert result == {"task_a": {"ok": True}}

    def test_run_session_delegates_to_anyio_run(self):
        """run_session forwards all args (incl. deterministic) positionally to
        anyio.run(arun_session, ...)."""
        with patch("sieval.cli.leaderboard.session.anyio.run") as run_mock:
            run_mock.return_value = {"task_b": {"ok": True}}
            result = run_session(
                "cfg.yaml",
                model="m1",
                resume=True,
                result_dir="out_dir",
                deterministic=True,
            )

        run_mock.assert_called_once_with(
            arun_session,
            "cfg.yaml",
            "m1",
            True,
            "out_dir",
            True,
            None,
            None,
            None,
            frozenset(),
        )
        assert result == {"task_b": {"ok": True}}

    @pytest.mark.anyio
    async def test_arun_session_forwards_endpoint_map_and_infer_plans(self):
        fake_arun = AsyncMock(return_value={"t": {}})
        fake_runner = types.SimpleNamespace(arun=fake_arun)
        endpoint_map = {"m": "http://host:8000/v1"}
        plans = {"m": {"backend": "vllm"}}

        with patch(
            "sieval.cli.leaderboard.session.EvalSession",
            return_value=fake_runner,
        ) as EvalSessionCls:
            await arun_session(
                "cfg.yaml",
                endpoint_map=endpoint_map,
                infer_plans=plans,
            )

        EvalSessionCls.assert_called_once_with(
            config_path="cfg.yaml",
            model_override=None,
            resume=False,
            result_dir_override=None,
            deterministic_override=None,
            endpoint_map=endpoint_map,
            infer_plans=plans,
            invocation=None,
            self_managed_endpoints=frozenset(),
        )

    def test_run_session_forwards_endpoint_map_and_infer_plans_positionally(self):
        with patch("sieval.cli.leaderboard.session.anyio.run") as run_mock:
            run_mock.return_value = {}
            endpoint_map = {"m": "http://host:8000/v1"}
            plans = {"m": {"backend": "vllm"}}
            run_session(
                "cfg.yaml",
                endpoint_map=endpoint_map,
                infer_plans=plans,
            )

        run_mock.assert_called_once_with(
            arun_session,
            "cfg.yaml",
            None,
            False,
            None,
            None,
            endpoint_map,
            plans,
            None,
            frozenset(),
        )

    def test_eval_session_run_calls_anyio_run(self):
        runner = object.__new__(EvalSession)
        runner.arun = AsyncMock(return_value={"task_c": {"ok": True}})

        with patch(
            "sieval.cli.leaderboard.session.anyio.run",
            return_value={"task_c": {"ok": True}},
        ) as run_mock:
            result = EvalSession.run(runner)

        run_mock.assert_called_once_with(runner.arun)
        assert result == {"task_c": {"ok": True}}

    @pytest.mark.anyio
    async def test_arun_raises_when_runner_missing_after_prepare(self):
        runner = object.__new__(EvalSession)
        runner.runner = None
        with (
            patch.object(runner, "_prepare_execution", AsyncMock(return_value=None)),
            patch.object(
                runner, "_persist_effective_config", AsyncMock(return_value=None)
            ),
            patch.object(runner, "_persist_infer_plans", AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="Runner not initialized"),
        ):
            await runner.arun()


# ===================================================================
# resolve_task_class: submodule search by naming convention
# ===================================================================


def _find_project_root() -> Path:
    """Find repo root: parent of the `sieval` package used in tests."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "sieval" / "tasks" / "aime_2024_0shot_gen.py").exists():
            return p  # p is parent of sieval package, i.e. repo root
        p = p.parent
    return Path(__file__).resolve().parents[4]


@contextmanager
def _project_sieval_first():
    """
    Ensure sieval.tasks is resolved from project root (not mutants).
    Under mutmut, resolve_task_class runs from mutants; import_module() uses
    sys.modules first, so if sieval/sieval.tasks are already loaded from mutants,
    we must clear them and prepend project root to sys.path so the next import
    re-resolves from the real package.
    """
    root = _find_project_root()
    inserted = str(root) not in sys.path[:1]
    if inserted:
        sys.path.insert(0, str(root))

    # Clear sieval* from sys.modules so import_module() re-resolves via sys.path.
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "sieval" or k.startswith("sieval.")
    }
    try:
        yield
    finally:
        if inserted and sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
        # Evict any sieval* modules loaded during the yield — otherwise they
        # shadow the originals (e.g. task modules that re-registered into a
        # transient TASK_REGISTRY) for subsequent tests.
        for k in [
            k
            for k in list(sys.modules)
            if (k == "sieval" or k.startswith("sieval.")) and k not in saved
        ]:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v


class TestResolveTaskClass:
    """
    Test resolve_task_class submodule search path (lines 141-163 in session.py).
    """

    def test_full_path_and_short_name_resolve(self):
        """Both full-path and short-name task specs should resolve."""
        from sieval.cli.leaderboard.session import resolve_task_class

        with _project_sieval_first():
            full_path_cls = resolve_task_class(
                "sieval.tasks.aime_2024_0shot_gen.AIME2024ZeroShotGenTask"
            )
            from sieval.tasks.aime_2024_0shot_gen import AIME2024ZeroShotGenTask

            short_name_cls = resolve_task_class("AIME2024ZeroShotGenTask")
        assert full_path_cls is AIME2024ZeroShotGenTask
        assert short_name_cls is AIME2024ZeroShotGenTask

    def test_unknown_class_raises_import_error(self):
        """A class that cannot be found should raise ImportError."""
        from sieval.cli.leaderboard.session import resolve_task_class

        with pytest.raises(ImportError, match="Could not find task class"):
            resolve_task_class("NonExistentTask12345")

    def test_short_name_resolves_when_exported_in_tasks_init(self):
        """Short-name resolution should work for tasks exported in sieval.tasks."""
        from sieval.cli.leaderboard.session import resolve_task_class

        with _project_sieval_first():
            from sieval.tasks import AIME2024ZeroShotGenTask

            resolved = resolve_task_class("AIME2024ZeroShotGenTask")
        assert resolved is AIME2024ZeroShotGenTask

    def test_short_name_resolves_via_submodule_when_tasks_import_fails(self):
        """If importing `sieval.tasks` fails, submodule search should still resolve."""
        from sieval.cli.leaderboard.session import resolve_task_class

        target_cls = type("SyntheticTask", (), {})
        fake_module = types.SimpleNamespace(SyntheticTask=target_cls)

        def _fake_import_module(module_name: str):
            if module_name == "sieval.tasks":
                raise ModuleNotFoundError(
                    "No module named 'sieval.tasks'",
                    name="sieval.tasks",
                )
            if module_name == "sieval.tasks.synthetic_task":
                return fake_module
            raise ModuleNotFoundError(
                f"No module named '{module_name}'", name=module_name
            )

        with (
            patch(
                "sieval.cli.leaderboard.session._guess_submodule_names",
                return_value=["synthetic_task"],
            ),
            patch(
                "sieval.cli.leaderboard.session.importlib.import_module",
                side_effect=_fake_import_module,
            ),
        ):
            resolved = resolve_task_class("SyntheticTask")

        assert resolved is target_cls

    def test_short_name_propagates_missing_dependency_error(self):
        """If a task module exists but has a missing dependency, propagate the error."""
        from sieval.cli.leaderboard.session import resolve_task_class

        def _fake_import_module(module_name: str):
            if module_name == "sieval.tasks":
                # sieval.tasks loads, but __getattr__ triggers import of the
                # task module which fails due to a missing dependency.
                raise ModuleNotFoundError("No module named 'scipy'", name="scipy")
            raise ModuleNotFoundError(
                f"No module named '{module_name}'", name=module_name
            )

        with (
            patch(
                "sieval.cli.leaderboard.session.importlib.import_module",
                side_effect=_fake_import_module,
            ),
            pytest.raises(ModuleNotFoundError, match="scipy"),
        ):
            resolve_task_class("SomeTask")


# ===================================================================
# infer_args: per-task inference parameter override
# ===================================================================
class TestInferArgs:
    """Test infer_args override mechanism in _setup_tasks()."""

    def _make_runner(self, tasks_cfg, models=None, datasets=None) -> EvalSession:
        runner = object.__new__(EvalSession)
        runner.config = {
            "tasks": tasks_cfg,
            "runner_config": {
                "show_progress": False,
                "detect_anomalies": False,
                "profile_io": False,
                "profile_stages": False,
                "profile_usage": False,
                "dump_progress": False,
            },
        }
        runner.config_path = Path("test.yaml")
        runner.model_override = None
        runner.resume_override = False
        runner.deterministic = False
        runner.models = models or {}
        runner.datasets = datasets or {}
        runner.runner = MultiTaskRunner()
        return runner

    def test_infer_args_override_model_kwargs(self):
        """infer_args should override model's default _kwargs via with_args()."""
        mock_model = MockChatModel(temperature=0.0, max_tokens=16384)
        mock_ds = MagicMock()
        mock_task_cls = MagicMock(return_value=MagicMock())

        runner = self._make_runner(
            {
                "eval_task": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                    "infer_args": {"max_tokens": 512, "temperature": 0.7},
                }
            },
            models={"m": mock_model},
            datasets={"ds": mock_ds},
        )

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()

        # The model passed to task constructor should have overridden _kwargs
        call_kwargs = mock_task_cls.call_args.kwargs
        derived_model = call_kwargs["model"]
        assert derived_model is not mock_model  # should be a new derived model
        assert derived_model._kwargs["max_tokens"] == 512
        assert derived_model._kwargs["temperature"] == 0.7

    def test_infer_args_empty_noop(self):
        """Empty or missing infer_args should not create a model copy."""
        mock_model = MockChatModel(temperature=0.0)
        mock_ds = MagicMock()
        mock_task_cls = MagicMock(return_value=MagicMock())

        # Test with missing infer_args
        runner = self._make_runner(
            {
                "eval_task": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                }
            },
            models={"m": mock_model},
            datasets={"ds": mock_ds},
        )

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()

        call_kwargs = mock_task_cls.call_args.kwargs
        assert call_kwargs["model"] is mock_model  # exact same object, not a copy

    def test_infer_args_nested_extra_reaches_model_extra(self):
        """infer_args nested extra reaches model.extra via with_args."""
        mock_model = MockChatModel(temperature=0.0)
        mock_ds = MagicMock()
        mock_task_cls = MagicMock(return_value=MagicMock())

        wrappers = {"dna": "<dna>{seq}</dna>", "rna": "<rna>{seq}</rna>"}
        runner = self._make_runner(
            {
                "t": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                    "infer_args": {"extra": {"sequence_wrappers": wrappers}},
                }
            },
            models={"m": mock_model},
            datasets={"ds": mock_ds},
        )

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()

        derived_model = mock_task_cls.call_args.kwargs["model"]
        assert derived_model.extra["sequence_wrappers"] == wrappers

    def test_infer_args_empty_explicit_noop(self):
        """Explicit empty infer_args should not create a model copy."""
        mock_model = MockChatModel(temperature=0.0)
        mock_ds = MagicMock()
        mock_task_cls = MagicMock(return_value=MagicMock())

        runner = self._make_runner(
            {
                "eval_task": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                    "infer_args": {},
                }
            },
            models={"m": mock_model},
            datasets={"ds": mock_ds},
        )

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()

        call_kwargs = mock_task_cls.call_args.kwargs
        assert call_kwargs["model"] is mock_model  # exact same object

    def test_infer_args_shared_client_and_limiter(self):
        """Derived model from infer_args should share _client and _limiter."""
        mock_model = MockChatModel(concurrency_limit=128, temperature=0.0)
        mock_ds = MagicMock()
        mock_task_cls = MagicMock(return_value=MagicMock())

        runner = self._make_runner(
            {
                "eval_task": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                    "infer_args": {"max_tokens": 512},
                }
            },
            models={"m": mock_model},
            datasets={"ds": mock_ds},
        )

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()

        call_kwargs = mock_task_cls.call_args.kwargs
        derived_model = call_kwargs["model"]
        # Shared client and limiter (with_args without concurrency_limit)
        assert derived_model._client is mock_model._client
        assert derived_model._limiter is mock_model._limiter
        assert derived_model._parent_limiter is mock_model._parent_limiter
        # But _kwargs differ
        assert derived_model._kwargs["max_tokens"] == 512
        assert derived_model._kwargs["temperature"] == 0.0

    def test_infer_args_e2e_yaml(self, tmp_path):
        """Full YAML E2E: infer_args overrides model defaults in task config."""
        yaml_content = """\
result_dir: "{result_dir}"

models:
  mock_model:
    name: "mock-chat"
    type: "chat"
    args:
      api_key: "fake"

datasets:
  test_ds:
    class: tests.conftest.MockDataset
    args: {{}}

tasks:
  infer_args_eval:
    class: tests.unit.core.runners.test_runner.MockTask
    dataset: test_ds
    model: mock_model
    infer_args:
      max_tokens: 256
      temperature: 0.9
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""".format(result_dir=str(tmp_path / "yaml_infer_args"))

        config_path = _write_yaml_config(
            tmp_path, "infer_args_config.yaml", yaml_content
        )

        # Build a model with known _kwargs so we can verify override
        mock_model = MockChatModel(temperature=0.0, max_tokens=16384)
        task_runner = _prepare_eval_session(
            config_path,
            models={"mock_model": mock_model},
        )

        # Verify the task's model has overridden kwargs
        assert len(task_runner._runners) == 1
        task_runner_entry = task_runner._runners[0]
        task = task_runner_entry._task
        task_model = task.model
        assert task_model is not mock_model  # should be derived
        assert task_model._kwargs["max_tokens"] == 256
        assert task_model._kwargs["temperature"] == 0.9
        # Shared client and limiter
        assert task_model._client is mock_model._client


# ===================================================================
# Deterministic mode: required-seed injection + transparent sampling
# ===================================================================
class TestDeterministicMode:
    """Deterministic mode flag resolution, seed injection, sampling pass-through.

    Deterministic mode injects only `seed` as required. All sampling
    parameters (temperature, top_p, top_k, ...) are transparent — users
    configure them freely and pass@k configurations with temperature > 0
    are first-class.
    """

    def _make_setup_models_runner(self, config: dict) -> EvalSession:
        """Create a minimal EvalSession for _setup_models tests."""
        runner = object.__new__(EvalSession)
        runner.config = config
        runner.model_override = None
        runner.resume_override = False
        runner.models = {}
        runner.datasets = {}
        runner.runner = None
        runner.deterministic = config.get("deterministic", False)
        return runner

    # ------------------------------------------------------------------
    # EvalSession resolves deterministic internally: the kwarg is the
    # override, YAML is the default, monotone OR is the rule. Covers the
    # full truth table here (resolve_deterministic's unit tests still
    # exercise the helper in isolation).
    # ------------------------------------------------------------------

    def test_session_stores_deterministic_true(self, tmp_path):
        """``deterministic_override=True`` forces on even when YAML is empty."""
        config_path = _write_yaml_config(tmp_path, "cfg.yaml", "result_dir: /tmp/x\n")
        session = EvalSession(config_path=str(config_path), deterministic_override=True)
        assert session.deterministic is True

    def test_session_default_deterministic_false(self, tmp_path):
        """Default (``None``) with YAML unset → False."""
        config_path = _write_yaml_config(tmp_path, "cfg.yaml", "result_dir: /tmp/x\n")
        session = EvalSession(config_path=str(config_path))
        assert session.deterministic is False

    def test_session_default_picks_up_yaml_true(self, tmp_path):
        """Default (``None``) defers to YAML — a programmatic caller that
        doesn't explicitly pass ``deterministic`` still gets the YAML
        intent, closing the prior silent-downgrade trap."""
        config_path = _write_yaml_config(
            tmp_path, "cfg.yaml", "deterministic: true\nresult_dir: /tmp/x\n"
        )
        session = EvalSession(config_path=str(config_path))
        assert session.deterministic is True

    def test_session_false_cannot_downgrade_yaml(self, tmp_path):
        """Monotone: explicit ``deterministic_override=False`` is a no-op when YAML
        says ``deterministic: true`` (reproducibility contract wins)."""
        config_path = _write_yaml_config(
            tmp_path, "cfg.yaml", "deterministic: true\nresult_dir: /tmp/x\n"
        )
        session = EvalSession(
            config_path=str(config_path), deterministic_override=False
        )
        assert session.deterministic is True

    # ------------------------------------------------------------------
    # seed: required key, injected if absent, user value preserved
    # ------------------------------------------------------------------

    def test_seed_auto_injected_when_absent(self):
        """Deterministic mode injects seed=0 when user doesn't specify."""
        runner = self._make_setup_models_runner(
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-chat", "type": "chat", "args": {}}},
            }
        )
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()
        assert MockCls.call_args.kwargs.get("seed") == 0

    def test_seed_user_override_preserved(self):
        """User's explicit seed=42 is preserved; no injection override."""
        runner = self._make_setup_models_runner(
            {
                "deterministic": True,
                "models": {
                    "m1": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"seed": 42},
                    }
                },
            }
        )
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()
        assert MockCls.call_args.kwargs.get("seed") == 42

    def test_no_seed_injection_when_not_deterministic(self):
        """seed is NOT auto-injected when deterministic=False."""
        runner = self._make_setup_models_runner(
            {
                "deterministic": False,
                "models": {"m1": {"name": "mock-chat", "type": "chat", "args": {}}},
            }
        )
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()
        assert "seed" not in MockCls.call_args.kwargs

    def test_derived_model_inherits_injected_seed(self):
        """Derived models inherit seed=0 from base via with_args kwarg merge.

        seed is injected only on the first (base) pass; derived models pick it
        up through ``base_model.with_args(**args)`` which merges
        ``{**base._kwargs, **args}``. Uses a real ``MockChatModel`` so the
        merge is exercised rather than mocked.
        """
        runner = self._make_setup_models_runner(
            {
                "deterministic": True,
                "models": {
                    "base": {"name": "mock-chat", "type": "chat", "args": {}},
                    # Non-empty args forces the with_args path (empty args +
                    # no concurrency_limit would just alias base).
                    "child": {"base": "base", "args": {"temperature": 0.7}},
                },
            }
        )
        # MockChatModel hardcodes model/api_key in its super().__init__; drop
        # those here so _setup_models's `model=...` kwarg doesn't collide.
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            side_effect=lambda **kw: MockChatModel(
                **{k: v for k, v in kw.items() if k not in ("model", "api_key")}
            ),
        ):
            runner._setup_models()
        base = runner.models["base"]
        child = runner.models["child"]
        assert base._kwargs.get("seed") == 0
        # Derived model keeps seed=0 from base and picks up its own override.
        assert child._kwargs.get("seed") == 0
        assert child._kwargs.get("temperature") == 0.7

    # ------------------------------------------------------------------
    # Sampling params: transparent — user configures freely, no lock
    # ------------------------------------------------------------------

    def test_temperature_sampling_allowed(self):
        """temperature > 0 is allowed under deterministic mode (seeded sampling)."""
        runner = self._make_setup_models_runner(
            {
                "deterministic": True,
                "models": {
                    "m1": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"temperature": 0.6},
                    }
                },
            }
        )
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()  # must not raise
        call_kwargs = MockCls.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.6
        assert call_kwargs.get("seed") == 0  # still injected

    def test_full_sampling_config_passes_through(self):
        """Full pass@k sampling config (temperature, top_p, top_k) all pass through."""
        runner = self._make_setup_models_runner(
            {
                "deterministic": True,
                "models": {
                    "m1": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "max_tokens": 32768,
                            "frequency_penalty": 0.1,
                        },
                    }
                },
            }
        )
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()
        call_kwargs = MockCls.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.6
        assert call_kwargs.get("top_p") == 0.95
        assert call_kwargs.get("top_k") == 20
        assert call_kwargs.get("max_tokens") == 32768
        assert call_kwargs.get("frequency_penalty") == 0.1
        assert call_kwargs.get("seed") == 0

    def test_no_temperature_injection_under_deterministic(self):
        """Deterministic mode does NOT inject temperature (only seed)."""
        runner = self._make_setup_models_runner(
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-chat", "type": "chat", "args": {}}},
            }
        )
        with patch(
            "sieval.cli.leaderboard.session.ChatModel",
            return_value=MagicMock(),
        ) as MockCls:
            runner._setup_models()
        # temperature left to engine default — not force-injected to 0.0
        assert "temperature" not in MockCls.call_args.kwargs

    def test_per_task_infer_args_temperature_allowed(self):
        """Per-task infer_args with temperature > 0 is accepted (no lock)."""
        mock_model = MagicMock()
        mock_model.with_args.return_value = mock_model
        mock_ds = MagicMock()
        mock_task_cls = MagicMock(return_value=MagicMock())

        runner = object.__new__(EvalSession)
        runner.config = {
            "tasks": {
                "eval_task": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                    "infer_args": {"temperature": 0.6, "top_k": 20},
                }
            },
            "runner_config": {
                "show_progress": False,
                "detect_anomalies": False,
                "profile_io": False,
                "profile_stages": False,
                "profile_usage": False,
                "dump_progress": False,
            },
        }
        runner.config_path = Path("test.yaml")
        runner.model_override = None
        runner.resume_override = False
        runner.models = {"m": mock_model}
        runner.datasets = {"ds": mock_ds}
        runner.runner = MultiTaskRunner()
        runner.deterministic = True

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()  # must not raise
        mock_model.with_args.assert_called_once_with(temperature=0.6, top_k=20)


# ===================================================================
# resolve_deterministic: monotone upper bound helper (shared by _run_all
# and EvalSession.__init__)
# ===================================================================
class TestResolveDeterministic:
    def test_both_false_stays_false(self):
        assert resolve_deterministic(None, {}) is False
        assert resolve_deterministic(False, {"deterministic": False}) is False

    def test_cli_true_wins(self):
        assert resolve_deterministic(True, {}) is True
        assert resolve_deterministic(True, {"deterministic": False}) is True

    def test_yaml_true_wins(self):
        assert resolve_deterministic(None, {"deterministic": True}) is True
        assert resolve_deterministic(False, {"deterministic": True}) is True

    def test_both_true_is_true(self):
        assert resolve_deterministic(True, {"deterministic": True}) is True


# ===================================================================
# Runner field classification: throughput vs strict vs non-match
# ===================================================================
class TestRunnerFieldClassification:
    def test_every_field_classified_exactly_once(self):
        all_fields = set(TaskRunnerConfig.__dataclass_fields__)
        buckets = [
            _THROUGHPUT_RUNNER_KEYS,
            _STRICT_RUNNER_KEYS,
            _NONMATCH_RUNNER_KEYS,
        ]
        union = set().union(*buckets)
        assert union == all_fields, f"unclassified: {all_fields ^ union}"
        # pairwise disjoint
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                assert buckets[i].isdisjoint(buckets[j])


class TestStripNoncomparableFields:
    def test_removes_top_level_concurrency_without_mutating_input(self):
        cfg = {"concurrency_limit": 8, "concurrency_limits": {"infer": 4}, "models": {}}
        out = _strip_noncomparable_fields(cfg)
        assert "concurrency_limit" not in out
        assert "concurrency_limits" not in out
        assert cfg["concurrency_limit"] == 8  # original untouched

    def test_removes_per_model_args_concurrency_only(self):
        cfg = {"models": {"m": {"args": {"concurrency_limit": 64, "temperature": 0.0}}}}
        out = _strip_noncomparable_fields(cfg)
        assert "concurrency_limit" not in out["models"]["m"]["args"]
        assert out["models"]["m"]["args"]["temperature"] == 0.0

    def test_removes_runner_config_throughput_keeps_strict(self):
        cfg = {
            "tasks": {
                "t": {
                    "runner_config": {
                        # Scheduling + console-only → stripped
                        "concurrency_limits": {"infer": 4},
                        "show_progress": False,
                        # Affect on-disk content / result semantics → kept strict
                        "max_retries": 3,
                        "profile_usage": False,
                        "detect_anomalies": False,
                        "dump_progress": False,
                        "shard_samples": 1024,
                        "max_iterations": 5,
                    }
                }
            }
        }
        out = _strip_noncomparable_fields(cfg)
        rc = out["tasks"]["t"]["runner_config"]
        # stripped (adjustable on resume)
        assert "concurrency_limits" not in rc
        assert "show_progress" not in rc
        # kept (must match on resume — touch disk content / failure signal)
        assert rc["max_retries"] == 3
        assert rc["profile_usage"] is False
        assert rc["detect_anomalies"] is False
        assert rc["dump_progress"] is False
        assert rc["shard_samples"] == 1024
        assert rc["max_iterations"] == 5

    def test_removes_top_level_runner_config_throughput_keeps_strict(self):
        # The top-level runner_config defaults block is merged into every task,
        # so it carries the same throughput knobs and must be stripped too.
        cfg = {
            "runner_config": {
                "concurrency_limits": {"infer": 4},
                "write_buffer_size": 64,
                "max_retries": 3,  # strict → kept
            }
        }
        out = _strip_noncomparable_fields(cfg)
        rc = out["runner_config"]
        assert "concurrency_limits" not in rc
        assert "write_buffer_size" not in rc
        assert rc["max_retries"] == 3


# ===================================================================
# Best-effort deterministic warning: fires when the session talks to an
# externally-managed api_base, because sieval can only pin `seed` — it
# cannot verify batch-invariant kernels on the remote engine.
# ===================================================================
@pytest.fixture
def loguru_caplog(caplog):
    """Bridge loguru warnings into pytest's caplog for the test duration."""
    import logging as _logging

    from loguru import logger as _logger

    sink_id = _logger.add(caplog.handler, level="WARNING")
    try:
        with caplog.at_level(_logging.WARNING):
            yield caplog
    finally:
        _logger.remove(sink_id)


class TestBestEffortDeterministicWarning:
    def _session(self, tmp_path, yaml_text: str, **kwargs):
        config_path = _write_yaml_config(tmp_path, "cfg.yaml", yaml_text)
        return EvalSession(config_path=str(config_path), **kwargs)

    def test_external_api_base_under_deterministic_warns(self, tmp_path, loguru_caplog):
        self._session(
            tmp_path,
            "deterministic: true\n"
            "models:\n"
            "  m1:\n"
            "    name: foo\n"
            "    api_base: http://external.example/v1\n",
            deterministic_override=True,
        )
        assert any("best-effort" in rec.message for rec in loguru_caplog.records)
        assert any("m1" in rec.message for rec in loguru_caplog.records)

    def test_self_managed_endpoints_suppress_warning(self, tmp_path, loguru_caplog):
        """api_base is present but sieval launched it → no warning."""
        self._session(
            tmp_path,
            "deterministic: true\n"
            "models:\n"
            "  m1:\n"
            "    name: foo\n"
            "    api_base: http://localhost:8000/v1\n",
            deterministic_override=True,
            self_managed_endpoints=frozenset({"m1"}),
        )
        assert not any("best-effort" in rec.message for rec in loguru_caplog.records)

    def test_no_api_base_no_warning(self, tmp_path, loguru_caplog):
        """Models without api_base aren't reachable, so no warning."""
        self._session(
            tmp_path,
            "deterministic: true\n"
            "models:\n"
            "  m1:\n"
            "    name: foo\n"
            "    path: /models/foo\n",
            deterministic_override=True,
        )
        assert not any("best-effort" in rec.message for rec in loguru_caplog.records)

    def test_non_deterministic_never_warns(self, tmp_path, loguru_caplog):
        self._session(
            tmp_path,
            "models:\n  m1:\n    name: foo\n    api_base: http://external.example/v1\n",
        )
        assert not any("best-effort" in rec.message for rec in loguru_caplog.records)

    def test_mixed_models_only_external_listed(self, tmp_path, loguru_caplog):
        self._session(
            tmp_path,
            "deterministic: true\n"
            "models:\n"
            "  self_hosted:\n"
            "    name: foo\n"
            "    api_base: http://localhost:8000/v1\n"
            "  external_api:\n"
            "    name: bar\n"
            "    api_base: http://external.example/v1\n",
            deterministic_override=True,
            self_managed_endpoints=frozenset({"self_hosted"}),
        )
        messages = [
            rec.message for rec in loguru_caplog.records if "best-effort" in rec.message
        ]
        assert len(messages) == 1
        assert "external_api" in messages[0]
        assert "self_hosted" not in messages[0]


# Tests for unwrap_proxies — recursive MappingProxyType → dict conversion.
# Rationale: dataclasses.asdict(DeploymentPlan) leaves RoleAssignment.engine_params
# as MappingProxyType (frozen via _freeze_dict); yaml.safe_dump raises
# RepresenterError on MappingProxyType nodes.


class TestUnwrapProxies:
    def test_unwraps_top_level_proxy(self):
        proxy = MappingProxyType({"a": 1, "b": 2})
        result = unwrap_proxies(proxy)
        assert type(result) is dict
        assert result == {"a": 1, "b": 2}

    def test_unwraps_nested_proxies_inside_dict(self):
        nested = {
            "outer": MappingProxyType({"inner": MappingProxyType({"x": 1})}),
        }
        result = unwrap_proxies(nested)
        assert type(result["outer"]) is dict
        assert type(result["outer"]["inner"]) is dict
        assert result["outer"]["inner"]["x"] == 1

    def test_unwraps_proxy_inside_list(self):
        mixed = [MappingProxyType({"k": "v"}), 42, "str"]
        result = unwrap_proxies(mixed)
        assert type(result[0]) is dict
        assert result[1] == 42
        assert result[2] == "str"

    def test_unwraps_proxy_inside_tuple(self):
        tup = (MappingProxyType({"k": "v"}),)
        result = unwrap_proxies(tup)
        # tuples become lists (YAML serialization doesn't care)
        assert type(result) is list
        assert type(result[0]) is dict

    def test_passes_through_primitives(self):
        assert unwrap_proxies("str") == "str"
        assert unwrap_proxies(42) == 42
        assert unwrap_proxies(None) is None
        assert unwrap_proxies(True) is True

    def test_unwraps_dataclass_with_nested_mapping_proxy(self):
        """Direct call on DeploymentPlan — the Task 10 use case."""
        from sieval.infer.topology.models import (
            DeploymentPlan,
            DeviceGroup,
            ParallelTopology,
            RoleAssignment,
            WellKnownRole,
        )

        plan = DeploymentPlan(
            checkpoint="/data/ckpts/m",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role=WellKnownRole.FULL,
                    devices=DeviceGroup(count=2, gpu_model="H100"),
                    topology=ParallelTopology(tp=2, dp=1, pp=1),
                    engine_params={"dtype": "bfloat16"},
                ),
            ),
            deterministic=True,
            seed=0,
        )
        result = unwrap_proxies(plan)

        assert type(result) is dict
        assert result["backend"] == "vllm"
        assert result["deterministic"] is True
        # Nested dataclass -> nested dict
        assignment = result["assignments"][0]
        assert type(assignment) is dict
        assert assignment["role"] == WellKnownRole.FULL
        # Nested MappingProxyType inside engine_params -> plain dict
        assert type(assignment["engine_params"]) is dict
        assert assignment["engine_params"] == {"dtype": "bfloat16"}

    def test_dataclass_walk_sidesteps_asdict_pickle_error(self):
        """Regression guard: dataclasses.asdict fails on DeploymentPlan under
        Python 3.13 (mappingproxy not picklable). unwrap_proxies must
        sidestep this by walking fields directly.
        """
        import dataclasses as _dc

        from sieval.infer.topology.models import (
            DeploymentPlan,
            DeviceGroup,
            ParallelTopology,
            RoleAssignment,
            WellKnownRole,
        )

        plan = DeploymentPlan(
            checkpoint="/p",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role=WellKnownRole.FULL,
                    devices=DeviceGroup(count=1, gpu_model="H100"),
                    topology=ParallelTopology(tp=1, dp=1, pp=1),
                    engine_params={"k": "v"},
                ),
            ),
        )

        # asdict is expected to fail — confirming why this helper exists.
        with pytest.raises(TypeError, match="mappingproxy"):
            _dc.asdict(plan)

        # unwrap_proxies handles it cleanly.
        result = unwrap_proxies(plan)
        assert isinstance(result, dict)
        assert result["backend"] == "sglang"


class TestReifyCliOverrides:
    def test_deterministic_sets_root_and_base_model_seed(self):
        cfg = {
            "models": {
                "base": {"name": "qwen3-4b", "args": {"max_tokens": 8192}},
                "derived": {"base": "base", "args": {"temperature": 0.6}},
            }
        }
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["deterministic"] is True
        assert out["models"]["base"]["args"]["seed"] == DETERMINISTIC_DEFAULT_SEED
        assert out["models"]["base"]["args"]["max_tokens"] == 8192
        assert "seed" not in out["models"]["derived"]["args"]

    def test_deterministic_preserves_user_seed(self):
        cfg = {"models": {"base": {"name": "m", "args": {"seed": 42}}}}
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["models"]["base"]["args"]["seed"] == 42

    def test_deterministic_adds_args_dict_when_missing(self):
        cfg = {"models": {"base": {"name": "m"}}}
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["models"]["base"]["args"] == {"seed": DETERMINISTIC_DEFAULT_SEED}

    def test_deterministic_idempotent_when_already_true(self):
        cfg = {
            "deterministic": True,
            "models": {"base": {"name": "m", "args": {"seed": 0}}},
        }
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["deterministic"] is True
        assert out["models"]["base"]["args"]["seed"] == 0

    def test_model_override_rewrites_base_names_only(self):
        cfg = {
            "models": {
                "base_a": {"name": "original-a"},
                "base_b": {"name": "original-b"},
                "derived": {"base": "base_a", "args": {"temperature": 0.5}},
            }
        }
        out = _reify_cli_overrides(cfg, model="new-name")
        assert out["models"]["base_a"]["name"] == "new-name"
        assert out["models"]["base_b"]["name"] == "new-name"
        assert "name" not in out["models"]["derived"]

    def test_result_dir_overrides_root(self):
        cfg = {"result_dir": "./old", "models": {}}
        out = _reify_cli_overrides(cfg, result_dir="./new")
        assert out["result_dir"] == "./new"

    def test_no_overrides_is_noop(self):
        cfg = {"models": {"base": {"name": "m"}}, "deterministic": False}
        out = _reify_cli_overrides(dict(cfg))
        assert out == cfg

    def test_all_three_compose(self):
        cfg = {
            "result_dir": "./old",
            "models": {"base": {"name": "old"}},
        }
        out = _reify_cli_overrides(
            cfg, deterministic=True, model="new", result_dir="./new"
        )
        assert out["deterministic"] is True
        assert out["result_dir"] == "./new"
        assert out["models"]["base"]["name"] == "new"
        assert out["models"]["base"]["args"]["seed"] == DETERMINISTIC_DEFAULT_SEED


class TestApplyEndpointInjection:
    def test_injects_api_base_for_mapped_model(self):
        cfg = {"models": {"m1": {"path": "/ckpts/m1"}}}
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["api_base"] == "http://host:8000/v1"

    def test_injects_placeholder_api_key_when_absent(self):
        cfg = {"models": {"m1": {"path": "/ckpts/m1"}}}
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["api_key"] == "local"

    def test_preserves_user_api_key(self):
        cfg = {"models": {"m1": {"path": "/ckpts/m1", "api_key": "sk-real"}}}
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["api_key"] == "sk-real"

    def test_autofills_name_from_checkpoint_basename(self):
        cfg = {"models": {"m1": {"path": "/data/ckpts/qwen3-4b-sft"}}}
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["name"] == "qwen3-4b-sft"

    def test_autofills_name_from_infer_checkpoint(self):
        cfg = {
            "models": {
                "m1": {"infer": {"checkpoint": "/data/ckpts/qwen3-32b-sft"}},
            }
        }
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["name"] == "qwen3-32b-sft"

    def test_preserves_user_name(self):
        cfg = {"models": {"m1": {"name": "custom", "path": "/ckpts/m1"}}}
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["name"] == "custom"

    def test_empty_endpoint_map_is_noop(self):
        cfg = {"models": {"m1": {"path": "/ckpts/m1"}}}
        before = {"models": {"m1": {"path": "/ckpts/m1"}}}
        out = _apply_endpoint_injection(cfg, {})
        assert out == before

    def test_unlisted_models_untouched(self):
        cfg = {
            "models": {
                "m1": {"path": "/ckpts/m1"},
                "m2": {"api_base": "https://external.example/v1"},
            }
        }
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out["models"]["m1"]["api_base"] == "http://host:8000/v1"
        assert out["models"]["m2"]["api_base"] == "https://external.example/v1"

    def test_returns_same_dict_instance(self):
        cfg = {"models": {"m1": {"path": "/ckpts/m1"}}}
        out = _apply_endpoint_injection(cfg, {"m1": "http://host:8000/v1"})
        assert out is cfg


class TestFormatCommentHeader:
    def test_contains_sieval_version(self):
        from sieval import __version__

        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/path/to/cfg.yaml",
            invocation="sieval eval cfg.yaml",
        )
        assert __version__ in header

    def test_contains_iso_8601_utc_timestamp(self):
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/path/to/cfg.yaml",
            invocation="sieval eval cfg.yaml",
        )
        assert re.search(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)", header
        )

    def test_contains_invocation_line(self):
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/path/to/cfg.yaml",
            invocation="sieval run cfg.yaml --deterministic",
        )
        assert "sieval run cfg.yaml --deterministic" in header

    def test_contains_source_config(self):
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/abs/path/to/cfg.yaml",
            invocation="sieval eval cfg.yaml",
        )
        assert "/abs/path/to/cfg.yaml" in header

    def test_every_line_starts_with_hash(self):
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/p",
            invocation="sieval eval cfg.yaml",
        )
        for line in header.strip().splitlines():
            assert line.startswith("#"), f"non-comment line in header: {line!r}"

    def test_ends_with_newline(self):
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/p",
            invocation="sieval eval cfg.yaml",
        )
        assert header.endswith("\n")

    def test_extra_lines_are_included_and_hash_prefixed(self):
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/p",
            invocation="sieval run cfg.yaml",
            extra_lines=[
                "Reproduce:",
                "  sieval eval <this file>",
            ],
        )
        # Both lines appear, prefixed with "# "
        assert "# Reproduce:" in header
        assert "#   sieval eval <this file>" in header

    def test_no_extra_lines_omits_reproduce_block(self):
        """Callers that don't opt in get a minimal header — prevents the
        reproduce hint from leaking into audit artifacts that aren't
        directly runnable (e.g. infer_plans.yaml)."""
        header = _format_comment_header(
            title="Persisted by sieval",
            source_config="/p",
            invocation="sieval run cfg.yaml",
        )
        assert "Reproduce:" not in header
        assert "sieval eval" not in header


class TestStripHeader:
    """``_strip_header`` is anchored to the ``# ---`` border emitted by
    ``_format_comment_header`` — not "any leading comment line"."""

    def test_strips_well_formed_header_block(self):
        header = _format_comment_header(
            title="Persisted by",
            source_config="/p",
            invocation="sieval eval cfg.yaml",
        )
        body = "models:\n  m:\n    name: foo\n"
        assert _strip_header(header + body) == body

    def test_returns_unchanged_when_no_border(self):
        body = "models:\n  m:\n    name: foo\n"
        assert _strip_header(body) == body

    def test_returns_unchanged_when_open_border_has_no_close(self):
        """A user who deletes the closing border leaves a malformed file —
        return original text so body comparison detects the tampering instead
        of silently consuming an unbounded prefix."""
        broken = (
            "# ---------\n"
            "# Persisted by sieval ...\n"
            "# Invocation: ...\n"
            "models:\n  m: {}\n"
        )
        assert _strip_header(broken) == broken

    def test_does_not_swallow_pre_border_user_comments(self):
        """User-added top-of-file comments outside the bordered block are
        preserved — anchoring on ``# -`` means a leading non-border comment
        line skips the strip entirely, so manual commentary survives the
        round-trip and any attempt to bypass strict-match via prepended
        comments shows up in the body comparison."""
        text = (
            "# my own note\n"
            "# ---------\n"
            "# Persisted by sieval ...\n"
            "# ---------\n"
            "\n"
            "models:\n  m: {}\n"
        )
        assert _strip_header(text) == text


class TestSplitHeader:
    def test_valid_header_is_an_exact_partition(self):
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        body = "models:\n  base:\n    name: m\n"
        h, b = _split_header(header + body)
        assert b == body
        assert h + b == header + body

    def test_no_header_returns_empty_header(self):
        body = "models:\n  base: {}\n"
        h, b = _split_header(body)
        assert h == ""
        assert b == body

    def test_malformed_header_returns_empty_header(self):
        broken = "# " + "-" * 70 + "\n# only one border\nmodels: {}\n"
        h, b = _split_header(broken)
        assert h == ""
        assert b == broken

    def test_strip_header_delegates_to_split(self):
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        body = "models:\n  base:\n    name: m\n"
        assert _strip_header(header + body) == _split_header(header + body)[1]


class TestEvalSessionRawConfig:
    def test_raw_config_is_pristine_after_reification(self, tmp_path):
        """Raw YAML is preserved for persistence — CLI overrides don't leak into it."""
        config_path = _write_yaml_config(
            tmp_path, "cfg.yaml", "models:\n  base:\n    name: original\n"
        )
        session = EvalSession(
            config_path=str(config_path),
            model_override="new-name",
        )
        # _raw_config preserves the on-disk YAML
        assert session._raw_config["models"]["base"]["name"] == "original"
        assert "deterministic" not in session._raw_config
        # self.config has CLI overrides applied
        assert session.config["models"]["base"]["name"] == "new-name"

    def test_raw_config_unaffected_by_deterministic_override(self, tmp_path):
        config_path = _write_yaml_config(
            tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n"
        )
        session = EvalSession(
            config_path=str(config_path),
            deterministic_override=True,
        )
        assert "deterministic" not in session._raw_config
        assert session._raw_config["models"]["base"].get("args", {}).get("seed") is None
        # self.config has reification applied
        assert session.config["deterministic"] is True
        assert session.config["models"]["base"]["args"]["seed"] == 0

    def test_raw_config_unaffected_by_endpoint_map(self, tmp_path):
        config_path = _write_yaml_config(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    path: /ckpts/m\n",
        )
        session = EvalSession(
            config_path=str(config_path),
            endpoint_map={"base": "http://localhost:8000/v1"},
        )
        assert "api_base" not in session._raw_config["models"]["base"]
        # self.config has endpoint injected
        assert (
            session.config["models"]["base"]["api_base"] == "http://localhost:8000/v1"
        )

    def test_infer_plans_kwarg_is_stored(self, tmp_path):
        config_path = _write_yaml_config(tmp_path, "cfg.yaml", "models: {}\n")
        plans = {"m1": {"backend": "vllm", "checkpoint": "/p"}}
        session = EvalSession(
            config_path=str(config_path),
            infer_plans=plans,
        )
        assert session._infer_plans == plans

    def test_defaults_unchanged_for_existing_callers(self, tmp_path):
        """Existing callers that don't pass new kwargs see unchanged behavior."""
        config_path = _write_yaml_config(
            tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n"
        )
        session = EvalSession(config_path=str(config_path))
        # _raw_config and self.config match since no overrides applied
        assert session._raw_config == session.config
        assert session._infer_plans is None

    def test_init_runner_preserves_reified_config(self, tmp_path):
        """Regression guard: runner setup must not overwrite reified self.config."""
        config_path = _write_yaml_config(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    name: m\n",
        )
        session = EvalSession(
            config_path=str(config_path),
            deterministic_override=True,
        )
        # After __init__, config has reification applied
        assert session.config["deterministic"] is True

        # _init_runner is what _prepare_execution now calls for runner setup.
        session._init_runner()

        # Reification must survive runner initialization.
        assert session.config["deterministic"] is True
        assert session.config["models"]["base"]["args"]["seed"] == 0


class TestDiffDicts:
    def test_reports_changed_scalar(self):
        out = _diff_dicts({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert "b" in out
        assert "2" in out and "3" in out

    def test_identical_reports_formatting_only(self):
        out = _diff_dicts({"a": 1}, {"a": 1})
        assert "formatting only" in out

    def test_reports_list_length_change(self):
        out = _diff_dicts({"xs": [1, 2]}, {"xs": [1, 2, 3]})
        assert "list length 2 → 3" in out


class TestDiffLines:
    def test_identical_returns_empty(self):
        assert _diff_lines({"a": 1}, {"a": 1}) == []

    def test_nested_leaf_path(self):
        lines = _diff_lines({"a": {"b": 1}}, {"a": {"b": 2}})
        assert lines == ["- a.b: 1 → 2"]


class TestAppendResumeNote:
    def test_note_inserted_before_closing_border_and_split_stable(self):
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        body = "models:\n  base:\n    name: m\n"
        out = _append_resume_note(header, ["- concurrency_limit: 8 → 2"])

        assert "Persisted by sieval" in out  # origin preserved
        assert "Resumed by sieval" in out
        assert "#   - concurrency_limit: 8 → 2" in out
        # The note sits inside the border pair: the whole block is still parsed
        # as header (body is not polluted) when prepended to a body.
        h, b = _split_header(out + body)
        assert b == body
        assert "Resumed by sieval" in h

    def test_second_append_accumulates(self):
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        once = _append_resume_note(header, ["- a: 1 → 2"])
        twice = _append_resume_note(once, ["- a: 2 → 3"])
        assert twice.count("Resumed by sieval") == 2
        assert "- a: 1 → 2" in twice and "- a: 2 → 3" in twice


class TestBriefDiff:
    """``_brief_diff`` is called from the resume-mismatch error message;
    its output quality directly affects how quickly users diagnose why
    a resume aborted."""

    def test_scalar_value_diff(self):
        existing = "deterministic: false\n"
        current = "deterministic: true\n"
        out = _brief_diff(existing, current)
        assert "deterministic: False → True" in out

    def test_nested_dict_diff_emits_dotted_path(self):
        existing = "models:\n  base:\n    name: old\n"
        current = "models:\n  base:\n    name: new\n"
        out = _brief_diff(existing, current)
        assert "models.base.name: 'old' → 'new'" in out

    def test_list_diff_descends_into_elements(self):
        """Regression: operations are a list of single-key dicts — a seed
        change inside one op must surface as the specific nested field,
        not the whole list repr."""
        existing = "datasets:\n  d:\n    operations:\n      - shuffle: {seed: 42}\n"
        current = "datasets:\n  d:\n    operations:\n      - shuffle: {seed: 43}\n"
        out = _brief_diff(existing, current)
        assert "datasets.d.operations[0].shuffle.seed: 42 → 43" in out

    def test_list_length_change_is_called_out(self):
        existing = "xs:\n  - 1\n  - 2\n"
        current = "xs:\n  - 1\n  - 2\n  - 3\n"
        out = _brief_diff(existing, current)
        assert "xs: list length 2 → 3" in out

    def test_invalid_yaml_falls_back_to_generic_message(self):
        """Parse errors in the existing file must not mask the caller's
        Resume aborted RuntimeError with a parse traceback."""
        out = _brief_diff("not: [valid: yaml", "deterministic: true\n")
        assert "not valid YAML" in out

    def test_whitespace_only_diff(self):
        """Structurally identical dicts still trip a body-byte mismatch
        (e.g. key reorder, trailing whitespace) — diff helper reports
        that clearly instead of an empty diff block."""
        existing = "a: 1\nb: 2\n"
        current = "b: 2\na: 1\n"
        out = _brief_diff(existing, current)
        assert "whitespace / formatting only" in out


# ── Tests for env expansion + error-hint wrapping in _setup_datasets ──


def test_dataset_path_env_expanded_before_instantiation(monkeypatch, tmp_path):
    """${SIEVAL_DATA_DIR} in path: must expand before being passed to the
    Dataset constructor."""
    from unittest.mock import MagicMock, patch

    from sieval.cli.leaderboard.session import EvalSession

    monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))

    session = EvalSession.__new__(EvalSession)
    session.datasets = {}
    session.config = {
        "datasets": {
            "drop": {
                "class": "sieval.datasets.drop.DROPDataset",
                "path": "${SIEVAL_DATA_DIR}/drop",
            }
        }
    }

    captured = {}

    class StubDS:
        _sieval_dataset_meta = MagicMock()
        _sieval_dataset_meta.name = "drop"

        def __init__(self, path=None, **kwargs):
            captured["path"] = path

    with (
        patch(
            "sieval.cli.leaderboard.session.resolve_dataset_class",
            return_value=StubDS,
        ),
        patch.object(
            EvalSession,
            "_get_named_config_map",
            return_value=session.config["datasets"],
        ),
        patch.object(EvalSession, "_normalize_dict", return_value={}),
        patch.object(EvalSession, "_normalize_list", return_value=[]),
        patch.object(
            EvalSession,
            "_apply_dataset_operations",
            side_effect=lambda ds, *a, **kw: ds,
        ),
    ):
        session._setup_datasets()

    assert captured["path"] == str(tmp_path / "drop")


def test_dataset_missing_file_error_appends_download_hint(tmp_path):
    """Missing-dataset errors get wrapped in a RuntimeError carrying the
    `sieval dataset download` hint. The original FileNotFoundError is kept
    on `__cause__` so attributes like `.filename` / `.errno` remain
    inspectable by any caller that needs them."""
    from unittest.mock import MagicMock, patch

    import pytest

    from sieval.cli.leaderboard.session import EvalSession

    session = EvalSession.__new__(EvalSession)
    session.datasets = {}
    session.config = {
        "datasets": {
            "drop": {
                "class": "sieval.datasets.drop.DROPDataset",
                "path": str(tmp_path / "does-not-exist"),
            }
        }
    }

    class RaisingDS:
        _sieval_dataset_meta = MagicMock()
        _sieval_dataset_meta.name = "drop"

        def __init__(self, *a, **kw):
            raise FileNotFoundError("bogus/missing/file.jsonl")

    with (
        patch(
            "sieval.cli.leaderboard.session.resolve_dataset_class",
            return_value=RaisingDS,
        ),
        patch.object(
            EvalSession,
            "_get_named_config_map",
            return_value=session.config["datasets"],
        ),
        patch.object(EvalSession, "_normalize_dict", return_value={}),
        patch.object(EvalSession, "_normalize_list", return_value=[]),
        pytest.raises(RuntimeError, match="sieval dataset download drop") as excinfo,
    ):
        session._setup_datasets()

    # The original exception type is preserved on the cause chain and still
    # rendered in the wrapper message, so users keep the diagnostic signal.
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert "FileNotFoundError" in str(excinfo.value)


def test_dataset_missing_file_preserves_original_on_cause_chain():
    """`DataFilesNotFoundError` from the datasets library must survive on
    `__cause__` so downstream isinstance checks against OSError subclasses
    continue to work (previously we reconstructed via `type(exc)(...)` which
    dropped OSError-specific attrs)."""
    from unittest.mock import MagicMock, patch

    import pytest
    from datasets.exceptions import DataFilesNotFoundError

    from sieval.cli.leaderboard.session import EvalSession

    session = EvalSession.__new__(EvalSession)
    session.datasets = {}
    session.config = {
        "datasets": {
            "drop": {
                "class": "sieval.datasets.drop.DROPDataset",
                "path": "/nope",
            }
        }
    }

    class RaisingDS:
        _sieval_dataset_meta = MagicMock()
        _sieval_dataset_meta.name = "drop"

        def __init__(self, *a, **kw):
            raise DataFilesNotFoundError("no files")

    with (
        patch(
            "sieval.cli.leaderboard.session.resolve_dataset_class",
            return_value=RaisingDS,
        ),
        patch.object(
            EvalSession,
            "_get_named_config_map",
            return_value=session.config["datasets"],
        ),
        patch.object(EvalSession, "_normalize_dict", return_value={}),
        patch.object(EvalSession, "_normalize_list", return_value=[]),
        pytest.raises(RuntimeError) as excinfo,
    ):
        session._setup_datasets()

    assert isinstance(excinfo.value.__cause__, DataFilesNotFoundError)


# ---------------------------------------------------------------------------
# Alignment block
# ---------------------------------------------------------------------------


class TestEvalSessionAlignment:
    """YAML-level alignment block parsing."""

    def _write_card(self, path: Path) -> Path:
        card = path / "test-card.md"
        card.write_text(
            """---
reference: {kind: tr, source: "arXiv:0000.00000", title: "Test"}
tolerance: 3.0
reference_scores: {m: {t: 1.0}}
---
""",
            encoding="utf-8",
        )
        return card

    def _minimal_yaml(self, tmp_path: Path, extra: str = "") -> Path:
        yaml_text = f"""
result_dir: ./outputs/test
models:
  m:
    name: m
    args:
      max_tokens: 16
datasets:
  t:
    class: TestDataset
    path: ./data/test
{extra}
"""
        yaml_path = tmp_path / "lb.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")
        return yaml_path

    def test_no_alignment_block(self, tmp_path: Path) -> None:
        yaml_path = self._minimal_yaml(tmp_path)
        session = EvalSession(yaml_path)
        assert session.alignment_card is None

    def test_alignment_block_loads_card(self, tmp_path: Path) -> None:
        card = self._write_card(tmp_path)
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra=f"alignment:\n  card: {card.name}\n",
        )
        session = EvalSession(yaml_path)
        assert session.alignment_card is not None
        assert session.alignment_card.title == "Test"
        assert session.alignment_card.tolerance == 3.0

    def test_alignment_block_card_path_stays_relative(self, tmp_path: Path) -> None:
        """Card path is stored verbatim in both raw and reified views.

        Absolutizing would pin ``effective_config.yaml`` to a host-specific
        path and break run-bundle portability. Readers re-resolve against
        ``config_path.parent``, which tracks the YAML wherever it's copied.
        """
        card = self._write_card(tmp_path)
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra=f"alignment:\n  card: {card.name}\n",
        )
        session = EvalSession(yaml_path)
        raw = session._raw_config.get("alignment")
        reified = session._reified_config.get("alignment")
        assert raw is not None and reified is not None
        assert raw["card"] == card.name
        assert reified["card"] == card.name

    def test_alignment_block_missing_card_field(self, tmp_path: Path) -> None:
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra="alignment: {}\n",
        )
        with pytest.raises(ValueError, match="alignment.card"):
            EvalSession(yaml_path)

    def test_alignment_block_unknown_key_rejected(self, tmp_path: Path) -> None:
        """Typos like ``cards`` must not silently succeed."""
        card = self._write_card(tmp_path)
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra=f"alignment:\n  card: {card.name}\n  cards: extra\n",
        )
        with pytest.raises(ValueError, match="unknown keys"):
            EvalSession(yaml_path)

    def test_alignment_block_not_a_mapping(self, tmp_path: Path) -> None:
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra="alignment: some-string\n",
        )
        with pytest.raises(ValueError, match="alignment.*mapping"):
            EvalSession(yaml_path)

    def test_alignment_block_list_rejected(self, tmp_path: Path) -> None:
        """A list-valued ``alignment:`` (common mis-indent) must be rejected."""
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra="alignment:\n  - card: x.md\n",
        )
        with pytest.raises(ValueError, match="alignment.*mapping"):
            EvalSession(yaml_path)

    def test_alignment_card_file_not_found(self, tmp_path: Path) -> None:
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra="alignment:\n  card: does-not-exist.md\n",
        )
        with pytest.raises(FileNotFoundError):
            EvalSession(yaml_path)

    def test_alignment_card_malformed(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.md"
        bad.write_text("# no frontmatter\n", encoding="utf-8")
        yaml_path = self._minimal_yaml(
            tmp_path,
            extra="alignment:\n  card: bad.md\n",
        )
        with pytest.raises(ValueError, match="frontmatter"):
            EvalSession(yaml_path)
