"""
Tests for session pure functions: class resolution, submodule guessing,
dataset operations, and runner config building.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import dataclasses
import json
import re
import types
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
import yaml

from sieval.cli._filter_spec import VALUES_DIGEST_KEY, compute_values_digest
from sieval.cli.leaderboard.session import (
    _DETERMINISTIC_SEED_CONTRACT_KEY,
    _NONMATCH_RUNNER_KEYS,
    _STRICT_RUNNER_KEYS,
    _THROUGHPUT_RUNNER_KEYS,
    DETERMINISTIC_DEFAULT_SEED,
    EvalSession,
    _append_resume_note,
    _apply_endpoint_injection,
    _apply_request_seed_decision_to_args,
    _apply_request_seed_decision_to_model,
    _brief_diff,
    _cross_version_resume_hint,
    _describe_order_change,
    _diff_dicts,
    _diff_key_shape,
    _diff_lines,
    _format_comment_header,
    _reify_cli_overrides,
    _resolve_deterministic_request_seed,
    _sort_versions,
    _split_header,
    _strip_header,
    _strip_noncomparable_fields,
    arun_session,
    resolve_deterministic,
    run_session,
    unwrap_proxies,
)
from sieval.cli.resolution import derive_model_type
from sieval.cli.validation import _VALID_OPERATIONS
from sieval.core.models.connection_factory import DEFAULT_REQUEST_TIMEOUT
from sieval.core.models.dialect_registry import RequestSeedSupport
from sieval.core.models.model import Model
from sieval.core.models.reconcile import CheckStage, Configured, DeferredCheck
from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    InlineModelBinding,
    InputKind,
    NamedModelBinding,
    RequirementContext,
    TaskModelRequirement,
    TaskRequirements,
)
from sieval.core.runners import TaskRunnerConfig
from sieval.core.runners.multi_runner import MultiTaskRunner
from tests.conftest import MockChatModel


def example_row_key(row: dict) -> str:
    """A stand-in for a real key function, referenced by dotted path below."""
    return f"{row.get('subset')}::{row.get('key')}"


NOT_CALLABLE = "I am a string, not a function"


def _write_yaml_config(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def _task_requirement_context_for_setup_test(
    task_cfg: object,
    models: dict[str, Model],
) -> RequirementContext:
    """Build the frozen candidate seam expected by direct ``_setup_tasks`` tests."""

    if not isinstance(task_cfg, dict):
        return RequirementContext()
    typed_task_cfg = cast(dict[str, Any], task_cfg)
    infer_args = typed_task_cfg.get("infer_args", {})
    if not isinstance(infer_args, dict):
        infer_args = {}
    model_name = typed_task_cfg.get("model")
    if not model_name and len(models) == 1:
        model_name = next(iter(models))
    bindings = {}
    if isinstance(model_name, str) and model_name:
        bindings["candidate"] = NamedModelBinding(
            binding_id=f"model:{model_name}",
            root_deployment_key=f"model:{model_name}",
            requested_model_id=model_name,
            config_name=model_name,
        )
    return RequirementContext(model_bindings=bindings, infer_args=infer_args)


def _prepare_eval_session(
    config_path: Path,
    *,
    models: dict[str, Model] | None = None,
    resume: bool = False,
) -> MultiTaskRunner:
    runner = EvalSession(config_path=str(config_path), resume=resume)
    if models is not None:
        runner.models = models
    tasks_cfg = runner._get_named_config_map("tasks")
    runner._task_requirement_contexts = {
        task_name: _task_requirement_context_for_setup_test(
            task_cfg,
            runner.models,
        )
        for task_name, task_cfg in tasks_cfg.items()
    }
    runner._init_runner()
    runner._setup_datasets()
    runner._setup_tasks()
    assert runner.runner is not None
    return runner.runner


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
            (
                "filter",
                {"by": "subset", "value": "gsm8k"},
                "filter",
                ("subset", "gsm8k"),
                {"require_all": False, "split": "test"},
            ),
            (
                "filter",
                {"by": "subset", "value": ["gsm8k", "svamp"]},
                "filter",
                ("subset", ["gsm8k", "svamp"]),
                {"require_all": False, "split": "test"},
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
            ("filter", {}, "'filter' requires 'by'"),
            (
                "filter",
                {"by": "subset"},
                "'filter' requires exactly one of 'value' or 'values_file'",
            ),
        ],
    )
    def test_operation_required_args(self, op, missing_args, error_match):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match=error_match):
            runner._apply_dataset_operations(ds, [{op: missing_args}], "test_ds")

    # A falsy value is a value. Checking `op_args.get("value")` for truthiness
    # would reject these as "omitted" and make integer/boolean columns
    # unfilterable.
    @pytest.mark.parametrize("value", [0, False, "", []])
    def test_filter_accepts_a_falsy_value(self, value):
        runner = self._make_runner()
        ds = MagicMock()
        ds.filter.return_value = ds

        runner._apply_dataset_operations(
            ds, [{"filter": {"by": "n", "value": value}}], "test_ds"
        )
        ds.filter.assert_called_once_with("n", value, require_all=False, split="test")

    def test_the_unknown_operation_message_lists_every_valid_operation(self):
        # This message omitted 'filter' for as long as 'filter' had existed, so
        # a user who mistyped an operation was told it was not among four when
        # it was among five. Derived from the validator's set rather than
        # restated by hand, because a hand-written list is what drifted.
        runner = self._make_runner()
        with pytest.raises(ValueError, match="Unknown operation") as excinfo:
            runner._apply_dataset_operations(MagicMock(), [{"nope": {}}], "test_ds")
        listed = str(excinfo.value).split("Valid operations:")[1]
        assert {op.strip() for op in listed.split(",")} == _VALID_OPERATIONS

    # -- `by` in its three config forms -----------------------------------

    def test_filter_by_a_list_of_columns_is_passed_through(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.filter.return_value = ds

        runner._apply_dataset_operations(
            ds, [{"filter": {"by": ["a", "b"], "value": [["x", "y"]]}}], "test_ds"
        )
        ds.filter.assert_called_once_with(
            ["a", "b"], [["x", "y"]], require_all=False, split="test"
        )

    def test_filter_by_a_callable_reference_resolves_to_the_function(self):
        # The parity claim at its narrowest: what YAML names, `filter` receives
        # as the very object a Python caller would have passed.
        runner = self._make_runner()
        ds = MagicMock()
        ds.filter.return_value = ds

        by = {"callable": f"{__name__}.example_row_key"}
        runner._apply_dataset_operations(
            ds, [{"filter": {"by": by, "value": "k"}}], "test_ds"
        )
        assert ds.filter.call_args.args[0] is example_row_key

    @pytest.mark.parametrize(
        "by,error_match",
        [
            ({"callable": "no_dots_here"}, "could not be resolved"),
            ({"callable": ".relative"}, "could not be resolved"),
            ({"callable": f"{__name__}.does_not_exist"}, "could not be resolved"),
            ({"callable": f"{__name__}.NOT_CALLABLE"}, "not a callable"),
            ({"callable": "x.y", "extra": 1}, "exactly one key, 'callable'"),
            (42, "must be a column name"),
            ([], "must name one or more columns"),
            (["a", 2], "must name one or more columns"),
        ],
    )
    def test_filter_by_rejects_a_reference_it_cannot_use(self, by, error_match):
        runner = self._make_runner()
        with pytest.raises(ValueError, match=error_match):
            runner._apply_dataset_operations(
                MagicMock(), [{"filter": {"by": by, "value": "k"}}], "test_ds"
            )

    def test_filter_by_callable_failure_names_the_dataset(self):
        # Same obligation the other operation errors carry: a config with many
        # datasets has to say which one is wrong.
        runner = self._make_runner()
        with pytest.raises(ValueError, match="Dataset 'ds7'"):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": {"callable": "no_such_pkg.fn"}, "value": "k"}}],
                "ds7",
            )

    # -- `values_file` ----------------------------------------------------

    def _runner_at(self, config_dir):
        runner = self._make_runner()
        runner.config_path = Path(config_dir) / "eval.yaml"
        return runner

    def _filtered_values(self, tmp_path, filename, text):
        (tmp_path / filename).write_text(text, encoding="utf-8")
        runner = self._runner_at(tmp_path)
        ds = MagicMock()
        ds.filter.return_value = ds
        runner._apply_dataset_operations(
            ds, [{"filter": {"by": "id", "values_file": filename}}], "test_ds"
        )
        return ds.filter.call_args.args[1]

    def test_values_file_reads_a_json_list(self, tmp_path):
        assert self._filtered_values(tmp_path, "k.json", '["a", "b"]') == ["a", "b"]

    def test_values_file_reads_a_json_object_as_its_keys(self, tmp_path):
        # A selection usually arrives as a map from id to whatever justified
        # picking it. Requiring it be stripped to a bare list first would mean
        # the file that is kept is not the file the config reads.
        assert self._filtered_values(
            tmp_path, "k.json", '{"a": {"why": "x"}, "b": {"why": "y"}}'
        ) == ["a", "b"]

    def test_values_file_reads_one_value_per_line(self, tmp_path):
        assert self._filtered_values(
            tmp_path, "k.txt", "# picked by hand\na\n\n  b  \n"
        ) == ["a", "b"]

    def test_values_file_resolves_against_the_config_directory(self, tmp_path):
        # The same rule `alignment.card` follows: stored verbatim, resolved
        # relative to the config that named it.
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "k.json").write_text('["a"]', encoding="utf-8")
        runner = self._runner_at(tmp_path)
        ds = MagicMock()
        ds.filter.return_value = ds

        runner._apply_dataset_operations(
            ds, [{"filter": {"by": "id", "values_file": "sub/k.json"}}], "test_ds"
        )
        assert ds.filter.call_args.args[1] == ["a"]

    def test_values_file_accepts_an_absolute_path(self, tmp_path):
        target = tmp_path / "elsewhere.json"
        target.write_text('["a"]', encoding="utf-8")
        runner = self._runner_at(tmp_path / "cfg")
        ds = MagicMock()
        ds.filter.return_value = ds

        runner._apply_dataset_operations(
            ds, [{"filter": {"by": "id", "values_file": str(target)}}], "test_ds"
        )
        assert ds.filter.call_args.args[1] == ["a"]

    @pytest.mark.parametrize(
        "filename,text,error_match",
        [
            ("k.json", "{not json", "not valid JSON"),
            ("k.json", '"a string"', "must hold a JSON list"),
        ],
    )
    def test_values_file_rejects_a_file_it_cannot_read_as_values(
        self, tmp_path, filename, text, error_match
    ):
        (tmp_path / filename).write_text(text, encoding="utf-8")
        runner = self._runner_at(tmp_path)
        with pytest.raises(ValueError, match=error_match):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": "id", "values_file": filename}}],
                "test_ds",
            )

    def test_values_file_missing_raises(self, tmp_path):
        runner = self._runner_at(tmp_path)
        with pytest.raises(ValueError, match="'values_file' not found"):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": "id", "values_file": "gone.json"}}],
                "test_ds",
            )

    def test_filter_rejects_both_value_and_values_file(self):
        runner = self._make_runner()
        with pytest.raises(ValueError, match="exactly one of 'value' or"):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": "id", "value": "a", "values_file": "k.json"}}],
                "test_ds",
            )

    # -- `require_all` ----------------------------------------------------

    def test_require_all_is_passed_through(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.filter.return_value = ds

        runner._apply_dataset_operations(
            ds,
            [{"filter": {"by": "id", "value": ["a"], "require_all": True}}],
            "test_ds",
        )
        assert ds.filter.call_args.kwargs["require_all"] is True

    def test_require_all_must_be_a_boolean(self):
        # `require_all: "no"` is truthy in Python, so a string would arm the
        # check while reading as if it turned it off.
        runner = self._make_runner()
        with pytest.raises(ValueError, match="'require_all' must be a boolean"):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": "id", "value": "a", "require_all": "no"}}],
                "test_ds",
            )

    # -- key names --------------------------------------------------------

    def test_a_misspelled_optional_key_is_rejected_rather_than_ignored(self):
        # The failure this closes: `require_all_keys` reads as `require_all`
        # left at its default, so the run proceeds with the assertion silently
        # disarmed — and reports a plausible number on a partial selection.
        runner = self._make_runner()
        with pytest.raises(
            ValueError, match=r"unknown key\(s\) \['require_all_keys'\]"
        ):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": "id", "value": "a", "require_all_keys": True}}],
                "test_ds",
            )

    def test_split_must_be_a_split_name(self):
        # A non-string `split` matches no split, which keeps every row.
        runner = self._make_runner()
        with pytest.raises(ValueError, match="'split' must be a split name"):
            runner._apply_dataset_operations(
                MagicMock(),
                [{"filter": {"by": "id", "value": "a", "split": ["test"]}}],
                "test_ds",
            )

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
            "Subject",
            num=800,
            per_group=None,
            fraction=None,
            min_per_group=5,
            seed=42,
            split="test",
        )

    def test_stratified_sample_fraction_dispatch(self):
        # MMMLU's efficient-eval setting: a share of every (Locale, Subject) cell.
        runner = self._make_runner()
        ds = MagicMock()
        ds.stratified_sample.return_value = ds
        runner._apply_dataset_operations(
            ds,
            [
                {
                    "stratified_sample": {
                        "by": ["Locale", "Subject"],
                        "fraction": 0.1,
                        "seed": 0,
                    }
                }
            ],
            "test_ds",
        )
        ds.stratified_sample.assert_called_once_with(
            ["Locale", "Subject"],
            num=None,
            per_group=None,
            fraction=0.1,
            min_per_group=None,
            seed=0,
            split="test",
        )

    def test_stratified_sample_fraction_accepts_min_per_group(self):
        runner = self._make_runner()
        ds = MagicMock()
        ds.stratified_sample.return_value = ds
        runner._apply_dataset_operations(
            ds,
            [
                {
                    "stratified_sample": {
                        "by": "Subject",
                        "fraction": 0.05,
                        "min_per_group": 3,
                    }
                }
            ],
            "test_ds",
        )
        ds.stratified_sample.assert_called_once_with(
            "Subject",
            num=None,
            per_group=None,
            fraction=0.05,
            min_per_group=3,
            seed=0,
            split="test",
        )

    @pytest.mark.parametrize("bad", [True, False, "0.1", 0, 1.5, -0.5])
    def test_stratified_sample_rejects_a_bad_fraction_naming_the_dataset(self, bad):
        # Guarded at this layer too so the message names the offending dataset,
        # like the 'by' and budget-count guards. `True` is the trap: an `int`
        # subclass that passes a bare range test and keeps every row.
        runner = self._make_runner()
        ds = MagicMock()
        ds.stratified_sample.return_value = ds
        with pytest.raises(ValueError, match=r"test_ds.*'fraction' must be a number"):
            runner._apply_dataset_operations(
                ds,
                [{"stratified_sample": {"by": "Subject", "fraction": bad}}],
                "test_ds",
            )
        ds.stratified_sample.assert_not_called()

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
            fraction=None,
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

    @pytest.mark.parametrize(
        "budgets",
        [
            {},
            {"num": 5, "per_group": 2},
            {"num": 5, "fraction": 0.5},
            {"per_group": 2, "fraction": 0.5},
        ],
    )
    def test_stratified_sample_requires_exactly_one_budget(self, budgets):
        runner = self._make_runner()
        ds = MagicMock()
        with pytest.raises(ValueError, match="exactly one of 'num', 'per_group'"):
            runner._apply_dataset_operations(
                ds, [{"stratified_sample": {"by": "Subject", **budgets}}], "test_ds"
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
            fraction=None,
            min_per_group=None,
            seed=42,
            split="test",
        )


# ===================================================================
# Model type derivation from normalized task requirements
# ===================================================================
class TestDeriveModelType:
    @staticmethod
    def _requirements(
        *kinds: InputKind,
    ) -> AggregatedTaskRequirements:
        return AggregatedTaskRequirements(
            input=frozenset(kinds),
            input_sources={kind: frozenset({f"{kind.value}_task"}) for kind in kinds},
        )

    def test_explicit_type_and_default_without_task_evidence(self):
        empty = self._requirements()
        assert derive_model_type("m", "gen", empty) == "gen"
        assert derive_model_type("m", "chat", empty) == "chat"
        assert derive_model_type("m", None, empty) == "chat"

    def test_normalized_completion_evidence_derives_gen(self):
        requirements = self._requirements(InputKind.COMPLETION)
        assert derive_model_type("m", None, requirements) == "gen"
        assert derive_model_type("m", "gen", requirements) == "gen"

    def test_explicit_type_is_assertion_when_evidence_exists(self):
        requirements = self._requirements(InputKind.COMPLETION)
        with pytest.raises(
            ValueError,
            match=r"declares type: chat.*require 'gen'.*checked assertion",
        ):
            derive_model_type("m", "chat", requirements)

    def test_conflicting_normalized_inputs_report_sources(self):
        requirements = self._requirements(InputKind.CHAT, InputKind.COMPLETION)
        with pytest.raises(ValueError, match="conflicting normalized input") as exc:
            derive_model_type("m", None, requirements)

        message = str(exc.value)
        assert "chat_task" in message
        assert "completion_task" in message
        assert "separate root model configs" in message

    def test_rejects_invalid_explicit_type_and_non_normalized_evidence(self):
        empty = self._requirements()
        with pytest.raises(ValueError, match="invalid type"):
            derive_model_type("m", "other", empty)
        with pytest.raises(TypeError, match="AggregatedTaskRequirements"):
            derive_model_type("m", None, cast(Any, object()))


class TestTaskModelConfigName:
    """Task -> candidate model-name resolution.

    This is the single surviving resolver: `_setup_tasks` reads the candidate
    from the prelaunch-frozen `RequirementContext`, so these errors are the
    only thing standing between a typo'd `model:` and a confusing failure
    much deeper in binding.
    """

    def _session(self):
        return object.__new__(EvalSession)

    def test_explicit_model_ref_is_returned(self):
        session = self._session()
        models_cfg = {"my_model": {"name": "org/m"}, "other": {"name": "org/o"}}

        assert (
            session._task_model_config_name("t1", {"model": "my_model"}, models_cfg)
            == "my_model"
        )

    def test_single_model_is_the_default_candidate(self):
        session = self._session()

        assert (
            session._task_model_config_name("t1", {}, {"only": {"name": "org/m"}})
            == "only"
        )

    def test_unknown_model_ref_is_rejected(self):
        session = self._session()
        with pytest.raises(ValueError, match="references unknown model 'bad_ref'"):
            session._task_model_config_name(
                "t1", {"model": "bad_ref"}, {"my_model": {"name": "org/m"}}
            )

    def test_no_models_defined_is_rejected(self):
        session = self._session()
        with pytest.raises(ValueError, match="no models defined in config"):
            session._task_model_config_name("t1", {}, {})

    def test_ambiguous_candidate_is_rejected(self):
        session = self._session()
        with pytest.raises(ValueError, match="'model' required when multiple models"):
            session._task_model_config_name(
                "t1", {}, {"a": {"name": "org/a"}, "b": {"name": "org/b"}}
            )

    def test_non_string_model_ref_is_rejected(self):
        session = self._session()
        with pytest.raises(ValueError, match="'model' must be a string reference"):
            session._task_model_config_name(
                "t1", {"model": 123}, {"my_model": {"name": "org/m"}}
            )


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
deterministic: true
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
        persisted = yaml.safe_load(
            (tmp_path / "arun_results" / "effective_config.yaml").read_text()
        )
        contract = persisted[_DETERMINISTIC_SEED_CONTRACT_KEY]
        assert set(contract) == {"bindings", "external_roles", "candidates"}
        binding = contract["bindings"]["model:mock_model"]
        assert binding["dialect_id"] == "openai_chat"
        assert binding["request_seed_support"] == "supported"
        assert binding["seed"] == DETERMINISTIC_DEFAULT_SEED
        assert binding["seed_provenance"] == "automatic"


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
# filter values_file: pinned into the config the resume gate compares
# ===================================================================
class TestFilterValuesFilePinning:
    """The selection lives outside the config, so the config must pin it.

    ``effective_config.yaml`` records ``values_file`` as a path. Without a
    digest beside it two runs whose persisted configs compare equal can score
    different sample sets, and ``--resume`` accepts the second — the failure is
    silent, which is what makes it worth a test at this level rather than a
    unit test of the digest helper alone.
    """

    CONFIG = """
result_dir: {result_dir}
datasets:
  curated:
    class: fake.Dataset
    operations:
      - filter: {{by: id, values_file: picked.json}}
"""

    def _session(self, tmp_path, contents, *, result_dir="out"):
        (tmp_path / "picked.json").write_text(contents, encoding="utf-8")
        config_path = _write_yaml_config(
            tmp_path,
            "eval.yaml",
            self.CONFIG.format(result_dir=str(tmp_path / result_dir)),
        )
        return EvalSession(config_path=str(config_path))

    def _op(self, session):
        return session._reified_config["datasets"]["curated"]["operations"][0]["filter"]

    def test_the_digest_reaches_the_persisted_config(self, tmp_path):
        # _reified_config is what _persist_effective_config dumps, so this is
        # the view the resume gate compares.
        session = self._session(tmp_path, '["a", "b"]')
        assert self._op(session)[VALUES_DIGEST_KEY] == (
            compute_values_digest(b'["a", "b"]')
        )

    def test_the_path_is_still_stored_verbatim(self, tmp_path):
        # Portability is why the path is not resolved: effective_config.yaml
        # has to stay meaningful on another machine.
        session = self._session(tmp_path, '["a"]')
        assert self._op(session)["values_file"] == "picked.json"

    def test_editing_the_file_changes_the_persisted_config(self, tmp_path):
        # The gate compares config bodies. Before this pin both bodies were
        # byte-identical while the selection differed.
        before = self._op(self._session(tmp_path, '["a", "b"]'))
        after = self._op(self._session(tmp_path, '["a"]'))
        assert before != after
        assert before[VALUES_DIGEST_KEY] != after[VALUES_DIGEST_KEY]

    def test_resume_aborts_when_the_values_file_changed(self, tmp_path):
        # End to end through the real gate: persist, edit the file, resume.
        session = self._session(tmp_path, '["a", "b"]')
        anyio.run(session._persist_effective_config)

        resumed = self._session(tmp_path, '["a"]')
        resumed.resume_override = True
        with pytest.raises(RuntimeError, match="does not match current invocation"):
            anyio.run(resumed._persist_effective_config)

    def test_resume_is_accepted_when_the_values_file_is_untouched(self, tmp_path):
        # The other half: an unchanged file must not start failing resumes.
        session = self._session(tmp_path, '["a", "b"]')
        anyio.run(session._persist_effective_config)

        resumed = self._session(tmp_path, '["a", "b"]')
        resumed.resume_override = True
        anyio.run(resumed._persist_effective_config)

    def test_rerunning_a_persisted_config_verifies_its_own_digest(self, tmp_path):
        # `sieval run effective_config.yaml` after the values file moved: the
        # digest it carries is checked rather than silently re-pinned.
        session = self._session(tmp_path, '["a", "b"]')
        anyio.run(session._persist_effective_config)
        persisted = Path(str(tmp_path / "out")) / "effective_config.yaml"
        (tmp_path / "out" / "picked.json").write_text('["a"]', encoding="utf-8")

        with pytest.raises(ValueError, match="has changed since this config"):
            EvalSession(config_path=str(persisted))

    def test_a_missing_values_file_is_reported_at_load(self, tmp_path):
        config_path = _write_yaml_config(
            tmp_path, "eval.yaml", self.CONFIG.format(result_dir=str(tmp_path / "out"))
        )
        with pytest.raises(ValueError, match="'values_file' not found"):
            EvalSession(config_path=str(config_path))

    def test_the_read_rechecks_the_digest(self, tmp_path):
        # Closes the window between the load-time read and the apply-time one.
        session = self._session(tmp_path, '["a"]')
        with pytest.raises(ValueError, match="changed while the run was starting"):
            session._read_filter_values("picked.json", "curated", "sha256:" + "0" * 64)


# ===================================================================
# _setup_models: canonical post-launch binding gate
# ===================================================================
class TestSetupModels:
    def test_requires_postlaunch_reconciliation(self):
        session = object.__new__(EvalSession)

        with pytest.raises(RuntimeError, match="post-launch capability"):
            session._setup_models()

    def test_delegates_to_canonical_binding_path(self):
        session = object.__new__(EvalSession)
        session.postlaunch_reconcile_result = MagicMock()

        with patch.object(session, "_setup_bound_models") as setup_bound:
            session._setup_models()

        setup_bound.assert_called_once_with()


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
        runner._task_requirement_contexts = (
            {
                task_name: _task_requirement_context_for_setup_test(
                    task_cfg,
                    runner.models,
                )
                for task_name, task_cfg in tasks_cfg.items()
            }
            if isinstance(tasks_cfg, dict)
            else {}
        )
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
# RFC #25 pre-launch requirement composition and reconciliation
# ===================================================================
class TestPrelaunchReconciliation:
    class ChatTask:
        model_type = "chat"

        def __init__(self, *, grader=None, models_by_role=None):
            del grader, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="chat_task",
                ),
            )

    class CompletionScoringTask:
        model_type = "gen"

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(
                        input=InputKind.COMPLETION,
                        input_scoring=True,
                        sampled_logprobs=True,
                        min_top_logprobs=7,
                    ),
                    source_task="completion_scoring",
                ),
            )

    class CompletionTask:
        model_type = "gen"

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.COMPLETION),
                    source_task="completion_task",
                ),
            )

    class NoInputTask:
        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(),
                    source_task="no_input",
                ),
            )

    class JudgeTask:
        model_type = "chat"

        def __init__(self, *, grader=None, models_by_role=None):
            del grader, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="judge_task",
                ),
                TaskModelRequirement(
                    role="grader",
                    binding=context.model_bindings["grader"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="judge_task",
                ),
            )

    class ExtractorTask:
        model_type = "chat"

        def __init__(self, *, extractor=None, models_by_role=None):
            del extractor, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="extractor_task",
                ),
                TaskModelRequirement(
                    role="extractor",
                    binding=context.model_bindings["extractor"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="extractor_task",
                ),
            )

    class ScoringJudgeTask:
        model_type = "chat"

        def __init__(self, *, grader=None, models_by_role=None):
            del grader, models_by_role

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="scoring_judge_task",
                ),
                TaskModelRequirement(
                    role="grader",
                    binding=context.model_bindings["grader"],
                    requires=TaskRequirements(
                        input=InputKind.COMPLETION,
                        input_scoring=True,
                        sampled_logprobs=True,
                        min_top_logprobs=1,
                    ),
                    source_task="scoring_judge_task",
                ),
            )

    @staticmethod
    def _config(tmp_path: Path, body: str) -> Path:
        return _write_yaml_config(tmp_path, "prelaunch.yaml", body)

    async def _external_seed_contract_entry(
        self,
        tmp_path: Path,
        *,
        requested_model_id: str = "org/external-grader",
        api_base: str = "https://external.example:8000/v1",
        completion: bool = False,
        explicit_seed: int | None = None,
    ) -> dict[str, Any]:
        from sieval.core.models import ChatModel, GenModel

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        if completion:
            if explicit_seed is None:
                external: Model = GenModel(
                    model=requested_model_id,
                    api_base=api_base,
                    api_key="external-key",
                )
            else:
                external = GenModel(
                    model=requested_model_id,
                    api_base=api_base,
                    api_key="external-key",
                    seed=explicit_seed,
                )
        else:
            if explicit_seed is None:
                external = ChatModel(
                    model=requested_model_id,
                    api_base=api_base,
                    api_key="external-key",
                )
            else:
                external = ChatModel(
                    model=requested_model_id,
                    api_base=api_base,
                    api_key="external-key",
                    seed=explicit_seed,
                )
        session = EvalSession(config_path)
        tasks = cast(dict[str, Any], session.config["tasks"])
        task = cast(dict[str, Any], tasks["judged"])
        cast(dict[str, Any], task["args"])["grader"] = external
        task_class = self.ScoringJudgeTask if completion else self.JudgeTask

        try:
            with patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=task_class,
            ):
                session.prepare_prelaunch()
                session._stamp_deterministic_seed_contract()
            contract = cast(
                dict[str, Any],
                session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY],
            )
            entry = cast(dict[str, Any], contract["external_roles"]["judged.grader"])
            return dict(entry)
        finally:
            await external.aclose()

    @pytest.mark.anyio
    async def test_external_seed_contract_uses_stable_semantic_identity(
        self, tmp_path: Path
    ) -> None:
        first = await self._external_seed_contract_entry(tmp_path)
        reconstructed = await self._external_seed_contract_entry(tmp_path)
        moved_endpoint = await self._external_seed_contract_entry(
            tmp_path,
            api_base="https://external.example:9000/v1",
        )

        assert first == reconstructed == moved_endpoint
        assert "binding_id" not in first
        assert "runtime_plan_fingerprint" not in first

    @pytest.mark.anyio
    async def test_external_seed_contract_retains_semantic_sensitivity(
        self, tmp_path: Path
    ) -> None:
        baseline = await self._external_seed_contract_entry(tmp_path)
        changed_model = await self._external_seed_contract_entry(
            tmp_path,
            requested_model_id="org/other-grader",
        )
        changed_dialect = await self._external_seed_contract_entry(
            tmp_path,
            completion=True,
        )
        changed_seed = await self._external_seed_contract_entry(
            tmp_path,
            explicit_seed=7,
        )

        assert changed_model != baseline
        assert changed_dialect != baseline
        assert changed_seed != baseline

    def test_implicit_single_model_completion_uses_normalized_hook_evidence(
        self, tmp_path: Path
    ) -> None:
        class CompletionWithMisleadingLegacyMetadata:
            # The resolver must not read this legacy projection.
            model_type = "chat"
            provisional_dialect: str | None = "not-called"

            @classmethod
            def model_requirements_for(cls, context):
                binding = context.model_bindings["candidate"]
                cls.provisional_dialect = binding.dialect_id
                return (
                    TaskModelRequirement(
                        role="candidate",
                        binding=binding,
                        requires=TaskRequirements(input=InputKind.COMPLETION),
                        source_task="implicit_completion",
                    ),
                )

        config_path = self._config(
            tmp_path,
            """
models:
  only:
    name: org/model
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=CompletionWithMisleadingLegacyMetadata,
        ):
            session.prepare_prelaunch()

        assert CompletionWithMisleadingLegacyMetadata.provisional_dialect is None
        finalized = session._normalized_model_bindings["model:only"]
        assert finalized.dialect_id == "openai_completions"
        assert session._model_types_by_root == {"model:only": "gen"}
        assert (
            session._task_requirement_contexts["completion"]
            .model_bindings["candidate"]
            .dialect_id
            == "openai_completions"
        )
        assert session._task_model_requirements[0].binding == finalized

    def test_task_requirement_hook_cannot_replace_context_binding(
        self, tmp_path: Path
    ) -> None:
        class ForgedBindingTask:
            @classmethod
            def model_requirements_for(cls, context):
                binding = dataclasses.replace(
                    context.model_bindings["candidate"],
                    binding_id="model:forged",
                )
                return (
                    TaskModelRequirement(
                        role="candidate",
                        binding=binding,
                        requires=TaskRequirements(input=InputKind.CHAT),
                        source_task="forged_binding",
                    ),
                )

        config_path = self._config(
            tmp_path,
            """
models:
  only:
    name: org/model
tasks:
  forged:
    class: fake.ForgedBindingTask
    dataset:
      class: fake.Dataset
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=ForgedBindingTask,
            ),
            pytest.raises(ValueError, match="changed the normalized binding"),
        ):
            session.prepare_prelaunch()

    def test_configured_role_missing_from_hook_fails_before_model_io(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  incomplete:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ChatTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=(
                    r"ChatTask\.model_requirements_for\(\) did not declare "
                    r"normalized model role\(s\): 'grader'"
                ),
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    @pytest.mark.parametrize(
        ("key", "yaml_value"),
        [
            ("name", "null"),
            ("dataset", "{}"),
            ("model", "null"),
            ("models_by_role", "{}"),
        ],
    )
    def test_composition_owned_task_args_fail_before_model_io(
        self, tmp_path: Path, key: str, yaml_value: str
    ) -> None:
        config_path = self._config(
            tmp_path,
            f"""
models:
  candidate:
    name: org/candidate
tasks:
  invalid:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      {key}: {yaml_value}
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ChatTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(ValueError, match=rf"composition-owned.*{key}"),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    @pytest.mark.parametrize(
        ("role", "yaml_value"),
        [
            ("grader", "null"),
            ("extractor", "null"),
            ("extractor", "self"),
            ("extractor", "{model: org/extractor}"),
        ],
    )
    def test_non_owner_model_role_fails_before_model_io(
        self, tmp_path: Path, role: str, yaml_value: str
    ) -> None:
        from sieval.tasks.gsm8k_0shot_gen import GSM8KZeroShotGenTask

        config_path = self._config(
            tmp_path,
            f"""
models:
  candidate:
    name: org/candidate
tasks:
  invalid:
    class: fake.GSM8KTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      {role}: {yaml_value}
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=GSM8KZeroShotGenTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(ValueError, match=rf"model role.*{role}"),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_derived_binding_evidence_is_resolved_once_for_shared_root(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
  derived:
    base: base
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: derived
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            patch(
                "sieval.cli.leaderboard.session.derive_model_type",
                wraps=derive_model_type,
            ) as resolver,
        ):
            session.prepare_prelaunch()

        resolver.assert_called_once()
        assert session._model_types_by_root == {"model:base": "gen"}
        assert session._normalized_model_bindings["model:base"].dialect_id == (
            "openai_completions"
        )
        assert session._normalized_model_bindings["model:derived"].dialect_id == (
            "openai_completions"
        )

    def test_root_type_assertion_cannot_override_derived_task_evidence(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
    type: chat
  derived:
    base: base
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: derived
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(
                ValueError,
                match=r"root 'base' declares type: chat.*require 'gen'",
            ),
        ):
            session.prepare_prelaunch()

    def test_type_and_dialect_conflict_fails_before_model_io(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  responses:
    name: org/model
    type: gen
    dialect: openai_responses
tasks:
  no_input:
    class: fake.NoInputTask
    dataset:
      class: fake.Dataset
    model: responses
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.NoInputTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=r"legacy type 'gen'.*openai_responses.*must agree",
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_inline_dialect_input_conflict_fails_before_model_io(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
    type: chat
    dialect: openai_chat
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        dialect: openai_completions
        api_base: https://grader.example/v1
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.JudgeTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=(
                    r"input_kind_unsupported.*inline:judged:grader.*"
                    r"openai_completions.*does not accept 'chat' input"
                ),
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_explicit_null_dialect_is_rejected_before_model_io(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: chat
    dialect: null
tasks:
  no_input:
    class: fake.NoInputTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.NoInputTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(ValueError, match=r"m.*dialect must be a non-empty string"),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_unused_derived_dialect_must_match_shared_root_type_before_io(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
    type: chat
  completion_view:
    base: base
    dialect: openai_completions
tasks:
  chat:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: base
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ChatTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=(r"completion_view.*legacy type 'chat'.*openai_completions"),
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_sibling_derived_kinds_conflict_at_shared_root(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
  chat_view:
    base: base
  completion_view:
    base: base
tasks:
  chat:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: chat_view
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: completion_view
""",
        )
        session = EvalSession(config_path)

        def resolve(task_spec: str):
            if task_spec.endswith("ChatTask"):
                return self.ChatTask
            return self.CompletionTask

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                side_effect=resolve,
            ),
            pytest.raises(ValueError, match="conflicting normalized input") as exc,
        ):
            session.prepare_prelaunch()

        message = str(exc.value)
        assert "chat_task" in message
        assert "completion_task" in message
        assert "separate root model configs" in message

    def test_derived_and_root_type_assertions_must_agree(self, tmp_path: Path) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
    type: chat
  derived:
    base: base
    type: gen
tasks: {}
""",
        )
        session = EvalSession(config_path)

        with pytest.raises(
            ValueError,
            match=r"sharing deployment root 'base'.*base='chat'.*derived='gen'",
        ):
            session.prepare_prelaunch()

    def test_external_completions_without_engine_remains_unknown(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: provider/model
    type: gen
    api_base: https://provider.example/v1
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.CompletionTask,
        ):
            session.prepare_prelaunch()

        deployment_input = session._prelaunch_deployment_inputs["model:m"]
        assert deployment_input.engine_id == "unknown"
        deployment = session._configured_deployment_for(
            session._normalized_model_bindings["model:m"],
            deployment_input,
            session._get_named_config_map("models"),
        )
        assert deployment.engine.engine_id == "unknown"
        assert deployment.engine_source == "unknown"
        assert deployment.deployment_id is None
        assert deployment.plan is None

    def test_native_external_dialect_requires_explicit_api_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: claude
    type: chat
    dialect: anthropic_messages
tasks: {}
""",
        )
        monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-family.example/v1")
        session = EvalSession(config_path)

        with pytest.raises(
            ValueError,
            match=r"anthropic_messages.*requires an explicit api_base",
        ):
            session.prepare_prelaunch()

    def test_managed_native_dialect_does_not_require_external_api_base(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: claude
    type: chat
    dialect: anthropic_messages
    infer:
      backend: vllm
      checkpoint: /models/claude
tasks: {}
""",
        )
        raw_plan = {
            "backend": "vllm",
            "checkpoint": "/models/claude",
            "assignments": [],
        }
        session = EvalSession(config_path, infer_plans={"m": raw_plan})
        models_cfg = session._get_named_config_map("models")
        binding = session._finalize_named_binding(
            session._provisional_named_binding("m", models_cfg),
            "chat",
            models_cfg,
        )

        deployment_input = session._deployment_input_for(binding, models_cfg)

        assert deployment_input.engine_id == "vllm"
        assert deployment_input.plan is not None

    def test_explicit_engine_assertion_must_match_deployment_plan(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    engine: sglang
    dialect: openai_completions
    infer:
      backend: vllm
      checkpoint: /models/org-model
tasks: {}
""",
        )
        raw_plan = {
            "backend": "vllm",
            "checkpoint": "/models/org-model",
            "assignments": [],
        }
        session = EvalSession(config_path, infer_plans={"m": raw_plan})

        with pytest.raises(
            ValueError,
            match=r"engine assertion 'sglang'.*deployment engine 'vllm'",
        ):
            session.prepare_prelaunch()

    def test_infer_backend_is_not_a_remote_engine_or_dialect_selector(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    infer:
      backend: sglang
      checkpoint: /models/org-model
tasks: {}
""",
        )
        session = EvalSession(config_path)
        models_cfg = session._get_named_config_map("models")
        binding = session._finalize_named_binding(
            session._provisional_named_binding("m", models_cfg),
            "gen",
            models_cfg,
        )

        assert binding.dialect_id == "openai_completions"
        deployment_input = session._deployment_input_for(binding, models_cfg)
        assert deployment_input.engine_id == "unknown"

    def test_managed_sglang_engine_does_not_select_legacy_dialect(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    infer:
      backend: sglang
      checkpoint: /models/org-model
tasks: {}
""",
        )
        raw_plan = {
            "backend": "sglang",
            "checkpoint": "/models/org-model",
            "assignments": [],
        }
        session = EvalSession(config_path, infer_plans={"m": raw_plan})
        models_cfg = session._get_named_config_map("models")
        binding = session._finalize_named_binding(
            session._provisional_named_binding("m", models_cfg),
            "gen",
            models_cfg,
        )

        assert binding.dialect_id == "openai_completions"
        deployment_input = session._deployment_input_for(binding, models_cfg)
        assert deployment_input.engine_id == "sglang"

    def test_remote_engine_identity_is_independent_of_chat_dialect(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: chat
    engine: hosted-vendor
    dialect: openai_chat
tasks: {}
""",
        )
        session = EvalSession(config_path)

        session.prepare_prelaunch()

        deployment_input = session._prelaunch_deployment_inputs["model:m"]
        assert deployment_input.engine_id == "hosted-vendor"
        deployment = session._configured_deployment_for(
            session._normalized_model_bindings["model:m"],
            deployment_input,
            session._get_named_config_map("models"),
        )
        assert deployment.engine.engine_id == "hosted-vendor"
        assert deployment.engine_source == "config"

    @pytest.mark.parametrize(
        ("field", "error_type", "expected"),
        [
            ("engine", TypeError, "engine must be a non-empty string"),
            ("service_role", ValueError, "service_role must be a non-empty string"),
        ],
    )
    def test_explicit_null_binding_identity_fields_are_not_treated_as_omitted(
        self,
        tmp_path: Path,
        field: str,
        error_type: type[Exception],
        expected: str,
    ) -> None:
        config_path = self._config(
            tmp_path,
            f"""
models:
  m:
    name: org/model
    type: chat
    {field}: null
tasks:
  no_input:
    class: fake.NoInputTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.NoInputTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(error_type, match=expected),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_infer_overrides_are_projected_as_explicit_engine_parameters(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    infer:
      backend: vllm
      checkpoint: /models/org-model
      overrides:
        max-logprobs: 20
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        raw_plan = {
            "backend": "vllm",
            "checkpoint": "/models/org-model",
            "assignments": [],
        }
        session = EvalSession(config_path, infer_plans={"m": raw_plan})

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.CompletionTask,
        ):
            result = session.prepare_prelaunch()

        deployment_input = session._prelaunch_deployment_inputs["model:m"]
        assert deployment_input.explicit_parameters == {"max_logprobs": 20}
        assert deployment_input.recipe_parameters == {}
        deployment_plan = result.deployment_plans["model:m"]
        assert deployment_plan.explicit_parameters == {"max_logprobs": 20}
        assert deployment_input.plan is not None
        assert deployment_plan.desired_plan_fingerprint == (
            deployment_input.plan.fingerprint
        )

    def test_missing_realized_managed_deployment_is_not_synthesized(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    api_base: https://unverified.example/v1
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        raw_plan = {
            "backend": "vllm",
            "checkpoint": "/models/org-model",
            "assignments": [],
        }
        session = EvalSession(config_path, infer_plans={"m": raw_plan})

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.CompletionTask,
        ):
            session.prepare_prelaunch()
            with pytest.raises(ValueError, match="no realized Deployment"):
                session._setup_postlaunch_reconciliation()

    @pytest.mark.anyio
    async def test_registered_future_family_needs_no_session_branch(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.leaderboard import session as session_module
        from sieval.core.models.connection_factory import (
            CONNECTION_FACTORY_REGISTRY,
            ConnectionFactorySpec,
            ConnectionRequest,
        )
        from sieval.core.models.deployment import (
            ConnectionIdentity,
            Deployment,
            Engine,
            ResolvedRoute,
            ServingFacts,
        )
        from sieval.core.models.reconcile import RuntimeBindingPlan

        class FutureConnection:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        connection = FutureConnection()
        requests: list[ConnectionRequest] = []

        def build(request: ConnectionRequest) -> FutureConnection:
            requests.append(request)
            return connection

        future_registry = CONNECTION_FACTORY_REGISTRY.with_factory(
            ConnectionFactorySpec(
                connection_family="future_sdk",
                retry_policy_prefix="future-sdk:max-retries=",
                builder=build,
            )
        )
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: future/model
    api_base: https://future.example/v1
    api_key: future-secret
    max_retries: 5
tasks: {}
""",
        )
        session = EvalSession(config_path)
        binding = NamedModelBinding(
            binding_id="model:m",
            root_deployment_key="model:m",
            requested_model_id="future/model",
            config_name="m",
            dialect_id="future_dialect",
        )
        models_cfg = session._get_named_config_map("models")

        with (
            patch.object(
                session_module,
                "CONNECTION_FACTORY_REGISTRY",
                future_registry,
            ),
            patch.object(
                session_module,
                "get_dialect_spec",
                return_value=types.SimpleNamespace(connection_family="future_sdk"),
            ),
        ):
            scope = session._connection_scope_for(binding, models_cfg)
            identity = ConnectionIdentity(
                endpoint="https://future.example/v1",
                connection_family="future_sdk",
                credential_scope=scope.credential_scope,
                retry_policy=scope.retry_policy,
                quota_scope=scope.quota_scope,
            )
            route = ResolvedRoute(
                service_role="default",
                endpoint=identity.endpoint,
                connection_family="future_sdk",
                fingerprint="future-route",
            )
            runtime_plan = cast(
                RuntimeBindingPlan,
                types.SimpleNamespace(
                    dialect_id="future_dialect",
                    resolved_route=route,
                    connection_identity=identity,
                ),
            )
            deployment = Deployment(
                deployment_id=None,
                plan=None,
                engine=Engine("future-engine"),
                engine_source="config",
                api_base=identity.endpoint,
                endpoints={},
                topology=None,
                metrics_url=None,
                facts=ServingFacts(),
            )
            pool = session._create_owned_pool(
                "model:m",
                deployment,
                runtime_plan,
                models_cfg["m"],
                nested_args=True,
            )

        assert scope.retry_policy == "future-sdk:max-retries=5"
        assert pool.connection is connection
        assert len(requests) == 1
        assert requests[0].endpoint == "https://future.example/v1"
        assert requests[0].credential == "future-secret"
        assert "future-secret" not in repr(identity)

        await pool.aclose()
        assert connection.closed is True

    def test_scoring_unknowns_retain_named_request_time_verifier(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
tasks:
  score:
    class: fake.ScoreTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.CompletionScoringTask,
        ):
            prepared = session.prepare_prelaunch()

        result = session.prelaunch_reconcile_result
        assert result is not None
        assert prepared is result
        plan = result.binding_plans["model:m"]
        assert plan.dialect_id == "openai_completions"
        assert plan.pending_capabilities == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
        }
        assert session._aggregated_requirements["model:m"].min_top_logprobs == 7
        checks = result.deployment_plans["model:m"].request_checks
        assert {check.verifier for check in checks} == {"validate_response_channel"}
        assert {check.capability for check in checks} == {
            "input_scoring",
            "sampled_logprobs",
            "top_logprobs",
        }

    @pytest.mark.anyio
    async def test_invalid_dialect_fails_before_model_setup(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    dialect: openai_chat
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)
        setup_models = MagicMock()

        with (
            patch.object(session, "_setup_models", setup_models),
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(ValueError, match=r"legacy type 'gen'.*must agree"),
        ):
            await session._prepare_execution()

        setup_models.assert_not_called()

    def test_explicit_sglang_without_dialect_stays_named_legacy_bypass(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    engine: sglang
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.CompletionTask,
        ):
            session._setup_prelaunch_reconciliation()

        assert session._legacy_bypass_bindings == {"model:m"}
        assert session._normalized_model_bindings["model:m"].dialect_id == (
            "sglang_legacy"
        )
        assert session.prelaunch_reconcile_result is not None
        assert not session.prelaunch_reconcile_result.binding_plans

    def test_explicit_sglang_uses_native_dialect_when_binder_is_active(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    engine: sglang
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)
        models_cfg = session._get_named_config_map("models")
        provisional = session._provisional_named_binding("m", models_cfg)
        assert provisional.dialect_id is None

        with patch(
            "sieval.cli.leaderboard.session.dialect_is_bindable",
            return_value=True,
        ) as bindable:
            binding = session._finalize_named_binding(provisional, "gen", models_cfg)

        bindable.assert_called_once_with("sglang_native")
        assert binding.dialect_id == "sglang_native"

    def test_sglang_legacy_bypass_rejects_capability_declarations(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    engine: sglang
    capabilities:
      input_scoring: true
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(ValueError, match="sglang_legacy bypass"),
        ):
            session._setup_prelaunch_reconciliation()

    def test_model_args_cannot_repeat_canonical_reasoning(self, tmp_path: Path) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    dialect: openai_chat
    capabilities:
      reasoning:
        effort: high
    args:
      reasoning_effort: high
tasks:
  chat:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ChatTask,
            ),
            pytest.raises(
                ValueError,
                match=(
                    r"models\.m\.capabilities and models\.m\.args.*"
                    r"reasoning via reasoning_effort"
                ),
            ),
        ):
            session.prepare_prelaunch()

    def test_inherited_capability_conflicts_with_derived_model_args(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
    dialect: openai_chat
    capabilities:
      reasoning: false
  derived:
    base: base
    args:
      reasoning_effort: low
tasks:
  chat:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: derived
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ChatTask,
            ),
            pytest.raises(
                ValueError,
                match=(
                    r"models\.derived\.capabilities and models\.derived\.args.*"
                    r"reasoning via reasoning_effort"
                ),
            ),
        ):
            session.prepare_prelaunch()

    @pytest.mark.parametrize(
        ("capability", "legacy_key", "legacy_value", "dialect", "task_class"),
        [
            ("function_tools", "tools", [], "openai_chat", ChatTask),
            (
                "structured_output",
                "response_format",
                {"type": "json_object"},
                "openai_chat",
                ChatTask,
            ),
            ("input_scoring", "echo", False, "openai_completions", CompletionTask),
            (
                "stateful_session",
                "session_id",
                "prior-response",
                "openai_chat",
                ChatTask,
            ),
            ("fim", "suffix", "tail", "openai_completions", CompletionTask),
        ],
    )
    def test_infer_args_cannot_repeat_canonical_capability(
        self,
        tmp_path: Path,
        capability: str,
        legacy_key: str,
        legacy_value: object,
        dialect: str,
        task_class: type,
    ) -> None:
        config = {
            "models": {
                "m": {
                    "name": "org/model",
                    "dialect": dialect,
                    # false is still an explicit canonical owner; allowing a
                    # legacy argument to override it would restore last-wins.
                    "capabilities": {capability: False},
                }
            },
            "tasks": {
                "eval": {
                    "class": "fake.Task",
                    "dataset": {"class": "fake.Dataset"},
                    "model": "m",
                    "infer_args": {legacy_key: legacy_value},
                }
            },
        }
        config_path = self._config(tmp_path, yaml.safe_dump(config, sort_keys=False))
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=task_class,
            ),
            pytest.raises(
                ValueError,
                match=rf"tasks\.eval\.infer_args.*{capability} via {legacy_key}",
            ),
        ):
            session.prepare_prelaunch()

    def test_sampling_infer_args_remain_outside_capability_ambiguity(
        self, tmp_path: Path
    ) -> None:
        config = {
            "models": {
                "m": {
                    "name": "org/model",
                    "dialect": "openai_chat",
                    "capabilities": {"reasoning": False},
                }
            },
            "tasks": {
                "chat": {
                    "class": "fake.ChatTask",
                    "dataset": {"class": "fake.Dataset"},
                    "model": "m",
                    "infer_args": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_tokens": 128,
                        "stop": ["done"],
                        "seed": 7,
                    },
                }
            },
        }
        config_path = self._config(tmp_path, yaml.safe_dump(config, sort_keys=False))
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.ChatTask,
        ):
            result = session.prepare_prelaunch()

        assert result.is_valid

    def test_active_infer_args_enter_reconcile_before_runtime(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    dialect: openai_completions
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
    infer_args:
      reasoning_effort: high
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(ValueError, match=r"dialect_unsupported.*reasoning"),
        ):
            session.prepare_prelaunch()

    def test_active_model_args_enter_reconcile_before_runtime(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    dialect: openai_completions
    args:
      response_format:
        type: json_object
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(
                ValueError,
                match=r"dialect_unsupported.*structured_output",
            ),
        ):
            session.prepare_prelaunch()

    def test_infer_logprobs_contributes_numeric_minimum(self, tmp_path: Path) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    dialect: openai_chat
tasks:
  chat:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: m
    infer_args:
      logprobs: 8
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.ChatTask,
        ):
            result = session.prepare_prelaunch()

        plan = result.binding_plans["model:m"]
        assert plan.required_capabilities >= {
            "sampled_logprobs",
            "top_logprobs",
        }
        assert plan.capability_minimums["top_logprobs"] == {"minimum": 8}
        assert {
            check.capability
            for check in result.deployment_plans["model:m"].request_checks
        } == {"sampled_logprobs", "top_logprobs"}

    def test_inline_grader_is_normalized_before_task_construction(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        api_base: https://grader.example/v1
        api_key: secret-value
        temperature: 0
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.JudgeTask,
        ):
            session._setup_prelaunch_reconciliation()

        context = session._task_requirement_contexts["judged"]
        assert "grader" not in context.task_args
        grader = context.model_bindings["grader"]
        assert isinstance(grader, InlineModelBinding)
        assert grader.dialect_id == "openai_chat"
        assert "api_key" not in grader.config
        grader_source = cast(
            dict[str, object],
            session._task_role_model_sources["judged"]["grader"],
        )
        assert grader_source["api_key"] == "secret-value"
        assert {record.role for record in session._task_model_requirements} == {
            "candidate",
            "grader",
        }
        result = session.prelaunch_reconcile_result
        assert result is not None
        assert set(result.binding_plans) == {
            "model:candidate",
            grader.binding_id,
        }

    def test_nested_inline_secret_is_redacted_from_stable_binding_identity(
        self, tmp_path: Path
    ) -> None:
        bindings: list[InlineModelBinding] = []
        plan_fingerprints: list[str] = []

        with patch(
            "sieval.core.models.connection_factory.AsyncOpenAI"
        ) as client_factory:
            for secret in ("SECRET-A", "SECRET-B"):
                config_path = self._config(
                    tmp_path,
                    f"""
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        args:
          api_key: {secret}
          temperature: 0
""",
                )
                session = EvalSession(config_path)
                with patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=self.JudgeTask,
                ):
                    result = session.prepare_prelaunch()

                binding = session._task_requirement_contexts["judged"].model_bindings[
                    "grader"
                ]
                assert isinstance(binding, InlineModelBinding)
                bindings.append(binding)
                plan_fingerprints.append(
                    result.binding_plans[binding.binding_id].fingerprint
                )
                source = session._task_role_model_sources["judged"]["grader"]
                assert secret in repr(source)

        first, second = bindings
        assert first.binding_id == second.binding_id
        assert first.config == second.config
        assert first.config["args"] == {"temperature": 0}
        assert "SECRET-A" not in repr(first.config)
        assert "SECRET-B" not in repr(second.config)
        assert plan_fingerprints[0] == plan_fingerprints[1]
        client_factory.assert_not_called()

    @pytest.mark.parametrize(
        ("nested", "expected_path"),
        [(False, "helper"), (True, "settings.helper")],
    )
    def test_unregistered_model_task_arg_fails_before_model_io(
        self,
        tmp_path: Path,
        nested: bool,
        expected_path: str,
    ) -> None:
        class HelperTask(self.ChatTask):
            def __init__(self, *, helper=None, settings=None):
                del helper, settings

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  hidden:
    class: fake.HelperTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        hidden_model = MockChatModel()
        session = EvalSession(config_path)
        tasks = session.config["tasks"]
        assert isinstance(tasks, dict)
        task = tasks["hidden"]
        args = task["args"]
        assert isinstance(args, dict)
        if nested:
            args["settings"] = {"helper": hidden_model}
        else:
            args["helper"] = hidden_model

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=HelperTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=rf"outside registered model roles: {re.escape(expected_path)}",
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    def test_inline_sglang_legacy_role_fails_during_prelaunch(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        dialect: sglang_legacy
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.JudgeTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=r"inline grader.*sglang_legacy.*named model",
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    @pytest.mark.anyio
    async def test_postlaunch_binds_pool_per_identity_and_shares_root_limiter(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import Deployment, Engine, GenModel, ServingFacts

        config_path = self._config(
            tmp_path,
            """
models:
  leaf:
    base: derived
    type: gen
    args:
      temperature: 0.3
  base:
    name: org/model
    type: gen
    api_base: https://configured.example/v1
    api_key: local
    service_role: decode
    args:
      concurrency_limit: 8
      temperature: 0.1
  derived:
    base: base
    type: gen
    service_role: prefill
    args:
      concurrency_limit: 2
      temperature: 0.2
tasks:
  root_task:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: base
  child_task:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: derived
  leaf_task:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: leaf
""",
        )
        realized = Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine("vllm"),
            engine_source="deployment",
            api_base="https://realized.example/v1",
            endpoints={
                "prefill": "https://prefill.example/v1",
                "decode": "https://decode.example/v1",
            },
            topology=None,
            metrics_url=None,
            facts=ServingFacts(engine_version="test"),
        )
        session = EvalSession(
            config_path,
            realized_deployments={"base": realized},
        )
        closes = [AsyncMock(), AsyncMock()]
        connections = [types.SimpleNamespace(close=close) for close in closes]

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI",
                side_effect=connections,
            ) as client_factory,
        ):
            session._setup_prelaunch_reconciliation()
            session._setup_postlaunch_reconciliation()
            session._setup_models()

        base = session.models["base"]
        derived = session.models["derived"]
        leaf = session.models["leaf"]
        assert isinstance(base, GenModel)
        assert isinstance(derived, GenModel)
        assert isinstance(leaf, GenModel)
        assert base.pool is not derived.pool
        assert derived.pool is leaf.pool
        assert base.pool.shared_limiter is derived.pool.shared_limiter
        assert base.deployment is realized
        assert base.runtime_plan is not None
        assert derived.runtime_plan is not None
        assert leaf.runtime_plan is not None
        assert base.runtime_plan.binding_id == "model:base"
        assert derived.runtime_plan.binding_id == "model:derived"
        assert leaf.runtime_plan.binding_id == "model:leaf"
        assert base._kwargs["temperature"] == 0.1
        assert derived._kwargs["temperature"] == 0.2
        assert leaf._kwargs["temperature"] == 0.3
        assert derived._parent_limiter is derived.pool.shared_limiter
        assert leaf._limiter is derived._limiter
        assert leaf._parent_limiter is derived.pool.shared_limiter
        assert len(session._owned_pools) == 2
        assert {call.kwargs["base_url"] for call in client_factory.call_args_list} == {
            "https://decode.example/v1",
            "https://prefill.example/v1",
        }
        assert all(
            call.kwargs["api_key"] == "local" and call.kwargs["max_retries"] == 3
            for call in client_factory.call_args_list
        )

        await session._close_owned_model_resources()
        await session._close_owned_model_resources()
        for close in closes:
            close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_managed_pre_and_postlaunch_keep_connection_scope_identity(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import Deployment, Engine, ServingFacts
        from sieval.infer import deployment_plan_projection

        config_path = self._config(
            tmp_path,
            """
models:
  base:
    name: org/model
    type: gen
    max_retries: 9
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: base
""",
        )
        raw_plan = {
            "backend": "vllm",
            "checkpoint": "/models/org-model",
            "assignments": [],
        }
        plan = deployment_plan_projection(raw_plan)
        realized = Deployment(
            deployment_id="served-base",
            plan=plan,
            engine=Engine("vllm"),
            engine_source="deployment",
            api_base="https://realized.example/v1",
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(engine_version="test"),
        )
        session = EvalSession(
            config_path,
            infer_plans={"base": raw_plan},
            realized_deployments={"base": realized},
        )
        close = AsyncMock()
        connection = types.SimpleNamespace(close=close)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=self.CompletionTask,
        ):
            prelaunch = session.prepare_prelaunch()
            pre_binding = prelaunch.binding_plans["model:base"]
            assert pre_binding.connection_scope.credential_scope == (
                "model:base:managed-local-credential"
            )
            assert pre_binding.connection_scope.retry_policy == (
                "openai-sdk:max-retries=9"
            )
            session._setup_postlaunch_reconciliation()

        postlaunch = session.postlaunch_reconcile_result
        assert postlaunch is not None
        post_binding = postlaunch.binding_plans["model:base"]
        runtime = postlaunch.runtime_plans["model:base"]
        assert post_binding.fingerprint == pre_binding.fingerprint
        assert runtime.binding_plan_fingerprint == pre_binding.fingerprint
        assert runtime.connection_identity.credential_scope == (
            pre_binding.connection_scope.credential_scope
        )
        assert runtime.connection_identity.retry_policy == (
            pre_binding.connection_scope.retry_policy
        )
        assert runtime.connection_identity.quota_scope == (
            pre_binding.connection_scope.quota_scope
        )
        assert runtime.connection_identity.endpoint == "https://realized.example/v1"

        with patch(
            "sieval.core.models.connection_factory.AsyncOpenAI",
            return_value=connection,
        ) as client_factory:
            session._setup_models()
        client_factory.assert_called_once_with(
            base_url="https://realized.example/v1",
            api_key=None,
            max_retries=9,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        await session._close_owned_model_resources()
        close.assert_awaited_once()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("api_key", "child-key"),
            ("api_base", "https://child.example/v1"),
            ("engine", "sglang"),
            ("max_retries", 7),
            ("connection_family", "native-http"),
            ("authorization", "Bearer secret"),
        ],
    )
    def test_derived_model_cannot_override_root_binding_resources(
        self, tmp_path: Path, field: str, value: object
    ) -> None:
        config = {
            "models": {
                "base": {
                    "name": "org/model",
                    "type": "gen",
                    "api_base": "https://base.example/v1",
                    "api_key": "base-key",
                },
                "child": {"base": "base", "type": "gen", field: value},
            },
            "tasks": {
                "completion": {
                    "class": "fake.CompletionTask",
                    "dataset": {"class": "fake.Dataset"},
                    "model": "child",
                }
            },
        }
        config_path = self._config(tmp_path, yaml.safe_dump(config, sort_keys=False))
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(ValueError, match="places binding resource"),
        ):
            session.prepare_prelaunch()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("api_key", "child-key"),
            ("api_base", "https://child.example/v1"),
            ("max_retries", 7),
            ("connection_family", "native-http"),
            ("authorization", "Bearer secret"),
        ],
    )
    def test_derived_model_args_cannot_override_root_binding_resources(
        self, tmp_path: Path, field: str, value: object
    ) -> None:
        config = {
            "models": {
                "base": {
                    "name": "org/model",
                    "type": "gen",
                    "api_base": "https://base.example/v1",
                },
                "child": {
                    "base": "base",
                    "type": "gen",
                    "args": {field: value},
                },
            },
            "tasks": {
                "completion": {
                    "class": "fake.CompletionTask",
                    "dataset": {"class": "fake.Dataset"},
                    "model": "child",
                }
            },
        }
        config_path = self._config(tmp_path, yaml.safe_dump(config, sort_keys=False))
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            pytest.raises(ValueError, match=rf"args\.{field}"),
        ):
            session.prepare_prelaunch()

    def test_task_infer_args_cannot_change_connection_family(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
tasks:
  chat:
    class: fake.ChatTask
    dataset:
      class: fake.Dataset
    model: m
    infer_args:
      connection_family: native-http
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ChatTask,
            ),
            pytest.raises(
                ValueError, match="infer_args cannot change.*connection_family"
            ),
        ):
            session.prepare_prelaunch()

    def test_inline_grader_rejects_authorization_before_redaction(
        self, tmp_path: Path
    ) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        api_base: https://grader.example/v1
        authorization: Bearer-secret
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.JudgeTask,
            ),
            pytest.raises(ValueError, match="inline grader.*authorization"),
        ):
            session.prepare_prelaunch()

    def test_inline_grader_rejects_non_string_engine(self, tmp_path: Path) -> None:
        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        engine: 17
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.JudgeTask,
            ),
            pytest.raises(TypeError, match="Inline binding.*engine"),
        ):
            session.prepare_prelaunch()

    def test_self_extractor_reuses_candidate_requirement(self, tmp_path: Path) -> None:
        from sieval.tasks.agieval_0shot_gen import AGIEvalZeroShotGenTask

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  extracted:
    class: fake.AGIEvalTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      extractor: self
    infer_args:
      max_tokens: 4096
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=AGIEvalZeroShotGenTask,
        ):
            session.prepare_prelaunch()

        context = session._task_requirement_contexts["extracted"]
        assert set(context.model_bindings) == {"candidate"}
        assert context.task_args["extractor"] == "self"
        assert context.infer_args["max_tokens"] == 4096
        assert [item.role for item in session._task_model_requirements] == ["candidate"]

    def test_agieval_inline_extractor_declares_chat_requirement(
        self, tmp_path: Path
    ) -> None:
        from sieval.tasks.agieval_0shot_gen import AGIEvalZeroShotGenTask

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  extracted:
    class: fake.AGIEvalTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      extractor:
        model: org/extractor
        api_base: https://extractor.example/v1
""",
        )
        session = EvalSession(config_path)

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=AGIEvalZeroShotGenTask,
        ):
            session.prepare_prelaunch()

        context = session._task_requirement_contexts["extracted"]
        requirements = {item.role: item for item in session._task_model_requirements}
        assert set(requirements) == {"candidate", "extractor"}
        assert requirements["extractor"].binding is context.model_bindings["extractor"]
        assert requirements["extractor"].requires.input is InputKind.CHAT

    def test_invalid_extractor_string_fails_before_model_io(
        self, tmp_path: Path
    ) -> None:
        from sieval.tasks.agieval_0shot_gen import AGIEvalZeroShotGenTask

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  extracted:
    class: fake.AGIEvalTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      extractor: itself
""",
        )
        session = EvalSession(config_path)

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=AGIEvalZeroShotGenTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI"
            ) as client_factory,
            pytest.raises(
                ValueError,
                match=(
                    "Task 'extracted' extractor must be 'self', an inline mapping, "
                    "or Model"
                ),
            ),
        ):
            session.prepare_prelaunch()

        client_factory.assert_not_called()

    @pytest.mark.anyio
    async def test_postlaunch_inline_extractor_gets_own_pool_and_role_binding(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import ChatModel

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  candidate:
    name: org/candidate
    api_base: https://candidate.example/v1
    api_key: candidate-key
tasks:
  extracted:
    class: fake.ExtractorTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      extractor:
        model: org/extractor
        api_base: https://extractor.example/v1
        api_key: extractor-key
        temperature: 0
        seed: 17
""",
        )
        session = EvalSession(config_path)
        closes = [AsyncMock(), AsyncMock()]
        connections = [types.SimpleNamespace(close=close) for close in closes]

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ExtractorTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI",
                side_effect=connections,
            ) as client_factory,
        ):
            session._setup_prelaunch_reconciliation()
            tasks = cast(dict, session.config["tasks"])
            extracted = cast(dict, tasks["extracted"])
            task_args = cast(dict, extracted["args"])
            cast(dict, task_args["extractor"])["seed"] = 71
            session._setup_postlaunch_reconciliation()
            session._setup_models()
            session._stamp_deterministic_seed_contract()

        extractor = session._bound_task_role_models["extracted"]["extractor"]
        assert isinstance(session.models["candidate"], ChatModel)
        assert isinstance(extractor, ChatModel)
        assert session.models["candidate"]._kwargs["seed"] == 0
        assert extractor._kwargs["seed"] == 17
        binding_id = (
            session._task_requirement_contexts["extracted"]
            .model_bindings["extractor"]
            .binding_id
        )
        contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
        inline_contract = contract["bindings"][binding_id]
        assert inline_contract["seed"] == 17
        assert inline_contract["seed_provenance"] == "binding_config"
        assert extractor is not session.models["candidate"]
        assert extractor.pool is not session.models["candidate"].pool
        assert len(session._owned_pools) == 2
        assert {call.kwargs["base_url"] for call in client_factory.call_args_list} == {
            "https://candidate.example/v1",
            "https://extractor.example/v1",
        }

        await session._close_owned_model_resources()
        for close in closes:
            close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_postlaunch_inline_grader_gets_own_pool_and_role_binding(
        self, tmp_path: Path, loguru_caplog
    ) -> None:
        from sieval.core.models import ChatModel

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  candidate:
    name: org/candidate
    api_base: https://candidate.example/v1
    api_key: candidate-key
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args:
      grader:
        model: org/grader
        api_base: https://grader.example/v1
        api_key: grader-key
        temperature: 0
""",
        )
        session = EvalSession(config_path)
        closes = [AsyncMock(), AsyncMock()]
        connections = [types.SimpleNamespace(close=close) for close in closes]

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.JudgeTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI",
                side_effect=connections,
            ) as client_factory,
        ):
            session._setup_prelaunch_reconciliation()
            session._setup_postlaunch_reconciliation()
            session._setup_models()

        grader = session._bound_task_role_models["judged"]["grader"]
        assert isinstance(session.models["candidate"], ChatModel)
        assert isinstance(grader, ChatModel)
        assert session.models["candidate"]._kwargs["seed"] == 0
        assert grader._kwargs["seed"] == 0
        assert grader is not session.models["candidate"]
        assert grader.pool is not session.models["candidate"].pool
        assert len(session._owned_pools) == 2
        assert {call.kwargs["base_url"] for call in client_factory.call_args_list} == {
            "https://candidate.example/v1",
            "https://grader.example/v1",
        }
        assert any(
            "judged.grader" in record.message
            for record in loguru_caplog.records
            if "best-effort" in record.message
        )

        await session._close_owned_model_resources()
        for close in closes:
            close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_external_grader_pool_is_borrowed_not_closed(
        self, tmp_path: Path, loguru_caplog
    ) -> None:
        from sieval.core.models import ChatModel

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  candidate:
    name: org/candidate
    api_base: https://candidate.example/v1
    api_key: candidate-key
tasks:
  judged:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        external = ChatModel(
            model="org/external-grader",
            api_base="https://external.example/v1",
            api_key="external-key",
        )
        session = EvalSession(config_path)
        tasks = session.config["tasks"]
        assert isinstance(tasks, dict)
        task = tasks["judged"]
        args = task["args"]
        assert isinstance(args, dict)
        args["grader"] = external
        candidate_close = AsyncMock()
        candidate_connection = types.SimpleNamespace(close=candidate_close)
        external_close = AsyncMock()

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.JudgeTask,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI",
                return_value=candidate_connection,
            ),
            patch.object(external.pool, "aclose", external_close),
        ):
            session._setup_prelaunch_reconciliation()
            session._setup_postlaunch_reconciliation()
            session._setup_models()
            rebound = session._bound_task_role_models["judged"]["grader"]
            assert rebound is not external
            assert rebound.pool is external.pool
            assert "seed" not in external._kwargs
            assert rebound._kwargs["seed"] == 0
            assert rebound.runtime_plan is not None
            postlaunch = session.postlaunch_reconcile_result
            assert postlaunch is not None
            assert external.runtime_plan is not None
            assert (
                rebound.runtime_plan
                is postlaunch.runtime_plans[external.runtime_plan.binding_id]
            )
            assert external.pool not in session._owned_pools.values()
            session._stamp_deterministic_seed_contract()
            contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
            external_contract = contract["external_roles"]["judged.grader"]
            assert external_contract["explicit_seed_present"] is False
            assert external_contract["seed"] == DETERMINISTIC_DEFAULT_SEED
            assert external_contract["seed_provenance"] == "automatic"
            assert any(
                "judged.grader" in record.message
                for record in loguru_caplog.records
                if "best-effort" in record.message
            )
            await session._close_owned_model_resources()

        candidate_close.assert_awaited_once()
        external_close.assert_not_awaited()
        await external._client.close()

    @pytest.mark.anyio
    @pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
    async def test_external_roles_with_one_binding_keep_each_source_derivation(
        self, tmp_path: Path, order: tuple[str, str]
    ) -> None:
        from sieval.core.models import ChatModel

        task_blocks = "\n".join(
            f"""  {name}:
    class: fake.JudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {{}}"""
            for name in order
        )
        config_path = self._config(
            tmp_path,
            f"""
deterministic: true
models:
  candidate:
    name: org/candidate
    api_base: https://candidate.example/v1
    api_key: candidate-key
tasks:
{task_blocks}
""",
        )
        base = ChatModel(
            model="org/external-grader",
            api_base="https://external.example/v1",
            api_key="external-key",
            concurrency_limit=8,
        )
        derived = {
            "a": base.with_args(
                temperature=0,
                seed=7,
                extra={"source": "a"},
                concurrency_limit=2,
            ),
            "b": base.with_args(
                temperature=1,
                extra={"source": "b"},
                concurrency_limit=3,
            ),
        }
        assert derived["a"].runtime_plan is derived["b"].runtime_plan

        session = EvalSession(config_path)
        tasks = cast(dict, session.config["tasks"])
        for name in order:
            task = cast(dict, tasks[name])
            cast(dict, task["args"])["grader"] = derived[name]

        candidate_close = AsyncMock()
        candidate_connection = types.SimpleNamespace(close=candidate_close)
        external_close = AsyncMock()
        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=self.JudgeTask,
                ),
                patch(
                    "sieval.core.models.connection_factory.AsyncOpenAI",
                    return_value=candidate_connection,
                ),
                patch.object(base.pool, "aclose", external_close),
            ):
                session._setup_prelaunch_reconciliation()
                # The exact external role defaults are frozen before rebind.
                # Later source mutation must not change contract or execution.
                derived["a"]._kwargs["seed"] = 70
                derived["b"]._kwargs["seed"] = 90
                session._setup_postlaunch_reconciliation()
                session._setup_models()
                session._stamp_deterministic_seed_contract()

                postlaunch = session.postlaunch_reconcile_result
                assert postlaunch is not None
                assert base.runtime_plan is not None
                rebound_plan = postlaunch.runtime_plans[base.runtime_plan.binding_id]
                for name in ("a", "b"):
                    rebound = session._bound_task_role_models[name]["grader"]
                    assert rebound._kwargs == {
                        "temperature": 0 if name == "a" else 1,
                        "seed": 7 if name == "a" else DETERMINISTIC_DEFAULT_SEED,
                    }
                    assert rebound.extra == {"source": name}
                    assert rebound._limiter is derived[name]._limiter
                    assert rebound.pool is base.pool
                    assert rebound.runtime_plan is rebound_plan

                contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
                explicit = contract["external_roles"]["a.grader"]
                automatic = contract["external_roles"]["b.grader"]
                assert explicit["seed"] == 7
                assert explicit["seed_provenance"] == "external_model"
                assert automatic["seed"] == DETERMINISTIC_DEFAULT_SEED
                assert automatic["seed_provenance"] == "automatic"

                assert base.pool not in session._owned_pools.values()
                await session._close_owned_model_resources()

            candidate_close.assert_awaited_once()
            external_close.assert_not_awaited()
        finally:
            await base.aclose()

    @pytest.mark.anyio
    async def test_external_runtime_plan_preserves_request_safety_checks(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import GenModel

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.ScoringJudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        external = GenModel(
            model="org/external-grader",
            api_base="https://external.example/v1",
            api_key="external-key",
        )
        assert external.runtime_plan is not None
        guarded_plan = dataclasses.replace(
            external.runtime_plan,
            request_checks=(
                DeferredCheck(
                    "top_logprobs",
                    CheckStage.REQUEST,
                    "validate_response_channel",
                    "preserve the caller's established response guard",
                ),
            ),
        )
        guarded_external = external.with_dialect(guarded_plan.dialect_id, guarded_plan)
        session = EvalSession(config_path)
        task = cast(dict, cast(dict, session.config["tasks"])["judged"])
        cast(dict, task["args"])["grader"] = guarded_external

        try:
            with patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ScoringJudgeTask,
            ):
                prelaunch = session.prepare_prelaunch()
                session._setup_postlaunch_reconciliation()

            binding_id = guarded_plan.binding_id
            prelaunch_plan = prelaunch.runtime_plans[binding_id]
            postlaunch = session.postlaunch_reconcile_result
            assert postlaunch is not None
            postlaunch_plan = postlaunch.runtime_plans[binding_id]
            expected = {"input_scoring", "sampled_logprobs", "top_logprobs"}
            assert {
                check.capability for check in prelaunch_plan.request_checks
            } == expected
            assert postlaunch_plan.request_checks == prelaunch_plan.request_checks
            assert postlaunch_plan.capability_minimums["top_logprobs"] == {"minimum": 1}
        finally:
            await external.aclose()

    @pytest.mark.anyio
    async def test_custom_reconciler_cannot_erase_external_baseline(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import GenModel

        class BlindReconciler:
            def reconcile(self, requirements, deployment):
                del deployment
                return {
                    requirement.capability: Configured(evidence={"probe": "ok"})
                    for requirement in requirements
                }

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.ScoringJudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        external = GenModel(
            model="org/external-grader",
            api_base="https://external.example/v1",
            api_key="external-key",
        )
        assert external.runtime_plan is not None
        guarded_plan = dataclasses.replace(
            external.runtime_plan,
            request_checks=(
                DeferredCheck(
                    "top_logprobs",
                    CheckStage.REQUEST,
                    "validate_response_channel",
                    "preserve the caller's established response guard",
                ),
            ),
        )
        guarded_external = external.with_dialect(guarded_plan.dialect_id, guarded_plan)
        session = EvalSession(config_path, serving_reconciler=BlindReconciler())
        task = cast(dict, cast(dict, session.config["tasks"])["judged"])
        cast(dict, task["args"])["grader"] = guarded_external

        try:
            with patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ScoringJudgeTask,
            ):
                result = session.prepare_prelaunch()

            rebound = result.runtime_plans[guarded_plan.binding_id]
            assert guarded_plan.request_checks[0] in rebound.request_checks
            evidence = result.deployment_plans[
                guarded_plan.root_deployment_key
            ].outcome_evidence["top_logprobs"]
            assert evidence["plan_fingerprints"] == [guarded_plan.fingerprint]
            assert evidence["injected_reconciler"] == {"probe": "ok"}
        finally:
            await external.aclose()

    @pytest.mark.anyio
    async def test_external_bindings_can_share_root_with_distinct_plans(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import GenModel

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged_a:
    class: fake.ScoringJudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
  judged_b:
    class: fake.ScoringJudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        external = GenModel(
            model="org/external-grader",
            api_base="https://external.example/v1",
            api_key="external-key",
        )
        assert external.runtime_plan is not None
        top_check = DeferredCheck(
            "top_logprobs",
            CheckStage.REQUEST,
            "validate_response_channel",
            "first binding guard",
        )
        input_check = DeferredCheck(
            "input_scoring",
            CheckStage.REQUEST,
            "validate_response_channel",
            "second binding guard",
        )
        first_plan = dataclasses.replace(
            external.runtime_plan,
            request_checks=(top_check,),
        )
        second_plan = dataclasses.replace(
            external.runtime_plan,
            binding_id=f"{external.runtime_plan.binding_id}:sibling",
            binding_plan_fingerprint="external:sibling-binding",
            request_checks=(input_check,),
        )
        first = external.with_dialect(first_plan.dialect_id, first_plan)
        second = external.with_dialect(second_plan.dialect_id, second_plan)
        session = EvalSession(config_path)
        tasks = cast(dict, session.config["tasks"])
        cast(dict, cast(dict, tasks["judged_a"])["args"])["grader"] = first
        cast(dict, cast(dict, tasks["judged_b"])["args"])["grader"] = second

        try:
            with patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ScoringJudgeTask,
            ):
                result = session.prepare_prelaunch()

            assert first_plan.binding_id in result.runtime_plans
            assert second_plan.binding_id in result.runtime_plans
            for binding_id in (first_plan.binding_id, second_plan.binding_id):
                rebound = result.runtime_plans[binding_id]
                assert top_check in rebound.request_checks
                assert input_check in rebound.request_checks
            evidence = result.deployment_plans[
                first_plan.root_deployment_key
            ].outcome_evidence["top_logprobs"]
            assert evidence["plan_fingerprints"] == sorted(
                [first_plan.fingerprint, second_plan.fingerprint]
            )
        finally:
            await external.aclose()

    @pytest.mark.anyio
    async def test_external_postlaunch_rejects_evidence_drift(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import GenModel

        class StatefulReconciler:
            def __init__(self) -> None:
                self.calls: dict[str, int] = {}

            def reconcile(self, requirements, deployment):
                round_ = self.calls.get(deployment.root_deployment_key, 0) + 1
                self.calls[deployment.root_deployment_key] = round_
                return {
                    requirement.capability: Configured(
                        evidence={"observation_round": round_}
                    )
                    for requirement in requirements
                }

        config_path = self._config(
            tmp_path,
            """
models:
  candidate:
    name: org/candidate
tasks:
  judged:
    class: fake.ScoringJudgeTask
    dataset:
      class: fake.Dataset
    model: candidate
    args: {}
""",
        )
        external = GenModel(
            model="org/external-grader",
            api_base="https://external.example/v1",
            api_key="external-key",
        )
        session = EvalSession(
            config_path,
            serving_reconciler=StatefulReconciler(),
        )
        task = cast(dict, cast(dict, session.config["tasks"])["judged"])
        cast(dict, task["args"])["grader"] = external

        try:
            with patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.ScoringJudgeTask,
            ):
                session.prepare_prelaunch()
                with pytest.raises(
                    RuntimeError,
                    match="changed serving evidence or checks",
                ):
                    session._setup_postlaunch_reconciliation()
        finally:
            await external.aclose()

    @pytest.mark.anyio
    async def test_sglang_legacy_remains_outside_runtime_plans(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import SglangGenModel

        config_path = self._config(
            tmp_path,
            """
models:
  m:
    name: org/model
    type: gen
    engine: sglang
    api_base: https://sglang.example/v1
    api_key: local
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)
        close = AsyncMock()
        connection = types.SimpleNamespace(
            close=close,
            base_url="https://sglang.example/v1",
        )

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=self.CompletionTask,
            ),
            patch(
                "sieval.core.models.sglang_gen_model.AsyncOpenAI",
                return_value=connection,
            ),
        ):
            session._setup_prelaunch_reconciliation()
            session._setup_postlaunch_reconciliation()
            session._setup_models()

        assert session.postlaunch_reconcile_result is not None
        assert not session.postlaunch_reconcile_result.runtime_plans
        assert isinstance(session.models["m"], SglangGenModel)
        assert not session._owned_pools
        assert set(session._owned_legacy_models) == {"model:m"}

        await session._close_owned_model_resources()
        close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_deterministic_sglang_legacy_does_not_auto_inject_request_seed(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import (
            Request,
            Response,
            SglangGenModel,
            TokenLogprob,
        )

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  m:
    name: org/model
    type: gen
    engine: sglang
    api_base: https://sglang.example/v1
    api_key: local
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)
        close = AsyncMock()
        connection = types.SimpleNamespace(
            close=close,
            base_url="https://sglang.example/v1",
        )
        transport_arun = AsyncMock(
            return_value=Response(
                texts=("ok",),
                logprobs=(TokenLogprob("ok", -0.1),),
            )
        )

        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=self.CompletionTask,
                ),
                patch(
                    "sieval.core.models.sglang_gen_model.AsyncOpenAI",
                    return_value=connection,
                ),
            ):
                session._setup_prelaunch_reconciliation()
                session._setup_postlaunch_reconciliation()
                session._setup_models()
                session._stamp_deterministic_seed_contract()

            model = session.models["m"]
            assert isinstance(model, SglangGenModel)
            assert "seed" not in model._kwargs
            contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
            binding = contract["bindings"]["model:m"]
            assert binding["request_seed_support"] == "unsupported"
            assert binding["seed_scope"] == "engine_level_only"
            assert binding["seed_present"] is False
            assert binding["seed"] is None
            assert binding["seed_provenance"] == "none"
            with patch.object(model._legacy_transport, "arun", transport_arun):
                generated = await model.agenerate("prompt")
                scored = await model.alogprobs("prompt", echo=False)

            assert generated.texts == ["ok"]
            assert scored.texts == ["ok"]
            requests = [
                cast(Request, call.args[0]) for call in transport_arun.await_args_list
            ]
            assert len(requests) == 2
            assert all(request.sampling.seed is None for request in requests)
        finally:
            await session._close_owned_model_resources()

        close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_explicit_sglang_legacy_seed_remains_engine_level_noop(
        self, tmp_path: Path
    ) -> None:
        from sieval.core.models import Request, Response, SglangGenModel

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  m:
    name: org/model
    type: gen
    engine: sglang
    api_base: https://sglang.example/v1
    api_key: local
    args:
      seed: 7
tasks:
  completion:
    class: fake.CompletionTask
    dataset:
      class: fake.Dataset
    model: m
""",
        )
        session = EvalSession(config_path)
        close = AsyncMock()
        connection = types.SimpleNamespace(
            close=close,
            base_url="https://sglang.example/v1",
        )
        transport_arun = AsyncMock(return_value=Response(texts=("ok",)))

        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=self.CompletionTask,
                ),
                patch(
                    "sieval.core.models.sglang_gen_model.AsyncOpenAI",
                    return_value=connection,
                ),
            ):
                session._setup_prelaunch_reconciliation()
                session._setup_postlaunch_reconciliation()
                session._setup_models()

            model = session.models["m"]
            assert isinstance(model, SglangGenModel)
            assert model._kwargs["seed"] == 7
            with patch.object(model._legacy_transport, "arun", transport_arun):
                output = await model.agenerate("prompt")

            assert output.texts == ["ok"]
            call = transport_arun.await_args
            assert call is not None
            request = cast(Request, call.args[0])
            assert request.sampling.seed == 7
        finally:
            await session._close_owned_model_resources()

        close.assert_awaited_once()

    @pytest.mark.anyio
    async def test_deterministic_legacy_candidate_survives_task_setup(
        self, tmp_path: Path
    ) -> None:
        """`_setup_tasks` applies the frozen decision to a `SglangGenModel`.

        The legacy facade is the one candidate that reaches
        `_apply_request_seed_decision_to_model` as a `Model` *subclass* whose
        `with_dialect` raises, so the helper's dialect guard has to be
        satisfied by `SglangGenModel.dialect_id` rather than a runtime plan.
        """
        from sieval.core.models import SglangGenModel
        from tests.unit.core.runners.test_runner import MockTask

        class LegacyGenTask(MockTask):
            model_type = "gen"

        config_path = self._config(
            tmp_path,
            """
deterministic: true
models:
  m:
    name: org/model
    type: gen
    engine: sglang
    api_base: https://sglang.example/v1
    api_key: local
datasets:
  ds:
    class: fake.Dataset
tasks:
  completion:
    class: fake.LegacyGenTask
    dataset: ds
    model: m
""",
        )
        session = EvalSession(config_path)
        close = AsyncMock()
        connection = types.SimpleNamespace(
            close=close,
            base_url="https://sglang.example/v1",
        )

        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=LegacyGenTask,
                ),
                patch(
                    "sieval.core.models.sglang_gen_model.AsyncOpenAI",
                    return_value=connection,
                ),
            ):
                session._setup_prelaunch_reconciliation()
                session._setup_postlaunch_reconciliation()
                session._setup_models()
                session._init_runner()
                session.datasets["ds"] = MagicMock()
                session._setup_tasks()

            assert session.runner is not None
            task_model = session.runner._runners[0]._task.model
            assert isinstance(task_model, SglangGenModel)
            assert task_model.dialect_id == "sglang_legacy"
            assert "seed" not in task_model._kwargs
            # The helper derives rather than mutates, so a distinct object is
            # the evidence that it ran against the legacy facade at all.
            assert task_model is not session.models["m"]
            assert "seed" not in session.models["m"]._kwargs
        finally:
            await session._close_owned_model_resources()


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
            realized_deployments=None,
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
            realized_deployments=None,
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
            None,
        )
        assert result == {"task_b": {"ok": True}}

    @pytest.mark.anyio
    async def test_arun_session_forwards_legacy_external_endpoint_adapter(self):
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
            realized_deployments=None,
        )

    def test_run_session_forwards_legacy_external_endpoint_adapter_positionally(
        self,
    ):
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
            None,
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
            patch.object(runner, "prepare_prelaunch", MagicMock(return_value=None)),
            patch.object(runner, "_stamp_deterministic_seed_contract"),
            patch.object(runner, "_prepare_execution", AsyncMock(return_value=None)),
            patch.object(
                runner, "_persist_effective_config", AsyncMock(return_value=None)
            ),
            patch.object(runner, "_persist_infer_plans", AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="Runner not initialized"),
        ):
            await runner.arun()

    @pytest.mark.anyio
    async def test_arun_closes_owned_pool_once_when_runner_fails(self):
        runner = object.__new__(EvalSession)
        runner.runner = types.SimpleNamespace(
            arun=AsyncMock(side_effect=RuntimeError("evaluation failed"))
        )
        pool = types.SimpleNamespace(aclose=AsyncMock())
        runner._owned_pools = {"model:m": pool}
        runner._owned_legacy_models = {}

        with (
            patch.object(runner, "prepare_prelaunch", MagicMock(return_value=None)),
            patch.object(runner, "_stamp_deterministic_seed_contract"),
            patch.object(runner, "_prepare_execution", AsyncMock(return_value=None)),
            patch.object(
                runner, "_persist_effective_config", AsyncMock(return_value=None)
            ),
            patch.object(runner, "_persist_infer_plans", AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="evaluation failed"),
        ):
            await runner.arun()

        pool.aclose.assert_awaited_once()
        await runner._close_owned_model_resources()
        pool.aclose.assert_awaited_once()

    @pytest.mark.anyio
    async def test_cleanup_deduplicates_same_pool_registered_under_two_roots(self):
        runner = object.__new__(EvalSession)
        pool = types.SimpleNamespace(aclose=AsyncMock())
        runner._owned_pools = {"model:a": pool, "model:b": pool}
        runner._owned_legacy_models = {}

        await runner._close_owned_model_resources()

        pool.aclose.assert_awaited_once()
        assert not runner._owned_pools

    @pytest.mark.anyio
    async def test_cleanup_continues_after_first_close_error_and_does_not_retry(self):
        runner = object.__new__(EvalSession)
        first = types.SimpleNamespace(
            aclose=AsyncMock(side_effect=RuntimeError("first close failed"))
        )
        second = types.SimpleNamespace(aclose=AsyncMock())
        legacy = types.SimpleNamespace(aclose=AsyncMock())
        runner._owned_pools = {"model:a": first, "model:b": second}
        runner._owned_legacy_models = {"model:legacy": legacy}

        with pytest.raises(RuntimeError, match="first close failed"):
            await runner._close_owned_model_resources()

        first.aclose.assert_awaited_once()
        second.aclose.assert_awaited_once()
        legacy.aclose.assert_awaited_once()
        await runner._close_owned_model_resources()
        first.aclose.assert_awaited_once()
        second.aclose.assert_awaited_once()
        legacy.aclose.assert_awaited_once()


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
        runner._task_requirement_contexts = {
            task_name: _task_requirement_context_for_setup_test(
                task_cfg,
                runner.models,
            )
            for task_name, task_cfg in tasks_cfg.items()
        }
        return runner

    @pytest.mark.parametrize("key", ["name", "dataset", "model", "models_by_role"])
    def test_setup_tasks_defensively_rejects_composition_owned_arg(self, key):
        mock_task_cls = MagicMock(return_value=MagicMock())
        runner = self._make_runner(
            {
                "eval_task": {
                    "class": "fake.Task",
                    "dataset": "ds",
                    "model": "m",
                    "args": {key: None},
                }
            },
            models={"m": MockChatModel()},
            datasets={"ds": MagicMock()},
        )

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=mock_task_cls,
            ),
            pytest.raises(ValueError, match=rf"composition-owned.*{key}"),
        ):
            runner._setup_tasks()

        mock_task_cls.assert_not_called()

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

    def test_self_extractor_uses_candidate_after_infer_args(self):
        """The self sentinel must resolve to the final task-specific model."""
        from sieval.tasks.agieval_0shot_gen import AGIEvalZeroShotGenTask

        base_model = MockChatModel(concurrency_limit=128, temperature=0.0)
        runner = self._make_runner(
            {
                "agieval": {
                    "class": "fake.AGIEvalTask",
                    "dataset": "ds",
                    "model": "candidate",
                    "args": {"extractor": "self"},
                    "infer_args": {"max_tokens": 4096},
                }
            },
            models={"candidate": base_model},
            datasets={"ds": MagicMock()},
        )

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=AGIEvalZeroShotGenTask,
        ):
            runner._setup_tasks()

        assert runner.runner is not None
        task = runner.runner._runners[0]._task
        assert isinstance(task, AGIEvalZeroShotGenTask)
        assert task.model is task._extractor
        assert task.model is not base_model
        assert task.model._kwargs["max_tokens"] == 4096
        assert task.model.pool is base_model.pool
        assert task.model._limiter is base_model._limiter
        assert task.model.runtime_plan is base_model.runtime_plan

    @pytest.mark.parametrize(
        "task_class_name",
        [
            "AdvancedIFZeroShotGenTask",
            "ComplexConstraintsZeroShotGenTask",
            "InverseIFEvalZeroShotGenTask",
        ],
    )
    def test_bound_grader_is_injected_into_real_role_aware_task(
        self, task_class_name: str
    ) -> None:
        from sieval.tasks.advanced_if_0shot_gen import AdvancedIFZeroShotGenTask
        from sieval.tasks.complex_constraints_0shot_gen import (
            ComplexConstraintsZeroShotGenTask,
        )
        from sieval.tasks.inverse_ifeval_0shot_gen import (
            InverseIFEvalZeroShotGenTask,
        )

        task_classes = {
            task_class.__name__: task_class
            for task_class in (
                AdvancedIFZeroShotGenTask,
                ComplexConstraintsZeroShotGenTask,
                InverseIFEvalZeroShotGenTask,
            )
        }
        task_class = task_classes[task_class_name]
        candidate = MockChatModel()
        grader = MockChatModel()
        runner = self._make_runner(
            {
                "judged": {
                    "class": f"fake.{task_class_name}",
                    "dataset": "ds",
                    "model": "candidate",
                    "args": {"grader": {"model": "raw-inline-grader"}},
                }
            },
            models={"candidate": candidate},
            datasets={"ds": MagicMock()},
        )
        runner._bound_task_role_models = {"judged": {"grader": grader}}

        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=task_class,
            ),
            patch(
                "sieval.tasks.advanced_if_0shot_gen.load_judge_prompts",
                return_value=None,
            ),
        ):
            runner._setup_tasks()

        assert runner.runner is not None
        task = runner.runner._runners[0]._task
        assert isinstance(task, task_class)
        assert task._grader is grader

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
# Deterministic mode: dialect-declared seed injection + transparent sampling
# ===================================================================
class TestDeterministicRequestSeedPolicy:
    @pytest.mark.parametrize("dialect_id", ["openai_chat", "openai_completions"])
    def test_supported_dialects_receive_the_automatic_seed(self, dialect_id):
        decision = _resolve_deterministic_request_seed(
            dialect_id=dialect_id,
            explicit_seed_present=False,
        )

        assert decision.seed_present is True
        assert decision.seed == DETERMINISTIC_DEFAULT_SEED
        assert decision.provenance.value == "automatic"

    def test_unsupported_dialect_requests_automatic_default_removal(self):
        with patch(
            "sieval.cli.leaderboard.session.get_dialect_spec",
            return_value=types.SimpleNamespace(
                request_seed_support=RequestSeedSupport.UNSUPPORTED
            ),
        ):
            decision = _resolve_deterministic_request_seed(
                dialect_id="future_seedless",
                explicit_seed_present=False,
            )

        assert decision.seed_present is False
        assert decision.seed is None
        assert decision.provenance.value == "none"

    @pytest.mark.anyio
    async def test_absent_decision_removes_existing_model_seed(self):
        with patch(
            "sieval.cli.leaderboard.session.get_dialect_spec",
            return_value=types.SimpleNamespace(
                request_seed_support=RequestSeedSupport.UNSUPPORTED
            ),
        ):
            decision = _resolve_deterministic_request_seed(
                dialect_id="openai_chat",
                explicit_seed_present=False,
            )

        source = MockChatModel(seed=7)
        try:
            model = _apply_request_seed_decision_to_model(source, decision)
            assert "seed" not in model.meta()["default_params"]
        finally:
            await source.aclose()

    def test_absent_decision_removes_existing_argument_seed(self):
        with patch(
            "sieval.cli.leaderboard.session.get_dialect_spec",
            return_value=types.SimpleNamespace(
                request_seed_support=RequestSeedSupport.UNSUPPORTED
            ),
        ):
            decision = _resolve_deterministic_request_seed(
                dialect_id="future_seedless",
                explicit_seed_present=False,
            )

        args = {"seed": 7, "temperature": 0.5}
        _apply_request_seed_decision_to_args(args, decision)

        assert args == {"temperature": 0.5}

    def test_present_decision_replaces_existing_argument_seed(self):
        decision = _resolve_deterministic_request_seed(
            dialect_id="openai_chat",
            explicit_seed_present=True,
            explicit_seed=0,
        )
        args = {"seed": 9, "temperature": 0.5}

        _apply_request_seed_decision_to_args(args, decision)

        assert args == {"seed": 0, "temperature": 0.5}

    def test_reserved_seed_policy_fails_loudly(self):
        with pytest.raises(ValueError, match="openai_responses.*has not declared"):
            _resolve_deterministic_request_seed(
                dialect_id="openai_responses",
                explicit_seed_present=False,
            )

    @pytest.mark.parametrize("seed", [None, 0, 42])
    def test_explicit_seed_is_never_rewritten(self, seed):
        with patch(
            "sieval.cli.leaderboard.session.get_dialect_spec",
            return_value=types.SimpleNamespace(
                request_seed_support=RequestSeedSupport.UNSUPPORTED
            ),
        ):
            decision = _resolve_deterministic_request_seed(
                dialect_id="future_seedless",
                explicit_seed_present=True,
                explicit_seed=seed,
            )

        assert decision.seed_present is True
        assert decision.seed == seed
        assert decision.provenance.value == "binding_config"


class TestDeterministicMode:
    """Deterministic mode flag resolution, seed injection, sampling pass-through.

    Deterministic mode injects only ``seed``, and only when the selected
    dialect declares that it can transmit one. All other sampling parameters
    (temperature, top_p, top_k, ...) remain transparent.
    """

    class ChatTask:
        model_type = "chat"

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.CHAT),
                    source_task="deterministic_test",
                ),
            )

    class GenTask:
        model_type = "gen"

        @classmethod
        def model_requirements_for(cls, context):
            return (
                TaskModelRequirement(
                    role="candidate",
                    binding=context.model_bindings["candidate"],
                    requires=TaskRequirements(input=InputKind.COMPLETION),
                    source_task="deterministic_test",
                ),
            )

    def _bind_models(
        self,
        tmp_path: Path,
        config: dict,
        *,
        task_class: type | None = None,
        connection: object | None = None,
    ) -> EvalSession:
        """Exercise the same prelaunch -> postlaunch -> bind path as a run."""

        task_class = task_class or self.ChatTask
        config.setdefault("tasks", {})["eval"] = {
            "class": f"fake.{task_class.__name__}",
            "dataset": {"class": "fake.Dataset"},
            "model": next(reversed(config["models"])),
        }
        config_path = _write_yaml_config(
            tmp_path,
            "deterministic.yaml",
            yaml.safe_dump(config, sort_keys=False),
        )
        session = EvalSession(config_path)
        connection = connection or types.SimpleNamespace(close=AsyncMock())
        with (
            patch(
                "sieval.cli.leaderboard.session.resolve_task_class",
                return_value=task_class,
            ),
            patch(
                "sieval.core.models.connection_factory.AsyncOpenAI",
                return_value=connection,
            ),
        ):
            session._setup_prelaunch_reconciliation()
            session._setup_postlaunch_reconciliation()
            session._setup_models()
        return session

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

    def test_seed_auto_injected_when_absent(self, tmp_path):
        """A supporting dialect receives seed=0 when the user omits it."""
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-chat", "type": "chat", "args": {}}},
            },
        )
        assert session.models["m1"]._kwargs.get("seed") == 0

    def test_completions_seed_auto_injected_when_absent(self, tmp_path):
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-gen", "type": "gen", "args": {}}},
            },
            task_class=self.GenTask,
        )

        assert session.models["m1"].dialect_id == "openai_completions"
        assert session.models["m1"]._kwargs.get("seed") == 0

    @pytest.mark.anyio
    async def test_chat_automatic_seed_reaches_wire_and_model_output(self, tmp_path):
        create = AsyncMock(
            return_value=types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        index=0,
                        message=types.SimpleNamespace(
                            content="ok",
                            reasoning=None,
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                usage=None,
                model="served-chat",
                system_fingerprint=None,
            )
        )
        connection = types.SimpleNamespace(
            close=AsyncMock(),
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            ),
        )
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-chat", "type": "chat"}},
            },
            connection=connection,
        )

        try:
            output = await session.models["m1"].agenerate("hello", stream=False)
            masked = await session.models["m1"].agenerate(
                "hello", stream=False, seed=None
            )
        finally:
            await session._close_owned_model_resources()

        seeded_call, masked_call = create.await_args_list
        assert seeded_call.kwargs["seed"] == DETERMINISTIC_DEFAULT_SEED
        assert "seed" not in masked_call.kwargs
        assert output.request_params is not None
        assert output.request_params["seed"] == DETERMINISTIC_DEFAULT_SEED
        assert masked.request_params is not None
        assert "seed" not in masked.request_params

    @pytest.mark.anyio
    async def test_completions_automatic_seed_reaches_wire_and_model_output(
        self, tmp_path
    ):
        create = AsyncMock(
            return_value=types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        index=0,
                        text="ok",
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                usage=None,
                model="served-completion",
                system_fingerprint=None,
            )
        )
        connection = types.SimpleNamespace(
            close=AsyncMock(),
            completions=types.SimpleNamespace(create=create),
        )
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-gen", "type": "gen"}},
            },
            task_class=self.GenTask,
            connection=connection,
        )

        try:
            output = await session.models["m1"].agenerate("hello", stream=False)
        finally:
            await session._close_owned_model_resources()

        call = create.await_args
        assert call is not None
        assert call.kwargs["seed"] == DETERMINISTIC_DEFAULT_SEED
        assert output.request_params is not None
        assert output.request_params["seed"] == DETERMINISTIC_DEFAULT_SEED

    @pytest.mark.anyio
    async def test_task_infer_seed_contract_matches_each_candidate_wire(self, tmp_path):
        from tests.unit.core.runners.test_runner import MockTask

        create = AsyncMock(
            return_value=types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        index=0,
                        message=types.SimpleNamespace(
                            content="ok",
                            reasoning=None,
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                usage=None,
                model="served-chat",
                system_fingerprint=None,
            )
        )
        connection = types.SimpleNamespace(
            close=AsyncMock(),
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            ),
        )
        config_path = _write_yaml_config(
            tmp_path,
            "candidate-seeds.yaml",
            yaml.safe_dump(
                {
                    "deterministic": True,
                    "models": {"m": {"name": "mock-chat", "type": "chat"}},
                    "datasets": {"ds": {"class": "fake.Dataset"}},
                    "tasks": {
                        "first": {
                            "class": "fake.MockTask",
                            "dataset": "ds",
                            "model": "m",
                            "infer_args": {"seed": 11},
                        },
                        "second": {
                            "class": "fake.MockTask",
                            "dataset": "ds",
                            "model": "m",
                            "infer_args": {"seed": 22},
                        },
                    },
                },
                sort_keys=False,
            ),
        )
        session = EvalSession(config_path)

        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=MockTask,
                ),
                patch(
                    "sieval.core.models.connection_factory.AsyncOpenAI",
                    return_value=connection,
                ),
            ):
                session._setup_prelaunch_reconciliation()
                session._stamp_deterministic_seed_contract()

                # Execution must consume RequirementContext.infer_args, not a
                # second read of the now-mutated session config.
                tasks = cast(dict, session.config["tasks"])
                cast(dict, tasks["first"])["infer_args"] = {"seed": 91}
                cast(dict, tasks["second"])["infer_args"] = {"seed": 92}

                session._setup_postlaunch_reconciliation()
                session._setup_models()
                session._init_runner()
                session.datasets["ds"] = MagicMock()
                session._setup_tasks()

            contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
            assert contract["bindings"]["model:m"]["seed"] == 0
            assert contract["candidates"]["first"]["seed"] == 11
            assert (
                contract["candidates"]["first"]["seed_provenance"] == "task_infer_args"
            )
            assert contract["candidates"]["second"]["seed"] == 22
            assert (
                contract["candidates"]["second"]["seed_provenance"] == "task_infer_args"
            )
            yaml.safe_dump(contract)
            assert "_request_seed_decisions" not in session._reified_config

            assert session.runner is not None
            task_models = {
                entry._task.name: entry._task.model for entry in session.runner._runners
            }
            await task_models["first"].agenerate("hello", stream=False)
            await task_models["second"].agenerate("hello", stream=False)

            first_call, second_call = create.await_args_list
            assert first_call.kwargs["seed"] == 11
            assert second_call.kwargs["seed"] == 22
        finally:
            await session._close_owned_model_resources()

    @pytest.mark.anyio
    async def test_task_candidate_binding_is_frozen_before_setup(self, tmp_path):
        from tests.unit.core.runners.test_runner import MockTask

        def response() -> object:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        index=0,
                        message=types.SimpleNamespace(
                            content="ok",
                            reasoning=None,
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                usage=None,
                model="served-chat",
                system_fingerprint=None,
            )

        frozen_create = AsyncMock(return_value=response())
        mutated_create = AsyncMock(return_value=response())
        frozen_connection = types.SimpleNamespace(
            close=AsyncMock(),
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=frozen_create)
            ),
        )
        mutated_connection = types.SimpleNamespace(
            close=AsyncMock(),
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=mutated_create)
            ),
        )
        config_path = _write_yaml_config(
            tmp_path,
            "frozen-candidate.yaml",
            yaml.safe_dump(
                {
                    "deterministic": True,
                    "models": {
                        "frozen": {
                            "name": "frozen-model",
                            "type": "chat",
                            "api_base": "https://frozen.example/v1",
                            "api_key": "frozen-key",
                            "args": {"seed": 11},
                        },
                        "mutated": {
                            "name": "mutated-model",
                            "type": "chat",
                            "api_base": "https://mutated.example/v1",
                            "api_key": "mutated-key",
                            "args": {"seed": 22},
                        },
                    },
                    "datasets": {"ds": {"class": "fake.Dataset"}},
                    "tasks": {
                        "eval": {
                            "class": "fake.MockTask",
                            "dataset": "ds",
                            "model": "frozen",
                        }
                    },
                },
                sort_keys=False,
            ),
        )
        session = EvalSession(config_path)

        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=MockTask,
                ),
                patch(
                    "sieval.core.models.connection_factory.AsyncOpenAI",
                    side_effect=[frozen_connection, mutated_connection],
                ),
            ):
                session._setup_prelaunch_reconciliation()
                session._stamp_deterministic_seed_contract()

                tasks = cast(dict, session.config["tasks"])
                cast(dict, tasks["eval"])["model"] = "mutated"

                session._setup_postlaunch_reconciliation()
                session._setup_models()
                session._init_runner()
                session.datasets["ds"] = MagicMock()
                session._setup_tasks()

            contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
            candidate_contract = contract["candidates"]["eval"]
            assert candidate_contract["binding_id"] == "model:frozen"
            assert candidate_contract["requested_model_id"] == "frozen-model"
            assert candidate_contract["seed"] == 11

            assert session.runner is not None
            task_model = session.runner._runners[0]._task.model
            assert task_model.runtime_plan is not None
            assert task_model.runtime_plan.binding_id == "model:frozen"
            assert task_model._kwargs["seed"] == 11

            await task_model.agenerate("hello", stream=False)
            frozen_create.assert_awaited_once()
            mutated_create.assert_not_awaited()
            call = frozen_create.await_args
            assert call is not None
            assert call.kwargs["model"] == "frozen-model"
            assert call.kwargs["seed"] == 11
        finally:
            await session._close_owned_model_resources()

    def test_seed_user_override_preserved(self, tmp_path):
        """User's explicit seed=42 is preserved; no injection override."""
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {
                    "m1": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"seed": 42},
                    }
                },
            },
        )
        assert session.models["m1"]._kwargs.get("seed") == 42

    @pytest.mark.anyio
    async def test_named_seed_value_and_provenance_are_frozen_before_binding(
        self, tmp_path
    ):
        create = AsyncMock(
            return_value=types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        index=0,
                        message=types.SimpleNamespace(
                            content="ok",
                            reasoning=None,
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                usage=None,
                model="served-chat",
                system_fingerprint=None,
            )
        )
        connection = types.SimpleNamespace(
            close=AsyncMock(),
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            ),
        )
        config_path = _write_yaml_config(
            tmp_path,
            "frozen-named-seed.yaml",
            yaml.safe_dump(
                {
                    "deterministic": True,
                    "models": {
                        "m": {
                            "name": "mock-chat",
                            "type": "chat",
                            "args": {"seed": 41},
                        }
                    },
                    "tasks": {
                        "eval": {
                            "class": "fake.ChatTask",
                            "dataset": {"class": "fake.Dataset"},
                            "model": "m",
                        }
                    },
                },
                sort_keys=False,
            ),
        )
        session = EvalSession(config_path)

        try:
            with (
                patch(
                    "sieval.cli.leaderboard.session.resolve_task_class",
                    return_value=self.ChatTask,
                ),
                patch(
                    "sieval.core.models.connection_factory.AsyncOpenAI",
                    return_value=connection,
                ),
            ):
                session._setup_prelaunch_reconciliation()
                models = cast(dict, session.config["models"])
                args = cast(dict, cast(dict, models["m"])["args"])
                args["seed"] = 99
                session._setup_postlaunch_reconciliation()
                session._setup_models()
                session._stamp_deterministic_seed_contract()

            contract = session._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
            assert contract["bindings"]["model:m"]["seed"] == 41
            assert (
                contract["bindings"]["model:m"]["seed_provenance"] == "binding_config"
            )
            assert contract["candidates"]["eval"]["seed"] == 41

            await session.models["m"].agenerate("hello", stream=False)
            call = create.await_args
            assert call is not None
            assert call.kwargs["seed"] == 41
        finally:
            await session._close_owned_model_resources()

    def test_explicit_none_seed_is_preserved_through_derived_chain(self, tmp_path):
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {
                    "base": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"seed": None},
                    },
                    "child": {"base": "base", "args": {"temperature": 0.7}},
                },
            },
        )

        assert session.models["base"]._kwargs["seed"] is None
        assert session.models["child"]._kwargs["seed"] is None

    def test_no_seed_injection_when_not_deterministic(self, tmp_path):
        """seed is NOT auto-injected when deterministic=False."""
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": False,
                "models": {"m1": {"name": "mock-chat", "type": "chat", "args": {}}},
            },
        )
        assert "seed" not in session.models["m1"]._kwargs

    def test_derived_model_keeps_binding_local_automatic_seed(self, tmp_path):
        """Every compatible binding receives the deterministic request seed."""
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {
                    "base": {"name": "mock-chat", "type": "chat", "args": {}},
                    # Non-empty args forces the with_args path (empty args +
                    # no concurrency_limit would just alias base).
                    "child": {"base": "base", "args": {"temperature": 0.7}},
                },
            },
        )
        base = session.models["base"]
        child = session.models["child"]
        assert base._kwargs.get("seed") == 0
        # Derived model keeps seed=0 from base and picks up its own override.
        assert child._kwargs.get("seed") == 0
        assert child._kwargs.get("temperature") == 0.7

    # ------------------------------------------------------------------
    # Sampling params: transparent — user configures freely, no lock
    # ------------------------------------------------------------------

    def test_temperature_sampling_allowed(self, tmp_path):
        """temperature > 0 is allowed under deterministic mode (seeded sampling)."""
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {
                    "m1": {
                        "name": "mock-chat",
                        "type": "chat",
                        "args": {"temperature": 0.6},
                    }
                },
            },
        )
        defaults = session.models["m1"]._kwargs
        assert defaults.get("temperature") == 0.6
        assert defaults.get("seed") == 0  # still injected

    def test_full_sampling_config_passes_through(self, tmp_path):
        """Full pass@k sampling config (temperature, top_p, top_k) all pass through."""
        session = self._bind_models(
            tmp_path,
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
            },
        )
        defaults = session.models["m1"]._kwargs
        assert defaults.get("temperature") == 0.6
        assert defaults.get("top_p") == 0.95
        assert defaults.get("top_k") == 20
        assert defaults.get("max_tokens") == 32768
        assert defaults.get("frequency_penalty") == 0.1
        assert defaults.get("seed") == 0

    def test_no_temperature_injection_under_deterministic(self, tmp_path):
        """Deterministic mode does NOT inject temperature (only seed)."""
        session = self._bind_models(
            tmp_path,
            {
                "deterministic": True,
                "models": {"m1": {"name": "mock-chat", "type": "chat", "args": {}}},
            },
        )
        # temperature left to engine default — not force-injected to 0.0
        assert "temperature" not in session.models["m1"]._kwargs

    def test_per_task_infer_args_temperature_allowed(self):
        """Per-task infer_args with temperature > 0 is accepted (no lock)."""
        mock_model = MockChatModel()
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
        runner._task_requirement_contexts = {
            "eval_task": RequirementContext(
                model_bindings={
                    "candidate": NamedModelBinding(
                        binding_id="model:m",
                        root_deployment_key="model:m",
                        requested_model_id="m",
                        config_name="m",
                        dialect_id="openai_chat",
                    )
                },
                infer_args={"temperature": 0.6, "top_k": 20},
            )
        }
        runner._request_seed_decisions_by_candidate = {
            "eval_task": _resolve_deterministic_request_seed(
                dialect_id="openai_chat",
                explicit_seed_present=False,
            )
        }
        runner._request_seed_decisions_frozen = True

        with patch(
            "sieval.cli.leaderboard.session.resolve_task_class",
            return_value=mock_task_cls,
        ):
            runner._setup_tasks()  # must not raise
        task_model = mock_task_cls.call_args.kwargs["model"]
        assert task_model._kwargs["temperature"] == 0.6
        assert task_model._kwargs["top_k"] == 20
        assert task_model._kwargs["seed"] == DETERMINISTIC_DEFAULT_SEED


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
    def test_deterministic_sets_root_without_persisting_automatic_seed(self):
        cfg = {
            "models": {
                "base": {"name": "qwen3-4b", "args": {"max_tokens": 8192}},
                "derived": {"base": "base", "args": {"temperature": 0.6}},
            }
        }
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["deterministic"] is True
        assert out["models"]["base"]["args"]["max_tokens"] == 8192
        assert "seed" not in out["models"]["base"]["args"]
        assert "seed" not in out["models"]["derived"]["args"]

    def test_deterministic_preserves_user_seed(self):
        cfg = {"models": {"base": {"name": "m", "args": {"seed": 42}}}}
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["models"]["base"]["args"]["seed"] == 42

    def test_deterministic_does_not_add_args_dict_when_missing(self):
        cfg = {"models": {"base": {"name": "m"}}}
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert "args" not in out["models"]["base"]

    def test_deterministic_idempotent_when_already_true(self):
        cfg = {
            "deterministic": True,
            "models": {"base": {"name": "m", "args": {"seed": 0}}},
        }
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["deterministic"] is True
        assert out["models"]["base"]["args"]["seed"] == 0

    def test_deterministic_preserves_explicit_none_seed(self):
        cfg = {"models": {"base": {"name": "m", "args": {"seed": None}}}}
        out = _reify_cli_overrides(cfg, deterministic=True)
        assert out["models"]["base"]["args"]["seed"] is None

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
        assert "args" not in out["models"]["base"]


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
        assert "args" not in session.config["models"]["base"]

    def test_raw_config_unaffected_by_legacy_external_endpoint_adapter(self, tmp_path):
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

    def test_rejects_legacy_endpoint_adapter_with_typed_deployment(self, tmp_path):
        from sieval.core.models import Deployment, Engine, ServingFacts

        config_path = _write_yaml_config(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    path: /ckpts/m\n",
        )
        deployment = Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine("vllm"),
            engine_source="deployment",
            api_base="http://localhost:8000/v1",
            endpoints={"full": "http://localhost:8000/v1"},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )

        with pytest.raises(
            ValueError,
            match="legacy endpoint-only adapter.*realized_deployments",
        ):
            EvalSession(
                config_path=str(config_path),
                endpoint_map={"base": "http://localhost:8000/v1"},
                realized_deployments={"base": deployment},
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
        assert "args" not in session.config["models"]["base"]


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

    def test_reports_order_instead_of_formatting_only(self):
        """A reorder has no structural diff, so this used to answer "(whitespace /
        formatting only)" while its caller aborted because the bytes differ."""
        out = _diff_dicts({"a": 1, "b": 2}, {"b": 2, "a": 1})
        assert "formatting only" not in out
        assert "same keys, different order" in out
        assert "'b' moved from position 1 to 0" in out

    def test_a_dropped_null_key_is_not_called_a_reorder(self):
        """The same conflation one step over: calling a removed key "same keys,
        different order" is as wrong as the message it replaced."""
        out = _diff_dicts({"role": "full", "scaling": None}, {"role": "full"})
        assert "formatting only" not in out
        assert "same keys" not in out
        assert "keys added or removed" in out
        assert "removed ['scaling']" in out

    def test_both_kinds_are_reported_together(self):
        out = _diff_dicts(
            {"top": {"x": None, "y": 1}, "other": {"a": 1, "b": 2}},
            {"top": {"y": 1}, "other": {"b": 2, "a": 1}},
        )
        assert "keys added or removed" in out
        assert "same keys, different order" in out

    def test_value_differences_still_win(self):
        out = _diff_dicts({"a": 1}, {"a": 2})
        assert out == "Diff:\n  - a: 1 → 2"


class TestDiffLines:
    def test_identical_returns_empty(self):
        assert _diff_lines({"a": 1}, {"a": 1}) == []

    def test_nested_leaf_path(self):
        lines = _diff_lines({"a": {"b": 1}}, {"a": {"b": 2}})
        assert lines == ["- a.b: 1 → 2"]


class TestDescribeOrderChange:
    """A reorder is named by the key that moved, not by dumping both sequences."""

    def test_names_the_first_key_that_moved_and_where_it_came_from(self):
        assert _describe_order_change(["a", "b"], ["b", "a"]) == (
            "'b' moved from position 1 to 0 (2 keys)"
        )

    def test_a_realistic_block_stays_readable(self):
        """Why this helper exists: dumped in full, a 12-key block is one
        ~500-character line."""
        keys = [
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "enable_prefix_caching",
            "tool_call_parser",
            "enable_auto_tool_choice",
            "reasoning_parser",
            "dtype",
            "max_num_seqs",
            "trust_remote_code",
            "seed",
            "disable_log_requests",
        ]
        moved = [keys[1], keys[0], *keys[2:]]
        out = _describe_order_change(keys, moved)
        assert out == "'gpu_memory_utilization' moved from position 1 to 0 (12 keys)"
        assert len(out) < 100

    def test_identical_sequences_have_nothing_to_name(self):
        """Callers only ask when the orders differ, but stay total."""
        assert _describe_order_change(["a"], ["a"]) == "identical order (1 keys)"


class TestDiffKeyShape:
    """A changed key SET (a null-valued key reads as an absent one) and a changed
    key ORDER both hide behind an empty `_diff_lines`, and differ in kind."""

    def test_same_shape_returns_nothing(self):
        assert _diff_key_shape({"a": 1, "b": 2}, {"a": 1, "b": 2}) == ([], [])

    def test_root_reorder_reported_as_order(self):
        presence, order = _diff_key_shape({"a": 1, "b": 2}, {"b": 2, "a": 1})
        assert presence == []
        assert order == ["- (root): 'b' moved from position 1 to 0 (2 keys)"]

    def test_nested_reorder_carries_its_path(self):
        presence, order = _diff_key_shape(
            {"m": {"p": {"x": 1, "y": 2}}},
            {"m": {"p": {"y": 2, "x": 1}}},
        )
        assert presence == []
        assert order == ["- m.p: 'y' moved from position 1 to 0 (2 keys)"]

    def test_reorder_inside_a_list_element(self):
        """The real shape: engine_params sits under assignments[0]."""
        presence, order = _diff_key_shape(
            {"models": {"m": {"assignments": [{"engine_params": {"a": 1, "b": 2}}]}}},
            {"models": {"m": {"assignments": [{"engine_params": {"b": 2, "a": 1}}]}}},
        )
        assert presence == []
        assert order == [
            "- models.m.assignments[0].engine_params: "
            "'b' moved from position 1 to 0 (2 keys)"
        ]

    def test_a_dropped_null_key_is_presence_not_order(self):
        """`.get()` makes `{'x': None}` and `{}` compare equal leaf-for-leaf, so
        the difference is only visible as a key set."""
        assert _diff_lines({"x": None, "y": 1}, {"y": 1}) == []
        presence, order = _diff_key_shape({"x": None, "y": 1}, {"y": 1})
        assert order == []
        assert presence == ["- (root): removed ['x']"]

    def test_added_and_removed_are_both_named(self):
        presence, order = _diff_key_shape({"a": 1, "b": 2}, {"a": 1, "c": 2})
        assert order == []
        assert presence == ["- (root): removed ['b'], added ['c']"]

    def test_a_changed_key_set_is_not_reported_as_a_reorder(self):
        """A differing key set makes `list(x) != list(y)` trivially true, and
        calling that an order change says "same keys" about two that differ."""
        presence, order = _diff_key_shape({"a": 1}, {"b": 1})
        assert order == []
        assert presence == ["- (root): removed ['a'], added ['b']"]

    def test_presence_and_order_can_both_be_present(self):
        presence, order = _diff_key_shape(
            {"top": {"x": None, "y": 1}, "other": {"a": 1, "b": 2}},
            {"top": {"y": 1}, "other": {"b": 2, "a": 1}},
        )
        assert presence == ["- top: removed ['x']"]
        assert order == ["- other: 'b' moved from position 1 to 0 (2 keys)"]


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
        """Same structure AND same key order — the difference really is layout.

        Previously this case was illustrated with a key reorder, which conflated
        two things the caller must tell apart: whitespace is cosmetic, key order
        is not (the strict comparison is byte-for-byte). See the reorder test
        below.
        """
        existing = "a: 1\nb: 2\n"
        current = "a:   1\nb: 2\n"
        out = _brief_diff(existing, current)
        assert "whitespace / formatting only" in out

    def test_key_reorder_is_named_not_called_formatting(self):
        """A reorder aborts a byte comparison, so it must be reported as one."""
        existing = "a: 1\nb: 2\n"
        current = "b: 2\na: 1\n"
        out = _brief_diff(existing, current)
        assert "same keys, different order" in out
        assert "formatting only" not in out

    def test_a_dropped_null_key_is_named_as_a_removed_key(self):
        """`scaling: null` is a real field on every role assignment, so a version
        that stops emitting it lands here — not on the reorder branch."""
        existing = "role: full\nreplicas: 1\nscaling: null\n"
        current = "role: full\nreplicas: 1\n"
        out = _brief_diff(existing, current)
        assert "keys added or removed" in out
        assert "removed ['scaling']" in out
        assert "same keys" not in out
        assert "formatting only" not in out


class TestSortVersions:
    def test_sorts_by_version_not_by_text(self):
        """`sorted()` on strings puts 0.10.0 before 0.7.0."""
        assert _sort_versions({"0.7.0", "0.10.0", "0.9.1"}) == [
            "0.7.0",
            "0.9.1",
            "0.10.0",
        ]

    def test_unparseable_strings_sort_last_by_text(self):
        """Rejected for being unparseable, so they reach this list."""
        assert _sort_versions({"0.7.0", "not-a-version", "0.8.0"}) == [
            "0.7.0",
            "0.8.0",
            "not-a-version",
        ]


class TestCrossVersionResumeHint:
    """`EvalSession` compares its artifacts before any TaskRunner exists, so
    `gate_resume_version` cannot pre-empt those aborts — a cross-version resume
    arrives as an artifact diff. The hint names the version so the user is not
    left auditing their own invocation.
    """

    def _write_meta(
        self, root: Path, task: str, payload: object, *, resumable: bool = True
    ) -> None:
        (root / task).mkdir(parents=True, exist_ok=True)
        (root / task / "meta.json").write_text(json.dumps(payload))
        if resumable:
            # The gate's own precondition for calling a dir resumable.
            (root / task / "manifest.json").write_text("{}")

    @pytest.fixture
    def released(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Pin the current version to a released one.

        A dev/local build rejects every non-identical pair, so without this the
        COMPATIBLE rung is unreachable and these tests would mean something
        different on a released install.
        """
        monkeypatch.setattr(
            "sieval.cli.leaderboard.session.__version__", "0.8.0", raising=True
        )
        return "0.8.0"

    @pytest.mark.anyio
    async def test_incompatible_version_is_named(self, tmp_path: Path, released: str):
        self._write_meta(tmp_path, "arc_easy", {"version": "0.1.0"})
        out = await _cross_version_resume_hint(anyio.Path(tmp_path))
        assert "0.1.0" in out
        assert "not resume-compatible" in out
        assert released in out

    @pytest.mark.anyio
    async def test_the_note_does_not_blame_the_invocation(
        self, tmp_path: Path, released: str
    ):
        """An older run and an edited invocation can both be true at once, and on
        a dev build every non-identical pair is incompatible — so a claim about
        cause would often be the wrong one."""
        self._write_meta(tmp_path, "arc_easy", {"version": "0.1.0"})
        out = await _cross_version_resume_hint(anyio.Path(tmp_path))
        assert "likely cause" not in out
        assert "not your invocation" not in out
        assert "independently of the difference above" in out

    @pytest.mark.anyio
    async def test_same_series_is_compatible_and_stays_quiet(
        self, tmp_path: Path, released: str
    ):
        """A non-exact but same-series pair resumes fine, so say nothing. Distinct
        from the exact-match case below: this is the COMPATIBLE rung."""
        self._write_meta(tmp_path, "arc_easy", {"version": "0.8.2"})
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_exact_version_yields_nothing(self, tmp_path: Path):
        from sieval import __version__

        self._write_meta(tmp_path, "arc_easy", {"version": __version__})
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_a_dir_the_gate_would_not_read_is_skipped(
        self, tmp_path: Path, released: str
    ):
        """meta.json lands at run START, but a dir is only resumable once
        manifest.json exists — a task that died before its first flush is never
        version-gated, so it must not be reported as if it were."""
        self._write_meta(tmp_path, "arc_easy", {"version": "0.1.0"}, resumable=False)
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_missing_meta_yields_nothing(self, tmp_path: Path, released: str):
        """Unlike the gate, this is NOT fail-closed: a task that never started
        has no meta.json, and that is not evidence of a version problem."""
        (tmp_path / "arc_easy").mkdir()
        (tmp_path / "arc_easy" / "manifest.json").write_text("{}")
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_unreadable_meta_yields_nothing(self, tmp_path: Path, released: str):
        """A hint must never mask the caller's own abort."""
        (tmp_path / "arc_easy").mkdir()
        (tmp_path / "arc_easy" / "manifest.json").write_text("{}")
        (tmp_path / "arc_easy" / "meta.json").write_text("{not json")
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_version_less_meta_yields_nothing(
        self, tmp_path: Path, released: str
    ):
        self._write_meta(tmp_path, "arc_easy", {"deterministic": False})
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_non_mapping_meta_yields_nothing(self, tmp_path: Path, released: str):
        self._write_meta(tmp_path, "arc_easy", ["not", "a", "mapping"])
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_absent_result_dir_yields_nothing(self, tmp_path: Path):
        assert await _cross_version_resume_hint(anyio.Path(tmp_path / "nope")) == ""

    @pytest.mark.anyio
    async def test_a_failing_scan_yields_nothing_rather_than_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The caller is mid-`raise`; an OSError from the scan must not replace
        the abort it was annotating."""

        def _boom(*_args):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "iterdir", _boom)
        assert await _cross_version_resume_hint(anyio.Path(tmp_path)) == ""

    @pytest.mark.anyio
    async def test_every_incompatible_version_is_listed_in_version_order(
        self, tmp_path: Path, released: str
    ):
        self._write_meta(tmp_path, "task_a", {"version": "0.10.0"})
        self._write_meta(tmp_path, "task_b", {"version": "0.2.0"})
        out = await _cross_version_resume_hint(anyio.Path(tmp_path))
        assert "0.2.0, 0.10.0" in out
        assert "none of which is resume-compatible" in out

    @pytest.mark.anyio
    async def test_strict_abort_carries_both_the_order_diff_and_the_hint(
        self, tmp_path: Path, released: str
    ):
        """End-to-end on the real path: the message a cross-version resume of a
        reordered plan actually produces."""
        cfg = _write_yaml_config(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        self._write_meta(result_dir, "arc_easy", {"version": "0.1.0"})

        def _plan(engine_params: dict) -> dict:
            return {
                "checkpoint": "/ckpt",
                "backend": "vllm",
                "assignments": [
                    {
                        "role": "full",
                        "devices": {"count": 1, "gpu_model": "H100"},
                        "topology": {"tp": 1, "dp": 1, "pp": 1},
                        "replicas": 1,
                        "engine_params": engine_params,
                        "scaling": None,
                    }
                ],
                "deterministic": False,
                "seed": 0,
            }

        before = {"max_model_len": 4096, "tool_call_parser": "hermes", "dtype": "bf16"}
        after = {"max_model_len": 4096, "dtype": "bf16", "tool_call_parser": "hermes"}

        await EvalSession(
            config_path=str(cfg),
            result_dir_override=str(result_dir),
            infer_plans={"m": _plan(before)},
            invocation="sieval run cfg.yaml",
        )._persist_infer_plans()

        resumed = EvalSession(
            config_path=str(cfg),
            result_dir_override=str(result_dir),
            infer_plans={"m": _plan(after)},
            resume=True,
            invocation="sieval run cfg.yaml --resume",
        )
        with pytest.raises(RuntimeError) as exc:
            await resumed._persist_infer_plans()

        message = str(exc.value)
        assert "same keys, different order" in message
        assert "engine_params" in message
        assert "'dtype' moved from position 2 to 1" in message
        assert "0.1.0" in message
        assert "formatting only" not in message

    @pytest.mark.anyio
    async def test_strict_abort_names_a_dropped_null_plan_field(
        self, tmp_path: Path, released: str
    ):
        """The other cross-version plan shape: a schema change that stops emitting
        a `None`-valued field. `scaling` is declared a placeholder, so filling it
        in or dropping it lands here."""
        cfg = _write_yaml_config(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        self._write_meta(result_dir, "arc_easy", {"version": "0.1.0"})

        def _plan(*, with_scaling: bool) -> dict:
            assignment: dict[str, Any] = {"role": "full", "replicas": 1}
            if with_scaling:
                assignment["scaling"] = None
            return {"checkpoint": "/ckpt", "assignments": [assignment]}

        await EvalSession(
            config_path=str(cfg),
            result_dir_override=str(result_dir),
            infer_plans={"m": _plan(with_scaling=True)},
            invocation="sieval run cfg.yaml",
        )._persist_infer_plans()

        resumed = EvalSession(
            config_path=str(cfg),
            result_dir_override=str(result_dir),
            infer_plans={"m": _plan(with_scaling=False)},
            resume=True,
            invocation="sieval run cfg.yaml --resume",
        )
        with pytest.raises(RuntimeError) as exc:
            await resumed._persist_infer_plans()

        message = str(exc.value)
        assert "keys added or removed" in message
        assert "removed ['scaling']" in message
        # The two labels this must NOT reach for: neither is true here.
        assert "same keys" not in message
        assert "formatting only" not in message


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
