"""
Tests for sieval.cli.validation — config pre-validation and dry-run.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from sieval.cli.main import app
from sieval.cli.validation import (
    ValidationResult,
    _is_builtin_class_spec,
    load_yaml_with_duplicate_check,
    validate_eval_config,
    validate_eval_config_imports,
)

cli_runner = CliRunner()


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_ok_true_when_no_errors(self):
        result = ValidationResult(warnings=["some warning"])
        assert result.ok is True

    def test_ok_false_when_errors_exist(self):
        result = ValidationResult(errors=["something broke"])
        assert result.ok is False

    def test_merge_combines_errors_and_warnings(self):
        a = ValidationResult(errors=["e1"], warnings=["w1"])
        b = ValidationResult(errors=["e2"], warnings=["w2"])
        a.merge(b)
        assert a.errors == ["e1", "e2"]
        assert a.warnings == ["w1", "w2"]


# ---------------------------------------------------------------------------
# YAML duplicate-key checking
# ---------------------------------------------------------------------------


class TestLoadYamlWithDuplicateCheck:
    def test_valid_yaml_no_duplicates(self):
        content = "models:\n  a:\n    name: foo\n  b:\n    name: bar\n"
        data, warnings = load_yaml_with_duplicate_check(content)
        assert data == {"models": {"a": {"name": "foo"}, "b": {"name": "bar"}}}
        assert warnings == []

    def test_duplicate_top_level_key(self):
        content = "models:\n  a:\n    name: foo\nmodels:\n  b:\n    name: bar\n"
        data, warnings = load_yaml_with_duplicate_check(content)
        assert data == {"models": {"b": {"name": "bar"}}}
        assert len(warnings) == 1
        assert "models" in warnings[0]

    def test_duplicate_nested_key(self):
        content = "models:\n  a:\n    name: foo\n  a:\n    name: bar\n"
        data, warnings = load_yaml_with_duplicate_check(content)
        assert data == {"models": {"a": {"name": "bar"}}}
        assert len(warnings) == 1
        assert "'a'" in warnings[0]

    def test_empty_yaml(self):
        data, warnings = load_yaml_with_duplicate_check("")
        assert data == {}
        assert warnings == []

    def test_invalid_yaml_raises(self):
        with pytest.raises(yaml.YAMLError):
            load_yaml_with_duplicate_check("models:\n  - [invalid\n")

    def test_sequential_calls_produce_independent_warnings(self):
        dup_content = "a: 1\na: 2\n"
        clean_content = "b: 1\nc: 2\n"

        _, w1 = load_yaml_with_duplicate_check(dup_content)
        _, w2 = load_yaml_with_duplicate_check(clean_content)
        _, w3 = load_yaml_with_duplicate_check(dup_content)

        assert len(w1) == 1
        assert w2 == []
        assert len(w3) == 1
        # Warnings must not leak between calls
        assert w1 is not w3


# ---------------------------------------------------------------------------
# Schema validation — structure
# ---------------------------------------------------------------------------


class TestDeterministicValidation:
    def test_deterministic_key_accepted(self, tmp_path):
        """deterministic: true should not generate an unrecognized-key warning."""
        from sieval.cli.validation import validate_eval_config

        config = {
            "deterministic": True,
            "result_dir": "./out",
            "models": {"base": {"name": "m"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(config)
        assert not any("deterministic" in w for w in result.warnings)


class TestValidateStructure:
    def test_valid_minimal_config(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"d": {"class": "MyDataset"}},
            "tasks": {"t": {"class": "MyTask", "dataset": "d", "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert result.ok
        assert result.warnings == []

    def test_models_not_dict(self):
        cfg = {"models": ["a", "b"], "datasets": {}, "tasks": {}}
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("models" in e and "dict" in e.lower() for e in result.errors)

    def test_datasets_not_dict(self):
        cfg = {"models": {}, "datasets": "bad", "tasks": {}}
        result = validate_eval_config(cfg)
        assert not result.ok

    def test_tasks_not_dict(self):
        cfg = {"models": {}, "datasets": {}, "tasks": 42}
        result = validate_eval_config(cfg)
        assert not result.ok

    def test_unrecognized_top_level_key(self):
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {},
            "unknown_key": "value",
        }
        result = validate_eval_config(cfg)
        assert result.ok  # warnings don't cause failure
        assert any("unknown_key" in w for w in result.warnings)

    def test_empty_models_warning(self):
        cfg = {"models": {}, "datasets": {}, "tasks": {}}
        result = validate_eval_config(cfg)
        assert result.ok
        assert any(
            "models" in w.lower() and "empty" in w.lower() for w in result.warnings
        )

    def test_empty_tasks_warning(self):
        cfg = {"models": {"m": {"name": "gpt-4"}}, "datasets": {}, "tasks": {}}
        result = validate_eval_config(cfg)
        assert result.ok
        assert any(
            "tasks" in w.lower() and "empty" in w.lower() for w in result.warnings
        )

    def test_non_dict_item_in_section_does_not_crash(self):
        cfg = {"models": {"m": "just a string"}, "datasets": {}, "tasks": {}}
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("models.m" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Schema validation — models
# ---------------------------------------------------------------------------


class TestValidateModels:
    def test_base_model_missing_name(self):
        cfg = {
            "models": {"m": {"type": "chat"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("name" in e for e in result.errors)

    def test_base_model_name_derivable_from_checkpoint(self):
        """No error when name is absent but infer.checkpoint can derive it."""
        cfg = {
            "models": {
                "m": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": "/models/Qwen2.5-1.5B",
                    },
                }
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert result.ok, result.errors

    def test_base_model_name_derivable_from_path(self):
        """No error when name is absent but path can derive it."""
        cfg = {
            "models": {"m": {"path": "/models/Qwen2.5-1.5B"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert result.ok, result.errors

    def test_invalid_model_type(self):
        cfg = {
            "models": {"m": {"name": "gpt-4", "type": "invalid"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("type" in e for e in result.errors)

    def test_invalid_engine_value(self):
        cfg = {
            "models": {"m": {"name": "x", "type": "gen", "engine": "bogus"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("engine must be" in e for e in result.errors)

    def test_engine_on_chat_model(self):
        cfg = {
            "models": {"m": {"name": "x", "type": "chat", "engine": "sglang"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("only valid for type: gen" in e for e in result.errors)

    def test_engine_on_derived_model(self):
        cfg = {
            "models": {
                "base": {"name": "x", "type": "gen", "engine": "sglang"},
                "d": {"base": "base", "engine": "sglang"},
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("cannot set 'engine'" in e for e in result.errors)

    def test_valid_engine_sglang(self):
        cfg = {
            "models": {"m": {"name": "x", "type": "gen", "engine": "sglang"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert result.ok, result.errors

    def test_valid_engine_vllm(self):
        cfg = {
            "models": {"m": {"name": "x", "type": "gen", "engine": "vllm"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert result.ok, result.errors

    def test_name_and_base_coexist(self):
        cfg = {
            "models": {"m": {"name": "gpt-4", "base": "other"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("name" in e and "base" in e for e in result.errors)

    def test_derived_model_base_not_found(self):
        cfg = {
            "models": {"m": {"base": "nonexistent"}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("nonexistent" in e for e in result.errors)

    def test_derived_model_cycle(self):
        cfg = {
            "models": {
                "a": {"base": "b"},
                "b": {"base": "a"},
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("cycl" in e.lower() for e in result.errors)

    def test_derived_model_base_not_string(self):
        cfg = {
            "models": {"m": {"base": 123}},
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("non-empty string" in e for e in result.errors)

    def test_valid_derived_model(self):
        cfg = {
            "models": {
                "base": {"name": "gpt-4"},
                "derived": {"base": "base", "type": "gen"},
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert result.ok

    def test_infer_section_without_checkpoint_warns(self):
        """`infer:` section with no checkpoint/path and no api_base warns —
        `sieval run` silently skips auto-serve in this case, and schema
        validation is the cross-entrypoint place to surface it."""
        cfg = {
            "models": {
                "m": {
                    "name": "foo",
                    "infer": {"backend": "vllm"},
                }
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert result.ok  # warning, not error
        assert any("skip auto-serve" in w and "'m'" in w for w in result.warnings)

    def test_infer_section_with_api_base_no_warning(self):
        """api_base present → serving is not needed, no warning."""
        cfg = {
            "models": {
                "m": {
                    "name": "foo",
                    "api_base": "http://localhost:8000/v1",
                    "infer": {"backend": "vllm"},
                }
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not any("auto-serve" in w for w in result.warnings)

    def test_infer_section_with_checkpoint_no_warning(self):
        """checkpoint present → we know what to serve, no warning."""
        cfg = {
            "models": {
                "m": {
                    "name": "foo",
                    "infer": {"backend": "vllm", "checkpoint": "/models/foo"},
                }
            },
            "datasets": {},
            "tasks": {},
        }
        result = validate_eval_config(cfg)
        assert not any("auto-serve" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Schema validation — datasets
# ---------------------------------------------------------------------------


class TestValidateDatasets:
    def test_non_dict_dataset_item_skipped(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": "not_a_dict"},
        }
        result = validate_eval_config(cfg)
        assert not result.ok  # caught by _validate_structure

    def test_operations_not_list(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": {"class": "X", "operations": "bad"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("list" in e for e in result.errors)

    def test_operation_args_not_dict(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": {"class": "X", "operations": [{"slice": "bad"}]}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("dict or null" in e for e in result.errors)

    def test_dataset_missing_class(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": {"path": "/data"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("class" in e for e in result.errors)

    def test_operations_invalid_format(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": {"class": "X", "operations": ["bad"]}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("operation" in e.lower() for e in result.errors)

    def test_operations_multi_key_dict(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {
                "d": {
                    "class": "X",
                    "operations": [{"slice": {"num": 10}, "shuffle": {}}],
                }
            },
        }
        result = validate_eval_config(cfg)
        assert not result.ok

    def test_operations_unknown_op(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": {"class": "X", "operations": [{"sort": {}}]}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("sort" in e for e in result.errors)

    def test_operations_renamed_op_gives_migration_hint(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {"d": {"class": "X", "operations": [{"select": {"num": 5}}]}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("'select' was renamed to 'slice'" in e for e in result.errors)

    def test_operations_never_shipped_name_is_unknown_not_renamed(self):
        # 'stratified_select' never shipped (introduced and renamed within the
        # same unreleased change), so it must read as an unknown op, not carry a
        # migration hint for a name users never saw.
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {
                "d": {"class": "X", "operations": [{"stratified_select": {"num": 5}}]}
            },
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("unknown operation 'stratified_select'" in e for e in result.errors)
        assert not any("was renamed" in e for e in result.errors)

    def test_valid_operations(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {
                "d": {
                    "class": "X",
                    "operations": [
                        {"shuffle": {"seed": 42}},
                        {"slice": {"num": 100}},
                    ],
                }
            },
        }
        result = validate_eval_config(cfg)
        assert result.ok

    def test_valid_stratified_sample_operation(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {
                "d": {
                    "class": "X",
                    "operations": [
                        {
                            "stratified_sample": {
                                "by": "Subject",
                                "num": 800,
                                "min_per_group": 5,
                                "seed": 42,
                            }
                        }
                    ],
                }
            },
        }
        result = validate_eval_config(cfg)
        assert result.ok

    def test_valid_stratified_sample_equal_composite_operation(self):
        cfg = {
            "models": {},
            "tasks": {},
            "datasets": {
                "d": {
                    "class": "X",
                    "operations": [
                        {
                            "stratified_sample": {
                                "by": ["locale", "subject"],
                                "per_group": 20,
                                "seed": 42,
                            }
                        }
                    ],
                }
            },
        }
        result = validate_eval_config(cfg)
        assert result.ok


# ---------------------------------------------------------------------------
# Schema validation — tasks
# ---------------------------------------------------------------------------


class TestValidateTasks:
    def test_non_dict_task_item_skipped(self):
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {"t": "not_a_dict"},
        }
        result = validate_eval_config(cfg)
        assert not result.ok

    def test_task_inline_dataset_operations_not_list(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {},
            "tasks": {
                "t": {
                    "class": "X",
                    "dataset": {"class": "DS", "operations": "bad"},
                    "model": "m",
                }
            },
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("list" in e for e in result.errors)

    def test_task_dataset_wrong_type(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {},
            "tasks": {"t": {"class": "X", "dataset": 42, "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("string or dict" in e for e in result.errors)

    def test_task_model_not_string(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"class": "X", "dataset": "d", "model": 123}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("'model' must be a string" in e for e in result.errors)

    def test_task_missing_class(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"dataset": "d", "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("class" in e for e in result.errors)

    def test_task_legacy_task_field_accepted(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"task": "MyTask", "dataset": "d", "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert result.ok

    def test_task_dataset_ref_not_found(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {},
            "tasks": {"t": {"class": "X", "dataset": "missing", "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("missing" in e for e in result.errors)

    def test_task_inline_dataset_missing_class(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {},
            "tasks": {"t": {"class": "X", "dataset": {"path": "/data"}, "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok

    def test_task_inline_dataset_valid(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {},
            "tasks": {"t": {"class": "X", "dataset": {"class": "DS"}, "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert result.ok

    def test_task_model_ref_not_found(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"class": "X", "dataset": "d", "model": "missing"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("missing" in e for e in result.errors)

    def test_task_model_omitted_single_model(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"class": "X", "dataset": "d"}},
        }
        result = validate_eval_config(cfg)
        assert result.ok

    def test_task_model_omitted_zero_models_error(self):
        cfg = {
            "models": {},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"class": "X", "dataset": "d"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("no models" in e.lower() for e in result.errors)

    def test_task_model_omitted_multi_model_error(self):
        cfg = {
            "models": {"a": {"name": "gpt-4"}, "b": {"name": "gpt-3.5"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"class": "X", "dataset": "d"}},
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("model" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Schema validation — runner_config
# ---------------------------------------------------------------------------


class TestValidateRunnerConfig:
    def test_runner_config_not_dict(self):
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {},
            "runner_config": "bad",
        }
        result = validate_eval_config(cfg)
        assert not result.ok
        assert any("runner_config" in e and "dict" in e for e in result.errors)

    def test_unknown_runner_config_field(self):
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {},
            "runner_config": {"unknown_field": 42},
        }
        result = validate_eval_config(cfg)
        assert result.ok
        assert any("unknown_field" in w for w in result.warnings)

    def test_invalid_concurrency_limits_key(self):
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {},
            "concurrency_limits": {"infer": 10, "typo_stage": 5},
        }
        result = validate_eval_config(cfg)
        assert result.ok
        assert any("typo_stage" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Schema validation — unreferenced entities
# ---------------------------------------------------------------------------


class TestUnreferencedEntities:
    def test_non_dict_task_item_in_unreferenced(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {},
            "tasks": {"t": "not_a_dict"},
        }
        # non-dict task items are caught by _validate_structure,
        # but _validate_unreferenced should not crash on them
        result = validate_eval_config(cfg)
        assert not result.ok

    def test_unreferenced_model(self):
        cfg = {
            "models": {"used": {"name": "gpt-4"}, "unused": {"name": "other"}},
            "datasets": {"d": {"class": "X"}},
            "tasks": {"t": {"class": "X", "dataset": "d", "model": "used"}},
        }
        result = validate_eval_config(cfg)
        assert result.ok
        assert any("unused" in w for w in result.warnings)

    def test_unreferenced_dataset(self):
        cfg = {
            "models": {"m": {"name": "gpt-4"}},
            "datasets": {"used": {"class": "X"}, "unused": {"class": "Y"}},
            "tasks": {"t": {"class": "X", "dataset": "used", "model": "m"}},
        }
        result = validate_eval_config(cfg)
        assert result.ok
        assert any("unused" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------


class TestValidateImports:
    def test_successful_imports(self):
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {"t": {"class": "MyTask", "dataset": {"class": "MyDS"}}},
        }
        with (
            patch("sieval.cli.validation.resolve_task_class") as mock_task,
            patch("sieval.cli.validation.resolve_dataset_class") as mock_ds,
        ):
            mock_task.return_value = type("MyTask", (), {})
            mock_ds.return_value = type("MyDS", (), {})
            result = validate_eval_config_imports(cfg)
        assert result.ok
        assert result.warnings == []

    def test_builtin_task_import_failure_becomes_error(self):
        """Simple name (no dot) -> built-in -> failure is an error."""
        cfg = {
            "models": {},
            "datasets": {},
            "tasks": {"t": {"class": "BadTask", "dataset": {"class": "MyDS"}}},
        }
        with (
            patch(
                "sieval.cli.validation.resolve_task_class",
                side_effect=ImportError("no module"),
            ),
            patch("sieval.cli.validation.resolve_dataset_class") as mock_ds,
        ):
            mock_ds.return_value = type("MyDS", (), {})
            result = validate_eval_config_imports(cfg)
        assert not result.ok  # built-in import failures are errors
        assert any("BadTask" in e for e in result.errors)

    def test_builtin_dataset_import_failure_becomes_error(self):
        """Simple name (no dot) -> built-in -> failure is an error."""
        cfg = {
            "models": {},
            "datasets": {"d": {"class": "BadDS"}},
            "tasks": {},
        }
        with patch(
            "sieval.cli.validation.resolve_dataset_class",
            side_effect=ImportError("no module"),
        ):
            result = validate_eval_config_imports(cfg)
        assert not result.ok
        assert any("BadDS" in e for e in result.errors)

    def test_sieval_dotted_path_failure_becomes_error(self):
        """Dotted path starting with ``sieval.`` -> built-in -> error."""
        cfg = {
            "models": {},
            "datasets": {"d": {"class": "sieval.datasets.NoSuch"}},
            "tasks": {},
        }
        with patch(
            "sieval.cli.validation.resolve_dataset_class",
            side_effect=ImportError("no module"),
        ):
            result = validate_eval_config_imports(cfg)
        assert not result.ok
        assert any("sieval.datasets.NoSuch" in e for e in result.errors)

    def test_thirdparty_import_failure_becomes_warning(self):
        """Dotted path outside ``sieval.`` -> third-party -> warning."""
        cfg = {
            "models": {},
            "datasets": {"d": {"class": "my_pkg.datasets.CustomDS"}},
            "tasks": {},
        }
        with patch(
            "sieval.cli.validation.resolve_dataset_class",
            side_effect=ImportError("no module"),
        ):
            result = validate_eval_config_imports(cfg)
        assert result.ok  # third-party failures stay as warnings
        assert any("my_pkg.datasets.CustomDS" in w for w in result.warnings)

    def test_skips_entries_without_class(self):
        cfg = {
            "models": {},
            "datasets": {"d": {}},
            "tasks": {"t": {}},
        }
        result = validate_eval_config_imports(cfg)
        assert result.ok
        assert result.warnings == []

    def test_non_dict_sections_return_early(self):
        cfg = {"tasks": [], "datasets": {}}
        result = validate_eval_config_imports(cfg)
        assert result.ok
        assert result.warnings == []

    def test_non_dict_items_skipped(self):
        cfg = {
            "tasks": {"t": "not_a_dict"},
            "datasets": {"d": "not_a_dict"},
        }
        result = validate_eval_config_imports(cfg)
        assert result.ok
        assert result.warnings == []

    def test_relative_import_becomes_error(self):
        """Relative import syntax (`.Foo`) is always invalid — must be an error."""
        cfg = {
            "models": {},
            "datasets": {"d": {"class": ".RelDS"}},
            "tasks": {"t": {"class": ".RelTask", "dataset": {"class": ".InlineDS"}}},
        }
        result = validate_eval_config_imports(cfg)
        assert not result.ok
        assert len(result.errors) == 3
        assert all("Relative import" in e for e in result.errors)


# ---------------------------------------------------------------------------
# _is_builtin_class_spec
# ---------------------------------------------------------------------------


class TestIsBuiltinClassSpec:
    def test_simple_name_is_builtin(self):
        assert _is_builtin_class_spec("AIME2024Dataset") is True

    def test_sieval_dotted_path_is_builtin(self):
        assert _is_builtin_class_spec("sieval.datasets.AIME2024Dataset") is True
        assert _is_builtin_class_spec("sieval.tasks.math.MathTask") is True

    def test_relative_import_is_not_builtin(self):
        assert _is_builtin_class_spec(".MyClass") is False

    def test_thirdparty_dotted_path_is_not_builtin(self):
        assert _is_builtin_class_spec("my_pkg.MyClass") is False
        assert _is_builtin_class_spec("lm_eval.tasks.Custom") is False


# ---------------------------------------------------------------------------
# CLI integration — eval --dry-run
# ---------------------------------------------------------------------------


class TestEvalDryRunCli:
    def test_dry_run_valid_config(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "models:\n"
            "  m:\n"
            "    name: gpt-4\n"
            "datasets:\n"
            "  d:\n"
            "    class: FakeDataset\n"
            "tasks:\n"
            "  t:\n"
            "    class: FakeTask\n"
            "    dataset: d\n"
            "    model: m\n",
            encoding="utf-8",
        )
        with (
            patch("sieval.cli.validation.resolve_task_class"),
            patch("sieval.cli.validation.resolve_dataset_class"),
        ):
            result = cli_runner.invoke(app, ["eval", "--dry-run", str(config)])
        assert result.exit_code == 0
        assert "Dry-run passed" in result.output

    def test_dry_run_invalid_config_exit_code_1(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "models:\n  m:\n    type: invalid\ndatasets: []\ntasks: {}\n",
            encoding="utf-8",
        )
        result = cli_runner.invoke(app, ["eval", "--dry-run", str(config)])
        assert result.exit_code == 1
        assert "Dry-run failed" in result.output

    def test_dry_run_file_not_found(self, tmp_path: Path):
        result = cli_runner.invoke(
            app, ["eval", "--dry-run", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_dry_run_invalid_yaml_syntax(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text("models:\n  - [broken\n", encoding="utf-8")
        result = cli_runner.invoke(app, ["eval", "--dry-run", str(config)])
        assert result.exit_code == 1
        assert "YAML syntax error" in result.output


# ---------------------------------------------------------------------------
# CLI integration — run --dry-run
# ---------------------------------------------------------------------------


class TestRunDryRunCli:
    """Test --dry-run flag on ``sieval run`` command."""

    def test_run_dry_run_valid_config(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "models:\n"
            "  m:\n"
            "    name: gpt-4\n"
            "datasets:\n"
            "  d:\n"
            "    class: FakeDataset\n"
            "tasks:\n"
            "  t:\n"
            "    class: FakeTask\n"
            "    dataset: d\n"
            "    model: m\n",
            encoding="utf-8",
        )
        with (
            patch("sieval.cli.validation.resolve_task_class"),
            patch("sieval.cli.validation.resolve_dataset_class"),
        ):
            result = cli_runner.invoke(app, ["run", "--dry-run", str(config)])
        assert result.exit_code == 0
        assert "Dry-run passed" in result.output

    def test_run_dry_run_invalid_config(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "models:\n  m:\n    type: invalid\ndatasets: []\ntasks: {}\n",
            encoding="utf-8",
        )
        result = cli_runner.invoke(app, ["run", "--dry-run", str(config)])
        assert result.exit_code == 1
        assert "Dry-run failed" in result.output

    def test_run_dry_run_file_not_found(self, tmp_path: Path):
        result = cli_runner.invoke(
            app, ["run", "--dry-run", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code == 1
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# run_dry_run() unit tests
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_returns_dict_on_valid_config(self, tmp_path):
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ndatasets: {}\ntasks: {}")
        result = run_dry_run(config)
        assert isinstance(result, dict)
        assert "checks" in result
        assert "n_errors" in result
        assert "n_warnings" in result
        for check in result["checks"]:
            assert "name" in check
            assert "ok" in check

    def test_returns_dict_on_missing_file(self, tmp_path):
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "nonexistent.yaml"
        result = run_dry_run(config)
        assert result["n_errors"] >= 1
        file_check = result["checks"][0]
        assert file_check["name"] == "file_exists"
        assert file_check["ok"] is False

    def test_returns_dict_on_bad_yaml(self, tmp_path):
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "bad.yaml"
        config.write_text("models: {\\n  invalid yaml")
        result = run_dry_run(config)
        yaml_check = next(c for c in result["checks"] if c["name"] == "yaml_syntax")
        assert yaml_check["ok"] is False

    def test_schema_validation_failure(self, tmp_path):
        """Config with bad schema returns schema check failure."""
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "bad_schema.yaml"
        # Missing required 'models' key ->  schema error
        # models as a list instead of dict triggers schema error
        config.write_text("models: [1, 2, 3]\ntasks: {}")
        result = run_dry_run(config)
        assert result["n_errors"] >= 1
        schema_check = next(
            (c for c in result["checks"] if c["name"] == "schema"), None
        )
        assert schema_check is not None
        assert schema_check["ok"] is False
        assert "detail" in schema_check

    def test_import_validation_failure(self, tmp_path):
        """Config with valid schema but bad task import."""
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "import_fail.yaml"
        config.write_text(
            "models:\n  m1:\n"
            "    name: test\n"
            "    api_base: http://localhost:8000/v1\n"
            "datasets:\n  d1:\n"
            "    class: sieval.datasets.nonexistent.Fake\n"
            "tasks:\n  t1:\n"
            "    class: sieval.tasks.nonexistent.Fake\n"
            "    dataset: d1\n    model: m1\n"
        )
        result = run_dry_run(config)
        imports_check = next(
            (c for c in result["checks"] if c["name"] == "imports"),
            None,
        )
        assert imports_check is not None
        assert imports_check["ok"] is False
        assert "detail" in imports_check

    def test_schema_warnings_attached_to_check(self, tmp_path):
        """Schema warnings (e.g. unrecognized keys) appear on the schema check."""
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "extra_key.yaml"
        config.write_text("models: {}\ndatasets: {}\ntasks: {}\nfoobar: 123\n")
        result = run_dry_run(config)
        schema_check = next(c for c in result["checks"] if c["name"] == "schema")
        assert schema_check["ok"] is True
        assert any("foobar" in w for w in schema_check.get("warnings", []))
        assert result["n_warnings"] >= 1

    def test_duplicate_key_warnings(self, tmp_path):
        """Config with duplicate keys returns warnings."""
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "dup.yaml"
        config.write_text(
            "models:\n  m1:\n    api_base: http://localhost:8000/v1\n"
            "models:\n  m2:\n    api_base: http://localhost:8001/v1\n"
            "datasets: {}\ntasks: {}\n"
        )
        result = run_dry_run(config)
        yaml_check = next(c for c in result["checks"] if c["name"] == "yaml_syntax")
        assert yaml_check["ok"] is True
        assert len(yaml_check.get("warnings", [])) > 0
        assert result["n_warnings"] > 0

    def test_schema_ok_no_detail_when_sections_not_dicts(self, tmp_path):
        """Schema check omits detail when models/datasets/tasks aren't all dicts."""
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "no_detail.yaml"
        # models is a string, not a dict -> detail is not set
        config.write_text("models: not_a_dict\ndatasets: {}\ntasks: {}\n")
        result = run_dry_run(config)
        schema_check = next(
            (c for c in result["checks"] if c["name"] == "schema"), None
        )
        assert schema_check is not None
        # schema fails because models isn't a dict
        assert "detail" not in schema_check or schema_check.get("ok") is False

    def test_import_warnings_attached(self, tmp_path):
        """Import warnings (e.g. third-party class) appear on the imports check."""
        from sieval.cli.validation import run_dry_run

        config = tmp_path / "import_warn.yaml"
        config.write_text(
            "models:\n  m1:\n"
            "    name: test\n"
            "    api_base: http://localhost:8000/v1\n"
            "datasets:\n  d1:\n"
            "    class: thirdparty.datasets.SomeDataset\n"
            "tasks:\n  t1:\n"
            "    class: thirdparty.tasks.SomeTask\n"
            "    dataset: d1\n    model: m1\n"
        )
        result = run_dry_run(config)
        imports_check = next(
            (c for c in result["checks"] if c["name"] == "imports"), None
        )
        assert imports_check is not None
        # Third-party imports produce warnings (not errors)
        warnings = imports_check.get("warnings", [])
        assert len(warnings) > 0
