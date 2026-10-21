"""Tests for EvalSession session-level persistence:
effective_config.yaml and infer_plans.yaml.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import contextlib
import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from sieval.cli.leaderboard.session import (
    _DETERMINISTIC_SEED_CONTRACT_KEY,
    EvalSession,
    _format_comment_header,
    _split_header,
)
from sieval.infer.topology.models import WellKnownRole


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class TestPersistEffectiveConfig:
    @pytest.mark.anyio
    async def test_writes_file_to_result_dir(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        await session._persist_effective_config()

        persisted = result_dir / "effective_config.yaml"
        assert persisted.exists()

    @pytest.mark.anyio
    async def test_body_is_raw_config_plus_cli_reify(self, tmp_path: Path):
        """CLI overrides are baked in; legacy endpoint injection is not."""
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    name: original\n    path: /ckpts/m\n",
        )
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            deterministic_override=True,
            model_override="new-name",
            result_dir_override=str(result_dir),
            endpoint_map={"base": "http://localhost:8000/v1"},
        )

        await session._persist_effective_config()

        persisted = result_dir / "effective_config.yaml"
        loaded = yaml.safe_load(persisted.read_text())
        # CLI reification baked in
        assert loaded["deterministic"] is True
        assert loaded["models"]["base"]["name"] == "new-name"
        assert "args" not in loaded["models"]["base"]
        # Endpoint injection NOT persisted
        assert "api_base" not in loaded["models"]["base"]
        assert "api_key" not in loaded["models"]["base"]

    @pytest.mark.anyio
    async def test_header_advertises_sieval_run_as_universal_reproduce(
        self, tmp_path: Path
    ):
        """``sieval run`` is the universal reproduce command: for auto-
        served sessions it re-launches services from the preserved
        ``path`` / ``infer`` fields (runtime endpoints are intentionally not
        persisted); for pre-served sessions it's a pass-through.

        ``sieval eval`` would silently fail for any auto-served session
        because the persisted body has no ``name`` / ``api_base`` — so
        the header must lead with ``sieval run`` and call out the
        ``sieval eval`` precondition explicitly."""
        cfg_path = _write_yaml(
            tmp_path, "cfg.yaml", "models:\n  base:\n    path: /ckpts/m\n"
        )
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        await session._persist_effective_config()

        text = (result_dir / "effective_config.yaml").read_text()
        # `sieval run` is advertised, and advertised FIRST.
        assert "sieval run <this file>" in text
        run_pos = text.find("sieval run <this file>")
        eval_pos = text.find("sieval eval <this file>")
        assert eval_pos == -1 or run_pos < eval_pos, (
            "sieval run must appear before sieval eval — it's the universal "
            "command; sieval eval is the specialized one."
        )
        # The `sieval eval` precondition is called out, not just listed bare.
        assert "already has api_base" in text

    @pytest.mark.anyio
    async def test_user_declared_api_base_is_preserved_and_secrets_redacted(
        self, tmp_path: Path
    ):
        """User-set ``api_base`` and ``api_key_env`` flow through to
        effective_config.yaml unchanged; every ``api_key`` / ``authorization``
        value, at any depth, is written as ``<redacted>``. Only runtime
        injection by the legacy external endpoint adapter is stripped."""
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "models:\n"
            "  remote:\n"
            "    name: gpt-4\n"
            "    api_base: https://api.openai.com/v1\n"
            "    api_key: SENTINEL-TOP\n"
            "  nested:\n"
            "    name: other\n"
            "    api_key_env: OTHER_KEY\n"
            "    args:\n"
            "      api_key: SENTINEL-ARGS\n"
            "tasks:\n"
            "  judged:\n"
            "    args:\n"
            "      grader:\n"
            "        model: judge\n"
            "        authorization: SENTINEL-AUTH\n"
            "        args:\n"
            "          api_key: SENTINEL-GRADER\n"
            "      extractor:\n"
            "        model: helper\n"
            "        api_key_env: HELPER_KEY\n",
        )
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        await session._persist_effective_config()

        text = (result_dir / "effective_config.yaml").read_text()
        assert "SENTINEL" not in text
        loaded = yaml.safe_load(text)
        m = loaded["models"]["remote"]
        assert m["api_base"] == "https://api.openai.com/v1"
        assert m["api_key"] == "<redacted>"
        assert loaded["models"]["nested"]["api_key_env"] == "OTHER_KEY"
        assert loaded["models"]["nested"]["args"]["api_key"] == "<redacted>"
        roles = loaded["tasks"]["judged"]["args"]
        assert roles["grader"]["authorization"] == "<redacted>"
        assert roles["grader"]["args"]["api_key"] == "<redacted>"
        assert roles["extractor"]["api_key_env"] == "HELPER_KEY"
        # The session's own config is untouched: the run still has its keys.
        assert session._reified_config["models"]["remote"]["api_key"] == (
            "SENTINEL-TOP"
        )

    @pytest.mark.anyio
    async def test_header_is_valid_yaml_and_contains_metadata(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        await session._persist_effective_config()

        text = (result_dir / "effective_config.yaml").read_text()
        # Header block is yaml-comments
        assert text.startswith("# ---")
        # Source path in header
        assert str(cfg_path) in text
        # Parses as YAML (header treated as comments)
        assert yaml.safe_load(text)["models"]["base"]["name"] == "m"

    @pytest.mark.anyio
    async def test_atomic_write_cleans_up_tmp_on_failure(
        self, tmp_path: Path, monkeypatch
    ):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        # Make the atomic rename fail.
        import anyio

        async def _fail(*_a, **_kw):
            raise OSError("simulated disk full")

        monkeypatch.setattr(anyio.Path, "replace", _fail)
        # Should not raise — persistence is best-effort
        await session._persist_effective_config()

        # Final file not written
        assert not (result_dir / "effective_config.yaml").exists()
        # Tmp cleaned up
        assert not (result_dir / "effective_config.yaml.tmp").exists()

    @pytest.mark.anyio
    async def test_mkdir_failure_does_not_crash_session(
        self, tmp_path: Path, monkeypatch
    ):
        """mkdir for result_dir failing (permission denied / NotADirectory) must
        not take down the session — persistence is best-effort."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        import anyio

        async def _fail_mkdir(*_a, **_kw):
            raise PermissionError("simulated EACCES on result_dir")

        monkeypatch.setattr(anyio.Path, "mkdir", _fail_mkdir)

        # Must not raise
        await session._persist_effective_config()

        # Nothing written
        assert not (result_dir / "effective_config.yaml").exists()
        assert not (result_dir / "effective_config.yaml.tmp").exists()

    @pytest.mark.anyio
    async def test_no_result_dir_logs_warning_and_returns(self, tmp_path: Path, caplog):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        session = EvalSession(config_path=str(cfg_path))
        # result_dir_override unset, YAML has no result_dir
        assert session.result_dir is None

        # Should not raise
        await session._persist_effective_config()
        # No exception, no file written (no dir to write into)

    @pytest.mark.anyio
    async def test_explicit_invocation_kwarg_appears_in_header(self, tmp_path: Path):
        """Programmatic callers can pass an explicit `invocation` string to
        override the sys.argv fallback — required so notebook / test / library
        use doesn't leak pytest's argv into audit files."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            invocation="library: custom.caller.run(cfg.yaml)",
        )

        await session._persist_effective_config()

        text = (result_dir / "effective_config.yaml").read_text()
        assert "library: custom.caller.run(cfg.yaml)" in text

    @pytest.mark.anyio
    async def test_default_invocation_falls_back_to_sys_argv(
        self, tmp_path: Path, monkeypatch
    ):
        """When no `invocation` is passed, EvalSession snapshots sys.argv at
        __init__ time. Monkeypatching sys.argv AFTER init must not affect
        the persisted header (snapshot is eager, not lazy)."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"

        import sys as _sys

        monkeypatch.setattr(_sys, "argv", ["sieval", "run", "/path/to/cfg.yaml"])
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        # Mutate sys.argv AFTER init — the session already captured its value
        monkeypatch.setattr(_sys, "argv", ["something", "else"])

        await session._persist_effective_config()

        text = (result_dir / "effective_config.yaml").read_text()
        assert "sieval run /path/to/cfg.yaml" in text
        assert "something else" not in text


def _real_plan_dict() -> dict:
    """Produce a plain dict mirroring a real DeploymentPlan.

    dataclasses.asdict cannot deepcopy MappingProxyType on Python 3.13
    (engine_params is frozen in RoleAssignment.__post_init__), so we
    build the dict manually and pass it through unwrap_proxies to
    exercise that code path with a real MappingProxyType value."""
    from types import MappingProxyType

    from sieval.cli.leaderboard.session import unwrap_proxies

    # Build a dict that contains a MappingProxyType node, mirroring
    # what dataclasses.asdict would produce if it could run.
    raw = {
        "checkpoint": "/data/ckpts/m",
        "backend": "vllm",
        "assignments": [
            {
                "role": WellKnownRole.FULL,
                "devices": {"count": 2, "gpu_model": "H100"},
                "topology": {"tp": 2, "dp": 1, "pp": 1},
                "replicas": 1,
                "engine_params": MappingProxyType(
                    {"dtype": "bfloat16", "max_model_len": 32768}
                ),
                "scaling": None,
            }
        ],
        "deterministic": True,
        "seed": 0,
    }
    return unwrap_proxies(raw)


class TestPersistInferPlans:
    @pytest.mark.anyio
    async def test_secrets_in_plans_are_redacted(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        plan = _real_plan_dict()
        plan["env"] = {"api_key": "SENTINEL-PLAN", "HF_HOME": "/cache"}
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": plan},
        )

        await session._persist_infer_plans()

        text = (result_dir / "infer_plans.yaml").read_text()
        assert "SENTINEL" not in text
        env = yaml.safe_load(text)["models"]["m"]["env"]
        assert env == {"api_key": "<redacted>", "HF_HOME": "/cache"}

    @pytest.mark.anyio
    async def test_no_file_when_infer_plans_none(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        await session._persist_infer_plans()

        assert not (result_dir / "infer_plans.yaml").exists()

    @pytest.mark.anyio
    async def test_no_file_when_infer_plans_empty(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={},
        )

        await session._persist_infer_plans()

        assert not (result_dir / "infer_plans.yaml").exists()

    @pytest.mark.anyio
    async def test_writes_file_with_models_section(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"model_a": _real_plan_dict()},
        )

        await session._persist_infer_plans()

        f = result_dir / "infer_plans.yaml"
        assert f.exists()
        loaded = yaml.safe_load(f.read_text())
        assert "models" in loaded
        assert "model_a" in loaded["models"]
        m = loaded["models"]["model_a"]
        assert m["backend"] == "vllm"
        assert m["deterministic"] is True
        assert m["seed"] == 0

    @pytest.mark.anyio
    async def test_header_does_not_advertise_sieval_eval(self, tmp_path: Path):
        """infer_plans.yaml is an audit artifact — not runnable via
        ``sieval eval``. Copy-pasting effective_config.yaml's reproduce hint
        here would mislead users into trying to evaluate this file directly.
        The header must instead make the reference-only role explicit."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": _real_plan_dict()},
        )

        await session._persist_infer_plans()

        text = (result_dir / "infer_plans.yaml").read_text()
        assert "sieval eval" not in text, (
            "infer_plans.yaml advertises `sieval eval`, but it's audit-only — "
            "users who try `sieval eval infer_plans.yaml` will get a confusing "
            "config parse error."
        )
        # Positive check: the reference role is called out explicitly.
        assert "Reference only" in text

    @pytest.mark.anyio
    async def test_engine_params_round_trip_no_proxy_leak(self, tmp_path: Path):
        """Regression: MappingProxyType inside RoleAssignment.engine_params
        must be unwrapped to plain dict before yaml.safe_dump."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": _real_plan_dict()},
        )

        # Must not raise yaml.representer.RepresenterError
        await session._persist_infer_plans()

        # Round-trip value integrity
        loaded = yaml.safe_load((result_dir / "infer_plans.yaml").read_text())
        ep = loaded["models"]["m"]["assignments"][0]["engine_params"]
        assert ep == {"dtype": "bfloat16", "max_model_len": 32768}

    @pytest.mark.anyio
    async def test_persist_config_failure_does_not_block_plans(
        self, tmp_path: Path, monkeypatch
    ):
        """Best-effort failures in _persist_effective_config don't block
        _persist_infer_plans. (Strict-match RuntimeErrors DO block by
        design — they're fail-fast, not best-effort.)"""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": _real_plan_dict()},
        )

        async def _fail(*_args, **_kwargs):
            raise OSError("boom")

        # Make only _persist_effective_config fail
        monkeypatch.setattr(session, "_persist_effective_config", _fail)

        # arun would normally call both; simulate manual sequence
        with contextlib.suppress(OSError):
            await session._persist_effective_config()
        await session._persist_infer_plans()

        assert (result_dir / "infer_plans.yaml").exists()


class TestResumePersistence:
    @pytest.mark.anyio
    async def test_identical_resume_is_silent(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"

        # First run — writes effective_config.yaml
        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        await s1._persist_effective_config()
        original = (result_dir / "effective_config.yaml").read_text()

        # Resume with same invocation — must not raise, must not rewrite
        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
        )
        await s2._persist_effective_config()

        # Header timestamp may differ; body should still match the original file
        assert (result_dir / "effective_config.yaml").read_text() == original

    @pytest.mark.anyio
    async def test_deterministic_resume_matches_without_persisted_automatic_seed(
        self, tmp_path: Path
    ):
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "deterministic: true\n"
            "models:\n"
            "  base:\n"
            "    name: mock-chat\n"
            "    type: chat\n"
            "    args:\n"
            "      max_tokens: 32\n"
            "tasks:\n"
            "  eval:\n"
            "    class: tests.unit.core.runners.test_runner.MockTask\n"
            "    dataset:\n"
            "      class: tests.conftest.MockDataset\n"
            "    model: base\n",
        )
        result_dir = tmp_path / "out"

        first = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        first.prepare_prelaunch()
        first._stamp_deterministic_seed_contract()
        await first._persist_effective_config()
        target = result_dir / "effective_config.yaml"
        original = target.read_text()
        persisted = yaml.safe_load(original)
        assert persisted["deterministic"] is True
        assert persisted["models"]["base"]["args"] == {"max_tokens": 32}
        original_contract = persisted[_DETERMINISTIC_SEED_CONTRACT_KEY]

        replayed = EvalSession(
            config_path=str(target),
            result_dir_override=str(tmp_path / "replayed"),
        )
        replayed.prepare_prelaunch()
        replayed._stamp_deterministic_seed_contract()
        assert (
            replayed._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]
            == original_contract
        )

        resumed = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
        )
        resumed.prepare_prelaunch()
        resumed._stamp_deterministic_seed_contract()
        await resumed._persist_effective_config()

        assert target.read_text() == original

    def test_explicit_seed_contract_ignores_unused_default_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "deterministic: true\n"
            "models:\n"
            "  m:\n"
            "    name: mock-chat\n"
            "    type: chat\n"
            "    args:\n"
            "      seed: 42\n"
            "tasks:\n"
            "  eval:\n"
            "    class: tests.unit.core.runners.test_runner.MockTask\n"
            "    dataset:\n"
            "      class: tests.conftest.MockDataset\n"
            "    model: m\n",
        )
        first = EvalSession(config_path=str(cfg_path))
        first.prepare_prelaunch()
        first._stamp_deterministic_seed_contract()
        original = first._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY]

        monkeypatch.setattr(
            "sieval.cli.leaderboard.session.DETERMINISTIC_DEFAULT_SEED", 1
        )
        replayed = EvalSession(config_path=str(cfg_path))
        replayed.prepare_prelaunch()
        replayed._stamp_deterministic_seed_contract()

        assert replayed._reified_config[_DETERMINISTIC_SEED_CONTRACT_KEY] == original

    @pytest.mark.anyio
    async def test_legacy_endpoint_change_does_not_trigger_mismatch(
        self, tmp_path: Path
    ):
        """Legacy runtime endpoints are not part of the strict match."""
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    path: /ckpts/m\n",
        )
        result_dir = tmp_path / "out"

        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            endpoint_map={"base": "http://host:8000/v1"},
        )
        await s1._persist_effective_config()

        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
            endpoint_map={"base": "http://host:9000/v1"},  # different port
        )
        # Must not raise — legacy adapter state is not persisted into
        # effective_config.yaml.
        await s2._persist_effective_config()

    @pytest.mark.anyio
    async def test_non_resume_overwrites(self, tmp_path: Path):
        """Without --resume, an existing file is overwritten — no strict check."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        (result_dir / "effective_config.yaml").write_text(
            "# stale\nmodels:\n  old: {}\n"
        )

        s = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        # Must not raise
        await s._persist_effective_config()
        loaded = yaml.safe_load((result_dir / "effective_config.yaml").read_text())
        # Written over the stale file
        assert "base" in loaded["models"]

    @pytest.mark.anyio
    async def test_matching_infer_plans_is_silent(self, tmp_path: Path):
        """Identical plan under --resume must not raise and must not rewrite."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"

        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": _real_plan_dict()},
        )
        await s1._persist_infer_plans()
        original = (result_dir / "infer_plans.yaml").read_text()

        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
            infer_plans={"m": _real_plan_dict()},
        )
        await s2._persist_infer_plans()

        # File unchanged (including header timestamp from first write)
        assert (result_dir / "infer_plans.yaml").read_text() == original

    @pytest.mark.anyio
    async def test_resume_tolerates_top_level_concurrency_change(self, tmp_path: Path):
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "concurrency_limit: 8\nmodels:\n  base:\n    name: m\n",
        )
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()
        original = (result_dir / "effective_config.yaml").read_text()
        original_header = _split_header(original)[0]
        assert original_header != ""  # sanity: a real header was written

        # User lowers concurrency and resumes.
        cfg_path.write_text("concurrency_limit: 2\nmodels:\n  base:\n    name: m\n")
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s2._persist_effective_config()  # must NOT raise

        after = (result_dir / "effective_config.yaml").read_text()
        loaded = yaml.safe_load(after)
        assert loaded["concurrency_limit"] == 2  # body updated to new value
        after_header = _split_header(after)[0]
        # The header keeps only the first run's provenance; the change goes to
        # each task's meta.json `runs[].config` instead.
        assert after_header == original_header
        assert s2._resume_config_diff == ["concurrency_limit: 8 → 2"]

    @pytest.mark.anyio
    async def test_resume_tolerates_per_model_args_concurrency_change(
        self, tmp_path: Path
    ):
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    name: m\n    args:\n"
            "      concurrency_limit: 64\n      temperature: 0.0\n",
        )
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        cfg_path.write_text(
            "models:\n  base:\n    name: m\n    args:\n"
            "      concurrency_limit: 8\n      temperature: 0.0\n"
        )
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s2._persist_effective_config()  # must NOT raise

        loaded = yaml.safe_load((result_dir / "effective_config.yaml").read_text())
        assert loaded["models"]["base"]["args"]["concurrency_limit"] == 8
        assert loaded["models"]["base"]["args"]["temperature"] == 0.0

    @pytest.mark.anyio
    async def test_resume_tolerates_console_progress_change(self, tmp_path: Path):
        # show_progress is console-only (never the progress.json dump) — adjustable.
        base = (
            "models:\n  base:\n    name: m\ntasks:\n  t:\n"
            "    runner_config:\n      show_progress: {}\n"
        )
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format("true"))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        cfg_path.write_text(base.format("false"))
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s2._persist_effective_config()  # must NOT raise

        loaded = yaml.safe_load((result_dir / "effective_config.yaml").read_text())
        assert loaded["tasks"]["t"]["runner_config"]["show_progress"] is False

    @pytest.mark.anyio
    async def test_resume_rewrite_with_missing_header_emits_fresh_header(
        self, tmp_path: Path
    ):
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        # Pre-seed a header-less persisted file whose throughput differs.
        (result_dir / "effective_config.yaml").write_text(
            "concurrency_limit: 8\nmodels:\n  base:\n    name: m\n"
        )
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "concurrency_limit: 2\nmodels:\n  base:\n    name: m\n",
        )
        # result_dir is injected into the body by result_dir_override but stripped from
        # the resume comparison, so it does not count as a diff here.
        s = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s._persist_effective_config()  # must NOT raise

        after = (result_dir / "effective_config.yaml").read_text()
        assert after.startswith("# -")  # a fresh header was emitted
        assert yaml.safe_load(after)["concurrency_limit"] == 2

    @pytest.mark.anyio
    async def test_resume_tolerates_formatting_only_diff(self, tmp_path: Path):
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        # Flow-style body, semantically identical to the canonical block dump.
        (result_dir / "effective_config.yaml").write_text(
            header + "models: {base: {name: m}}\n"
        )
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        # result_dir is injected into the body by result_dir_override but stripped from
        # the resume comparison, so it does not count as a diff here.
        s = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s._persist_effective_config()  # must NOT raise (no semantic change)

        # A formatting-only rewrite records no resume note (nothing changed).
        after = (result_dir / "effective_config.yaml").read_text()
        assert "Resumed by sieval" not in _split_header(after)[0]

    @pytest.mark.anyio
    async def test_resume_tolerates_top_level_runner_config_throughput(
        self, tmp_path: Path
    ):
        # Throughput knobs in the TOP-LEVEL runner_config defaults block (merged
        # into every task) are as resume-mutable as per-task ones.
        base = (
            "runner_config:\n  concurrency_limits:\n    infer: {}\n"
            "models:\n  base:\n    name: m\n"
        )
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format(4))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        cfg_path.write_text(base.format(16))
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s2._persist_effective_config()  # must NOT raise

        after = (result_dir / "effective_config.yaml").read_text()
        loaded = yaml.safe_load(after)
        assert loaded["runner_config"]["concurrency_limits"]["infer"] == 16
        assert "Resumed by sieval" not in _split_header(after)[0]
        assert s2._resume_config_diff == [
            "runner_config.concurrency_limits.infer: 4 → 16"
        ]

    @pytest.mark.anyio
    async def test_resume_keeps_the_first_header_and_records_each_diff(
        self, tmp_path: Path
    ):
        base = "concurrency_limit: {}\nmodels:\n  base:\n    name: m\n"
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format(8))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()
        first_header = _split_header(
            (result_dir / "effective_config.yaml").read_text()
        )[0]

        # Two successive resumes, each changing only throughput.
        diffs = []
        for value in (4, 2):
            cfg_path.write_text(base.format(value))
            s = EvalSession(
                config_path=str(cfg_path),
                resume=True,
                result_dir_override=str(result_dir),
            )
            await s._persist_effective_config()
            diffs.append(s._resume_config_diff)

        after = (result_dir / "effective_config.yaml").read_text()
        header, body = _split_header(after)
        # Origin recorded once and never amended; each resume reports only its own
        # change, which the runner appends to meta.json `runs[]`.
        assert header == first_header
        assert diffs == [["concurrency_limit: 8 → 4"], ["concurrency_limit: 4 → 2"]]
        assert yaml.safe_load(body)["concurrency_limit"] == 2
        assert _split_header(after) == (header, body)

    @pytest.mark.anyio
    async def test_a_changed_config_is_rewritten_not_refused(self, tmp_path: Path):
        """The strict match is gone: each task compares its stage identities.
        A changed config is rewritten under the first run's header, and what
        changed goes to `runs[].config`."""
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "models:\n  base:\n    name: m\nrunner_config:\n  max_retries: 1\n",
        )
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()
        first_header = _split_header(
            (result_dir / "effective_config.yaml").read_text()
        )[0]

        cfg_path.write_text(
            "models:\n  base:\n    name: m\nrunner_config:\n  max_retries: 3\n"
        )
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s2._persist_effective_config()
        header, body = _split_header((result_dir / "effective_config.yaml").read_text())
        assert header == first_header
        assert "max_retries: 3" in body
        assert s2._resume_config_diff == ["runner_config.max_retries: 1 → 3"]

    @pytest.mark.anyio
    async def test_changed_infer_plans_do_not_block_a_resume(self, tmp_path: Path):
        """The serving recipe is in each task's inference identity; the file
        stays as an audit record of the latest plan."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  m:\n    name: m\n")
        result_dir = tmp_path / "out"
        plan = _real_plan_dict()
        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": plan},
        )
        await s1._persist_infer_plans()
        changed = copy.deepcopy(plan)
        changed["recipe"] = "changed-recipe"
        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
            infer_plans={"m": changed},
        )
        await s2._persist_infer_plans()
        assert "changed-recipe" in (result_dir / "infer_plans.yaml").read_text()
        # Not a config change: the plan is recorded in the inference identity.
        assert s2._resume_config_diff == []

    @pytest.mark.anyio
    async def test_an_unreadable_existing_file_is_rewritten(
        self, tmp_path: Path, monkeypatch
    ):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        (result_dir / "effective_config.yaml").write_text("- not\n- a mapping\n")
        s = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        await s._persist_effective_config()
        assert "name: m" in (result_dir / "effective_config.yaml").read_text()
        assert s._resume_config_diff == []


class TestPersistTimingBeforeRunnerArun:
    """Regression guard for spec §3.7: persistence must happen BEFORE
    runner.arun() starts, so the file survives a mid-run crash."""

    @pytest.mark.anyio
    async def test_effective_config_written_before_runner_arun(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        # Stub _prepare_execution and make runner.arun raise. Persistence
        # runs before _prepare_execution, so the file must exist even when
        # the runner never completes.
        async def stub_prepare(self):
            self.runner = MagicMock()
            self.runner.arun = AsyncMock(side_effect=RuntimeError("task crashed"))

        with (
            patch.object(EvalSession, "_prepare_execution", stub_prepare),
            pytest.raises(RuntimeError, match="task crashed"),
        ):
            await session.arun()

        # File written despite the crash
        assert (result_dir / "effective_config.yaml").exists()


class TestResumeFromSurface:
    """`--resume DIR` / `runner_config.resume: <dir>`, as Harmony mounts it."""

    CONFIG = (
        "models:\n"
        "  base:\n"
        "    name: m\n"
        "    api_base: http://a/v1\n"
        "datasets:\n"
        "  ds:\n"
        "    class: tests.conftest.MockDataset\n"
        "tasks:\n"
        "  qa:\n"
        "    class: tests.unit.core.runners.test_runner.MockTask\n"
        "    dataset: ds\n"
        "    args:\n"
        "      n: 1\n"
    )

    def _session(self, tmp_path: Path, resume=None, body=None) -> EvalSession:
        cfg = _write_yaml(tmp_path, "cfg.yaml", body or self.CONFIG)
        return EvalSession(
            config_path=str(cfg),
            result_dir_override=str(tmp_path / "out"),
            resume=resume,
        )

    def test_a_session_dir_resolves_per_task(self, tmp_path: Path):
        (tmp_path / "old" / "qa").mkdir(parents=True)
        session = self._session(tmp_path, resume=tmp_path / "old")
        assert session._cli_resume_for("qa") == tmp_path / "old" / "qa"
        rc = session._build_runner_config({}, {}, "qa")
        assert rc.resume == tmp_path / "old" / "qa"

    def test_a_task_dir_serves_a_one_task_config(self, tmp_path: Path):
        (tmp_path / "old").mkdir()
        (tmp_path / "old" / "meta.json").write_text("{}")
        session = self._session(tmp_path, resume=tmp_path / "old")
        assert session._cli_resume_for("qa") == tmp_path / "old"

    def test_a_task_missing_there_runs_fresh(self, tmp_path: Path):
        (tmp_path / "old").mkdir()
        session = self._session(tmp_path, resume=tmp_path / "old")
        assert session._cli_resume_for("qa") is None
        assert session._build_runner_config({}, {}, "qa").resume is None

    def test_no_value_resumes_in_place(self, tmp_path: Path):
        session = self._session(tmp_path, resume=True)
        assert session._build_runner_config({}, {}, "qa").resume is True

    def test_the_sessions_own_dir_resumes_in_place(self, tmp_path: Path):
        session = self._session(tmp_path, resume=tmp_path / "out")
        assert session.resume_override is True
        assert session._build_runner_config({}, {}, "qa").resume is True

    def test_the_tasks_own_dir_resumes_in_place(self, tmp_path: Path):
        """A one-task config may name the task directory, as a copy can."""
        own = tmp_path / "out" / "qa"
        own.mkdir(parents=True)
        (own / "meta.json").write_text("{}")
        session = self._session(tmp_path, resume=own)
        assert session._build_runner_config({}, {}, "qa").resume is True

    def test_a_string_is_a_directory(self, tmp_path: Path):
        (tmp_path / "old" / "qa").mkdir(parents=True)
        session = self._session(tmp_path, resume=str(tmp_path / "old"))
        assert session._build_runner_config({}, {}, "qa").resume == (
            tmp_path / "old" / "qa"
        )

    def test_a_yaml_dir_wins_over_the_cli(self, tmp_path: Path):
        (tmp_path / "old" / "qa").mkdir(parents=True)
        session = self._session(tmp_path, resume=tmp_path / "old")
        rc = session._build_runner_config({}, {"resume": "/mnt/x/qa"}, "qa")
        assert rc.resume == Path("/mnt/x/qa")

    def test_a_per_task_dir_wins_over_the_defaults_and_cli(self, tmp_path: Path):
        session = self._session(tmp_path, resume=True)
        rc = session._build_runner_config(
            {"runner_config": {"resume": "/mnt/task/qa"}},
            {"resume": "/mnt/top/qa"},
            "qa",
        )
        assert rc.resume == Path("/mnt/task/qa")

    def test_a_yaml_dir_equal_to_the_tasks_own_resumes_in_place(self, tmp_path: Path):
        session = self._session(tmp_path)
        rc = session._build_runner_config(
            {"runner_config": {"resume": str(tmp_path / "out" / "qa")}}, {}, "qa"
        )
        assert rc.resume is True

    def test_yaml_true_resumes_without_the_cli(self, tmp_path: Path):
        session = self._session(tmp_path)
        rc = session._build_runner_config({}, {"resume": True}, "qa")
        assert rc.resume is True

    def test_a_cli_resume_wins_over_yaml_false(self, tmp_path: Path):
        """`resume: false` in YAML is the default; `--resume` turns it on, as
        `--resume` did over `auto_resume: false`."""
        session = self._session(tmp_path, resume=True)
        rc = session._build_runner_config({}, {"resume": False}, "qa")
        assert rc.resume is True

    def test_legacy_match_reads_the_recorded_config(self, tmp_path: Path):
        old = tmp_path / "old"
        (old / "qa").mkdir(parents=True)
        (old / "effective_config.yaml").write_text(
            self.CONFIG.replace("http://a/v1", "http://b/v1")
        )
        session = self._session(tmp_path, resume=old)
        rc = session._build_runner_config({}, {}, "qa")
        # An endpoint is not inference.
        assert session._legacy_match_for("qa", rc) == {
            "data": True,
            "inference": True,
        }

        (old / "effective_config.yaml").write_text(self.CONFIG.replace("n: 1", "n: 4"))
        assert session._legacy_match_for("qa", rc) == {
            "data": True,
            "inference": False,
        }

    @pytest.mark.parametrize(
        ("recorded", "current"),
        [
            ("api_key: <redacted>", "api_key: sk-live"),
            ("api_key_env: KEY_A", "api_key_env: KEY_B"),
            ("api_key: sk-old", "api_key_env: KEY_B"),
        ],
    )
    def test_a_credential_in_args_is_not_inference(
        self, tmp_path: Path, recorded: str, current: str
    ):
        def with_args(line: str) -> str:
            return self.CONFIG.replace(
                "    api_base: http://a/v1\n",
                f"    api_base: http://a/v1\n    args:\n      {line}\n",
            )

        old = tmp_path / "old"
        (old / "qa").mkdir(parents=True)
        (old / "effective_config.yaml").write_text(with_args(recorded))
        session = self._session(tmp_path, resume=old, body=with_args(current))
        rc = session._build_runner_config({}, {}, "qa")
        assert session._legacy_match_for("qa", rc) == {
            "data": True,
            "inference": True,
        }

    def test_no_recorded_config_leaves_stages_unknown(self, tmp_path: Path):
        (tmp_path / "old" / "qa").mkdir(parents=True)
        session = self._session(tmp_path, resume=tmp_path / "old")
        rc = session._build_runner_config({}, {}, "qa")
        assert session._legacy_match_for("qa", rc) is None

    @pytest.mark.anyio
    async def test_a_same_dir_resume_compares_the_config_before_its_rewrite(
        self, tmp_path: Path
    ):
        """`--resume` rewrites effective_config.yaml before tasks are built, so
        the recorded config is the one read before that rewrite."""
        await self._session(tmp_path)._persist_effective_config()
        cfg = _write_yaml(tmp_path, "cfg.yaml", self.CONFIG.replace("n: 1", "n: 4"))
        session = EvalSession(
            config_path=str(cfg),
            result_dir_override=str(tmp_path / "out"),
            resume=True,
        )
        await session._persist_effective_config()
        rc = session._build_runner_config({}, {}, "qa")
        assert session._legacy_match_for("qa", rc) == {
            "data": True,
            "inference": False,
        }

    @pytest.mark.anyio
    async def test_a_resume_dir_is_no_config_change(self, tmp_path: Path):
        """How a run starts is not what it runs: `resume` never reaches
        `runs[].config`, so it never regrades or refuses a resume."""
        await self._session(tmp_path)._persist_effective_config()
        body = self.CONFIG.replace(
            "    dataset: ds\n",
            "    dataset: ds\n    runner_config:\n      resume: /mnt/old/qa\n",
        )
        cfg = _write_yaml(tmp_path, "cfg.yaml", body)
        session = EvalSession(
            config_path=str(cfg),
            result_dir_override=str(tmp_path / "out"),
            resume=True,
        )
        await session._persist_effective_config()
        assert session._resume_config_diff == []

    def test_a_fresh_run_has_nothing_to_compare(self, tmp_path: Path):
        session = self._session(tmp_path)
        rc = session._build_runner_config({}, {}, "qa")
        assert session._legacy_match_for("qa", rc) is None


class TestDeprecatedResumeKeys:
    """`runner_config.auto_resume` and `.resume_from` fold into `.resume`."""

    BASE = TestResumeFromSurface.CONFIG

    def _session(self, tmp_path: Path, top: str = "", task: str = "", **kw):
        body = self.BASE
        if top:
            body = f"runner_config:\n{top}" + body
        if task:
            body = body.replace(
                "    dataset: ds\n", f"    dataset: ds\n    runner_config:\n{task}"
            )
        cfg = _write_yaml(tmp_path, "cfg.yaml", body)
        return EvalSession(
            config_path=str(cfg), result_dir_override=str(tmp_path / "out"), **kw
        )

    def test_auto_resume_becomes_resume(self, tmp_path: Path, warnings):
        session = self._session(tmp_path, top="  auto_resume: true\n")
        assert session._reified_config["runner_config"] == {"resume": True}
        rc = session._build_runner_config({}, session.config["runner_config"], "qa")
        assert rc.resume is True
        assert warnings == [
            "runner_config.auto_resume is deprecated; use runner_config.resume: true"
        ]

    def test_resume_from_becomes_resume(self, tmp_path: Path, warnings):
        session = self._session(tmp_path, task="      resume_from: /mnt/old/qa\n")
        task_rc = session._reified_config["tasks"]["qa"]["runner_config"]
        assert task_rc == {"resume": "/mnt/old/qa"}
        rc = session._build_runner_config(session.config["tasks"]["qa"], {}, "qa")
        assert rc.resume == Path("/mnt/old/qa")
        assert warnings == [
            "tasks.qa.runner_config.resume_from is deprecated; "
            "use tasks.qa.runner_config.resume <dir>"
        ]

    @pytest.mark.anyio
    async def test_effective_config_records_only_resume(self, tmp_path: Path):
        session = self._session(
            tmp_path,
            top="  auto_resume: false\n",
            task="      resume_from: /mnt/old/qa\n",
        )
        await session._persist_effective_config()
        persisted = yaml.safe_load(
            (tmp_path / "out" / "effective_config.yaml").read_text()
        )
        assert persisted["runner_config"] == {"resume": False}
        assert persisted["tasks"]["qa"]["runner_config"] == {"resume": "/mnt/old/qa"}

    def test_an_alias_beside_a_different_resume_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="disagree; set only"):
            self._session(
                tmp_path,
                task="      resume: /mnt/a/qa\n      resume_from: /mnt/b/qa\n",
            )

    def test_auto_resume_false_beside_resume_true_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="runner_config.auto_resume"):
            self._session(tmp_path, top="  resume: true\n  auto_resume: false\n")

    def test_a_per_task_alias_beats_the_cli(self, tmp_path: Path, warnings):
        session = self._session(
            tmp_path, task="      resume_from: /mnt/old/qa\n", resume=True
        )
        rc = session._build_runner_config(session.config["tasks"]["qa"], {}, "qa")
        assert rc.resume == Path("/mnt/old/qa")

    def test_the_session_alias_maps_to_resume(self, tmp_path: Path, warnings):
        (tmp_path / "old" / "qa").mkdir(parents=True)
        session = self._session(tmp_path, resume_from=tmp_path / "old")
        assert session.resume_override == tmp_path / "old"
        rc = session._build_runner_config({}, {}, "qa")
        assert rc.resume == tmp_path / "old" / "qa"
        assert warnings == ["resume_from is deprecated; use resume <dir>"]

    def test_run_session_takes_the_alias_alone(self, tmp_path: Path, warnings):
        """``resume`` left at its default must not read as ``false``, which
        disagrees with any ``resume_from``."""
        from sieval.cli.leaderboard import session as session_mod

        built: list[EvalSession] = []

        def build(*args, **kwargs):
            built.append(EvalSession(*args, **kwargs))
            raise KeyboardInterrupt  # stop before anything runs

        cfg = _write_yaml(tmp_path, "cfg.yaml", self.BASE)
        with (
            patch.object(session_mod, "EvalSession", build),
            pytest.raises(KeyboardInterrupt),
        ):
            session_mod.run_session(cfg, resume_from=tmp_path / "old")
        assert built[0].resume_override == tmp_path / "old"
        assert warnings == ["resume_from is deprecated; use resume <dir>"]
