"""Tests for EvalSession session-level persistence:
effective_config.yaml and infer_plans.yaml.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from sieval.cli.leaderboard.session import (
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
        """CLI overrides ARE baked into the body; endpoint injections are NOT."""
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
        assert loaded["models"]["base"]["args"]["seed"] == 0
        # Endpoint injection NOT persisted
        assert "api_base" not in loaded["models"]["base"]
        assert "api_key" not in loaded["models"]["base"]

    @pytest.mark.anyio
    async def test_header_advertises_sieval_run_as_universal_reproduce(
        self, tmp_path: Path
    ):
        """``sieval run`` is the universal reproduce command: for auto-
        served sessions it re-launches services from the preserved
        ``path`` / ``infer`` fields (the endpoint_map is intentionally
        NOT persisted); for pre-served sessions it's a pass-through.

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
    async def test_user_declared_api_base_is_preserved(self, tmp_path: Path):
        """User-set ``api_base`` / ``api_key`` in the source YAML flow through
        to effective_config.yaml unchanged. Only ``endpoint_map``-driven
        auto-injection (the auto-serve path in ``sieval run``) is stripped;
        the PR's "no api_base baked in" claim applies to injection only."""
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "models:\n"
            "  remote:\n"
            "    name: gpt-4\n"
            "    api_base: https://api.openai.com/v1\n"
            "    api_key: sk-user\n",
        )
        result_dir = tmp_path / "out"
        session = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )

        await session._persist_effective_config()

        loaded = yaml.safe_load((result_dir / "effective_config.yaml").read_text())
        m = loaded["models"]["remote"]
        assert m["api_base"] == "https://api.openai.com/v1"
        assert m["api_key"] == "sk-user"

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


class TestStrictResumeMatch:
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
    async def test_mismatched_deterministic_raises(self, tmp_path: Path):
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"

        # First run: no deterministic
        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        await s1._persist_effective_config()

        # Resume with --deterministic → mismatch
        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            deterministic_override=True,
            result_dir_override=str(result_dir),
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted"):
            await s2._persist_effective_config()

    @pytest.mark.anyio
    async def test_mismatched_model_override_raises(self, tmp_path: Path):
        cfg_path = _write_yaml(
            tmp_path, "cfg.yaml", "models:\n  base:\n    name: original\n"
        )
        result_dir = tmp_path / "out"

        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        await s1._persist_effective_config()

        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            model_override="different",
            result_dir_override=str(result_dir),
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted"):
            await s2._persist_effective_config()

    @pytest.mark.anyio
    async def test_endpoint_map_change_does_not_trigger_mismatch(self, tmp_path: Path):
        """Endpoints change across runs by nature — not part of strict match."""
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
        # Must not raise — endpoint changes are expected, endpoint_map is
        # not persisted into effective_config.yaml
        await s2._persist_effective_config()

    @pytest.mark.anyio
    async def test_resume_read_failure_surfaces_as_resume_aborted(
        self, tmp_path: Path, monkeypatch
    ):
        """If the persisted file is unreadable under ``--resume`` (EACCES,
        unlink race, etc.), the user gets a resume-shaped RuntimeError —
        not a bare OSError from the persistence internals. Rationale: a
        user who triggers this sees the same "Resume aborted" class of
        error as a body mismatch, with an actionable message, instead of
        a file-io traceback they may not know originates from persistence."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        result_dir = tmp_path / "out"

        # Seed an existing persisted file.
        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
        )
        await s1._persist_effective_config()
        assert (result_dir / "effective_config.yaml").exists()

        # Now resume with a read that simulates EACCES on the existing file.
        import anyio

        async def _fail_read(*_a, **_kw):
            raise PermissionError("simulated EACCES")

        monkeypatch.setattr(anyio.Path, "read_text", _fail_read)

        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
        )

        with pytest.raises(RuntimeError, match=r"Resume aborted: cannot read"):
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
    async def test_mismatched_infer_plans_raises(self, tmp_path: Path):
        """Symmetric to test_mismatched_*_raises for effective_config:
        _persist_infer_plans must also abort on strict-match mismatch."""
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models: {}\n")
        result_dir = tmp_path / "out"

        s1 = EvalSession(
            config_path=str(cfg_path),
            result_dir_override=str(result_dir),
            infer_plans={"m": _real_plan_dict()},
        )
        await s1._persist_infer_plans()

        # Resume with a plan that differs (different backend) → must mismatch
        mutated = _real_plan_dict()
        mutated["backend"] = "sglang"
        s2 = EvalSession(
            config_path=str(cfg_path),
            resume=True,
            result_dir_override=str(result_dir),
            infer_plans={"m": mutated},
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted"):
            await s2._persist_infer_plans()

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
        # Origin provenance preserved, and the change is appended as a record.
        assert "Persisted by sieval" in after_header
        assert "Resumed by sieval" in after_header
        assert "concurrency_limit: 8 → 2" in after_header

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
    async def test_resume_aborts_on_max_retries_change(self, tmp_path: Path):
        # max_retries is the failure signal (written into FAILED records) — strict.
        base = (
            "models:\n  base:\n    name: m\ntasks:\n  t:\n"
            "    runner_config:\n      max_retries: {}\n"
        )
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format(3))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        cfg_path.write_text(base.format(10))
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted"):
            await s2._persist_effective_config()

    @pytest.mark.anyio
    async def test_resume_aborts_on_profile_change(self, tmp_path: Path):
        # profile_* write per-record meta + the profiler summary file — strict.
        base = (
            "models:\n  base:\n    name: m\ntasks:\n  t:\n"
            "    runner_config:\n      profile_usage: {}\n"
        )
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format("true"))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        cfg_path.write_text(base.format("false"))
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted"):
            await s2._persist_effective_config()

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
    async def test_resume_aborts_on_throughput_plus_result_change(self, tmp_path: Path):
        cfg_path = _write_yaml(
            tmp_path,
            "cfg.yaml",
            "concurrency_limit: 8\nmodels:\n  base:\n    name: m\n",
        )
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        # Throughput AND a result-affecting field change together.
        cfg_path.write_text(
            "concurrency_limit: 2\ndeterministic: true\nmodels:\n  base:\n    name: m\n"
        )
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted") as exc:
            await s2._persist_effective_config()
        # Diff surfaces the result field, not the throughput noise.
        assert "deterministic" in str(exc.value)
        assert "concurrency_limit" not in str(exc.value)

    @pytest.mark.anyio
    async def test_resume_aborts_on_shard_samples_change(self, tmp_path: Path):
        base = (
            "models:\n  base:\n    name: m\ntasks:\n  t:\n"
            "    runner_config:\n      shard_samples: {}\n"
        )
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format(1024))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        cfg_path.write_text(base.format(512))
        s2 = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        with pytest.raises(RuntimeError, match=r"Resume aborted"):
            await s2._persist_effective_config()

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
        header = _split_header(after)[0]
        assert "runner_config.concurrency_limits.infer: 4 → 16" in header

    @pytest.mark.anyio
    async def test_resume_appends_accumulating_notes(self, tmp_path: Path):
        base = "concurrency_limit: {}\nmodels:\n  base:\n    name: m\n"
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", base.format(8))
        result_dir = tmp_path / "out"
        s1 = EvalSession(config_path=str(cfg_path), result_dir_override=str(result_dir))
        await s1._persist_effective_config()

        # Two successive resumes, each changing only throughput.
        for value in (4, 2):
            cfg_path.write_text(base.format(value))
            s = EvalSession(
                config_path=str(cfg_path),
                resume=True,
                result_dir_override=str(result_dir),
            )
            await s._persist_effective_config()

        after = (result_dir / "effective_config.yaml").read_text()
        header, body = _split_header(after)
        # Origin recorded once; every tolerated change appended in order.
        assert header.count("Persisted by sieval") == 1
        assert header.count("Resumed by sieval") == 2
        assert "concurrency_limit: 8 → 4" in header
        assert "concurrency_limit: 4 → 2" in header
        # Body reflects the latest run; header/body partition stays intact.
        assert yaml.safe_load(body)["concurrency_limit"] == 2
        assert _split_header(after) == (header, body)

    @pytest.mark.anyio
    async def test_resume_aborts_on_non_mapping_existing_file(self, tmp_path: Path):
        # A tampered persisted file that parses to a non-mapping must surface as
        # the documented RuntimeError, not an opaque AttributeError from the strip.
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        (result_dir / "effective_config.yaml").write_text(header + "- a\n- b\n")
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        s = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        with pytest.raises(RuntimeError, match=r"not a YAML mapping"):
            await s._persist_effective_config()

    @pytest.mark.anyio
    async def test_resume_aborts_on_unparseable_existing_file(self, tmp_path: Path):
        # A tampered persisted body that is not valid YAML must surface as the
        # documented RuntimeError, not the raw yaml.YAMLError from the strip path.
        result_dir = tmp_path / "out"
        result_dir.mkdir()
        header = _format_comment_header(
            title="Persisted by", source_config="/x", invocation="sieval run x"
        )
        (result_dir / "effective_config.yaml").write_text(header + "key: [unclosed\n")
        cfg_path = _write_yaml(tmp_path, "cfg.yaml", "models:\n  base:\n    name: m\n")
        s = EvalSession(
            config_path=str(cfg_path), resume=True, result_dir_override=str(result_dir)
        )
        with pytest.raises(RuntimeError, match=r"cannot parse existing"):
            await s._persist_effective_config()


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
