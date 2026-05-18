"""Tests for sieval.infer.deployer — collect_basic_env().

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sieval.infer.config import InferEnv
from sieval.infer.deployer import collect_basic_env


@pytest.mark.anyio
async def test_collect_basic_env_with_nvidia_smi():
    query_output = (
        "NVIDIA H100 80GB HBM3, 550.54.15\nNVIDIA H100 80GB HBM3, 550.54.15\n"
    )
    header_output = (
        "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |"
    )

    def mock_run(cmd, **_kwargs):
        mock_result = MagicMock()
        mock_result.returncode = 0
        if any("--query-gpu" in arg for arg in cmd):
            mock_result.stdout = query_output
        else:
            mock_result.stdout = header_output
        mock_result.stderr = ""
        return mock_result

    with patch("sieval.infer.deployer.subprocess.run", side_effect=mock_run):
        env = await collect_basic_env()

    assert isinstance(env, InferEnv)
    assert env.framework == "unknown"
    assert env.gpu_count == 2
    assert "H100" in env.gpu_model
    assert env.driver_version == "550.54.15"
    assert env.cuda_version == "12.4"
    assert env.python_version != ""


@pytest.mark.anyio
async def test_collect_basic_env_no_gpu():
    with patch(
        "sieval.infer.deployer.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        env = await collect_basic_env()

    assert isinstance(env, InferEnv)
    assert env.framework == "unknown"
    assert env.gpu_count == 0
    assert env.gpu_model == ""
    assert env.cuda_version == ""
    assert env.python_version != ""


@pytest.mark.anyio
async def test_collect_basic_env_detects_sglang_version():
    mock_sglang = SimpleNamespace(__version__="0.4.5")
    with (
        patch(
            "sieval.infer.deployer.subprocess.run",
            side_effect=FileNotFoundError,
        ),
        patch.dict("sys.modules", {"sglang": mock_sglang}),
    ):
        env = await collect_basic_env("sglang")

    assert env.framework == "sglang==0.4.5"


@pytest.mark.anyio
async def test_collect_basic_env_detects_vllm_version():
    mock_info = SimpleNamespace(vllm_version="0.8.3")
    mock_collect_env = SimpleNamespace(get_env_info=lambda: mock_info)
    mock_vllm = SimpleNamespace(collect_env=mock_collect_env)
    with (
        patch(
            "sieval.infer.deployer.subprocess.run",
            side_effect=FileNotFoundError,
        ),
        patch.dict(
            "sys.modules",
            {"vllm": mock_vllm, "vllm.collect_env": mock_collect_env},
        ),
    ):
        env = await collect_basic_env("vllm")

    assert env.framework == "vllm==0.8.3"


@pytest.mark.anyio
async def test_collect_basic_env_framework_fallback_on_import_error():
    """When the framework package is not installed, fall back gracefully."""
    with (
        patch(
            "sieval.infer.deployer.subprocess.run",
            side_effect=FileNotFoundError,
        ),
        patch.dict("sys.modules", {"sglang": None}),
    ):
        env = await collect_basic_env("sglang")

    # Should still contain the backend name, just no version
    assert env.framework == "sglang"
