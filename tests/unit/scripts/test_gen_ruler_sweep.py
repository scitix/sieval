"""Tests for scripts/gen_ruler_sweep.py — multi-length RULER sweep generator.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

# scripts/ is not a package — add it to sys.path so we can import directly.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gen_ruler_sweep import _len_tag, build  # noqa: E402


def _args(**overrides):
    base = {
        "lengths": "4096,8192,16384,32768,65536,131072",
        "native_ctx": 32768,
        "checkpoint": "/path/to/model",
        "tokenizer_model": None,
        "model_base": "model",
        "backend": "sglang",
        "endpoint": "chat",
        "num_samples": 500,
        "result_dir": "./outputs/ruler-sweep",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_len_tag():
    assert _len_tag(4096) == "4k"
    assert _len_tag(131072) == "128k"
    assert _len_tag(1000) == "1000"  # non-power-of-1024 stays raw


def test_full_sweep_has_13_tasks_per_length():
    doc = yaml.safe_load(build(_args()))
    # 6 lengths × 13 tasks = 78 datasets + 78 tasks.
    assert len(doc["datasets"]) == 78
    assert len(doc["tasks"]) == 78
    # Exactly 13 tasks carry each length suffix.
    for tag in ("4k", "8k", "16k", "32k", "64k", "128k"):
        n = sum(1 for name in doc["tasks"] if name.endswith(f"_{tag}"))
        assert n == 13, f"{tag}: {n}"


def test_yarn_only_above_native_ctx():
    doc = yaml.safe_load(build(_args()))
    models = doc["models"]
    # <= 32k native → one shared no-YARN model; 64k and 128k → YARN models.
    assert "model-native" in models
    assert "model-yarn64k" in models
    assert "model-yarn128k" in models

    # native model carries no rope_scaling.
    native_ov = models["model-native"]["infer"]["overrides"]
    assert "json_model_override_args" not in native_ov
    assert native_ov["context_length"] == 32768

    # factor = ceil(length / native): 64k→2, 128k→4.
    import json

    f64 = json.loads(
        models["model-yarn64k"]["infer"]["overrides"]["json_model_override_args"]
    )
    f128 = json.loads(
        models["model-yarn128k"]["infer"]["overrides"]["json_model_override_args"]
    )
    assert f64["rope_scaling"]["factor"] == 2.0
    assert f128["rope_scaling"]["factor"] == 4.0
    assert f128["rope_scaling"]["original_max_position_embeddings"] == 32768


def test_vllm_backend_uses_max_model_len_and_rope_scaling():
    doc = yaml.safe_load(build(_args(backend="vllm")))
    ov = doc["models"]["model-yarn128k"]["infer"]["overrides"]
    assert "max_model_len" in ov
    assert "rope_scaling" in ov  # vllm flag name, not json_model_override_args


def test_tokenizer_model_defaults_to_checkpoint():
    # Synthesis must size prompts with the evaluated model's tokenizer (RULER
    # aligns these): every dataset inherits the checkpoint unless overridden.
    doc = yaml.safe_load(build(_args(checkpoint="/models/Qwen3-32B")))
    tms = {ds["args"]["tokenizer_model"] for ds in doc["datasets"].values()}
    assert tms == {"/models/Qwen3-32B"}


def test_tokenizer_model_override():
    doc = yaml.safe_load(build(_args(tokenizer_model="gpt-4")))
    tms = {ds["args"]["tokenizer_model"] for ds in doc["datasets"].values()}
    assert tms == {"gpt-4"}


def test_single_native_length_has_no_yarn_model():
    doc = yaml.safe_load(build(_args(lengths="4096,32768")))
    assert list(doc["models"]) == ["model-native"]
