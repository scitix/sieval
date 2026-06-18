#!/usr/bin/env python3
"""Generate a multi-length RULER sweep config (the full 13-task suite per length).

RULER's headline number is the 13-task average at each context length; its
"effective length" is the longest length whose average still clears a fixed
threshold (see ``sieval leaderboard ruler-effective``). Reproducing that means
running the same 13 configs at every length tier — RULER (``config_tasks.sh``) and
OpenCompass (``eval_ruler.py``) both emit these programmatically rather than by
hand. This script does the same: it defines the 13 RULER configs once and expands
them across the requested lengths, so there is a single source of truth and no
copy-paste drift across 78+ near-identical blocks.

YARN: a model is only extrapolated past its native context. For each length tier
``> --native-ctx`` the script attaches an engine override with
``factor = ceil(length / native_ctx)`` (e.g. native 32768 → 64K uses factor 2,
128K uses factor 4); tiers ``<= native-ctx`` get no YARN (static YARN would hurt
short-context scores, which is why the tiers are deployed separately).

Usage:
    python scripts/gen_ruler_sweep.py \
        --lengths 4096,8192,16384,32768,65536,131072 \
        --checkpoint /path/to/Qwen3-32B --native-ctx 32768 \
        --backend sglang --endpoint chat \
        --out examples/ruler-multilength.yaml

This emits a config for the local-launch path (``sieval run``), where YARN is set
via engine overrides. For an already-running API endpoint, YARN is fixed
server-side at deploy time — drop the overrides and point ``api_base`` at it.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import argparse
import json
import math
import re

# Task class mapping: inferred from task type in synthetic.yaml
_TASK_CLASS_MAP = {
    "niah": "RulerNiahZeroShotGenTask",
    "variable_tracking": "RulerVtZeroShotGenTask",
    "common_words_extraction": "RulerCweZeroShotGenTask",
    "freq_words_extraction": "RulerFweZeroShotGenTask",
    "qa": "RulerQaZeroShotGenTask",
}

# Dataset class mapping: inferred from task type in synthetic.yaml
_DATASET_CLASS_MAP = {
    "niah": "RulerNiahDataset",
    "variable_tracking": "RulerVtDataset",
    "common_words_extraction": "RulerCweDataset",
    "freq_words_extraction": "RulerFweDataset",
    "qa": "RulerQaDataset",
}

# Dataset path mapping: where to find data in SIEVAL_DATA_DIR
_DATASET_PATH_MAP = {
    "niah": "ruler_niah",
    "variable_tracking": None,
    "common_words_extraction": "ruler_cwe",
    "freq_words_extraction": None,
    "qa": None,  # Multi-path: "ruler_qa" or "hotpotqa"
}

# NIAH variants kept for compatibility (args loaded from synthetic.yaml)
_NIAH_VARIANTS = [
    ("single_1", {}),
    ("single_2", {}),
    ("single_3", {}),
    ("multikey_1", {}),
    ("multikey_2", {}),
    ("multikey_3", {}),
    ("multivalue", {}),
    ("multiquery", {}),
]

# Other tasks: (dataset_class, subdir, task_type_for_class_lookup)
_OTHER_TASKS = {
    "vt": ("RulerVtDataset", None, "variable_tracking"),
    "cwe": ("RulerCweDataset", "ruler_cwe", "common_words_extraction"),
    "fwe": ("RulerFweDataset", None, "freq_words_extraction"),
    "qa_squad": ("RulerQaDataset", "ruler_qa", "qa"),
    "qa_hotpotqa": ("RulerQaDataset", "hotpotqa/hotpot_qa", "qa"),
}


def _len_tag(length: int) -> str:
    """4096 -> '4k', 131072 -> '128k'."""
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


def _load_synthetic_config(path: str) -> dict:
    """Load NIAH and other task configs from synthetic.yaml."""
    import yaml
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def _scalar(v) -> str:
    """Render a YAML flow scalar, quoting strings that aren't safe bare tokens.

    A JSON blob like ``{"rope_scaling":...}`` must be double-quoted, else YAML
    parses it as a nested mapping instead of a string (the engine override would
    then reach the launcher with the wrong type).
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if not isinstance(v, str):
        return str(v)
    # Bare-safe: plain word/number/path tokens with no YAML-significant chars.
    if re.fullmatch(r"[A-Za-z0-9_./-]+", v):
        return v
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _flow(d: dict) -> str:
    """Render a dict as a compact YAML flow mapping."""
    return "{ " + ", ".join(f"{k}: {_scalar(v)}" for k, v in d.items()) + " }"


def _model_name(base: str, length: int, native: int) -> str:
    if length <= native:
        return f"{base}-native"
    return f"{base}-yarn{_len_tag(length)}"


def build(args) -> str:
    # Parse lengths: either direct numbers or multipliers of 1024
    raw_lengths = [int(x) for x in args.lengths.split(",")]
    lengths = [x * 1024 if x < 1024 else x for x in raw_lengths]
    native = args.native_ctx
    model_type = "chat"
    ctx_key = "max_model_len" if args.backend == "vllm" else "context_length"
    # Size prompts with the evaluated model's own tokenizer (RULER aligns these),
    # falling back to the checkpoint path when not given.
    tokenizer_model = args.tokenizer_model or args.checkpoint

    # Load synthetic.yaml config to reference task-level args
    try:
        import os
        import yaml
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "../sieval/community/ruler/synthetic.yaml"
        )
        with open(yaml_path, encoding="utf-8") as f:
            synthetic_config = yaml.safe_load(f) or {}
    except Exception:
        synthetic_config = {}

    # --- models: one per distinct serving config (native, then one per YARN tier) ---
    model_blocks: list[str] = []
    model_for_length: dict[int, str] = {}
    seen: set[str] = set()
    for length in lengths:
        name = _model_name(args.model_base, length, native)
        model_for_length[length] = name
        if name in seen:
            continue
        seen.add(name)

        serve_ctx = native if length <= native else length
        overrides = {ctx_key: serve_ctx}

        # Enable SGLang deterministic inference and increase max_seq_len (server-side parameter)
        if args.backend == "sglang":
            # overrides["enable_deterministic_inference"] = True
            # For 128K+ sequences, need to disable CUDA graphs to avoid memory limits
            if serve_ctx >= 131072:
                overrides["disable_cuda_graph"] = True

        yarn_note = ""
        if length > native:
            # Use fixed YARN factor if specified, else compute adaptive factor
            if args.yarn_factor:
                factor = float(args.yarn_factor)
            else:
                factor = math.ceil(length / native)
            yarn_note = f"  # YARN factor={factor} ({_len_tag(length)} > native {_len_tag(native)})"  # noqa: E501
            scaling = {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": native,
            }
            if args.backend == "vllm":
                overrides["rope_scaling"] = json.dumps(scaling)
            else:  # sglang injects HF-config overrides as a JSON blob
                overrides["json_model_override_args"] = json.dumps(
                    {"rope_scaling": scaling}
                )

        block = [
            f"  {name}:{yarn_note}",
            "    args:",
            "      concurrency_limit: 64",
            "      temperature: 0.7",
            "      top_p: 0.8",
            "      presence_penalty: 1.5",
            "      extra_body:",
            "        enable_thinking: false",
            "        top_k: 20",
            "        continue_final_message: True",
            "        add_generation_prompt: False",
        ]
        block += [
            "    infer:",
            f"      backend: {args.backend}",
            f"      recipe: {args.recipe}",
            f"      checkpoint: {args.checkpoint}        # EDIT ME",
            f"      overrides: {_flow(overrides)}",
            "    infer_meta:",
            f"      gpu: {args.gpu}",
            "      image: lmsysorg/sglang:latest",
        ]
        model_blocks.append("\n".join(block))

    # --- datasets + tasks, expanded across every length tier ---
    ds_lines: list[str] = []
    task_lines: list[str] = []
    for length in lengths:
        tag = _len_tag(length)
        ns = args.num_samples
        model = model_for_length[length]

        for variant, vargs in _NIAH_VARIANTS:
            name = f"ruler_niah_{variant}_{tag}"
            # Use args from synthetic.yaml if available, else fall back to local config
            synth_key = f"niah_{variant}"
            synth_args = synthetic_config.get(synth_key, {}).get("args", {})
            a = {
                "max_seq_length": length,
                "num_samples": ns,
                "tokenizer_type": "hf",
                "tokenizer_path": tokenizer_model,
                "enable_thinking": args.enable_thinking,
                **(synth_args or vargs),
            }
            ds_lines.append(f"  {name}:")
            ds_lines.append("    class: RulerNiahDataset")
            ds_lines.append('    path: "${SIEVAL_DATA_DIR}/ruler_niah"')
            ds_lines.append(f"    args: {_flow(a)}")
            task_lines.append(
                f"  {name}: {_flow({'class': _TASK_CLASS_MAP['niah'], 'dataset': name, 'model': model})}"  # noqa: E501
            )

        for key, (ds_cls, subdir, task_type) in _OTHER_TASKS.items():
            name = f"ruler_{key}_{tag}"
            # Map internal keys to synthetic.yaml keys
            synth_key_map = {"qa_squad": "qa_1", "qa_hotpotqa": "qa_2"}
            synth_key = synth_key_map.get(key, key)
            # Use args from synthetic.yaml if available, else empty dict
            synth_args = synthetic_config.get(synth_key, {}).get("args", {})
            a = {
                "max_seq_length": length,
                "num_samples": ns,
                "tokenizer_type": "hf",
                "tokenizer_path": tokenizer_model,
                "enable_thinking": args.enable_thinking,
                **synth_args,
            }
            ds_lines.append(f"  {name}:")
            ds_lines.append(f"    class: {ds_cls}")
            ds_lines.append(
                f'    path: "${{SIEVAL_DATA_DIR}}/{subdir}"'
                if subdir
                else '    path: "."'
            )
            ds_lines.append(f"    args: {_flow(a)}")
            task_lines.append(
                f"  {name}: {_flow({'class': _TASK_CLASS_MAP[task_type], 'dataset': name, 'model': model})}"  # noqa: E501
            )

    bar = "# " + "-" * 78
    header = f"""{bar}
# RULER multi-length sweep — {len(lengths)} length tiers x 13 tasks
{bar}
# GENERATED by scripts/gen_ruler_qwen3_8b_sglang.py — edit that script, not this file.
#   lengths: {", ".join(_len_tag(x) for x in lengths)}   native ctx: {_len_tag(native)}
#   backend: {args.backend}   tokenizer_model: {tokenizer_model}  (prompts sized with this; keep == model)
#
# Each length tier runs the full 13-task RULER suite; the per-tier 13-task
# average is RULER's score at that length, and the "effective length" is the
# longest tier still clearing the threshold — compute both with
#   sieval leaderboard ruler-effective {args.result_dir}
#
# YARN: tiers <= native ctx run with no extrapolation; tiers > native ctx get an
# engine override with factor=ceil(length/native). For an API endpoint, YARN is
# fixed server-side — delete `overrides` and point `api_base` at the deployment.
#
# `num_samples` is {args.num_samples} here; RULER uses 500. Large lengths are slow
# (synthesis tokenizes every sample).
# ------------------------------------------------------------------------------
result_dir: {args.result_dir}
"""

    return (
        header
        + "\nmodels:\n"
        + "\n\n".join(model_blocks)
        + "\n\ndatasets:\n"
        + "\n".join(ds_lines)
        + "\n\ntasks:\n"
        + "\n".join(task_lines)
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lengths",
        default="4,8,16,32,128",
        help="Context lengths in 1K units (4 -> 4096, 8 -> 8192, etc). "
        "Can also be raw byte values for lengths >= 1024.",
    )
    p.add_argument("--native-ctx", type=int, default=32768, help="model native context")
    p.add_argument("--checkpoint", default="/root/models/Qwen3-8b")
    p.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer used to size prompts; default = --checkpoint so synthesis "
        "matches the evaluated model (RULER aligns these). Use 'gpt-4' for tiktoken, "
        "or an HF id / local path otherwise.",
    )
    p.add_argument("--model-base", default="Qwen3-8B", help="model name")
    p.add_argument("--backend", choices=["sglang", "vllm"], default="sglang")
    p.add_argument(
        "--yarn-factor",
        type=float,
        default=None,
        help="Fixed YARN scaling factor (e.g., 4). If not specified, uses adaptive factor "
        "(ceil(length / native_ctx)) for each length > native_ctx.",
    )
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument(
        "--recipe",
        default="qwen3-8b",
        help="Recipe name from sieval/infer/recipes/ (for sieval infer). "
        "Must match the model size: qwen3-8b for ~8B params.",
    )
    p.add_argument(
        "--gpu",
        default="H200-141G",
        help="GPU model for infer_meta (e.g., H200-141G, H100-80G, A100-40G). "
        "Must match a profile key in the recipe.",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable reasoning/thinking mode in Qwen3 (longer generation, higher tokens). "
        "When enabled, consider increasing max_completion_tokens.",
    )
    p.add_argument("--result-dir", default="./outputs/ruler_qwen3_8b_sglang_test")
    p.add_argument("--out", default="-", help="output path, or '-' for stdout")
    args = p.parse_args()

    text = build(args)
    if args.out == "-":
        print(text, end="")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
