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

# The 13 RULER configs, defined once. Each NIAH variant shares RulerNiahDataset
# but differs by args; vt/cwe/fwe/qa map to their own datasets.
_NIAH_VARIANTS = [
    (
        "single_1",
        {
            "type_haystack": "repeat",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "single_2",
        {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "single_3",
        {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "uuids",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "multikey_1",
        {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 4,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "multikey_2",
        {
            "type_haystack": "needle",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "multikey_3",
        {
            "type_haystack": "needle",
            "type_needle_k": "uuids",
            "type_needle_v": "uuids",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "multivalue",
        {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 4,
            "num_needle_q": 1,
        },
    ),  # noqa: E501
    (
        "multiquery",
        {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 4,
        },
    ),  # noqa: E501
]

# task_key -> (dataset class, base args, data subdir under SIEVAL_DATA_DIR or None)
_OTHER_TASKS = {
    "vt": ("RulerVtDataset", {"num_chains": 1, "num_hops": 4}, None),
    "cwe": ("RulerCweDataset", {"freq_cw": 30, "freq_ucw": 3, "num_cw": 10}, None),
    "fwe": ("RulerFweDataset", {"alpha": 2.0}, None),
    "qa_squad": ("RulerQaDataset", {"dataset": "squad"}, "ruler_qa"),
    "qa_hotpotqa": ("RulerQaDataset", {"dataset": "hotpotqa"}, "ruler_qa"),
}

# Endpoint → (chat task suffix, model_type)
_TASK_CLASS = {
    "chat": {
        "niah": "RulerNiahZeroShotGenTask",
        "vt": "RulerVtZeroShotGenTask",
        "cwe": "RulerCweZeroShotGenTask",
        "fwe": "RulerFweZeroShotGenTask",
        "qa": "RulerQaZeroShotGenTask",
    },
    "base": {
        "niah": "RulerNiahZeroShotBaseGenTask",
        "vt": "RulerVtZeroShotBaseGenTask",
        "cwe": "RulerCweZeroShotBaseGenTask",
        "fwe": "RulerFweZeroShotBaseGenTask",
        "qa": "RulerQaZeroShotBaseGenTask",
    },
}


def _len_tag(length: int) -> str:
    """4096 -> '4k', 131072 -> '128k'."""
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


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
    lengths = [int(x) for x in args.lengths.split(",")]
    native = args.native_ctx
    endpoint = args.endpoint
    model_type = "chat" if endpoint == "chat" else "gen"
    cls = _TASK_CLASS[endpoint]
    ctx_key = "max_model_len" if args.backend == "vllm" else "context_length"

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
        yarn_note = ""
        if length > native:
            factor = math.ceil(length / native)
            yarn_note = f"  # YARN factor={factor} ({_len_tag(length)} > native {_len_tag(native)})"  # noqa: E501
            scaling = {
                "rope_type": "yarn",
                "factor": float(factor),
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
            "      temperature: 0.0                # RULER uses greedy decoding",
        ]
        if endpoint == "chat":
            block += [
                "      extra_body:",
                "        chat_template_kwargs:",
                "          enable_thinking: false  # set true + raise max_tokens for thinking",  # noqa: E501
            ]
        block += [
            "    infer:",
            f"      backend: {args.backend}",
            f"      checkpoint: {args.checkpoint}        # EDIT ME",
            f"      overrides: {_flow(overrides)}",
            "    infer_meta:",
            "      gpu: H100-80G",
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
            a = {"max_seq_length": length, "num_samples": ns, **vargs}
            ds_lines.append(f"  {name}:")
            ds_lines.append("    class: RulerNiahDataset")
            ds_lines.append('    path: "${SIEVAL_DATA_DIR}/ruler_niah"')
            ds_lines.append(f"    args: {_flow(a)}")
            task_lines.append(
                f"  {name}: {_flow({'class': cls['niah'], 'dataset': name, 'model': model})}"  # noqa: E501
            )

        for key, (ds_cls, bargs, subdir) in _OTHER_TASKS.items():
            name = f"ruler_{key}_{tag}"
            a = {"max_seq_length": length, "num_samples": ns, **bargs}
            ds_lines.append(f"  {name}:")
            ds_lines.append(f"    class: {ds_cls}")
            ds_lines.append(
                f'    path: "${{SIEVAL_DATA_DIR}}/{subdir}"'
                if subdir
                else '    path: "."'
            )
            ds_lines.append(f"    args: {_flow(a)}")
            tkey = "qa" if key.startswith("qa") else key
            task_lines.append(
                f"  {name}: {_flow({'class': cls[tkey], 'dataset': name, 'model': model})}"  # noqa: E501
            )

    bar = "# " + "-" * 78
    header = f"""{bar}
# RULER multi-length sweep — {len(lengths)} length tiers x 13 tasks
{bar}
# GENERATED by scripts/gen_ruler_sweep.py — edit that script, not this file.
#   lengths: {", ".join(_len_tag(x) for x in lengths)}   native ctx: {_len_tag(native)}
#   endpoint: {endpoint} ({model_type})   backend: {args.backend}
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
    p.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072")
    p.add_argument("--native-ctx", type=int, default=32768, help="model native context")
    p.add_argument("--checkpoint", default="/path/to/your/model")
    p.add_argument("--model-base", default="model", help="model-name prefix per tier")
    p.add_argument("--backend", choices=["sglang", "vllm"], default="sglang")
    p.add_argument("--endpoint", choices=["chat", "base"], default="chat")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--result-dir", default="./outputs/ruler-sweep")
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
