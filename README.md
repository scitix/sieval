# SiEval

SiEval is a **model delivery quality verification system** with an asynchronous streaming evaluation engine, iterative feedback loop, and resilient sharded persistence. It verifies the entire model delivery pipeline — training → conversion → inference → evaluation.

## Features

- **Asynchronous streaming** — process samples concurrently without waiting for batch completion
- **Iterative feedback loop** — multi-turn evaluation with feedback
- **Resilient persistence** — sharded, append-only storage for crash recovery
- **15 registered benchmark datasets** — AIME 2024, AIME 2025, AIME 2026, CMMLU, DROP, GPQA-Diamond, GSM8K, HMMT Feb 2026, HumanEval, IFEval, LiveCodeBench, MATH-500, MMLU, MMLU-Pro, T-Eval (math, code, reasoning, knowledge, instruction-following, tool-use)
- **Type-safe pipelines** — fully typed task stages (preprocess → infer → postprocess → feedback)
- **YAML-based configuration** — batch evaluation with model derivation and quota allocation
- **Inference orchestration** — recipe-driven inference with auto-resolve and backend abstraction (vLLM, SGLang)
- **Anomaly detection** — built-in detection rules for output quality, performance, and correctness
- **Profiling** — stage timing, I/O metrics, and token usage tracking

## Installation

**Requirements:** Unix (Linux, macOS), Python ≥ 3.12, [PDM](https://pdm-project.org/) (recommended) or pip

```bash
git clone https://github.com/scitix/sieval.git
cd sieval
pdm install          # or: pip install -e .
```

Optional extras (per-benchmark dependencies):

```bash
pip install -e ".[math]"     # AIME 2024/2025/2026, HMMT Feb 2026, MATH-500 (math-verify)
pip install -e ".[drop]"     # DROP (numpy, scipy)
pip install -e ".[ifeval]"   # IFEval (absl, langdetect, nltk, immutabledict)
pip install -e ".[t-eval]"   # T-Eval (numpy, sentence-transformers)
pip install -e ".[math,drop,ifeval,t-eval]"   # all extras at once
```

## Quick Start

> Dataset paths below use HuggingFace repo ids for HF-sourced datasets
> (e.g. `HuggingFaceH4/aime_2024`) and `${SIEVAL_DATA_DIR}/<name>` for
> URL-sourced datasets. Set `SIEVAL_DATA_DIR` (default `~/.sieval/data`)
> before running any command that resolves a URL-sourced dataset.

**Start from an example — two-step flow:**

```bash
cp examples/quickstart.yaml eval.yaml
$EDITOR eval.yaml           # set model checkpoint + container image

# Step 1: stage the data
sieval dataset download aime_2024

# Step 2: run eval
sieval eval eval.yaml
```

See [examples/README.md](examples/README.md) for more scenarios (leaderboard,
recipe overrides) and [examples/hardware/](examples/hardware/) for
hardware-pinned reference configs.

**Discover tasks / datasets:**

```bash
sieval dataset list                   # registered datasets + licenses + download status
sieval task list --domain Mathematics # filter tasks by domain
sieval dataset show aime_2024         # dataset detail, incl. the YAML path: to paste
sieval dataset download aime_2024     # stage data into $SIEVAL_DATA_DIR
```

**All-in-one** (launch inference, evaluate, cleanup — recommended entry point):

```bash
sieval run config.yaml
sieval run config.yaml --resume
```

**Evaluate against an already-online endpoint:**

```bash
sieval eval leaderboards/sft_fast_202511.yaml --model gpt-4o
# `sieval eval` is a shortcut for the underlying resource verb:
sieval leaderboard run leaderboards/sft_fast_202511.yaml --model gpt-4o
```

**Inference management:**

```bash
sieval infer start /path/to/Qwen3-8B          # auto-resolve and launch
sieval infer list                               # show running services
sieval infer logs qwen3-8b -f                   # stream engine logs
sieval infer stop qwen3-8b                      # graceful shutdown
```

**Programmatic usage:**

```python
import anyio

from sieval.datasets import MMLUDataset
from sieval.tasks import MMLUZeroShotGenTask
from sieval.core.models import ChatModel
from sieval.core.runners import TaskRunner, TaskRunnerConfig

async def main():
    dataset = MMLUDataset("cais/mmlu")
    model = ChatModel("gpt-4o", max_retries=3, concurrency_limit=128)
    task = MMLUZeroShotGenTask(dataset=dataset, model=model)

    runner = TaskRunner(
        task=task,
        config=TaskRunnerConfig(result_dir="./outputs/mmlu", auto_resume=True),
    )
    results = await runner.arun()
    print(results)

anyio.run(main)
```

## Documentation

- [Configuration Guide](docs/guide/configuration.md) — YAML format, task pipeline, model resource pool, anomaly detection
- [Concurrency Control](docs/guide/concurrency.md) — four-level concurrency model
- [Profiling & Observability](docs/guide/profiling.md) — stage timing, I/O metrics, token tracking
- [Inference Management](docs/guide/infer.md) — full infer subcommand reference

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, project architecture, code conventions, and the PR process.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{sieval2026,
  title = {SiEval: Asynchronous Streaming Evaluation Framework},
  author = {{ScitiX}},
  year = {2026},
  url = {https://github.com/scitix/sieval}
}
```
