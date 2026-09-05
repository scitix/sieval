"""QuoteBench, raw contract: the reply is executed verbatim.

The matched baseline of the paper's crossover. Its sibling
`quotebench_nested_shell_0shot_gen` is the other matched path; the interesting
number is the pair, not either alone.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.community.quotebench.core import SYSTEM_PROMPT
from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._quotebench_base import QuoteBenchTask

QUOTEBENCH_COMMIT = "693325a671e65f889e5cd9d83965db9cc3b26dc2"


@sieval_task(
    name="quotebench_raw_0shot_gen",
    display_name="QuoteBench (raw contract, 0-shot)",
    description="QuoteBench — literal intent through one shell boundary, raw.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "shell", "code-exec"),
    model_type="chat",
    status="experimental",
    reference_kind="procedure",
    reference_impl=ReferenceImpl(
        source="quotebench",
        url=(
            "https://github.com/LeonardNJU/quoteBench/blob/"
            f"{QUOTEBENCH_COMMIT}/quotebench/scenarios.py"
        ),
        notes=(
            "System prompt is upstream's SYSTEM_PROMPT verbatim; the reply is "
            "sent unmodified, with no extraction, because upstream passes it "
            "straight to `bash -c` and scores a fenced or chatty answer as the "
            "shell failure it becomes. Grading runs in the vendored "
            "code-evaluator's `quotebench` source (fixture setup, one bounded "
            "`bash -c`, exact-final-state check), never in-process. Anchored on "
            "upstream's released rollouts (HF lsamc/QuoteBench-Rollouts @ "
            "69957a53): replaying the raw-vs-nested arm's stored replies "
            "through the HTTP path reproduces upstream's recorded GNU verdicts "
            "224/224 on `passed` and 224/224 on failure class, across all four "
            "crossover cells, and recomputes the published gpt-5.5 row exactly. "
            "`experimental`: the shipped path grades inside a GNU userland image "
            "that has not yet been built or run, so no score impact is "
            "quantified against it. Upstream bounds each command at 15s; "
            "measured cost on the frozen core is a 2ms median and a 17ms max "
            "per attempt, so no bound binds. Single-draw upstream: the published "
            "crossover is one reply per (task, contract), so n=1 matches it — "
            "and n!=1 is refused at construction rather than defaulted, because "
            "every axis reported here (headline, the seven failure-class counts, "
            "the per-tier and per-scenario rates) reads one rollout per sample, "
            "and a failure class has no agreed reading across repeats. "
            "The off-diagonal damage and compensation effects need the same "
            "stored reply re-executed under the other transport, which is a "
            "replay across both tasks' persisted finals rather than a metric "
            "either one can report."
        ),
    ),
)
class QuoteBenchRawZeroShotGenTask(QuoteBenchTask):
    SYSTEM_PROMPT = SYSTEM_PROMPT
    CONTRACT = "raw"
