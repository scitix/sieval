"""NL2SH-ALFA — 0-shot natural-language-to-Bash, upstream's Base reading.

The benchmark with no method layered on it: the prompt asks for a bare command
and the reply is graded as it arrives. This is Table 5's **Base** column
(arXiv:2502.06858) -- gpt-4o-2024-08-06 0.74, gpt-4o-mini-2024-07-18 0.71,
gpt-4-0613 0.68, gpt-3.5-turbo-0125 0.58 -- measured at n=300, where one standard
error is about 2.5 pp.

The bare name is this reading rather than the ``_parse`` sibling because the
paper presents parsing as one of four *methods* (constrained decoding, parsing,
in-context learning, in-weight learning) applied on top of the benchmark, and the
unqualified name belongs to the un-intervened protocol.

**One divergence is unavoidable and is not in the sibling.** The prompt is
Figure 7 of the paper, and Figure 7 is the only place it exists: upstream ships
``example.ipynb`` and ``model_comparison.ipynb``, and the sentences that
distinguish this reading -- "You will not output markdown or other formatting.
You will not include additional information." -- appear in neither. So how
upstream split that prompt across a system and a user turn is not recoverable
from any artifact. This task uses the same split its shipped scripts use for the
sibling figure (instruction as the system turn, the bare instruction as the user
turn), which is the most defensible inference available and is still an
inference. A system/user split can move a score, so it is the reason this reading
cannot be argued to byte-fidelity the way ``nl2sh_alfa_0shot_gen_parse`` can.

Expect this reading to charge modern chat models for formatting compliance as
well as translation: a reply that arrives fenced is executed fenced. That is
upstream's measurement, and the paper's own finding is that parsing recovers up
to 32% of it -- which is what the sibling task exists to show.

References:

* Paper: <https://arxiv.org/abs/2502.06858>
* Harness: <https://github.com/westenfelder/InterCode-ALFA>
* Dataset: <https://huggingface.co/datasets/westenfelder/NL2SH-ALFA>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._nl2sh_alfa_base import (
    NL2SH_ALFA_HARNESS_URL,
    NL2SH_ALFA_SHARED_NOTES,
    NL2SHAlfaSharedZeroShotGenTask,
)

#: Paper Figure 7, "Translation Prompt: Baseline", verbatim minus the trailing
#: `natural_language_prompt` placeholder, which becomes the user turn.
BASELINE_SYSTEM_PROMPT = (
    "Your task is to translate a natural language instruction to a Bash "
    "command. You will receive an instruction in English and output a Bash "
    "command that can be run in a Linux terminal. You will not output markdown "
    "or other formatting. You will not include additional information."
)


@sieval_task(
    name="nl2sh_alfa_0shot_gen",
    display_name="NL2SH-ALFA (0-shot, generative)",
    description=(
        "Natural language to Bash, execution-graded; upstream's Base prompt, "
        "reply used verbatim."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "nl2bash", "code-exec"),
    deps_group="nl2sh-alfa",
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="intercode-alfa",
        url=NL2SH_ALFA_HARNESS_URL,
        notes=(
            "BASE reading: paper Figure 7's prompt, model reply graded verbatim "
            "with no markdown stripping. Table 5 Base column -- gpt-4o-2024-08-06 "
            "0.74, gpt-4o-mini-2024-07-18 0.71, gpt-4-0613 0.68, "
            "gpt-3.5-turbo-0125 0.58, at temperature 0 and seed 123. NOT "
            "byte-faithful in one respect that the _parse sibling is: no upstream "
            "script ships this prompt (neither example.ipynb nor "
            "model_comparison.ipynb contains its 'will not output markdown' "
            "sentences), so its split across a system and a user turn is inferred "
            "from how the shipped scripts render the sibling figure. "
            "experimental until a per-sample verdict comparison against upstream "
            "icalfa and an alignment run against those published numbers both "
            f"land. {NL2SH_ALFA_SHARED_NOTES}"
        ),
    ),
)
class NL2SHAlfaZeroShotGenTask(NL2SHAlfaSharedZeroShotGenTask):
    SYSTEM_PROMPT = BASELINE_SYSTEM_PROMPT
    PARSE_MARKDOWN = False
