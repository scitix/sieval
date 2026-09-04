"""NL2SH-ALFA — 0-shot natural-language-to-Bash, upstream's Parse reading.

Same 300 instructions and the same grader as ``nl2sh_alfa_0shot_gen``; two
differences, both upstream's:

* the prompt drops the baseline's two formatting sentences, so the model is not
  told to avoid markdown;
* the reply passes through ``parse_bash``, which strips one markdown fence --
  ```` ```bash ````, then a bare ```` ``` ````, then a single backtick span, and
  returns the reply unchanged when none matches.

This is Table 5's **Parse** column: gpt-4o-2024-08-06 0.73,
gpt-4o-mini-2024-07-18 0.72, gpt-4-0613 0.68, gpt-3.5-turbo-0125 0.67,
qwen2.5-coder-7b-instruct 0.62, llama-3.1-8b-instruct 0.53.

**This is the reading with the stronger fidelity claim.** It is the protocol
upstream actually ships runnable code for -- ``code/example.ipynb`` and
``code/model_comparison.ipynb`` both use exactly this system prompt, this bare
user turn, this ``parse_bash``, and ``submit_command(index, command,
eval_mode="embed", eval_param=0.75)``. So prompt, extraction and FEH setting are
copied rather than inferred, which the Base sibling's prompt split cannot be.

The ``_parse`` variant name is a deliberate extension of the variant vocabulary
in ``sieval/tasks/CLAUDE.md`` rather than a coined one: one paper publishes two
protocols over one benchmark and one grader, differing only in prompt text and an
output extractor, with a separate published column for each. Neither is a defect
fix, so neither is ``_fixed``; neither is a different measurement regime, so
neither is a mode.

Why both readings ship: the gap between them is the paper's own headline finding.
Reading Table 5's two columns, it is ~1 pp on gpt-4o, 9 pp on gpt-3.5-turbo,
20 pp on llama-3.2-1b and 32 pp on llama-3.2-3b (0.17 to 0.49) -- so a single
number for "NL2SH accuracy" hides whether a model was scored on translation or on
formatting compliance.

References:

* Paper: <https://arxiv.org/abs/2502.06858>
* Reproduction script: <https://github.com/westenfelder/NL2SH/blob/b405201aeac7630353e7b46ebe7d069b70b4506c/code/model_comparison.ipynb>
* Harness: <https://github.com/westenfelder/InterCode-ALFA>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._nl2sh_alfa_base import (
    NL2SH_ALFA_HARNESS_URL,
    NL2SH_ALFA_SHARED_NOTES,
    NL2SHAlfaSharedZeroShotGenTask,
)

#: The system prompt every shipped upstream script uses, verbatim. Paper
#: Figure 8 ("Translation Prompt: Parser, Constrained Decoding and In-Weight
#: Learning") is the same text; the scripts are what pin the system/user split.
PARSE_SYSTEM_PROMPT = (
    "Your task is to translate a natural language instruction to a Bash "
    "command. You will receive an instruction in English and output a Bash "
    "command that can be run in a Linux terminal."
)


@sieval_task(
    name="nl2sh_alfa_0shot_gen_parse",
    display_name="NL2SH-ALFA (0-shot, generative, parsed)",
    description=(
        "Natural language to Bash, execution-graded; upstream's shipped prompt "
        "with markdown stripping."
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
            "PARSE reading: the prompt and extractor upstream ships runnable code "
            "for (NL2SH@b405201a code/example.ipynb and code/model_comparison.ipynb "
            "-- same system prompt, bare user turn, parse_bash, and "
            "submit_command(eval_mode='embed', eval_param=0.75)), so prompt, "
            "extraction and FEH setting are copied rather than inferred. Table 5 "
            "Parse column -- gpt-4o-2024-08-06 0.73, gpt-4o-mini-2024-07-18 0.72, "
            "gpt-4-0613 0.68, gpt-3.5-turbo-0125 0.67, qwen2.5-coder-7b-instruct "
            "0.62, llama-3.1-8b-instruct 0.53, at temperature 0 and seed 123. "
            "parse_bash strips one fence (```bash, then ```, then `) and returns "
            "the reply unchanged otherwise. The sibling nl2sh_alfa_0shot_gen is "
            "the same benchmark without this method; the paper's finding is that "
            "the gap reaches 32%. experimental until a per-sample verdict "
            "comparison against upstream icalfa and an alignment run against those "
            f"published numbers both land. {NL2SH_ALFA_SHARED_NOTES}"
        ),
    ),
)
class NL2SHAlfaZeroShotGenParseTask(NL2SHAlfaSharedZeroShotGenTask):
    SYSTEM_PROMPT = PARSE_SYSTEM_PROMPT
    PARSE_MARKDOWN = True
