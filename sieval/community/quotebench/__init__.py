"""QuoteBench: vendored task definitions and contract prompts.

Upstream: https://github.com/LeonardNJU/quoteBench/tree/693325a671e65f889e5cd9d83965db9cc3b26dc2
@ ``693325a6``, Apache-2.0.

Only the halves sieval needs are vendored: :mod:`core` (contract prompts, the
``Task`` type, the literal-text markers) and :mod:`scenarios` (the 56 frozen
tasks), plus :mod:`shellesc`, which ``scenarios`` imports to construct oracles.
Execution and final-state checking are the code-evaluator's half -- no sieval
task executes model output.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

#: Upstream's own, carried over verbatim from the pinned ``__init__``. It is not
#: in the digest -- that covers the three modules whose content decides a
#: verdict -- so it pins nothing; it is here because a partial vendoring that
#: drops a version string makes the copy answer differently from the package it
#: claims to be.
__version__ = "0.1.0"
