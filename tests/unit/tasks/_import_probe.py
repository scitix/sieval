"""Fresh-interpreter probe behind the task import-discipline family test.

Registering a task must not import its optional grading dependency, and proving
that needs an interpreter where the dependency was never imported. A separate
interpreter per task costs ~2.4s, nearly all of it spent importing the shared
`sieval.tasks` base — so this pays that once and measures every task against it.

Tasks stay independent because after each one this drops every module that task
added, *except* the `sieval.tasks` / `sieval.datasets` / `sieval.core` subtrees:
their module bodies register into process-global registries, and re-executing
one raises a duplicate-name error. What that leaves droppable — third-party
packages, and the `sieval.community` graders, which register nothing — is
exactly what a forbidden name can be, so no task inherits another's imports.

Reads `{module: [forbidden, ...]}` as JSON on stdin, importing in that order.
Writes `{module: {"present": [...]}}` — or `{module: {"error": "..."}}` if the
import itself raised — as JSON on stdout.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import importlib
import json
import sys

# Subtrees to leave loaded: a second execution of any module body under them
# would re-run `@sieval_task` / `@sieval_dataset` against a registry that
# already holds the name.
_STICKY = ("sieval.tasks", "sieval.datasets", "sieval.core")


def _drop_addenda(baseline: frozenset[str]) -> None:
    """Unload every module added since *baseline* that is safe to re-import."""
    for name in sys.modules.keys() - baseline:
        if not name.startswith(_STICKY):
            del sys.modules[name]


def main() -> int:
    manifest: dict[str, list[str]] = json.load(sys.stdin)

    # Some optional dependencies print on import, which would corrupt the reply.
    # Hand the protocol a handle nothing else holds and send stray output away.
    reply_to, sys.stdout = sys.stdout, sys.stderr

    importlib.import_module("sieval.tasks")
    baseline = frozenset(sys.modules)

    results: dict[str, dict] = {}
    for module, forbidden in manifest.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            results[module] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            results[module] = {
                "present": [name for name in forbidden if name in sys.modules]
            }
        _drop_addenda(baseline)

    json.dump(results, reply_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
