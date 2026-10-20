"""Processes a task calls besides the model, and their provenance for ``meta.json``.

A task names them in :attr:`Task.services <sieval.core.tasks.task.Task.services>`.
Each is reached through one environment variable (:data:`SERVICE_ENDPOINT_ENV`):
unset, the CLI starts SiEval's vendored copy and sets it; set, it is used as given.

The CLI records what it started in :data:`VENDORED_SERVICES_ENV`. A declared
service listed there, whose endpoint variable still holds the endpoint started,
is ``vendored``; any other is ``external``, and its URL is never recorded.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import os
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Literal, NotRequired, TypedDict

#: The code-execution service vendored at ``vendor/code-evaluator``.
CODE_EVAL = "code-eval"

#: Environment variable each service's endpoint is read from.
SERVICE_ENDPOINT_ENV: Mapping[str, str] = MappingProxyType(
    {CODE_EVAL: "SIEVAL_CODE_EVAL_API"}
)

#: Services this SiEval process started from its vendored sources, by name.
VENDORED_SERVICES_ENV = "SIEVAL_VENDORED_SERVICES"


class ServiceRuntime(TypedDict):
    """The interpreter a vendored service ran under, which graded programs share.

    ``packages`` maps each package graded programs import to its version,
    ``None`` when absent. Toolchains on ``PATH`` (``g++``, ``node``) are not
    recorded.
    """

    python: str
    platform: str
    packages: dict[str, str | None]


class ServiceIdentity(TypedDict):
    """One service a task calls while grading.

    ``commit`` is the upstream commit of the vendored copy, ``None`` when it
    does not say; ``digest`` covers the sources it ran, local patches included.
    """

    name: str
    source: Literal["vendored", "external"]
    commit: NotRequired[str | None]
    digest: NotRequired[str]
    runtime: NotRequired[ServiceRuntime]


def check_service_names(names: Iterable[str], owner: str) -> None:
    """Raise ``ValueError`` for a service name SiEval does not know."""
    unknown = sorted(set(names) - set(SERVICE_ENDPOINT_ENV))
    if unknown:
        raise ValueError(
            f"{owner} declares unknown service(s) {unknown}; "
            f"known: {sorted(SERVICE_ENDPOINT_ENV)}"
        )


def vendored_services(environ: Mapping[str, str] | None = None) -> dict[str, dict]:
    """Read :data:`VENDORED_SERVICES_ENV`; empty when unset or malformed."""
    raw = (os.environ if environ is None else environ).get(VENDORED_SERVICES_ENV, "")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: v for k, v in parsed.items() if isinstance(v, dict)}


def service_identities(
    names: Iterable[str], environ: Mapping[str, str] | None = None
) -> list[ServiceIdentity]:
    """Describe each of *names* as vendored (commit, digest, runtime) or external."""
    env = os.environ if environ is None else environ
    started = vendored_services(env)
    out: list[ServiceIdentity] = []
    for name in sorted(set(names)):
        entry = started.get(name)
        if (
            entry is None
            or not isinstance(entry.get("digest"), str)
            or entry.get("endpoint") != env.get(SERVICE_ENDPOINT_ENV.get(name, ""))
        ):
            out.append({"name": name, "source": "external"})
            continue
        commit = entry.get("commit")
        identity: ServiceIdentity = {
            "name": name,
            "source": "vendored",
            "commit": commit if isinstance(commit, str) else None,
            "digest": entry["digest"],
        }
        runtime = entry.get("runtime")
        if isinstance(runtime, dict):
            identity["runtime"] = runtime
        out.append(identity)
    return out
