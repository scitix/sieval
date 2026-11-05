"""How ``sieval run --dry-run`` sets a session up, by ``--mode`` name.

A mode decides, for each heavy thing planning does as the run does, whether the
dry-run does it or checks it without doing it. ``light`` (the default) reads
datasets in memory and checks vendored services without starting them; ``full``
does both as the run does. A new mode is one more :data:`DRY_RUN_MODES` entry;
the CLI flag and the report's shape do not change.

The report names what each mode verified per item (:class:`DryRunRecord`), so a
caller reads ``verified`` / ``unverified`` instead of inferring them from the
mode's name.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from sieval.core.utils.hf_uncached import BuilderRead

#: ``memory``: read each dataset in memory, writing no Arrow cache
#: (:func:`~sieval.core.utils.hf_uncached.uncached_reads`). ``cache``: load it
#: through the Arrow cache, as the run does.
type DatasetReads = Literal["memory", "cache"]

#: ``check``: check each vendored service the run would start can start, without
#: starting it (:func:`~sieval.cli.services.checked_services_for`). ``start``:
#: start it, as the run does.
type ServiceStarts = Literal["check", "start"]


@dataclass(frozen=True, slots=True)
class DryRunMode:
    """One ``--mode``: how planning treats each heavy thing it does."""

    name: str
    datasets: DatasetReads
    services: ServiceStarts


#: What the run itself does; a session outside a dry-run behaves so.
RUN = DryRunMode("full", datasets="cache", services="start")

DRY_RUN_MODES: Mapping[str, DryRunMode] = MappingProxyType(
    {
        "light": DryRunMode("light", datasets="memory", services="check"),
        "full": RUN,
    }
)
DEFAULT_DRY_RUN_MODE = "light"


def dry_run_mode(name: str) -> DryRunMode:
    """The mode called *name*; ``ValueError`` naming the known ones otherwise."""
    try:
        return DRY_RUN_MODES[name]
    except KeyError:
        raise ValueError(
            f"unknown mode {name!r}; known: {', '.join(DRY_RUN_MODES)}"
        ) from None


@dataclass(slots=True)
class ServiceCheck:
    """What planning did with one vendored service the run would start.

    *started* is whether it ran; *verified* names the checks that passed, in
    order, and *unverified* those the run makes that this planning did not.
    *setup_unreached* lists the tasks whose ``setup()`` handshake could not
    reach it because it was not started.
    """

    started: bool
    verified: list[str]
    unverified: list[str]
    setup_unreached: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DryRunRecord:
    """What a planning session did, item by item, for the dry-run report."""

    datasets: dict[str, list[BuilderRead]] = field(default_factory=dict)
    services: dict[str, ServiceCheck] = field(default_factory=dict)
