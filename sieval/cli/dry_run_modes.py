"""The ``--mode`` of ``sieval run --dry-run``: how planning does its heavy setup.

``light`` (the default) reads datasets in memory and checks vendored services
without starting them; ``full`` does both as the run does. A new mode is one
more :data:`DRY_RUN_MODES` entry. The report says what each item verified, so a
caller never infers it from the mode's name.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from sieval.core.utils.hf_uncached import BuilderRead

#: ``memory``: no Arrow cache (:func:`~sieval.core.utils.hf_uncached.uncached_reads`);
#: ``cache``: as the run does.
type DatasetReads = Literal["memory", "cache"]

#: ``check``: :func:`~sieval.cli.services.checked_services_for`; ``start``: as
#: the run does.
type ServiceStarts = Literal["check", "start"]


@dataclass(frozen=True, slots=True)
class DryRunMode:
    """One ``--mode``."""

    name: str
    datasets: DatasetReads
    services: ServiceStarts


#: What the run itself does.
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
    """One vendored service in the report; *setup_unreached* names the tasks
    whose ``setup()`` could not connect to it because it was not started."""

    started: bool
    verified: list[str]
    unverified: list[str]
    setup_unreached: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DryRunRecord:
    """What planning did per dataset and service, for the dry-run report."""

    datasets: dict[str, list[BuilderRead]] = field(default_factory=dict)
    services: dict[str, ServiceCheck] = field(default_factory=dict)

    def merge(self, other: "DryRunRecord") -> None:
        """Add *other*, one serving group's; a service several groups check
        keeps every group's unreached tasks."""
        self.datasets |= other.datasets
        for name, check in other.services.items():
            mine = self.services.get(name)
            if mine is None:
                self.services[name] = replace(
                    check, setup_unreached=list(check.setup_unreached)
                )
            else:
                mine.setup_unreached = sorted(
                    {*mine.setup_unreached, *check.setup_unreached}
                )
