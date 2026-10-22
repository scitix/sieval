"""``--checkpoint`` / ``--api-base``: the one model a task-only config lacks.

A benchmark config often carries only ``datasets:`` and ``tasks:`` so it can
run against many models. These flags name that model on the command line. They
become one ``models:`` entry built as a hand-written one would be, so it reaches
``effective_config.yaml`` and a rerun of that file needs no flags.

They apply only when the config has no ``models:``. A config that defines its
own models keeps one source for each, so the flags are refused there.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Any

import typer

CheckpointOption = Annotated[
    str | None,
    typer.Option(
        "--checkpoint",
        help=(
            "Checkpoint to launch and evaluate, for a config with no 'models:' "
            "section. Uses the default infer plan; name the served model with "
            "--model."
        ),
    ),
]

ApiBaseOption = Annotated[
    str | None,
    typer.Option(
        "--api-base",
        help=(
            "Endpoint to evaluate, for a config with no 'models:' section. "
            "Requires --model."
        ),
    ),
]


@dataclass(frozen=True)
class CliModel:
    """The model ``--checkpoint`` or ``--api-base`` names. Exactly one is set."""

    checkpoint: str | None = None
    api_base: str | None = None

    def __post_init__(self) -> None:
        if (self.checkpoint is None) == (self.api_base is None):
            raise ValueError("CliModel takes exactly one of checkpoint and api_base")
        for flag, value in (
            ("--checkpoint", self.checkpoint),
            ("--api-base", self.api_base),
        ):
            if value is not None and not value:
                raise ValueError(f"{flag} must not be empty")

    @classmethod
    def from_options(
        cls, *, checkpoint: str | None, api_base: str | None, model: str | None
    ) -> "CliModel | None":
        """The model the flags name, or ``None`` when neither is given."""
        if checkpoint is not None and api_base is not None:
            raise ValueError(
                "--checkpoint and --api-base name two models; give --checkpoint "
                "to launch and evaluate it, or --api-base to evaluate an "
                "endpoint that is already up"
            )
        if checkpoint is None and api_base is None:
            return None
        if api_base is not None and not model:
            raise ValueError(
                "--api-base needs --model to name the model the endpoint serves"
            )
        return cls(checkpoint=checkpoint, api_base=api_base)

    def entry(self, model: str | None) -> tuple[str, dict[str, Any]]:
        """The ``models:`` key and entry, as a hand-written config spells them.

        *model* (``--model``) is the served name, written as ``name:``.
        Without it a checkpoint is served under its directory name.
        """
        if self.checkpoint is not None:
            entry: dict[str, Any] = {"path": self.checkpoint}
        else:
            entry = {"api_base": self.api_base}
        if model:
            entry["name"] = model
        key = PurePosixPath(model or self.checkpoint or "").name
        if not key:
            raise ValueError("Cannot name a model entry; pass --model")
        return key, entry


def apply_cli_model(
    cfg: dict[str, Any], cli_model: CliModel | None, model: str | None
) -> dict[str, Any]:
    """Add *cli_model* as the only ``models:`` entry of *cfg*, in place.

    Raises:
        ValueError: *cfg* already defines models.
    """
    if cli_model is None:
        return cfg
    existing = cfg.get("models")
    if existing:
        names = ", ".join(sorted(existing)) if isinstance(existing, dict) else "…"
        flag = "--checkpoint" if cli_model.checkpoint is not None else "--api-base"
        raise ValueError(
            f"{flag} supplies the model for a config with no 'models:' "
            f"section, but this config defines {names}. Remove {flag} to run "
            "the models the config defines, or remove its 'models:' section."
        )
    key, entry = cli_model.entry(model)
    cfg["models"] = {key: entry}
    return cfg
