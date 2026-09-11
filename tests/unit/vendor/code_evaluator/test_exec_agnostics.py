"""Tests for the code-evaluator's agnostics source -- the digest-pin contract.

`exec_agnostics` is an upstream-bound patch (`vendor/code-evaluator/VENDORED.md`),
whose convention puts tests beside the code so they travel with it. These are
here instead, deliberately: `[tool.pytest] testpaths` covers `tests/` only, so a
test under `vendor/code-evaluator/tests/` is never collected by this suite and
gates nothing. The contract below has already regressed once -- `verifier_command`
enforced the pin on the default path but not on the override -- and was caught by
a manual validation run rather than by CI, which is the gap this file closes. A
re-vendor that moves these upstream should leave an equivalent gate behind.

Only `app.exec_agnostics` is imported, never `app.server` -- the latter needs
fastapi, which is the evaluator service's dependency and not sieval's. Nothing
here spawns a container: `verifier_command` is a pure function of the environment,
and the two `execute_agnostics` cases asserted below both return before the spawn.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import re
import sys
from pathlib import Path

import pytest

# vendor/code-evaluator is a service root, not an installed package — put it on
# sys.path so `app` resolves the way it does inside the service.
_EVALUATOR_DIR = str(Path(__file__).resolve().parents[4] / "vendor" / "code-evaluator")
if _EVALUATOR_DIR not in sys.path:
    sys.path.insert(0, _EVALUATOR_DIR)

from app.exec_agnostics import (  # noqa: E402  # type: ignore[unresolved-import]  # vendor/code-evaluator added to sys.path at runtime
    _COMMAND_ENV_VAR,
    _IMAGE_DIGESTS,
    _REGISTRY,
    execute_agnostics,
    verifier_command,
)

# What `ghcr.io/nuprl/agnostics` actually publishes. Spelled out rather than
# derived from `_IMAGE_DIGESTS`, so a tag deleted from or misspelled in the table
# fails here instead of the table agreeing with itself. Note these are FILE
# EXTENSIONS, not language names -- Julia is `jl`, OCaml `ml`, Fortran `f90`.
_PUBLISHED_TAGS = frozenset({"lua", "r", "python", "jl", "java", "cpp", "ml", "f90"})

# Upstream publishes no image for these, so the table must not carry one.
_UNPINNED = ("rust", "c", "julia", "ocaml", "fortran")


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    """Every test states its own override, or the absence of one."""
    monkeypatch.delenv(_COMMAND_ENV_VAR, raising=False)


# --------------------------------------------------------------------------- #
# The digest table
# --------------------------------------------------------------------------- #
def test_the_table_covers_exactly_upstreams_published_tags() -> None:
    assert set(_IMAGE_DIGESTS) == _PUBLISHED_TAGS


@pytest.mark.parametrize("lang", sorted(_PUBLISHED_TAGS))
def test_every_pinned_entry_is_a_wellformed_digest(lang) -> None:
    # A tag slipping in where a digest belongs is the failure the pin exists to
    # prevent, and it reads as a plausible string everywhere else.
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", _IMAGE_DIGESTS[lang])


# --------------------------------------------------------------------------- #
# verifier_command -- the default (podman) path
# --------------------------------------------------------------------------- #
def test_the_default_path_runs_the_pinned_digest_and_reports_it() -> None:
    argv, image = verifier_command("lua")

    assert image == f"{_REGISTRY}@{_IMAGE_DIGESTS['lua']}"
    # By digest, never `registry:lua` -- a tag moved under a finished
    # leaderboard would otherwise change a score with nothing on disk changing.
    assert argv[-1] == image
    assert ":lua" not in argv[-1]
    assert argv[:3] == ["podman", "run", "--rm"]


@pytest.mark.parametrize("lang", _UNPINNED)
def test_the_default_path_refuses_an_unpinned_language(lang) -> None:
    # Refusing beats floating the mutable tag: an unpinned verifier scores
    # silently, which is the whole reason the table exists.
    with pytest.raises(KeyError):
        verifier_command(lang)


# --------------------------------------------------------------------------- #
# verifier_command -- the override path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", _UNPINNED)
def test_an_image_templated_override_refuses_an_unpinned_language(
    monkeypatch, lang
) -> None:
    # REGRESSION: this branch once substituted "" into the `{image}` slot instead
    # of raising, so the documented `infra:unpinned-lang` refusal never fired
    # under any non-podman runtime -- upstream's own apptainer included, since
    # podman is the one runtime needing no override. A template asking this table
    # for a digest is asking for the pin, and gets the pin's refusal.
    monkeypatch.setenv(_COMMAND_ENV_VAR, "apptainer run --contain {image}")

    with pytest.raises(KeyError):
        verifier_command(lang)


def test_an_image_templated_override_still_reports_the_pinned_digest(
    monkeypatch,
) -> None:
    monkeypatch.setenv(_COMMAND_ENV_VAR, "apptainer run --contain {image}")

    argv, image = verifier_command("jl")

    assert image == f"{_REGISTRY}@{_IMAGE_DIGESTS['jl']}"
    assert argv == ["apptainer", "run", "--contain", image]


@pytest.mark.parametrize("lang", ["lua", *_UNPINNED])
def test_an_override_naming_its_own_image_is_left_alone(monkeypatch, lang) -> None:
    # The override taking responsibility, which is the point of having one: no
    # `{image}`, so the table is never consulted and an unpinned language is not
    # refused. It reports NO image even for a language the table could name --
    # what ran is the override's own, and naming a digest that did not run is
    # worse than naming none.
    monkeypatch.setenv(_COMMAND_ENV_VAR, "apptainer run --contain /opt/{lang}.sif")

    argv, image = verifier_command(lang)

    assert image is None
    assert argv == ["apptainer", "run", "--contain", f"/opt/{lang}.sif"]


# --------------------------------------------------------------------------- #
# execute_agnostics -- the refusals that never reach a container
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_an_unpinned_language_is_refused_with_its_own_code() -> None:
    # The split this module exists to make: the operator reads the cause, not
    # whatever the runtime says about a missing argument.
    passed, msg, _stats, image = await execute_agnostics(
        code="print(1)", inputs=["1"], expect_outputs=["1"], lang="rust", timeout=15.0
    )

    assert passed is False
    assert msg.startswith("infra:unpinned-lang")
    assert image is None


@pytest.mark.anyio
@pytest.mark.parametrize("lang", ["../etc", "lua; rm -rf /", "LUA", "", "x" * 33])
async def test_a_lang_that_cannot_be_a_tag_is_refused_before_any_spawn(lang) -> None:
    # `lang` reaches an argv slot. Not a sandbox -- the command template is
    # trusted and this is not -- but a tag cannot smuggle in a separator.
    passed, msg, _stats, image = await execute_agnostics(
        code="print(1)", inputs=["1"], expect_outputs=["1"], lang=lang, timeout=15.0
    )

    assert passed is False
    assert msg.startswith("infra:bad-lang")
    assert image is None
