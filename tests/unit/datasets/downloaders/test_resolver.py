from unittest.mock import MagicMock, patch

import pytest

from sieval.datasets.downloaders import resolver
from sieval.datasets.downloaders.resolver import (
    _extras_requirements_for,
    extras_unsatisfied,
)

# ---------------------------------------------------------------------------
# extras_unsatisfied — authoritative check (replaces static probe table)
# ---------------------------------------------------------------------------


def _stub_distribution(requires: list[str]) -> MagicMock:
    """Build a Distribution-like object that returns *requires* verbatim."""
    dist = MagicMock()
    dist.requires = requires
    return dist


@pytest.fixture(autouse=True)
def _clear_extras_cache():
    """extras_requirements() is lru_cached; reset per test to isolate."""
    resolver._extras_requirements.cache_clear()
    yield
    resolver._extras_requirements.cache_clear()


def test_extras_requirements_indexes_by_marker_group():
    """A requires-list containing markers for two groups fans out correctly."""
    dist = _stub_distribution(
        [
            'scipy>=1.16.3; extra == "drop"',
            'numpy<=2.2; extra == "drop"',
            'rdkit>=2024.9.1; extra == "drugassist"',
            "pytest>=9.0",  # no extra marker — not an extras dep
        ]
    )
    out = _extras_requirements_for(dist)
    assert set(out) == {"drop", "drugassist"}
    assert sorted(r.name for r in out["drop"]) == ["numpy", "scipy"]
    assert [r.name for r in out["drugassist"]] == ["rdkit"]


def test_extras_unsatisfied_empty_when_all_versions_ok():
    """A transitively-installed-but-satisfying dep is not reported."""
    with (
        patch.object(
            resolver.metadata,
            "distribution",
            return_value=_stub_distribution(['scipy>=1.16.3; extra == "drop"']),
        ),
        patch.object(resolver.metadata, "version", return_value="1.20.0"),
    ):
        assert extras_unsatisfied("drop") == []


def test_extras_unsatisfied_flags_missing_package():
    def fake_version(name):
        raise resolver.metadata.PackageNotFoundError(name)

    with (
        patch.object(
            resolver.metadata,
            "distribution",
            return_value=_stub_distribution(['math-verify>=0.8.0; extra == "math"']),
        ),
        patch.object(resolver.metadata, "version", side_effect=fake_version),
    ):
        unmet = extras_unsatisfied("math")
    assert len(unmet) == 1
    assert "math-verify" in unmet[0]


def test_extras_unsatisfied_flags_wrong_version():
    """Regression: transitively installed `scipy` at a version below the pin
    must still register as unsatisfied — this is the scenario the old
    ``find_spec`` probe couldn't distinguish from a real extras install."""
    with (
        patch.object(
            resolver.metadata,
            "distribution",
            return_value=_stub_distribution(['scipy>=1.16.3; extra == "drop"']),
        ),
        patch.object(resolver.metadata, "version", return_value="1.14.0"),
    ):
        unmet = extras_unsatisfied("drop")
    assert len(unmet) == 1
    assert "installed: 1.14.0" in unmet[0]


def test_extras_unsatisfied_unknown_group_returns_empty():
    with patch.object(
        resolver.metadata,
        "distribution",
        return_value=_stub_distribution([]),
    ):
        assert extras_unsatisfied("nonexistent_group") == []


def test_extras_unsatisfied_handles_missing_distribution():
    """Running from a non-installed checkout — no authoritative info available."""
    with patch.object(
        resolver.metadata,
        "distribution",
        side_effect=resolver.metadata.PackageNotFoundError("sieval"),
    ):
        assert extras_unsatisfied("drop") == []
