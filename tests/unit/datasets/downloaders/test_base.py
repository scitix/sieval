from pathlib import Path

import pytest

from sieval.datasets.downloaders import base


class DummyHandler:
    scheme = "dummy"

    def download(self, source, dest_root, dataset_name, force):
        return Path("/tmp/dummy")

    def is_downloaded(self, source, dest_root, dataset_name):
        return False


def _isolate_registry(monkeypatch):
    monkeypatch.setattr(base, "_HANDLERS", {})
    monkeypatch.setattr(base, "_builtin_registered", True)


def test_register_and_resolve(monkeypatch):
    _isolate_registry(monkeypatch)
    h = DummyHandler()
    base.register_handler(h)
    assert base.resolve("dummy:something") is h


def test_resolve_unknown_scheme(monkeypatch):
    _isolate_registry(monkeypatch)
    with pytest.raises(NotImplementedError, match="Unknown source scheme"):
        base.resolve("weird:x")


def test_register_duplicate_rejected(monkeypatch):
    _isolate_registry(monkeypatch)
    base.register_handler(DummyHandler())
    with pytest.raises(ValueError, match="already registered"):
        base.register_handler(DummyHandler())


def test_builtin_registration_survives_preoccupied_slot():
    """If a pre-existing handler occupies the `hf` slot, `_ensure_builtin_handlers`
    silently skips HFHandler (via the ``scheme not in _HANDLERS`` guard) and
    still goes on to register URLHandler. The pre-registered Impostor must
    remain reachable, and URL resolution must work."""
    import sieval.datasets.downloaders.base as base_mod

    # Snapshot + reset module state so we can exercise the one-shot init path
    saved_handlers = base_mod._HANDLERS.copy()
    saved_flag = base_mod._builtin_registered
    base_mod._HANDLERS.clear()
    base_mod._builtin_registered = False

    try:
        # Pre-register an impostor `hf` handler so HFHandler's slot is taken
        # and the builtin init path short-circuits for `hf`.
        class Impostor:
            scheme = "hf"

            def download(self, *a, **kw):
                raise NotImplementedError

            def is_downloaded(self, *a, **kw):
                return False

        base_mod.register_handler(Impostor())

        # Trigger builtin registration — HFHandler skip, URLHandler registers
        handler = base_mod.resolve("hf:anything")
        assert handler is not None  # resolves to the Impostor (already registered)

        # URL handler must be registered despite the `hf` slot being preoccupied
        handler = base_mod.resolve("url:https://example.com/x.csv")
        assert handler.scheme == "url"
    finally:
        base_mod._HANDLERS.clear()
        base_mod._HANDLERS.update(saved_handlers)
        base_mod._builtin_registered = saved_flag
