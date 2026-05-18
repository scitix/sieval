from pathlib import Path

from sieval.core.utils.paths import resolve_data_dir


def test_cli_flag_wins(monkeypatch):
    monkeypatch.setenv("SIEVAL_DATA_DIR", "/env/path")
    assert resolve_data_dir(cli_flag="/cli/path") == Path("/cli/path")


def test_env_var_second(monkeypatch):
    monkeypatch.setenv("SIEVAL_DATA_DIR", "/env/path")
    assert resolve_data_dir(cli_flag=None) == Path("/env/path")


def test_default_fallback(monkeypatch):
    monkeypatch.delenv("SIEVAL_DATA_DIR", raising=False)
    assert resolve_data_dir(cli_flag=None) == Path.home() / ".sieval" / "data"
