"""
Tests for the one atomic write every result-directory writer uses.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import anyio
import pytest

from sieval.core.utils import atomic
from sieval.core.utils.atomic import aatomic_write, atomic_write


def test_a_new_file_gets_the_umask_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "_UMASK", 0o022)
    atomic_write(tmp_path / "report.json", b"{}")
    assert (tmp_path / "report.json").stat().st_mode & 0o777 == 0o644


def test_a_replaced_file_keeps_its_mode(tmp_path):
    target = tmp_path / "meta.json"
    target.write_bytes(b"old")
    target.chmod(0o640)
    atomic_write(target, b"new")
    assert target.read_bytes() == b"new"
    assert target.stat().st_mode & 0o777 == 0o640


def test_a_failed_write_leaves_the_target_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "meta.json"
    target.write_bytes(b"old")

    def _fail(*_args):
        raise OSError("EIO")

    monkeypatch.setattr(atomic.os, "replace", _fail)
    with pytest.raises(OSError, match="EIO"):
        atomic_write(target, b"new")
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json"]


@pytest.mark.anyio
async def test_concurrent_writers_never_share_a_temp_file(tmp_path):
    async with anyio.create_task_group() as tg:
        for i in range(8):
            tg.start_soon(aatomic_write, tmp_path / "p.json", b"%d" % i)
    assert (tmp_path / "p.json").read_bytes() in {b"%d" % i for i in range(8)}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.json"]
