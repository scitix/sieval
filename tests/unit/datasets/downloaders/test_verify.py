from sieval.core.datasets.meta import Category, DatasetMeta, Level1Category
from sieval.datasets.downloaders.verify import (
    ChecksumMismatch,
    compute_sha256,
    verify_checksums,
)


def _meta(name: str, checksums: tuple[tuple[str, str], ...]) -> DatasetMeta:
    return DatasetMeta(
        name=name,
        display_name=name,
        description="d",
        source=("url:https://example.com/f.bin",),
        categories=(Category(Level1Category.CODE, "CodeGeneration"),),
        checksums=checksums,
    )


def test_compute_sha256_format(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    digest = compute_sha256(p)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert digest == (
        "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_verify_match_returns_empty(tmp_path):
    (tmp_path / "ds").mkdir()
    f = tmp_path / "ds" / "f.bin"
    f.write_bytes(b"hello")
    meta = _meta("ds", (("f.bin", compute_sha256(f)),))
    assert verify_checksums(meta, tmp_path) == []


def test_verify_corruption_detected(tmp_path):
    (tmp_path / "ds").mkdir()
    f = tmp_path / "ds" / "f.bin"
    f.write_bytes(b"hello")
    good = compute_sha256(f)
    f.write_bytes(b"tampered")
    out = verify_checksums(_meta("ds", (("f.bin", good),)), tmp_path)
    assert len(out) == 1 and out[0].basename == "f.bin"
    assert out[0].actual is not None and out[0].actual != good
    assert out[0].expected == good


def test_verify_missing_file(tmp_path):
    out = verify_checksums(_meta("ds", (("f.bin", "sha256:" + "a" * 64),)), tmp_path)
    assert out == [ChecksumMismatch("f.bin", "sha256:" + "a" * 64, None)]


def test_verify_empty_checksums_is_noop(tmp_path):
    assert verify_checksums(_meta("ds", ()), tmp_path) == []
