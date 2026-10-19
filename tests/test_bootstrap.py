"""Tests for the setup command's safety checks (no network, no downloads)."""
import hashlib
import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from tdq_adapters import bootstrap as bs


# ---------------------------------------------------------------- path containment
def test_safe_target_accepts_inside_paths(tmp_path):
    root = tmp_path / "setup-x"
    assert bs.safe_target(root, "a/b.txt") == (root / "a" / "b.txt").resolve()


@pytest.mark.parametrize("name", ["../setup-x-evil/file.txt",     # sibling folder sharing the prefix
                                  "../outside.txt", "a/../../outside.txt", "/etc/passwd", "C:/Windows/x.txt", ""])
def test_safe_target_rejects_escapes(tmp_path, name):
    with pytest.raises(bs.SetupError):
        bs.safe_target(tmp_path / "setup-x", name)


def _zip(path: Path, entries: list[tuple[str, bytes, int | None]]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            z.writestr(info, data)
    return path


def test_extract_zip_writes_regular_files(tmp_path):
    archive = _zip(tmp_path / "ok.zip", [("pkg/a.txt", b"hello", None), ("pkg/b/c.txt", b"x", None)])
    assert bs.extract_zip(archive, tmp_path / "out") == 2
    assert (tmp_path / "out" / "pkg" / "a.txt").read_bytes() == b"hello"


def test_extract_zip_rejects_traversal_before_writing_anything(tmp_path):
    archive = _zip(tmp_path / "bad.zip", [("fine.txt", b"1", None), ("../out-evil/x.txt", b"2", None)])
    with pytest.raises(bs.SetupError, match="outside"):
        bs.extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "fine.txt").exists() and not (tmp_path / "out-evil").exists()


def test_extract_zip_rejects_symlink_entries(tmp_path):
    archive = _zip(tmp_path / "link.zip", [("link", b"/etc/passwd", stat.S_IFLNK | 0o777)])
    with pytest.raises(bs.SetupError, match="not a regular file"):
        bs.extract_zip(archive, tmp_path / "out")


def _tar(path: Path, members: list[tarfile.TarInfo], data: dict[str, bytes] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as t:
        for m in members:
            payload = (data or {}).get(m.name)
            if payload is not None:
                m.size = len(payload)
                t.addfile(m, io.BytesIO(payload))
            else:
                t.addfile(m)
    return path


def test_extract_tar_rejects_links_and_traversal(tmp_path):
    link = tarfile.TarInfo("inside_link")
    link.type, link.linkname = tarfile.SYMTYPE, "target.txt"
    with pytest.raises(bs.SetupError, match="not a regular file"):
        bs.extract_tar(_tar(tmp_path / "l.tar.gz", [link]), tmp_path / "out1")
    evil = tarfile.TarInfo("../evil.txt")
    with pytest.raises(Exception):
        bs.extract_tar(_tar(tmp_path / "e.tar.gz", [evil], {"../evil.txt": b"x"}), tmp_path / "out2")
    assert not (tmp_path / "evil.txt").exists()


# ---------------------------------------------------------------- archive and extraction checks
def _spec(archive: Path, into: Path, kind: str) -> dict:
    data = archive.read_bytes()
    return {"md5": hashlib.md5(data).hexdigest(), "bytes": len(data), "extract": kind, "into": into}


def test_corrupted_archive_is_not_accepted(tmp_path):
    archive = _zip(tmp_path / "a.zip", [("x.txt", b"abc", None)])
    spec = _spec(archive, tmp_path / "out", "zip")
    assert bs.archive_ok(archive, spec)
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                     # same size, different content
    archive.write_bytes(bytes(raw))
    assert not bs.archive_ok(archive, spec)
    archive.write_bytes(bytes(raw) + b"x")           # different size
    assert not bs.archive_ok(archive, spec)


def test_extraction_problems_finds_missing_and_changed_files(tmp_path):
    archive = _zip(tmp_path / "a.zip", [("d/x.txt", b"abc", None), ("d/y.txt", b"defg", None)])
    out = tmp_path / "out"
    spec = _spec(archive, out, "zip")
    bs.extract_zip(archive, out)
    assert bs.extraction_problems(archive, spec) == []
    (out / "d" / "x.txt").write_bytes(b"changed!")
    (out / "d" / "y.txt").unlink()
    assert sorted(bs.extraction_problems(archive, spec)) == ["missing d/y.txt", "size differs: d/x.txt"]


# ---------------------------------------------------------------- shipped authors' code
def test_check_repo_ignores_ds_store_and_detects_changes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_bytes(b"print(1)\n")
    expected = {"src/a.py": hashlib.sha256(b"print(1)\n").hexdigest(), "src/.DS_Store": "whatever"}
    assert bs.check_repo(repo, expected) == []                      # .DS_Store missing locally: fine
    (repo / "src" / "a.py").write_bytes(b"print(2)\n")
    (repo / "extra.txt").write_text("x")
    assert bs.check_repo(repo, expected) == ["unexpected extra.txt", "changed src/a.py"]
    assert bs.check_repo(tmp_path / "nope", expected) == ["missing"]
