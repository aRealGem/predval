"""Hashing must be deterministic, or provenance claims mean nothing."""

from __future__ import annotations

from pathlib import Path

from predval import hash_bytes, hash_file, hash_inputs
from predval.hashing import _CHUNK_BYTES, DIGEST_SIZE


def test_hashing_is_deterministic_across_calls(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"predval" * 1000)
    assert hash_file(p) == hash_file(p)


def test_digest_shape(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"x")
    digest = hash_file(p)
    assert len(digest) == DIGEST_SIZE * 2
    assert set(digest) <= set("0123456789abcdef")


def test_hash_depends_only_on_content_not_filename(tmp_path: Path) -> None:
    (tmp_path / "one.bin").write_bytes(b"same bytes")
    (tmp_path / "two.bin").write_bytes(b"same bytes")
    assert hash_file(tmp_path / "one.bin") == hash_file(tmp_path / "two.bin")


def test_differing_content_differs(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"alpha")
    (tmp_path / "b").write_bytes(b"beta")
    assert hash_file(tmp_path / "a") != hash_file(tmp_path / "b")


def test_chunking_does_not_change_the_digest(tmp_path: Path) -> None:
    """A file larger than one read chunk must hash the same as its bytes in memory.

    This is the test that catches a chunked-read bug: a naive implementation that resets the
    hasher per chunk still produces a plausible-looking digest.
    """
    payload = bytes(range(256)) * (_CHUNK_BYTES // 256 + 7)
    assert len(payload) > _CHUNK_BYTES
    p = tmp_path / "big.bin"
    p.write_bytes(payload)
    assert hash_file(p) == hash_bytes(payload)


def test_empty_file_hashes(tmp_path: Path) -> None:
    p = tmp_path / "empty"
    p.write_bytes(b"")
    assert hash_file(p) == hash_bytes(b"")


def test_hash_inputs_skips_none(tmp_path: Path) -> None:
    p = tmp_path / "a"
    p.write_bytes(b"a")
    out = hash_inputs(cohort=p, predictions=None)
    assert set(out) == {"cohort"}
    assert out["cohort"] == hash_file(p)
