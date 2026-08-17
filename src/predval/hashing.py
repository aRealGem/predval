"""Deterministic content hashing for provenance.

A validation report asserts something about specific inputs. These digests are what let a reader
confirm, months later, that the inputs were the ones named. The hash is therefore a pure
function of file bytes -- no paths, no timestamps, no environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: blake2b digest size in bytes. 32 gives a 64-character hex string.
DIGEST_SIZE = 32

#: Read size. Bounded so hashing a multi-gigabyte parquet does not fault the machine.
_CHUNK_BYTES = 1024 * 1024


def hash_file(path: str | Path) -> str:
    """Return the blake2b hex digest of a file's contents.

    Deterministic: the same bytes always produce the same digest, in this process or any other.
    """
    p = Path(path)
    h = hashlib.blake2b(digest_size=DIGEST_SIZE)
    with p.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Return the blake2b hex digest of a bytes object.

    Uses the same parameters as :func:`hash_file`, so hashing a file and hashing its contents
    read into memory agree.
    """
    return hashlib.blake2b(data, digest_size=DIGEST_SIZE).hexdigest()


def hash_inputs(**paths: str | Path | None) -> dict[str, str]:
    """Hash a set of named input files, skipping any whose path is None.

    >>> hash_inputs(cohort="cohort.yaml", predictions="predictions.parquet")  # doctest: +SKIP
    {'cohort': '9f86d081...', 'predictions': 'a3f5b2c1...'}
    """
    return {name: hash_file(p) for name, p in paths.items() if p is not None}
