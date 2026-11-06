"""Crash-safe filesystem helpers shared by the TACC and GCP executors (stdlib only).

Every durable write in `hpc_core` / `tacc_exec` goes through `atomic_write_bytes`:

1. write to a unique temp file in the *same* directory (`tempfile.mkstemp`), so concurrent
   writers never share a temp name and `os.replace` stays a same-filesystem rename;
2. `flush` + `os.fsync` the temp file, so the data blocks are on stable storage before the
   rename makes them visible (POSIX does not order data vs. metadata otherwise; this matters
   on Lustre `$WORK`/`$SCRATCH`, NFS `$HOME` and local ext4/xfs alike);
3. `os.replace` onto the target (atomic for readers: old or new content, never a mix);
4. best-effort `fsync` of the parent directory, so the rename itself survives a crash.

A failure at any step removes the temp file and re-raises; the target is left untouched.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# One identifier grammar for run ids, campaign ids and equilibrium ids that become path
# components (queue folders, dataset directories, GCS object prefixes). It cannot contain a
# path separator and cannot start with "." or "-", so "..", ".hidden" and "-flag" are rejected.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_DIR_FSYNC_IGNORED_ERRNOS = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.EACCES})


def is_safe_id(value: object) -> bool:
    """Return True if `value` is a string usable as a single, non-traversing path component."""
    return isinstance(value, str) and SAFE_ID_RE.fullmatch(value) is not None


def fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory so a completed rename/create inside it is durable."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        # Some filesystems (and some network mounts) refuse fsync on a directory fd.
        if exc.errno not in _DIR_FSYNC_IGNORED_ERRNOS:
            raise
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """Durably and atomically replace `path` with `data` (temp file + fsync + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # Permission bits are best effort: GCS FUSE has no POSIX modes (they come from
        # --file-mode) and may refuse chmod; that must not fail a durable write.
        with contextlib.suppress(OSError):
            if mode is not None:
                os.chmod(tmp_name, mode)
            elif target.exists():
                # Preserve the permission bits of the file being replaced (mkstemp uses 0o600).
                os.chmod(tmp_name, target.stat().st_mode & 0o7777)
            else:
                umask = os.umask(0)
                os.umask(umask)
                os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    fsync_dir(target.parent)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Durably and atomically replace `path` with `text`."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: Path, obj: Any, *, indent: int | None = 2, sort_keys: bool = True
) -> None:
    """Durably and atomically write strict JSON (NaN/Infinity are rejected, not emitted)."""
    rendered = json.dumps(obj, indent=indent, sort_keys=sort_keys, allow_nan=False) + "\n"
    atomic_write_text(path, rendered)


def append_line_durable(path: Path, line: str, *, encoding: str = "utf-8") -> None:
    """Append one newline-terminated record and fsync it (for audit/outbox JSONL files)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = line if line.endswith("\n") else line + "\n"
    with target.open("a", encoding=encoding) as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, streaming in `chunk_size` blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
