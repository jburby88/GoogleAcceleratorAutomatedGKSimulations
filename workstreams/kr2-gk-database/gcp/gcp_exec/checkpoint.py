"""Structural validation of GX NetCDF files (restart checkpoints and diagnostics outputs).

GX writes NetCDF-4 (HDF5) files and recreates ``<stem>.restart.nc`` from scratch with
``NC_CLOBBER | NC_NETCDF4`` every ``nsave`` steps (``src/diagnostics.cu:360``), so a copy
taken while GX is writing can be truncated or internally inconsistent.
Publishing such a copy as "the" checkpoint, or resuming from it, silently corrupts a run
(GCP-03). This module answers one question cheaply, with the standard library only:
*is this file a complete NetCDF/HDF5 file?*

Checks, in order:
1. non-empty regular file;
2. NetCDF classic magic (``CDF\\x01``/``CDF\\x02``/``CDF\\x05``) or the HDF5 signature;
3. HDF5 superblock: the recorded end-of-file address must be defined and not exceed the
   file size (a truncated copy fails this); a v3 superblock with the "open for writing" or
   SWMR status flag set is rejected (a copy of a file still being written);
4. when the ``netCDF4`` module is importable (it is in the gx-runner image), the file must
   also open and expose at least one variable (and ``G`` + ``time`` for ``*.restart.nc``).

HDF5 superblock layout: https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html#Superblock
"""

from __future__ import annotations

import struct
from pathlib import Path

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
NETCDF_CLASSIC_MAGICS = (b"CDF\x01", b"CDF\x02", b"CDF\x05")
# HDF5 allows a user block, so the superblock may start at 0, 512, 1024, 2048, ...
_HDF5_SEARCH_OFFSETS = (0, 512, 1024, 2048, 4096, 8192)


def _read_uint(buf: bytes, offset: int, size: int) -> int | None:
    if size not in (2, 4, 8) or offset + size > len(buf):
        return None
    fmt = {2: "<H", 4: "<I", 8: "<Q"}[size]
    return int(struct.unpack_from(fmt, buf, offset)[0])


def _hdf5_superblock_problem(head: bytes, base: int, file_size: int) -> str | None:
    """Validate the HDF5 superblock that starts at ``base`` (``head`` = file prefix)."""
    sb = head[base:]
    if len(sb) < 12:
        return "truncated HDF5 superblock"
    version = sb[8]
    if version in (0, 1):
        size_of_offsets = sb[13]
        addr_start = 24 if version == 0 else 28
        # base address, free-space info address, end-of-file address, driver info address
        eof_offset = addr_start + 2 * size_of_offsets
        flags = 0
    elif version in (2, 3):
        size_of_offsets = sb[9]
        flags = sb[11]
        # base address, superblock extension address, end-of-file address, root group addr
        addr_start = 12
        eof_offset = addr_start + 2 * size_of_offsets
    else:
        return f"unsupported HDF5 superblock version {version}"

    base_addr = _read_uint(sb, addr_start, size_of_offsets)
    eof_addr = _read_uint(sb, eof_offset, size_of_offsets)
    if base_addr is None or eof_addr is None:
        return "truncated HDF5 superblock"
    undefined = (1 << (8 * size_of_offsets)) - 1
    if eof_addr == undefined or eof_addr == 0:
        return "HDF5 end-of-file address is undefined"
    if version == 3 and flags & 0b101:
        return "HDF5 file is marked as open for writing (copy of a file still being written)"
    expected = (base if base_addr == 0 else base_addr) + eof_addr
    if file_size < expected:
        return f"truncated HDF5 file: {file_size} bytes < end-of-file address {expected}"
    return None


def netcdf_file_problem(path: Path, *, deep: bool = True) -> str | None:
    """Return ``None`` if ``path`` looks like a complete NetCDF file, else a reason string."""
    p = Path(path)
    try:
        if not p.is_file():
            return "missing or not a regular file"
        file_size = p.stat().st_size
        if file_size == 0:
            return "empty file"
        with p.open("rb") as fh:
            head = fh.read(max(_HDF5_SEARCH_OFFSETS) + 128)
    except OSError as exc:
        return f"unreadable: {exc.__class__.__name__}"

    if head[:4] in NETCDF_CLASSIC_MAGICS:
        problem = None
    else:
        base = next(
            (
                off
                for off in _HDF5_SEARCH_OFFSETS
                if head[off : off + len(HDF5_SIGNATURE)] == HDF5_SIGNATURE
            ),
            None,
        )
        if base is None:
            return "not a NetCDF/HDF5 file (bad magic)"
        problem = _hdf5_superblock_problem(head, base, file_size)
    if problem or not deep:
        return problem

    try:
        import netCDF4  # type: ignore[import-not-found]
    except ImportError:
        return None  # structural checks passed; the full open check needs netCDF4
    try:
        with netCDF4.Dataset(str(p), "r") as ds:
            if p.name.endswith(".restart.nc"):
                has_restart_vars = "G" in ds.variables and "time" in ds.variables
                if not has_restart_vars:
                    return "NetCDF restart file is missing required 'G' and 'time' variables"
                return None
            has_vars = bool(ds.variables) or any(bool(g.variables) for g in ds.groups.values())
    except Exception as exc:  # netCDF4 raises OSError/RuntimeError subclasses
        return f"netCDF4 cannot open file: {exc.__class__.__name__}"
    return None if has_vars else "NetCDF file has no variables"


def is_valid_netcdf(path: Path, *, deep: bool = True) -> bool:
    return netcdf_file_problem(path, deep=deep) is None
