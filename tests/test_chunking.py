"""Tests for the default (chunk-cache-scaled) chunk-shape policy."""

from __future__ import annotations

from pathlib import Path

import h5py

from h5col import (
    ColumnSpec,
    FixedString,
    LeafValuesSpec,
    ListColumnSpec,
    StringValuesSpec,
    Table,
    bool_dtype,
)
from h5col._hdf5 import MAX_CHUNK_BYTES, MIN_CHUNK_BYTES, target_chunk_bytes

MiB = 1 << 20


def _clen(col: object) -> int:
    return col.dataset.chunks[0]  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Target derivation
# --------------------------------------------------------------------------- #
def test_target_is_half_the_cache_clamped(h5file: h5py.File) -> None:
    cache = h5file.id.get_access_plist().get_cache()[2]
    expected = min(MAX_CHUNK_BYTES, max(MIN_CHUNK_BYTES, cache // 2))
    assert target_chunk_bytes(h5file) == expected


def test_hdf5_2_default_cache_is_8mib(h5file: h5py.File) -> None:
    # HDF5 >= 2.0 defaults rdcc_nbytes to 8 MiB, so the target is 4 MiB.
    assert h5file.id.get_access_plist().get_cache()[2] == 8 * MiB
    assert target_chunk_bytes(h5file) == 4 * MiB


def test_explicit_override_bypasses_clamp(h5file: h5py.File) -> None:
    # A user-supplied target is honored verbatim, even below the 2 MiB floor.
    assert target_chunk_bytes(h5file, override=512 * 1024) == 512 * 1024


# --------------------------------------------------------------------------- #
# Chunk length across item sizes
# --------------------------------------------------------------------------- #
def test_chunk_len_tracks_itemsize(h5file: h5py.File) -> None:
    target = target_chunk_bytes(h5file)
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="a", dtype="f8"),  # 8 bytes
            ColumnSpec(name="b", dtype="f4"),  # 4 bytes
            ColumnSpec(name="c", dtype="i2"),  # 2 bytes
        ],
    )
    assert _clen(t["a"]) == target // 8
    assert _clen(t["b"]) == target // 4
    assert _clen(t["c"]) == target // 2


def test_one_byte_column_reaches_full_target(h5file: h5py.File) -> None:
    # Regression: the old row cap (1<<20) pinned 1-byte columns to 1 MiB. A
    # boolean column (1 byte) must now span the whole byte target.
    target = target_chunk_bytes(h5file)
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="flags", dtype=bool_dtype())])
    assert _clen(t["flags"]) == target  # e.g. 4,194,304 rows, not 1,048,576


def test_fixed_string_sized_by_byte_width(h5file: h5py.File) -> None:
    target = target_chunk_bytes(h5file)
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="s", dtype=FixedString(10))])
    assert _clen(t["s"]) == target // 10


# --------------------------------------------------------------------------- #
# Clamp bounds via custom chunk caches
# --------------------------------------------------------------------------- #
def test_large_cache_capped(h5path: Path) -> None:
    with h5py.File(h5path, "w", rdcc_nbytes=64 * MiB) as f:
        assert target_chunk_bytes(f) == MAX_CHUNK_BYTES  # 8 MiB cap, not 32 MiB
        t = Table.create(f.create_group("t"), [ColumnSpec(name="a", dtype="f8")])
        assert _clen(t["a"]) == MAX_CHUNK_BYTES // 8


def test_small_cache_floored(h5path: Path) -> None:
    with h5py.File(h5path, "w", rdcc_nbytes=1 * MiB) as f:
        assert target_chunk_bytes(f) == MIN_CHUNK_BYTES  # 2 MiB floor
        t = Table.create(f.create_group("t"), [ColumnSpec(name="a", dtype="f8")])
        assert _clen(t["a"]) == MIN_CHUNK_BYTES // 8


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def test_default_chunk_bytes_override(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="a", dtype="f8")], default_chunk_bytes=1 * MiB)
    assert _clen(t["a"]) == 1 * MiB // 8  # bypasses the cache-scaled default


def test_explicit_column_chunks_wins(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ColumnSpec(name="a", dtype="f8", chunks=1234)], default_chunk_bytes=1 * MiB
    )
    assert _clen(t["a"]) == 1234


# --------------------------------------------------------------------------- #
# List columns
# --------------------------------------------------------------------------- #
def test_list_members_use_policy(h5file: h5py.File) -> None:
    target = target_chunk_bytes(h5file)
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="tags", values=StringValuesSpec())])
    grp = t["tags"].group
    assert grp["OFFSETS"].chunks[0] == target // 8  # uint64
    assert grp["VALUES/OFFSETS"].chunks[0] == target // 8  # uint64
    assert grp["VALUES/CHARS"].chunks[0] == target // 1  # uint8


def test_list_default_chunk_bytes_flows(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))],
        default_chunk_bytes=1 * MiB,
    )
    assert t["r"].group["VALUES"].chunks[0] == 1 * MiB // 4
