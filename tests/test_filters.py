"""Tests for the filter-pipeline API (h5col.filters)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest
from h5py import h5d, h5p, h5s, h5t

from h5col import ColumnSpec, Table
from h5col.exceptions import FilterError
from h5col.filters import (
    Deflate,
    Filter,
    FilterPipeline,
    Fletcher32,
    Shuffle,
    from_hdf5plugin,
)


def test_builtin_plugin_ids() -> None:
    assert Deflate(5).plugin_id == 1
    assert Deflate(5).cd_values == (5,)
    assert Shuffle().plugin_id == 2
    assert Fletcher32().plugin_id == 3


def test_deflate_level_validation() -> None:
    with pytest.raises(FilterError):
        Deflate(10)
    with pytest.raises(FilterError):
        Deflate(-1)


def test_optional_flag() -> None:
    assert Filter(1, (5,)).flags == 0
    assert Filter(1, (5,), optional=True).flags == 1


def test_from_hdf5plugin() -> None:
    f = from_hdf5plugin(hdf5plugin.Zstd(clevel=5))
    assert f.plugin_id == 32015
    assert f.cd_values == (5,)


def test_pipeline_coerces_hdf5plugin_and_preserves_order() -> None:
    pipe = FilterPipeline([Shuffle(), hdf5plugin.Zstd(clevel=5)])
    assert len(pipe) == 2
    assert [f.plugin_id for f in pipe] == [2, 32015]


def test_pipeline_rejects_bad_entry() -> None:
    with pytest.raises(FilterError):
        FilterPipeline([object()])


def test_to_h5py_kwargs_shuffle_and_compressor() -> None:
    kw = FilterPipeline([Shuffle(), hdf5plugin.Zstd(clevel=5)]).to_h5py_kwargs()
    assert kw == {"shuffle": True, "compression": 32015, "compression_opts": (5,)}


def test_to_h5py_kwargs_deflate_uses_int_level() -> None:
    kw = FilterPipeline([Deflate(6)]).to_h5py_kwargs()
    assert kw == {"compression": "gzip", "compression_opts": 6}


def test_to_h5py_kwargs_rejects_two_compressors() -> None:
    pipe = FilterPipeline([hdf5plugin.Zstd(clevel=5), hdf5plugin.LZ4()])
    with pytest.raises(FilterError):
        pipe.to_h5py_kwargs()


def test_to_h5py_kwargs_fletcher32() -> None:
    kw = FilterPipeline([Fletcher32()]).to_h5py_kwargs()
    assert kw == {"fletcher32": True}


def test_pipeline_apply_roundtrip(h5file: h5py.File) -> None:
    data = np.arange(10_000, dtype="i4")
    pipe = FilterPipeline([Shuffle(), hdf5plugin.Zstd(clevel=5)])

    space = h5s.create_simple(data.shape, (h5s.UNLIMITED,))
    dcpl = h5p.create(h5p.DATASET_CREATE)
    dcpl.set_chunk((1000,))
    pipe.apply(dcpl)
    tid = h5t.py_create(data.dtype, logical=True)
    h5d.create(h5file.id, b"z", tid, space, dcpl=dcpl)
    h5file["z"][...] = data

    # Filters are stored in pipeline order.
    plist = h5file["z"].id.get_create_plist()
    ids = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    assert ids == [2, 32015]

    # Data survives the round trip.
    assert np.array_equal(h5file["z"][...], data)


# The reader script imports h5py and h5col only — it never imports hdf5plugin
# itself — so the read fails with an HDF5 "can't open directory" error unless
# importing h5col is what registered the Zstandard plugin.
_READER = """
import sys

assert "hdf5plugin" not in sys.modules, "the interpreter preloaded hdf5plugin"

import h5py

from h5col import Table

with h5py.File(sys.argv[1], "r") as f:
    values = Table.open(f["t"])["x"].read()
print(int(values.sum()))
"""


def test_plugin_compressed_column_reads_without_importing_hdf5plugin(
    h5path: Path,
) -> None:
    data = np.arange(1000, dtype="i8")
    with h5py.File(h5path, "w") as f:
        table = Table.create(
            f.create_group("t"),
            [
                ColumnSpec(
                    name="x",
                    dtype="i8",
                    filters=FilterPipeline([hdf5plugin.Zstd(clevel=5)]),
                )
            ],
        )
        table.append({"x": data})

    proc = subprocess.run(
        [sys.executable, "-c", _READER, str(h5path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(data.sum())
