"""Tests for the Table + Column write/read core (h5col.table, h5col.column)."""

from __future__ import annotations

import h5py
import hdf5plugin
import numpy as np
import pytest

from h5col import (
    Column,
    ColumnSpec,
    Deflate,
    Filter,
    FilterPipeline,
    FixedString,
    Shuffle,
    Table,
    bool_dtype,
)
from h5col._hdf5 import write_utf8_attr
from h5col.exceptions import (
    ConformanceError,
    FillValueError,
    FilterError,
    OversizedStringError,
    ReservedNameError,
    SchemaError,
)


def test_create_empty_table(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    assert Table.is_table_group(g)
    assert t.nrows == 0
    assert t.version == "1.0"
    assert t.column_names == ["x"]


def test_roundtrip_numeric_string_bool(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    arrays = {
        "id": np.arange(3, dtype="i8"),
        "name": ["ab", "cde", "f"],
        "flag": np.array([True, False, True]),
        "val": np.array([1.5, 2.5, 3.5], dtype="f4"),
    }
    t = Table.from_arrays(g, arrays, index_columns=["id"])
    assert t.nrows == 3
    out = t.read()
    assert list(out["id"]) == [0, 1, 2]
    assert list(out["name"]) == ["ab", "cde", "f"]
    assert list(out["flag"]) == [True, False, True]
    assert np.allclose(out["val"], [1.5, 2.5, 3.5])
    assert t.index_columns == ["id"]

    # Reopen from a fresh handle.
    t2 = Table.open(g)
    assert list(t2.read()["name"]) == ["ab", "cde", "f"]
    assert t2.column_names == ["id", "name", "flag", "val"]


def test_root_group_as_table(h5file: h5py.File) -> None:
    t = Table.from_arrays(h5file, {"x": np.arange(3, dtype="i4")})
    assert t.nrows == 3
    assert list(t.read()["x"]) == [0, 1, 2]


def test_append_batches_commit_nrows_last(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    t.append({"x": [1, 2, 3]})
    t.append({"x": [4, 5]})
    assert t.nrows == 5
    assert list(t.read()["x"]) == [1, 2, 3, 4, 5]


def test_string_oversize_raises_and_leaves_table_untouched(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="s", dtype=FixedString(3))])
    with pytest.raises(OversizedStringError):
        t.append({"s": ["abcd"]})
    assert t.nrows == 0  # not committed


def test_missing_via_absent_column(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ColumnSpec(name="a", dtype="i4"), ColumnSpec(name="b", dtype="i4")]
    )
    t.append({"a": [1, 2, 3]})  # b absent -> fill (missing)
    assert list(t["b"].is_missing()) == [True, True, True]
    assert list(t["a"].is_missing()) == [False, False, False]
    # b reads back as its fill value, and masked as missing.
    assert list(t["b"].read(masked=False)) == [-2147483647] * 3
    assert t["b"].read().tolist() == [None] * 3


def test_boolean_no_fill_and_required_in_append(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="a", dtype="i4"),
            ColumnSpec(name="flag", dtype=bool_dtype()),
        ],
    )
    assert t["flag"].fill_value is None
    with pytest.raises(SchemaError):
        t.append({"a": [1, 2]})  # flag can't be filled
    assert t.nrows == 0


def test_boolean_fill_value_rejected_at_create(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(SchemaError):
        Table.create(g, [ColumnSpec(name="flag", dtype=bool_dtype(), fill_value=0)])


def test_fill_inside_valid_range_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(FillValueError):
        Table.create(
            g,
            [ColumnSpec(name="x", dtype="i4", fill_value=5, valid_min=0, valid_max=10)],
        )


def test_open_non_table_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("plain")
    g.create_dataset("d", data=[1, 2, 3])
    with pytest.raises(ConformanceError):
        Table.open(g)


def test_create_on_existing_table_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    with pytest.raises(SchemaError):
        Table.create(g, [ColumnSpec(name="y", dtype="i4")])


def test_reserved_column_name_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(ReservedNameError):
        Table.create(g, [ColumnSpec(name="CATEGORIES", dtype="i4")])


def test_column_order_controls_logical_order(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [ColumnSpec(name="b", dtype="i4"), ColumnSpec(name="a", dtype="i4")],
        column_order=["a", "b"],
    )
    assert t.column_names == ["a", "b"]


def test_units_and_description_attrs(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ColumnSpec(name="x", dtype="f4", units="MeV", description="energy")]
    )
    assert t["x"].units == "MeV"
    assert t["x"].description == "energy"


def test_column_filters_roundtrip(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    spec = ColumnSpec(
        name="v",
        dtype="i4",
        filters=FilterPipeline([Shuffle(), hdf5plugin.Zstd(clevel=5)]),
        chunks=1000,
    )
    t = Table.create(g, [spec])
    t.append({"v": np.arange(5000, dtype="i4")})
    assert list(t.read()["v"][:5]) == [0, 1, 2, 3, 4]
    plist = t["v"].dataset.id.get_create_plist()
    ids = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    assert 2 in ids and 32015 in ids  # shuffle + zstd


def test_validate_passes_on_wellformed_table(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.from_arrays(
        g,
        {"id": np.arange(4, dtype="i8"), "s": ["a", "bb", "ccc", "d"]},
        index_columns=["id"],
    )
    t.validate()


def test_validate_catches_unequal_extents(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.from_arrays(
        g, {"a": np.arange(3, dtype="i4"), "b": np.arange(3, dtype="i4")}
    )
    # Corrupt the invariant behind the API.
    g["a"].resize((5,))
    with pytest.raises(ConformanceError):
        t.validate()


def test_equal_extents_and_nrows_semantics(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    t.append({"x": [1, 2, 3]})
    # NROWS is authoritative; the dataset extent is >= NROWS.
    assert t.nrows == 3
    assert g["x"].shape[0] >= t.nrows


# -- review-driven regression tests ---------------------------------------- #
def test_failed_create_leaves_no_table_and_allows_retry(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(ReservedNameError):
        Table.create(
            g,
            [
                ColumnSpec(name="ok", dtype="i4"),
                ColumnSpec(name="CATEGORIES", dtype="i4"),
            ],
        )
    # No CLASS marker and no stray column left behind.
    assert not Table.is_table_group(g)
    assert "CLASS" not in g.attrs
    assert list(g.keys()) == []
    # A retry works.
    t = Table.create(g, [ColumnSpec(name="ok", dtype="i4")])
    assert t.column_names == ["ok"]


def test_failed_create_rolls_back_partial_writes(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    # A string column with two compressors fails during creation (after the
    # first column and the CLASS marker were written) -> everything rolls back.
    two_compressors = FilterPipeline([hdf5plugin.Zstd(clevel=5), hdf5plugin.LZ4()])
    with pytest.raises(FilterError):
        Table.create(
            g,
            [
                ColumnSpec(name="a", dtype="i4"),
                ColumnSpec(name="s", dtype=FixedString(4), filters=two_compressors),
            ],
        )
    assert not Table.is_table_group(g)
    assert list(g.keys()) == []


def test_column_without_user_fill_reports_no_missing(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.from_arrays(g, {"a": np.arange(3, dtype="i4")})
    # A dataset created with no user-defined fill (h5py library default).
    nf = g.create_dataset("nf", data=np.zeros(3, dtype="i4"))
    col = Column(nf, t)
    assert col.fill_value is None
    assert list(col.is_missing()) == [False, False, False]


def test_add_column_backfills_missing(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.from_arrays(g, {"a": np.arange(3, dtype="i4")})
    t.add_column(ColumnSpec(name="b", dtype="i4"))
    assert "b" in t.column_names
    assert list(t["b"].is_missing()) == [True, True, True]


def test_add_boolean_column_to_nonempty_table_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.from_arrays(g, {"a": np.arange(3, dtype="i4")})
    with pytest.raises(SchemaError):
        t.add_column(ColumnSpec(name="flag", dtype=bool_dtype()))


def test_validate_rejects_non_uint64_nrows(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    del g.attrs["NROWS"]
    g.attrs.create("NROWS", np.int32(0))
    with pytest.raises(ConformanceError):
        Table.open(g).validate()


def test_validate_rejects_nonscalar_nrows(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    del g.attrs["NROWS"]
    g.attrs.create("NROWS", np.array([0, 1], dtype="u8"))
    with pytest.raises(ConformanceError):
        Table.open(g).validate()


def test_validate_rejects_index_disagreeing_with_underscore_index(
    h5file: h5py.File,
) -> None:
    g = h5file.create_group("t")
    t = Table.from_arrays(
        g,
        {"id": np.arange(3, dtype="i8"), "x": np.arange(3, dtype="i4")},
        index_columns=["id"],
    )
    del g.attrs["_index"]
    write_utf8_attr(g, "_index", "x")  # disagrees with INDEX_COLUMNS[0] == "id"
    with pytest.raises(ConformanceError):
        t.validate()


def test_numeric_filter_order_preserved(h5file: h5py.File) -> None:
    # Declared deflate-then-shuffle order is kept exactly (high-level would reorder).
    g = h5file.create_group("t")
    pipe = FilterPipeline([Deflate(4), Shuffle()])
    t = Table.create(g, [ColumnSpec(name="v", dtype="i4", filters=pipe, chunks=1000)])
    t.append({"v": np.arange(5000, dtype="i4")})
    plist = t["v"].dataset.id.get_create_plist()
    ids = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    assert ids == [1, 2]
    assert list(t.read()["v"][:4]) == [0, 1, 2, 3]


def test_optional_filter_flag_preserved(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    pipe = FilterPipeline([Filter(1, (4,), optional=True)])
    t = Table.create(g, [ColumnSpec(name="v", dtype="i4", filters=pipe, chunks=1000)])
    t.append({"v": np.arange(2000, dtype="i4")})
    fid, flags, _, _ = t["v"].dataset.id.get_create_plist().get_filter(0)
    assert fid == 1
    assert flags & 1  # H5Z_FLAG_OPTIONAL


def test_append_rejects_scalar_values(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="x", dtype="i4")])
    with pytest.raises(SchemaError):
        t.append({"x": 5})  # not a 1-D sequence
