"""Tests for the __repr__ of the live-HDF5 wrapper classes (phase 5B).

Reprs must be informative, show the *logical* row count, and — critically —
never raise (a repr is what you reach for in a debugger on a possibly-dead
object).
"""

from __future__ import annotations

import h5py

from h5col import (
    ColumnSpec,
    Deflate,
    Filter,
    FilterPipeline,
    LeafValuesSpec,
    ListColumnSpec,
    Table,
    bool_dtype,
)
from h5col.reserved import ATTR_NROWS
from h5col.strings import FixedString


def make_table(h5file: h5py.File, n: int = 20) -> Table:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=8),
            ColumnSpec(name="size", categories=["S", "M", "L", "XL"], chunks=8),
            ColumnSpec(name="ok", dtype=bool_dtype(), chunks=8),
        ],
    )
    t.append(
        {
            "x": list(range(n)),
            "size": [["S", "M", "L", "XL"][i % 4] for i in range(n)],
            "ok": [i % 2 == 0 for i in range(n)],
        }
    )
    return t


class TestTableRepr:
    def test_shows_nrows(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=12)
        assert repr(t) == "<h5col.Table '/t' nrows=12>"

    def test_raise_safe_on_closed_file(self, tmp_path) -> None:
        f = h5py.File(tmp_path / "t.h5", "w")
        t = make_table(f)
        f.close()
        assert repr(t) == "<h5col.Table (closed or invalid)>"  # must not raise

    def test_raise_safe_on_missing_nrows(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        del t.group.attrs[ATTR_NROWS]  # foreign/half-built group
        assert repr(t) == "<h5col.Table (closed or invalid)>"


class TestColumnRepr:
    def test_shows_dtype_and_logical_nrows(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=20)
        r = repr(t.columns["x"])
        assert r.startswith("<h5col.Column 'x' dtype=")
        assert "nrows=20>" in r

    def test_logical_nrows_not_reserved_extent(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=20)
        t.truncate(7)  # extent stays 20, logical nrows is 7
        assert t.columns["x"].dataset.shape[0] == 20
        assert "nrows=7>" in repr(t.columns["x"])

    def test_categorical_shows_count(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        r = repr(t.columns["size"])
        assert "categories=4" in r  # flags categorical; dtype alone hides it
        assert "nrows=20>" in r

    def test_non_categorical_has_no_categories(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        assert "categories=" not in repr(t.columns["x"])

    def test_raise_safe_on_closed_file(self, tmp_path) -> None:
        f = h5py.File(tmp_path / "t.h5", "w")
        t = make_table(f)
        col = t.columns["x"]
        f.close()
        assert repr(col) == "<h5col.Column (closed or invalid)>"


class TestListColumnRepr:
    def test_shows_nullable_and_nrows(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [
                ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=8),
                ListColumnSpec(
                    name="r", values=LeafValuesSpec(dtype="f4"), nullable=True
                ),
            ],
        )
        t.append({"x": [1, 2, 3], "r": [[1.0], None, [3.0]]})
        r = repr(t.columns["r"])
        assert r == "<h5col.ListColumn 'r' nullable=True nrows=3>"


class TestSearchIndexRepr:
    def test_bound_wrapper_shows_column(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("x", "SORTED_ROWS")
        bound = t.columns["x"].search_indexes[0]  # bound to its column
        r = repr(bound)
        assert r.startswith("<h5col.SortedRowsIndex 'x__sorted_rows'")
        assert "column='x'" in r
        assert "kind='SORTED_ROWS'" in r
        assert "valid=True>" in r

    def test_unbound_wrapper_has_no_column(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("x", "SORTED_ROWS")
        unbound = t.search_indexes["x__sorted_rows"]  # resolved table-wide
        assert "column=" not in repr(unbound)  # must not scan to find it

    def test_bitmap_shows_nvalues_and_exhaustive(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        bm = t.add_search_index("size", "BITMAP")  # 4 distinct categories
        r = repr(bm)
        assert "kind='BITMAP'" in r
        assert "nvalues=4" in r
        assert "exhaustive=True" in r

    def test_chunk_minmax_has_no_per_kind_extra(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        cm = t.add_search_index("x", "CHUNK_MINMAX")
        r = repr(cm)
        assert "nvalues=" not in r
        assert r.endswith("valid=True>")

    def test_raise_safe_on_closed_file(self, tmp_path) -> None:
        f = h5py.File(tmp_path / "t.h5", "w")
        t = make_table(f)
        si = t.add_search_index("size", "BITMAP")
        f.close()
        assert repr(si) == "<h5col.BitmapIndex (closed or invalid)>"


class TestValueObjectReprsUnchanged:
    """Dataclass / Pydantic value objects keep their faithful auto reprs."""

    def test_fixed_string(self) -> None:
        assert repr(FixedString(6)) == "FixedString(nbytes=6, encoding='utf-8')"

    def test_filter_and_pipeline(self) -> None:
        assert "Filter(" in repr(Deflate(4))
        assert repr(Deflate(4)).startswith("Filter(")
        assert "FilterPipeline(" in repr(FilterPipeline([Deflate(4)]))
        assert repr(Filter(1, (4,))).startswith("Filter(")

    def test_spec(self) -> None:
        assert "ColumnSpec(" in repr(ColumnSpec(name="x", dtype="i8"))
