"""Tests for the BITMAP search-index family."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest

from h5col import (
    BitmapIndex,
    ChunkMinMaxIndex,
    ColumnSpec,
    ConformanceError,
    FixedString,
    SchemaError,
    StaleIndexError,
    Table,
    bool_dtype,
    indexes,
    references,
)
from h5col._hdf5 import write_ascii_token_attr, write_bool_attr, write_uint64_attr
from h5col.reserved import (
    ATTR_EXHAUSTIVE,
    ATTR_KIND,
    ATTR_ORDERED,
    ATTR_SEARCH_INDEX_LIST,
    ATTR_SOURCE_GENERATION,
    ATTR_SOURCE_NROWS,
    ATTR_VALUES,
    GROUP_SEARCH_INDEXES,
    KIND_BITMAP,
)

DATA = [3, 1, 3, -1, 7, 1, 3, -1, 7, 3, 1]  # fill -1, 11 rows (pad bits exist)


def make_table(f: h5py.File) -> Table:
    t = Table.create(
        f.create_group("t"),
        [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
    )
    t.append({"x": DATA})
    return t


def scan_rows(values: Any, fill: Any, value: Any) -> list[int]:
    return [r for r, v in enumerate(values) if v != fill and v == value]


def hand_bitmap(
    t: Table,
    column: str,
    values: np.ndarray,
    bits: np.ndarray,
    *,
    exhaustive: bool = True,
    name: str = "hand__bitmap",
) -> Any:
    """Hand-build a conformant BITMAP with valid tokens over *column*."""
    g = t.group
    gen = indexes.ensure_generation(g)
    si_group = g.require_group(GROUP_SEARCH_INDEXES)
    values_ds = si_group.create_dataset(f"{name}_values", data=values)
    bitmap_ds = si_group.create_dataset(name, data=bits, dtype="u1")
    write_ascii_token_attr(bitmap_ds, ATTR_KIND, KIND_BITMAP)
    references.write_ref_attr(bitmap_ds, ATTR_VALUES, values_ds)
    write_bool_attr(bitmap_ds, ATTR_ORDERED, False)
    write_bool_attr(bitmap_ds, ATTR_EXHAUSTIVE, exhaustive)
    write_uint64_attr(bitmap_ds, ATTR_SOURCE_GENERATION, gen)
    write_uint64_attr(bitmap_ds, ATTR_SOURCE_NROWS, t.nrows)
    references.append_ref_to_array_attr(g[column], ATTR_SEARCH_INDEX_LIST, bitmap_ds)
    return bitmap_ds


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
class TestBitmapContent:
    def test_int_enumeration_and_bits(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        assert isinstance(si, BitmapIndex)
        assert si.values().tolist() == [1, 3, 7]  # sorted, fill excluded
        assert si.ordered is True
        assert si.exhaustive is True
        assert si.dataset.shape == (3, 2)  # ceil(11 / 8) == 2
        for k, v in enumerate([1, 3, 7]):
            got = np.unpackbits(si.dataset[k], bitorder="little", count=t.nrows)
            assert np.flatnonzero(got).tolist() == scan_rows(DATA, -1, v)

    def test_values_dataset_is_kindless_sibling(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        values_ds = si.values_dataset
        assert values_ds.name == f"{t.group.name}/SEARCH_INDEXES/x__bitmap_values"
        assert ATTR_KIND not in values_ds.attrs
        assert values_ds.dtype == t.group["x"].dtype
        # accompanying dataset: not listed among the search indexes
        assert set(t.search_indexes) == {"x__bitmap"}

    def test_pad_bits_are_zero(self, h5file: h5py.File) -> None:
        t = make_table(h5file)  # 11 rows: 5 pad bits in byte 1
        si = t.add_search_index("x", KIND_BITMAP)
        last = si.dataset[:, 1]
        assert not np.any(last & np.uint8(0b11111000))

    def test_boolean_column(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="b", dtype=bool_dtype(), chunks=4)],
        )
        t.append({"b": [True, False, True, False, False]})
        si = t.add_search_index("b")  # auto-pick
        assert isinstance(si, BitmapIndex)
        assert si.values().tolist() == [False, True]
        assert si.rows(False).tolist() == [1, 3, 4]
        assert si.rows(True).tolist() == [0, 2]

    def test_categorical_column_by_label_and_code(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="c", categories=["red", "green", "blue"], chunks=4)],
        )
        t.append({"c": ["red", "blue", None, "blue", "green", "red"]})
        si = t.add_search_index("c")  # auto-pick -> BITMAP
        assert isinstance(si, BitmapIndex)
        assert si.values().tolist() == [0, 1, 2]
        assert si.rows("blue").tolist() == [1, 3]
        assert si.rows(2).tolist() == [1, 3]  # raw code queries work too
        assert si.rows("nope").tolist() == []  # label outside the category set

    def test_string_column(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="s", dtype=FixedString(4), fill_value="", chunks=4)],
        )
        t.append({"s": ["fig", "", "fig", "pear", "b"]})
        si = t.add_search_index("s", KIND_BITMAP)
        assert si.values().tolist() == [b"b", b"fig", b"pear"]
        assert si.rows("fig").tolist() == [0, 2]
        assert si.rows(b"fig").tolist() == [0, 2]

    def test_negative_zero_is_one_value(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", fill_value=-9999.0, chunks=4)],
        )
        t.append({"v": [-0.0, 1.5, 0.0]})
        si = t.add_search_index("v", KIND_BITMAP)
        assert si.dataset.shape[0] == 2  # -0.0 == +0.0: one enumerated value
        assert si.rows(0.0).tolist() == [0, 2]
        assert si.rows(-0.0).tolist() == [0, 2]

    def test_nonmissing_nan_makes_exhaustive_false(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", fill_value=-9999.0, chunks=4)],
        )
        t.append({"v": [1.0, np.nan, 2.0]})
        si = t.add_search_index("v", KIND_BITMAP)
        assert si.values().tolist() == [1.0, 2.0]
        assert si.exhaustive is False
        assert si.rows(1.0).tolist() == [0]
        assert si.rows(5.0) is None  # partial enumeration cannot prove absence

    def test_nan_fill_column_is_exhaustive(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", fill_value=np.nan, chunks=4)],
        )
        t.append({"v": [1.0, np.nan, 2.0]})
        si = t.add_search_index("v", KIND_BITMAP)
        assert si.exhaustive is True  # every NaN is missing here
        assert si.rows(5.0).tolist() == []

    def test_empty_table(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", chunks=4)],
        )
        si = t.add_search_index("x", KIND_BITMAP)
        assert si.is_valid
        assert si.dataset.shape == (0, 0)
        assert si.rows(1).tolist() == []
        t.validate(deep=True)

    def test_reopen_from_disk(self, h5path: Any) -> None:
        with h5py.File(h5path, "w") as f:
            t = make_table(f)
            t.add_search_index("x", KIND_BITMAP)
        with h5py.File(h5path, "r") as f:
            t = Table.open(f["t"])
            si = t.search_indexes["x__bitmap"]
            assert isinstance(si, BitmapIndex)
            assert si.is_valid
            assert si.rows(3).tolist() == scan_rows(DATA, -1, 3)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
class TestBitmapQueries:
    def test_rows_match_scan_for_every_value(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        for v in (1, 3, 7):
            assert si.rows(v).tolist() == scan_rows(DATA, -1, v)

    def test_absent_value_exhaustive_empty(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        assert si.rows(999).tolist() == []

    def test_absent_value_non_exhaustive_none(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        write_bool_attr(si.dataset, ATTR_EXHAUSTIVE, False)
        assert si.rows(999) is None
        assert si.rows(3).tolist() == scan_rows(DATA, -1, 3)  # hits still exact

    def test_missing_exhaustive_attr_means_none(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        del si.dataset.attrs[ATTR_EXHAUSTIVE]
        assert si.exhaustive is False
        assert si.rows(999) is None

    def test_fill_value_query_returns_empty(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        assert si.rows(-1).tolist() == []  # missing rows never match

    def test_foreign_bitmap_enumerating_fill_still_excludes_missing(
        self, h5file: h5py.File
    ) -> None:
        # Bit semantics are raw equality, so a foreign producer MAY enumerate
        # the fill value with bits set on missing rows; the H5Col query rule
        # (missing rows never match) must still hold.
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        t.append({"x": [5, 5, -1, 7]})
        values = np.array([-1, 5, 7], dtype="i8")
        bits = np.array([[0b0100], [0b0011], [0b1000]], dtype="u1")
        hand_bitmap(t, "x", values, bits)
        si = t.search_indexes["hand__bitmap"]
        assert si.is_valid
        assert si.rows(-1).tolist() == []
        assert si.rows(5).tolist() == [0, 1]
        t.validate(deep=True)  # raw-equality bits are conformant

    def test_pad_bits_ignored_by_queries(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        si.dataset[0, 1] = si.dataset[0, 1] | 0b11111000  # corrupt pad bits
        assert si.rows(1).tolist() == scan_rows(DATA, -1, 1)  # unaffected
        with pytest.raises(ConformanceError, match="padding"):
            t.validate()  # but producers MUST write them as 0

    def test_isin_union(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        expected = sorted(set(scan_rows(DATA, -1, 1)) | set(scan_rows(DATA, -1, 7)))
        assert si.isin([1, 7]).tolist() == expected
        assert si.isin([1, 999]).tolist() == sorted(scan_rows(DATA, -1, 1))

    def test_isin_none_when_unprovable(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        write_bool_attr(si.dataset, ATTR_EXHAUSTIVE, False)
        assert si.isin([1, 999]) is None

    def test_nan_query_rejected(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", chunks=4)],
        )
        t.append({"v": [1.0]})
        si = t.add_search_index("v", KIND_BITMAP)
        with pytest.raises(SchemaError, match="NaN"):
            si.rows(float("nan"))

    def test_stale_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        t.append({"x": [1]})
        with pytest.raises(StaleIndexError):
            si.rows(1)


# --------------------------------------------------------------------------- #
# Lifecycle: append / truncate / refresh maintenance
# --------------------------------------------------------------------------- #
class TestBitmapLifecycle:
    def test_append_maintain_grows_enumeration(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        t.append({"x": [42, 3]}, maintain_indexes=True)
        assert si.is_valid
        assert si.values().tolist() == [1, 3, 7, 42]
        data = DATA + [42, 3]
        assert si.rows(42).tolist() == scan_rows(data, -1, 42)
        assert si.rows(3).tolist() == scan_rows(data, -1, 3)
        t.validate(deep=True)

    def test_truncate_maintain_shrinks_enumeration_exactly(
        self, h5file: h5py.File
    ) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        t.truncate(3, maintain_indexes=True)  # rows [3, 1, 3]: value 7 gone
        assert si.is_valid
        assert si.values().tolist() == [1, 3]
        # K is defined by the datasets, so no residue row may survive
        assert si.dataset.shape == (2, 1)
        assert si.values_dataset.shape == (2,)
        assert si.rows(7).tolist() == []  # exhaustive: provably absent
        t.validate(deep=True)

    def test_append_default_stale_then_refresh(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        t.append({"x": [42]})
        assert not si.is_valid
        assert t.refresh_indexes() == 1
        assert si.is_valid
        assert si.values().tolist() == [1, 3, 7, 42]

    def test_foreign_non_resizable_skipped_by_maintenance(
        self, h5file: h5py.File
    ) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        t.append({"x": [5, 5, -1, 7]})
        values = np.array([5, 7], dtype="i8")
        bits = np.array([[0b0011], [0b1000]], dtype="u1")
        hand_bitmap(t, "x", values, bits)  # contiguous, fixed-shape datasets
        si = t.search_indexes["hand__bitmap"]
        assert si.is_valid

        t.append({"x": [11]}, maintain_indexes=True)
        assert not si.is_valid  # skipped: cannot resize to the new shape
        assert si.source_nrows == 4  # tokens untouched

    def test_dangling_values_ref_skipped_by_maintenance(
        self, h5file: h5py.File
    ) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        del t.group[f"{GROUP_SEARCH_INDEXES}/x__bitmap_values"]
        t.append({"x": [1]}, maintain_indexes=True)
        assert not si.is_valid
        assert si.source_nrows == len(DATA)  # tokens untouched


# --------------------------------------------------------------------------- #
# Validation (rules 3 and 9)
# --------------------------------------------------------------------------- #
class TestBitmapValidate:
    def _build(self, h5file: h5py.File) -> tuple[Table, BitmapIndex]:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        assert isinstance(si, BitmapIndex)
        return t, si

    def test_validate_deep_passes(self, h5file: h5py.File) -> None:
        t, _ = self._build(h5file)
        t.validate(deep=True)

    def test_orphan_bitmap_values_still_vouched(self, h5file: h5py.File) -> None:
        # Rule 3: the KIND-less values dataset is vouched for by the bitmap's
        # VALUES reference even when no column claims the bitmap.
        t, _ = self._build(h5file)
        del t.group["x"].attrs[ATTR_SEARCH_INDEX_LIST]
        t.validate()

    def test_bit_flip_caught_only_deep(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset[0, 0] = si.dataset[0, 0] ^ 0b0000_0001
        t.validate()  # structural checks cannot see it
        with pytest.raises(ConformanceError, match="deep"):
            t.validate(deep=True)

    def test_kind_on_values_dataset_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        values_ds = si.values_dataset
        write_ascii_token_attr(values_ds, ATTR_KIND, "FANCY")
        # tokens keep rule 12 quiet so the bitmap-specific rule is exercised
        write_uint64_attr(values_ds, ATTR_SOURCE_GENERATION, 0)
        write_uint64_attr(values_ds, ATTR_SOURCE_NROWS, t.nrows)
        with pytest.raises(ConformanceError, match="KIND"):
            t.validate()

    def test_values_dtype_mismatch_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        g = t.group[GROUP_SEARCH_INDEXES]
        wrong = g.create_dataset("wrong_values", data=np.array([1, 3, 7], "i4"))
        del si.dataset.attrs[ATTR_VALUES]
        references.write_ref_attr(si.dataset, ATTR_VALUES, wrong)
        # drop the now-unvouched original so rule 3 does not fire first
        del g["x__bitmap_values"]
        with pytest.raises(ConformanceError, match="dtype"):
            t.validate()

    def test_row_count_mismatch_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset.resize((si.dataset.shape[0] + 1, si.dataset.shape[1]))
        with pytest.raises(ConformanceError, match="values"):
            t.validate()

    def test_array_values_ref_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        values_ds = si.values_dataset
        ref = references.make_ref(values_ds)
        del si.dataset.attrs[ATTR_VALUES]
        si.dataset.attrs.create(
            ATTR_VALUES, np.array([ref], dtype=references.ref_dtype())
        )
        # An array-valued VALUES cannot vouch for the values dataset either,
        # so drop it to reach the scalar-reference check itself.
        del t.group[GROUP_SEARCH_INDEXES]["x__bitmap_values"]
        with pytest.raises(ConformanceError, match="scalar"):
            t.validate()
        with pytest.raises(ConformanceError, match="values dataset"):
            si.rows(1)

    def test_missing_values_attr_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        del si.dataset.attrs[ATTR_VALUES]
        # the orphaned values dataset now trips rule 3 first; remove it to
        # exercise the bitmap's own required-attribute check
        del t.group[GROUP_SEARCH_INDEXES]["x__bitmap_values"]
        with pytest.raises(ConformanceError, match="VALUES"):
            t.validate()

    def test_exhaustive_claim_miss_caught_deep(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        t.append({"x": [1, 1, 2, 2, 3]})
        values = np.array([1, 2], dtype="i8")  # 3 is missing from the claim
        bits = np.array([[0b00011], [0b01100]], dtype="u1")
        hand_bitmap(t, "x", values, bits, exhaustive=True)
        t.validate()  # bits themselves are correct
        with pytest.raises(ConformanceError, match="exhaustive"):
            t.validate(deep=True)

    def test_exhaustive_with_nonmissing_nan_caught_deep(
        self, h5file: h5py.File
    ) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", fill_value=-9999.0, chunks=4)],
        )
        t.append({"v": [1.0, np.nan]})
        values = np.array([1.0], dtype="f8")
        bits = np.array([[0b01]], dtype="u1")
        hand_bitmap(t, "v", values, bits, exhaustive=True)
        with pytest.raises(ConformanceError, match="NaN"):
            t.validate(deep=True)

    def test_malformed_ordered_attr_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        del si.dataset.attrs[ATTR_ORDERED]
        si.dataset.attrs.create(ATTR_ORDERED, np.int8(1))
        with pytest.raises(ConformanceError, match="boolean"):
            t.validate()

    def test_absent_ordered_and_exhaustive_ok(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        del si.dataset.attrs[ATTR_ORDERED]
        del si.dataset.attrs[ATTR_EXHAUSTIVE]
        t.validate(deep=False)
        assert si.ordered is None
        assert si.exhaustive is False

    def test_stale_index_exempt(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset[0, 0] = 0xFF  # corrupt bits AND pad
        t.append({"x": [1]})  # GENERATION bump disables the index
        t.validate(deep=True)


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #
class TestBitmapApi:
    def test_autopick_numeric_still_minmax(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x")
        assert isinstance(si, ChunkMinMaxIndex)

    def test_default_names_and_collisions(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        assert si.name == "x__bitmap"
        with pytest.raises(SchemaError, match="already contains"):
            t.add_search_index("x", KIND_BITMAP)
        # the values-dataset name is claimed too
        si2 = t.add_search_index("x", KIND_BITMAP, name="other")
        with pytest.raises(SchemaError, match="already contains"):
            t.add_search_index("x", KIND_BITMAP, name="other_values")
        assert si2.name == "other"

    def test_unknown_kind_rejected(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="not implemented"):
            t.add_search_index("x", "CHUNK_BLOOM")

    def test_column_bound_wrapper(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        col = t["x"]
        si = col.add_search_index(KIND_BITMAP)
        assert isinstance(si, BitmapIndex)
        assert si.column is not None and si.column.name == "x"


# --------------------------------------------------------------------------- #
# Adversarial-review regressions (4b review)
# --------------------------------------------------------------------------- #
class TestBitmapReviewRegressions:
    def test_enumerated_but_clipped_value_raises(self, h5file: h5py.File) -> None:
        # Review: rows() used to clip the enumeration to the bitmap row count,
        # so a value present in VALUES but without a bitmap row fell into the
        # miss branch — and exhaustive=True turned that into a silently wrong
        # "provably zero rows".
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        t.append({"x": [1, 2, 3, 2]})
        values = np.array([1, 2, 3], dtype="i8")
        bits = np.array([[0b0001], [0b1010]], dtype="u1")  # row for 3 missing
        g = t.group
        gen = indexes.ensure_generation(g)
        sig = g.require_group(GROUP_SEARCH_INDEXES)
        vals = sig.create_dataset("v", data=values)
        bm = sig.create_dataset("b", data=bits, dtype="u1")
        write_ascii_token_attr(bm, ATTR_KIND, KIND_BITMAP)
        references.write_ref_attr(bm, ATTR_VALUES, vals)
        write_bool_attr(bm, ATTR_EXHAUSTIVE, True)
        write_uint64_attr(bm, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(bm, ATTR_SOURCE_NROWS, 4)
        references.append_ref_to_array_attr(g["x"], ATTR_SEARCH_INDEX_LIST, bm)
        si = t.search_indexes["b"]
        assert si.is_valid
        with pytest.raises(ConformanceError, match="more values"):
            si.rows(3)
        assert si.rows(2).tolist() == [1, 3]  # rows that exist still answer
        with pytest.raises(ConformanceError):  # K mismatch is a rule-9 error
            t.validate()

    def test_values_charset_mismatch_rejected_and_not_maintained(
        self, h5file: h5py.File
    ) -> None:
        # Review: the values dataset must hold "the same datatype as the
        # source column" — the HDF5 datatype, so an ASCII values dataset over
        # a UTF-8 string column is nonconformant even though NumPy sees the
        # same |S4 on both sides.
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="s", dtype=FixedString(4), fill_value="")],
        )
        t.append({"s": ["a", "b", "a"]})
        si = t.add_search_index("s", KIND_BITMAP)
        sig = t.group[GROUP_SEARCH_INDEXES]
        ascii_vals = sig.create_dataset(
            "ascii_values", data=np.array([b"a", b"b"], dtype="S4")
        )
        del si.dataset.attrs[ATTR_VALUES]
        references.write_ref_attr(si.dataset, ATTR_VALUES, ascii_vals)
        del sig["s__bitmap_values"]
        with pytest.raises(ConformanceError, match="dtype"):
            t.validate()
        # maintenance must skip it too, leaving the tokens untouched
        src_nrows = si.source_nrows
        t.append({"s": ["c"]}, maintain_indexes=True)
        assert not si.is_valid
        assert si.source_nrows == src_nrows

    def test_create_rollback_between_dataset_creations(
        self, h5file: h5py.File, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review: _create_bitmap_datasets ran outside the rollback try-block,
        # so a failure after the values dataset was created orphaned a
        # KIND-less dataset in SEARCH_INDEXES (a rule-3 violation).
        t = make_table(h5file)

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(references, "write_ref_attr", boom)
        with pytest.raises(RuntimeError, match="boom"):
            t.add_search_index("x", KIND_BITMAP)
        sig = t.group.get(GROUP_SEARCH_INDEXES)
        if sig is not None:
            assert "x__bitmap" not in sig and "x__bitmap_values" not in sig
        t.validate()  # no rule-3 leftovers

    def test_multi_reference_values_attr_no_crash(self, h5file: h5py.File) -> None:
        # Review: rule-3 vouching passed the VALUES attribute straight into
        # the null-reference test; an array of two references crashed
        # validate() with ValueError instead of ConformanceError.
        t = make_table(h5file)
        si = t.add_search_index("x", KIND_BITMAP)
        values_ds = si.values_dataset
        ref = references.make_ref(values_ds)
        del si.dataset.attrs[ATTR_VALUES]
        si.dataset.attrs.create(
            ATTR_VALUES, np.array([ref, ref], dtype=references.ref_dtype())
        )
        with pytest.raises(ConformanceError):
            t.validate()
