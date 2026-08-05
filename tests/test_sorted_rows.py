"""Tests for the SORTED_ROWS search-index family."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest

from h5col import (
    ColumnSpec,
    ConformanceError,
    FixedString,
    ReservedNameError,
    SchemaError,
    SortedRowsIndex,
    StaleIndexError,
    Table,
    bool_dtype,
    indexes,
    references,
)
from h5col._hdf5 import write_ascii_token_attr, write_uint64_attr
from h5col.reserved import (
    ATTR_FILL_TAIL_LENGTH,
    ATTR_KIND,
    ATTR_NAN_TAIL_LENGTH,
    ATTR_ORDERED,
    ATTR_SEARCH_INDEX_LIST,
    ATTR_SOURCE_GENERATION,
    ATTR_SOURCE_NROWS,
    GROUP_SEARCH_INDEXES,
    KIND_SORTED_ROWS,
)

CHUNK = 16


def make_int_table(f: h5py.File, n: int = 100) -> Table:
    """An int64 column with fill -1 (every 9th row missing), seeded random."""
    rng = np.random.default_rng(7)
    t = Table.create(
        f.create_group("t"),
        [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=CHUNK)],
    )
    x = rng.integers(0, 50, n)  # low range: plenty of ties
    x[::9] = -1
    t.append({"x": x})
    return t


def oracle_perm(
    values: Any, fill: Any, *, is_float: bool = False
) -> tuple[list[int], int, int]:
    """Independent permutation oracle: sorted() with row-position tie-break."""
    nan_rows = [r for r, v in enumerate(values) if is_float and np.isnan(v)]
    if fill is not None and not (is_float and np.isnan(fill)):
        fill_rows = [
            r for r, v in enumerate(values) if r not in set(nan_rows) and v == fill
        ]
    elif fill is not None:  # NaN fill: every missing row is a NaN row
        fill_rows = []
    else:
        fill_rows = []
    in_tails = set(nan_rows) | set(fill_rows)
    body = [r for r in range(len(values)) if r not in in_tails]
    body.sort(key=lambda r: (values[r], r))
    return body + fill_rows + nan_rows, len(fill_rows), len(nan_rows)


def brute_rows(values: Any, present: Any, op: str, value: Any) -> set[int]:
    """Row positions with a present value satisfying the predicate (oracle)."""
    ops = {
        "<": lambda v: v < value,
        "<=": lambda v: v <= value,
        ">": lambda v: v > value,
        ">=": lambda v: v >= value,
        "==": lambda v: v == value,
        "between": lambda v: value[0] <= v <= value[1],
    }
    return {r for r in range(len(values)) if present[r] and ops[op](values[r])}


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
class TestSortedRowsContent:
    def test_int_permutation_matches_oracle(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert isinstance(si, SortedRowsIndex)
        x = t.group["x"][: t.nrows]
        expected, fill_tail, nan_tail = oracle_perm(x.tolist(), -1)
        assert si.permutation().tolist() == expected
        assert si.fill_tail_length == fill_tail
        assert si.nan_tail_length == nan_tail == 0
        assert si.dataset.dtype == np.uint64
        assert si.ordered is True

    def test_tie_break_increasing_row(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=4)],
        )
        t.append({"x": [3, 1, 3, 1, 3]})
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert si.permutation().tolist() == [1, 3, 0, 2, 4]

    def test_float_fill_tail_then_nan_tail(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", fill_value=-9999.0, chunks=4)],
        )
        vals = [2.5, np.nan, -9999.0, 1.0, np.nan, -9999.0, 0.5]
        t.append({"v": vals})
        si = t.add_search_index("v", KIND_SORTED_ROWS)
        # body sorted, then fill rows (2, 5), then NaN rows (1, 4)
        assert si.permutation().tolist() == [6, 3, 0, 2, 5, 1, 4]
        assert si.fill_tail_length == 2
        assert si.nan_tail_length == 2

    def test_nan_fill_column_has_empty_fill_tail(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", fill_value=np.nan, chunks=4)],
        )
        t.append({"v": [3.0, np.nan, 1.0, np.nan]})
        si = t.add_search_index("v", KIND_SORTED_ROWS)
        assert si.permutation().tolist() == [2, 0, 1, 3]
        assert si.fill_tail_length == 0
        assert si.nan_tail_length == 2  # every missing row is a NaN row

    def test_string_byte_wise_order(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="s", dtype=FixedString(4), fill_value="~~~~", chunks=4)],
        )
        # Byte-wise UTF-8: "a" < "aa" < "b" < "É" (0xC3 0x89); fill last.
        t.append({"s": ["b", "É", "a", "~~~~", "aa"]})
        si = t.add_search_index("s", KIND_SORTED_ROWS)
        assert si.permutation().tolist() == [2, 4, 0, 1, 3]
        assert si.fill_tail_length == 1

    def test_boolean_false_before_true(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="b", dtype=bool_dtype(), chunks=4)],
        )
        t.append({"b": [True, False, True, False]})
        si = t.add_search_index("b", KIND_SORTED_ROWS)
        assert si.permutation().tolist() == [1, 3, 0, 2]

    def test_categorical_sorts_by_code(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="c", categories=["low", "mid", "high"], chunks=4)],
        )
        t.append({"c": ["high", "low", None, "mid"]})
        si = t.add_search_index("c", KIND_SORTED_ROWS)
        # codes: 2, 0, fill, 1 -> body [1, 3, 0], fill tail [2]
        assert si.permutation().tolist() == [1, 3, 0, 2]
        assert si.fill_tail_length == 1

    def test_empty_table(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", chunks=4)],
        )
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert si.is_valid
        assert si.permutation().tolist() == []
        assert si.rows("==", 1).tolist() == []
        t.validate(deep=True)

    def test_reopen_from_disk(self, h5path: Any) -> None:
        with h5py.File(h5path, "w") as f:
            t = make_int_table(f)
            t.add_search_index("x", KIND_SORTED_ROWS)
            expected = t.search_indexes["x__sorted_rows"].permutation().tolist()
        with h5py.File(h5path, "r") as f:
            t = Table.open(f["t"])
            si = t.search_indexes["x__sorted_rows"]
            assert isinstance(si, SortedRowsIndex)
            assert si.is_valid
            assert si.permutation().tolist() == expected
            assert set(si.rows("==", 10)) == brute_rows(
                f["t/x"][: t.nrows].tolist(),
                f["t/x"][: t.nrows] != -1,
                "==",
                10,
            )


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
class TestSortedRowsQueries:
    @pytest.mark.parametrize(
        ("op", "value"),
        [
            ("<", 25),
            ("<=", 25),
            (">", 25),
            (">=", 25),
            ("==", 10),
            ("between", (10, 30)),
            ("<", 0),
            (">", 49),
            ("==", 999),
        ],
    )
    def test_int_queries_match_scan(
        self, h5file: h5py.File, op: str, value: Any
    ) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        x = t.group["x"][: t.nrows].tolist()
        present = [v != -1 for v in x]
        assert set(si.rows(op, value).tolist()) == brute_rows(x, present, op, value)

    def test_results_in_rank_order(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        rows = si.rows(">=", 25)
        x = t.group["x"][: t.nrows]
        vals = x[rows]
        assert (np.diff(vals) >= 0).all()  # ascending by value

    def test_string_queries_match_scan(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="s", dtype=FixedString(6), fill_value="", chunks=4)],
        )
        vals = ["pear", "fig", "", "apple", "fig", "plum", "b", ""]
        t.append({"s": vals})
        si = t.add_search_index("s", KIND_SORTED_ROWS)
        present = [v != "" for v in vals]
        bvals = [v.encode() for v in vals]
        for op, q in [
            ("==", b"fig"),
            ("<", b"fig"),
            (">=", b"b"),
            ("between", (b"apple", b"fig")),
        ]:
            got = set(si.rows(op, q).tolist())
            assert got == brute_rows(bvals, present, op, q)
        # str and bytes queries agree
        assert set(si.rows("==", "fig")) == set(si.rows("==", b"fig"))

    def test_between_inverted_bounds_empty(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert si.rows("between", (30, 10)).tolist() == []

    def test_fill_value_query_matches_no_missing_row(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert si.rows("==", -1).tolist() == []  # fill rows live in the tail
        # but range predicates spanning the fill value still exclude them
        got = si.rows("<", 0)
        assert got.tolist() == []

    def test_exact_at_huge_int_magnitudes(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        base = 2**60
        t.append({"x": [base, base + 1, base - 127, base + 129]})
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        # Python-int comparisons are exact where float64 would round.
        assert si.rows("==", base + 1).tolist() == [1]
        assert set(si.rows(">", base).tolist()) == {1, 3}
        assert si.rows(">", 2**64).tolist() == []  # beyond uint64 range too

    def test_nan_query_rejected(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="v", dtype="f8", chunks=4)],
        )
        t.append({"v": [1.0, 2.0]})
        si = t.add_search_index("v", KIND_SORTED_ROWS)
        with pytest.raises(SchemaError, match="NaN"):
            si.rows("==", float("nan"))

    def test_unknown_operator_rejected(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        with pytest.raises(SchemaError, match="operator"):
            si.rows("!=", 1)

    def test_vlen_foreign_index_is_queryable(self, h5file: h5py.File) -> None:
        # The spec orders vlen strings; a conformant foreign SORTED_ROWS over
        # a vlen column must be queryable even though our builder refuses it.
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=4)],
        )
        t.append({"x": [1, 2, 3]})
        g = t.group
        del g.attrs["column-order"]
        vals = ["pear", "apple", "fig"]
        s = g.create_dataset(
            "s",
            data=vals,
            dtype=h5py.string_dtype(),
            chunks=(4,),
            maxshape=(None,),
            fillvalue="",
        )
        gen = indexes.ensure_generation(g)
        si_group = g.require_group(GROUP_SEARCH_INDEXES)
        order = sorted(range(3), key=lambda r: vals[r])
        ds = si_group.create_dataset(
            "s__sr", data=np.array(order, dtype="u8"), chunks=(8,), maxshape=(None,)
        )
        write_ascii_token_attr(ds, ATTR_KIND, KIND_SORTED_ROWS)
        write_uint64_attr(ds, ATTR_FILL_TAIL_LENGTH, 0)
        write_uint64_attr(ds, ATTR_NAN_TAIL_LENGTH, 0)
        ds.attrs.create(ATTR_ORDERED, np.True_)
        write_uint64_attr(ds, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(ds, ATTR_SOURCE_NROWS, t.nrows)
        references.append_ref_to_array_attr(s, ATTR_SEARCH_INDEX_LIST, ds)

        si = t.search_indexes["s__sr"]
        assert si.is_valid
        assert si.rows("==", "fig").tolist() == [2]
        assert set(si.rows(">=", "fig").tolist()) == {0, 2}
        t.validate(deep=True)  # deep skips the vlen oracle, structure passes


# --------------------------------------------------------------------------- #
# Lifecycle: append / truncate / refresh maintenance
# --------------------------------------------------------------------------- #
class TestSortedRowsLifecycle:
    def test_append_default_stale_then_refresh(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        t.append({"x": [7, 3]})
        assert not si.is_valid
        with pytest.raises(StaleIndexError):
            si.rows("==", 7)
        assert t.refresh_indexes() == 1
        assert si.is_valid
        x = t.group["x"][: t.nrows].tolist()
        expected, _, _ = oracle_perm(x, -1)
        assert si.permutation().tolist() == expected

    def test_append_maintain_rebuilds(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        t.append({"x": [7, 3]}, maintain_indexes=True)
        assert si.is_valid
        x = t.group["x"][: t.nrows].tolist()
        expected, _, _ = oracle_perm(x, -1)
        assert si.permutation().tolist() == expected
        t.validate(deep=True)

    def test_truncate_maintain_rebuilds(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file, n=50)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        t.truncate(20, maintain_indexes=True)
        assert si.is_valid
        assert si.source_nrows == 20
        x = t.group["x"][:20].tolist()
        expected, _, _ = oracle_perm(x, -1)
        assert si.permutation().tolist() == expected
        # residue beyond NROWS is permitted; the dataset never shrank
        assert si.dataset.shape[0] == 50
        t.validate(deep=True)

    def test_foreign_narrow_dtype_skipped_by_maintenance(
        self, h5file: h5py.File
    ) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=64)],
        )
        t.append({"x": np.arange(10, dtype="i8")})
        g = t.group
        gen = indexes.ensure_generation(g)
        si_group = g.require_group(GROUP_SEARCH_INDEXES)
        perm, fill_tail, nan_tail = indexes.compute_sorted_rows(g["x"], 10)
        ds = si_group.create_dataset(
            "x__sr8",
            data=perm.astype("u1"),
            dtype="u1",
            chunks=(256,),
            maxshape=(None,),
        )
        write_ascii_token_attr(ds, ATTR_KIND, KIND_SORTED_ROWS)
        write_uint64_attr(ds, ATTR_FILL_TAIL_LENGTH, fill_tail)
        write_uint64_attr(ds, ATTR_NAN_TAIL_LENGTH, nan_tail)
        ds.attrs.create(ATTR_ORDERED, np.True_)
        write_uint64_attr(ds, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(ds, ATTR_SOURCE_NROWS, 10)
        references.append_ref_to_array_attr(g["x"], ATTR_SEARCH_INDEX_LIST, ds)
        si = t.search_indexes["x__sr8"]
        assert si.is_valid

        # 300 rows cannot be addressed by uint8: maintenance must skip the
        # index, leaving its (now stale) tokens untouched.
        t.append({"x": np.arange(300, dtype="i8")}, maintain_indexes=True)
        assert not si.is_valid
        assert si.source_nrows == 10  # untouched, not clobbered

    def test_foreign_non_growable_skipped_by_maintenance(
        self, h5file: h5py.File
    ) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=8)],
        )
        t.append({"x": [4, 2, 9]})
        g = t.group
        gen = indexes.ensure_generation(g)
        si_group = g.require_group(GROUP_SEARCH_INDEXES)
        perm, fill_tail, nan_tail = indexes.compute_sorted_rows(g["x"], 3)
        ds = si_group.create_dataset("x__sr", data=perm)  # contiguous, fixed
        write_ascii_token_attr(ds, ATTR_KIND, KIND_SORTED_ROWS)
        write_uint64_attr(ds, ATTR_FILL_TAIL_LENGTH, fill_tail)
        write_uint64_attr(ds, ATTR_NAN_TAIL_LENGTH, nan_tail)
        ds.attrs.create(ATTR_ORDERED, np.True_)
        write_uint64_attr(ds, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(ds, ATTR_SOURCE_NROWS, 3)
        references.append_ref_to_array_attr(g["x"], ATTR_SEARCH_INDEX_LIST, ds)

        t.append({"x": [1]}, maintain_indexes=True)
        si = t.search_indexes["x__sr"]
        assert not si.is_valid
        assert si.source_nrows == 3  # untouched
        # but truncation back to a size it can hold maintains it in place
        t.truncate(2, maintain_indexes=True)
        assert si.is_valid
        assert si.permutation().tolist() == [1, 0]

    def test_refresh_index_raises_on_unfit(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=8)],
        )
        t.append({"x": [4, 2, 9]})
        g = t.group
        gen = indexes.ensure_generation(g)
        si_group = g.require_group(GROUP_SEARCH_INDEXES)
        perm, fill_tail, nan_tail = indexes.compute_sorted_rows(g["x"], 3)
        ds = si_group.create_dataset("x__sr", data=perm)  # contiguous, fixed
        write_ascii_token_attr(ds, ATTR_KIND, KIND_SORTED_ROWS)
        write_uint64_attr(ds, ATTR_FILL_TAIL_LENGTH, fill_tail)
        write_uint64_attr(ds, ATTR_NAN_TAIL_LENGTH, nan_tail)
        ds.attrs.create(ATTR_ORDERED, np.True_)
        write_uint64_attr(ds, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(ds, ATTR_SOURCE_NROWS, 3)
        references.append_ref_to_array_attr(g["x"], ATTR_SEARCH_INDEX_LIST, ds)

        t.append({"x": [1, 1]})  # 5 rows no longer fit the fixed dataset
        with pytest.raises(SchemaError, match="cannot rebuild"):
            indexes.refresh_index(g, ds, g["x"])


# --------------------------------------------------------------------------- #
# Validation (rule 9)
# --------------------------------------------------------------------------- #
class TestSortedRowsValidate:
    def _build(self, h5file: h5py.File) -> tuple[Table, SortedRowsIndex]:
        t = make_int_table(h5file, n=40)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert isinstance(si, SortedRowsIndex)
        return t, si

    def test_validate_deep_passes(self, h5file: h5py.File) -> None:
        t, _ = self._build(h5file)
        t.validate(deep=True)

    def test_duplicate_entry_caught_structurally(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset[0] = si.dataset[1]  # no longer a permutation
        with pytest.raises(ConformanceError, match="permutation"):
            t.validate()

    def test_out_of_range_entry_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset[0] = t.nrows + 5
        with pytest.raises(ConformanceError, match="permutation"):
            t.validate()

    def test_wrong_order_caught_only_deep(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        x = t.group["x"][: t.nrows]
        perm = si.permutation()
        # find two adjacent body rows with distinct values and swap them
        i = next(i for i in range(len(perm) - 1) if x[perm[i]] != x[perm[i + 1]])
        si.dataset[i], si.dataset[i + 1] = perm[i + 1], perm[i]
        t.validate()  # still a permutation: structural check passes
        with pytest.raises(ConformanceError, match="deep"):
            t.validate(deep=True)

    def test_wrong_tail_attr_caught_deep(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        stored = si.fill_tail_length
        assert stored is not None and stored > 0
        write_uint64_attr(si.dataset, ATTR_FILL_TAIL_LENGTH, stored - 1)
        with pytest.raises(ConformanceError, match="deep"):
            t.validate(deep=True)

    def test_missing_ordered_attr_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        del si.dataset.attrs[ATTR_ORDERED]
        with pytest.raises(ConformanceError, match="ordered"):
            t.validate()
        with pytest.raises(ConformanceError, match="ordered"):
            si.rows("==", 1)

    def test_ordered_false_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset.attrs.modify(ATTR_ORDERED, np.False_)
        with pytest.raises(ConformanceError, match="must be true"):
            t.validate()
        with pytest.raises(ConformanceError, match="ordered"):
            si.rows("==", 1)

    def test_malformed_tail_attr_dtype_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        del si.dataset.attrs[ATTR_NAN_TAIL_LENGTH]
        si.dataset.attrs.create(ATTR_NAN_TAIL_LENGTH, np.int32(0))
        with pytest.raises(ConformanceError, match="uint64"):
            t.validate()
        with pytest.raises(ConformanceError, match="tail"):
            si.rows("==", 1)

    def test_tails_exceeding_nrows_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        write_uint64_attr(si.dataset, ATTR_FILL_TAIL_LENGTH, t.nrows + 1)
        with pytest.raises(ConformanceError, match="tail"):
            t.validate()

    def test_undersized_dataset_caught(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset.resize((t.nrows - 2,))
        with pytest.raises(ConformanceError, match="entries"):
            t.validate()
        with pytest.raises(ConformanceError, match="entries"):
            si.rows("==", 1)

    def test_signed_dtype_caught(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=8)],
        )
        t.append({"x": [3, 1, 2]})
        g = t.group
        gen = indexes.ensure_generation(g)
        si_group = g.require_group(GROUP_SEARCH_INDEXES)
        ds = si_group.create_dataset("x__sr", data=np.array([1, 2, 0], dtype="i8"))
        write_ascii_token_attr(ds, ATTR_KIND, KIND_SORTED_ROWS)
        write_uint64_attr(ds, ATTR_FILL_TAIL_LENGTH, 0)
        write_uint64_attr(ds, ATTR_NAN_TAIL_LENGTH, 0)
        ds.attrs.create(ATTR_ORDERED, np.True_)
        write_uint64_attr(ds, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(ds, ATTR_SOURCE_NROWS, 3)
        references.append_ref_to_array_attr(g["x"], ATTR_SEARCH_INDEX_LIST, ds)
        with pytest.raises(ConformanceError, match="unsigned"):
            t.validate()

    def test_stale_index_exempt_from_rule9(self, h5file: h5py.File) -> None:
        t, si = self._build(h5file)
        si.dataset[0] = si.dataset[1]  # corrupt content
        t.append({"x": [1]})  # GENERATION bump disables the index
        assert not si.is_valid
        t.validate(deep=True)  # exempt: treated as absent


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #
class TestSortedRowsApi:
    def test_default_and_custom_names(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        assert si.name == "x__sorted_rows"
        si2 = t.add_search_index("x", KIND_SORTED_ROWS, name="by_x")
        assert si2.name == "by_x"

    def test_duplicate_name_raises(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        t.add_search_index("x", KIND_SORTED_ROWS)
        with pytest.raises(SchemaError, match="already contains"):
            t.add_search_index("x", KIND_SORTED_ROWS)

    def test_bad_names_rejected_without_leftovers(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        for bad in ("NROWS", "a/b"):
            with pytest.raises((ReservedNameError, SchemaError)):
                t.add_search_index("x", KIND_SORTED_ROWS, name=bad)
        assert GROUP_SEARCH_INDEXES not in t.group

    def test_column_bound_wrapper(self, h5file: h5py.File) -> None:
        t = make_int_table(h5file)
        col = t["x"]
        si = col.add_search_index(KIND_SORTED_ROWS)
        assert isinstance(si, SortedRowsIndex)
        assert si.column is not None and si.column.name == "x"
        got = col.search_indexes
        assert len(got) == 1 and isinstance(got[0], SortedRowsIndex)

    def test_unsupported_dtype_refused(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", chunks=4)],
        )
        g = t.group
        del g.attrs["column-order"]
        g.create_dataset("v", data=["a", "b"], dtype=h5py.string_dtype())
        with pytest.raises(SchemaError, match="SORTED_ROWS"):
            t.add_search_index("v", KIND_SORTED_ROWS)


# --------------------------------------------------------------------------- #
# Adversarial-review regressions (4b review)
# --------------------------------------------------------------------------- #
class TestSortedRowsReviewRegressions:
    def test_maintenance_restores_missing_ordered_attr(self, h5file: h5py.File) -> None:
        # Review: maintenance rewrote content and tails but not the mandatory
        # `ordered` attribute, stamping valid a state validate() rejects.
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        t.append({"x": [3, 1, 2]})
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        del si.dataset.attrs[ATTR_ORDERED]  # foreign tool dropped it
        t.append({"x": [4]}, maintain_indexes=True)
        assert si.is_valid
        assert si.ordered is True
        t.validate(deep=True)

    def test_refresh_leaves_valid_indexes_untouched(self, h5file: h5py.File) -> None:
        # Review: refreshing a currently VALID index rewrote its content in
        # place, opening a crash window with torn content behind passing
        # tokens. Valid indexes are now left alone.
        t = make_int_table(h5file, n=20)
        si = t.add_search_index("x", KIND_SORTED_ROWS)
        p0, p1 = int(si.dataset[0]), int(si.dataset[1])
        si.dataset[0], si.dataset[1] = p1, p0  # sentinel: content mutated
        assert t.refresh_indexes() == 0
        indexes.refresh_index(t.group, si.dataset, t.group["x"])  # no-op
        assert [int(si.dataset[0]), int(si.dataset[1])] == [p1, p0]

    def test_doubly_claimed_index_not_maintained(self, h5file: h5py.File) -> None:
        # Review: maintained mutations rebuilt a doubly-claimed index once
        # per claimant (last column won) and stamped it valid — exactly the
        # wrong-column content find_index_column refuses to resolve.
        t = Table.create(
            h5file.create_group("t"),
            [
                ColumnSpec(name="a", dtype="i8", fill_value=-1, chunks=4),
                ColumnSpec(name="b", dtype="i8", fill_value=-1, chunks=4),
            ],
        )
        t.append({"a": [3, 1, 2], "b": [9, 8, 7]})
        si = t.add_search_index("a", KIND_SORTED_ROWS)
        references.append_ref_to_array_attr(
            t.group["b"], ATTR_SEARCH_INDEX_LIST, si.dataset
        )
        t.append({"a": [0], "b": [6]}, maintain_indexes=True)
        assert not indexes.index_is_valid(si.dataset, t.group)
        assert indexes._scalar_uint64(si.dataset.attrs, ATTR_SOURCE_NROWS) == 3
