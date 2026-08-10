"""Tests for gathered (chunk-coalesced) row reads.

The contract is equivalence: ``col.read_rows(rows)`` must be indistinguishable
from ``col.read()[rows]`` for every column datatype, so that the selective
path a query takes can never change an answer.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from h5col import ColumnSpec, FixedString, Table, bool_dtype, field
from h5col._hdf5 import gather_rows
from h5col.query import _worth_gathering

NROWS = 500
CHUNK = 32


def _table(h5file: h5py.File) -> Table:
    table = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="num", dtype="int32", chunks=CHUNK, fill_value=-1),
            ColumnSpec(name="flt", dtype="float64", chunks=CHUNK, fill_value=np.nan),
            ColumnSpec(name="txt", dtype=FixedString(nbytes=8), chunks=CHUNK),
            ColumnSpec(name="flag", dtype=bool_dtype(), chunks=CHUNK),
            ColumnSpec(name="cat", categories=["a", "b", "c"], chunks=CHUNK),
        ],
    )
    labels = ["a", "b", "c", None]
    table.append(
        {
            "num": [i if i % 7 else None for i in range(NROWS)],
            "flt": [float(i) if i % 5 else None for i in range(NROWS)],
            "txt": [f"s{i:04d}" for i in range(NROWS)],
            "flag": [bool(i % 2) for i in range(NROWS)],
            "cat": [labels[i % 4] for i in range(NROWS)],
        }
    )
    return table


ROW_SETS = {
    "clustered": np.arange(64, 80),
    "scattered": np.array([0, 97, 198, 301, 499]),
    "single": np.array([250]),
    "first_and_last": np.array([0, NROWS - 1]),
    "empty": np.empty(0, dtype=np.int64),
    "dense": np.arange(0, NROWS, 3),
    "all": np.arange(NROWS),
}


@pytest.mark.parametrize("colname", ["num", "flt", "txt", "flag", "cat"])
@pytest.mark.parametrize("case", list(ROW_SETS))
def test_read_rows_matches_full_read_then_subset(
    h5file: h5py.File, colname: str, case: str
) -> None:
    table = _table(h5file)
    rows = ROW_SETS[case]
    col = table[colname]

    gathered = col.read_rows(rows)
    expected = col.read()[rows]

    assert gathered.shape == expected.shape
    assert gathered.dtype == expected.dtype
    if colname == "flt":
        np.testing.assert_array_equal(gathered, expected)  # NaN-aware
    else:
        assert list(gathered) == list(expected)


def test_read_rows_preserves_caller_order_and_duplicates(h5file: h5py.File) -> None:
    table = _table(h5file)
    rows = np.array([300, 1, 300, 64, 1])
    assert list(table["txt"].read_rows(rows)) == list(table["txt"].read()[rows])


def test_read_rows_rejects_out_of_range(h5file: h5py.File) -> None:
    table = _table(h5file)
    with pytest.raises(IndexError):
        table["num"].read_rows([NROWS])
    with pytest.raises(IndexError):
        table["num"].read_rows([-NROWS - 1])


def test_read_rows_rejects_non_1d(h5file: h5py.File) -> None:
    table = _table(h5file)
    with pytest.raises(ValueError, match="1-D"):
        table["num"].read_rows(np.array([[0, 1], [2, 3]]))


def test_gather_reads_only_the_needed_chunks(h5path: Path) -> None:
    # Rows confined to one chunk must not pull the rest of the column: read
    # from a file opened with a tiny chunk cache and compare element counts by
    # instrumenting the dataset's __getitem__ through a proxy.
    with h5py.File(h5path, "w") as f:
        table = Table.create(
            f.create_group("t"),
            [ColumnSpec(name="x", dtype="int64", chunks=CHUNK)],
        )
        table.append({"x": np.arange(NROWS)})

    with h5py.File(h5path, "r") as f:
        table = Table.open(f["t"])
        ds = table["x"].dataset
        seen: list[int] = []

        class Counting:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __getitem__(self, key):
                block = self._real[key]
                seen.append(int(np.size(block)))
                return block

        # A contiguous run is taken as a single slice of exactly that width.
        rows = np.arange(64, 80)
        out = gather_rows(Counting(ds), rows, NROWS)
        assert list(out) == list(rows)
        assert seen == [16]

        # Non-contiguous rows fall to the chunk-run path: still one read, one
        # chunk wide, and in neither case the whole 500-row column.
        seen.clear()
        scattered = np.array([64, 70, 79])
        out = gather_rows(Counting(ds), scattered, NROWS)
        assert list(out) == list(scattered)
        assert seen == [CHUNK]


def test_worth_gathering_declines_when_rows_span_the_column(h5file: h5py.File) -> None:
    table = _table(h5file)
    col = table["num"]
    total_chunks = -(-NROWS // CHUNK)

    assert _worth_gathering(col, np.arange(0, 16), NROWS) is True
    assert _worth_gathering(col, np.empty(0, dtype=np.int64), NROWS) is True
    # One row in every chunk: a gather would read the whole column anyway.
    spread = np.arange(0, NROWS, CHUNK)
    assert spread.size == total_chunks
    assert _worth_gathering(col, spread, NROWS) is False


def test_selection_read_matches_the_unoptimized_path(h5file: h5py.File) -> None:
    table = _table(h5file)
    for expr in (
        field("num") > 480,  # selective, clustered at the end
        field("cat") == "b",  # spread across every chunk
        field("num") > 10**9,  # matches nothing
    ):
        sel = table.select(expr)
        rows = sel.row_positions
        got = sel.read(["num", "flt", "txt", "flag", "cat"])
        for name, values in got.items():
            expected = table[name].read()[rows]
            np.testing.assert_array_equal(values, expected)


def test_selection_read_on_contiguous_column(h5path: Path) -> None:
    # A column with no chunking has nothing to coalesce; the result must still
    # be correct via the full-read path.
    with h5py.File(h5path, "w") as f:
        g = f.create_group("t")
        table = Table.create(g, [ColumnSpec(name="x", dtype="int32", chunks=16)])
        table.append({"x": np.arange(100, dtype="int32")})
        contiguous = g.create_dataset("y", data=np.arange(100, dtype="int32"))
        assert contiguous.chunks is None
        out = gather_rows(contiguous, np.array([5, 60]), 100)
        assert list(out) == [5, 60]


# --------------------------------------------------------------------------- #
# Slices
#
# A slice never reaches gather_rows: it is one hyperslab, so it skips the sort
# and scatter the scattered path needs. The contract is the same either way —
# the result must equal what NumPy would give for the same slice of read().
# --------------------------------------------------------------------------- #

SLICES = {
    "clustered": slice(64, 80),
    "whole": slice(None),
    "strided": slice(10, 200, 7),
    "reversed": slice(None, None, -1),
    "reversed_strided": slice(None, None, -3),
    "backwards_range": slice(100, 40, -2),
    "negative_bounds": slice(-20, -5),
    "open_start": slice(None, 50),
    "open_stop": slice(450, None),
    "empty": slice(20, 10),
    "degenerate": slice(5, 5),
    "past_the_end": slice(NROWS - 3, NROWS + 999),
    "last_row": slice(-1, None),
}


@pytest.mark.parametrize("colname", ["num", "flt", "txt", "flag", "cat"])
@pytest.mark.parametrize("case", list(SLICES))
def test_slice_matches_full_read_then_slice(
    h5file: h5py.File, colname: str, case: str
) -> None:
    table = _table(h5file)
    key = SLICES[case]
    col = table[colname]

    sliced = col.read_rows(key)
    expected = col.read()[key]

    assert sliced.shape == expected.shape
    assert sliced.dtype == expected.dtype
    if colname == "flt":
        np.testing.assert_array_equal(sliced, expected)  # NaN-aware
    else:
        assert list(sliced) == list(expected)


@pytest.mark.parametrize("case", list(SLICES))
def test_slice_carries_the_matching_mask(h5file: h5py.File, case: str) -> None:
    # The mask has to describe the rows that were read, not the rows of the
    # whole column, so a slice that starts partway in cannot be off by its
    # offset.
    table = _table(h5file)
    col = table["num"]
    key = SLICES[case]
    np.testing.assert_array_equal(
        np.ma.getmaskarray(col.read_rows(key)),
        np.ma.getmaskarray(col.read())[key],
    )


def test_slice_stops_at_nrows_not_the_dataset_extent(h5file: h5py.File) -> None:
    # truncate lowers NROWS and leaves the rows above it as reserved storage,
    # which a read must never surface.
    table = _table(h5file)
    table.truncate(10)
    col = table["num"]
    assert col.dataset.shape[0] > 10

    assert col.read_rows(slice(None)).shape == (10,)
    assert col.read_rows(slice(0, 500)).shape == (10,)
    assert list(col.read_rows(slice(-3, None))) == list(col.read()[-3:])


def test_slice_matches_the_equivalent_fancy_selection(h5file: h5py.File) -> None:
    table = _table(h5file)
    col = table["txt"]
    assert list(col.read_rows(slice(64, 80))) == list(col.read_rows(np.arange(64, 80)))


def test_slice_step_of_zero_is_rejected(h5file: h5py.File) -> None:
    table = _table(h5file)
    with pytest.raises(ValueError):
        table["num"].read_rows(slice(None, None, 0))


# --------------------------------------------------------------------------- #
# Boolean masks and negative positions
# --------------------------------------------------------------------------- #


def test_boolean_mask_selects_the_marked_rows(h5file: h5py.File) -> None:
    # A mask cast to an integer dtype becomes a run of ones and zeros, which
    # reads rows 1 and 0 over and over instead of the marked ones — wrong
    # rows, no error.
    table = _table(h5file)
    mask = np.zeros(NROWS, dtype=bool)
    mask[[3, 7, 9, 498]] = True
    col = table["txt"]

    assert list(col.read_rows(mask)) == list(col.read()[mask])
    assert list(col.read_rows(mask)) == list(col.read_rows([3, 7, 9, 498]))


def test_all_false_boolean_mask_reads_nothing(h5file: h5py.File) -> None:
    table = _table(h5file)
    assert table["num"].read_rows(np.zeros(NROWS, dtype=bool)).shape == (0,)


@pytest.mark.parametrize("length", [NROWS - 1, NROWS + 1, 0])
def test_boolean_mask_of_the_wrong_length_is_rejected(
    h5file: h5py.File, length: int
) -> None:
    table = _table(h5file)
    with pytest.raises(IndexError, match="one entry per row"):
        table["num"].read_rows(np.zeros(length, dtype=bool))


def test_negative_positions_count_back_from_the_end(h5file: h5py.File) -> None:
    table = _table(h5file)
    col = table["txt"]
    assert list(col.read_rows([-1, -2, -NROWS])) == list(col.read()[[-1, -2, -NROWS]])


def test_negative_and_positive_positions_mix(h5file: h5py.File) -> None:
    table = _table(h5file)
    col = table["num"]
    rows = [0, -1, 250, -250]
    assert list(col.read_rows(rows)) == list(col.read()[rows])


@pytest.mark.parametrize("rows", [[1.5], [1.0, 2.0], np.array([1.0])])
def test_non_integer_positions_are_rejected(h5file: h5py.File, rows: object) -> None:
    # Casting these would truncate silently, which this package does not do.
    table = _table(h5file)
    with pytest.raises(TypeError, match="must be integers"):
        table["num"].read_rows(rows)


def test_empty_selection_still_works(h5file: h5py.File) -> None:
    table = _table(h5file)
    for rows in ([], np.empty(0, dtype=np.int64)):
        assert table["num"].read_rows(rows).shape == (0,)
