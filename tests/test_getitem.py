"""Tests for reading a column by subscript.

``column[key]`` is :meth:`Column.read_rows` with its defaults, so the contract
is agreement: for every key both spellings must give the same answer, and a
list column must accept the same keys a scalar one does.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import (
    ColumnSpec,
    FixedString,
    LeafValuesSpec,
    ListColumnSpec,
    Table,
    bool_dtype,
)

NROWS = 40


def _table(h5file: h5py.File) -> Table:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="num", dtype="int32", chunks=8, fill_value=-1),
            ColumnSpec(name="txt", dtype=FixedString(nbytes=8), fill_value="N/A"),
            ColumnSpec(name="flag", dtype=bool_dtype()),
            ColumnSpec(name="cat", categories=["a", "b", "c"]),
            ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="f8"), nullable=True),
        ],
    )
    labels = ["a", "b", "c", None]
    t.append(
        {
            "num": [i if i % 5 else None for i in range(NROWS)],
            "txt": [f"s{i:03d}" if i % 4 else None for i in range(NROWS)],
            "flag": [bool(i % 2) for i in range(NROWS)],
            "cat": [labels[i % 4] for i in range(NROWS)],
            # A null row, an empty row, and rows of differing length.
            "xs": [
                None if i % 7 == 0 else [float(j) for j in range(i % 4)]
                for i in range(NROWS)
            ],
        }
    )
    return t


SCALAR_COLUMNS = ["num", "txt", "flag", "cat"]
ALL_COLUMNS = [*SCALAR_COLUMNS, "xs"]

KEYS = {
    "slice": slice(4, 12),
    "whole": slice(None),
    "strided": slice(1, 30, 4),
    "reversed": slice(None, None, -1),
    "negative_bounds": slice(-10, -2),
    "empty_slice": slice(9, 4),
    "positions": [7, 2, 7],
    "negative_positions": [-1, -NROWS],
    "empty_positions": [],
}


@pytest.mark.parametrize("colname", ALL_COLUMNS)
@pytest.mark.parametrize("case", list(KEYS))
def test_subscript_agrees_with_read_rows(
    h5file: h5py.File, colname: str, case: str
) -> None:
    col = _table(h5file)[colname]
    key = KEYS[case]
    assert list(col[key]) == list(col.read_rows(key))


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_whole_column_subscript_equals_read(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    assert list(col[:]) == list(col.read())


def _same(a: object, b: object) -> bool:
    """Equality that treats two masked values as equal.

    ``np.ma.masked == np.ma.masked`` is ``masked``, which is falsy, so a plain
    ``==`` would report two missing rows as different.
    """
    if a is np.ma.masked or b is np.ma.masked:
        return a is b
    return bool(a == b)


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_boolean_mask_subscript(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    # Built rather than taken from is_missing(), because a boolean column has
    # no missing rows and its mask would select nothing.
    mask = np.zeros(NROWS, dtype=bool)
    mask[[1, 2, 3, NROWS - 1]] = True
    assert list(col[mask]) == list(col.read_rows(mask))
    assert len(col[mask]) == 4


# --------------------------------------------------------------------------- #
# An integer key drops the dimension
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("colname", SCALAR_COLUMNS)
def test_integer_key_returns_one_value_not_an_array(
    h5file: h5py.File, colname: str
) -> None:
    col = _table(h5file)[colname]
    value = col[1]
    assert not isinstance(value, np.ndarray)
    assert value == col.read()[1]


def test_integer_key_on_a_missing_row_is_masked(h5file: h5py.File) -> None:
    t = _table(h5file)
    # row 0 is missing in both "num" and "txt"
    assert t["num"].is_missing()[0]
    assert t["num"][0] is np.ma.masked
    assert t["txt"][0] is np.ma.masked


def test_integer_key_on_a_present_row_is_the_decoded_value(h5file: h5py.File) -> None:
    t = _table(h5file)
    assert t["txt"][1] == "s001"
    assert t["num"][1] == 1
    assert t["cat"][1] == "b"
    assert t["flag"][1] is np.True_


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_negative_integer_key_counts_back(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    assert _same(col[-1], col[NROWS - 1])
    assert _same(col[-NROWS], col[0])


def test_integer_key_on_a_list_column(h5file: h5py.File) -> None:
    col = _table(h5file)["xs"]
    rows = col.read()
    assert col[0] is None  # a null row, not an empty one
    assert col[1] == rows[1]
    assert col[7 * 2] is None
    assert col[-1] == rows[-1]


def test_empty_list_row_is_not_a_null_row(h5file: h5py.File) -> None:
    col = _table(h5file)["xs"]
    assert col[8] == []  # i % 7 != 0 and i % 4 == 0
    assert col[8] is not None


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("colname", ALL_COLUMNS)
@pytest.mark.parametrize("key", [NROWS, -NROWS - 1])
def test_out_of_range_integer_is_rejected(
    h5file: h5py.File, colname: str, key: int
) -> None:
    col = _table(h5file)[colname]
    with pytest.raises(IndexError, match="out of range"):
        col[key]


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_tuple_key_is_rejected(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    with pytest.raises(IndexError, match="one-dimensional"):
        col[0, 1]


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_bare_boolean_is_rejected(h5file: h5py.File, colname: str) -> None:
    # True is an int in Python, so without a guard column[True] would quietly
    # read row 1.
    col = _table(h5file)[colname]
    with pytest.raises(TypeError, match="not a row selection"):
        col[True]


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_non_integer_key_is_rejected(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    with pytest.raises(TypeError, match="not a row selection"):
        col[1.5]
    with pytest.raises(TypeError, match="must be integers"):
        col[[1.5]]


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_wrong_length_boolean_mask_is_rejected(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    with pytest.raises(IndexError, match="one entry per row"):
        col[np.zeros(NROWS - 1, dtype=bool)]


# --------------------------------------------------------------------------- #
# len() and iteration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_len_is_nrows(h5file: h5py.File, colname: str) -> None:
    t = _table(h5file)
    assert len(t[colname]) == t.nrows == NROWS


@pytest.mark.parametrize("colname", ALL_COLUMNS)
def test_iteration_matches_read(h5file: h5py.File, colname: str) -> None:
    col = _table(h5file)[colname]
    assert list(iter(col)) == list(col.read())


def test_iteration_reads_the_column_once(h5file: h5py.File) -> None:
    # Without __iter__, Python falls back on __getitem__ and reads one row at
    # a time — NROWS reads instead of one.
    t = _table(h5file)
    col = t["num"]
    reads = []
    real = col.dataset

    class Counting:
        def __getattr__(self, name: str) -> object:
            return getattr(real, name)

        def __getitem__(self, key: object) -> object:
            reads.append(key)
            return real[key]

    col._ds = Counting()  # type: ignore[assignment]
    assert len(list(col)) == NROWS
    assert len(reads) == 1


# --------------------------------------------------------------------------- #
# Subscript reads committed rows, not reserved storage
# --------------------------------------------------------------------------- #


def test_subscript_respects_nrows_after_truncate(h5file: h5py.File) -> None:
    t = _table(h5file)
    t.truncate(6)
    col = t["num"]
    assert col.dataset.shape[0] > 6

    assert len(col) == 6
    assert len(col[:]) == 6
    assert len(list(col)) == 6
    with pytest.raises(IndexError):
        col[6]
    # The raw dataset still shows the reserved rows; the column must not.
    assert col.dataset[:].shape[0] > 6
