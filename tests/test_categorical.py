"""Tests for categorical columns and the CATEGORIES group."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from h5col import ColumnSpec, Table
from h5col.exceptions import ConformanceError, SchemaError


def test_categorical_roundtrip_string_labels(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [ColumnSpec(name="color", categories=["red", "green", "blue"], ordered=False)],
    )
    t.append({"color": ["red", "blue", "red", "green"]})
    assert t.nrows == 4
    col = t["color"]
    assert col.is_categorical
    assert list(col.categories) == ["red", "green", "blue"]
    assert list(col.read()) == ["red", "blue", "red", "green"]
    assert list(col.codes) == [0, 2, 0, 1]
    assert col.ordered is False


def test_categorical_missing_via_none(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b"])])
    t.append({"c": ["a", None, "b"]})
    col = t["c"]
    assert list(col.read()) == ["a", None, "b"]
    assert list(col.is_missing()) == [False, True, False]


def test_categorical_unknown_label_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b"])])
    with pytest.raises(SchemaError):
        t.append({"c": ["a", "z"]})
    assert t.nrows == 0  # not committed


def test_categorical_code_dtype_auto_int8(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b", "c"])])
    assert t["c"].dtype == np.dtype("i1")


def test_categorical_explicit_integer_dtype(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b"], dtype="i4")])
    assert t["c"].dtype == np.dtype("i4")


def test_categorical_noninteger_dtype_rejected() -> None:
    with pytest.raises(ValidationError):
        ColumnSpec(name="c", categories=["a", "b"], dtype="f4")


def test_categorical_fill_collision_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(SchemaError):
        Table.create(g, [ColumnSpec(name="c", categories=["a", "b"], fill_value=0)])
    assert not Table.is_table_group(g)  # pre-flight failure leaves group unmarked


def test_categorical_duplicate_categories_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(SchemaError):
        Table.create(g, [ColumnSpec(name="c", categories=["a", "a"])])


def test_categorical_numeric_labels(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="n", categories=[10, 20, 30])])
    t.append({"n": [10, 30, 20]})
    assert list(t["n"].read()) == [10, 30, 20]


def test_categories_group_and_column_discovery(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="id", dtype="i8"),
            ColumnSpec(name="c", categories=["a", "b"]),
        ],
    )
    assert "CATEGORIES" in g
    # The code column is a column; the categories dataset is not.
    assert t.column_names == ["id", "c"]
    assert "CATEGORIES" not in t.column_names


def test_categorical_validate_and_reopen(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    spec = ColumnSpec(name="c", categories=["x", "y", "z"], ordered=True)
    t = Table.create(g, [spec])
    t.append({"c": ["x", "z", "y"]})
    t.validate()
    t2 = Table.open(g)
    assert list(t2["c"].read()) == ["x", "z", "y"]
    assert t2["c"].ordered is True


def test_categorical_unsigned_code_dtype_default_fill(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b"], dtype="u1")])
    assert t["c"].dtype == np.dtype("u1")
    # Default unsigned fill is the type max (255), outside [0, 2).
    assert int(t["c"].fill_value) == 255
    t.append({"c": ["a", None, "b"]})
    assert list(t["c"].read()) == ["a", None, "b"]
    assert list(t["c"].is_missing()) == [False, True, False]


def test_categorical_unrepresentable_fill_raises_cleanly(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(SchemaError):  # not a raw OverflowError
        Table.create(
            g, [ColumnSpec(name="c", categories=["a", "b"], dtype="u1", fill_value=-1)]
        )
    assert not Table.is_table_group(g)


def test_categorical_code_dtype_too_small_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    cats = [str(i) for i in range(200)]  # needs codes up to 199
    with pytest.raises(SchemaError):
        Table.create(g, [ColumnSpec(name="c", categories=cats, dtype="i1")])


def test_categorical_auto_dtype_scales_to_int16(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    cats = [str(i) for i in range(200)]
    t = Table.create(g, [ColumnSpec(name="c", categories=cats)])
    assert t["c"].dtype == np.dtype("i2")


def test_categorical_validate_catches_orphan_categories(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b"])])
    # Inject an orphan dataset into CATEGORIES (referenced by no column).
    g["CATEGORIES"].create_dataset("orphan", data=np.arange(2))
    with pytest.raises(ConformanceError):
        t.validate()
