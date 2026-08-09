"""Tests for categorical columns and the CATEGORIES group."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from h5col import ColumnSpec, Table, categorical, references
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


# --------------------------------------------------------------------------- #
# The fill value is the only missing-category marker
# --------------------------------------------------------------------------- #
def _fill_less_code_dataset(h5file: h5py.File) -> tuple[h5py.Group, h5py.Dataset]:
    """A categorical code dataset that declares no fill value.

    Built by hand because h5col's own writer always sets one; the gap only shows
    up on a file written by another producer.
    """
    g = h5file.create_group("t")
    cat_ds = categorical.create_categories_dataset(
        g.create_group("CATEGORIES"), "c__CATEGORIES", ["a", "b", "c"]
    )
    ds = g.create_dataset("c", data=np.array([0, 1, 2], dtype="i1"))
    references.write_ref_attr(ds, "CATEGORIES", cat_ds)
    return g, ds


def test_categorical_decode_without_user_fill_keeps_code_zero(
    h5file: h5py.File,
) -> None:
    # h5py reports a library-default fillvalue of 0 for a dataset that declares
    # none. Read ungated, that turns the *first* category into a missing value.
    g, ds = _fill_less_code_dataset(h5file)
    assert ds.fillvalue == 0
    assert categorical.user_fill_code(ds) is None
    assert list(categorical.decode_codes(g, ds, ds[...])) == ["a", "b", "c"]


def test_categorical_encode_none_without_user_fill_raises(h5file: h5py.File) -> None:
    g, ds = _fill_less_code_dataset(h5file)
    with pytest.raises(SchemaError, match="no fill value"):
        categorical.encode_labels(g, ds, ["a", None])


def test_categorical_out_of_range_code_raises(h5file: h5py.File) -> None:
    # An unindexable code is a malformed file, not another spelling of missing:
    # decoding it to None would contradict is_missing() and the query layer.
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b", "c"])])
    t.append({"c": ["a", "b", "c", "a"]})
    t["c"].dataset[2] = 7
    with pytest.raises(ConformanceError, match="neither the fill code"):
        t["c"].read()


def test_categorical_out_of_range_code_raises_via_read_rows(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b"])])
    t.append({"c": ["a", "b", "a", "b"]})
    t["c"].dataset[3] = 9
    assert list(t["c"].read_rows([0, 1])) == ["a", "b"]  # untouched rows still read
    with pytest.raises(ConformanceError, match="categorical code 9"):
        t["c"].read_rows([1, 3])


def test_categorical_read_and_is_missing_never_disagree(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ColumnSpec(name="c", categories=["a", "b", "c"])])
    t.append({"c": ["a", None, "b", None, "c"]})
    col = t["c"]
    from_read = np.array([v is None for v in col.read()])
    np.testing.assert_array_equal(from_read, col.is_missing())
