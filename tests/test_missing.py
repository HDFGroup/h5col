"""Tests for fill values and the canonical missing-value test (h5col.missing)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import ColumnSpec, Table
from h5col.booleans import bool_dtype
from h5col.exceptions import FillValueError, SchemaError
from h5col.missing import (
    is_missing,
    masked_to_none,
    recommended_fill,
    validate_fill_outside_range,
)
from h5col.strings import FixedString


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("int8", -127),
        ("int16", -32767),
        ("int32", -2147483647),
        ("int64", -9223372036854775807),
        ("uint8", 255),
        ("uint16", 65535),
        ("uint32", 4294967295),
        ("uint64", 18446744073709551615),
    ],
)
def test_recommended_integer_fills(dtype: str, expected: int) -> None:
    fill = recommended_fill(np.dtype(dtype))
    assert int(fill) == expected
    assert fill.dtype == np.dtype(dtype)


def test_recommended_float_fill_is_exact_in_both_widths() -> None:
    f32 = recommended_fill(np.dtype("float32"))
    f64 = recommended_fill(np.dtype("float64"))
    assert f32 == np.float32(9.9692099683868690e36)
    # Bit-preserving across the width cast (per the spec).
    assert np.float32(f64) == f32


def test_recommended_string_fill_is_empty() -> None:
    assert recommended_fill(FixedString(8).dtype) == b""


def test_recommended_fill_unsupported_dtype() -> None:
    with pytest.raises(FillValueError):
        recommended_fill(np.dtype("float16"))


def test_recommended_fill_enum_returns_missing_code() -> None:
    dt = h5py.enum_dtype({"FALSE": 0, "TRUE": 1, "MISSING": 2}, basetype="i4")
    fill = recommended_fill(dt)
    assert int(fill) == 2
    assert np.dtype(fill.dtype).name == "int32"


def test_recommended_fill_enum_without_missing_raises() -> None:
    dt = h5py.enum_dtype({"RED": 0, "GREEN": 1}, basetype="i4")
    with pytest.raises(FillValueError):
        recommended_fill(dt)


def test_recommended_fill_boolean_raises() -> None:
    # Boolean columns MUST NOT declare a fill value.
    with pytest.raises(FillValueError):
        recommended_fill(bool_dtype())


def test_is_missing_integer() -> None:
    values = np.array([1, -127, 3, -127], dtype="int8")
    mask = is_missing(values, np.int8(-127))
    assert list(mask) == [False, True, False, True]


def test_is_missing_float_equality() -> None:
    fill = np.float32(9.9692099683868690e36)
    values = np.array([1.0, fill, 2.0], dtype="float32")
    assert list(is_missing(values, fill)) == [False, True, False]


def test_is_missing_nan_fill_uses_isnan_branch() -> None:
    values = np.array([1.0, np.nan, 2.0], dtype="float64")
    assert list(is_missing(values, np.float64("nan"))) == [False, True, False]


def test_is_missing_nan_fill_as_0d_ndarray() -> None:
    # A NaN fill delivered as a 0-d array must still take the isnan branch.
    values = np.array([1.0, np.nan, 2.0], dtype="float64")
    assert list(is_missing(values, np.array(np.nan))) == [False, True, False]


def test_is_missing_nonnan_fill_does_not_flag_stray_nan() -> None:
    # With a non-NaN fill, a NaN data value is not "missing" (NaN != fill).
    fill = np.float64(9.9692099683868690e36)
    values = np.array([1.0, np.nan, fill], dtype="float64")
    assert list(is_missing(values, fill)) == [False, False, True]


def test_is_missing_strings() -> None:
    values = np.array([b"a", b"", b"c"], dtype="S3")
    assert list(is_missing(values, b"")) == [False, True, False]


def test_validate_fill_outside_range_ok() -> None:
    validate_fill_outside_range(-127, valid_min=0, valid_max=100)
    validate_fill_outside_range(-127, valid_min=None, valid_max=None)
    validate_fill_outside_range(200, valid_min=0, valid_max=100)


def test_validate_fill_inside_range_raises() -> None:
    with pytest.raises(FillValueError):
        validate_fill_outside_range(50, valid_min=0, valid_max=100)
    with pytest.raises(FillValueError):
        validate_fill_outside_range(0, valid_min=0, valid_max=100)  # boundary inclusive


def test_validate_fill_partial_bounds() -> None:
    validate_fill_outside_range(-1, valid_min=0, valid_max=None)  # below min: ok
    with pytest.raises(FillValueError):
        validate_fill_outside_range(5, valid_min=0, valid_max=None)  # >= min: inside


# --------------------------------------------------------------------------- #
# None in append data means "this row is missing"
# --------------------------------------------------------------------------- #
def _one_column_table(h5file: h5py.File, spec: ColumnSpec) -> Table:
    return Table.create(h5file.create_group("t"), [spec])


def test_append_none_writes_the_integer_fill(h5file: h5py.File) -> None:
    table = _one_column_table(
        h5file, ColumnSpec(name="x", dtype="int32", fill_value=-1, valid_min=0)
    )
    table.append({"x": [1, None, 3]})

    assert list(table["x"].read()) == [1, -1, 3]
    assert list(table["x"].is_missing()) == [False, True, False]


def test_append_none_writes_the_nan_fill(h5file: h5py.File) -> None:
    table = _one_column_table(
        h5file, ColumnSpec(name="x", dtype="float64", fill_value=np.nan)
    )
    table.append({"x": [1.5, None, 3.5]})

    values = table["x"].read()
    assert np.isnan(values[1])
    assert list(table["x"].is_missing()) == [False, True, False]


def test_append_none_writes_the_recommended_sentinel_fill(h5file: h5py.File) -> None:
    # No explicit fill: None must become the column's own recommended sentinel,
    # not NaN.
    table = _one_column_table(h5file, ColumnSpec(name="x", dtype="float64"))
    table.append({"x": [1.5, None]})

    values = table["x"].read()
    assert values[1] == recommended_fill(np.dtype("float64"))
    assert not np.isnan(values[1])
    assert list(table["x"].is_missing()) == [False, True]


def test_append_none_writes_the_string_fill_not_the_word_none(
    h5file: h5py.File,
) -> None:
    table = _one_column_table(h5file, ColumnSpec(name="x", dtype=FixedString(nbytes=8)))
    table.append({"x": ["ab", None, "cd"]})

    values = table["x"].read()
    assert values[1] != "None"  # never str(None)
    assert list(values) == ["ab", "", "cd"]
    assert list(table["x"].is_missing()) == [False, True, False]


def test_append_none_in_object_array_is_honored(h5file: h5py.File) -> None:
    table = _one_column_table(
        h5file, ColumnSpec(name="x", dtype="int32", fill_value=-1, valid_min=0)
    )
    table.append({"x": np.array([1, None, 3], dtype=object)})

    assert list(table["x"].is_missing()) == [False, True, False]


def test_append_none_in_boolean_column_raises(h5file: h5py.File) -> None:
    table = _one_column_table(h5file, ColumnSpec(name="x", dtype=bool_dtype()))
    with pytest.raises(SchemaError, match="boolean"):
        table.append({"x": [True, None]})


def test_append_none_rejected_leaves_the_table_unchanged(h5file: h5py.File) -> None:
    # Validation happens before any column is extended, so a rejected append
    # commits nothing.
    table = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="ok", dtype="int32", fill_value=-1, valid_min=0),
            ColumnSpec(name="flag", dtype=bool_dtype()),
        ],
    )
    table.append({"ok": [1], "flag": [True]})
    with pytest.raises(SchemaError):
        table.append({"ok": [2], "flag": [None]})

    assert table.nrows == 1
    assert list(table["ok"].read()) == [1]


# --------------------------------------------------------------------------- #
# A masked element means the same as None on append
# --------------------------------------------------------------------------- #
def test_masked_to_none_passes_through_non_masked_input() -> None:
    plain = np.array([1, 2, 3])
    assert masked_to_none(plain) is plain
    assert masked_to_none([1, None, 3]) == [1, None, 3]


def test_masked_to_none_keeps_typed_array_when_nothing_is_masked() -> None:
    # `.mask` is the scalar `nomask` here; the result must stay a typed array so
    # the fast path downstream is not turned into a Python list.
    out = masked_to_none(np.ma.masked_array([1.0, 2.0]))
    assert isinstance(out, np.ndarray) and not isinstance(out, np.ma.MaskedArray)
    assert list(out) == [1.0, 2.0]


def test_append_masked_string_column_records_missing(h5file: h5py.File) -> None:
    # Regression: the masked element used to reach FixedString.encode as a
    # MaskedConstant and be stringified to the literal characters "--".
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="s", dtype=FixedString(nbytes=4), fill_value=b"")],
    )
    t.append(
        {
            "s": np.ma.masked_array(
                np.array(["zz", "bb", "cc"], dtype=object), mask=[True, False, False]
            )
        }
    )
    assert list(t["s"].read()) == ["", "bb", "cc"]
    assert list(t["s"].is_missing()) == [True, False, False]


def test_append_masked_numeric_column_ignores_data_under_the_mask(
    h5file: h5py.File,
) -> None:
    # The payload under a mask is arbitrary — here 0.0, not the fill — so a
    # dtype fast path that trusts `.data` writes it as a real value.
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="x", dtype="f4", fill_value=np.float32(-999))],
    )
    t.append({"x": np.ma.masked_array([0.0, 1.0, 2.0], mask=[True, False, False])})
    assert list(t["x"].is_missing()) == [True, False, False]
    assert float(t["x"].dataset[0]) == -999.0


def test_append_masked_categorical_column_records_missing(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"), [ColumnSpec(name="k", categories=["a", "b"])]
    )
    t.append(
        {
            "k": np.ma.masked_array(
                np.array(["a", "b"], dtype=object), mask=[True, False]
            )
        }
    )
    assert list(t["k"].read()) == [None, "b"]
    assert list(t["k"].is_missing()) == [True, False]


def test_append_masked_boolean_column_rejected(h5file: h5py.File) -> None:
    # H5Col forbids a boolean column from declaring a fill, so it has no way to
    # record a missing row — a mask must be refused, not silently dropped.
    t = Table.create(h5file.create_group("t"), [ColumnSpec(name="f", dtype="bool")])
    with pytest.raises(SchemaError, match="cannot hold a missing"):
        t.append({"f": np.ma.masked_array([True, False], mask=[True, False])})
    assert t.nrows == 0
