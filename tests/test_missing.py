"""Tests for fill values and the canonical missing-value test (h5col.missing)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import (
    Column,
    ColumnSpec,
    LeafValuesSpec,
    ListColumnSpec,
    Table,
    field,
)
from h5col.booleans import bool_dtype
from h5col.exceptions import FillValueError, SchemaError
from h5col.missing import (
    is_missing,
    masked_to_none,
    recommended_fill,
    validate_fill_outside_range,
)
from h5col.opaque import is_opaque_dtype, opaque_fill_bytes
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

    assert list(table["x"].read(masked=False)) == [1, -1, 3]
    assert table["x"].read().tolist() == [1, None, 3]
    assert list(table["x"].is_missing()) == [False, True, False]


def test_append_none_writes_the_nan_fill(h5file: h5py.File) -> None:
    table = _one_column_table(
        h5file, ColumnSpec(name="x", dtype="float64", fill_value=np.nan)
    )
    table.append({"x": [1.5, None, 3.5]})

    values = table["x"].read(masked=False)
    assert np.isnan(values[1])
    assert list(table["x"].is_missing()) == [False, True, False]


def test_append_none_writes_the_recommended_sentinel_fill(h5file: h5py.File) -> None:
    # No explicit fill: None must become the column's own recommended sentinel,
    # not NaN.
    table = _one_column_table(h5file, ColumnSpec(name="x", dtype="float64"))
    table.append({"x": [1.5, None]})

    values = table["x"].read(masked=False)
    assert values[1] == recommended_fill(np.dtype("float64"))
    assert not np.isnan(values[1])
    assert list(table["x"].is_missing()) == [False, True]


def test_append_none_writes_the_string_fill_not_the_word_none(
    h5file: h5py.File,
) -> None:
    table = _one_column_table(h5file, ColumnSpec(name="x", dtype=FixedString(nbytes=8)))
    table.append({"x": ["ab", None, "cd"]})

    values = table["x"].read(masked=False)
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
    assert list(t["s"].read(masked=False)) == ["", "bb", "cc"]
    assert t["s"].read().tolist() == [None, "bb", "cc"]
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
    assert t["k"].read().tolist() == [None, "b"]
    assert list(t["k"].is_missing()) == [True, False]


def test_append_masked_boolean_column_rejected(h5file: h5py.File) -> None:
    # H5Col forbids a boolean column from declaring a fill, so it has no way to
    # record a missing row — a mask must be refused, not silently dropped.
    t = Table.create(h5file.create_group("t"), [ColumnSpec(name="f", dtype="bool")])
    with pytest.raises(SchemaError, match="cannot hold a missing"):
        t.append({"f": np.ma.masked_array([True, False], mask=[True, False])})
    assert t.nrows == 0


# --------------------------------------------------------------------------- #
# Masked output: `read()` carries the missing rows in a mask
# --------------------------------------------------------------------------- #
def _mixed_table(h5file: h5py.File) -> Table:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="num", dtype="f4", fill_value=np.float32(-999)),
            ColumnSpec(name="i8", dtype="int8", fill_value=np.int8(-127)),
            ColumnSpec(name="s", dtype=FixedString(nbytes=8)),
            ColumnSpec(name="cat", categories=["a", "b"]),
            ColumnSpec(name="flag", dtype="bool"),
            ColumnSpec(name="full", dtype="f4", fill_value=None),
        ],
    )
    t.append(
        {
            "num": [1.0, None, 3.0],
            "i8": [1, None, 3],
            "s": ["ab", None, "cd"],
            "cat": ["a", None, "b"],
            "flag": [True, False, True],
            "full": [1.0, 2.0, 3.0],
        }
    )
    return t


def test_masked_is_the_default_for_every_scalar_column(h5file: h5py.File) -> None:
    for name, values in _mixed_table(h5file).read().items():
        assert isinstance(values, np.ma.MaskedArray), name


def test_uniform_even_where_missing_is_impossible(h5file: h5py.File) -> None:
    # H5Col forbids a boolean column from declaring a fill, so it can never
    # have a missing row. It still comes back masked, all-False, so generic
    # code over the dict never has to branch on the column.
    col = _mixed_table(h5file)["flag"]
    assert col.fill_value is None
    assert isinstance(col.read(), np.ma.MaskedArray)
    assert not col.read().mask.any()


def test_column_with_no_declared_fill_reads_all_present(h5file: h5py.File) -> None:
    # h5col's own writer always sets a fill -- ColumnSpec(fill_value=None) takes
    # the recommended sentinel -- so a fill-less non-boolean column only turns
    # up in a file from another producer. Nothing is missing when nothing
    # declares what missing looks like.
    t = _mixed_table(h5file)
    assert t["full"].fill_value is not None  # the recommended sentinel, not None
    ds = t.group.create_dataset("foreign", data=np.arange(3, dtype="f4"))
    col = Column(ds, t)
    assert col.fill_value is None
    assert not col.read().mask.any()
    assert col.read().tolist() == [0.0, 1.0, 2.0]


def test_mask_is_materialised_not_nomask(h5file: h5py.File) -> None:
    # `nomask` is a scalar; `.mask[i]` on it raises IndexError. Always handing
    # back a real array keeps indexing the mask safe on every column.
    m = _mixed_table(h5file)["flag"].read().mask
    assert isinstance(m, np.ndarray) and m.shape == (3,)
    assert m[0] is np.False_ or m[0] == False  # noqa: E712 - indexable at all


def test_filled_reproduces_the_unmasked_read(h5file: h5py.File) -> None:
    t = _mixed_table(h5file)
    for name in ("num", "i8", "s"):
        col = t[name]
        assert list(col.read().filled()) == list(col.read(masked=False))


def test_fill_value_is_the_columns_sentinel_not_numpys(h5file: h5py.File) -> None:
    t = _mixed_table(h5file)
    # NumPy would default to 999999 for int8 -- which wraps to 63 -- and the
    # string "N/A" for a string column.
    assert t["i8"].read().fill_value == np.int8(-127)
    assert t["num"].read().fill_value == np.float32(-999)
    assert t["s"].read().fill_value == ""


def test_masked_false_is_the_previous_behaviour(h5file: h5py.File) -> None:
    t = _mixed_table(h5file)
    plain = t["num"].read(masked=False)
    assert not isinstance(plain, np.ma.MaskedArray)
    assert list(plain) == [1.0, -999.0, 3.0]


def test_mask_matches_is_missing_on_every_column(h5file: h5py.File) -> None:
    t = _mixed_table(h5file)
    for name in t.column_names:
        col = t[name]
        np.testing.assert_array_equal(col.read().mask, col.is_missing(), err_msg=name)


def test_read_rows_is_masked_too(h5file: h5py.File) -> None:
    t = _mixed_table(h5file)
    got = t["num"].read_rows([2, 1, 1])
    assert isinstance(got, np.ma.MaskedArray)
    assert got.tolist() == [3.0, None, None]
    assert list(t["num"].read_rows([2, 1], masked=False)) == [3.0, -999.0]


def test_selection_read_is_masked_on_both_paths(h5file: h5py.File) -> None:
    t = _mixed_table(h5file)
    for masked, kind in ((True, np.ma.MaskedArray), (False, np.ndarray)):
        out = t.select(field("num") != 1.0).read(["num", "s"], masked=masked)
        for name, values in out.items():
            assert isinstance(values, kind), (name, masked)


def test_list_columns_are_untouched_by_masked(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="i", dtype="i8"),
            ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="f8"), nullable=True),
        ],
    )
    t.append({"i": [1, 2], "xs": [[1.0], None]})
    for masked in (True, False):
        out = t.read(masked=masked)
        assert out["xs"] == [[1.0], None]
        assert isinstance(out["i"], np.ma.MaskedArray if masked else np.ndarray)


def test_filled_reproduces_unmasked_read_for_categoricals(h5file: h5py.File) -> None:
    # A category whose label collides with NumPy's own string sentinel: the
    # missing row and the genuine "N/A" row must stay distinguishable.
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="c", categories=["red", "N/A", "blue"])],
    )
    t.append({"c": ["red", None, "N/A", "blue"]})
    col = t["c"]
    assert col.read().filled().tolist() == col.read(masked=False).tolist()
    assert col.read().filled().tolist() == ["red", None, "N/A", "blue"]


def test_filled_keeps_numeric_categories_numeric(h5file: h5py.File) -> None:
    # NumPy's default fill for an object array is the string "?", which would
    # splice a str into a column of integer labels.
    t = Table.create(
        h5file.create_group("t"), [ColumnSpec(name="n", categories=[10, 20, 30])]
    )
    t.append({"n": [10, None, 30]})
    assert t["n"].read().filled().tolist() == [10, None, 30]


def test_list_column_read_accepts_and_ignores_masked(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="i", dtype="i8"),
            ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f8")),
        ],
    )
    t.append({"i": [1], "r": [[1.0]]})
    for masked in (True, False):
        # Uniform across the table: a caller may pass the keyword to any column.
        assert t["r"].read(masked=masked) == [[1.0]]
        t["i"].read(masked=masked)


def test_undecodable_fill_does_not_abort_the_read(h5file: h5py.File) -> None:
    # A fill only another producer could write. Setting fill_value is a
    # convenience; it must never cost the caller the column.
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="s", dtype=FixedString(nbytes=8), fill_value=b"\xff\xfe")],
    )
    t.append({"s": ["ab", None, "ef"]})
    got = t["s"].read()
    assert got.mask.tolist() == [False, True, False]
    assert got[0] == "ab"


def test_read_then_append_preserves_missing_rows(h5file: h5py.File) -> None:
    spec = [
        ColumnSpec(name="x", dtype="f4", fill_value=np.float32(-999)),
        ColumnSpec(name="s", dtype=FixedString(nbytes=4)),
        ColumnSpec(name="c", categories=["a", "b"]),
    ]
    a = Table.create(h5file.create_group("a"), spec)
    a.append({"x": [1.0, None, 3.0], "s": ["p", None, "q"], "c": ["a", None, "b"]})
    b = Table.create(h5file.create_group("b"), spec)
    b.append(a.read())  # masked arrays straight back in
    for name in ("x", "s", "c"):
        np.testing.assert_array_equal(
            a[name].is_missing(), b[name].is_missing(), err_msg=name
        )
        assert a[name].read().tolist() == b[name].read().tolist()


# --------------------------------------------------------------------------- #
# The opaque fill pattern
# --------------------------------------------------------------------------- #
def test_the_opaque_fill_is_the_marker_then_rising_bytes() -> None:
    assert opaque_fill_bytes(8) == b"FILL\x01\x02\x03\x04"
    assert opaque_fill_bytes(4) == b"FILL"
    assert opaque_fill_bytes(6) == b"FILL\x01\x02"


def test_a_narrow_opaque_column_gets_what_fits_of_the_marker() -> None:
    assert opaque_fill_bytes(1) == b"F"
    assert opaque_fill_bytes(2) == b"FI"
    assert all(len(opaque_fill_bytes(n)) == n for n in range(1, 40))


def test_the_rising_tail_wraps_through_zero() -> None:
    # Byte i of the tail is (i + 1) % 256, so 0xFF is followed by 0x00 and the
    # count starts again.
    wide = opaque_fill_bytes(4 + 258)
    tail = wide[4:]
    assert tail[254] == 0xFF
    assert tail[255] == 0x00
    assert tail[256] == 0x01


def test_the_pattern_is_neither_all_zeros_nor_all_ones() -> None:
    # Those two are what padding, erased flash and uninitialized memory leave
    # behind, which is exactly why the fill must not be either.
    for n in (1, 4, 8, 16, 64):
        value = opaque_fill_bytes(n)
        assert value != bytes(n)
        assert value != b"\xff" * n


def test_recommended_fill_uses_the_pattern_for_opaque() -> None:
    for n in (1, 4, 8, 32):
        fill = recommended_fill(np.dtype(f"V{n}"))
        assert bytes(fill) == opaque_fill_bytes(n)


def test_a_compound_dtype_is_not_opaque() -> None:
    # Compound and sub-array datatypes share NumPy's V kind and have no
    # recommended fill of their own.
    assert not is_opaque_dtype(np.dtype([("a", "i4")]))
    assert not is_opaque_dtype(np.dtype(("f8", (3,))))
    assert is_opaque_dtype(np.dtype("V8"))
    with pytest.raises(FillValueError):
        recommended_fill(np.dtype([("a", "i4")]))
