"""Tests for inferring H5Col column specs from an Arrow table.

Arrow's model is wider than H5Col's, so the contract here is mostly about what
is *refused*: a type with no exact equivalent, a name HDF5 cannot store, and
metadata that would shadow the convention's own attributes. Nothing in this
module touches a file.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest

from h5col import (
    ColumnSpec,
    FixedString,
    ListColumnSpec,
    Table,
    bool_dtype,
    is_bool_dtype,
    specs_from_arrow,
)
from h5col.exceptions import ReservedNameError, SchemaError

pa = pytest.importorskip("pyarrow", reason="the arrow extra is not installed")


def _one(table: Any) -> Any:
    specs = specs_from_arrow(table)
    assert len(specs) == 1
    return specs[0]


def _field_table(name: str, arrow_type: Any, values: Any, metadata: Any = None) -> Any:
    field = pa.field(name, arrow_type, metadata=metadata)
    return pa.table([pa.array(values, type=arrow_type)], schema=pa.schema([field]))


# --------------------------------------------------------------------------- #
# The types that map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("arrow_type", "expected"),
    [
        (pa.int8(), "int8"),
        (pa.int16(), "int16"),
        (pa.int32(), "int32"),
        (pa.int64(), "int64"),
        (pa.uint8(), "uint8"),
        (pa.uint16(), "uint16"),
        (pa.uint32(), "uint32"),
        (pa.uint64(), "uint64"),
        (pa.float32(), "float32"),
        (pa.float64(), "float64"),
    ],
)
def test_primitive_types_keep_their_width(arrow_type: Any, expected: str) -> None:
    spec = _one(_field_table("v", arrow_type, [1, 2]))
    assert spec.resolved_dtype() == np.dtype(expected)


def test_boolean_becomes_the_h5col_boolean_not_a_plain_int8() -> None:
    # A H5Col boolean column is a specific one-byte enumeration; losing that
    # would leave an ordinary integer column that no longer reads as bool.
    spec = _one(_field_table("flag", pa.bool_(), [True, False]))
    assert is_bool_dtype(spec.resolved_dtype())
    assert spec.is_boolean


@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
def test_strings_are_sized_to_the_widest_encoded_value(arrow_type: Any) -> None:
    spec = _one(_field_table("s", arrow_type, ["a", "café", "日本語"]))
    assert isinstance(spec.dtype, FixedString)
    assert spec.dtype.nbytes == 9  # 日本語 is 9 UTF-8 bytes, not 3 characters


def test_empty_string_column_still_gets_a_width() -> None:
    spec = _one(_field_table("s", pa.string(), []))
    assert spec.dtype.nbytes >= 1


def test_nulls_do_not_count_towards_the_string_width() -> None:
    spec = _one(_field_table("s", pa.string(), ["ab", None]))
    assert spec.dtype.nbytes == 2


def test_dictionary_becomes_a_categorical() -> None:
    arr = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0], type=pa.int8()), pa.array(["manned", "automatic"])
    )
    spec = _one(pa.table({"kind": arr}))
    assert spec.is_categorical
    assert spec.categories == ["manned", "automatic"]


def test_dictionary_chunks_are_unified_before_the_categories_are_read() -> None:
    # Two chunks may carry different dictionaries, in which case code 0 means
    # two different things. Taking only the first chunk's labels would lose the
    # second's entirely.
    chunked = pa.chunked_array(
        [
            pa.DictionaryArray.from_arrays(
                pa.array([0, 1], type=pa.int8()), pa.array(["x", "y"])
            ),
            pa.DictionaryArray.from_arrays(
                pa.array([0], type=pa.int8()), pa.array(["p"])
            ),
        ]
    )
    spec = _one(pa.table({"k": chunked}))
    assert spec.categories == ["x", "y", "p"]


def test_arrow_ordered_flag_is_honoured() -> None:
    ty = pa.dictionary(pa.int8(), pa.string(), ordered=True)
    arr = pa.DictionaryArray.from_arrays(
        pa.array([0], type=pa.int8()), pa.array(["a"])
    ).cast(ty)
    assert _one(pa.table({"k": arr})).ordered is True


@pytest.mark.parametrize("outer", [pa.list_, pa.large_list])
def test_list_columns(outer: Any) -> None:
    spec = _one(_field_table("xs", outer(pa.float64()), [[1.0, 2.0], None]))
    assert isinstance(spec, ListColumnSpec)
    assert spec.nullable is True  # the null row needs a MASK to live in


def test_nested_lists_recurse() -> None:
    spec = _one(
        _field_table("nest", pa.large_list(pa.large_list(pa.int64())), [[[1], [2, 3]]])
    )
    assert isinstance(spec, ListColumnSpec)
    assert spec.values.values.resolved_dtype() == np.dtype("int64")


def test_list_of_strings_uses_string_values() -> None:
    from h5col import StringValuesSpec

    spec = _one(_field_table("tags", pa.large_list(pa.large_string()), [["a", "b"]]))
    assert isinstance(spec.values, StringValuesSpec)


# --------------------------------------------------------------------------- #
# The types that do not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "arrow_type"),
    [
        ("timestamp", pa.timestamp("us")),
        ("date32", pa.date32()),
        ("time64", pa.time64("us")),
        ("duration", pa.duration("s")),
        ("decimal128", pa.decimal128(10, 2)),
        ("struct", pa.struct([("a", pa.int8())])),
        ("map", pa.map_(pa.string(), pa.int8())),
        ("binary", pa.binary()),
        ("null", pa.null()),
    ],
)
def test_unrepresentable_types_are_refused_by_name(label: str, arrow_type: Any) -> None:
    table = pa.table({"c": pa.array([], type=arrow_type)})
    with pytest.raises(SchemaError, match="cannot be stored in H5Col"):
        specs_from_arrow(table)


def test_float16_is_refused() -> None:
    # H5Col has no recommended fill for float16, so a missing value would have
    # nowhere to go.
    with pytest.raises(SchemaError):
        specs_from_arrow(pa.table({"c": pa.array([], type=pa.float16())}))


def test_unsupported_list_element_type_is_refused() -> None:
    ty = pa.large_list(pa.timestamp("us"))
    with pytest.raises(SchemaError, match="list element type"):
        specs_from_arrow(pa.table({"c": pa.array([], type=ty)}))


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def test_duplicate_field_names_are_refused() -> None:
    # Arrow allows two fields of the same name; HDF5 links are unique.
    table = pa.table({"a": [1]}).append_column("a", pa.array([2]))
    with pytest.raises(SchemaError, match="more than once"):
        specs_from_arrow(table)


@pytest.mark.parametrize("name", ["a/b", ""])
def test_names_hdf5_cannot_store_are_refused(name: str) -> None:
    with pytest.raises(SchemaError):
        specs_from_arrow(pa.table({name: [1]}))


@pytest.mark.parametrize("name", ["CLASS", "NROWS", "VERSION", "CATEGORIES"])
def test_reserved_column_names_are_refused(name: str) -> None:
    with pytest.raises(ReservedNameError):
        specs_from_arrow(pa.table({name: [1]}))


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_h5col_metadata_becomes_annotations() -> None:
    spec = _one(
        _field_table(
            "v",
            pa.float64(),
            [1.0],
            metadata={
                "h5col.units": "degC",
                "h5col.description": "air temp",
                "h5col.valid_min": "-90.0",
                "h5col.valid_max": "60.0",
            },
        )
    )
    assert spec.units == "degC"
    assert spec.description == "air temp"
    assert spec.valid_min == np.float64(-90.0)
    assert spec.valid_max == np.float64(60.0)


def test_bounds_are_parsed_against_the_column_dtype() -> None:
    spec = _one(_field_table("v", pa.int32(), [1], metadata={"h5col.valid_min": "-5"}))
    assert spec.valid_min == np.int32(-5)
    assert np.asarray(spec.valid_min).dtype == np.dtype("int32")


def test_other_metadata_is_carried_as_producer_attributes() -> None:
    spec = _one(
        _field_table(
            "v",
            pa.float64(),
            [1.0],
            metadata={"instrument": "AWS-11", "h5col.units": "m"},
        )
    )
    assert spec.units == "m"
    assert spec.attributes == {"instrument": "AWS-11"}


@pytest.mark.parametrize("key", ["CLASS", "NROWS", "VERSION", "valid_min"])
def test_producer_metadata_may_not_shadow_a_reserved_attribute(key: str) -> None:
    with pytest.raises(ReservedNameError):
        specs_from_arrow(_field_table("v", pa.float64(), [1.0], metadata={key: "x"}))


@pytest.mark.parametrize("key", ["units", "units_vocabulary", "description", "ordered"])
def test_producer_metadata_may_not_shadow_an_annotation_h5col_writes(key: str) -> None:
    # These are not reserved *names* — a column may be called "units" — but an
    # attribute of that name is the one h5col.units writes, so free-form
    # metadata using it would overwrite the annotation.
    with pytest.raises(ReservedNameError):
        specs_from_arrow(_field_table("v", pa.float64(), [1.0], metadata={key: "x"}))


def test_unknown_h5col_metadata_key_is_refused() -> None:
    # A key under the h5col prefix claims a meaning this importer does not
    # know; guessing would be worse than saying so.
    with pytest.raises(SchemaError, match="unknown"):
        specs_from_arrow(
            _field_table("v", pa.float64(), [1.0], metadata={"h5col.bogus": "1"})
        )


def test_a_bound_on_a_column_with_no_scalar_dtype_is_refused() -> None:
    # np.dtype(None) is float64, so parsing a bound without a datatype would
    # quietly turn it into a float.
    with pytest.raises(SchemaError, match="cannot carry"):
        specs_from_arrow(
            _field_table("flag", pa.bool_(), [True], metadata={"h5col.valid_min": "0"})
        )


def test_ordered_is_refused_on_a_column_that_cannot_hold_it() -> None:
    with pytest.raises(SchemaError, match="cannot carry"):
        specs_from_arrow(
            _field_table("v", pa.float64(), [1.0], metadata={"h5col.ordered": "true"})
        )


# --------------------------------------------------------------------------- #
# What comes back is usable
# --------------------------------------------------------------------------- #
def test_inferred_specs_create_a_table(h5file: h5py.File) -> None:
    tbl = pa.table(
        {
            "station": pa.array(["KBOS", "KJFK"]),
            "t_air": pa.array([21.5, None]),
            "flag": pa.array([True, False]),
        }
    )
    specs = specs_from_arrow(tbl)
    table = Table.create(h5file.create_group("t"), specs)
    assert table.column_names == ["station", "t_air", "flag"]


def test_specs_are_adjustable_before_use(h5file: h5py.File) -> None:
    # The reason they are returned at all: chunking and filters have no Arrow
    # equivalent, so they can only be set by hand.
    specs = specs_from_arrow(pa.table({"v": pa.array([1.0, 2.0])}))
    specs[0].chunks = 4096
    table = Table.create(h5file.create_group("t"), specs)
    assert table["v"].dataset.chunks == (4096,)


def test_round_trip_from_a_h5col_table(h5file: h5py.File) -> None:
    source = Table.create(
        h5file.create_group("src"),
        [
            ColumnSpec(name="v", dtype="f8", fill_value=np.nan, units="degC"),
            ColumnSpec(name="k", categories=["a", "b"], ordered=True),
            ColumnSpec(name="f", dtype=bool_dtype()),
        ],
    )
    source.append({"v": [1.5, None], "k": ["a", None], "f": [True, False]})

    specs = specs_from_arrow(source.to_arrow())
    rebuilt = Table.create(h5file.create_group("dst"), specs)
    assert rebuilt.column_names == source.column_names
    assert rebuilt["v"].units == "degC"
    assert list(rebuilt["k"].categories) == ["a", "b"]
    assert rebuilt["k"].ordered is True
    assert rebuilt["f"].is_boolean
