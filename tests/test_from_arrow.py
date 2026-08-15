"""Tests for importing a pyarrow Table into H5Col.

The spec inference is covered in ``test_specs_from_arrow``; what is checked here
is that the rows actually written match the Arrow table they came from, nulls
included, and that the guards cannot be stepped around by supplying specs.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest

from h5col import ColumnSpec, FilterPipeline, Shuffle, Table, specs_from_arrow
from h5col.exceptions import SchemaError

pa = pytest.importorskip("pyarrow", reason="the arrow extra is not installed")


def _sample() -> Any:
    return pa.table(
        {
            "station": pa.array(["KBOS", "KJFK", None]),
            "t_air": pa.array([21.5, None, 23.1]),
            "count": pa.array([1, None, 3], type=pa.int32()),
            "flag": pa.array([True, False, True]),
            "kind": pa.DictionaryArray.from_arrays(
                pa.array([0, 1, None], type=pa.int8()),
                pa.array(["manned", "automatic"]),
            ),
        }
    )


# --------------------------------------------------------------------------- #
# The data that arrives
# --------------------------------------------------------------------------- #
def test_every_value_survives_the_round_trip(h5file: h5py.File) -> None:
    source = _sample()
    table = Table.from_arrow(h5file.create_group("t"), source)
    back = table.to_arrow()
    for name in source.schema.names:
        assert back.column(name).to_pylist() == source.column(name).to_pylist(), name


def test_the_imported_table_is_conformant(h5file: h5py.File) -> None:
    Table.from_arrow(h5file.create_group("t"), _sample()).validate(deep=True)


def test_nulls_become_missing_rows(h5file: h5py.File) -> None:
    table = Table.from_arrow(h5file.create_group("t"), _sample())
    assert list(table["t_air"].is_missing()) == [False, True, False]
    assert list(table["station"].is_missing()) == [False, False, True]
    assert list(table["kind"].is_missing()) == [False, False, True]


def test_an_integer_column_with_nulls_keeps_its_width(h5file: h5py.File) -> None:
    # to_numpy on a nullable integer column upcasts to float64 and turns the
    # nulls into NaN, which would change the datatype of every value.
    table = Table.from_arrow(h5file.create_group("t"), _sample())
    assert table["count"].dtype == np.dtype("int32")
    assert table["count"].read().tolist() == [1, None, 3]


def test_table_kwargs_reach_create(h5file: h5py.File) -> None:
    table = Table.from_arrow(
        h5file.create_group("t"), _sample(), title="imported", description="from arrow"
    )
    assert table.title == "imported"
    assert table.description == "from arrow"


def test_an_empty_table_imports(h5file: h5py.File) -> None:
    empty = pa.table({"v": pa.array([], type=pa.int32())})
    table = Table.from_arrow(h5file.create_group("t"), empty)
    assert table.nrows == 0


def test_many_batches_are_all_written(h5file: h5py.File) -> None:
    # A column arrives as a ChunkedArray; every chunk has to reach the file.
    chunked = pa.chunked_array(
        [pa.array([1, 2], type=pa.int32()), pa.array([3], type=pa.int32())]
    )
    table = Table.from_arrow(h5file.create_group("t"), pa.table({"v": chunked}))
    assert table.nrows == 3
    assert table["v"].read().tolist() == [1, 2, 3]


def test_columns_with_different_chunk_layouts_stay_aligned(h5file: h5py.File) -> None:
    # Two columns may be chunked differently; a row must not be assembled from
    # mismatched pieces.
    source = pa.Table.from_arrays(
        [
            pa.chunked_array(
                [pa.array([1, 2], type=pa.int32()), pa.array([3], type=pa.int32())]
            ),
            pa.chunked_array([pa.array(["a"]), pa.array(["b", "c"])]),
        ],
        names=["n", "s"],
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table["n"].read().tolist() == [1, 2, 3]
    assert table["s"].read().tolist() == ["a", "b", "c"]


def test_chunks_with_different_dictionaries_import_correctly(h5file: h5py.File) -> None:
    # Two chunks may carry different dictionaries, in which case code 0 stands
    # for two different labels. The import reads labels rather than codes, and
    # the category set comes from the unified dictionary, so both chunks land
    # on the right labels.
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
    table = Table.from_arrow(h5file.create_group("t"), pa.table({"k": chunked}))
    assert table["k"].read().tolist() == ["x", "y", "p"]


# --------------------------------------------------------------------------- #
# Supplied specs
# --------------------------------------------------------------------------- #
def test_supplied_specs_set_what_arrow_cannot_express(h5file: h5py.File) -> None:
    source = pa.table({"v": pa.array([1.0, 2.0, 3.0])})
    specs = specs_from_arrow(source)
    specs[0].chunks = 4096
    specs[0].filters = FilterPipeline([Shuffle()])
    table = Table.from_arrow(h5file.create_group("t"), source, specs=specs)
    assert table["v"].dataset.chunks == (4096,)


def test_supplied_specs_are_not_mutated(h5file: h5py.File) -> None:
    source = pa.table({"v": pa.array([1.0, None])})
    specs = [ColumnSpec(name="v", dtype="f8")]
    assert specs[0].fill_value is None
    Table.from_arrow(h5file.create_group("t"), source, specs=specs)
    assert specs[0].fill_value is None  # the caller's object is left alone


def test_supplied_specs_cannot_skip_the_fill_check(h5file: h5py.File) -> None:
    # The collision guard is the one check that must not be optional.
    source = pa.table({"v": pa.array([7, 1, None], type=pa.int32())})
    specs = [ColumnSpec(name="v", dtype="int32", fill_value=7)]
    with pytest.raises(SchemaError, match="occurs in the data"):
        Table.from_arrow(h5file.create_group("t"), source, specs=specs)


def test_specs_must_name_exactly_the_columns(h5file: h5py.File) -> None:
    source = pa.table({"a": pa.array([1]), "b": pa.array([2])})
    with pytest.raises(SchemaError, match="exactly the table's columns"):
        Table.from_arrow(
            h5file.create_group("t"), source, specs=[ColumnSpec(name="a", dtype="i8")]
        )


def test_a_spec_for_an_unknown_column_is_refused(h5file: h5py.File) -> None:
    source = pa.table({"a": pa.array([1])})
    with pytest.raises(SchemaError, match="exactly the table's columns"):
        Table.from_arrow(
            h5file.create_group("t"),
            source,
            specs=[ColumnSpec(name="a", dtype="i8"), ColumnSpec(name="z", dtype="i8")],
        )


# --------------------------------------------------------------------------- #
# What is refused
# --------------------------------------------------------------------------- #
def test_an_unsupported_type_is_refused_before_the_group_is_touched(
    h5file: h5py.File,
) -> None:
    source = pa.table({"t": pa.array([0], type=pa.timestamp("us"))})
    group = h5file.create_group("t")
    with pytest.raises(SchemaError):
        Table.from_arrow(group, source)
    assert not Table.is_table_group(group)


def test_a_boolean_with_nulls_is_refused(h5file: h5py.File) -> None:
    source = pa.table({"flag": pa.array([True, None])})
    with pytest.raises(SchemaError, match="nowhere to be stored"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_a_boolean_only_table_imports_in_a_fresh_interpreter() -> None:
    """A table of only boolean columns imports without any other path running.

    Booleans take neither the fill check nor the numeric conversion, so this is
    the one import that touches almost none of the machinery — and in a shared
    interpreter it borrows state from whatever ran before it. A subprocess is
    the only way to see it on its own.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import tempfile, os
        import h5py, pyarrow as pa
        from h5col import Table

        path = os.path.join(tempfile.mkdtemp(), "b.h5")
        with h5py.File(path, "w") as f:
            t = Table.from_arrow(
                f.create_group("t"), pa.table({"flag": pa.array([True, False])})
            )
            assert t.nrows == 2
            assert t["flag"].read().tolist() == [True, False]
        print("ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# List columns
#
# A list column marks a null row with a MASK but a null *element* with a fill
# value, so the collision hazard of a scalar column exists again at every level
# of nesting — and once more for each level below that.
# --------------------------------------------------------------------------- #
def _list_sample() -> Any:
    return pa.table(
        {
            "i": pa.array([1, 2, 3]),
            "xs": pa.array([[1.0, None], None, []], type=pa.large_list(pa.float64())),
            "tags": pa.array(
                [["red", None], [], ["blue"]], type=pa.large_list(pa.large_string())
            ),
            "nest": pa.array(
                [[[1.0], [2.0, 3.0]], None, [[4.0]]],
                type=pa.large_list(pa.large_list(pa.float64())),
            ),
        }
    )


def test_list_columns_round_trip(h5file: h5py.File) -> None:
    source = _list_sample()
    table = Table.from_arrow(h5file.create_group("t"), source)
    back = table.to_arrow()
    for name in source.schema.names:
        assert back.column(name).to_pylist() == source.column(name).to_pylist(), name


def test_an_imported_list_table_is_conformant(h5file: h5py.File) -> None:
    Table.from_arrow(h5file.create_group("t"), _list_sample()).validate(deep=True)


def test_a_null_row_stays_distinct_from_an_empty_one(h5file: h5py.File) -> None:
    table = Table.from_arrow(h5file.create_group("t"), _list_sample())
    rows = table["xs"].read()
    assert rows[1] is None  # the null row
    assert rows[2] == []  # the empty row, which is a value


@pytest.mark.parametrize("name", ["xs", "nest"])
def test_nullable_levels_follow_the_data(h5file: h5py.File, name: str) -> None:
    table = Table.from_arrow(h5file.create_group("t"), _list_sample())
    assert table[name].nullable is True


def test_no_mask_is_created_when_nothing_is_null(h5file: h5py.File) -> None:
    # A MASK is a dataset per level; one that marks nothing is pure overhead.
    source = pa.table(
        {"xs": pa.array([[1.0, 2.0], [3.0]], type=pa.large_list(pa.float64()))}
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table["xs"].nullable is False
    assert "MASK" not in table["xs"].group


def test_a_fill_among_the_list_elements_is_refused(h5file: h5py.File) -> None:
    from h5col import recommended_fill

    collides = int(recommended_fill(np.dtype("int8")))
    source = pa.table(
        {"xs": pa.array([[1, collides], [2]], type=pa.large_list(pa.int8()))}
    )
    with pytest.raises(SchemaError, match="occurs among the list elements"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_the_leaf_check_reaches_through_nesting(h5file: h5py.File) -> None:
    from h5col import recommended_fill

    collides = int(recommended_fill(np.dtype("int8")))
    source = pa.table(
        {"n": pa.array([[[1, collides]]], type=pa.large_list(pa.large_list(pa.int8())))}
    )
    with pytest.raises(SchemaError, match="occurs among the list elements"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_a_boolean_leaf_with_a_null_element_is_refused(h5file: h5py.File) -> None:
    source = pa.table(
        {"flags": pa.array([[True, None]], type=pa.large_list(pa.bool_()))}
    )
    with pytest.raises(SchemaError, match="nowhere to be stored"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_a_boolean_leaf_without_nulls_is_fine(h5file: h5py.File) -> None:
    source = pa.table(
        {"flags": pa.array([[True, False], []], type=pa.large_list(pa.bool_()))}
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table["flags"].read() == [[True, False], []]


def test_null_string_elements_need_no_fill(h5file: h5py.File) -> None:
    # STRING_VALUES marks a null element with a MASK bit, so there is no value
    # that could collide with the data — including the empty string, which is
    # a real value here rather than a missing marker.
    source = pa.table(
        {"tags": pa.array([["", None, "x"]], type=pa.large_list(pa.large_string()))}
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table["tags"].read() == [["", None, "x"]]


def test_list_column_annotations_survive(h5file: h5py.File) -> None:
    field = pa.field(
        "xs",
        pa.large_list(pa.float64()),
        metadata={"h5col.units": "m", "h5col.description": "depths"},
    )
    source = pa.table(
        [pa.array([[1.0]], type=pa.large_list(pa.float64()))], schema=pa.schema([field])
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table["xs"].units == "m"
    assert table["xs"].description == "depths"


def test_a_mixed_table_of_scalar_and_list_columns(h5file: h5py.File) -> None:
    source = _list_sample()
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table.column_names == ["i", "xs", "tags", "nest"]
    assert table["i"].read().tolist() == [1, 2, 3]


def test_inner_levels_get_no_mask_when_they_hold_no_nulls(h5file: h5py.File) -> None:
    # Nullability is decided per level from that level's own data. A nested
    # list whose inner lists are never null needs no MASK there, even though
    # the rows above it may.
    source = pa.table(
        {
            "nest": pa.array(
                [[[1.0], [2.0]], None],
                type=pa.large_list(pa.large_list(pa.float64())),
            )
        }
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    group = table["nest"].group
    assert "MASK" in group  # the null row at the top
    assert "MASK" not in group["VALUES"]  # but no null inner list


def test_string_values_get_no_mask_when_no_element_is_null(h5file: h5py.File) -> None:
    source = pa.table(
        {"tags": pa.array([["a", "b"], []], type=pa.large_list(pa.large_string()))}
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert "MASK" not in table["tags"].group["VALUES"]


def test_string_values_get_a_mask_when_an_element_is_null(h5file: h5py.File) -> None:
    source = pa.table(
        {"tags": pa.array([["a", None]], type=pa.large_list(pa.large_string()))}
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert "MASK" in table["tags"].group["VALUES"]
    assert table["tags"].read() == [["a", None]]


# --------------------------------------------------------------------------- #
# Opaque columns: raw bytes of a fixed width
# --------------------------------------------------------------------------- #
def _digests(with_null: bool = False) -> Any:
    values: list[Any] = [b"\x01" * 8, b"\x02" * 8]
    if with_null:
        values.append(None)
    return pa.table({"digest": pa.array(values, type=pa.binary(8))})


def test_fixed_size_binary_becomes_an_opaque_column(h5file: h5py.File) -> None:
    table = Table.from_arrow(h5file.create_group("t"), _digests())
    table.validate(deep=True)
    assert table["digest"].dtype == np.dtype("V8")
    assert [bytes(v) for v in table["digest"].read()] == [b"\x01" * 8, b"\x02" * 8]
    assert table.to_arrow().column("digest").to_pylist() == [b"\x01" * 8, b"\x02" * 8]


def test_an_opaque_column_gets_the_recommended_byte_pattern(h5file: h5py.File) -> None:
    # No byte string is out of range for opaque data, so the fill is one chosen
    # to be unlikely rather than impossible: ASCII FILL, then rising bytes.
    table = Table.from_arrow(h5file.create_group("t"), _digests())
    assert bytes(table["digest"].fill_value) == b"FILL\x01\x02\x03\x04"


def test_opaque_nulls_survive_the_round_trip(h5file: h5py.File) -> None:
    table = Table.from_arrow(h5file.create_group("t"), _digests(with_null=True))
    assert list(table["digest"].is_missing()) == [False, False, True]
    assert table.to_arrow().column("digest").to_pylist() == [
        b"\x01" * 8,
        b"\x02" * 8,
        None,
    ]


def test_data_holding_the_fill_pattern_is_refused(h5file: h5py.File) -> None:
    # Unlikely is not impossible, so the collision check applies here as it
    # does to every other datatype.
    source = pa.table(
        {"digest": pa.array([b"FILL\x01\x02\x03\x04", b"\x02" * 8], type=pa.binary(8))}
    )
    with pytest.raises(SchemaError, match="occurs in the data"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_a_fill_can_be_chosen_for_an_opaque_column(h5file: h5py.File) -> None:
    source = pa.table(
        {"digest": pa.array([b"FILL\x01\x02\x03\x04", None], type=pa.binary(8))}
    )
    specs = specs_from_arrow(_digests())
    specs[0].fill_value = np.void(b"\xff" * 8)
    table = Table.from_arrow(h5file.create_group("t"), source, specs=specs)
    assert list(table["digest"].is_missing()) == [False, True]
    assert table.to_arrow().column("digest").to_pylist() == [
        b"FILL\x01\x02\x03\x04",
        None,
    ]


def test_a_list_of_fixed_size_binary_imports(h5file: h5py.File) -> None:
    source = pa.table(
        {
            "digests": pa.array(
                [[b"\x01" * 4], [], [b"\x02" * 4, b"\x03" * 4]],
                type=pa.large_list(pa.binary(4)),
            )
        }
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    table.validate(deep=True)
    assert table.to_arrow().column("digests").to_pylist() == [
        [b"\x01" * 4],
        [],
        [b"\x02" * 4, b"\x03" * 4],
    ]


# --------------------------------------------------------------------------- #
# When the recommended fill value is already in the data
#
# Refusing is right: the column needs a fill and this one would make real values
# read as missing. What the caller then needs is a way to name another.
# --------------------------------------------------------------------------- #
def _byte_levels() -> Any:
    # 255 is the recommended uint8 fill and an entirely ordinary byte value.
    return pa.table({"level": pa.array([1, 255, 3], type=pa.uint8())})


def test_inference_returns_specs_even_when_the_fill_collides() -> None:
    # The remedy is to set a fill value on the specs, so getting the specs
    # cannot itself be what fails.
    specs = specs_from_arrow(_byte_levels())
    assert specs[0].name == "level"
    assert specs[0].fill_value is None  # unset means the recommended one


def test_writing_still_refuses_the_collision(h5file: h5py.File) -> None:
    with pytest.raises(SchemaError, match="occurs in the data"):
        Table.from_arrow(h5file.create_group("t"), _byte_levels())


def test_the_refusal_names_a_fill_value_that_would_work(h5file: h5py.File) -> None:
    with pytest.raises(SchemaError, match=r"254.*does not occur"):
        Table.from_arrow(h5file.create_group("t"), _byte_levels())


def test_the_suggested_value_actually_works(h5file: h5py.File) -> None:
    # A suggestion that did not import would be worse than none at all.
    specs = specs_from_arrow(_byte_levels())
    specs[0].fill_value = np.uint8(254)
    table = Table.from_arrow(h5file.create_group("t"), _byte_levels(), specs=specs)
    table.validate(deep=True)
    assert table["level"].read().tolist() == [1, 255, 3]
    assert not table["level"].is_missing().any()


def test_a_column_using_its_whole_datatype_is_told_to_widen(h5file: h5py.File) -> None:
    full = pa.table({"b": pa.array(list(range(256)), type=pa.uint8())})
    with pytest.raises(SchemaError, match="widen the datatype"):
        Table.from_arrow(h5file.create_group("t"), full)


def test_a_list_leaf_gets_a_suggestion_too(h5file: h5py.File) -> None:
    # Where this bites hardest: list elements are often small integers using
    # their whole range, so a list of bytes containing 255 is unremarkable.
    source = pa.table({"xs": pa.array([[1, 255], [3]], type=pa.large_list(pa.uint8()))})
    with pytest.raises(SchemaError, match=r"254.*does not occur"):
        Table.from_arrow(h5file.create_group("t"), source)

    specs = specs_from_arrow(source)
    specs[0].values.fill_value = np.uint8(254)
    table = Table.from_arrow(h5file.create_group("ok"), source, specs=specs)
    table.validate(deep=True)
    assert table["xs"].read() == [[1, 255], [3]]


def test_a_signed_suggestion_skips_the_types_minimum(h5file: h5py.File) -> None:
    # H5Col recommends INT_MIN + 1 and leaves INT_MIN alone, keeping a margin
    # against operations that land on it. A suggestion should respect that.
    source = pa.table({"v": pa.array([1, -127], type=pa.int8())})
    with pytest.raises(SchemaError, match=r"-126.*does not occur"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_no_suggestion_is_offered_for_a_string_column(h5file: h5py.File) -> None:
    # The candidate space for text is not one to walk; the caller chooses.
    source = pa.table({"s": pa.array(["a", ""], type=pa.large_string())})
    with pytest.raises(SchemaError, match="a fill_value the data does not contain"):
        Table.from_arrow(h5file.create_group("t"), source)


def test_a_boolean_with_nulls_is_refused_when_written(h5file: h5py.File) -> None:
    # Inference now returns specs for it — changing the dtype is a real remedy
    # — but writing it still refuses.
    source = pa.table({"flag": pa.array([True, None])})
    assert specs_from_arrow(source)[0].is_boolean
    with pytest.raises(SchemaError, match="nowhere to be stored"):
        Table.from_arrow(h5file.create_group("t"), source)
