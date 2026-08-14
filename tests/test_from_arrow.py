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
def test_list_columns_are_not_implemented_yet(h5file: h5py.File) -> None:
    source = pa.table({"xs": pa.array([[1.0, 2.0]], type=pa.large_list(pa.float64()))})
    with pytest.raises(SchemaError, match="not implemented yet"):
        Table.from_arrow(h5file.create_group("t"), source)


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
