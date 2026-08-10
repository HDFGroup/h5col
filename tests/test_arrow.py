"""Tests for the Arrow export (h5col.arrow)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import (
    ColumnSpec,
    FixedString,
    LeafValuesSpec,
    ListColumnSpec,
    NestedListSpec,
    StringValuesSpec,
    Table,
    field,
)

pa = pytest.importorskip("pyarrow", reason="the arrow extra is not installed")


def _table(h5file: h5py.File) -> Table:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="station", dtype=FixedString(nbytes=8), description="id"),
            ColumnSpec(
                name="t_air",
                dtype="f4",
                fill_value=np.float32(-999),
                units="degC",
                valid_min=np.float32(-80),
                valid_max=np.float32(60),
            ),
            ColumnSpec(name="kind", categories=["manned", "automatic"], ordered=False),
            ColumnSpec(name="ok", dtype="bool"),
            ColumnSpec(name="n", categories=[10, 20]),
        ],
    )
    t.append(
        {
            "station": ["KBOS", "KJFK", None],
            "t_air": [21.5, None, 23.1],
            "kind": ["manned", None, "automatic"],
            "ok": [True, False, True],
            "n": [10, None, 20],
        }
    )
    return t


# --------------------------------------------------------------------------- #
# Type mapping
# --------------------------------------------------------------------------- #
def test_column_types(h5file: h5py.File) -> None:
    tb = _table(h5file).to_arrow()
    assert tb.schema.field("station").type == pa.large_string()
    assert tb.schema.field("t_air").type == pa.float32()
    assert tb.schema.field("ok").type == pa.bool_()
    kind = tb.schema.field("kind").type
    assert pa.types.is_dictionary(kind)
    assert kind.index_type == pa.int8() and kind.value_type == pa.string()


def test_missing_rows_become_real_nulls_not_the_fill(h5file: h5py.File) -> None:
    tb = _table(h5file).to_arrow()
    assert {n: tb[n].null_count for n in tb.column_names} == {
        "station": 1,
        "t_air": 1,
        "kind": 1,
        "ok": 0,
        "n": 1,
    }
    # The sentinel must appear nowhere in the exported values.
    assert -999.0 not in [v for v in tb["t_air"].to_pylist() if v is not None]
    assert tb["t_air"].to_pylist()[1] is None


def test_categorical_keeps_codes_and_labels(h5file: h5py.File) -> None:
    col = _table(h5file)["kind"]
    arr = col.to_arrow()
    assert arr.dictionary.to_pylist() == ["manned", "automatic"]
    assert arr.indices.to_pylist() == [0, None, 1]
    assert arr.to_pylist() == ["manned", None, "automatic"]


def test_categorical_with_numeric_labels(h5file: h5py.File) -> None:
    arr = _table(h5file)["n"].to_arrow()
    assert arr.dictionary.to_pylist() == [10, 20]
    assert arr.to_pylist() == [10, None, 20]


def test_boolean_column(h5file: h5py.File) -> None:
    assert _table(h5file)["ok"].to_arrow().to_pylist() == [True, False, True]


# --------------------------------------------------------------------------- #
# Strings: the one scalar kind that is a real conversion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    ["", "a", "abcdefgh", "héllo", "a\x00b", "ab\x00", "\x00", "日本", "x" * 12],
)
def test_string_export_matches_the_decode_oracle(value: str) -> None:
    fs = FixedString(nbytes=12)
    raw = np.asarray(fs.encode([value]))
    from h5col.arrow import string_array

    got = string_array(raw, 12, np.zeros(1, dtype=bool)).to_pylist()
    assert got == list(fs.decode(raw))


def test_string_column_roundtrips_through_the_table(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="s", dtype=FixedString(nbytes=12))],
    )
    values = ["a", "héllo", "日本", "x" * 12]
    t.append({"s": values})
    assert t["s"].to_arrow().to_pylist() == values


# --------------------------------------------------------------------------- #
# Attributes ride along as field metadata
# --------------------------------------------------------------------------- #
def test_attributes_become_field_metadata(h5file: h5py.File) -> None:
    tb = _table(h5file).to_arrow()

    def meta(name: str) -> dict[str, str]:
        m = tb.schema.field(name).metadata or {}
        return {k.decode(): v.decode() for k, v in m.items()}

    assert meta("t_air") == {
        "h5col.units": "degC",
        "h5col.valid_min": "-80.0",
        "h5col.valid_max": "60.0",
    }
    assert meta("station") == {"h5col.description": "id"}
    assert meta("kind") == {"h5col.ordered": "false"}
    assert meta("ok") == {}


def test_metadata_survives_a_parquet_round_trip(h5file: h5py.File) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    tb = _table(h5file).to_arrow()
    buf = pa.BufferOutputStream()
    pq.write_table(tb, buf)
    back = pq.read_table(pa.BufferReader(buf.getvalue()))
    assert back.schema.field("t_air").metadata[b"h5col.units"] == b"degC"
    assert back["t_air"].to_pylist() == tb["t_air"].to_pylist()


# --------------------------------------------------------------------------- #
# Rows, selections, and the table-level API
# --------------------------------------------------------------------------- #
def test_column_to_arrow_takes_rows(h5file: h5py.File) -> None:
    col = _table(h5file)["t_air"]
    assert col.to_arrow([2, 0]).to_pylist() == [23.100000381469727, 21.5]
    assert col.to_arrow([1]).to_pylist() == [None]


def test_selection_to_arrow(h5file: h5py.File) -> None:
    t = _table(h5file)
    tb = t.select(field("t_air") > 22.0).to_arrow(["station", "t_air"])
    assert tb.column_names == ["station", "t_air"]
    assert tb.num_rows == 1
    assert tb["t_air"].to_pylist() == [23.100000381469727]


def test_table_to_arrow_where_matches_selection(h5file: h5py.File) -> None:
    t = _table(h5file)
    where = t.to_arrow(["t_air"], where=field("t_air") > 22.0)
    assert (
        where.to_pylist()
        == t.select(field("t_air") > 22.0).to_arrow(["t_air"]).to_pylist()
    )


def test_column_subset_and_unknown_name(h5file: h5py.File) -> None:
    t = _table(h5file)
    assert t.to_arrow(["ok", "station"]).column_names == ["ok", "station"]
    with pytest.raises(KeyError):
        t.to_arrow(["nope"])


def test_empty_table(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="x", dtype="f4"), ColumnSpec(name="s", dtype=FixedString(4))],
    )
    tb = t.to_arrow()
    assert tb.num_rows == 0
    assert tb.schema.field("s").type == pa.large_string()


# --------------------------------------------------------------------------- #
# List columns (provisional in this phase: built by conversion, not by wrapping)
# --------------------------------------------------------------------------- #
def test_list_columns(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="i", dtype="i8"),
            ListColumnSpec(
                name="xs", values=LeafValuesSpec(dtype="f8"), nullable=True, units="mm"
            ),
            ListColumnSpec(name="tags", values=StringValuesSpec()),
            ListColumnSpec(
                name="nest", values=NestedListSpec(values=LeafValuesSpec(dtype="f8"))
            ),
        ],
    )
    t.append(
        {
            "i": [1, 2],
            "xs": [[1.0, 2.0], None],
            "tags": [["red"], []],
            "nest": [[[1.0], [2.0, 3.0]], [[4.0]]],
        }
    )
    tb = t.to_arrow()
    assert tb["xs"].to_pylist() == [[1.0, 2.0], None]
    assert tb["tags"].to_pylist() == [["red"], []]
    assert tb["nest"].to_pylist() == [[[1.0], [2.0, 3.0]], [[4.0]]]
    m = tb.schema.field("xs").metadata or {}
    assert m[b"h5col.units"] == b"mm"


# --------------------------------------------------------------------------- #
# Differential: Arrow and the masked read must agree everywhere
# --------------------------------------------------------------------------- #
def test_arrow_agrees_with_the_masked_read(h5file: h5py.File) -> None:
    t = _table(h5file)
    tb = t.to_arrow()
    read = t.read()
    for name in t.column_names:
        assert tb[name].to_pylist() == read[name].tolist(), name
