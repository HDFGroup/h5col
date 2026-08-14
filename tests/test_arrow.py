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

from h5col.arrow import _offsets_buffer, string_array  # noqa: E402
from h5col.exceptions import ConformanceError, SchemaError  # noqa: E402


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
# List columns: H5Col's storage is Arrow's layout, so the buffers are shared
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


def _list_table(h5file: h5py.File) -> Table:
    """Every VALUES form the convention allows, with nulls at each level."""
    t = Table.create(
        h5file.create_group("lt"),
        [
            ColumnSpec(name="i", dtype="i8"),
            ListColumnSpec(
                name="xs",
                values=LeafValuesSpec(dtype="f8", fill_value=-999.0),
                nullable=True,
            ),
            ListColumnSpec(name="tags", values=StringValuesSpec()),
            ListColumnSpec(name="flags", values=LeafValuesSpec(dtype="bool")),
            ListColumnSpec(name="codes", values=LeafValuesSpec(dtype=FixedString(4))),
            ListColumnSpec(
                name="nest", values=NestedListSpec(values=LeafValuesSpec(dtype="f8"))
            ),
        ],
    )
    t.append(
        {
            "i": [1, 2, 3],
            # A null row *and* a null element inside a row.
            "xs": [[1.0, None], None, [3.0]],
            "tags": [["red", "green"], [], ["blue"]],
            "flags": [[True, False], [], [True]],
            "codes": [["ab", "cd"], [], ["ef"]],
            "nest": [[[1.0], [2.0, 3.0]], [], [[4.0]]],
        }
    )
    return t


def test_list_column_types(h5file: h5py.File) -> None:
    tb = _list_table(h5file).to_arrow()
    f64 = pa.large_list(pa.float64())
    assert tb.schema.field("xs").type == f64
    assert tb.schema.field("tags").type == pa.large_list(pa.large_string())
    assert tb.schema.field("flags").type == pa.large_list(pa.bool_())
    # A fixed-length string leaf is a string, not opaque bytes.
    assert tb.schema.field("codes").type == pa.large_list(pa.large_string())
    assert tb.schema.field("nest").type == pa.large_list(f64)


def test_list_columns_match_the_python_read(h5file: h5py.File) -> None:
    t = _list_table(h5file)
    tb, read = t.to_arrow(), t.read()
    for name in ("xs", "tags", "flags", "codes", "nest"):
        assert tb[name].to_pylist() == read[name], name


def test_inner_null_survives(h5file: h5py.File) -> None:
    # No top-level mask can express a null *element* inside a present row.
    tb = _list_table(h5file).to_arrow()
    assert tb["xs"].to_pylist() == [[1.0, None], None, [3.0]]


def test_list_buffers_are_shared_not_copied(h5file: h5py.File) -> None:

    group = _list_table(h5file)["xs"].group
    raw = group["OFFSETS"][0:4]
    assert raw.dtype == np.uint64
    # uint64 -> int64 is a reinterpret of the same memory, and py_buffer wraps
    # rather than converts, so Arrow ends up pointing at the array h5py filled.
    view = raw.view(np.int64)
    assert view.__array_interface__["data"][0] == raw.__array_interface__["data"][0]
    assert pa.py_buffer(view).address == raw.__array_interface__["data"][0]
    assert _offsets_buffer(group["OFFSETS"], 3).size == raw.nbytes


def test_arrow_table_outlives_the_hdf5_file(h5file: h5py.File, tmp_path) -> None:
    # The buffers Arrow borrows are the arrays h5py read into, not mapped file
    # pages, so closing the file must leave the table intact.
    import gc

    path = tmp_path / "own.h5"
    with h5py.File(path, "w") as f:
        _list_table(f)
    handle = h5py.File(path, "r")
    tb = Table.open(handle["lt"]).to_arrow()
    expected = tb.to_pylist()
    handle.close()
    del handle
    gc.collect()
    assert tb.to_pylist() == expected


def test_empty_list_column(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="f8"))],
    )
    tb = t.to_arrow()
    assert tb.num_rows == 0
    assert tb.schema.field("xs").type == pa.large_list(pa.float64())


def test_list_column_selection_takes_rows(h5file: h5py.File) -> None:
    t = _list_table(h5file)
    tb = t.select(field("i") > 1).to_arrow(["i", "xs"])
    assert tb["i"].to_pylist() == [2, 3]
    assert tb["xs"].to_pylist() == [None, [3.0]]


def test_non_uint64_offsets_are_rejected(h5file: h5py.File) -> None:

    g = h5file.create_group("g")
    ds = g.create_dataset("OFFSETS", data=np.array([0, 1, 2], dtype="i4"))
    with pytest.raises(ConformanceError, match="must be uint64"):
        _offsets_buffer(ds, 2)


# --------------------------------------------------------------------------- #
# Malformed files must raise, never corrupt or abort
# --------------------------------------------------------------------------- #
def _one_list_column(h5file: h5py.File) -> Table:
    t = Table.create(
        h5file.create_group("t"),
        [ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="f8"))],
    )
    t.append({"xs": [[1.0, 2.0], [3.0, 4.0], [5.0]]})
    return t


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        # The half-written file: a tail still zero. Arrow builds this happily
        # and then aborts the process when anything reads it.
        ([0, 2, 4, 0], "must not decrease"),
        ([1, 2, 4, 5], r"OFFSETS\[0\] must be 0"),
        ([0, 4, 2, 5], "must not decrease"),
        ([0, 2, 4, 2**63], "exceeds the signed"),
    ],
)
def test_malformed_offsets_raise(
    h5file: h5py.File, offsets: list[int], message: str
) -> None:

    t = _one_list_column(h5file)
    t["xs"].group["OFFSETS"][:] = np.array(offsets, dtype="u8")
    with pytest.raises(ConformanceError, match=message):
        t.to_arrow()


def test_short_offsets_raise(h5file: h5py.File) -> None:

    ds = h5file.create_group("g").create_dataset(
        "OFFSETS", data=np.array([0, 1], dtype="u8")
    )
    with pytest.raises(ConformanceError, match="must hold 4 entries"):
        _offsets_buffer(ds, 3)


def test_non_uint8_chars_raise(h5file: h5py.File) -> None:

    t = Table.create(
        h5file.create_group("t"),
        [ListColumnSpec(name="tags", values=StringValuesSpec())],
    )
    t.append({"tags": [["red"], ["blue"]]})
    sv = t["tags"].group["VALUES"]
    data = sv["CHARS"][...]
    del sv["CHARS"]
    sv.create_dataset("CHARS", data=data.astype("u4"))
    with pytest.raises(ConformanceError, match="CHARS must be uint8"):
        t.to_arrow()


# --------------------------------------------------------------------------- #
# Byte order: h5py hands back the file's, Arrow refuses anything but native
# --------------------------------------------------------------------------- #
def test_big_endian_columns_export(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [
            ColumnSpec(name="s", dtype=">i4"),
            ListColumnSpec(
                name="xs", values=LeafValuesSpec(dtype=">f8"), nullable=True
            ),
        ],
    )
    t.append({"s": [1, 2, 3], "xs": [[1.5, 2.5], [], None]})
    tb = t.to_arrow()
    assert tb["s"].to_pylist() == [1, 2, 3]
    assert tb["xs"].to_pylist() == [[1.5, 2.5], [], None]
    assert tb["xs"].to_pylist() == t.read()["xs"]


# --------------------------------------------------------------------------- #
# Mask packing: the validity bitmap is bits, so row counts off a multiple of 8
# are where padding errors would hide
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nrows", [1, 7, 8, 9, 63, 64, 65])
def test_null_positions_survive_bit_packing(h5file: h5py.File, nrows: int) -> None:
    rows: list[list[float] | None] = [
        None if i % 3 == 0 else [float(i)] for i in range(nrows)
    ]
    t = Table.create(
        h5file.create_group("t"),
        [ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="f8"), nullable=True)],
    )
    t.append({"xs": rows})
    assert t.to_arrow()["xs"].to_pylist() == rows == t.read()["xs"]


def test_single_null_at_every_position(h5file: h5py.File) -> None:
    # A polarity error cannot hide behind symmetry if the one null moves.
    for pos in range(8):
        g = h5file.create_group(f"t{pos}")
        rows = [None if i == pos else [float(i)] for i in range(8)]
        t = Table.create(
            g,
            [
                ListColumnSpec(
                    name="xs", values=LeafValuesSpec(dtype="f8"), nullable=True
                )
            ],
        )
        t.append({"xs": rows})
        assert t.to_arrow()["xs"].to_pylist() == rows, pos


# --------------------------------------------------------------------------- #
# Slices through the Arrow export
# --------------------------------------------------------------------------- #
def test_column_to_arrow_takes_a_slice(h5file: h5py.File) -> None:
    col = _table(h5file)["t_air"]
    assert col.to_arrow(slice(0, 2)).to_pylist() == col.to_arrow([0, 1]).to_pylist()
    assert col.to_arrow(slice(None)).to_pylist() == col.to_arrow(None).to_pylist()
    assert col.to_arrow(slice(5, 5)).to_pylist() == []


@pytest.mark.parametrize("name", ["station", "t_air", "kind"])
def test_reversed_slice_survives_the_buffer_handover(
    h5file: h5py.File, name: str
) -> None:
    # A reversed hyperslab is a non-contiguous view, and the numeric export
    # hands its buffer to Arrow as it is — so the block has to be made
    # contiguous first or Arrow reads the wrong bytes.
    col = _table(h5file)[name]
    forward = col.to_arrow(None).to_pylist()
    assert col.to_arrow(slice(None, None, -1)).to_pylist() == forward[::-1]


def test_select_narrows_a_list_column_the_same_way_for_every_row_spec(
    h5file: h5py.File,
) -> None:
    from h5col.arrow import _select, list_array

    t = _list_table(h5file)
    col = t["xs"]
    whole = list_array(col.group, t.nrows)
    rows = col.read()

    # step 1 takes Arrow's zero-copy slice; the rest go through take.
    assert _select(pa, whole, slice(0, 2), t.nrows, "xs").to_pylist() == rows[0:2]
    assert (
        _select(pa, whole, slice(None, None, -1), t.nrows, "xs").to_pylist()
        == (rows[::-1])
    )
    assert _select(pa, whole, slice(0, 3, 2), t.nrows, "xs").to_pylist() == rows[0:3:2]
    assert _select(pa, whole, [2, 0], t.nrows, "xs").to_pylist() == [rows[2], rows[0]]
    assert _select(pa, whole, [-1], t.nrows, "xs").to_pylist() == [rows[-1]]
    mask = np.array([False, True, False])
    assert _select(pa, whole, mask, t.nrows, "xs").to_pylist() == [rows[1]]


def test_select_rejects_a_bad_row_spec(h5file: h5py.File) -> None:
    from h5col.arrow import _select, list_array

    t = _list_table(h5file)
    whole = list_array(t["xs"].group, t.nrows)
    with pytest.raises(IndexError):
        _select(pa, whole, [t.nrows], t.nrows, "xs")
    with pytest.raises(IndexError, match="one entry per row"):
        _select(pa, whole, np.zeros(99, dtype=bool), t.nrows, "xs")


# --------------------------------------------------------------------------- #
# Datatypes H5Col stores but Arrow has no type for
# --------------------------------------------------------------------------- #
def _opaque_column(h5file: h5py.File, where: str = "t") -> Table:
    """An opaque column, which the convention permits and h5col writes."""
    t = Table.create(
        h5file.create_group(where),
        [
            ColumnSpec(
                name="digest", dtype=np.dtype("V8"), fill_value=np.void(b"\xff" * 8)
            )
        ],
    )
    t.append({"digest": np.array([b"\x01" * 8, b"\x02" * 8], dtype="V8")})
    return t


def test_an_opaque_column_is_refused_by_name(h5file: h5py.File) -> None:
    # Left to pyarrow this fails with "Unsupported numpy type 20", which says
    # nothing about the column or about what to do instead.
    t = _opaque_column(h5file)
    t.validate(deep=True)  # the table itself is perfectly conformant
    with pytest.raises(SchemaError, match="opaque, compound, or array datatype"):
        t.to_arrow()
    with pytest.raises(SchemaError, match="digest"):
        t["digest"].to_arrow()


def test_a_complex_column_is_refused_by_name(h5file: h5py.File) -> None:
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="z", dtype="c16", fill_value=complex(-9e36, 0))],
    )
    t.append({"z": np.array([1 + 2j, 3 + 4j])})
    with pytest.raises(SchemaError, match="complex datatype"):
        t.to_arrow()


def test_an_opaque_list_leaf_is_refused_by_name(h5file: h5py.File) -> None:
    # The leaf path builds its own arrays, so it needs its own guard.
    t = Table.create(
        h5file.create_group("t"),
        [
            ListColumnSpec(
                name="digests",
                values=LeafValuesSpec(dtype="V8", fill_value=np.void(b"\xff" * 8)),
            )
        ],
    )
    t.append({"digests": [[np.void(b"\x01" * 8)], []]})
    with pytest.raises(SchemaError, match="opaque, compound, or array datatype"):
        t.to_arrow()


def test_a_variable_length_string_column_still_exports(h5file: h5py.File) -> None:
    # The guard covers the two dtype kinds that actually fail. A vlen string
    # column converts (as Arrow binary), so it must not be swept up as well.
    t = Table.create(
        h5file.create_group("t"),
        [ColumnSpec(name="note", dtype=h5py.string_dtype())],
    )
    t.append({"note": np.array(["ab", "cd"], dtype=object)})
    assert t.to_arrow().column("note").to_pylist() == [b"ab", b"cd"]
