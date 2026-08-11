"""Tests for list columns: LIST_COLUMN / STRING_VALUES, nesting, MASK, no-vlen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from h5col import (
    ColumnSpec,
    LeafValuesSpec,
    ListColumn,
    ListColumnSpec,
    NestedListSpec,
    StringValuesSpec,
    Table,
    TableSpec,
    bool_dtype,
    field,
)
from h5col._hdf5 import (
    read_str_attr,
    write_ascii_token_attr,
    write_uint64_attr,
)
from h5col.exceptions import ConformanceError, OversizedStringError, SchemaError
from h5col.lists import reject_vlen
from h5col.reserved import (
    ATTR_CLASS,
    ATTR_KIND,
    ATTR_NROWS,
    ATTR_VERSION,
    CLASS_LIST_COLUMN,
    CLASS_STRING_VALUES,
    MEMBER_CHARS,
    MEMBER_OFFSETS,
    MEMBER_VALUES,
)
from h5col.strings import FixedString


def _norm(x: Any) -> Any:
    """Recursively convert NumPy arrays/scalars to plain Python for comparison."""
    if x is None:
        return None
    if isinstance(x, str | bytes):
        return x
    if isinstance(x, np.ndarray):
        return [_norm(e) for e in x.tolist()]
    if isinstance(x, list):
        return [_norm(e) for e in x]
    item = getattr(x, "item", None)
    return item() if callable(item) else x


# --------------------------------------------------------------------------- #
# Leaf numeric lists
# --------------------------------------------------------------------------- #
def test_list_float_roundtrip(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))])
    t.append({"r": [[1.0, 2.0], [], [3.0, 4.0, 5.0]]})
    assert t.nrows == 3
    col = t["r"]
    assert isinstance(col, ListColumn)
    assert col.is_list is True
    assert _norm(col.read()) == [[1.0, 2.0], [], [3.0, 4.0, 5.0]]


def test_list_multiple_appends_offset_continuity(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="i8"))])
    t.append({"r": [[1], [2, 3]]})
    t.append({"r": [[4, 5, 6], []]})
    t.append({"r": [[7]]})
    assert t.nrows == 5
    assert _norm(t["r"].read()) == [[1], [2, 3], [4, 5, 6], [], [7]]
    # OFFSETS must be monotonic across the append boundaries.
    offs = t["r"].group[MEMBER_OFFSETS][0:6]
    assert offs.tolist() == [0, 1, 3, 6, 6, 7]


def test_list_missing_leaf_element_via_fill(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    # A None element inside a list becomes the leaf fill value on disk and reads
    # back as None (the canonical missing-value test).
    t = Table.create(g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="i4"))])
    t.append({"r": [[1, None, 3]]})
    assert _norm(t["r"].read()) == [[1, None, 3]]


# --------------------------------------------------------------------------- #
# Null lists (top-level MASK)
# --------------------------------------------------------------------------- #
def test_list_nullable_null_vs_empty(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f8"), nullable=True)]
    )
    t.append({"r": [[1.0], None, []]})
    col = t["r"]
    assert col.nullable is True
    assert _norm(col.read()) == [[1.0], None, []]
    assert list(col.is_missing()) == [False, True, False]
    # A null entry occupies zero VALUES elements (OFFSETS[i+1] == OFFSETS[i]).
    offs = col.group[MEMBER_OFFSETS][0:4]
    assert offs.tolist() == [0, 1, 1, 1]


def test_list_non_nullable_none_row_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))])
    with pytest.raises(SchemaError):
        t.append({"r": [[1.0], None]})
    assert t.nrows == 0


# --------------------------------------------------------------------------- #
# String element lists (STRING_VALUES)
# --------------------------------------------------------------------------- #
def test_list_string_values_roundtrip(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="tags", values=StringValuesSpec())])
    t.append({"tags": [["red", "green"], ["blue"], []]})
    col = t["tags"]
    assert col.read() == [["red", "green"], ["blue"], []]
    # The VALUES member is a STRING_VALUES group with a uint8 CHARS buffer.
    sv = col.group[MEMBER_VALUES]
    assert read_str_attr(sv, "CLASS") == CLASS_STRING_VALUES
    assert sv[MEMBER_CHARS].dtype == np.dtype("u1")
    assert sv[MEMBER_OFFSETS].dtype == np.dtype("u8")


def test_list_string_null_vs_empty_element(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [ListColumnSpec(name="tags", values=StringValuesSpec(nullable=True))],
    )
    # Within one row: a present string, an empty string, and a null string.
    t.append({"tags": [["a", "", None]]})
    assert t["tags"].read() == [["a", "", None]]


def test_list_string_non_nullable_none_element_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="tags", values=StringValuesSpec())])
    with pytest.raises(SchemaError):
        t.append({"tags": [["a", None]]})


def test_list_string_utf8_multibyte(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="tags", values=StringValuesSpec())])
    t.append({"tags": [["café", "naïve"], ["日本語"]]})
    assert t["tags"].read() == [["café", "naïve"], ["日本語"]]


# --------------------------------------------------------------------------- #
# Fixed-string leaf and boolean leaf
# --------------------------------------------------------------------------- #
def test_list_fixed_string_leaf(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ListColumnSpec(name="codes", values=LeafValuesSpec(dtype=FixedString(4)))]
    )
    t.append({"codes": [["ab", "cde"], ["x"]]})
    assert t["codes"].read() == [["ab", "cde"], ["x"]]


def test_list_fixed_string_oversized_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ListColumnSpec(name="codes", values=LeafValuesSpec(dtype=FixedString(4)))]
    )
    with pytest.raises(OversizedStringError):
        t.append({"codes": [["abcde"]]})  # 5 bytes > 4


def test_list_boolean_leaf(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ListColumnSpec(name="flags", values=LeafValuesSpec(dtype=bool_dtype()))]
    )
    t.append({"flags": [[True, False], [True]]})
    assert _norm(t["flags"].read()) == [[True, False], [True]]


def test_list_boolean_leaf_none_element_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ListColumnSpec(name="flags", values=LeafValuesSpec(dtype=bool_dtype()))]
    )
    with pytest.raises(SchemaError):
        t.append({"flags": [[True, None]]})


# --------------------------------------------------------------------------- #
# Nested lists
# --------------------------------------------------------------------------- #
def test_nested_list_of_lists(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    spec = ListColumnSpec(
        name="m", values=NestedListSpec(values=LeafValuesSpec(dtype="i2"))
    )
    t = Table.create(g, [spec])
    t.append({"m": [[[1, 2], [3]], [[4]], []]})
    assert _norm(t["m"].read()) == [[[1, 2], [3]], [[4]], []]
    # The VALUES member is itself a LIST_COLUMN group.
    inner = t["m"].group[MEMBER_VALUES]
    assert isinstance(inner, h5py.Group)
    assert read_str_attr(inner, "CLASS") == CLASS_LIST_COLUMN


def test_nested_list_inner_nullable(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    spec = ListColumnSpec(
        name="m",
        values=NestedListSpec(values=LeafValuesSpec(dtype="i4"), nullable=True),
    )
    t = Table.create(g, [spec])
    # An inner list may be null (distinct from an empty inner list).
    t.append({"m": [[[1], None, []]]})
    assert _norm(t["m"].read()) == [[[1], None, []]]


# --------------------------------------------------------------------------- #
# No variable-length datatypes below a list column (rule 11)
# --------------------------------------------------------------------------- #
def test_list_vlen_string_leaf_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    with pytest.raises(SchemaError):
        Table.create(
            g,
            [
                ListColumnSpec(
                    name="x", values=LeafValuesSpec(dtype=h5py.string_dtype())
                )
            ],
        )
    assert not Table.is_table_group(g)


def test_list_vlen_sequence_leaf_rejected(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    vlen = h5py.vlen_dtype(np.dtype("i4"))
    with pytest.raises(SchemaError):
        Table.create(g, [ListColumnSpec(name="x", values=LeafValuesSpec(dtype=vlen))])


def test_reject_vlen_descends_into_compound_and_array() -> None:
    # Vlen hidden inside a compound field, or under an array subtype, must be
    # caught (rule 11 forbids vlen ANYWHERE below a list column).
    with pytest.raises(SchemaError):
        reject_vlen(np.dtype([("id", "i4"), ("tags", h5py.string_dtype())]))
    with pytest.raises(SchemaError):
        reject_vlen(np.dtype([("a", "i4"), ("b", h5py.vlen_dtype(np.int32))]))
    with pytest.raises(SchemaError):
        reject_vlen(np.dtype((h5py.vlen_dtype(np.int32), (3,))))
    # A compound of only fixed-width fields is fine.
    reject_vlen(np.dtype([("a", "i4"), ("b", "f8")]))


def test_list_compound_vlen_leaf_rejected_on_create(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    compound = np.dtype([("id", "i4"), ("tags", h5py.string_dtype())])
    with pytest.raises(SchemaError):
        Table.create(
            g,
            [
                ListColumnSpec(
                    name="x", values=LeafValuesSpec(dtype=compound, fill_value=(0, b""))
                )
            ],
        )
    assert not Table.is_table_group(g)


def test_validate_rejects_externally_built_compound_vlen_leaf(
    h5file: h5py.File,
) -> None:
    # Hand-build a structurally valid list column whose leaf VALUES carries a
    # variable-length field, mimicking a non-conformant externally-produced file.
    g = h5file.create_group("t")
    write_ascii_token_attr(g, ATTR_CLASS, "COLUMN_TABLE")
    write_ascii_token_attr(g, ATTR_VERSION, "1.0")
    write_uint64_attr(g, ATTR_NROWS, 0)
    lc = g.create_group("x")
    write_ascii_token_attr(lc, ATTR_CLASS, CLASS_LIST_COLUMN)
    write_ascii_token_attr(lc, ATTR_KIND, "OFFSETS")
    off = lc.create_dataset("OFFSETS", shape=(1,), maxshape=(None,), dtype="u8")
    off[0] = 0
    compound = np.dtype([("id", "i4"), ("tags", h5py.string_dtype())])
    lc.create_dataset(
        "VALUES", shape=(0,), maxshape=(None,), dtype=compound, chunks=(8,)
    )
    with pytest.raises(ConformanceError):
        Table.open(g).validate()


# --------------------------------------------------------------------------- #
# Discovery, column-order, mixing with scalar columns
# --------------------------------------------------------------------------- #
def test_list_and_scalar_columns_mixed(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="id", dtype="i8"),
            ListColumnSpec(name="readings", values=LeafValuesSpec(dtype="f4")),
        ],
    )
    t.append({"id": [10, 20], "readings": [[1.0, 2.0], [3.0]]})
    assert t.nrows == 2
    assert t.column_names == ["id", "readings"]
    assert _norm(t["id"].read()) == [10, 20]
    assert _norm(t["readings"].read()) == [[1.0, 2.0], [3.0]]
    assert t["readings"].is_list and not t["id"].is_list


def test_list_column_in_column_order(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="id", dtype="i8"),
            ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4")),
        ],
        column_order=["r", "id"],
    )
    assert t.column_names == ["r", "id"]


def test_list_column_cannot_be_index_column() -> None:
    with pytest.raises(ValidationError):
        TableSpec(
            columns=[
                ColumnSpec(name="id", dtype="i8"),
                ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4")),
            ],
            index_columns=["r"],
        )


def test_list_column_annotations(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ListColumnSpec(
                name="r",
                values=LeafValuesSpec(dtype="f4", units="V"),
                description="sensor readings",
            )
        ],
    )
    assert t["r"].description == "sensor readings"
    assert t["r"].group[MEMBER_VALUES].attrs  # leaf carries units


# --------------------------------------------------------------------------- #
# Absent columns during append
# --------------------------------------------------------------------------- #
def test_absent_nullable_list_becomes_null(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="id", dtype="i8"),
            ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"), nullable=True),
        ],
    )
    t.append({"id": [1, 2]})  # 'r' omitted -> null lists
    assert _norm(t["r"].read()) == [None, None]
    assert list(t["r"].is_missing()) == [True, True]


def test_absent_non_nullable_list_raises(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="id", dtype="i8"),
            ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4")),
        ],
    )
    with pytest.raises(SchemaError):
        t.append({"id": [1, 2]})
    assert t.nrows == 0


# --------------------------------------------------------------------------- #
# validate()
# --------------------------------------------------------------------------- #
def test_validate_passes_for_list_table(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g,
        [
            ColumnSpec(name="id", dtype="i8"),
            ListColumnSpec(name="r", values=StringValuesSpec(nullable=True)),
            ListColumnSpec(
                name="m", values=NestedListSpec(values=LeafValuesSpec(dtype="i2"))
            ),
        ],
    )
    t.append({"id": [1, 2], "r": [["a"], []], "m": [[[1]], [[2, 3], [4]]]})
    t.validate()  # must not raise


def test_validate_catches_non_monotonic_offsets(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))])
    t.append({"r": [[1.0], [2.0, 3.0], [4.0]]})
    # Corrupt a committed OFFSETS value to break monotonicity.
    t["r"].group[MEMBER_OFFSETS][2] = 0
    with pytest.raises(ConformanceError):
        t.validate()


def test_validate_catches_unexpected_member(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))])
    t.append({"r": [[1.0]]})
    t["r"].group.create_dataset("EXTRA", data=np.arange(3))
    with pytest.raises(ConformanceError):
        t.validate()


def test_validate_catches_null_entry_with_nonempty_slice(h5file: h5py.File) -> None:
    g = h5file.create_group("t")
    t = Table.create(
        g, [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"), nullable=True)]
    )
    t.append({"r": [[1.0, 2.0], None]})
    # Force the null row (index 1) to appear to span elements: set its MASK False
    # already; now make OFFSETS[2] != OFFSETS[1].
    grp = t["r"].group
    grp[MEMBER_OFFSETS][2] = grp[MEMBER_OFFSETS][1] + 1
    with pytest.raises(ConformanceError):
        t.validate()


# --------------------------------------------------------------------------- #
# True close/reopen round-trip
# --------------------------------------------------------------------------- #
def test_list_reopen_from_disk(h5path: Path) -> None:
    with h5py.File(h5path, "w") as f:
        g = f.create_group("t")
        t = Table.create(
            g,
            [
                ColumnSpec(name="id", dtype="i8"),
                ListColumnSpec(name="tags", values=StringValuesSpec()),
                ListColumnSpec(
                    name="m", values=NestedListSpec(values=LeafValuesSpec(dtype="i2"))
                ),
            ],
        )
        t.append({"id": [1, 2], "tags": [["x", "y"], []], "m": [[[1, 2]], [[3]]]})

    with h5py.File(h5path, "r") as f:
        t = Table.open(f["t"])
        assert t.nrows == 2
        assert _norm(t["id"].read()) == [1, 2]
        assert t["tags"].read() == [["x", "y"], []]
        assert _norm(t["m"].read()) == [[[1, 2]], [[3]]]


# --------------------------------------------------------------------------- #
# Ranged reads
#
# A list column's rows are reached through OFFSETS, so reading a range means
# narrowing the child read to the span those offsets describe — at every level
# of nesting. Offsets are absolute while the block just read is not, so each
# level has to rebase them; these tests exist mostly to pin that down.
# --------------------------------------------------------------------------- #

RANGE_ROWS = 60


def _range_table(h5file: h5py.File) -> Table:
    """Every VALUES form, with null rows, null elements and empty rows."""
    t = Table.create(
        h5file.create_group("rt"),
        [
            ListColumnSpec(
                name="nums",
                values=LeafValuesSpec(dtype="f8", fill_value=-999.0),
                nullable=True,
            ),
            ListColumnSpec(name="tags", values=StringValuesSpec(), nullable=True),
            ListColumnSpec(name="flags", values=LeafValuesSpec(dtype=bool_dtype())),
            ListColumnSpec(
                name="nest",
                values=NestedListSpec(values=LeafValuesSpec(dtype="i8")),
                nullable=True,
            ),
        ],
    )
    t.append(
        {
            # Null rows, empty rows, a null element, and varying lengths so no
            # two rows share an offset pattern.
            "nums": [
                None if i % 11 == 0 else [float(i + k) for k in range(i % 4)]
                for i in range(RANGE_ROWS)
            ],
            "tags": [
                None if i % 13 == 0 else [f"t{i}-{k}" for k in range(i % 3)]
                for i in range(RANGE_ROWS)
            ],
            "flags": [
                [bool((i + k) % 2) for k in range(i % 3)] for i in range(RANGE_ROWS)
            ],
            "nest": [
                None if i % 17 == 0 else [[i + k] * (k + 1) for k in range(i % 3)]
                for i in range(RANGE_ROWS)
            ],
        }
    )
    return t


RANGE_COLUMNS = ["nums", "tags", "flags", "nest"]
RANGES = [
    (0, 5),
    (1, 4),
    (7, 8),
    (0, RANGE_ROWS),
    (RANGE_ROWS - 1, RANGE_ROWS),
    (RANGE_ROWS - 5, RANGE_ROWS),
    (11, 12),  # a null row on its own
    (10, 14),  # a null row inside a range
    (23, 23),  # empty
]


@pytest.mark.parametrize("colname", RANGE_COLUMNS)
@pytest.mark.parametrize(("start", "stop"), RANGES)
def test_ranged_read_matches_the_whole_column(
    h5file: h5py.File, colname: str, start: int, stop: int
) -> None:
    col = _range_table(h5file)[colname]
    whole = col.read()
    assert _norm(col.read_rows(slice(start, stop))) == _norm(whole[start:stop])


@pytest.mark.parametrize("colname", RANGE_COLUMNS)
def test_ranged_read_keeps_null_rows_distinct_from_empty_ones(
    h5file: h5py.File, colname: str
) -> None:
    col = _range_table(h5file)[colname]
    whole = col.read()
    for i in range(RANGE_ROWS):
        one = col[i]
        if whole[i] is None:
            assert one is None, f"row {i} of {colname} lost its null"
        else:
            assert one is not None
            assert _norm(one) == _norm(whole[i])


@pytest.mark.parametrize("colname", RANGE_COLUMNS)
def test_scattered_positions_match_the_whole_column(
    h5file: h5py.File, colname: str
) -> None:
    col = _range_table(h5file)[colname]
    whole = col.read()
    for rows in (
        [5, 2, 5],
        [0, RANGE_ROWS - 1],
        [11, 13, 12],
        [-1, -2],
        list(range(0, RANGE_ROWS, 7)),
    ):
        assert _norm(col.read_rows(rows)) == _norm([whole[i] for i in rows])


@pytest.mark.parametrize("colname", RANGE_COLUMNS)
def test_strided_and_reversed_slices(h5file: h5py.File, colname: str) -> None:
    col = _range_table(h5file)[colname]
    whole = col.read()
    for key in (
        slice(None, None, -1),
        slice(2, 40, 5),
        slice(None, None, 3),
        slice(-6, None),
    ):
        assert _norm(col.read_rows(key)) == _norm(whole[key])


def _count_reads(fn: object) -> int:
    """Total HDF5 elements *fn* pulls, counted at the h5py level."""
    real = h5py.Dataset.__getitem__
    total = 0

    def logged(self: object, key: object) -> object:
        nonlocal total
        block = real(self, key)
        total += int(np.size(block))
        return block

    h5py.Dataset.__getitem__ = logged  # type: ignore[method-assign]
    try:
        fn()  # type: ignore[operator]
    finally:
        h5py.Dataset.__getitem__ = real  # type: ignore[method-assign]
    return total


@pytest.mark.parametrize("colname", RANGE_COLUMNS)
def test_a_range_reads_only_that_range(h5file: h5py.File, colname: str) -> None:
    col = _range_table(h5file)[colname]
    whole = _count_reads(col.read)
    ranged = _count_reads(lambda: col.read_rows(slice(10, 15)))
    assert ranged < whole / 5, (
        f"{colname}: {ranged} elements is not much less than {whole}"
    )


def test_clustered_positions_read_little_and_spread_ones_are_never_worse(
    h5file: h5py.File,
) -> None:
    col = _range_table(h5file)["nums"]
    whole = _count_reads(col.read)
    clustered = _count_reads(lambda: col.read_rows([20, 22, 21]))
    spread = _count_reads(lambda: col.read_rows([0, RANGE_ROWS - 1]))

    assert clustered < whole / 5
    # Spanning the column costs what the column costs, and no more.
    assert spread <= whole


def test_single_row_subscript_reads_one_row(h5file: h5py.File) -> None:
    col = _range_table(h5file)["nest"]
    whole = _count_reads(col.read)
    one = _count_reads(lambda: col[30])
    assert one < whole / 5


def test_ranged_read_survives_multiple_appends(h5file: h5py.File) -> None:
    # Offsets carry across appends, so a range that straddles an append
    # boundary is the case most likely to expose a rebasing mistake.
    t = Table.create(
        h5file.create_group("multi"),
        [ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="i8"), nullable=True)],
    )
    t.append({"xs": [[1], [2, 2], None]})
    t.append({"xs": [[3, 3, 3], None, [4]]})
    t.append({"xs": [[], [5, 5]]})
    col = t["xs"]
    whole = col.read()
    for start in range(len(whole)):
        for stop in range(start, len(whole) + 1):
            assert _norm(col.read_rows(slice(start, stop))) == _norm(whole[start:stop])


def test_selection_read_narrows_a_list_column(h5file: h5py.File) -> None:
    # Selection.read used to send every list column down the read-whole path,
    # so a query matching a handful of rows still paid for the entire column.
    t = Table.create(
        h5file.create_group("q"),
        [
            ColumnSpec(name="i", dtype="i8", fill_value=-1),
            ListColumnSpec(name="xs", values=LeafValuesSpec(dtype="f8"), nullable=True),
        ],
    )
    n = 300
    t.append(
        {
            "i": list(range(n)),
            "xs": [None if i % 23 == 0 else [float(i)] * (i % 3) for i in range(n)],
        }
    )

    sel = t.select(field("i") >= n - 5)
    got = sel.read(["xs"])["xs"]
    whole = t["xs"].read()
    assert _norm(got) == _norm(whole[-5:])

    xs_whole = _count_reads(t["xs"].read)
    xs_via_query = _count_reads(lambda: sel.read(["xs"]))
    # The query still scans "i"; what must shrink is the list column's share.
    assert xs_via_query - n < xs_whole / 5
