"""The Arrow round trip: export a table, import it, export it again.

``to_arrow`` and ``from_arrow`` are inverses only over the part of Arrow's model
H5Col can hold, so the property worth stating is a fixed point rather than an
identity: exporting a table, importing the result, and exporting that must give
an equal Arrow table — same types, same values, same field metadata. A column
kind that loses something on the way through shows up here as an inequality,
and the failure names the kind.

The cases below aim to cover every column kind H5Col can store, since a round
trip that holds for floats and fails for ordered categories is not much of a
guarantee.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest

from h5col import (
    ColumnSpec,
    LeafValuesSpec,
    ListColumnSpec,
    NestedListSpec,
    StringValuesSpec,
    Table,
    specs_from_arrow,
)
from h5col.exceptions import SchemaError
from h5col.strings import FixedString

pa = pytest.importorskip("pyarrow", reason="the arrow extra is not installed")


# --------------------------------------------------------------------------- #
# The cases: one per column kind, each written by h5col itself
# --------------------------------------------------------------------------- #
def _numeric(dtype: str) -> tuple[list[Any], dict[str, list[Any]]]:
    return (
        [ColumnSpec(name="v", dtype=dtype)],
        {"v": [1, None, 3]},
    )


CASES: dict[str, tuple[list[Any], dict[str, list[Any]]]] = {
    # Every primitive with an exact Arrow equivalent, missing rows included.
    **{f"numeric-{d}": _numeric(d) for d in ("i1", "i2", "i4", "i8", "f4", "f8")},
    **{f"numeric-{d}": _numeric(d) for d in ("u1", "u2", "u4", "u8")},
    "boolean": (
        # A boolean declares no fill, so it has no missing rows to carry.
        [ColumnSpec(name="v", dtype="bool")],
        {"v": [True, False, True]},
    ),
    "string": (
        [ColumnSpec(name="v", dtype=FixedString(8))],
        {"v": ["KBOS", "KJFK", None]},
    ),
    "categorical": (
        [ColumnSpec(name="v", categories=["manned", "automatic"])],
        {"v": ["manned", None, "automatic"]},
    ),
    "categorical-ordered": (
        [ColumnSpec(name="v", categories=["low", "high"], ordered=True)],
        {"v": ["low", "high", None]},
    ),
    "categorical-wide-codes": (
        # Two labels fit in int8; this column says int32 anyway. Importing must
        # not quietly narrow it, or the exported schema changes under you.
        [ColumnSpec(name="v", dtype="i4", categories=["low", "high"])],
        {"v": ["low", "high", None]},
    ),
    "opaque": (
        # Raw bytes of one width per row, with a missing one. The fill is
        # H5Col's recommended byte pattern, which the data does not contain.
        [ColumnSpec(name="v", dtype=np.dtype("V8"))],
        {"v": [b"\x01" * 8, None, b"\xff" * 8]},
    ),
    "list-opaque": (
        [ListColumnSpec(name="v", values=LeafValuesSpec(dtype=np.dtype("V4")))],
        {"v": [[b"\x01" * 4], [], [b"\x02" * 4, b"\x03" * 4]]},
    ),
    "list-numeric": (
        [ListColumnSpec(name="v", values=LeafValuesSpec(dtype="f8"), nullable=True)],
        # A null row, an empty row, and a null element inside a row.
        {"v": [[1.0, None], None, []]},
    ),
    "list-string": (
        [ListColumnSpec(name="v", values=StringValuesSpec(nullable=True))],
        {"v": [["red", None], [], ["", "blue"]]},
    ),
    "list-boolean": (
        [ListColumnSpec(name="v", values=LeafValuesSpec(dtype="bool"))],
        {"v": [[True, False], [], [True]]},
    ),
    "list-nested": (
        [
            ListColumnSpec(
                name="v",
                values=NestedListSpec(values=LeafValuesSpec(dtype="f8")),
                nullable=True,
            )
        ],
        {"v": [[[1.0], [2.0, 3.0]], None, [[4.0]]]},
    ),
    "annotated": (
        [
            ColumnSpec(
                name="v",
                dtype="f4",
                units="degC",
                units_vocabulary="UDUNITS-2",
                description="air temperature",
                valid_min=np.float32(-90.0),
                valid_max=np.float32(60.0),
            )
        ],
        {"v": [1.5, None, 3.25]},
    ),
    "annotated-list": (
        [
            ListColumnSpec(
                name="v",
                values=LeafValuesSpec(dtype="f8"),
                units="mm",
                units_vocabulary="UDUNITS-2",
                description="rainfall per hour",
            )
        ],
        {"v": [[1.0], [], [2.0, 3.0]]},
    ),
}


def _written(h5file: h5py.File, case: str, where: str = "src") -> Table:
    """Build the named case as a real H5Col table."""
    specs, rows = CASES[case]
    table = Table.create(h5file.create_group(where), specs)
    table.append(rows)
    return table


# --------------------------------------------------------------------------- #
# The property
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(CASES))
def test_export_import_export_is_a_fixed_point(h5file: h5py.File, case: str) -> None:
    first = _written(h5file, case).to_arrow()
    second = Table.from_arrow(h5file.create_group("again"), first).to_arrow()
    assert second.equals(first, check_metadata=True), (
        f"{case}\n  exported: {first.schema}\n  re-exported: {second.schema}"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_imported_table_is_conformant(h5file: h5py.File, case: str) -> None:
    exported = _written(h5file, case).to_arrow()
    Table.from_arrow(h5file.create_group("again"), exported).validate(deep=True)


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_third_generation_changes_nothing(h5file: h5py.File, case: str) -> None:
    # One hop reaching a fixed point could still be a coincidence of the first
    # import; a second hop that also changes nothing rules that out.
    first = _written(h5file, case).to_arrow()
    second = Table.from_arrow(h5file.create_group("g2"), first).to_arrow()
    third = Table.from_arrow(h5file.create_group("g3"), second).to_arrow()
    assert third.equals(first, check_metadata=True), case


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_values_themselves_survive(h5file: h5py.File, case: str) -> None:
    # Equal schemas with both tables empty would satisfy the property above.
    first = _written(h5file, case).to_arrow()
    second = Table.from_arrow(h5file.create_group("again"), first).to_arrow()
    assert first.num_rows == 3, case
    assert second.column("v").to_pylist() == first.column("v").to_pylist(), case


def test_every_case_together_in_one_table(h5file: h5py.File) -> None:
    # Each case in isolation could pass while some interaction between column
    # kinds does not — a shared batch boundary, say.
    specs: list[Any] = []
    rows: dict[str, list[Any]] = {}
    for case, (case_specs, case_rows) in CASES.items():
        name = case.replace("-", "_")
        for spec in case_specs:
            specs.append(spec.model_copy(update={"name": name}))
        rows[name] = case_rows["v"]
    table = Table.create(h5file.create_group("wide"), specs)
    table.append(rows)

    first = table.to_arrow()
    second = Table.from_arrow(h5file.create_group("again"), first).to_arrow()
    assert second.equals(first, check_metadata=True)


# --------------------------------------------------------------------------- #
# The ordered flag, which Arrow carries on the type itself
# --------------------------------------------------------------------------- #
def test_an_ordered_categorical_exports_an_ordered_arrow_type(
    h5file: h5py.File,
) -> None:
    # Without this the exported type is not self-describing: a consumer that
    # reads the type and not the metadata sees an unordered dictionary.
    exported = _written(h5file, "categorical-ordered").to_arrow()
    assert exported.schema.field("v").type.ordered is True


def test_an_unordered_categorical_exports_an_unordered_arrow_type(
    h5file: h5py.File,
) -> None:
    exported = _written(h5file, "categorical").to_arrow()
    assert exported.schema.field("v").type.ordered is False


def test_an_ordered_arrow_dictionary_imports_as_ordered(h5file: h5py.File) -> None:
    # No h5col metadata at all: the Arrow type's own flag has to be read.
    source = pa.table(
        {
            "v": pa.DictionaryArray.from_arrays(
                pa.array([0, 1], type=pa.int8()),
                pa.array(["low", "high"]),
                ordered=True,
            )
        }
    )
    table = Table.from_arrow(h5file.create_group("t"), source)
    assert table["v"].ordered is True
    assert table.to_arrow().schema.field("v").type.ordered is True


def test_h5col_metadata_outranks_the_arrow_flag(h5file: h5py.File) -> None:
    # A table exported by h5col 0.3.0 carries h5col.ordered but left the Arrow
    # flag at 0. Reading the flag first would drop the ordering.
    source = pa.table(
        {
            "v": pa.DictionaryArray.from_arrays(
                pa.array([0, 1], type=pa.int8()),
                pa.array(["low", "high"]),
                ordered=False,
            )
        }
    ).replace_schema_metadata()
    field = source.schema.field("v").with_metadata({"h5col.ordered": "true"})
    source = source.cast(pa.schema([field]))
    assert specs_from_arrow(source)[0].ordered is True


# --------------------------------------------------------------------------- #
# The empty string, which is the recommended fill for a string column
# --------------------------------------------------------------------------- #
def _with_an_empty_string() -> Any:
    return pa.table({"v": pa.array(["KBOS", "", None], type=pa.large_string())})


def test_an_empty_string_is_refused_by_the_recommended_fill(h5file: h5py.File) -> None:
    # H5Col recommends the empty string as a string column's fill value, so a
    # column that contains one has to say what marks a missing row instead.
    # Importing it regardless would read that row back as missing.
    with pytest.raises(SchemaError, match="occurs in the data"):
        Table.from_arrow(h5file.create_group("t"), _with_an_empty_string())


def test_an_empty_string_survives_a_fill_chosen_for_it(h5file: h5py.File) -> None:
    source = _with_an_empty_string()
    specs = specs_from_arrow(_arrow_without_empty_string())
    specs[0].fill_value = b"\x01"
    table = Table.from_arrow(h5file.create_group("t"), source, specs=specs)
    assert table["v"].read().tolist() == ["KBOS", "", None]
    assert table.to_arrow().column("v").to_pylist() == ["KBOS", "", None]


def _arrow_without_empty_string() -> Any:
    # specs_from_arrow refuses the column with the empty string in it, so the
    # specs are inferred from the same schema holding a value it can size.
    return pa.table({"v": pa.array(["KBOS", "xxxx", None], type=pa.large_string())})


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def test_annotations_reach_the_hdf5_attributes(h5file: h5py.File) -> None:
    # The fixed-point tests would also pass if an annotation were dropped on
    # every hop; this checks it actually lands on the column.
    exported = _written(h5file, "annotated").to_arrow()
    column = Table.from_arrow(h5file.create_group("again"), exported)["v"]
    assert column.units == "degC"
    assert column.units_vocabulary == "UDUNITS-2"
    assert column.description == "air temperature"
    assert column.valid_min == np.float32(-90.0)
    assert column.valid_max == np.float32(60.0)


def test_list_column_annotations_reach_the_hdf5_attributes(h5file: h5py.File) -> None:
    exported = _written(h5file, "annotated-list").to_arrow()
    column = Table.from_arrow(h5file.create_group("again"), exported)["v"]
    assert column.units == "mm"
    assert column.units_vocabulary == "UDUNITS-2"
    assert column.description == "rainfall per hour"
