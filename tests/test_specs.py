"""Tests for the Pydantic write-side specs (h5col.specs)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from h5col.booleans import bool_dtype
from h5col.specs import ColumnSpec, TableSpec
from h5col.strings import FixedString


def test_column_spec_resolved_dtype() -> None:
    assert ColumnSpec(name="x", dtype="i4").resolved_dtype() == np.dtype("i4")
    assert ColumnSpec(name="s", dtype=FixedString(8)).resolved_dtype().kind == "S"


def test_column_spec_is_boolean() -> None:
    assert ColumnSpec(name="f", dtype=bool_dtype()).is_boolean
    assert ColumnSpec(name="g", dtype="bool").is_boolean
    assert not ColumnSpec(name="x", dtype="i4").is_boolean


def test_table_spec_ordered_names_defaults_to_column_order() -> None:
    spec = TableSpec(
        columns=[ColumnSpec(name="a", dtype="i4"), ColumnSpec(name="b", dtype="i4")]
    )
    assert spec.ordered_names == ["a", "b"]


def test_table_spec_rejects_duplicate_names() -> None:
    with pytest.raises(ValidationError):
        TableSpec(
            columns=[ColumnSpec(name="a", dtype="i4"), ColumnSpec(name="a", dtype="i8")]
        )


def test_table_spec_rejects_unknown_index_column() -> None:
    with pytest.raises(ValidationError):
        TableSpec(columns=[ColumnSpec(name="a", dtype="i4")], index_columns=["b"])


def test_table_spec_rejects_bad_column_order() -> None:
    with pytest.raises(ValidationError):
        TableSpec(
            columns=[
                ColumnSpec(name="a", dtype="i4"),
                ColumnSpec(name="b", dtype="i4"),
            ],
            column_order=["a", "a"],
        )


def test_table_spec_column_order_permutation_ok() -> None:
    spec = TableSpec(
        columns=[
            ColumnSpec(name="a", dtype="i4"),
            ColumnSpec(name="b", dtype="i4"),
        ],
        column_order=["b", "a"],
    )
    assert spec.ordered_names == ["b", "a"]
