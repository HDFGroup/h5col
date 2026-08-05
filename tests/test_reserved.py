"""Tests for reserved names and validators (h5col.reserved)."""

from __future__ import annotations

import pytest

from h5col import reserved
from h5col.exceptions import ReservedNameError, SchemaError


def test_class_and_kind_tokens() -> None:
    assert reserved.CLASS_COLUMN_TABLE == "COLUMN_TABLE"
    assert reserved.CLASS_LIST_COLUMN == "LIST_COLUMN"
    assert reserved.CLASS_STRING_VALUES == "STRING_VALUES"
    assert reserved.KIND_OFFSETS == "OFFSETS"
    assert reserved.SEARCH_INDEX_KINDS == {
        "CHUNK_MINMAX",
        "SORTED_ROWS",
        "BITMAP",
        "CHUNK_BLOOM",
    }


def test_is_valid_link_name() -> None:
    assert reserved.is_valid_link_name("energy")
    assert not reserved.is_valid_link_name("")
    assert not reserved.is_valid_link_name("a/b")
    assert not reserved.is_valid_link_name("a\x00b")
    assert not reserved.is_valid_link_name(123)


def test_validate_column_name_accepts_normal() -> None:
    assert reserved.validate_column_name("energy") == "energy"
    # Unicode is permitted.
    assert reserved.validate_column_name("naïve") == "naïve"


@pytest.mark.parametrize(
    "name",
    ["CATEGORIES", "SEARCH_INDEXES", "OFFSETS", "VALUES", "MASK", "CHARS"],
)
def test_validate_column_name_rejects_reserved(name: str) -> None:
    with pytest.raises(ReservedNameError):
        reserved.validate_column_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "CLASS",
        "VERSION",
        "NROWS",
        "TITLE",
        "INDEX_COLUMNS",
        "GENERATION",
        "KIND",
        "SEARCH_INDEX_LIST",
        "SOURCE_GENERATION",
        "SOURCE_NROWS",
        "valid_min",
        "valid_max",
    ],
)
def test_validate_column_name_rejects_reserved_attributes(name: str) -> None:
    # H5Col reserved-names rule 2: a column name must not be any catalog name,
    # including reserved attribute names.
    with pytest.raises(ReservedNameError):
        reserved.validate_column_name(name)


@pytest.mark.parametrize("name", ["a/b", "", "with\x00nul"])
def test_validate_column_name_rejects_bad_link_names(name: str) -> None:
    with pytest.raises(SchemaError):
        reserved.validate_column_name(name)


def test_reserved_name_error_is_schema_error() -> None:
    assert issubclass(ReservedNameError, SchemaError)


def test_discouraged_names() -> None:
    assert reserved.is_discouraged_column_name("_index")
    assert not reserved.is_discouraged_column_name("index")


def test_reserved_attribute_names_membership() -> None:
    for name in ("CLASS", "VERSION", "NROWS", "valid_min", "valid_max", "MASK"):
        assert name in reserved.RESERVED_ATTRIBUTE_NAMES
