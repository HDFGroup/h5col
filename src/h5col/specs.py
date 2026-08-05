"""Pydantic v2 specifications for creating H5Col tables and columns.

These model the *write-side* schema — what columns a table has and how each is
stored. They validate structure at construction time; per-value enforcement
(string byte budgets, fill-value ranges, boolean domains) happens when the data
is written and raises the H5Col exception family.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .booleans import is_bool_dtype
from .categorical import choose_code_dtype
from .filters import FilterPipeline
from .strings import FixedString


def _resolve_dtype(dtype: Any) -> np.dtype:
    """Resolve a dtype-like (incl. :class:`FixedString`) to a NumPy dtype."""
    if isinstance(dtype, FixedString):
        return dtype.dtype
    return np.dtype(dtype)


class ColumnSpec(BaseModel):
    """Specification of one column dataset.

    ``dtype`` accepts a NumPy dtype-like, a :class:`~h5col.strings.FixedString`,
    or the boolean dtype from :func:`~h5col.booleans.bool_dtype`. For a
    categorical column, set ``categories`` (the label values); ``dtype`` is then
    the integer code type and may be omitted (a fitting signed int is chosen).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    dtype: Any = None
    chunks: int | tuple[int, ...] | None = None
    filters: FilterPipeline | None = None
    fill_value: Any = None
    valid_min: Any = None
    valid_max: Any = None
    units: str | None = None
    units_vocabulary: str | None = None
    description: str | None = None
    categories: list[Any] | None = None
    ordered: bool | None = None

    @model_validator(mode="after")
    def _check_dtype(self) -> ColumnSpec:
        if self.dtype is None and self.categories is None:
            raise ValueError("dtype is required unless categories is given")
        if self.categories is not None and self.dtype is not None:
            if self.resolved_dtype().kind not in ("i", "u"):
                raise ValueError("a categorical column's dtype must be an integer type")
        return self

    def resolved_dtype(self) -> np.dtype:
        """Return the concrete NumPy dtype for this column."""
        if self.dtype is None:
            if self.categories is not None:
                return choose_code_dtype(len(self.categories))
            raise ValueError("column has no dtype")
        return _resolve_dtype(self.dtype)

    @property
    def is_categorical(self) -> bool:
        """True if this is a categorical column."""
        return self.categories is not None

    @property
    def is_boolean(self) -> bool:
        """True if this is a H5Col boolean column."""
        return not self.is_categorical and is_bool_dtype(self.resolved_dtype())


class LeafValuesSpec(BaseModel):
    """A *leaf* ``VALUES`` member of a list column: a rank-1 element dataset.

    The element ``dtype`` may be any datatype permitted for a column dataset
    except a variable-length datatype (H5Col forbids those below a list column).
    Missing elements are expressed with the fill value, exactly as for column
    datasets; boolean leaves declare no fill (a boolean cannot be missing).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype: Any
    chunks: int | None = None
    filters: FilterPipeline | None = None
    fill_value: Any = None
    valid_min: Any = None
    valid_max: Any = None
    units: str | None = None
    units_vocabulary: str | None = None
    description: str | None = None

    def resolved_dtype(self) -> np.dtype:
        """Return the concrete NumPy dtype for this leaf's elements."""
        return _resolve_dtype(self.dtype)

    @property
    def is_boolean(self) -> bool:
        """True if this leaf holds H5Col boolean values."""
        return is_bool_dtype(self.resolved_dtype())


class StringValuesSpec(BaseModel):
    """A ``STRING_VALUES`` member: variable-length UTF-8 via ``OFFSETS`` + ``CHARS``.

    Set ``nullable=True`` to add a ``MASK`` that distinguishes a null string
    element from an empty one. ``filters`` applies to the ``CHARS`` byte
    buffer; ``chunks`` sets the chunk size of both the group's ``OFFSETS``
    dataset and ``CHARS``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nullable: bool = False
    chunks: int | None = None
    filters: FilterPipeline | None = None


class NestedListSpec(BaseModel):
    """A nested ``LIST_COLUMN`` level: its ``VALUES`` member plus this level's mask.

    ``values`` is the member stored under this level (leaf, string values, or a
    deeper list). ``nullable=True`` adds a ``MASK`` marking null inner lists at
    this level. Used recursively for lists of lists.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: LeafValuesSpec | StringValuesSpec | NestedListSpec
    nullable: bool = False
    chunks: int | None = None
    filters: FilterPipeline | None = None


class ListColumnSpec(BaseModel):
    """Specification of a list column (a ``CLASS=LIST_COLUMN`` group).

    ``values`` describes the ``VALUES`` member. ``nullable=True`` adds the
    top-level ``MASK`` distinguishing a null list from an empty list per row.
    ``chunks``/``filters`` apply to the top-level ``OFFSETS`` dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    values: LeafValuesSpec | StringValuesSpec | NestedListSpec
    nullable: bool = False
    chunks: int | None = None
    filters: FilterPipeline | None = None
    units: str | None = None
    units_vocabulary: str | None = None
    description: str | None = None


# Resolve the forward references in the recursive union.
NestedListSpec.model_rebuild()
ListColumnSpec.model_rebuild()


class TableSpec(BaseModel):
    """Specification of a whole table: its columns and table-level attributes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    columns: list[ColumnSpec | ListColumnSpec]
    title: str | None = None
    description: str | None = None
    index_columns: list[str] = Field(default_factory=list)
    column_order: list[str] | None = None
    units_vocabulary: str | None = None
    encoding_type: str | None = None
    encoding_version: str | None = None

    @model_validator(mode="after")
    def _check_structure(self) -> TableSpec:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        list_names = {c.name for c in self.columns if isinstance(c, ListColumnSpec)}
        for ic in self.index_columns:
            if ic not in names:
                raise ValueError(
                    f"index column {ic!r} is not one of the declared columns"
                )
            if ic in list_names:
                raise ValueError(
                    f"index column {ic!r} is a list column; list columns cannot "
                    "serve as row-index columns"
                )
        if self.column_order is not None and sorted(self.column_order) != sorted(names):
            raise ValueError("column_order must be a permutation of the column names")
        return self

    @property
    def ordered_names(self) -> list[str]:
        """Column names in their logical order (column_order, else spec order)."""
        if self.column_order is not None:
            return list(self.column_order)
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnSpec | ListColumnSpec:
        """Return the column spec named *name*.

        Raises
        ------
        KeyError
            If no column with that name is defined.
        """
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)
