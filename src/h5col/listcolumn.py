"""The ListColumn class: a read-friendly wrapper over one H5Col list column."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import lists
from ._hdf5 import read_str_attr, row_positions
from .booleans import decode_bool
from .reserved import (
    ATTR_DESCRIPTION,
    ATTR_UNITS,
    ATTR_UNITS_VOCABULARY,
    MEMBER_MASK,
)

if TYPE_CHECKING:
    from .table import Table


class ListColumn:
    """One list column of a H5Col table, wrapping its ``CLASS=LIST_COLUMN`` group.

    Reading returns one Python ``list`` per row (or ``None`` for a null list),
    with elements decoded to friendly values — nested lists become nested Python
    lists, string elements become ``str``, and missing leaf elements become
    ``None``.
    """

    is_list = True

    def __init__(self, group: Any, table: Table) -> None:
        self._g = group
        self._table = table

    def __len__(self) -> int:
        """The number of rows in the column, which is the table's ``NROWS``."""
        return int(self._table.nrows)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the rows, each a list or ``None``."""
        return iter(self.read())

    def __getitem__(self, key: Any) -> Any:
        """Read rows by subscript, the same keys a scalar column takes.

        An integer returns one row — a list, or ``None`` if the row is null —
        and every other key returns a list of rows. Negative positions count
        back from the last row.

        Be aware that this reads the whole column and then narrows it, unlike
        a scalar column, which reads only the rows asked for. A list column's
        rows are found through its ``OFFSETS``, and reading a range of those
        directly is not implemented yet. The answer is the same; the cost is
        not. :meth:`~h5col.Table.to_arrow` is the cheaper route to part of a
        large list column.

        Raises
        ------
        IndexError
            If a row position is out of range, a boolean mask is the wrong
            length, or *key* is a tuple.
        TypeError
            If *key* holds values that are neither integers nor booleans.
        """
        if isinstance(key, tuple):
            raise IndexError(
                f"a column is one-dimensional and takes one index; got a "
                f"{len(key)}-tuple for column {self.name!r}"
            )
        if isinstance(key, (bool, np.bool_)):
            raise TypeError(
                f"a single boolean is not a row selection for column "
                f"{self.name!r}; pass a mask with one entry per row"
            )
        if isinstance(key, (int, np.integer)):
            # Wrapped in a list so a bare integer gets the same range check,
            # and the same message, as one inside a sequence.
            positions = row_positions([int(key)], self._table.nrows, self.name)
            return self.read()[int(positions[0])]
        return self.read_rows(key)

    def __repr__(self) -> str:
        try:
            return (
                f"<h5col.ListColumn {self.name!r} "
                f"nullable={self.nullable} nrows={self._table.nrows}>"
            )
        except Exception:
            return "<h5col.ListColumn (closed or invalid)>"

    @property
    def name(self) -> str:
        """The list column's name (the final component of its HDF5 path)."""
        return self._g.name.rsplit("/", 1)[-1]

    @property
    def group(self) -> Any:
        """The underlying h5py Group (for advanced/low-level access)."""
        return self._g

    @property
    def nullable(self) -> bool:
        """True if the top level carries a ``MASK`` (null lists are possible)."""
        return MEMBER_MASK in self._g

    @property
    def units(self) -> str | None:
        """The list column's ``units`` attribute, or None when unset."""
        return read_str_attr(self._g, ATTR_UNITS)

    @property
    def units_vocabulary(self) -> str | None:
        """The list column's ``units_vocabulary`` attribute, or None when unset."""
        return read_str_attr(self._g, ATTR_UNITS_VOCABULARY)

    @property
    def description(self) -> str | None:
        """The list column's ``description`` attribute, or None when unset."""
        return read_str_attr(self._g, ATTR_DESCRIPTION)

    def read(self, *, masked: bool = True) -> list[Any]:
        """Read rows ``[0, NROWS)`` as a list of per-row lists (``None`` = null).

        ``masked`` is accepted and ignored, so a caller can pass it uniformly
        across a table's columns. A list column is ragged and so cannot be a
        :class:`numpy.ma.MaskedArray`. It already spells a null row ``None``.
        """
        return lists.read_list_column(self._g, self._table.nrows)

    def read_rows(self, rows: Any, *, masked: bool = True) -> list[Any]:
        """Read just *rows*, in the order given, as a list of (list | None).

        Accepts the same row specs as :meth:`Column.read_rows` — a slice, a
        sequence of positions, or a boolean mask — so a caller can select rows
        the same way whatever kind of column it holds. ``masked`` is accepted
        and ignored, as it is by :meth:`read`.

        The whole column is read and then narrowed; see :meth:`__getitem__`
        for why.
        """
        if isinstance(rows, slice):
            return self.read()[rows]
        positions = row_positions(rows, self._table.nrows, self.name)
        full = self.read()
        return [full[int(i)] for i in positions]

    def is_missing(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of null-list rows over ``[0, NROWS)``.

        A list column with no top-level ``MASK`` cannot mark a row missing, so
        every row reads as present.
        """
        n = self._table.nrows
        if MEMBER_MASK not in self._g:
            return np.zeros(n, dtype=np.bool_)
        return ~decode_bool(self._g[MEMBER_MASK][0:n])
