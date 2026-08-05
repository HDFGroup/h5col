"""The ListColumn class: a read-friendly wrapper over one H5Col list column."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import lists
from ._hdf5 import read_str_attr
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

    def read(self) -> list[Any]:
        """Read rows ``[0, NROWS)`` as a list of per-row lists (``None`` = null)."""
        return lists.read_list_column(self._g, self._table.nrows)

    def is_missing(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of null-list rows over ``[0, NROWS)``.

        A list column with no top-level ``MASK`` cannot mark a row missing, so
        every row reads as present.
        """
        n = self._table.nrows
        if MEMBER_MASK not in self._g:
            return np.zeros(n, dtype=np.bool_)
        return ~decode_bool(self._g[MEMBER_MASK][0:n])
