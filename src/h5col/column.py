"""The Column class: a read-friendly wrapper over one H5Col column dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import categorical, indexes, missing
from ._hdf5 import gather_rows, read_str_attr
from .booleans import decode_bool, is_bool_dtype
from .reserved import (
    ATTR_CATEGORIES,
    ATTR_DESCRIPTION,
    ATTR_UNITS,
    ATTR_VALID_MAX,
    ATTR_VALID_MIN,
)
from .searchindex import SearchIndex, wrap_index
from .strings import FixedString, decoded_string_dtype

if TYPE_CHECKING:
    from .table import Table


class Column:
    """One column of a H5Col table, wrapping its rank-1 HDF5 dataset.

    Reads decode to friendly Python values: fixed-length strings become ``str``,
    boolean columns become NumPy ``bool``, and numeric columns pass through.
    """

    is_list = False

    def __init__(self, dataset: Any, table: Table) -> None:
        self._ds = dataset
        self._table = table

    def __repr__(self) -> str:
        try:
            parts = [f"<h5col.Column {self.name!r} dtype={self.dtype!r}"]
            if self.is_categorical:
                n = categorical.n_categories(self._table.group, self._ds)
                parts.append(f" categories={n}")
            parts.append(f" nrows={self._table.nrows}>")
            return "".join(parts)
        except Exception:
            return "<h5col.Column (closed or invalid)>"

    @property
    def name(self) -> str:
        """The column's name (the final component of its HDF5 path)."""
        return self._ds.name.rsplit("/", 1)[-1]

    @property
    def dataset(self) -> Any:
        """The underlying h5py Dataset (for advanced/low-level access)."""
        return self._ds

    @property
    def dtype(self) -> np.dtype:
        """The column's NumPy dtype (its stored HDF5 datatype).

        For a categorical column this is the integer *code* dtype, not the
        category labels (see :attr:`categories`).
        """
        return self._ds.dtype

    @property
    def is_boolean(self) -> bool:
        """True if this is an H5Col boolean column."""
        return is_bool_dtype(self._ds.dtype)

    @property
    def is_string(self) -> bool:
        """True if this is a fixed-length string column."""
        return FixedString.is_fixed_string(self._ds.dtype)

    @property
    def is_categorical(self) -> bool:
        """True if this is a categorical column (its values are category codes)."""
        return ATTR_CATEGORIES in self._ds.attrs

    @property
    def categories(self) -> npt.NDArray[Any] | None:
        """The category labels, or None for a non-categorical column.

        String labels come back as a compact NumPy string array; numeric labels
        keep their own dtype.
        """
        if not self.is_categorical:
            return None
        labels = categorical.load_category_labels(self._table.group, self._ds)
        if labels and all(isinstance(lab, str) for lab in labels):
            return np.asarray(labels, dtype=decoded_string_dtype())
        return np.array(labels, dtype=object)

    @property
    def ordered(self) -> bool | None:
        """The categories' ``ordered`` flag, or None if not categorical/unset."""
        if not self.is_categorical:
            return None
        return categorical.is_ordered(self._table.group, self._ds)

    @property
    def codes(self) -> npt.NDArray[Any]:
        """Raw integer category codes over ``[0, NROWS)`` (categorical columns)."""
        return self._ds[0 : self._table.nrows]

    def _has_user_fill(self) -> bool:
        # H5D_FILL_VALUE_USER_DEFINED == 2 (matches table.py usage).
        return self._ds.id.get_create_plist().fill_value_defined() == 2

    @property
    def fill_value(self) -> Any:
        """The column's fill value, or None when it declares none.

        None is returned for boolean columns and for any column not in the
        ``H5D_FILL_VALUE_USER_DEFINED`` state (e.g. a full-domain column that
        declares no missing-row semantics). h5py's library-default fill value is
        not a H5Col sentinel and must not be surfaced as one.
        """
        if self.is_boolean or not self._has_user_fill():
            return None
        return self._ds.fillvalue

    @property
    def units(self) -> str | None:
        """The column's ``units`` attribute, or None when unset."""
        return read_str_attr(self._ds, ATTR_UNITS)

    @property
    def description(self) -> str | None:
        """The column's ``description`` attribute, or None when unset."""
        return read_str_attr(self._ds, ATTR_DESCRIPTION)

    @property
    def valid_min(self) -> Any:
        """The column's ``valid_min`` attribute, or None when unset."""
        a = self._ds.attrs
        return a[ATTR_VALID_MIN] if ATTR_VALID_MIN in a else None

    @property
    def valid_max(self) -> Any:
        """The column's ``valid_max`` attribute, or None when unset."""
        a = self._ds.attrs
        return a[ATTR_VALID_MAX] if ATTR_VALID_MAX in a else None

    def _decode(self, raw: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Decode stored values to friendly ones.

        Elementwise, so it applies equally to a whole column or to any subset
        of one — which is what lets :meth:`read_rows` share it with
        :meth:`read`.
        """
        if self.is_categorical:
            return categorical.decode_codes(self._table.group, self._ds, raw)
        if self.is_string:
            return FixedString.from_dtype(self._ds.dtype).decode(raw)
        if self.is_boolean:
            return decode_bool(raw)
        return raw

    def read(self) -> npt.NDArray[Any]:
        """Read the logical rows ``[0, NROWS)``, decoded to friendly values."""
        n = self._table.nrows
        return self._decode(self._ds[0:n])

    def read_rows(self, rows: Any) -> npt.NDArray[Any]:
        """Read just *rows*, decoded, in the order given.

        Values are fetched with coalesced, chunk-aligned block reads, so a
        selection confined to a few chunks costs a few chunks rather than the
        whole column — in both time and peak memory. Rows may be given in any
        order and may repeat.

        Raises
        ------
        IndexError
            If a row position is negative or not below ``NROWS``.
        ValueError
            If *rows* is not one-dimensional.
        """
        idx = np.asarray(rows, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"rows must be a 1-D sequence, got {idx.ndim}-D")
        n = self._table.nrows
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise IndexError(
                f"row positions must lie in [0, {n}) for column {self.name!r}"
            )
        # gather_rows needs ascending input; restore the caller's order after.
        order = np.argsort(idx, kind="stable")
        raw = gather_rows(self._ds, idx[order], n)
        restored = np.empty_like(raw)
        restored[order] = raw
        return self._decode(restored)

    @property
    def search_indexes(self) -> list[SearchIndex]:
        """Search indexes bound to this column, from ``SEARCH_INDEX_LIST``.

        The wrappers are bound to this column, so their queries always run
        against it — even on a non-conformant file where another column also
        claims the same index dataset.

        Raises
        ------
        ConformanceError
            If the column's ``SEARCH_INDEX_LIST`` attribute is malformed (not a
            1-D array of object references).
        """
        return [
            wrap_index(ds, self._table, self._ds)
            for ds in indexes.column_index_datasets(self._table.group, self._ds)
        ]

    def add_search_index(
        self,
        kind: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> SearchIndex:
        """Build a search index over this column (:meth:`Table.add_search_index`).

        Raises
        ------
        SchemaError
            If no index family applies to the column's dtype (``kind=None``),
            *kind* is unimplemented, or ``SEARCH_INDEXES`` already holds a
            dataset of the chosen name.
        ReservedNameError
            If *name* is a H5Col reserved name.
        ConformanceError
            If the table carries no ``NROWS`` attribute.
        """
        return self._table.add_search_index(
            self.name, kind, name=name, description=description
        )

    def build_index(
        self,
        kind: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> SearchIndex:
        """Build a search index over this column (alias of :meth:`add_search_index`)."""
        return self.add_search_index(kind, name=name, description=description)

    def is_missing(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of missing rows over ``[0, NROWS)``.

        A column with no user-defined fill value (boolean columns, or full-domain
        columns per H5Col) declares no missing-row semantics, so every row reads
        as present.
        """
        n = self._table.nrows
        if self.is_boolean or not self._has_user_fill():
            return np.zeros(n, dtype=np.bool_)
        return missing.is_missing(self._ds[0:n], self._ds.fillvalue)
