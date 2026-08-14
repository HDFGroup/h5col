"""The Column class: a read-friendly wrapper over one H5Col column dataset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt

from . import arrow, categorical, indexes, missing
from ._hdf5 import gather_rows, read_str_attr, row_positions
from .booleans import decode_bool, is_bool_dtype
from .reserved import (
    ATTR_CATEGORIES,
    ATTR_DESCRIPTION,
    ATTR_UNITS,
    ATTR_UNITS_VOCABULARY,
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

    Rows can be read by subscript — ``column[17:98]``, ``column[-1]``,
    ``column[[3, 1]]`` — which is :meth:`read_rows` with its defaults. Note
    that ``column[...]`` and ``column.dataset[...]`` are not the same read:
    the second goes straight to h5py, so it skips decoding and can return rows
    above ``NROWS`` that :meth:`~h5col.Table.truncate` left behind as reserved
    storage.
    """

    is_list = False

    def __init__(self, dataset: Any, table: Table) -> None:
        self._ds = dataset
        self._table = table

    def __len__(self) -> int:
        """The number of rows in the column, which is the table's ``NROWS``.

        .. versionadded:: 0.3.0
        """
        return int(self._table.nrows)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the decoded rows.

        Defined so that iterating reads the column once. Without it Python
        falls back on :meth:`__getitem__` and fetches every row separately,
        which is one HDF5 read per row.

        .. versionadded:: 0.3.0
        """
        return iter(self.read())

    def __getitem__(self, key: Any) -> Any:
        """Read rows by subscript, decoded and masked.

        An integer key returns that one row's value — :data:`numpy.ma.masked`
        if the row is missing — and every other key returns an array, the way
        NumPy behaves.

        This is :meth:`read_rows` with the defaults. Subscript has nowhere to
        put a keyword, so it always decodes and always masks; call
        :meth:`read` or :meth:`read_rows` when you want ``masked=False``.

        .. versionadded:: 0.3.0

        Parameters
        ----------
        key:
            An integer, a slice, a sequence of positions, or a boolean array
            with one entry per row. A negative position counts back from the
            last row.

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
            n = self._table.nrows
            pos = int(key)
            if pos < 0:
                pos += n
            if not 0 <= pos < n:
                raise IndexError(
                    f"row {int(key)} is out of range for column {self.name!r}, "
                    f"which has {n} rows"
                )
            # One row still goes through the slice path, so it is one hyperslab
            # rather than a gather.
            return self.read_rows(slice(pos, pos + 1))[0]
        return self.read_rows(key)

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
    def units_vocabulary(self) -> str | None:
        """The column's ``units_vocabulary`` attribute, or None when unset.

        Names the vocabulary :attr:`units` is drawn from — UDUNITS-2, say — so
        a reader can tell which spelling of a unit was meant. A table declares
        one for all its columns; this is the per-column override.

        .. versionadded:: 0.4.0
        """
        return read_str_attr(self._ds, ATTR_UNITS_VOCABULARY)

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

    def _slice_block(self, key: slice, n: int) -> npt.NDArray[Any]:
        """The stored values for a slice of rows, read as one hyperslab.

        A slice is the one selection HDF5 serves directly, so it is kept off
        the scattered path entirely. That path has to sort the positions,
        gather, and scatter the result back into the caller's order, and for a
        contiguous range every step of it is wasted: a million-row range spends
        several milliseconds sorting a sequence that was already sorted. Going
        straight to h5py skips all of it.
        """
        start, stop, step = key.indices(n)
        if step > 0:
            return self._ds[start:stop:step]
        # HDF5 hyperslabs only run forwards, so read the same rows ascending
        # and reverse the block. The copy keeps the result contiguous, which
        # the Arrow export needs to hand the buffer over as it is.
        wanted = range(start, stop, step)
        if not wanted:
            return self._ds[0:0]
        block = self._ds[wanted[-1] : wanted[0] + 1 : -step]
        return np.ascontiguousarray(block[::-1])

    def _raw_block(self, rows: Any = None) -> npt.NDArray[Any]:
        """The stored values for *rows*, or all of ``[0, NROWS)`` when None.

        Shared by every reader — decoded, masked or Arrow — so they agree on
        which rows exist and how a selection is fetched. *rows* may be a slice,
        a sequence of integers, or a boolean mask with one entry per row.
        Negative positions count back from ``NROWS``.

        Raises
        ------
        IndexError
            If a row position is out of range, or a boolean mask is the wrong
            length.
        TypeError
            If *rows* holds values that are neither integers nor booleans.
        ValueError
            If *rows* is not one-dimensional.
        """
        n = self._table.nrows
        if rows is None:
            return self._ds[0:n]
        if isinstance(rows, slice):
            return self._slice_block(rows, n)
        idx = row_positions(rows, n, self.name)
        # gather_rows needs ascending input; restore the caller's order after.
        order = np.argsort(idx, kind="stable")
        raw = gather_rows(self._ds, idx[order], n)
        restored = np.empty_like(raw)
        restored[order] = raw
        return restored

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

    def _missing_mask(self, raw: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        """Missing-row mask for a block of *stored* values already in hand.

        Taken from the block the caller just read rather than by re-reading the
        column, so a masked read costs one pass over the data, not two.

        A boolean column, which H5Col forbids from declaring a fill value, and a
        full-domain column that declares none, have no missing rows — their mask
        is all-False rather than absent, so every scalar column comes back the
        same shape of object.
        """
        if self.is_boolean or not self._has_user_fill():
            return np.zeros(raw.shape[0], dtype=np.bool_)
        return missing.is_missing(raw, self._ds.fillvalue)

    def _mask(self, raw: npt.NDArray[Any], decoded: npt.NDArray[Any]) -> Any:
        """Pair *decoded* values with the missing rows *raw* implies."""
        mask = self._missing_mask(raw)
        out = np.ma.MaskedArray(decoded, mask=mask, copy=False)
        if mask.any():
            # Every masked position already holds the decoded fill, so take it
            # from there: NumPy's own default is unusable (999999 for an int8
            # column, the string "N/A" for a string one), and decoding the
            # stored sentinel separately would re-read a categorical's whole
            # labels dataset on every read. This is what makes
            # ``read().filled()`` reproduce ``read(masked=False)``.
            #
            # It must be a 0-d array: assigning the bare ``None`` a categorical
            # decodes to is accepted and then silently ignored, leaving NumPy's
            # sentinel to masquerade as a label.
            try:
                # NumPy's stub types fill_value as a scalar; at runtime only
                # the 0-d array form is honoured for these dtypes.
                out.fill_value = np.asarray(  # type: ignore[assignment]
                    decoded[int(np.argmax(mask))], dtype=decoded.dtype
                )
            except UnicodeDecodeError:
                # A fill a non-conformant producer wrote as invalid UTF-8.
                # Reading the column must not depend on decoding it.
                pass
        return out

    @overload
    def read(self, *, masked: Literal[True] = ...) -> np.ma.MaskedArray: ...
    @overload
    def read(self, *, masked: Literal[False]) -> npt.NDArray[Any]: ...
    @overload
    def read(self, *, masked: bool) -> Any: ...

    def read(self, *, masked: bool = True) -> Any:
        """Read the logical rows ``[0, NROWS)``, decoded to friendly values.

        Parameters
        ----------
        masked:
            Return a :class:`numpy.ma.MaskedArray` whose mask marks the
            column's missing rows (the default). Pass False for the plain
            array, in which case a missing row holds the column's fill value
            with nothing to distinguish it from data.
        """
        raw = self._raw_block()
        decoded = self._decode(raw)
        return self._mask(raw, decoded) if masked else decoded

    @overload
    def read_rows(
        self, rows: Any, *, masked: Literal[True] = ...
    ) -> np.ma.MaskedArray: ...
    @overload
    def read_rows(self, rows: Any, *, masked: Literal[False]) -> npt.NDArray[Any]: ...
    @overload
    def read_rows(self, rows: Any, *, masked: bool) -> Any: ...

    def read_rows(self, rows: Any, *, masked: bool = True) -> Any:
        """Read just *rows*, decoded, in the order given.

        A slice is read as a single hyperslab. Anything else is fetched with
        coalesced, chunk-aligned block reads, so a selection confined to a few
        chunks costs a few chunks rather than the whole column — in both time
        and peak memory.

        Parameters
        ----------
        rows:
            A slice, a sequence of integer positions, or a boolean mask with
            one entry per row. A negative position counts back from the end, so
            ``-1`` is the last row. Integer positions may be given in any order
            and may repeat; the result follows the order given.
        masked:
            As for :meth:`read`.

        Raises
        ------
        IndexError
            If a row position is out of range, or a boolean mask is the wrong
            length.
        TypeError
            If *rows* holds values that are neither integers nor booleans.
        ValueError
            If *rows* is not one-dimensional.
        """
        raw = self._raw_block(rows)
        decoded = self._decode(raw)
        return self._mask(raw, decoded) if masked else decoded

    def to_arrow(self, rows: Any = None) -> Any:
        """Convert the column to an Arrow array, or just *rows* of it.

        Missing rows become real Arrow nulls rather than the fill value, and a
        categorical column becomes a ``DictionaryArray`` of the codes and
        labels H5Col already stores — neither of which NumPy can express.

        Needs the optional ``pyarrow`` dependency (``pip install
        h5col[arrow]``).

        .. versionadded:: 0.2.0

        Parameters
        ----------
        rows:
            Which rows to convert, in any form :meth:`read_rows` accepts. None
            (the default) converts the whole column.
        """
        return arrow.column_array(self, rows)

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

        Parameters
        ----------
        kind:
            Which index family to build — ``CHUNK_MINMAX``, ``SORTED_ROWS`` or
            ``BITMAP``. None (the default) picks the family that suits the
            column's datatype.
        name:
            Name for the index dataset under ``SEARCH_INDEXES``. None derives
            one from the column name and the kind.
        description:
            Free text stored on the index as its ``DESCRIPTION`` attribute.

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
        """Build a search index over this column (alias of :meth:`add_search_index`).

        Parameters
        ----------
        kind:
            As for :meth:`add_search_index`.
        name:
            As for :meth:`add_search_index`.
        description:
            As for :meth:`add_search_index`.
        """
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
