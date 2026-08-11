"""Read-friendly wrappers over search-index datasets.

:class:`SearchIndex` wraps any ``KIND``-tagged dataset under ``SEARCH_INDEXES``.
Its subclasses add the Layer-1 query primitives per family:
:meth:`ChunkMinMaxIndex.prune` maps a predicate to the candidate chunks a scan
must still read and verify (a **superset** — the query answer is always
defined by the data, never by the index), while :meth:`SortedRowsIndex.rows`
and :meth:`BitmapIndex.rows` answer predicates **exactly**, with the matching
row positions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from . import categorical, indexes
from ._hdf5 import read_bool_attr, read_str_attr, read_uint64_attr
from .exceptions import ConformanceError, SchemaError, StaleIndexError
from .ordering import is_spacepad, normalize_strings
from .reserved import (
    ATTR_CATEGORIES,
    ATTR_DESCRIPTION,
    ATTR_EXHAUSTIVE,
    ATTR_FILL_TAIL_LENGTH,
    ATTR_NAN_TAIL_LENGTH,
    ATTR_ORDERED,
    ATTR_SOURCE_GENERATION,
    ATTR_SOURCE_NROWS,
    KIND_BITMAP,
    KIND_CHUNK_MINMAX,
    KIND_SORTED_ROWS,
)
from .strings import FixedString

if TYPE_CHECKING:
    from .column import Column
    from .table import Table

#: Comparison operators understood by the range-query methods
#: (:meth:`ChunkMinMaxIndex.prune`, :meth:`SortedRowsIndex.rows`).
QUERY_OPS = ("<", "<=", ">", ">=", "==", "between")

#: Backwards-compatible name from sub-phase 4a.
PRUNE_OPS = QUERY_OPS


def _encode_query_value(col_ds: Any, value: Any) -> Any:
    """Canonicalize a query value for exact comparison against stored values.

    Returns a plain Python scalar — ``bytes`` for string columns (compared
    byte-wise, like the stored values) or ``int``/``float``/``bool``
    otherwise — so every comparison happens in the Python object domain,
    where mixed int/float comparisons are exact. NumPy would promote
    int64/uint64 values to float64 and round them, silently corrupting
    comparisons beyond 2**53.
    """
    dtype = col_ds.dtype
    if h5py.check_string_dtype(dtype) is not None:
        # Always UTF-8: the spec orders ASCII columns "as if they were
        # UTF-8", so a non-ASCII query against an ASCII column is a
        # well-defined comparison, not an encoding error — the charset
        # restricts what is storable, never what is comparable.
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, bytes | np.bytes_):
            return bytes(value)
        raise SchemaError(
            f"string-column query value must be str or bytes, got {value!r}"
        )
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int):
        return value  # exact at any magnitude (e.g. 2**64 on uint64)
    if isinstance(value, float):
        if math.isnan(value):
            raise SchemaError(
                "NaN is unordered and never matches a value predicate; "
                "query missing rows with Column.is_missing() instead"
            )
        return value
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise SchemaError(f"expected a scalar query value, got {value!r}")
    if arr.dtype.kind == "f" and np.isnan(arr):
        raise SchemaError(
            "NaN is unordered and never matches a value predicate; query "
            "missing rows with Column.is_missing() instead"
        )
    if arr.dtype.kind not in ("i", "u", "f", "b"):
        raise SchemaError(
            f"query value {value!r} is not comparable with column dtype {dtype!r}"
        )
    return arr.item()


def _canon_element(value: Any, *, spacepad: bool = False) -> Any:
    """Canonicalize one stored element into the query-value domain.

    NumPy scalar reads already strip trailing NUL padding from fixed strings
    (the spec's byte-wise rule for NULLTERM/NULLPAD); SPACEPAD storage strips
    its trailing spaces here.
    """
    if isinstance(value, bytes | np.bytes_):
        out = bytes(value)
        return out.rstrip(b" ") if spacepad else out
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class SearchIndex:
    """One search-index dataset of a table, wrapping its HDF5 dataset.

    A wrapper obtained *through a column* (``Column.search_indexes``) is bound
    to that column; one obtained table-wide (``Table.search_indexes``) resolves
    its column by scanning ``SEARCH_INDEX_LIST`` attributes, the only linkage
    the spec defines. Binding matters on non-conformant files where two columns
    claim the same index: the bound wrapper still answers for its own column,
    while the scan refuses to pick one.
    """

    def __init__(self, dataset: Any, table: Table, column_ds: Any = None) -> None:
        self._ds = dataset
        self._table = table
        self._column_ds = column_ds

    def __repr__(self) -> str:
        try:
            parts = [f"<h5col.{type(self).__name__} {self.name!r}"]
            if self._column_ds is not None:
                # Only when the wrapper is column-bound — never scan
                # SEARCH_INDEX_LIST from a repr.
                parts.append(f" column={self._column_ds.name.rsplit('/', 1)[-1]!r}")
            parts.append(f" kind={self.kind!r}")
            parts.extend(self._repr_extra())
            parts.append(f" valid={self.is_valid}>")
            return "".join(parts)
        except Exception:
            return f"<h5col.{type(self).__name__} (closed or invalid)>"

    def _repr_extra(self) -> list[str]:
        """Per-family extra ``repr`` fields (subclasses override)."""
        return []

    @property
    def name(self) -> str:
        """The index dataset's name (the final component of its HDF5 path)."""
        return self._ds.name.rsplit("/", 1)[-1]

    @property
    def dataset(self) -> Any:
        """The underlying h5py Dataset (for advanced/low-level access)."""
        return self._ds

    @property
    def kind(self) -> str | None:
        """The index's on-disk ``KIND``, or None when absent or malformed."""
        return indexes.index_kind(self._ds)

    @property
    def description(self) -> str | None:
        """The index's ``description`` attribute, or None when unset."""
        return read_str_attr(self._ds, ATTR_DESCRIPTION)

    @property
    def source_generation(self) -> int | None:
        """The ``SOURCE_GENERATION`` validity token, or None when absent."""
        return read_uint64_attr(self._ds, ATTR_SOURCE_GENERATION)

    @property
    def source_nrows(self) -> int | None:
        """The ``SOURCE_NROWS`` validity token, or None when absent."""
        return read_uint64_attr(self._ds, ATTR_SOURCE_NROWS)

    @property
    def is_valid(self) -> bool:
        """The H5Col consumer validity check; False means "treat as absent"."""
        return indexes.index_is_valid(self._ds, self._table.group)

    @property
    def column(self) -> Column | None:
        """The column this index accelerates.

        The bound column when the wrapper came from one; otherwise resolved by
        scanning ``SEARCH_INDEX_LIST`` attributes.
        """
        from .column import Column

        col_ds = self._column_ds
        if col_ds is None:
            col_ds = indexes.find_index_column(self._table.group, self._ds)
        if col_ds is None:
            return None
        return Column(col_ds, self._table)

    def _column_dataset(self) -> Any:
        if self._column_ds is not None:
            return self._column_ds
        col_ds = indexes.find_index_column(self._table.group, self._ds)
        if col_ds is None:
            raise SchemaError(
                f"no column's SEARCH_INDEX_LIST references index {self.name!r}"
            )
        return col_ds

    def _require_valid(self) -> None:
        if not self.is_valid:
            raise StaleIndexError(
                f"search index {self.name!r} fails the validity check; "
                "refresh it or fall back to a scan"
            )


class ChunkMinMaxIndex(SearchIndex):
    """A ``CHUNK_MINMAX`` zone map: per-chunk min/max plus missing counts."""

    @property
    def n_chunks(self) -> int:
        """Data-bearing chunks of the source column at the current ``NROWS``."""
        return indexes.data_chunk_count(self._column_dataset(), self._table.nrows)

    @property
    def chunk_len(self) -> int:
        """Rows per chunk of the source column."""
        return indexes.source_chunk_len(self._column_dataset(), self._table.nrows)

    def entries(self) -> np.ndarray:
        """The index entries for the data-bearing chunks (tail residue clipped).

        Raises
        ------
        SchemaError
            If this is an unbound wrapper whose column cannot be resolved from
            any ``SEARCH_INDEX_LIST``.
        """
        return self._ds[: self.n_chunks]

    def chunk_row_range(self, chunk_id: int) -> tuple[int, int]:
        """Row interval ``[start, stop)`` that chunk *chunk_id* covers.

        Parameters
        ----------
        chunk_id:
            A zero-based chunk position, as returned by :meth:`prune`.

        Raises
        ------
        IndexError
            If *chunk_id* is not a data-bearing chunk (outside ``[0, NROWS)``).
        """
        nrows = self._table.nrows
        chunk_len = self.chunk_len
        start = chunk_id * chunk_len
        if not 0 <= start < nrows:
            raise IndexError(f"chunk {chunk_id} is not a data-bearing chunk")
        return start, min(nrows, start + chunk_len)

    def prune(self, op: str, value: Any) -> np.ndarray:
        """Candidate chunk ids that may hold a non-missing match for the predicate.

        The Layer-1 primitive: returns a **superset** of the chunks containing
        rows whose (non-missing) value satisfies ``op value`` — every returned
        chunk must still be read and verified, but no matching chunk is ever
        excluded. Chunks with no orderable, non-missing element carry
        placeholder bounds and are never candidates; missing rows never match a
        value predicate (query them with :meth:`Column.is_missing`, not with an
        index).

        Parameters
        ----------
        op:
            One of ``<``, ``<=``, ``>``, ``>=``, ``==`` or ``between``.
        value:
            The value to compare against, in the column's decoded form. For
            ``between`` it is an inclusive ``(low, high)`` pair instead.

        Raises
        ------
        StaleIndexError
            If the validity check fails.
        ConformanceError
            If a token-valid index does not cover every data-bearing chunk (a
            rule-9 violation a consumer must not silently clip, because the
            uncovered chunks could hold matches).
        SchemaError
            If *op* is unknown or *value* is not a valid query value for the
            column's datatype.
        """
        self._require_valid()
        if op not in PRUNE_OPS:
            raise SchemaError(f"unknown prune operator {op!r}; use one of {PRUNE_OPS}")

        col_ds = self._column_dataset()
        n_chunks = indexes.data_chunk_count(col_ds, self._table.nrows)
        if self._ds.shape[0] < n_chunks:
            raise ConformanceError(
                f"search index {self.name!r} has {self._ds.shape[0]} entries "
                f"but the column has {n_chunks} data-bearing chunks"
            )
        if n_chunks == 0:
            return np.empty(0, dtype=np.int64)
        entries = self._ds[:n_chunks]

        vmin, vmax = entries["min"], entries["max"]
        if FixedString.is_fixed_string(col_ds.dtype) and is_spacepad(col_ds):
            vmin = normalize_strings(vmin, spacepad=True)
            vmax = normalize_strings(vmax, spacepad=True)

        # Compare in the Python object domain: mixed int/float comparisons are
        # exact there, whereas NumPy would promote int64/uint64 bounds to
        # float64 and round them — silently pruning chunks whose true bounds
        # satisfy the predicate (a false negative) for values beyond 2**53.
        # tolist() also strips trailing NUL padding from fixed strings, which
        # is exactly the spec's byte-wise comparison rule. Index entries are
        # few, so the scalar loop costs nothing next to reading a chunk.
        mins: list[Any] = vmin.tolist()
        maxs: list[Any] = vmax.tolist()

        if op == "between":
            low = _encode_query_value(col_ds, value[0])
            high = _encode_query_value(col_ds, value[1])
            if low > high:
                return np.empty(0, dtype=np.int64)
            hit = [mn <= high and mx >= low for mn, mx in zip(mins, maxs, strict=True)]
        else:
            v = _encode_query_value(col_ds, value)
            if op == "<":
                hit = [mn < v for mn in mins]
            elif op == "<=":
                hit = [mn <= v for mn in mins]
            elif op == ">":
                hit = [mx > v for mx in maxs]
            elif op == ">=":
                hit = [mx >= v for mx in maxs]
            else:  # ==
                hit = [mn <= v <= mx for mn, mx in zip(mins, maxs, strict=True)]

        # A chunk with no orderable, non-missing element holds placeholder
        # bounds that must not participate in pruning decisions — and it cannot
        # contain a match either, so it is simply not a candidate.
        usable = self._usable_mask(col_ds, entries)
        return np.flatnonzero(usable & np.asarray(hit, dtype=np.bool_)).astype(np.int64)

    @staticmethod
    def _usable_mask(col_ds: Any, entries: np.ndarray) -> np.ndarray:
        """Chunks that contain at least one orderable, non-missing element."""
        n = entries["n"]
        fill_count = entries["fill_count"]
        if col_ds.dtype.kind != "f":
            return fill_count < n
        fill = indexes._user_fill(col_ds)
        nan_count = entries["nan_count"]
        if fill is not None and not np.isnan(fill):
            # NaN elements and fill matches are disjoint: both are unorderable.
            return (fill_count + nan_count) < n
        # NaN fill (fill_count == nan_count) or no fill at all: every
        # unorderable element is a NaN, and NaN-fill rows are missing.
        return np.maximum(fill_count, nan_count) < n


class SortedRowsIndex(SearchIndex):
    """A ``SORTED_ROWS`` permutation: row positions in sorted-value order."""

    @property
    def nan_tail_length(self) -> int | None:
        """Rows whose value is NaN, at the permutation's very end."""
        return indexes._scalar_uint64(self._ds.attrs, ATTR_NAN_TAIL_LENGTH)

    @property
    def fill_tail_length(self) -> int | None:
        """Missing (non-NaN fill) rows, immediately before the NaN tail."""
        return indexes._scalar_uint64(self._ds.attrs, ATTR_FILL_TAIL_LENGTH)

    @property
    def ordered(self) -> bool | None:
        """The ``ordered`` flag (must be true for ``SORTED_ROWS``)."""
        return read_bool_attr(self._ds, ATTR_ORDERED)

    def permutation(self) -> np.ndarray:
        """The full permutation over ``[0, NROWS)`` (tail residue clipped)."""
        return self._ds[: self._table.nrows]

    def _body_length(self, nrows: int) -> int:
        """Sorted-body length after both tails, validating the tail attrs."""
        nan_tail = self.nan_tail_length
        fill_tail = self.fill_tail_length
        if nan_tail is None or fill_tail is None:
            raise ConformanceError(
                f"SORTED_ROWS {self.name!r} has missing or malformed "
                "nan_tail_length/fill_tail_length attributes"
            )
        if nan_tail + fill_tail > nrows:
            raise ConformanceError(
                f"SORTED_ROWS {self.name!r} tail lengths exceed NROWS {nrows}"
            )
        return nrows - nan_tail - fill_tail

    def rows(self, op: str, value: Any) -> np.ndarray:
        """Row positions whose (non-missing) value satisfies ``op value``.

        Exact, not a superset: binary search over the sorted body — reading
        O(log NROWS) individual column elements, never the full column — then
        one contiguous permutation slice. The rows come back in sorted-value
        rank order, not row order (sort them before a chunked read). Missing
        rows and NaN rows live in the tails and never match (query them with
        :meth:`Column.is_missing`).

        Parameters
        ----------
        op:
            One of ``<``, ``<=``, ``>``, ``>=``, ``==`` or ``between``.
        value:
            The value to compare against, in the column's decoded form. For
            ``between`` it is an inclusive ``(low, high)`` pair instead.

        Raises
        ------
        StaleIndexError
            If the validity check fails.
        ConformanceError
            For structural violations a consumer must not paper over (undersized
            dataset, bad tail attributes, ``ordered`` not true).
        SchemaError
            If *op* is unknown or *value* is not a valid query value for the
            column's datatype.
        """
        self._require_valid()
        if op not in QUERY_OPS:
            raise SchemaError(f"unknown operator {op!r}; use one of {QUERY_OPS}")

        col_ds = self._column_dataset()
        nrows = self._table.nrows
        if self._ds.shape[0] < nrows:
            raise ConformanceError(
                f"SORTED_ROWS {self.name!r} has {self._ds.shape[0]} entries "
                f"but the table has {nrows} rows"
            )
        if self.ordered is not True:
            raise ConformanceError(
                f"SORTED_ROWS {self.name!r} does not declare a total order "
                "(ordered must be true)"
            )
        body = self._body_length(nrows)
        empty = np.empty(0, dtype=np.int64)
        if body == 0:
            return empty

        spacepad = FixedString.is_fixed_string(col_ds.dtype) and is_spacepad(col_ds)

        def probe(i: int) -> Any:
            return _canon_element(col_ds[int(self._ds[i])], spacepad=spacepad)

        def first(pred: Any) -> int:
            # Least i in [0, body] where pred(probe(i)) holds; the body is
            # sorted, so pred is monotone False -> True.
            lo, hi = 0, body
            while lo < hi:
                mid = (lo + hi) // 2
                if pred(probe(mid)):
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        if op == "between":
            low = _encode_query_value(col_ds, value[0])
            high = _encode_query_value(col_ds, value[1])
            if low > high:
                return empty
            a, b = first(lambda x: x >= low), first(lambda x: x > high)
        else:
            v = _encode_query_value(col_ds, value)
            if op == "<":
                a, b = 0, first(lambda x: x >= v)
            elif op == "<=":
                a, b = 0, first(lambda x: x > v)
            elif op == ">":
                a, b = first(lambda x: x > v), body
            elif op == ">=":
                a, b = first(lambda x: x >= v), body
            else:  # ==
                a, b = first(lambda x: x >= v), first(lambda x: x > v)
        if a >= b:
            return empty
        return self._ds[a:b].astype(np.int64)


class BitmapIndex(SearchIndex):
    """A ``BITMAP``: per-value packed row bitmaps for equality predicates."""

    @property
    def values_dataset(self) -> Any:
        """The accompanying values dataset, or None when unusable."""
        return indexes.bitmap_values_dataset(self._table.group, self._ds)

    @property
    def ordered(self) -> bool | None:
        """Whether the values enumeration order is semantically meaningful."""
        return read_bool_attr(self._ds, ATTR_ORDERED)

    @property
    def exhaustive(self) -> bool:
        """Whether the enumeration provably covers every non-missing value.

        False when the attribute is false, absent, or malformed — exactly the
        cases where the spec forbids treating an enumeration miss as proof of
        absence.
        """
        return bool(read_bool_attr(self._ds, ATTR_EXHAUSTIVE))

    def _repr_extra(self) -> list[str]:
        values_ds = self.values_dataset
        n = values_ds.shape[0] if values_ds is not None else "?"
        return [f" nvalues={n}", f" exhaustive={self.exhaustive}"]

    def values(self) -> np.ndarray:
        """The indexed values, in enumeration (bitmap row) order.

        Raises
        ------
        ConformanceError
            If the BITMAP has no usable values dataset.
        """
        values_ds = self.values_dataset
        if values_ds is None:
            raise ConformanceError(f"BITMAP {self.name!r} has no usable values dataset")
        return values_ds[...]

    def rows(self, value: Any) -> np.ndarray | None:
        """Row positions equal to *value*, or None when the index cannot say.

        Exact when it answers: the union of the bitmap rows whose indexed
        value equals *value* (row order, ascending). Returns an empty array
        when the value is missing from an ``exhaustive`` enumeration (provably
        zero rows) and None when it is missing from a partial one — the caller
        must fall back to a scan. On a categorical column, a ``str`` *value*
        is first encoded to its category code; an unknown label provably
        matches zero rows. Missing rows never match a value predicate, so a
        query equal to the column's fill value returns an empty array.

        Parameters
        ----------
        value:
            The value to match, in the column's decoded form. On a categorical
            column a ``str`` label is accepted and encoded to its code.

        Raises
        ------
        StaleIndexError
            If the validity check fails.
        ConformanceError
            If the bitmap is not a 2-D ``uint8`` dataset, is too narrow for
            ``NROWS``, or has no usable values dataset.
        SchemaError
            If *value* is not a valid query value for the column's datatype.
        """
        self._require_valid()
        col_ds = self._column_dataset()
        nrows = self._table.nrows
        if self._ds.ndim != 2 or self._ds.dtype != np.uint8:
            raise ConformanceError(f"BITMAP {self.name!r} must be a 2-D uint8 dataset")
        n_bytes = indexes.bitmap_bytes(nrows)
        if self._ds.shape[1] < n_bytes:
            raise ConformanceError(
                f"BITMAP {self.name!r} rows hold {self._ds.shape[1]} bytes "
                f"but ceil(NROWS / 8) is {n_bytes}"
            )
        values_ds = self.values_dataset
        if values_ds is None:
            raise ConformanceError(f"BITMAP {self.name!r} has no usable values dataset")
        empty = np.empty(0, dtype=np.int64)
        if nrows == 0:
            return empty

        if ATTR_CATEGORIES in col_ds.attrs and isinstance(value, str):
            try:
                value = int(
                    categorical.encode_labels(self._table.group, col_ds, [value])[0]
                )
            except SchemaError:
                # A label outside the category set has no code, so no stored
                # row can hold it — provably zero matches, indexed or not.
                return empty
        v = _encode_query_value(col_ds, value)

        # Missing rows never match a value predicate; a foreign bitmap MAY
        # enumerate the fill value (its bit semantics are raw equality), so
        # the fill guard runs before the enumeration lookup.
        fill = indexes._user_fill(col_ds)
        if fill is not None:
            spacepad = FixedString.is_fixed_string(col_ds.dtype) and is_spacepad(col_ds)
            if _canon_element(fill, spacepad=spacepad) == v:
                return empty

        stored = values_ds[...]
        vals_spacepad = FixedString.is_fixed_string(values_ds.dtype) and is_spacepad(
            values_ds
        )
        matches = [
            k
            for k, sv in enumerate(stored.tolist())
            if _canon_element(sv, spacepad=vals_spacepad) == v
        ]
        if not matches:
            return empty if self.exhaustive else None
        if any(k >= self._ds.shape[0] for k in matches):
            # The value IS enumerated, but its bitmap row does not exist —
            # answering "no rows" (exhaustive) or "unindexed" (partial) from
            # a truncated bitmap would be silently wrong; this is a rule-9
            # violation the consumer must not paper over.
            raise ConformanceError(
                f"BITMAP {self.name!r} values dataset enumerates more values "
                f"than the bitmap has rows; the index cannot answer"
            )

        acc = np.zeros(nrows, dtype=np.bool_)
        for k in matches:
            acc |= np.unpackbits(
                self._ds[k, :n_bytes], bitorder="little", count=nrows
            ).astype(np.bool_)
        return np.flatnonzero(acc).astype(np.int64)

    def isin(self, values: Any) -> np.ndarray | None:
        """Row positions equal to any of *values* (row order, ascending).

        None when any value is missing from a non-exhaustive enumeration —
        the union cannot be proven complete and the caller must scan.

        Parameters
        ----------
        values:
            An iterable of values, each in whatever form :meth:`rows` accepts.
            Duplicates are harmless.

        Raises
        ------
        StaleIndexError, ConformanceError, SchemaError
            Whatever :meth:`rows` raises for a value (it is called per value).
        """
        acc = np.zeros(self._table.nrows, dtype=np.bool_)
        for value in values:
            r = self.rows(value)
            if r is None:
                return None
            acc[r] = True
        return np.flatnonzero(acc).astype(np.int64)


def wrap_index(dataset: Any, table: Table, column_ds: Any = None) -> SearchIndex:
    """Wrap a search-index dataset in the class matching its ``KIND``.

    Parameters
    ----------
    dataset:
        A search-index dataset. Its ``KIND`` attribute picks the class; an
        unrecognized kind yields the base :class:`SearchIndex`.
    table:
        The table the index belongs to.
    column_ds:
        The column dataset the index was obtained through, binding the wrapper
        to that column (see :class:`SearchIndex`). None leaves it unbound.
    """
    kind = indexes.index_kind(dataset)
    if kind == KIND_CHUNK_MINMAX:
        return ChunkMinMaxIndex(dataset, table, column_ds)
    if kind == KIND_SORTED_ROWS:
        return SortedRowsIndex(dataset, table, column_ds)
    if kind == KIND_BITMAP:
        return BitmapIndex(dataset, table, column_ds)
    return SearchIndex(dataset, table, column_ds)
