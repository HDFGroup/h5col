"""Export H5Col columns and tables as Apache Arrow.

Arrow is the one representation that carries the whole H5Col data model without
loss. NumPy has no type for three of the things H5Col stores: a null distinct
from the value that marks it, a dictionary of codes and labels, and (from the
list-column work) ragged values with nulls at every level. Arrow has all three
natively, and its memory layout is close enough to H5Col's that most of the
conversion is handing buffers over rather than rewriting them.

``pyarrow`` is an optional dependency — install ``h5col[arrow]``. Nothing else
in the package imports it.

Type mapping
------------
=====================  =========================================
H5Col column           Arrow type
=====================  =========================================
numeric                the matching primitive, data buffer shared
boolean                ``bool_``
fixed-length string    ``large_string``
categorical            ``dictionary<indices=<code dtype>, values=...>``
list column            ``large_list<...>``, buffers shared
=====================  =========================================

Missing rows become real Arrow nulls in every case, so an Arrow consumer never
sees the fill value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import numpy.typing as npt

from . import categorical, missing
from ._hdf5 import read_str_attr, row_positions
from .booleans import decode_bool, is_bool_dtype
from .exceptions import ConformanceError
from .reserved import (
    ATTR_CLASS,
    ATTR_DESCRIPTION,
    ATTR_UNITS,
    ATTR_VALID_MAX,
    ATTR_VALID_MIN,
    CLASS_LIST_COLUMN,
    CLASS_STRING_VALUES,
    MEMBER_CHARS,
    MEMBER_MASK,
    MEMBER_OFFSETS,
    MEMBER_VALUES,
)
from .strings import FixedString

if TYPE_CHECKING:
    from .column import Column

#: Prefix for the Arrow field-metadata keys carrying HDF5 column attributes, so
#: they cannot collide with a consumer's own metadata.
METADATA_PREFIX = "h5col."


def require_pyarrow() -> Any:
    """Import and return ``pyarrow``, or explain how to get it.

    Raises
    ------
    ModuleNotFoundError
        If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise ModuleNotFoundError(
            "to_arrow() needs pyarrow, which h5col does not require by default. "
            "Install it with `pip install h5col[arrow]` or `conda install -c "
            "conda-forge pyarrow`."
        ) from exc
    return pa


def _native(raw: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """*raw* in native byte order, for Arrow, which rejects byte-swapped input.

    h5py hands back the file's byte order rather than normalizing it, and
    big-endian datasets are ordinary in this corner of the world — NetCDF-4,
    Fortran and IDL producers all write them. H5Col reads them fine, so the
    Arrow export must too. A no-op on native data.
    """
    # isnative covers both cases that need nothing done: already native, and
    # dtypes with no byte order to speak of (S, bool, enum).
    if raw.dtype.isnative:
        return raw
    return raw.astype(raw.dtype.newbyteorder("="), copy=False)


def _validity(mask: npt.NDArray[np.bool_]) -> Any:
    """Arrow's validity bitmap for a H5Col missing-row mask.

    H5Col carries presence as one byte per row; Arrow wants one *bit*, set when
    the row is valid. Returns None when nothing is missing, which is how Arrow
    spells "no nulls" and costs it no buffer at all.
    """
    pa = require_pyarrow()
    if not mask.any():
        return None
    return pa.py_buffer(np.packbits(~mask, bitorder="little"))


def string_array(
    raw: npt.NDArray[Any], nbytes: int, mask: npt.NDArray[np.bool_]
) -> Any:
    """Build an Arrow ``large_string`` from a fixed-width string column block.

    A fixed-width column has no offsets to lend, so unlike a list column this is
    a real conversion. It is still done with array arithmetic rather than a
    Python loop: the trailing NULs HDF5 pads with are counted per row, the
    lengths accumulate into the offsets buffer, and the payload bytes are taken
    in one masked gather.

    Parameters
    ----------
    raw:
        The stored fixed-width bytes, one row per element.
    nbytes:
        The column's fixed width, i.e. the stored size of every row.
    mask:
        True where the row is missing. Masked rows become Arrow nulls and
        contribute no payload bytes.
    """
    pa = require_pyarrow()
    n = raw.shape[0]
    flat = raw.view(np.uint8).reshape(n, nbytes)
    # Index of the last non-NUL byte, counting from the right.
    lens = nbytes - (flat[:, ::-1] != 0).argmax(axis=1)
    # argmax gives 0 for an all-NUL row, which would read as full width.
    lens = np.where((flat != 0).any(axis=1), lens, 0).astype(np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    payload = np.ascontiguousarray(
        flat.ravel()[(np.arange(nbytes) < lens[:, None]).ravel()]
    )
    return pa.Array.from_buffers(
        pa.large_string(),
        n,
        [_validity(mask), pa.py_buffer(offsets), pa.py_buffer(payload)],
    )


def _categorical_array(col: Column, raw: npt.NDArray[Any], mask: Any) -> Any:
    """Build an Arrow ``DictionaryArray`` from codes and labels.

    H5Col already stores a categorical as codes plus a separate label dataset,
    which is exactly Arrow's dictionary layout — so the codes go across as they
    are and only the (small) label set is materialized.
    """
    pa = require_pyarrow()
    labels = categorical.load_category_labels(col._table.group, col.dataset)
    # A fill code is outside [0, ncats) and would fail Arrow's index check, so
    # park masked rows on index 0 and let the validity bitmap hide them.
    codes = np.where(mask, 0, raw).astype(raw.dtype, copy=False)
    return pa.DictionaryArray.from_arrays(pa.array(codes, mask=mask), pa.array(labels))


def column_array(col: Column, rows: Any = None) -> Any:
    """Convert one scalar column (or *rows* of it) to an Arrow array.

    Parameters
    ----------
    col:
        The scalar column to convert. List columns go through
        :func:`list_array` instead.
    rows:
        Which rows to convert, in any form
        :meth:`~h5col.Column.read_rows` accepts. None converts the whole
        column.
    """
    pa = require_pyarrow()
    raw = col._raw_block(rows)
    mask = col._missing_mask(raw)
    if col.is_categorical:
        return _categorical_array(col, raw, mask)
    if col.is_string:
        nbytes = FixedString.from_dtype(col.dtype).nbytes
        return string_array(raw, nbytes, mask)
    if col.is_boolean:
        return pa.array(decode_bool(raw), mask=mask)
    return pa.array(_native(raw), mask=mask)


def column_metadata(col: Column) -> dict[str, str]:
    """The column's HDF5 attributes, as Arrow field metadata.

    Arrow metadata is a flat string-to-string map, so numeric attributes are
    rendered with ``str``. The keys are prefixed with ``h5col.`` and survive a
    Parquet round trip, which is what makes them worth carrying at all.

    Parameters
    ----------
    col:
        The scalar column whose attributes are collected. Attributes that are
        unset are simply left out.
    """
    meta: dict[str, str] = {}
    for key, value in (
        (ATTR_UNITS, col.units),
        (ATTR_DESCRIPTION, col.description),
        (ATTR_VALID_MIN, col.valid_min),
        (ATTR_VALID_MAX, col.valid_max),
    ):
        if value is not None:
            meta[f"{METADATA_PREFIX}{key}"] = str(value)
    if col.is_categorical and col.ordered is not None:
        meta[f"{METADATA_PREFIX}ordered"] = str(bool(col.ordered)).lower()
    return meta


def list_column_metadata(col: Any) -> dict[str, str]:
    """A list column's HDF5 attributes, as Arrow field metadata.

    Parameters
    ----------
    col:
        The list column whose attributes are collected. Attributes that are
        unset are simply left out.
    """
    meta: dict[str, str] = {}
    for key, value in (
        (ATTR_UNITS, col.units),
        ("units_vocabulary", col.units_vocabulary),
        (ATTR_DESCRIPTION, col.description),
    ):
        if value is not None:
            meta[f"{METADATA_PREFIX}{key}"] = str(value)
    return meta


def _select(pa: Any, array: Any, rows: Any, nrows: int, name: str) -> Any:
    """*rows* of an already-built Arrow array.

    A list column is wrapped whole and then narrowed, because a subset of
    OFFSETS cannot be shared as a buffer. This keeps the row spec meaning the
    same thing here as it does for a scalar column, where the selection is
    applied while reading instead.
    """
    if isinstance(rows, slice):
        start, stop, step = rows.indices(nrows)
        if step == 1:
            # Zero-copy: a contiguous run needs no take at all.
            return array.slice(start, max(0, stop - start))
        positions = np.arange(start, stop, step, dtype=np.int64)
    else:
        positions = row_positions(rows, nrows, name)
    return array.take(pa.array(positions))


def table_arrow(table: Any, columns: Any = None, rows: Any = None) -> Any:
    """Convert a table (or *rows* of it) to an Arrow table.

    Parameters
    ----------
    table:
        The table to convert.
    columns:
        Names to convert, in the order given. None converts every column.
    rows:
        Which rows to convert, as a slice, a sequence of positions, or a
        boolean mask. None converts every row.

    Raises
    ------
    KeyError
        If a requested column name is not a column of the table.
    """
    from .listcolumn import ListColumn

    pa = require_pyarrow()
    names = list(columns) if columns is not None else table.column_names
    cols = table.columns
    arrays, fields = [], []
    for name in names:
        if name not in cols:
            raise KeyError(name)
        col = cols[name]
        if isinstance(col, ListColumn):
            array = list_array(col.group, table.nrows)
            if rows is not None:
                array = _select(pa, array, rows, table.nrows, name)
            meta = list_column_metadata(col)
        else:
            array = column_array(col, rows)
            meta = column_metadata(col)
        arrays.append(array)
        fields.append(pa.field(name, array.type, metadata=meta or None))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


# --------------------------------------------------------------------------- #
# List columns
#
# These are the case Arrow fits best. H5Col already stores a list column the way
# Arrow lays one out — an OFFSETS array plus a values buffer — so almost nothing
# is converted; the buffers are handed over as they are. The exceptions are the
# null masks, which H5Col keeps as one byte per element and Arrow as one bit.
# --------------------------------------------------------------------------- #
#: Largest offset the uint64 → int64 reinterpret can carry. Beyond it the value
#: reads as negative, which Arrow does not check and which crashes on use.
_MAX_OFFSET = 2**63 - 1


def _offsets_buffer(ds: Any, count: int) -> Any:
    """H5Col's uint64 OFFSETS as Arrow's int64 offsets buffer.

    Same width, so the reinterpret is a view rather than a conversion. The
    signed reading is why the Arrow types here are the ``large_`` variants.

    The convention's invariants are checked here rather than left to Arrow.
    ``Array.from_buffers`` validates almost nothing, so a malformed OFFSETS —
    a half-written file whose tail is still zero, say — builds an array happily
    and then aborts the process when something reads it. Checking costs a pass
    over an array we have just read.

    Raises
    ------
    ConformanceError
        If OFFSETS is the wrong datatype or length, does not start at zero, is
        not monotonically non-decreasing, or exceeds the signed range.
    """
    pa = require_pyarrow()
    raw = ds[0 : count + 1]
    if raw.dtype != np.uint64:
        raise ConformanceError(
            f"{ds.name!r}: list-column OFFSETS must be uint64, got {raw.dtype}"
        )
    if raw.shape[0] != count + 1:
        raise ConformanceError(
            f"{ds.name!r}: OFFSETS must hold {count + 1} entries for {count} rows, "
            f"found {raw.shape[0]}"
        )
    if count >= 0 and raw.shape[0] and raw[0] != 0:
        raise ConformanceError(f"{ds.name!r}: OFFSETS[0] must be 0, got {raw[0]}")
    if raw.shape[0] > 1:
        bad = np.flatnonzero(raw[1:] < raw[:-1])
        if bad.size:
            i = int(bad[0])
            raise ConformanceError(
                f"{ds.name!r}: OFFSETS must not decrease, but entry {i + 1} "
                f"({raw[i + 1]}) is below entry {i} ({raw[i]})"
            )
        if int(raw[-1]) > _MAX_OFFSET:
            raise ConformanceError(
                f"{ds.name!r}: final OFFSETS entry {raw[-1]} exceeds the signed "
                "64-bit range Arrow offsets use"
            )
    return pa.py_buffer(raw.view(np.int64))


def _element_mask(ds: Any, raw: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    """Missing-element mask for a leaf VALUES dataset.

    Gated on ``H5D_FILL_VALUE_USER_DEFINED`` exactly as a column's is, so h5py's
    library-default fill value is never mistaken for a H5Col sentinel.
    """
    if is_bool_dtype(ds.dtype) or ds.id.get_create_plist().fill_value_defined() != 2:
        return np.zeros(raw.shape[0], dtype=np.bool_)
    return missing.is_missing(raw, ds.fillvalue)


def _leaf_array(ds: Any, count: int) -> Any:
    """A leaf VALUES dataset as an Arrow array.

    A leaf may be any datatype a column may be except a variable-length one, so
    this covers the fixed-length string and boolean cases as well as numerics.
    """
    pa = require_pyarrow()
    raw = ds[0:count]
    mask = _element_mask(ds, raw)
    if FixedString.is_fixed_string(ds.dtype):
        return string_array(raw, FixedString.from_dtype(ds.dtype).nbytes, mask)
    if is_bool_dtype(ds.dtype):
        return pa.array(decode_bool(raw), mask=mask)
    return pa.array(_native(raw), mask=mask)


def _string_values_array(group: Any, count: int) -> Any:
    """A STRING_VALUES group as an Arrow ``large_string``.

    H5Col's OFFSETS plus a UTF-8 CHARS buffer is precisely Arrow's string
    layout, so both buffers go across untouched.
    """
    pa = require_pyarrow()
    chars = group[MEMBER_CHARS]
    if chars.dtype != np.uint8:
        raise ConformanceError(
            f"{chars.name!r}: STRING_VALUES CHARS must be uint8, got {chars.dtype}; "
            "the OFFSETS index bytes, so any other width misreads every value"
        )
    nbytes = int(group[MEMBER_OFFSETS][count])
    buffers = [
        _mask_validity(group, count),
        _offsets_buffer(group[MEMBER_OFFSETS], count),
        pa.py_buffer(chars[0:nbytes]),
    ]
    return pa.Array.from_buffers(pa.large_string(), count, buffers)


def _mask_validity(group: Any, count: int) -> Any:
    """The group's MASK member as an Arrow validity bitmap, or None if absent."""
    if MEMBER_MASK not in group:
        return None
    return _validity(~decode_bool(group[MEMBER_MASK][0:count]))


def _values_array(obj: Any, count: int) -> Any:
    """Whichever of the three VALUES forms *obj* is, as an Arrow array."""
    if isinstance(obj, h5py.Dataset):
        return _leaf_array(obj, count)
    cls = read_str_attr(obj, ATTR_CLASS)
    if cls == CLASS_STRING_VALUES:
        return _string_values_array(obj, count)
    if cls == CLASS_LIST_COLUMN:
        return list_array(obj, count)
    raise ConformanceError(f"{obj.name!r}: VALUES group has unexpected CLASS {cls!r}")


def list_array(group: Any, count: int) -> Any:
    """One level of a list column as an Arrow ``large_list``.

    Recurses through nesting, so an inner null — which no top-level mask can
    express — survives at whatever depth it was written.

    Parameters
    ----------
    group:
        A list column group, or an inner nesting level of one.
    count:
        How many entries of this level to wrap, counted from row 0.
    """
    pa = require_pyarrow()
    child = _values_array(group[MEMBER_VALUES], int(group[MEMBER_OFFSETS][count]))
    return pa.Array.from_buffers(
        pa.large_list(child.type),
        count,
        [_mask_validity(group, count), _offsets_buffer(group[MEMBER_OFFSETS], count)],
        children=[child],
    )
