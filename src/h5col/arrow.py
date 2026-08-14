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
from .booleans import bool_dtype, decode_bool, is_bool_dtype
from .categorical import choose_code_dtype
from .exceptions import ConformanceError, SchemaError
from .missing import recommended_fill, validate_fill_outside_range
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
    validate_attribute_names,
    validate_column_name,
)
from .specs import (
    ColumnSpec,
    LeafValuesSpec,
    ListColumnSpec,
    NestedListSpec,
    StringValuesSpec,
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


# --------------------------------------------------------------------------- #
# Import: Arrow schema -> H5Col specs
#
# The inverse of the export above is not a mirror image, because the mapping is
# not onto: Arrow has types H5Col cannot hold, permits column names HDF5 cannot
# store, and marks missing values with a null rather than a value drawn from the
# column's own domain. Everything that cannot be represented exactly is refused
# here, by name, rather than approximated.
# --------------------------------------------------------------------------- #

#: Arrow primitive types that map to a NumPy dtype one-for-one.
_PRIMITIVE_DTYPES = {
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float": "float32",
    "double": "float64",
}

#: Why each unsupported family is refused, so the error can say something useful.
_UNSUPPORTED = {
    "timestamp": "H5Col has no datetime type; store the epoch offsets as an "
    "integer column and describe the encoding in its attributes",
    "date": "H5Col has no date type; store the day or epoch offsets as an "
    "integer column",
    "time": "H5Col has no time type; store the offsets as an integer column",
    "duration": "H5Col has no duration type; store the count as an integer column",
    "decimal": "H5Col has no decimal type; store the unscaled integer and its "
    "scale separately",
    "struct": "H5Col columns are rank-1; flatten the struct into one column per field",
    "map": "H5Col has no map type",
    "union": "H5Col has no union type",
    "null": "a column of only nulls has no datatype to store",
    "binary": "H5Col stores text, not opaque bytes; a fixed-length string column "
    "needs a known encoding",
    "fixed_size_list": "H5Col list columns are variable-length; convert to a "
    "list type first",
}

#: Metadata keys under ``h5col.`` that this importer understands.
_KNOWN_METADATA = frozenset(
    {"units", "units_vocabulary", "description", "valid_min", "valid_max", "ordered"}
)


def _refuse(field: Any) -> None:
    """Raise for an Arrow type H5Col cannot represent exactly."""
    name = str(field.type)
    for family, why in _UNSUPPORTED.items():
        if name.startswith(family):
            raise SchemaError(
                f"column {field.name!r}: Arrow type {name} cannot be stored in "
                f"H5Col — {why}"
            )
    raise SchemaError(
        f"column {field.name!r}: Arrow type {name} is not supported by this importer"
    )


def _decoded_metadata(field: Any) -> dict[str, str]:
    """*field*'s metadata as ``str`` keys and values (Arrow stores bytes)."""
    raw = field.metadata or {}
    return {
        (k.decode() if isinstance(k, bytes) else k): (
            v.decode() if isinstance(v, bytes) else v
        )
        for k, v in raw.items()
    }


def _parse_bound(text: str, dtype: Any) -> Any:
    """Parse a stringified ``valid_min``/``valid_max`` back to the column's dtype.

    :func:`column_metadata` writes these with ``str``, so they arrive as text and
    have to be read back against the datatype they belong to rather than
    guessed. ``np.dtype(None)`` is ``float64``, so a caller with no dtype in
    hand must not reach this — it would turn any bound into a float in silence.
    """
    if dtype is None:
        raise SchemaError(f"cannot read the bound {text!r} without a datatype")
    try:
        return np.dtype(dtype).type(text)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"cannot read {text!r} as a {np.dtype(dtype)} bound") from exc


#: Annotations every kind of column spec can hold.
_COMMON_METADATA = frozenset({"units", "units_vocabulary", "description"})

#: Plus the valid range, which only a scalar column has a datatype to express.
_SCALAR_METADATA = _COMMON_METADATA | {"valid_min", "valid_max"}


def _annotations(field: Any, dtype: Any, *, allowed: frozenset[str]) -> dict[str, Any]:
    """Split *field*'s metadata into H5Col annotations and producer attributes.

    Keys under ``h5col.`` are the ones :func:`column_metadata` wrote and become
    the column's own annotations. Every other key is carried across as a
    producer attribute, where a name H5Col reserves is refused — otherwise a
    field carrying, say, a bare ``units`` key would silently shadow the value
    taken from ``h5col.units``.

    Parameters
    ----------
    field:
        The Arrow field whose metadata is being read.
    dtype:
        The column's datatype, which ``valid_min``/``valid_max`` are parsed
        against. None where the column kind has no scalar datatype.
    allowed:
        Which annotations this kind of column can hold. A spec model silently
        drops a field it does not define, so a key that cannot be stored is
        refused here rather than disappearing.
    """
    out: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in _decoded_metadata(field).items():
        if not key.startswith(METADATA_PREFIX):
            extra[key] = value
            continue
        short = key[len(METADATA_PREFIX) :]
        if short not in _KNOWN_METADATA:
            raise SchemaError(
                f"column {field.name!r}: unknown {METADATA_PREFIX}{short!r} "
                f"metadata key; this importer understands "
                f"{sorted(_KNOWN_METADATA)}"
            )
        if short not in allowed:
            raise SchemaError(
                f"column {field.name!r}: a {str(field.type)} column cannot carry "
                f"{METADATA_PREFIX}{short}"
            )
        if short in ("valid_min", "valid_max"):
            out[short] = _parse_bound(value, dtype)
        elif short == "ordered":
            out[short] = value.lower() == "true"
        else:
            out[short] = value
    validate_attribute_names(extra, field.name)
    if extra:
        out["attributes"] = extra
    return out


def _max_utf8_bytes(chunked: Any) -> int:
    """Widest encoded value in a string column, which sizes its fixed width.

    Scanning is the only way to learn this: Arrow strings are variable-length
    and H5Col's are not. The column is therefore sized to its data, so a later
    append of a longer value raises rather than truncating — pass a
    :class:`~h5col.ColumnSpec` to choose a wider budget deliberately.
    """
    widest = 0
    for chunk in chunked.chunks:
        for value in chunk.to_pylist():
            if value is not None:
                widest = max(widest, len(value.encode("utf-8")))
    return max(1, widest)


def _values_spec_from_type(field_name: str, arrow_type: Any, nullable: bool) -> Any:
    """The ``VALUES`` member for one level of a list column."""
    pa = require_pyarrow()
    if pa.types.is_large_string(arrow_type) or pa.types.is_string(arrow_type):
        return StringValuesSpec(nullable=nullable)
    if pa.types.is_large_list(arrow_type) or pa.types.is_list(arrow_type):
        inner = arrow_type.value_type
        return NestedListSpec(
            values=_values_spec_from_type(field_name, inner, nullable),
            nullable=nullable,
        )
    key = str(arrow_type)
    if pa.types.is_boolean(arrow_type):
        return LeafValuesSpec(dtype=bool_dtype())
    if key in _PRIMITIVE_DTYPES:
        return LeafValuesSpec(dtype=np.dtype(_PRIMITIVE_DTYPES[key]))
    raise SchemaError(
        f"column {field_name!r}: Arrow list element type {key} cannot be stored "
        f"in H5Col"
    )


def _spec_from_field(field: Any, column: Any) -> Any:
    """One :class:`ColumnSpec` or :class:`ListColumnSpec` for one Arrow field."""
    pa = require_pyarrow()
    ty = field.type
    key = str(ty)

    if pa.types.is_large_list(ty) or pa.types.is_list(ty):
        # A null anywhere below the top level needs somewhere to go, and the
        # element mask is the only place a list column has.
        nullable = column.null_count > 0
        return ListColumnSpec(
            name=field.name,
            values=_values_spec_from_type(field.name, ty.value_type, nullable=True),
            nullable=nullable,
            **_annotations(field, None, allowed=_COMMON_METADATA),
        )

    if pa.types.is_dictionary(ty):
        unified = column.unify_dictionaries()
        labels = unified.chunk(0).dictionary.to_pylist() if unified.num_chunks else []
        codes = choose_code_dtype(len(labels))
        annotations = _annotations(field, codes, allowed=_COMMON_METADATA | {"ordered"})
        # Arrow's own flag is authoritative; h5col.ordered is the fallback for a
        # table written before the export set the Arrow type's flag.
        annotations.setdefault("ordered", ty.ordered)
        return ColumnSpec(name=field.name, categories=labels, **annotations)

    if pa.types.is_boolean(ty):
        return ColumnSpec(
            name=field.name,
            dtype=bool_dtype(),
            **_annotations(field, None, allowed=_COMMON_METADATA),
        )

    if pa.types.is_large_string(ty) or pa.types.is_string(ty):
        text = FixedString(_max_utf8_bytes(column))
        return ColumnSpec(
            name=field.name,
            dtype=text,
            **_annotations(field, text.dtype, allowed=_SCALAR_METADATA),
        )

    if key in _PRIMITIVE_DTYPES:
        numeric = np.dtype(_PRIMITIVE_DTYPES[key])
        return ColumnSpec(
            name=field.name,
            dtype=numeric,
            **_annotations(field, numeric, allowed=_SCALAR_METADATA),
        )

    _refuse(field)


def specs_from_arrow(table: Any) -> list[Any]:
    """The column specs this package would import an Arrow table with.

    Returned so they can be inspected and adjusted before anything is written.
    Chunking and filters have no Arrow equivalent and so cannot be inferred at
    all. The string widths and category sets are inferred from the data, which
    is worth checking on a table you did not write::

        specs = h5col.specs_from_arrow(tbl)
        specs[2].chunks = 8192
        specs[2].filters = FilterPipeline([Shuffle(), Deflate(4)])

    Nothing is read from or written to a file here.

    Arrow's model is wider than H5Col's, so a type with no exact H5Col
    equivalent is refused by name rather than approximated: timestamps, dates,
    times, durations, decimals, structs, maps, unions, opaque binary and
    fixed-size lists.

    Field metadata under ``h5col.`` becomes the column's own annotations, the
    same keys :func:`column_metadata` writes. Any other metadata is carried
    across as a producer attribute, unless its name is one H5Col reserves.

    Arrow marks a missing value with a null; H5Col marks one with a value drawn
    from the column's own domain. Each column therefore gets a fill value, the
    recommended one for its datatype unless a supplied spec chooses otherwise —
    and a fill that already occurs in the data is refused, because those rows
    would read as missing. A boolean column with nulls is refused outright:
    H5Col forbids a fill there, so the nulls have nowhere to go.

    Parameters
    ----------
    table:
        A :class:`pyarrow.Table`.

    Raises
    ------
    SchemaError
        If two fields share a name (HDF5 links are unique), if a column's type
        has no H5Col equivalent, if a ``h5col.`` metadata key is not one this
        importer understands, if a boolean column holds nulls, or if a column's
        fill value occurs in its data.
    FillValueError
        If a column's fill value falls inside a valid range its metadata
        declares.
    ReservedNameError
        If a column name, or a producer metadata key, is a name H5Col reserves.
    """
    require_pyarrow()
    names = list(table.schema.names)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SchemaError(
            f"Arrow fields {duplicates} appear more than once; H5Col column "
            f"names are HDF5 links and must be unique"
        )
    specs = []
    for field in table.schema:
        validate_column_name(field.name)
        column = table.column(field.name)
        spec = _spec_from_field(field, column)
        if isinstance(spec, ColumnSpec):
            spec.fill_value = _fill_for(spec, field, column)
        specs.append(spec)
    return specs


def _fill_scalar(pa: Any, fill: Any, arrow_type: Any) -> Any:
    """The chosen H5Col fill value as an Arrow scalar of the column's type."""
    if isinstance(fill, bytes | bytearray):
        # A fixed-length string column's fill is stored as bytes; the Arrow
        # column holds text.
        return pa.scalar(bytes(fill).decode("utf-8"), arrow_type)
    value = fill.item() if hasattr(fill, "item") else fill
    return pa.scalar(value, arrow_type)


def _fill_occurs_in_data(column: Any, fill: Any) -> bool:
    """True if *fill* appears as a real value in *column*.

    A fill value is how H5Col spells "this row is missing", so a genuine value
    equal to it becomes unreadable: :meth:`~h5col.Column.is_missing`, the query
    layer and the Arrow export would all agree the row is absent, on a file that
    passes ``validate(deep=True)``.
    """
    pa = require_pyarrow()
    # Importing the submodule registers it on the package; reaching it through
    # `pa` then keeps its dynamically generated functions out of the type
    # checker's way, which cannot see them.
    import pyarrow.compute  # noqa: F401

    compute = pa.compute
    if fill is None:
        return False
    if isinstance(fill, float | np.floating) and np.isnan(fill):
        found = compute.any(compute.is_nan(column))
    else:
        found = compute.any(compute.equal(column, _fill_scalar(pa, fill, column.type)))
    # An all-null or empty column answers null rather than false.
    return found.as_py() is True


def _fill_for(spec: Any, field: Any, column: Any) -> Any:
    """The fill value to import *column* with, or None when it declares none.

    Raises rather than choosing something unsafe. There is no correct fill for a
    column that already contains the one H5Col recommends, and none at all for a
    boolean, which the convention forbids from declaring one.
    """
    if spec.is_boolean:
        if column.null_count:
            raise SchemaError(
                f"column {field.name!r}: H5Col boolean columns cannot declare a "
                f"fill value, so this column's {column.null_count} null "
                f"value(s) have nowhere to be stored; drop the nulls or import "
                f"the column as an integer"
            )
        return None
    if spec.is_categorical:
        # The fill code is chosen outside [0, ncategories) by construction, so
        # it cannot collide with a code that stands for a label.
        return None

    fill = spec.fill_value
    chosen_by_caller = fill is not None
    if not chosen_by_caller:
        fill = recommended_fill(spec.resolved_dtype())
    if _fill_occurs_in_data(column, fill):
        source = "supplied" if chosen_by_caller else "recommended"
        raise SchemaError(
            f"column {field.name!r}: the {source} fill value {fill!r} occurs in "
            f"the data, so those rows would read as missing; pass a ColumnSpec "
            f"with a fill_value the column does not contain"
        )
    validate_fill_outside_range(fill, spec.valid_min, spec.valid_max)
    return fill


def prepared_specs(table: Any, specs: Any = None) -> list[Any]:
    """Specs for importing *table*, with every fill value checked against data.

    The fill checks are not skippable by supplying specs: choosing a value that
    already occurs in the column is the one importing mistake that produces a
    conformant file with unreadable rows, so it is verified whatever the specs
    came from. Supplied specs are copied rather than mutated.

    Parameters
    ----------
    table:
        The :class:`pyarrow.Table` being imported.
    specs:
        A complete list of column specs, as from :func:`specs_from_arrow`, or
        None to infer them.

    Raises
    ------
    SchemaError
        If *specs* does not name exactly the table's columns.
    """
    inferred = (
        specs_from_arrow(table) if specs is None else [s.model_copy() for s in specs]
    )
    by_name = {s.name: s for s in inferred}
    wanted = list(table.schema.names)
    if sorted(by_name) != sorted(wanted):
        missing_specs = sorted(set(wanted) - set(by_name))
        extra = sorted(set(by_name) - set(wanted))
        raise SchemaError(
            f"specs must name exactly the table's columns; missing "
            f"{missing_specs}, unexpected {extra}"
        )
    out = []
    for field in table.schema:
        spec = by_name[field.name]
        if isinstance(spec, ColumnSpec):
            spec.fill_value = _fill_for(spec, field, table.column(field.name))
        out.append(spec)
    return out


def append_values(spec: Any, column: Any) -> Any:
    """One record-batch column in a form :meth:`~h5col.Table.append` accepts.

    Numeric columns have their nulls replaced by the column's fill value before
    NumPy sees them: ``to_numpy`` on a nullable integer column upcasts to
    ``float64`` and turns the nulls into NaN, which would change the datatype of
    every value in the column, not only the missing ones.

    Parameters
    ----------
    spec:
        The column's spec, which carries the fill value and the datatype.
    column:
        The batch's column, as a :class:`pyarrow.Array`.
    """
    pa = require_pyarrow()
    if spec.is_categorical or FixedString.is_fixed_string(spec.resolved_dtype()):
        # Labels and text go across as Python objects; `append` maps None to the
        # column's fill for both.
        return column.to_pylist()
    if spec.is_boolean:
        return column.to_numpy(zero_copy_only=False)
    filled = column
    if column.null_count:
        # `pyarrow.compute` is not an attribute of the package until something
        # imports the submodule. Every route here has already been through
        # _fill_occurs_in_data, which imports it, but relying on that would
        # leave this function broken by a change somewhere else and no test
        # able to see it — the submodule stays registered for the rest of the
        # process once any one caller has imported it. Reaching it through
        # `pa` afterwards keeps its dynamically generated functions out of the
        # type checker's way, which cannot see them.
        import pyarrow.compute  # noqa: F401

        filled = pa.compute.fill_null(
            column, _fill_scalar(pa, spec.fill_value, column.type)
        )
    return np.asarray(filled.to_numpy(zero_copy_only=False)).astype(
        spec.resolved_dtype(), copy=False
    )
