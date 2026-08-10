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
=====================  =========================================

Missing rows become real Arrow nulls in every case, so an Arrow consumer never
sees the fill value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import categorical
from .booleans import decode_bool
from .reserved import ATTR_DESCRIPTION, ATTR_UNITS, ATTR_VALID_MAX, ATTR_VALID_MIN
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
    """Convert one scalar column (or *rows* of it) to an Arrow array."""
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
    return pa.array(raw, mask=mask)


def column_metadata(col: Column) -> dict[str, str]:
    """The column's HDF5 attributes, as Arrow field metadata.

    Arrow metadata is a flat string-to-string map, so numeric attributes are
    rendered with ``str``. The keys are prefixed with ``h5col.`` and survive a
    Parquet round trip, which is what makes them worth carrying at all.
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
    """A list column's HDF5 attributes, as Arrow field metadata."""
    meta: dict[str, str] = {}
    for key, value in (
        (ATTR_UNITS, col.units),
        ("units_vocabulary", col.units_vocabulary),
        (ATTR_DESCRIPTION, col.description),
    ):
        if value is not None:
            meta[f"{METADATA_PREFIX}{key}"] = str(value)
    return meta


def table_arrow(table: Any, columns: Any = None, rows: Any = None) -> Any:
    """Convert a table (or *rows* of it) to an Arrow table.

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
            values = col.read()
            if rows is not None:
                values = [values[int(i)] for i in rows]
            array = pa.array(values)
            meta = list_column_metadata(col)
        else:
            array = column_array(col, rows)
            meta = column_metadata(col)
        arrays.append(array)
        fields.append(pa.field(name, array.type, metadata=meta or None))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))
