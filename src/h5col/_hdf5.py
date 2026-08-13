"""Internal low-level HDF5 helpers shared by the table and column layers.

Not part of the public API. These wrap the awkward corners of h5py: creating a
rank-1 appendable column dataset with a correct fill value and filter pipeline,
extending it, and reading/writing the typed H5Col attributes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from h5py import h5d, h5p, h5s, h5t

from .booleans import encode_bool, is_bool_dtype
from .exceptions import SchemaError
from .filters import FilterPipeline
from .missing import masked_to_none
from .strings import FixedString, ascii_token_dtype

# Default (uncompressed) chunk-size policy. H5Col columns are write-once /
# read-often and frequently cloud-hosted, so chunks should be large: fewer chunks
# (less metadata), better compression, and fewer/larger object-store reads. The
# target scales with the file's raw-data chunk cache — which is per-dataset and
# defaults to 8 MiB on HDF5 >= 2.0 (1 MiB before) — so a bigger cache yields
# bigger chunks, clamped to stay friendly to readers that open with the default
# cache. An explicit chunk shape or ``default_chunk_bytes`` always overrides this.
MIN_CHUNK_BYTES = 2 << 20  # 2 MiB floor
MAX_CHUNK_BYTES = 8 << 20  # cap a modern default chunk cache
CACHE_FRACTION = 0.5  # aim for ~2 chunks resident per dataset


def _file_chunk_cache_bytes(group: Any) -> int:
    """The file's default raw-data chunk cache size (``rdcc_nbytes``), in bytes."""
    try:
        return int(group.file.id.get_access_plist().get_cache()[2])
    except Exception:
        return MIN_CHUNK_BYTES


def target_chunk_bytes(group: Any, override: int | None = None) -> int:
    """Uncompressed bytes to target for one default chunk.

    An explicit *override* (``Table.create(default_chunk_bytes=...)``) is used as
    given. Otherwise the target is ``CACHE_FRACTION`` of the file's chunk cache,
    clamped to ``[MIN_CHUNK_BYTES, MAX_CHUNK_BYTES]``.
    """
    if override is not None:
        return max(1, int(override))
    scaled = _file_chunk_cache_bytes(group) * CACHE_FRACTION
    return int(min(MAX_CHUNK_BYTES, max(MIN_CHUNK_BYTES, scaled)))


def default_chunk_len(
    dtype: Any, group: Any, default_chunk_bytes: int | None = None
) -> int:
    """Rows per chunk for a column of *dtype* under the default chunk policy."""
    itemsize = max(1, int(np.dtype(dtype).itemsize))
    return max(1, target_chunk_bytes(group, default_chunk_bytes) // itemsize)


def create_column_dataset(
    group: Any,
    name: str,
    dtype: Any,
    *,
    chunks: int | tuple[int, ...] | None = None,
    fill_value: Any = None,
    filters: FilterPipeline | None = None,
    initial_len: int = 0,
    default_chunk_bytes: int | None = None,
) -> Any:
    """Create a rank-1, unlimited-along-axis-0 column dataset.

    ``fill_value=None`` means no user-defined fill (used for boolean columns,
    which H5Col forbids from declaring one).

    Numeric and boolean columns are built through the low-level dataset-creation
    property list, which preserves the filter pipeline's exact order and each
    filter's optional flag. Fixed-length string columns go through h5py's
    high-level API instead, because its low-level ``set_fill_value`` is broken
    for strings; that path (via :meth:`FilterPipeline.to_h5py_kwargs`) cannot
    express an arbitrary filter order or optional flags.
    """
    chunk: tuple[int, ...]
    if chunks is None:
        chunk = (default_chunk_len(dtype, group, default_chunk_bytes),)
    elif isinstance(chunks, int):
        chunk = (chunks,)
    else:
        chunk = tuple(int(c) for c in chunks)

    if FixedString.is_fixed_string(dtype):
        kwargs: dict[str, Any] = {}
        if filters is not None:
            kwargs.update(filters.to_h5py_kwargs())
        if fill_value is not None:
            kwargs["fillvalue"] = fill_value
        return group.create_dataset(
            name,
            shape=(initial_len,),
            maxshape=(None,),
            chunks=chunk,
            dtype=dtype,
            **kwargs,
        )

    dcpl = h5p.create(h5p.DATASET_CREATE)
    dcpl.set_chunk(chunk)
    if fill_value is not None:
        dcpl.set_fill_value(np.asarray(fill_value, dtype=dtype))
    if filters is not None:
        filters.apply(dcpl)  # preserves declared order + per-filter optional flags
    space = h5s.create_simple((initial_len,), (h5s.UNLIMITED,))
    tid = h5t.py_create(dtype, logical=True)
    h5d.create(group.id, name.encode("utf-8"), tid, space, dcpl=dcpl)
    return group[name]


def extend_to(dataset: Any, new_len: int) -> None:
    """Grow *dataset*'s first dimension to at least *new_len* rows."""
    if dataset.shape[0] < new_len:
        dataset.resize((new_len,))


def row_positions(rows: Any, nrows: int, name: str) -> np.ndarray:
    """Normalize a scattered row selection to positions in ``[0, nrows)``.

    Accepts a sequence of integers or a boolean mask with one entry per row.
    A negative position counts back from the end, as it does in a slice, so
    ``-1`` is the last row. Positions may be given in any order and may repeat.

    A boolean mask has to be recognized rather than converted, because casting
    one to an integer dtype turns it into a run of ones and zeros — a silently
    wrong selection rather than an error.

    Parameters
    ----------
    rows:
        A sequence of integer positions, or a boolean mask with one entry per
        row. Positions may be in any order and may repeat.
    nrows:
        The column's row count, which positions are normalized and bounds
        checked against, and which a boolean mask must match in length.
    name:
        The column's name, used only to make the error messages specific.

    Raises
    ------
    IndexError
        If a position is out of range, or a boolean mask is the wrong length.
    TypeError
        If *rows* holds values that are neither integers nor booleans.
    ValueError
        If *rows* is not one-dimensional.
    """
    arr = np.asarray(rows)
    if arr.ndim == 0:
        raise TypeError(
            f"{rows!r} is not a row selection for column {name!r}: use an "
            f"integer, a slice, a sequence of positions, or a boolean mask"
        )
    if arr.ndim != 1:
        raise ValueError(f"rows must be a 1-D sequence, got {arr.ndim}-D")
    if arr.dtype == np.bool_:
        if arr.shape[0] != nrows:
            raise IndexError(
                f"a boolean mask needs one entry per row: got {arr.shape[0]} "
                f"for column {name!r}, which has {nrows} rows"
            )
        return np.flatnonzero(arr).astype(np.int64, copy=False)
    if arr.size == 0:
        # An empty list arrives as float64, so let it through before the dtype
        # check below; there are no positions to validate either way.
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"row positions must be integers, got dtype {arr.dtype} for column {name!r}"
        )
    idx = arr.astype(np.int64)
    negative = idx < 0
    if negative.any():
        idx[negative] += nrows
    out_of_range = (idx < 0) | (idx >= nrows)
    if out_of_range.any():
        first = int(np.flatnonzero(out_of_range)[0])
        raise IndexError(
            f"row {int(arr[first])} is out of range for column {name!r}, "
            f"which has {nrows} rows"
        )
    return idx


def gather_rows(dataset: Any, rows: np.ndarray, nrows: int) -> np.ndarray:
    """Read *rows* of *dataset* with chunk-aligned, coalesced block reads.

    *rows* must be sorted ascending and within ``[0, nrows)``. Consecutive
    chunks are merged into single contiguous reads, so a selection that lands
    in a few chunks touches only those chunks — the whole point being that a
    compressed column is decompressed chunk by chunk, and the chunks holding
    no wanted row are never fetched at all.

    A contiguous (unchunked) dataset has nothing to coalesce, so it is read
    once and indexed.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        # Nothing wanted: build the empty result without touching the data.
        return np.empty(0, dtype=dataset.dtype)
    if dataset.chunks is None:
        return dataset[0:nrows][rows]

    # A contiguous ascending run — every full scan, and any candidate set that
    # came out of whole chunks — is just a slice. Reading it directly skips the
    # chunk bookkeeping and, more to the point, the element-by-element copy
    # into `out` that the general path would make for every row.
    lo, hi = int(rows[0]), int(rows[-1])
    if rows.size == hi - lo + 1 and bool(np.all(np.diff(rows) == 1)):
        return dataset[lo : hi + 1]

    chunk_len = int(dataset.chunks[0])
    chunk_ids = np.unique(rows // chunk_len)
    # Merge runs of consecutive chunk ids into one read each.
    breaks = np.flatnonzero(np.diff(chunk_ids) != 1)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [chunk_ids.size]))

    out: np.ndarray | None = None
    for s, e in zip(starts, stops, strict=True):
        run_lo = int(chunk_ids[s]) * chunk_len
        run_hi = min(nrows, (int(chunk_ids[e - 1]) + 1) * chunk_len)
        block = dataset[run_lo:run_hi]
        if out is None:
            # Take the dtype from what h5py actually returned rather than from
            # dataset.dtype: enum columns (the H5Col boolean) come back as
            # NumPy bool, which is not the on-disk datatype.
            out = np.empty(rows.shape, dtype=block.dtype)
        lo = int(np.searchsorted(rows, run_lo))
        hi = int(np.searchsorted(rows, run_hi))
        out[lo:hi] = block[rows[lo:hi] - run_lo]
    assert out is not None
    return out


def substitute_fill_for_none(dataset: Any, values: Any, name: str) -> Any:
    """Return *values* with every missing element replaced by *dataset*'s fill value.

    H5Col marks a missing row with the column's fill value, so ``None`` in
    append data means "this row is missing" — the same reading list-column
    leaf elements already give it. A masked element of a
    :class:`numpy.ma.MaskedArray` means exactly that too, and is normalized to
    ``None`` first (see :func:`~h5col.missing.masked_to_none`) so both spellings
    take one path. A column that cannot represent a missing row (a boolean,
    which H5Col forbids from declaring a fill, or a foreign column that declares
    none) rejects them instead of coercing.

    Values that cannot hold a ``None`` — anything carrying a non-object dtype,
    such as a typed NumPy array or pandas Series — are returned unchanged, so
    the already-typed fast path is not scanned or copied. Inputs that are not
    sequences are returned as-is for the caller's own 1-D check to reject.

    Raises
    ------
    SchemaError
        If *values* holds a missing element and the column declares no fill
        value.
    """
    values = masked_to_none(values)
    kind = getattr(getattr(values, "dtype", None), "kind", None)
    if kind is not None and kind != "O":
        return values
    if isinstance(values, str | bytes | bytearray):
        return values
    try:
        items = list(values)
    except TypeError:
        return values
    if not any(v is None for v in items):
        return items
    if is_bool_dtype(dataset.dtype):
        raise SchemaError(
            f"column {name!r} is boolean and cannot hold a missing (None) row"
        )
    if dataset.id.get_create_plist().fill_value_defined() != 2:
        raise SchemaError(
            f"a missing (None) row in column {name!r} requires the column to "
            "declare a fill value"
        )
    fill = dataset.fillvalue
    return [fill if v is None else v for v in items]


def prepare_column_data(dtype: Any, values: Any) -> np.ndarray:
    """Encode *values* for writing into a column of *dtype*.

    Fixed-length strings are byte-checked (raising rather than truncating),
    booleans are domain-checked, and numeric values are cast to the column dtype.
    """
    if FixedString.is_fixed_string(dtype):
        return FixedString.from_dtype(dtype).encode(values)
    if is_bool_dtype(dtype):
        return encode_bool(values).astype(np.bool_)
    return np.asarray(values, dtype=dtype)


# --------------------------------------------------------------------------- #
# Typed attribute helpers
# --------------------------------------------------------------------------- #
def _to_str(value: Any) -> str:
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).rstrip(b"\x00").decode("utf-8")
    return str(value)


def write_ascii_token_attr(obj: Any, name: str, value: str) -> None:
    """Write a scalar fixed-length ASCII reserved-token attribute (CLASS, ...)."""
    obj.attrs.create(
        name, np.array(value.encode("ascii"), dtype=ascii_token_dtype(value))
    )


def write_utf8_attr(obj: Any, name: str, value: str) -> None:
    """Write a scalar fixed-length UTF-8 string attribute."""
    nbytes = len(value.encode("utf-8")) + 1
    obj.attrs.create(
        name, np.array(value.encode("utf-8"), dtype=FixedString(nbytes).dtype)
    )


def write_utf8_array_attr(obj: Any, name: str, values: Sequence[str]) -> None:
    """Write a 1-D fixed-length UTF-8 string array attribute (e.g. column-order)."""
    if len(values) == 0:
        obj.attrs.create(name, np.array([], dtype=FixedString(1).dtype))
        return
    nbytes = max(len(v.encode("utf-8")) for v in values) + 1
    dt = FixedString(nbytes).dtype
    arr = np.array([v.encode("utf-8") for v in values], dtype=dt)
    obj.attrs.create(name, arr)


def write_uint64_attr(obj: Any, name: str, value: int) -> None:
    """Write (creating or updating) a scalar ``uint64`` attribute (NROWS, ...).

    An existing attribute of the wrong shape or datatype (e.g. an ``int32``
    ``GENERATION`` left by a foreign tool) is deleted and recreated as scalar
    uint64 — ``attrs.modify`` alone would silently preserve the malformed
    datatype forever.
    """
    if name in obj.attrs:
        existing = np.asarray(obj.attrs[name])
        if existing.shape == () and existing.dtype == np.uint64:
            obj.attrs.modify(name, np.uint64(value))
            return
        del obj.attrs[name]
    obj.attrs.create(name, np.uint64(value))


def write_bool_attr(obj: Any, name: str, value: bool) -> None:
    """Write a scalar H5Col boolean attribute (the ``FALSE``/``TRUE`` enum).

    h5py maps NumPy ``bool`` to exactly the H5Col boolean datatype on disk.
    Like :func:`write_uint64_attr`, an existing attribute of the wrong shape or
    datatype is deleted and recreated rather than silently preserved.
    """
    if name in obj.attrs:
        existing = obj.attrs[name]
        if isinstance(existing, np.bool_):
            obj.attrs.modify(name, np.bool_(value))
            return
        del obj.attrs[name]
    obj.attrs.create(name, np.bool_(value))


def read_bool_attr(obj: Any, name: str) -> bool | None:
    """Read a scalar boolean attribute, or None when absent or not a boolean.

    h5py reads any one-byte ``FALSE``/``TRUE`` enum — exactly the set the
    H5Col tolerance rule admits — as NumPy ``bool``; anything else (a plain
    integer, a string, an array) is not a H5Col boolean and yields None.
    """
    if name not in obj.attrs:
        return None
    value = obj.attrs[name]
    if isinstance(value, np.bool_):
        return bool(value)
    return None


def read_str_attr(obj: Any, name: str) -> str | None:
    """Read a scalar string attribute as ``str`` (or None if absent)."""
    if name not in obj.attrs:
        return None
    return _to_str(obj.attrs[name])


def read_str_array_attr(obj: Any, name: str) -> list[str] | None:
    """Read a 1-D string attribute as a list of ``str`` (or None if absent)."""
    if name not in obj.attrs:
        return None
    return [_to_str(v) for v in obj.attrs[name]]


def read_uint64_attr(obj: Any, name: str) -> int | None:
    """Read a scalar integer attribute as ``int`` (or None if absent)."""
    if name not in obj.attrs:
        return None
    return int(obj.attrs[name])


def has_attr(obj: Any, name: str) -> bool:
    return name in obj.attrs


def write_extra_attributes(target: Any, attributes: dict[str, Any] | None) -> None:
    """Write producer-supplied attributes onto a column.

    Names are validated separately, before anything is created, so that an
    invalid one cannot leave a half-built column behind.

    Parameters
    ----------
    target:
        The dataset or group to write on.
    attributes:
        Name-to-value mapping, or None. ``str`` values are written as UTF-8;
        everything else goes through NumPy, so ints, floats, bools and arrays
        keep their datatype rather than being stringified.
    """
    for name, value in (attributes or {}).items():
        if isinstance(value, str):
            write_utf8_attr(target, name, value)
        else:
            target.attrs.create(name, np.asarray(value))
