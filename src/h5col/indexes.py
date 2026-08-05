"""Search-index engine: validity tokens, ``SEARCH_INDEXES``, and index families.

Like :mod:`h5col.lists`, this engine is file-driven: every operation reads the
structure it needs from the file, so indexes built in one session are
maintainable and queryable after reopening.

The validity-token protocol (spec, "Index validity tokens") is the backbone:

- ``GENERATION`` (table group) identifies the current state of the column data;
  it is created the first time an index is added and incremented by every
  mutation thereafter.
- ``SOURCE_GENERATION`` / ``SOURCE_NROWS`` (each index dataset) name the table
  state the index content was built against. :func:`index_is_valid` is the
  consumer check; a failed check means "treat the index as absent", never an
  error.
- Write ordering keeps every crash state detectable. Mutations gated by the
  ``NROWS`` commit (append) write *future-valued* tokens **before** the index
  content; building or refreshing an index over an already-committed state
  writes content first and current-valued tokens **last**.
"""

from __future__ import annotations

import math
from typing import Any

import h5py
import numpy as np
from h5py import h5t

from . import references
from ._hdf5 import (
    read_bool_attr,
    read_str_attr,
    read_uint64_attr,
    write_ascii_token_attr,
    write_bool_attr,
    write_uint64_attr,
    write_utf8_attr,
)
from .exceptions import ConformanceError, ObjectReferenceError, SchemaError
from .missing import is_missing
from .ordering import is_orderable, is_spacepad, min_max, normalize_strings
from .reserved import (
    ATTR_CLASS,
    ATTR_DESCRIPTION,
    ATTR_EXHAUSTIVE,
    ATTR_FILL_TAIL_LENGTH,
    ATTR_GENERATION,
    ATTR_KIND,
    ATTR_NAN_TAIL_LENGTH,
    ATTR_NROWS,
    ATTR_ORDERED,
    ATTR_SEARCH_INDEX_LIST,
    ATTR_SOURCE_GENERATION,
    ATTR_SOURCE_NROWS,
    ATTR_VALUES,
    CLASS_LIST_COLUMN,
    GROUP_SEARCH_INDEXES,
    KIND_BITMAP,
    KIND_CHUNK_MINMAX,
    KIND_SORTED_ROWS,
    SEARCH_INDEX_KINDS,
    validate_index_dataset_name,
)
from .strings import FixedString

#: Byte target for one chunk of a search-index dataset. Index datasets scale
#: with the source column's *chunk count*, not its row count, so the column
#: chunk policy would waste a mostly-empty multi-MiB chunk here; 64 KiB keeps
#: index datasets compact while still amortizing appends.
INDEX_CHUNK_BYTES = 64 << 10

#: ``CHUNK_MINMAX`` compound fields, in required declaration order.
MINMAX_FIELDS = ("min", "max", "nan_count", "fill_count", "n")

#: Index kinds this implementation can build and maintain (grows in 4c).
SUPPORTED_KINDS = frozenset({KIND_CHUNK_MINMAX, KIND_SORTED_ROWS, KIND_BITMAP})


# --------------------------------------------------------------------------- #
# Validity tokens
# --------------------------------------------------------------------------- #
def _scalar_uint64(attrs: Any, name: str) -> int | None:
    """Read attribute *name* if it is a scalar uint64; None otherwise."""
    if name not in attrs:
        return None
    val = np.asarray(attrs[name])
    if val.shape != () or not (val.dtype.kind == "u" and val.dtype.itemsize == 8):
        return None
    return int(val)


def table_generation(table_group: Any) -> int | None:
    """The table's ``GENERATION``, or None when it carries none."""
    return read_uint64_attr(table_group, ATTR_GENERATION)


def ensure_generation(table_group: Any) -> int:
    """Return the table's ``GENERATION``, creating or repairing it when needed.

    Per the spec, a table acquires ``GENERATION`` the first time a search index
    is built over it ("writing ``GENERATION`` first if the table did not
    previously carry it"); building over an unchanged table is not a mutation,
    so no increment happens here.

    A *missing-with-indexes* or *malformed* ``GENERATION`` (rule-12 violations
    some foreign tool left behind) fails the strict validity check, so every
    token this producer would write against it is dead on arrival. It is
    repaired as scalar uint64 with a value strictly above the old value *and*
    above every index's ``SOURCE_GENERATION`` — a spurious increment is
    explicitly safe (it can only disable indexes, never validate stale ones),
    whereas any reused value could equal some index's token and spuriously
    validate content nobody has verified.
    """
    gen = _scalar_uint64(table_group.attrs, ATTR_GENERATION)
    if gen is not None:
        return gen
    malformed = ATTR_GENERATION in table_group.attrs
    floor = 0
    if malformed:
        try:
            floor = max(0, int(table_group.attrs[ATTR_GENERATION]))
        except (TypeError, ValueError):
            floor = 0
    indexes_present = search_index_datasets(table_group)
    if not malformed and not indexes_present:
        # A table without indexes acquires GENERATION fresh; no token can
        # reference it yet, so the recommended initial value is safe.
        write_uint64_attr(table_group, ATTR_GENERATION, 0)
        return 0
    for index_ds in indexes_present.values():
        src = _scalar_uint64(index_ds.attrs, ATTR_SOURCE_GENERATION)
        if src is not None and src > floor:
            floor = src
    if malformed:
        del table_group.attrs[ATTR_GENERATION]
    write_uint64_attr(table_group, ATTR_GENERATION, floor + 1)
    return floor + 1


def mutation_generation(table_group: Any) -> int | None:
    """The pre-mutation ``GENERATION`` (``g_old``) for append/truncate.

    Strict read: a well-formed token is returned as-is, and an absent one is
    None (a table that does not carry ``GENERATION`` skips the bump steps). A
    *malformed* token must not simply be incremented — its lenient integer
    value bypasses the safety property, because ``g_old + 1`` could equal some
    index's residue ``SOURCE_GENERATION`` and spuriously validate unverified
    content once step 5 rewrites the attribute as uint64. It is repaired via
    :func:`ensure_generation`, which picks a value above every source token.
    """
    gen = _scalar_uint64(table_group.attrs, ATTR_GENERATION)
    if gen is not None:
        return gen
    if ATTR_GENERATION in table_group.attrs:
        return ensure_generation(table_group)
    return None


def index_is_valid(index_ds: Any, table_group: Any) -> bool:
    """The consumer validity check for one search-index dataset.

    ``SOURCE_GENERATION == GENERATION AND SOURCE_NROWS == NROWS``, with any
    absent or wrong-datatype token failing the check. A False result means
    "behave as if the index were not present" — it is never an error.
    """
    gen = _scalar_uint64(table_group.attrs, ATTR_GENERATION)
    nrows = _scalar_uint64(table_group.attrs, ATTR_NROWS)
    src_gen = _scalar_uint64(index_ds.attrs, ATTR_SOURCE_GENERATION)
    src_nrows = _scalar_uint64(index_ds.attrs, ATTR_SOURCE_NROWS)
    if gen is None or nrows is None or src_gen is None or src_nrows is None:
        return False
    return src_gen == gen and src_nrows == nrows


def _write_tokens(index_ds: Any, generation: int, nrows: int) -> None:
    write_uint64_attr(index_ds, ATTR_SOURCE_GENERATION, generation)
    write_uint64_attr(index_ds, ATTR_SOURCE_NROWS, nrows)


# --------------------------------------------------------------------------- #
# Discovery and linkage
# --------------------------------------------------------------------------- #
def index_kind(index_ds: Any) -> str | None:
    """The dataset's ``KIND`` value, or None when absent or not a scalar string.

    A malformed ``KIND`` (non-string or non-scalar value) yields None so that
    no kind-dispatched code path ever acts on it; ``validate`` flags the
    malformed attribute separately.
    """
    if ATTR_KIND not in index_ds.attrs:
        return None
    value = index_ds.attrs[ATTR_KIND]
    if isinstance(value, bytes | np.bytes_):
        try:
            return bytes(value).rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        return value
    return None


def search_index_datasets(table_group: Any) -> dict[str, Any]:
    """Every search-index dataset (carries ``KIND``) under ``SEARCH_INDEXES``.

    A ``SEARCH_INDEXES`` child that is not a group (a misuse of the reserved
    name) holds no index datasets; ``validate`` flags it, the consumer paths
    simply see no indexes.
    """
    si_group = table_group.get(GROUP_SEARCH_INDEXES)
    if si_group is None or not isinstance(si_group, h5py.Group):
        return {}
    return {
        name: obj
        for name, obj in si_group.items()
        if isinstance(obj, h5py.Dataset) and ATTR_KIND in obj.attrs
    }


def column_datasets(table_group: Any) -> dict[str, Any]:
    """Direct-child rank-1 datasets of the table group (the scalar columns)."""
    return {
        name: obj
        for name, obj in table_group.items()
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1
    }


def column_index_datasets(table_group: Any, column_ds: Any) -> list[Any]:
    """Resolve the column's ``SEARCH_INDEX_LIST`` references, in order.

    Consumer-lenient: null, dangling, and unlinked references are skipped —
    their indexes are treated as absent, per the spec's tolerance rules —
    while ``validate`` reports them as rule-4 violations. A malformed
    (non-1-D) attribute raises, because no reference can be read from it.
    """
    if ATTR_SEARCH_INDEX_LIST not in column_ds.attrs:
        return []
    refs = np.asarray(column_ds.attrs[ATTR_SEARCH_INDEX_LIST])
    if refs.ndim != 1:
        raise ConformanceError(
            f"column {column_ds.name!r} SEARCH_INDEX_LIST must be a 1-D array "
            "of object references"
        )
    out: list[Any] = []
    for ref in refs:
        if references.is_null_ref(ref):
            continue
        try:
            obj = references.resolve(table_group, ref)
        except ObjectReferenceError:
            continue
        if obj.name is None:  # resolves, but the object was unlinked
            continue
        out.append(obj)
    return out


def find_index_column(table_group: Any, index_ds: Any) -> Any | None:
    """The column whose ``SEARCH_INDEX_LIST`` references *index_ds*, or None.

    The column-side attribute is the only linkage the spec defines — index
    datasets carry no back-pointer — so this scans the table's column datasets.
    A malformed (non-1-D) ``SEARCH_INDEX_LIST`` on some column is skipped, so
    one bad column cannot break lookups for every other column's indexes.

    An index claimed by more than one column violates the spec's "a single
    search-index dataset MUST NOT cover multiple columns"; there is then no
    correct answer, and silently picking one would make pruning against the
    wrong column's data possible — so this raises instead.
    """
    claimants: list[Any] = []
    for col_ds in column_datasets(table_group).values():
        if ATTR_SEARCH_INDEX_LIST not in col_ds.attrs:
            continue
        refs = np.asarray(col_ds.attrs[ATTR_SEARCH_INDEX_LIST])
        if refs.ndim != 1:
            continue
        for ref in refs:
            if references.is_null_ref(ref):
                continue
            try:
                resolved = references.resolve(table_group, ref)
            except ObjectReferenceError:
                continue
            if resolved.name == index_ds.name:
                claimants.append(col_ds)
                break
    if len(claimants) > 1:
        names = sorted(c.name.rsplit("/", 1)[-1] for c in claimants)
        raise ConformanceError(
            f"search index {index_ds.name!r} is referenced by multiple columns "
            f"{names}; an index dataset must cover exactly one column"
        )
    return claimants[0] if claimants else None


# --------------------------------------------------------------------------- #
# CHUNK_MINMAX: shape and content
# --------------------------------------------------------------------------- #
def source_chunk_len(column_ds: Any, nrows: int) -> int:
    """Rows per chunk of the source column (its full extent when contiguous)."""
    if column_ds.chunks is not None:
        return int(column_ds.chunks[0])
    return max(1, nrows)


def data_chunk_count(column_ds: Any, nrows: int) -> int:
    """Chunks of *column_ds* that contain logical-table rows.

    ``ceil(nrows / chunk_len)`` for a chunked column, ``1`` for a contiguous
    column, ``0`` when ``nrows == 0``. Tail-only chunks are not counted.
    """
    if nrows == 0:
        return 0
    if column_ds.chunks is None:
        return 1
    return math.ceil(nrows / column_ds.chunks[0])


def minmax_dtype(element_dtype: Any) -> np.dtype:
    """The ``CHUNK_MINMAX`` compound dtype for a column of *element_dtype*."""
    return np.dtype(
        [
            ("min", element_dtype),
            ("max", element_dtype),
            ("nan_count", "<u8"),
            ("fill_count", "<u8"),
            ("n", "<u8"),
        ]
    )


def _user_fill(dataset: Any) -> Any | None:
    """The dataset's user-defined fill value, or None when it declares none."""
    if dataset.id.get_create_plist().fill_value_defined() == 2:
        return dataset.fillvalue
    return None


def supported_index_dtype(dtype: Any) -> bool:
    """True if this implementation can build its index families over *dtype*.

    A producer subset of :func:`~h5col.ordering.is_orderable`, shared by every
    supported family: the spec also orders variable-length strings and opaque
    values, but this implementation does not build indexes over them (building
    an index is always optional for a producer).
    """
    if not is_orderable(dtype):
        return False
    dt = np.dtype(dtype)
    if h5py.check_string_dtype(dtype) is not None:
        return FixedString.is_fixed_string(dtype)
    return dt.kind in ("i", "u", "f", "b") or h5py.check_enum_dtype(dtype) is not None


#: Backwards-compatible name from sub-phase 4a; the same predicate applies to
#: every family this implementation builds.
supported_minmax_dtype = supported_index_dtype


def compute_chunk_minmax(column_ds: Any, nrows: int) -> np.ndarray:
    """Recompute the ``CHUNK_MINMAX`` entries for rows ``[0, nrows)``.

    This is the build oracle: creation, append maintenance, refresh, and deep
    validation all derive the index content from this one function.
    """
    dtype = column_ds.dtype
    if not supported_index_dtype(dtype):
        raise SchemaError(f"CHUNK_MINMAX is not supported over column dtype {dtype!r}")
    n_chunks = data_chunk_count(column_ds, nrows)
    entries = np.zeros(n_chunks, dtype=minmax_dtype(dtype))
    if n_chunks == 0:
        return entries

    fill = _user_fill(column_ds)
    is_float = np.dtype(dtype).kind == "f"
    is_string = FixedString.is_fixed_string(dtype)
    spacepad = is_spacepad(column_ds) if is_string else False
    chunk_len = source_chunk_len(column_ds, nrows)

    for cid in range(n_chunks):
        start = cid * chunk_len
        stop = min(nrows, start + chunk_len)
        raw = column_ds[start:stop]
        if is_string:
            raw = normalize_strings(raw, spacepad=spacepad)
        n = stop - start

        fill_mask = (
            is_missing(raw, fill) if fill is not None else np.zeros(n, dtype=np.bool_)
        )
        if is_float:
            nan_mask = np.isnan(raw)
            orderable = ~fill_mask & ~nan_mask
            entries[cid]["nan_count"] = int(nan_mask.sum())
        else:
            orderable = ~fill_mask
        entries[cid]["fill_count"] = int(fill_mask.sum())
        entries[cid]["n"] = n

        values = raw[orderable]
        if values.shape[0] > 0:
            vmin, vmax = min_max(values)
        elif fill is not None:
            # Placeholders: no non-missing, orderable element in this chunk.
            # Consumers recognize the state from the counts and must not use
            # these values in pruning decisions.
            vmin = vmax = fill
        else:
            # Only reachable for a float column with no user fill whose chunk
            # is all NaN; there is no fill to use, and nan_count == n already
            # marks the entry as a placeholder.
            vmin = vmax = np.nan
        entries[cid]["min"] = vmin
        entries[cid]["max"] = vmax
    return entries


# --------------------------------------------------------------------------- #
# SORTED_ROWS: content
# --------------------------------------------------------------------------- #
def compute_sorted_rows(column_ds: Any, nrows: int) -> tuple[np.ndarray, int, int]:
    """Recompute the ``SORTED_ROWS`` permutation for rows ``[0, nrows)``.

    Returns ``(permutation, fill_tail_length, nan_tail_length)``. The
    permutation is total and deterministic: the body is sorted under the H5Col
    order with ties broken by increasing row position (a stable argsort over
    rows already in increasing order), followed by the fill tail and then the
    NaN tail, each in increasing row order. A row goes to the NaN tail if its
    value is NaN, and otherwise to the fill tail if it matches a non-NaN fill;
    with a NaN fill every missing row is a NaN row and the fill tail is empty.
    """
    dtype = column_ds.dtype
    if not supported_index_dtype(dtype):
        raise SchemaError(f"SORTED_ROWS is not supported over column dtype {dtype!r}")

    raw = column_ds[0:nrows]
    if FixedString.is_fixed_string(dtype):
        raw = normalize_strings(raw, spacepad=is_spacepad(column_ds))

    if np.dtype(dtype).kind == "f":
        nan_mask = np.isnan(raw)
    else:
        nan_mask = np.zeros(nrows, dtype=np.bool_)
    fill = _user_fill(column_ds)
    if fill is not None:
        # With a NaN fill, is_missing == nan_mask and the intersection removal
        # leaves the fill tail empty, exactly as the spec requires.
        fill_mask = is_missing(raw, fill) & ~nan_mask
    else:
        fill_mask = np.zeros(nrows, dtype=np.bool_)

    rows = np.arange(nrows, dtype=np.uint64)
    body_mask = ~nan_mask & ~fill_mask
    body_rows = rows[body_mask]
    order = np.argsort(raw[body_mask], kind="stable")
    perm = np.concatenate([body_rows[order], rows[fill_mask], rows[nan_mask]])
    return perm, int(fill_mask.sum()), int(nan_mask.sum())


# --------------------------------------------------------------------------- #
# BITMAP: content
# --------------------------------------------------------------------------- #
def bitmap_bytes(nrows: int) -> int:
    """Bytes per bitmap row for a table of *nrows* (``ceil(nrows / 8)``)."""
    return (nrows + 7) // 8


def bitmap_values_dataset(table_group: Any, index_ds: Any) -> Any | None:
    """Resolve a ``BITMAP`` index's accompanying values dataset, or None.

    Consumer-lenient: a missing, non-scalar, null, dangling, or unlinked
    ``VALUES`` reference — or a target that is not a KIND-less rank-1 dataset
    sitting next to the index under ``SEARCH_INDEXES`` — yields None, making
    the bitmap unusable rather than an error; ``validate`` reports the
    violation separately.
    """
    if ATTR_VALUES not in index_ds.attrs:
        return None
    ref = index_ds.attrs[ATTR_VALUES]
    # A scalar object-reference attribute reads back as a bare (non-iterable)
    # h5py Reference; an array-valued attribute is malformed here.
    if not isinstance(ref, h5py.h5r.Reference) or references.is_null_ref(ref):
        return None
    try:
        target = references.resolve(table_group, ref)
    except ObjectReferenceError:
        return None
    if (
        not isinstance(target, h5py.Dataset)
        or target.name is None
        or target.ndim != 1
        or ATTR_KIND in target.attrs
    ):
        return None
    si_group = table_group.get(GROUP_SEARCH_INDEXES)
    if si_group is None or not isinstance(si_group, h5py.Group):
        return None
    if not target.name.startswith(f"{si_group.name}/"):
        return None
    return target


def compute_bitmap(column_ds: Any, nrows: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Recompute a ``BITMAP`` enumeration for rows ``[0, nrows)``.

    Returns ``(values, bits, exhaustive)``: the distinct non-missing values in
    H5Col order, the ``(K, ceil(nrows / 8))`` uint8 bit matrix with bit
    ``r % 8`` of byte ``r // 8`` set where row ``r`` equals the ``k``-th value
    (pad bits zero), and the ``exhaustive`` claim. NaN cannot be enumerated —
    IEEE 754 equality never matches it — so non-missing NaN elements are left
    out of the enumeration and make the claim ``exhaustive = False``.
    """
    dtype = column_ds.dtype
    if not supported_index_dtype(dtype):
        raise SchemaError(f"BITMAP is not supported over column dtype {dtype!r}")

    raw = column_ds[0:nrows]
    if FixedString.is_fixed_string(dtype):
        raw = normalize_strings(raw, spacepad=is_spacepad(column_ds))

    if np.dtype(dtype).kind == "f":
        nan_mask = np.isnan(raw)
    else:
        nan_mask = np.zeros(nrows, dtype=np.bool_)
    fill = _user_fill(column_ds)
    if fill is not None:
        fill_mask = is_missing(raw, fill)
    else:
        fill_mask = np.zeros(nrows, dtype=np.bool_)

    values = np.unique(raw[~fill_mask & ~nan_mask])
    exhaustive = not bool((nan_mask & ~fill_mask).any())

    n_bytes = bitmap_bytes(nrows)
    bits = np.zeros((len(values), n_bytes), dtype=np.uint8)
    for k, value in enumerate(values):
        # Missing rows cannot set a bit: a non-missing value never equals the
        # fill (that is the canonical missing test), and NaN equals nothing.
        bits[k] = np.packbits(
            np.asarray(raw == value, dtype=np.bool_), bitorder="little"
        )
    return values, bits, exhaustive


# --------------------------------------------------------------------------- #
# Build, refresh, and mutation maintenance (all families)
# --------------------------------------------------------------------------- #
def _create_index_dataset(
    si_group: Any, name: str, dtype: np.dtype, length: int
) -> Any:
    chunk_len = max(1, INDEX_CHUNK_BYTES // max(1, dtype.itemsize))
    return si_group.create_dataset(
        name, shape=(length,), maxshape=(None,), chunks=(chunk_len,), dtype=dtype
    )


def _entries_fit(index_ds: Any, n_entries: int) -> bool:
    """True if *index_ds* can hold *n_entries* — in place or by growing.

    A foreign index dataset need not be resizable (chunked + unlimited is only
    a SHOULD); one that cannot fit the new entry count cannot be maintained.
    """
    if index_ds.shape[0] >= n_entries:
        return True
    if index_ds.chunks is None:
        return False
    maxlen = index_ds.maxshape[0]
    return maxlen is None or maxlen >= n_entries


def _write_entries(index_ds: Any, entries: np.ndarray) -> None:
    """Write *entries* into ``[0, len)``, growing the dataset when needed.

    Never shrinks: entries beyond the data-bearing chunk count are permitted
    tail residue that consumers ignore (spec, "How consumers interpret NROWS").
    """
    n = len(entries)
    if index_ds.shape[0] < n:
        index_ds.resize((n,))
    if n:
        index_ds[:n] = entries


def _create_bitmap_datasets(
    si_group: Any, name: str, element_dtype: Any, k: int, n_bytes: int
) -> tuple[Any, Any]:
    """Create the bitmap matrix and its accompanying values dataset."""
    values_len = max(1, INDEX_CHUNK_BYTES // max(1, np.dtype(element_dtype).itemsize))
    values_ds = si_group.create_dataset(
        f"{name}_values",
        shape=(k,),
        maxshape=(None,),
        chunks=(values_len,),
        dtype=element_dtype,
    )
    bitmap_ds = si_group.create_dataset(
        name,
        shape=(k, n_bytes),
        maxshape=(None, None),
        chunks=(1, INDEX_CHUNK_BYTES),
        dtype=np.uint8,
    )
    return bitmap_ds, values_ds


def _exactly_resizable(dataset: Any, shape: tuple[int, ...]) -> bool:
    """True if *dataset* holds exactly *shape* now or can be resized to it.

    Unlike the grow-only families, a bitmap's first dimension is ``K``, the
    number of indexed values — a length *defined by* the datasets themselves,
    so stale residue rows would masquerade as indexed values. Rebuilds
    therefore resize to the exact shape, which requires a chunked dataset (or
    one that already matches).
    """
    if dataset.shape == shape:
        return True
    if dataset.chunks is None or len(dataset.shape) != len(shape):
        return False
    return all(
        m is None or m >= s for m, s in zip(dataset.maxshape, shape, strict=True)
    )


def create_chunk_minmax(
    table_group: Any,
    column_ds: Any,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Build a ``CHUNK_MINMAX`` index over *column_ds* and link it.

    Building over an already-committed table state writes the index content
    first and the current-valued tokens last, so a crash mid-build leaves a
    dataset that fails the validity check.

    Raises
    ------
    ConformanceError
        If the table group has no ``NROWS`` attribute.
    SchemaError
        If the column's datatype is unsupported, or ``SEARCH_INDEXES`` already
        holds a dataset of the chosen name.
    ReservedNameError
        If *name* is a H5Col reserved name.
    """
    nrows = read_uint64_attr(table_group, ATTR_NROWS)
    if nrows is None:
        raise ConformanceError("table group has no NROWS attribute")
    if not supported_minmax_dtype(column_ds.dtype):
        raise SchemaError(
            f"cannot build CHUNK_MINMAX over column {column_ds.name!r} with "
            f"dtype {column_ds.dtype!r}"
        )

    if name is None:
        col_name = column_ds.name.rsplit("/", 1)[-1]
        name = f"{col_name}__chunk_minmax"
    validate_index_dataset_name(name)

    # GENERATION comes first when the table did not previously carry it, so
    # the tokens written below have something to be compared against.
    generation = ensure_generation(table_group)
    si_group = table_group.require_group(GROUP_SEARCH_INDEXES)
    if name in si_group:
        raise SchemaError(f"SEARCH_INDEXES already contains a dataset {name!r}")

    entries = compute_chunk_minmax(column_ds, nrows)
    index_ds = _create_index_dataset(si_group, name, entries.dtype, len(entries))
    try:
        if len(entries):
            index_ds[...] = entries
        write_ascii_token_attr(index_ds, ATTR_KIND, KIND_CHUNK_MINMAX)
        if description is not None:
            write_utf8_attr(index_ds, ATTR_DESCRIPTION, description)
        # Content before tokens: the index becomes valid only once complete.
        _write_tokens(index_ds, generation, nrows)
        references.append_ref_to_array_attr(column_ds, ATTR_SEARCH_INDEX_LIST, index_ds)
    except BaseException:
        # Do not leave a half-built dataset behind (a KIND-less dataset in
        # SEARCH_INDEXES would violate consistency rule 3).
        del si_group[name]
        raise
    table_group.file.flush()
    return index_ds


def create_sorted_rows(
    table_group: Any,
    column_ds: Any,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Build a ``SORTED_ROWS`` index over *column_ds* and link it.

    Building over an already-committed table state writes the index content
    first and the current-valued tokens last, so a crash mid-build leaves a
    dataset that fails the validity check.

    Raises
    ------
    ConformanceError
        If the table group has no ``NROWS`` attribute.
    SchemaError
        If the column's datatype is unsupported, or ``SEARCH_INDEXES`` already
        holds a dataset of the chosen name.
    ReservedNameError
        If *name* is a H5Col reserved name.
    """
    nrows = read_uint64_attr(table_group, ATTR_NROWS)
    if nrows is None:
        raise ConformanceError("table group has no NROWS attribute")
    if not supported_index_dtype(column_ds.dtype):
        raise SchemaError(
            f"cannot build SORTED_ROWS over column {column_ds.name!r} with "
            f"dtype {column_ds.dtype!r}"
        )

    if name is None:
        col_name = column_ds.name.rsplit("/", 1)[-1]
        name = f"{col_name}__sorted_rows"
    validate_index_dataset_name(name)

    generation = ensure_generation(table_group)
    si_group = table_group.require_group(GROUP_SEARCH_INDEXES)
    if name in si_group:
        raise SchemaError(f"SEARCH_INDEXES already contains a dataset {name!r}")

    perm, fill_tail, nan_tail = compute_sorted_rows(column_ds, nrows)
    index_ds = _create_index_dataset(si_group, name, np.dtype("<u8"), nrows)
    try:
        if nrows:
            index_ds[...] = perm
        write_ascii_token_attr(index_ds, ATTR_KIND, KIND_SORTED_ROWS)
        if description is not None:
            write_utf8_attr(index_ds, ATTR_DESCRIPTION, description)
        write_uint64_attr(index_ds, ATTR_FILL_TAIL_LENGTH, fill_tail)
        write_uint64_attr(index_ds, ATTR_NAN_TAIL_LENGTH, nan_tail)
        write_bool_attr(index_ds, ATTR_ORDERED, True)
        # Content before tokens: the index becomes valid only once complete.
        _write_tokens(index_ds, generation, nrows)
        references.append_ref_to_array_attr(column_ds, ATTR_SEARCH_INDEX_LIST, index_ds)
    except BaseException:
        del si_group[name]
        raise
    table_group.file.flush()
    return index_ds


def create_bitmap(
    table_group: Any,
    column_ds: Any,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Build a ``BITMAP`` index over *column_ds*, with its values dataset.

    The accompanying values dataset is created as ``<name>_values`` next to
    the bitmap (its name carries no meaning; the linkage is the bitmap's
    scalar ``VALUES`` object reference). The enumeration is the distinct
    non-missing values in H5Col order, so ``ordered`` is true; ``exhaustive``
    is true unless the column holds non-missing NaN elements, which IEEE 754
    equality makes impossible to enumerate.

    Raises
    ------
    ConformanceError
        If the table group has no ``NROWS`` attribute.
    SchemaError
        If the column's datatype is unsupported, or ``SEARCH_INDEXES`` already
        holds the bitmap name or its ``<name>_values`` name.
    ReservedNameError
        If *name* (or ``<name>_values``) is a H5Col reserved name.
    """
    nrows = read_uint64_attr(table_group, ATTR_NROWS)
    if nrows is None:
        raise ConformanceError("table group has no NROWS attribute")
    if not supported_index_dtype(column_ds.dtype):
        raise SchemaError(
            f"cannot build BITMAP over column {column_ds.name!r} with "
            f"dtype {column_ds.dtype!r}"
        )

    if name is None:
        col_name = column_ds.name.rsplit("/", 1)[-1]
        name = f"{col_name}__bitmap"
    validate_index_dataset_name(name)
    validate_index_dataset_name(f"{name}_values")

    generation = ensure_generation(table_group)
    si_group = table_group.require_group(GROUP_SEARCH_INDEXES)
    for taken in (name, f"{name}_values"):
        if taken in si_group:
            raise SchemaError(f"SEARCH_INDEXES already contains a dataset {taken!r}")

    values, bits, exhaustive = compute_bitmap(column_ds, nrows)
    try:
        bitmap_ds, values_ds = _create_bitmap_datasets(
            si_group, name, column_ds.dtype, len(values), bitmap_bytes(nrows)
        )
        if len(values):
            values_ds[...] = values
            bitmap_ds[...] = bits
        write_ascii_token_attr(bitmap_ds, ATTR_KIND, KIND_BITMAP)
        if description is not None:
            write_utf8_attr(bitmap_ds, ATTR_DESCRIPTION, description)
        references.write_ref_attr(bitmap_ds, ATTR_VALUES, values_ds)
        write_bool_attr(bitmap_ds, ATTR_ORDERED, True)
        write_bool_attr(bitmap_ds, ATTR_EXHAUSTIVE, exhaustive)
        # Content before tokens: the index becomes valid only once complete.
        _write_tokens(bitmap_ds, generation, nrows)
        references.append_ref_to_array_attr(
            column_ds, ATTR_SEARCH_INDEX_LIST, bitmap_ds
        )
    except BaseException:
        # A failure at any point (even between the two creations) must not
        # leave a KIND-less or token-less dataset behind (rules 3 and 12).
        for leftover in (name, f"{name}_values"):
            if leftover in si_group:
                del si_group[leftover]
        raise
    table_group.file.flush()
    return bitmap_ds


# --------------------------------------------------------------------------- #
# Rebuild machinery shared by refresh and mutation maintenance
# --------------------------------------------------------------------------- #
def _compute_index_content(kind: str, column_ds: Any, nrows: int) -> Any:
    """Recompute the content of one index (read-only, never touches tokens)."""
    if kind == KIND_CHUNK_MINMAX:
        return compute_chunk_minmax(column_ds, nrows)
    if kind == KIND_SORTED_ROWS:
        return compute_sorted_rows(column_ds, nrows)
    if kind == KIND_BITMAP:
        return compute_bitmap(column_ds, nrows)
    raise SchemaError(f"cannot rebuild search-index kind {kind!r}")


def _minmax_dtype_compatible(index_dtype: np.dtype, column_dtype: Any) -> bool:
    """True if a ``CHUNK_MINMAX`` compound dtype can hold entries losslessly.

    Writing into a mismatched foreign compound would let HDF5 silently
    convert the ``min``/``max`` fields — clamping an int64 bound into an
    int32 field, say — and stamp the corrupted bounds valid.
    """
    if index_dtype.names != MINMAX_FIELDS:
        return False
    for counter in ("nan_count", "fill_count", "n"):
        fdt = index_dtype[counter]
        if not (fdt.kind == "u" and fdt.itemsize == 8):
            return False
    return _same_element_dtype(
        index_dtype["min"], column_dtype
    ) and _same_element_dtype(index_dtype["max"], column_dtype)


def _index_content_fits(
    kind: str,
    table_group: Any,
    index_ds: Any,
    column_ds: Any,
    content: Any,
    nrows: int,
) -> bool:
    """True if the recomputed *content* can be written into *index_ds*.

    A foreign index dataset need not be resizable (chunked + unlimited is only
    a SHOULD), may be too narrow to address every row, hold a datatype the
    content cannot be written into losslessly, or lack a usable values
    dataset; such an index cannot be maintained and is left untouched.
    """
    if kind == KIND_CHUNK_MINMAX:
        return (
            index_ds.ndim == 1
            and _minmax_dtype_compatible(index_ds.dtype, column_ds.dtype)
            and _entries_fit(index_ds, len(content))
        )
    if kind == KIND_SORTED_ROWS:
        if index_ds.ndim != 1 or index_ds.dtype.kind != "u":
            return False
        if nrows and int(np.iinfo(index_ds.dtype).max) < nrows - 1:
            return False
        return _entries_fit(index_ds, nrows)
    if kind == KIND_BITMAP:
        values, bits, _ = content
        if index_ds.ndim != 2 or index_ds.dtype != np.uint8:
            return False
        values_ds = bitmap_values_dataset(table_group, index_ds)
        if values_ds is None:
            return False
        # The spec requires the values dataset to hold "the same datatype as
        # the source column" — the *HDF5* datatype, so character set and enum
        # identity matter, not just the NumPy view of them.
        if not values_ds.id.get_type().equal(column_ds.id.get_type()):
            return False
        return _exactly_resizable(values_ds, values.shape) and _exactly_resizable(
            index_ds, bits.shape
        )
    return False


def _write_index_content(
    kind: str, table_group: Any, index_ds: Any, content: Any
) -> None:
    """Write recomputed *content* into *index_ds* (fit already verified)."""
    if kind == KIND_CHUNK_MINMAX:
        _write_entries(index_ds, content)
        return
    if kind == KIND_SORTED_ROWS:
        perm, fill_tail, nan_tail = content
        _write_entries(index_ds, perm.astype(index_ds.dtype))
        write_uint64_attr(index_ds, ATTR_FILL_TAIL_LENGTH, fill_tail)
        write_uint64_attr(index_ds, ATTR_NAN_TAIL_LENGTH, nan_tail)
        # The rebuilt permutation is total, so the mandatory `ordered` claim
        # is this producer's to (re)assert — a foreign index missing it would
        # otherwise be stamped valid in a state validate() rejects.
        write_bool_attr(index_ds, ATTR_ORDERED, True)
        return
    if kind == KIND_BITMAP:
        values, bits, exhaustive = content
        values_ds = bitmap_values_dataset(table_group, index_ds)
        if values_ds is None:  # unreachable after _index_content_fits
            raise SchemaError(f"bitmap {index_ds.name!r} has no usable values dataset")
        if values_ds.shape != values.shape:
            values_ds.resize(values.shape)
        if index_ds.shape != bits.shape:
            index_ds.resize(bits.shape)
        if len(values):
            values_ds[...] = values
            index_ds[...] = bits
        # The rebuilt enumeration is sorted and complete, so the semantic
        # claims are this producer's, whatever the previous writer declared.
        write_bool_attr(index_ds, ATTR_ORDERED, True)
        write_bool_attr(index_ds, ATTR_EXHAUSTIVE, exhaustive)
        return
    raise SchemaError(f"cannot rebuild search-index kind {kind!r}")


def _refresh_one(
    table_group: Any, index_ds: Any, column_ds: Any, nrows: int, generation: int
) -> bool:
    """Rebuild one index over committed state: content first, tokens last.

    Returns False — leaving the index entirely untouched, tokens included —
    when this producer cannot rebuild it (unsupported kind, unsupported
    element dtype, or content that does not fit the existing datasets).
    """
    kind = index_kind(index_ds)
    if kind not in SUPPORTED_KINDS or not supported_index_dtype(column_ds.dtype):
        return False
    # Compute first (read-only): a failure here must not have already
    # clobbered the tokens of a still-valid index.
    content = _compute_index_content(kind, column_ds, nrows)
    if not _index_content_fits(kind, table_group, index_ds, column_ds, content, nrows):
        return False
    _write_index_content(kind, table_group, index_ds, content)
    _write_tokens(index_ds, generation, nrows)
    return True


def refresh_index(table_group: Any, index_ds: Any, column_ds: Any) -> None:
    """Rebuild *index_ds* against the table's current committed state.

    The current ``GENERATION``/``NROWS`` are already committed, so the order is
    content first, tokens last (writing current-valued tokens before the
    content would let a mid-rebuild index pass the check).

    A currently *valid* index is left untouched: its content already describes
    the committed state (rule 9), and rewriting it in place would open a crash
    window where torn content sits behind still-passing tokens — the one state
    the token protocol exists to prevent.

    Raises
    ------
    ConformanceError
        If the table group has no ``NROWS`` attribute.
    """
    nrows = read_uint64_attr(table_group, ATTR_NROWS)
    if nrows is None:
        raise ConformanceError("table group has no NROWS attribute")
    generation = ensure_generation(table_group)
    if index_is_valid(index_ds, table_group):
        return
    if not _refresh_one(table_group, index_ds, column_ds, nrows, generation):
        raise SchemaError(
            f"cannot rebuild search index {index_ds.name!r} (unsupported kind "
            "or element dtype, or the recomputed content does not fit)"
        )


def append_refresh_indexes(table_group: Any, g_old: int, n_new: int) -> bool:
    """Mutation-protocol step 4: maintain every supported index for the new state.

    Used by any ``NROWS``-gated mutation — append and truncation share the
    same steps 4-6, with *n_new* the post-mutation row count. For each
    maintained index, the *future-valued* tokens
    (``SOURCE_GENERATION = g_old + 1``, ``SOURCE_NROWS = n_new``) are written
    **before** the content — the index fails the validity check throughout its
    own rebuild, because the new generation does not yet exist on the table
    group. The caller commits ``GENERATION`` and ``NROWS`` afterwards.

    Indexes this producer cannot rebuild are left untouched, with their tokens
    intact — including any index claimed by more than one column, where there
    is no correct column to rebuild against (the spec forbids the state, and
    ``validate`` reports it). Returns True when any index was rewritten, so
    the caller knows to flush.
    """
    columns = [
        col_ds
        for col_ds in column_datasets(table_group).values()
        if ATTR_SEARCH_INDEX_LIST in col_ds.attrs
    ]
    claims: dict[str, int] = {}
    for col_ds in columns:
        for index_ds in column_index_datasets(table_group, col_ds):
            claims[index_ds.name] = claims.get(index_ds.name, 0) + 1

    touched = False
    for col_ds in columns:
        for index_ds in column_index_datasets(table_group, col_ds):
            if claims.get(index_ds.name, 0) != 1:
                continue
            kind = index_kind(index_ds)
            if kind not in SUPPORTED_KINDS or not supported_index_dtype(col_ds.dtype):
                continue
            # Compute first (read-only): a failure here must not have already
            # clobbered the tokens of a still-valid index.
            content = _compute_index_content(kind, col_ds, n_new)
            if not _index_content_fits(
                kind, table_group, index_ds, col_ds, content, n_new
            ):
                continue
            _write_tokens(index_ds, g_old + 1, n_new)
            _write_index_content(kind, table_group, index_ds, content)
            touched = True
    return touched


def refresh_all_indexes(table_group: Any) -> int:
    """Rebuild every supported *stale* index against the committed state.

    Returns the number of indexes refreshed. Indexes this producer cannot
    rebuild are left untouched — and stay detectably stale if they already
    were. Currently valid indexes are also left untouched: they already
    describe the committed state, and rewriting them in place would open a
    crash window with torn content behind passing tokens.
    """
    nrows = read_uint64_attr(table_group, ATTR_NROWS)
    if nrows is None:
        raise ConformanceError("table group has no NROWS attribute")
    generation = ensure_generation(table_group)
    count = 0
    for index_ds in search_index_datasets(table_group).values():
        if index_is_valid(index_ds, table_group):
            continue
        col_ds = find_index_column(table_group, index_ds)
        if col_ds is None:
            continue
        if _refresh_one(table_group, index_ds, col_ds, nrows, generation):
            count += 1
    if count:
        table_group.file.flush()
    return count


# --------------------------------------------------------------------------- #
# Validation (consistency rules 3, 4, 9, 12)
# --------------------------------------------------------------------------- #
def _require_scalar_uint64(obj: Any, attr: str, what: str) -> None:
    if attr not in obj.attrs:
        raise ConformanceError(f"{what} has no {attr} attribute")
    val = np.asarray(obj.attrs[attr])
    if val.shape != ():
        raise ConformanceError(f"{what} {attr} must be scalar, got shape {val.shape}")
    if not (val.dtype.kind == "u" and val.dtype.itemsize == 8):
        raise ConformanceError(f"{what} {attr} must be uint64, got {val.dtype}")


def _require_bool_attr(
    obj: Any, attr: str, what: str, *, must_be_true: bool = False
) -> None:
    """Require a scalar H5Col boolean attribute (with the spec's tolerance).

    Accepts any enumeration whose base is a one-byte integer of either
    signedness with members exactly ``FALSE = 0`` and ``TRUE = 1``.
    """
    if attr not in obj.attrs:
        raise ConformanceError(f"{what} has no {attr} attribute")
    if np.asarray(obj.attrs[attr]).shape != ():
        raise ConformanceError(f"{what} {attr} must be a scalar attribute")
    tid = obj.attrs.get_id(attr).get_type()
    if (
        tid.get_class() != h5t.ENUM
        or tid.get_super().get_size() != 1
        or tid.get_nmembers() != 2
        or {
            tid.get_member_name(i): tid.get_member_value(i)
            for i in range(tid.get_nmembers())
        }
        != {b"FALSE": 0, b"TRUE": 1}
    ):
        raise ConformanceError(f"{what} {attr} must be a H5Col boolean")
    if must_be_true and not bool(obj.attrs[attr]):
        raise ConformanceError(f"{what} {attr} must be true")


def _require_kind_attr(obj: Any, name: str) -> None:
    """``KIND`` MUST be a scalar fixed-length ASCII string attribute."""
    if np.asarray(obj.attrs[ATTR_KIND]).shape != ():
        raise ConformanceError(f"search index {name!r} KIND must be a scalar attribute")
    tid = obj.attrs.get_id(ATTR_KIND).get_type()
    if (
        tid.get_class() != h5t.STRING
        or tid.is_variable_str()
        or tid.get_cset() != h5t.CSET_ASCII
    ):
        raise ConformanceError(
            f"search index {name!r} KIND must be a fixed-length ASCII string"
        )


def _same_element_dtype(field_dtype: np.dtype, column_dtype: np.dtype) -> bool:
    """Loose element-dtype equality across h5py's read-side conversions.

    h5py normalizes the H5Col boolean enum to NumPy ``bool`` when reading a
    compound field, so an exact dtype comparison would wrongly reject a
    conformant boolean min/max field.
    """
    from .booleans import is_bool_dtype

    if is_bool_dtype(field_dtype) and is_bool_dtype(column_dtype):
        return True
    return np.dtype(field_dtype) == np.dtype(column_dtype)


def _validate_minmax_structure(
    index_ds: Any, column_ds: Any, nrows: int, name: str
) -> None:
    """Cheap structural checks of a *valid* ``CHUNK_MINMAX`` dataset (rule 9)."""
    if index_ds.ndim != 1:
        raise ConformanceError(f"CHUNK_MINMAX {name!r} must be 1-D")
    if not is_orderable(column_ds.dtype):
        raise ConformanceError(
            f"CHUNK_MINMAX {name!r} is built over a column whose datatype "
            f"{column_ds.dtype!r} has no H5Col-defined order"
        )
    fields = index_ds.dtype.names
    if fields != MINMAX_FIELDS:
        raise ConformanceError(
            f"CHUNK_MINMAX {name!r} fields {fields} != {MINMAX_FIELDS}"
        )
    for counter in ("nan_count", "fill_count", "n"):
        fdt = index_ds.dtype[counter]
        if not (fdt.kind == "u" and fdt.itemsize == 8):
            raise ConformanceError(
                f"CHUNK_MINMAX {name!r} field {counter!r} must be uint64"
            )
    if not _same_element_dtype(index_ds.dtype["min"], column_ds.dtype):
        raise ConformanceError(
            f"CHUNK_MINMAX {name!r} min/max dtype does not match its column"
        )
    n_chunks = data_chunk_count(column_ds, nrows)
    if index_ds.shape[0] < n_chunks:
        raise ConformanceError(
            f"CHUNK_MINMAX {name!r} has {index_ds.shape[0]} entries but the "
            f"column has {n_chunks} data-bearing chunks"
        )
    if n_chunks:
        # The n field is derivable without reading column data: every chunk but
        # the last covers chunk_len rows.
        chunk_len = source_chunk_len(column_ds, nrows)
        expected = np.minimum(
            chunk_len, nrows - chunk_len * np.arange(n_chunks, dtype=np.int64)
        )
        got = index_ds[:n_chunks]["n"].astype(np.int64)
        if not np.array_equal(got, expected):
            raise ConformanceError(
                f"CHUNK_MINMAX {name!r} n fields do not match the column's "
                "chunk coverage"
            )


def _minmax_entries_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Field-wise equality of two entry arrays, with NaN == NaN for floats."""
    for field in MINMAX_FIELDS:
        fa, fb = a[field], b[field]
        if fa.dtype.kind == "f":
            if not np.array_equal(fa, fb, equal_nan=True):
                return False
        elif not np.array_equal(fa, fb):
            return False
    return True


def _validate_sorted_rows_structure(
    index_ds: Any, column_ds: Any, nrows: int, name: str
) -> None:
    """Structural rule-9 checks of a *valid* ``SORTED_ROWS`` dataset.

    Reads the permutation (it is the index) but never the column data.
    """
    if index_ds.ndim != 1:
        raise ConformanceError(f"SORTED_ROWS {name!r} must be 1-D")
    if index_ds.dtype.kind != "u":
        raise ConformanceError(
            f"SORTED_ROWS {name!r} must have an unsigned integer datatype, "
            f"got {index_ds.dtype}"
        )
    if not is_orderable(column_ds.dtype):
        raise ConformanceError(
            f"SORTED_ROWS {name!r} is built over a column whose datatype "
            f"{column_ds.dtype!r} has no H5Col-defined order"
        )
    if nrows and int(np.iinfo(index_ds.dtype).max) < nrows - 1:
        raise ConformanceError(
            f"SORTED_ROWS {name!r} datatype {index_ds.dtype} cannot address "
            f"every row of a {nrows}-row table"
        )
    if index_ds.shape[0] < nrows:
        raise ConformanceError(
            f"SORTED_ROWS {name!r} has {index_ds.shape[0]} entries but the "
            f"table has {nrows} rows"
        )
    _require_scalar_uint64(index_ds, ATTR_NAN_TAIL_LENGTH, f"SORTED_ROWS {name!r}")
    _require_scalar_uint64(index_ds, ATTR_FILL_TAIL_LENGTH, f"SORTED_ROWS {name!r}")
    _require_bool_attr(
        index_ds, ATTR_ORDERED, f"SORTED_ROWS {name!r}", must_be_true=True
    )
    nan_tail = int(index_ds.attrs[ATTR_NAN_TAIL_LENGTH])
    fill_tail = int(index_ds.attrs[ATTR_FILL_TAIL_LENGTH])
    if nan_tail + fill_tail > nrows:
        raise ConformanceError(
            f"SORTED_ROWS {name!r} tail lengths {fill_tail}+{nan_tail} exceed "
            f"NROWS {nrows}"
        )
    if nrows:
        perm = index_ds[:nrows]
        if int(perm.max()) >= nrows:
            raise ConformanceError(
                f"SORTED_ROWS {name!r} is not a permutation of [0, {nrows})"
            )
        counts = np.bincount(perm.astype(np.int64), minlength=nrows)
        if not bool((counts == 1).all()):
            raise ConformanceError(
                f"SORTED_ROWS {name!r} is not a permutation of [0, {nrows})"
            )


def _validate_bitmap_structure(
    table_group: Any, index_ds: Any, column_ds: Any, nrows: int, name: str
) -> None:
    """Structural rule-9 checks of a *valid* ``BITMAP`` dataset."""
    if index_ds.ndim != 2:
        raise ConformanceError(f"BITMAP {name!r} must be 2-D")
    if index_ds.dtype != np.uint8:
        raise ConformanceError(f"BITMAP {name!r} must be uint8, got {index_ds.dtype}")
    if ATTR_VALUES not in index_ds.attrs:
        raise ConformanceError(f"BITMAP {name!r} has no VALUES attribute")
    ref = index_ds.attrs[ATTR_VALUES]
    if not isinstance(ref, h5py.h5r.Reference):
        raise ConformanceError(
            f"BITMAP {name!r} VALUES must be a scalar object reference"
        )
    if references.is_null_ref(ref):
        raise ConformanceError(f"BITMAP {name!r} VALUES is a null reference")
    try:
        values_ds = references.resolve(table_group, ref)
    except ObjectReferenceError as exc:
        raise ConformanceError(
            f"BITMAP {name!r} VALUES reference does not resolve"
        ) from exc
    si_group = table_group.get(GROUP_SEARCH_INDEXES)
    si_prefix = f"{si_group.name}/" if isinstance(si_group, h5py.Group) else None
    if (
        not isinstance(values_ds, h5py.Dataset)
        or values_ds.name is None
        or si_prefix is None
        or not values_ds.name.startswith(si_prefix)
    ):
        raise ConformanceError(
            f"BITMAP {name!r} VALUES must resolve to a sibling dataset under "
            "SEARCH_INDEXES"
        )
    if ATTR_KIND in values_ds.attrs:
        raise ConformanceError(
            f"BITMAP {name!r} values dataset carries a KIND attribute; an "
            "accompanying dataset must not"
        )
    if values_ds.ndim != 1:
        raise ConformanceError(f"BITMAP {name!r} values dataset must be 1-D")
    # "In the same datatype as the source column" is the HDF5 datatype:
    # character set, enum identity, and padding are part of it, so the loose
    # NumPy-level comparison is not enough here.
    if not values_ds.id.get_type().equal(column_ds.id.get_type()):
        raise ConformanceError(
            f"BITMAP {name!r} values dataset dtype does not match its column"
        )
    k = values_ds.shape[0]
    if index_ds.shape[0] != k:
        raise ConformanceError(
            f"BITMAP {name!r} has {index_ds.shape[0]} rows but its values "
            f"dataset holds {k} values"
        )
    n_bytes = bitmap_bytes(nrows)
    if index_ds.shape[1] < n_bytes:
        raise ConformanceError(
            f"BITMAP {name!r} rows hold {index_ds.shape[1]} bytes but "
            f"ceil(NROWS / 8) is {n_bytes}"
        )
    for attr in (ATTR_ORDERED, ATTR_EXHAUSTIVE):
        if attr in index_ds.attrs:
            _require_bool_attr(index_ds, attr, f"BITMAP {name!r}")
    if nrows % 8 and k:
        # Pad bits (positions >= NROWS in the final data byte) MUST be 0.
        pad_mask = np.uint8((0xFF << (nrows % 8)) & 0xFF)
        if bool((index_ds[:, n_bytes - 1] & pad_mask).any()):
            raise ConformanceError(
                f"BITMAP {name!r} has nonzero padding bits at positions >= NROWS"
            )


def _deep_check_bitmap(
    table_group: Any, index_ds: Any, column_ds: Any, nrows: int, name: str
) -> None:
    """Semantic rule-9 check: the stored bits describe the column exactly."""
    values_ds = bitmap_values_dataset(table_group, index_ds)
    if values_ds is None:  # unreachable after _validate_bitmap_structure
        raise ConformanceError(f"BITMAP {name!r} has no usable values dataset")
    raw = column_ds[0:nrows]
    if FixedString.is_fixed_string(column_ds.dtype):
        raw = normalize_strings(raw, spacepad=is_spacepad(column_ds))
    stored = values_ds[...]
    if FixedString.is_fixed_string(values_ds.dtype):
        stored = normalize_strings(stored, spacepad=is_spacepad(values_ds))
    n_bytes = bitmap_bytes(nrows)
    for k in range(stored.shape[0]):
        expected = np.packbits(
            np.asarray(raw == stored[k], dtype=np.bool_), bitorder="little"
        )
        if not np.array_equal(index_ds[k, :n_bytes], expected):
            raise ConformanceError(
                f"BITMAP {name!r} bits for value index {k} do not describe "
                "its column (deep check)"
            )
    if read_bool_attr(index_ds, ATTR_EXHAUSTIVE):
        if np.dtype(column_ds.dtype).kind == "f":
            nan_mask = np.isnan(raw)
        else:
            nan_mask = np.zeros(nrows, dtype=np.bool_)
        fill = _user_fill(column_ds)
        fill_mask = (
            is_missing(raw, fill)
            if fill is not None
            else np.zeros(nrows, dtype=np.bool_)
        )
        if bool((nan_mask & ~fill_mask).any()):
            raise ConformanceError(
                f"BITMAP {name!r} claims an exhaustive enumeration but the "
                "column holds non-missing NaN values, which IEEE 754 equality "
                "cannot enumerate"
            )
        distinct = np.unique(raw[~fill_mask & ~nan_mask])
        if not bool(np.isin(distinct, stored).all()):
            raise ConformanceError(
                f"BITMAP {name!r} claims an exhaustive enumeration but misses "
                "distinct non-missing column values (deep check)"
            )


def validate_search_indexes(
    table_group: Any, nrows: int, *, deep: bool = False
) -> None:
    """Enforce consistency rules 3, 4, 12, and rule 9 for every supported kind.

    Rule 9 is applied to every index family this implementation understands
    (``CHUNK_MINMAX``, ``SORTED_ROWS``, ``BITMAP``); an index of an unsupported
    kind is skipped. Rule 9 applies only to indexes whose validity check passes;
    a stale index is exempt (consumers treat it as absent). Structural rule-9
    checks always run; the semantic check — recomputing the index from its
    column — is O(index build) and runs only with ``deep=True``.

    Raises
    ------
    ConformanceError
        On the first consistency violation found.
    """
    si_group = table_group.get(GROUP_SEARCH_INDEXES)
    if si_group is not None and not isinstance(si_group, h5py.Group):
        raise ConformanceError(
            "SEARCH_INDEXES is a reserved name for a group, "
            f"got {type(si_group).__name__}"
        )
    indexes: dict[str, Any] = {}
    accompanying: dict[str, Any] = {}
    if si_group is not None:
        for name, obj in si_group.items():
            if not isinstance(obj, h5py.Dataset):
                raise ConformanceError(
                    f"SEARCH_INDEXES contains a non-dataset object {name!r}"
                )
            if ATTR_KIND in obj.attrs:
                _require_kind_attr(obj, name)
                indexes[name] = obj
            else:
                accompanying[name] = obj

        # Rule 3: a KIND-less dataset is permitted only as an accompanying
        # dataset some search index REQUIRES. Of the defined families, only
        # BITMAP defines one (its VALUES dataset); a CHUNK_MINMAX index
        # carrying a VALUES reference cannot vouch for anything. Unknown kinds
        # get the benefit of the doubt — a future HEP may define accompanying
        # datasets for them, and consumers must not reject what they cannot
        # interpret.
        referenced: set[str] = set()
        for obj in indexes.values():
            kind = index_kind(obj)
            if kind in SEARCH_INDEX_KINDS and kind != KIND_BITMAP:
                continue
            if ATTR_VALUES in obj.attrs:
                ref = obj.attrs[ATTR_VALUES]
                if not isinstance(ref, h5py.h5r.Reference):
                    # A non-scalar VALUES cannot vouch (and would crash the
                    # null-reference test); rule 9 reports the malformation.
                    continue
                try:
                    target = references.resolve(table_group, ref)
                except ObjectReferenceError:
                    continue
                referenced.add(target.name)
        for name, obj in accompanying.items():
            if obj.name not in referenced:
                raise ConformanceError(
                    f"dataset {name!r} in SEARCH_INDEXES carries no KIND and is "
                    "not an accompanying dataset of any search index"
                )

    # Rule 12: with >= 1 search-index dataset, the table must carry a scalar
    # uint64 GENERATION and every index scalar uint64 source tokens.
    if indexes:
        _require_scalar_uint64(table_group, ATTR_GENERATION, "table group")
        for name, obj in indexes.items():
            _require_scalar_uint64(obj, ATTR_SOURCE_GENERATION, f"index {name!r}")
            _require_scalar_uint64(obj, ATTR_SOURCE_NROWS, f"index {name!r}")

    # A list column group MUST NOT carry SEARCH_INDEX_LIST (spec, "List column
    # attributes"): this revision defines no search indexes over list columns.
    for name, obj in table_group.items():
        if (
            isinstance(obj, h5py.Group)
            and read_str_attr(obj, ATTR_CLASS) == CLASS_LIST_COLUMN
            and ATTR_SEARCH_INDEX_LIST in obj.attrs
        ):
            raise ConformanceError(
                f"list column {name!r} must not carry SEARCH_INDEX_LIST"
            )

    # Rule 4: every SEARCH_INDEX_LIST reference resolves to a KIND-tagged
    # dataset under this table's SEARCH_INDEXES subgroup — and no index
    # dataset is claimed by more than one column ("a single search-index
    # dataset MUST NOT cover multiple columns").
    si_prefix = None if si_group is None else f"{si_group.name}/"
    claimed_by: dict[str, str] = {}
    columns = column_datasets(table_group)
    for col_name, col_ds in columns.items():
        if ATTR_SEARCH_INDEX_LIST not in col_ds.attrs:
            continue
        refs = np.asarray(col_ds.attrs[ATTR_SEARCH_INDEX_LIST])
        if refs.ndim != 1:
            raise ConformanceError(
                f"column {col_name!r} SEARCH_INDEX_LIST must be a 1-D array of "
                "object references"
            )
        for ref in refs:
            if references.is_null_ref(ref):
                raise ConformanceError(
                    f"column {col_name!r} SEARCH_INDEX_LIST contains a null reference"
                )
            try:
                target = references.resolve(table_group, ref)
            except ObjectReferenceError as exc:
                raise ConformanceError(
                    f"column {col_name!r} SEARCH_INDEX_LIST reference does not resolve"
                ) from exc
            # A reference to a deleted (unlinked) object may still dereference;
            # such a target has no path and is not under SEARCH_INDEXES.
            if target.name is None:
                raise ConformanceError(
                    f"column {col_name!r} SEARCH_INDEX_LIST reference resolves "
                    "to an unlinked object"
                )
            if (
                si_prefix is None
                or not isinstance(target, h5py.Dataset)
                or not target.name.startswith(si_prefix)
                or ATTR_KIND not in target.attrs
            ):
                raise ConformanceError(
                    f"column {col_name!r} SEARCH_INDEX_LIST entry {target.name!r} "
                    "is not a search-index dataset under SEARCH_INDEXES"
                )
            other = claimed_by.setdefault(target.name, col_name)
            if other != col_name:
                raise ConformanceError(
                    f"search index {target.name!r} is referenced by columns "
                    f"{other!r} and {col_name!r}; an index dataset must cover "
                    "exactly one column"
                )

    # Rule 9 for the families this implementation understands, on indexes
    # that pass the validity check. The semantic (deep) half recomputes the
    # index from its column; a conformant foreign index over a dtype this
    # builder cannot recompute (e.g. vlen strings) has no oracle, so its
    # deep check is skipped.
    for name, obj in indexes.items():
        kind = index_kind(obj)
        if kind not in SUPPORTED_KINDS:
            continue
        if not index_is_valid(obj, table_group):
            continue
        col_ds = find_index_column(table_group, obj)
        if col_ds is None:
            continue  # orphan index: no column claims it, nothing to describe
        deep_here = deep and supported_index_dtype(col_ds.dtype)
        if kind == KIND_CHUNK_MINMAX:
            _validate_minmax_structure(obj, col_ds, nrows, name)
            if deep_here:
                n_chunks = data_chunk_count(col_ds, nrows)
                expected = compute_chunk_minmax(col_ds, nrows)
                if not _minmax_entries_equal(obj[:n_chunks], expected):
                    raise ConformanceError(
                        f"CHUNK_MINMAX {name!r} content does not describe its "
                        "column (deep check)"
                    )
        elif kind == KIND_SORTED_ROWS:
            _validate_sorted_rows_structure(obj, col_ds, nrows, name)
            if deep_here:
                perm, fill_tail, nan_tail = compute_sorted_rows(col_ds, nrows)
                stored_perm = obj[:nrows].astype(np.uint64)
                if (
                    not np.array_equal(stored_perm, perm)
                    or int(obj.attrs[ATTR_FILL_TAIL_LENGTH]) != fill_tail
                    or int(obj.attrs[ATTR_NAN_TAIL_LENGTH]) != nan_tail
                ):
                    raise ConformanceError(
                        f"SORTED_ROWS {name!r} content does not describe its "
                        "column (deep check)"
                    )
        else:  # KIND_BITMAP
            _validate_bitmap_structure(table_group, obj, col_ds, nrows, name)
            if deep_here:
                _deep_check_bitmap(table_group, obj, col_ds, nrows, name)


__all__ = [
    "INDEX_CHUNK_BYTES",
    "MINMAX_FIELDS",
    "SUPPORTED_KINDS",
    "append_refresh_indexes",
    "bitmap_bytes",
    "bitmap_values_dataset",
    "column_datasets",
    "column_index_datasets",
    "compute_bitmap",
    "compute_chunk_minmax",
    "compute_sorted_rows",
    "create_bitmap",
    "create_chunk_minmax",
    "create_sorted_rows",
    "data_chunk_count",
    "ensure_generation",
    "find_index_column",
    "index_is_valid",
    "index_kind",
    "minmax_dtype",
    "mutation_generation",
    "refresh_all_indexes",
    "refresh_index",
    "search_index_datasets",
    "source_chunk_len",
    "supported_index_dtype",
    "supported_minmax_dtype",
    "table_generation",
    "validate_search_indexes",
]
