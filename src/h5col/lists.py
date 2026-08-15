"""List columns: the H5Col offsets encoding for variable-length row values.

A list column is an HDF5 group (a direct child of the table group) with
``CLASS="LIST_COLUMN"`` and ``KIND="OFFSETS"``. It stores a variable-length
list per row using the Apache Arrow offsets layout: all elements are flattened
back-to-back into a ``VALUES`` member, and a monotonic ``OFFSETS`` dataset
records each entry's slice. ``VALUES`` is one of three things — a rank-1 *leaf*
dataset, a nested ``LIST_COLUMN`` group (lists of lists), or a ``STRING_VALUES``
group (variable-length UTF-8 via a second ``OFFSETS``/``CHARS`` level). An
optional ``MASK`` at any level distinguishes a null entry from an empty one.

The create/append/read/validate functions here are driven by the *file*
structure (not the Python spec), so a table can be reopened and appended without
its original :class:`~h5col.specs.ListColumnSpec`. Writing follows the H5Col
leaf-first order (deepest elements first, each enclosing ``OFFSETS`` last) so
that committed rows stay fully described at every moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from . import missing
from ._hdf5 import (
    create_column_dataset,
    extend_to,
    read_str_attr,
    write_ascii_token_attr,
    write_utf8_attr,
)
from .booleans import bool_dtype, decode_bool, encode_bool, is_bool_dtype
from .exceptions import ConformanceError, SchemaError
from .missing import recommended_fill, validate_fill_outside_range
from .reserved import (
    ATTR_CLASS,
    ATTR_DESCRIPTION,
    ATTR_KIND,
    ATTR_UNITS,
    ATTR_UNITS_VOCABULARY,
    ATTR_VALID_MAX,
    ATTR_VALID_MIN,
    CLASS_LIST_COLUMN,
    CLASS_STRING_VALUES,
    KIND_OFFSETS,
    MEMBER_CHARS,
    MEMBER_MASK,
    MEMBER_OFFSETS,
    MEMBER_VALUES,
    validate_column_name,
)
from .specs import LeafValuesSpec, ListColumnSpec, NestedListSpec, StringValuesSpec
from .strings import FixedString

_U8 = np.dtype("u8")


# --------------------------------------------------------------------------- #
# No variable-length datatypes below a list column (H5Col rule 11)
# --------------------------------------------------------------------------- #
def reject_vlen(dtype: Any) -> None:
    """Raise if *dtype* — or any datatype nested inside it — is variable-length.

    H5Col rule 11 forbids *any* HDF5 variable-length datatype anywhere below a
    list column, including one hidden inside a compound field or an array
    subtype. h5py's ``check_string_dtype`` / ``check_vlen_dtype`` only inspect
    the top level, so this descends into compound fields and array bases first.

    Parameters
    ----------
    dtype:
        Any dtype, including a compound or array dtype whose members are
        inspected recursively.
    """
    dt = np.dtype(dtype)
    if dt.subdtype is not None:
        reject_vlen(dt.subdtype[0])
        return
    if dt.fields is not None:
        for name in dt.names:
            reject_vlen(dt.fields[name][0])
        return
    info = h5py.check_string_dtype(dt)
    if info is not None and info.length is None:
        raise SchemaError(
            "variable-length strings are prohibited below a list column; "
            "use a STRING_VALUES member or a fixed-length string leaf"
        )
    if h5py.check_vlen_dtype(dt) is not None:
        raise SchemaError(
            "variable-length sequences are prohibited below a list column; "
            "use a nested list column instead"
        )


# --------------------------------------------------------------------------- #
# Spec validation (no file I/O)
# --------------------------------------------------------------------------- #
def validate_list_column_spec(spec: ListColumnSpec) -> None:
    """Validate a list column spec without touching the file.

    Parameters
    ----------
    spec:
        The spec to check. Its name and its whole ``values`` tree, however
        deeply nested, are validated.
    """
    validate_column_name(spec.name)
    _validate_values_spec(spec.values)


def _validate_values_spec(vs: Any) -> None:
    """Validate one ``values`` spec, descending through nested list levels.

    Parameters
    ----------
    vs:
        The spec for a ``VALUES`` member: a :class:`~h5col.specs.LeafValuesSpec`,
        a :class:`~h5col.specs.StringValuesSpec` (nothing to check) or a
        :class:`~h5col.specs.NestedListSpec`, which is recursed into so that one
        call covers a whole ``values`` tree. Anything else raises.
    """
    if isinstance(vs, LeafValuesSpec):
        dtype = vs.resolved_dtype()
        reject_vlen(dtype)
        if vs.is_boolean:
            if vs.fill_value is not None:
                raise SchemaError("boolean list leaf must not declare a fill value")
            if vs.valid_min is not None or vs.valid_max is not None:
                raise SchemaError("boolean list leaf must not declare valid_min/max")
        else:
            fill = (
                vs.fill_value if vs.fill_value is not None else recommended_fill(dtype)
            )
            validate_fill_outside_range(fill, vs.valid_min, vs.valid_max)
    elif isinstance(vs, StringValuesSpec):
        return
    elif isinstance(vs, NestedListSpec):
        _validate_values_spec(vs.values)
    else:  # pragma: no cover - guarded by Pydantic typing
        raise SchemaError(f"unsupported list values spec {type(vs).__name__}")


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #
def _create_offsets(
    group: Any, chunks: int | None, filters: Any, *, default_chunk_bytes: int | None
) -> Any:
    """Create a level's ``OFFSETS`` dataset, already holding its leading 0.

    Parameters
    ----------
    group:
        The list column level, or ``STRING_VALUES`` group, to create the
        ``OFFSETS`` dataset in.
    chunks:
        Rows per chunk, or ``None`` to leave the chunk size to
        *default_chunk_bytes*.
    filters:
        The filter pipeline for this dataset, or ``None`` for an unfiltered one.
    default_chunk_bytes:
        A byte target for one chunk, used only when *chunks* is ``None``.
    """
    ds = create_column_dataset(
        group,
        MEMBER_OFFSETS,
        _U8,
        chunks=chunks,
        fill_value=None,
        filters=filters,
        initial_len=1,
        default_chunk_bytes=default_chunk_bytes,
    )
    ds[0] = 0  # OFFSETS[0] MUST be 0
    return ds


def _create_mask(group: Any, *, default_chunk_bytes: int | None) -> Any:
    """Create a level's ``MASK`` dataset, empty and with no fill value.

    Parameters
    ----------
    group:
        The list column level, or ``STRING_VALUES`` group, to create the
        ``MASK`` dataset in.
    default_chunk_bytes:
        A byte target for one chunk. The dataset asks for no explicit chunk
        size of its own, so this is what sizes it.
    """
    return create_column_dataset(
        group,
        MEMBER_MASK,
        bool_dtype(),
        fill_value=None,
        default_chunk_bytes=default_chunk_bytes,
    )


def _create_leaf(
    parent: Any, leaf: LeafValuesSpec, *, default_chunk_bytes: int | None
) -> Any:
    """Create a leaf ``VALUES`` dataset and write its descriptive attributes.

    Parameters
    ----------
    parent:
        The list column level to create the ``VALUES`` dataset in.
    leaf:
        The leaf spec. Its dtype is rejected if variable-length; its chunks,
        filters, fill value and valid range go to the dataset, and its units,
        units vocabulary and description are written as attributes. A boolean
        leaf gets no fill value.
    default_chunk_bytes:
        A byte target for one chunk, used only when the spec sets no
        ``chunks``.
    """
    dtype = leaf.resolved_dtype()
    reject_vlen(dtype)
    if leaf.is_boolean:
        fill: Any = None
    else:
        fill = (
            leaf.fill_value if leaf.fill_value is not None else recommended_fill(dtype)
        )
        validate_fill_outside_range(fill, leaf.valid_min, leaf.valid_max)
    ds = create_column_dataset(
        parent,
        MEMBER_VALUES,
        dtype,
        chunks=leaf.chunks,
        fill_value=fill,
        filters=leaf.filters,
        default_chunk_bytes=default_chunk_bytes,
    )
    if leaf.valid_min is not None:
        ds.attrs.create(ATTR_VALID_MIN, np.asarray(leaf.valid_min, dtype=dtype))
    if leaf.valid_max is not None:
        ds.attrs.create(ATTR_VALID_MAX, np.asarray(leaf.valid_max, dtype=dtype))
    if leaf.units is not None:
        write_utf8_attr(ds, ATTR_UNITS, leaf.units)
    if leaf.units_vocabulary is not None:
        write_utf8_attr(ds, ATTR_UNITS_VOCABULARY, leaf.units_vocabulary)
    if leaf.description is not None:
        write_utf8_attr(ds, ATTR_DESCRIPTION, leaf.description)
    return ds


def _create_string_values(
    parent: Any, sv: StringValuesSpec, *, default_chunk_bytes: int | None
) -> Any:
    """Create a ``STRING_VALUES`` member with its ``OFFSETS`` and ``CHARS``.

    Parameters
    ----------
    parent:
        The list column level to create the ``VALUES`` group in.
    sv:
        The string values spec. Its ``chunks`` sizes both ``OFFSETS`` and
        ``CHARS``, its ``filters`` apply to ``CHARS`` alone, and ``nullable``
        adds a ``MASK``.
    default_chunk_bytes:
        A byte target for one chunk, used only when the spec sets no
        ``chunks``.
    """
    g = parent.create_group(MEMBER_VALUES)
    write_ascii_token_attr(g, ATTR_CLASS, CLASS_STRING_VALUES)
    _create_offsets(g, sv.chunks, None, default_chunk_bytes=default_chunk_bytes)
    create_column_dataset(
        g,
        MEMBER_CHARS,
        np.dtype("u1"),
        chunks=sv.chunks,
        fill_value=None,
        filters=sv.filters,
        default_chunk_bytes=default_chunk_bytes,
    )
    if sv.nullable:
        _create_mask(g, default_chunk_bytes=default_chunk_bytes)
    return g


def _create_values(
    parent: Any, values_spec: Any, *, default_chunk_bytes: int | None
) -> None:
    """Create the ``VALUES`` member of a list level, whichever kind it is.

    Parameters
    ----------
    parent:
        The list column level to create the ``VALUES`` member in.
    values_spec:
        The spec deciding what ``VALUES`` becomes: a leaf dataset, a
        ``STRING_VALUES`` group, or a nested ``LIST_COLUMN`` group whose own
        level is then created recursively. Anything else raises.
    default_chunk_bytes:
        A byte target for one chunk, passed down to every dataset created
        below *parent*.
    """
    if isinstance(values_spec, LeafValuesSpec):
        _create_leaf(parent, values_spec, default_chunk_bytes=default_chunk_bytes)
    elif isinstance(values_spec, StringValuesSpec):
        _create_string_values(
            parent, values_spec, default_chunk_bytes=default_chunk_bytes
        )
    elif isinstance(values_spec, NestedListSpec):
        g = parent.create_group(MEMBER_VALUES)
        write_ascii_token_attr(g, ATTR_CLASS, CLASS_LIST_COLUMN)
        write_ascii_token_attr(g, ATTR_KIND, KIND_OFFSETS)
        _create_list_level(
            g,
            values_spec.values,
            values_spec.nullable,
            values_spec.chunks,
            values_spec.filters,
            default_chunk_bytes=default_chunk_bytes,
        )
    else:  # pragma: no cover - guarded by Pydantic typing
        raise SchemaError(f"unsupported list values spec {type(values_spec).__name__}")


def _create_list_level(
    group: Any,
    values_spec: Any,
    nullable: bool,
    chunks: int | None,
    filters: Any,
    *,
    default_chunk_bytes: int | None,
) -> None:
    """Fill an existing group with one list level: ``OFFSETS``, ``MASK``, ``VALUES``.

    Parameters
    ----------
    group:
        The group to fill. Its ``CLASS`` and ``KIND`` attributes are written by
        the caller, not here.
    values_spec:
        The spec for this level's ``VALUES`` member.
    nullable:
        Whether this level gets a ``MASK``, which is what allows a null entry
        to be told apart from an empty one.
    chunks:
        Rows per chunk for this level's ``OFFSETS``.
    filters:
        The filter pipeline for this level's ``OFFSETS``.
    default_chunk_bytes:
        A byte target for one chunk, used wherever no explicit ``chunks`` is
        set.
    """
    _create_offsets(group, chunks, filters, default_chunk_bytes=default_chunk_bytes)
    if nullable:
        _create_mask(group, default_chunk_bytes=default_chunk_bytes)
    _create_values(group, values_spec, default_chunk_bytes=default_chunk_bytes)


def create_list_column(
    table_group: Any, spec: ListColumnSpec, *, default_chunk_bytes: int | None = None
) -> Any:
    """Create an empty list column group under *table_group* from *spec*.

    Parameters
    ----------
    table_group:
        The table group to create the column group in.
    spec:
        The list column's spec, including its nesting and its leaf values.
    default_chunk_bytes:
        Target chunk size for the datasets created here, when the spec sets no
        explicit ``chunks`` shape.
    """
    name = validate_column_name(spec.name)
    g = table_group.create_group(name)
    write_ascii_token_attr(g, ATTR_CLASS, CLASS_LIST_COLUMN)
    write_ascii_token_attr(g, ATTR_KIND, KIND_OFFSETS)
    if spec.units is not None:
        write_utf8_attr(g, ATTR_UNITS, spec.units)
    if spec.units_vocabulary is not None:
        write_utf8_attr(g, ATTR_UNITS_VOCABULARY, spec.units_vocabulary)
    if spec.description is not None:
        write_utf8_attr(g, ATTR_DESCRIPTION, spec.description)
    _create_list_level(
        g,
        spec.values,
        spec.nullable,
        spec.chunks,
        spec.filters,
        default_chunk_bytes=default_chunk_bytes,
    )
    return g


# --------------------------------------------------------------------------- #
# Encoding (pure: row values -> a write plan)
# --------------------------------------------------------------------------- #
@dataclass
class _LeafData:
    array: np.ndarray


@dataclass
class _StringData:
    byte_counts: list[int]
    mask: list[bool] | None
    buffer: np.ndarray  # uint8


@dataclass
class _LevelData:
    counts: list[int]
    mask: list[bool] | None
    child: _LeafData | _StringData | _LevelData


def _encode_level(level_group: Any, entries: list[Any]) -> _LevelData:
    """Encode *entries* for one list level into a plan the writer can apply.

    Parameters
    ----------
    level_group:
        The list column level the entries are destined for. It is only read
        here: whether it has a ``MASK`` decides if null entries are allowed,
        its ``VALUES`` member decides how the flattened elements are encoded,
        and its name goes into the error message.
    entries:
        One entry per row: an iterable of elements, or ``None`` for a null
        entry, which is allowed only when the level has a ``MASK``. A null
        entry contributes no elements and a count of 0.
    """
    nullable = MEMBER_MASK in level_group
    counts: list[int] = []
    mask: list[bool] | None = [] if nullable else None
    child_entries: list[Any] = []
    for e in entries:
        if e is None:
            if mask is None:
                raise SchemaError(
                    f"null entry in list {level_group.name!r} which has no MASK "
                    "(not nullable)"
                )
            counts.append(0)
            mask.append(False)
        else:
            elems = list(e)
            counts.append(len(elems))
            child_entries.extend(elems)
            if mask is not None:
                mask.append(True)
    child = _encode_values(level_group[MEMBER_VALUES], child_entries)
    return _LevelData(counts, mask, child)


def _encode_values(obj: Any, entries: list[Any]) -> Any:
    """Encode elements for a ``VALUES`` member, dispatching on what it is.

    Parameters
    ----------
    obj:
        The ``VALUES`` member the elements are destined for: a leaf dataset, or
        a group whose ``CLASS`` is ``STRING_VALUES`` or ``LIST_COLUMN``. Any
        other ``CLASS`` raises.
    entries:
        The elements to encode, already flattened by the enclosing level.
    """
    if isinstance(obj, h5py.Dataset):
        return _encode_leaf(obj, entries)
    cls = read_str_attr(obj, ATTR_CLASS)
    if cls == CLASS_STRING_VALUES:
        return _encode_string(obj, entries)
    if cls == CLASS_LIST_COLUMN:
        return _encode_level(obj, entries)
    raise ConformanceError(f"{obj.name!r}: VALUES group has unexpected CLASS {cls!r}")


def _encode_leaf(ds: Any, entries: list[Any]) -> _LeafData:
    """Encode *entries* into one array of the leaf dataset's datatype.

    Parameters
    ----------
    ds:
        The leaf ``VALUES`` dataset. Only its datatype and fill value are read;
        nothing is written to it here.
    entries:
        The elements to encode. ``None`` means a missing element and becomes
        the dataset's fill value, so a leaf declaring none rejects it — as does
        a boolean leaf, which the convention forbids a fill value.
    """
    dtype = ds.dtype
    if is_bool_dtype(dtype):
        if any(v is None for v in entries):
            raise SchemaError("a boolean list element cannot be missing (None)")
        arr = encode_bool(entries).astype(np.bool_) if entries else np.empty(0, bool)
        return _LeafData(arr)
    # One check for every non-boolean leaf. A fixed-length string leaf used to
    # fall back to empty bytes here, which is a real value rather than a marker:
    # with no fill value to compare against, the element would read back as an
    # empty string that is present, not as the missing one it was written for.
    has_fill = ds.id.get_create_plist().fill_value_defined() == 2
    if any(v is None for v in entries) and not has_fill:
        raise SchemaError(
            "a missing (None) leaf element requires the VALUES dataset to declare "
            "a fill value"
        )
    if FixedString.is_fixed_string(dtype):
        fs = FixedString.from_dtype(dtype)
        vals = [ds.fillvalue if v is None else v for v in entries]
        return _LeafData(fs.encode(vals) if vals else np.empty(0, dtype=dtype))
    arr = np.empty(len(entries), dtype=dtype)
    fillv = np.asarray(ds.fillvalue, dtype=dtype) if has_fill else None
    for i, v in enumerate(entries):
        arr[i] = fillv if v is None else v
    return _LeafData(arr)


def _encode_string(sv_group: Any, entries: list[Any]) -> _StringData:
    """Encode *entries* into one UTF-8 buffer plus per-element byte counts.

    Parameters
    ----------
    sv_group:
        The ``STRING_VALUES`` group the elements are destined for. It is only
        read here: whether it has a ``MASK`` decides if null elements are
        allowed, and its name goes into the error message.
    entries:
        The string elements to encode. A :class:`str` is encoded as UTF-8,
        anything else is taken as bytes. ``None`` marks a null element, which
        is allowed only when the group has a ``MASK``, and contributes no
        bytes.
    """
    nullable = MEMBER_MASK in sv_group
    byte_counts: list[int] = []
    mask: list[bool] | None = [] if nullable else None
    buf = bytearray()
    for v in entries:
        if v is None:
            if mask is None:
                raise SchemaError(
                    f"null string element in {sv_group.name!r} which has no MASK"
                )
            byte_counts.append(0)
            mask.append(False)
        else:
            b = v.encode("utf-8") if isinstance(v, str) else bytes(v)
            buf.extend(b)
            byte_counts.append(len(b))
            if mask is not None:
                mask.append(True)
    return _StringData(byte_counts, mask, np.frombuffer(bytes(buf), dtype="u1"))


# --------------------------------------------------------------------------- #
# Writing (leaf-first) a plan into the file's reserved tail
# --------------------------------------------------------------------------- #
def _write_level(level_group: Any, data: _LevelData, cur_count: int) -> None:
    """Write one level's plan into the file, leaf-first.

    The child elements go down first, then this level's ``MASK``, and its
    ``OFFSETS`` last, so the entries already committed stay fully described at
    every moment.

    Parameters
    ----------
    level_group:
        The list column level to write into. Its ``VALUES``, ``MASK`` and
        ``OFFSETS`` are extended as needed.
    data:
        The plan for this level, as produced by :func:`_encode_level`.
    cur_count:
        How many entries this level already holds. The new entries begin
        there, and ``OFFSETS[cur_count]`` says where their elements begin in
        the child.
    """
    offs = level_group[MEMBER_OFFSETS]
    base = int(offs[cur_count])
    k = len(data.counts)
    # 1) child elements first (leaf-first ordering).
    _write_values(level_group[MEMBER_VALUES], data.child, base)
    # 2) this level's MASK.
    if data.mask is not None:
        m = level_group[MEMBER_MASK]
        extend_to(m, cur_count + k)
        if k:
            m[cur_count : cur_count + k] = np.asarray(data.mask, dtype=bool)
    # 3) this level's OFFSETS last.
    extend_to(offs, cur_count + k + 1)
    if k:
        new_offsets = base + np.cumsum(np.asarray(data.counts, dtype=_U8), dtype=_U8)
        offs[cur_count + 1 : cur_count + k + 1] = new_offsets


def _write_values(obj: Any, data: Any, cur_count: int) -> None:
    """Write a ``VALUES`` plan, dispatching on which kind of plan *data* is.

    Parameters
    ----------
    obj:
        The ``VALUES`` member to write into, matching *data*: a leaf dataset, a
        ``STRING_VALUES`` group, or a nested list level. It is extended as
        needed.
    data:
        The plan for that member. A plan of no recognized kind writes nothing.
    cur_count:
        Where in *obj* the write begins: the element count already stored,
        which the enclosing level takes from its own ``OFFSETS``.
    """
    if isinstance(data, _LeafData):
        n = int(data.array.shape[0])
        extend_to(obj, cur_count + n)
        if n:
            obj[cur_count : cur_count + n] = data.array
    elif isinstance(data, _StringData):
        _write_string(obj, data, cur_count)
    elif isinstance(data, _LevelData):
        _write_level(obj, data, cur_count)


def _write_string(sv_group: Any, data: _StringData, cur_count: int) -> None:
    """Write a string plan into a ``STRING_VALUES`` group, characters first.

    Parameters
    ----------
    sv_group:
        The ``STRING_VALUES`` group to write into. Its ``CHARS``, ``MASK`` and
        ``OFFSETS`` are extended as needed, in that order.
    data:
        The plan for these elements, as produced by :func:`_encode_string`.
    cur_count:
        How many elements the group already holds. The new elements begin
        there, and ``OFFSETS[cur_count]`` says where their bytes begin in
        ``CHARS``.
    """
    svoffs = sv_group[MEMBER_OFFSETS]
    chars = sv_group[MEMBER_CHARS]
    base = int(svoffs[cur_count])
    ne = len(data.byte_counts)
    nb = int(data.buffer.shape[0])
    extend_to(chars, base + nb)
    if nb:
        chars[base : base + nb] = data.buffer
    if data.mask is not None:
        m = sv_group[MEMBER_MASK]
        extend_to(m, cur_count + ne)
        if ne:
            m[cur_count : cur_count + ne] = np.asarray(data.mask, dtype=bool)
    extend_to(svoffs, cur_count + ne + 1)
    if ne:
        new_offsets = base + np.cumsum(
            np.asarray(data.byte_counts, dtype=_U8), dtype=_U8
        )
        svoffs[cur_count + 1 : cur_count + ne + 1] = new_offsets


def append_list_column(level_group: Any, rows: list[Any], n_old: int) -> None:
    """Append *rows* (a list of per-row list values) to a list column group.

    Parameters
    ----------
    level_group:
        The list column group, or an inner nesting level of one.
    rows:
        One entry per new row: a list of values, or ``None`` for a null row.
    n_old:
        The row count before this append, which is where the new rows are
        written.
    """
    data = _encode_level(level_group, rows)
    _write_level(level_group, data, n_old)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_list_column(level_group: Any, count: int, *, start: int = 0) -> list[Any]:
    """Read *count* entries from row *start* as a list of (list | None).

    Only the rows asked for are read, at every level of nesting: the range's
    own ``OFFSETS`` say where its values begin and end in the child, so the
    child read is narrowed to that span, and so on down.

    The one thing to be careful about is that offsets are absolute positions
    in the child, while the child block just read begins at the range's first
    offset. Every offset therefore has to be rebased against that first one
    before it can index the block.

    Parameters
    ----------
    level_group:
        The list column group, or an inner nesting level of one.
    count:
        How many entries to read. Zero or fewer reads nothing and touches no
        dataset.
    start:
        The first row to read. Defaults to 0, so the older two-argument call
        still reads ``[0, count)``.
    """
    if count <= 0:
        return []
    offs = level_group[MEMBER_OFFSETS][start : start + count + 1]
    mask = None
    if MEMBER_MASK in level_group:
        mask = decode_bool(level_group[MEMBER_MASK][start : start + count])
    child_start, child_stop = int(offs[0]), int(offs[count])
    child_values = _read_values(
        level_group[MEMBER_VALUES], child_stop - child_start, start=child_start
    )
    out: list[Any] = []
    for i in range(count):
        if mask is not None and not mask[i]:
            out.append(None)
        else:
            lo = int(offs[i]) - child_start
            hi = int(offs[i + 1]) - child_start
            out.append(child_values[lo:hi])
    return out


def _read_values(obj: Any, count: int, *, start: int = 0) -> list[Any]:
    """Read *count* elements from a ``VALUES`` member, whichever kind it is.

    Parameters
    ----------
    obj:
        The ``VALUES`` member to read from: a leaf dataset, or a group whose
        ``CLASS`` is ``STRING_VALUES`` or ``LIST_COLUMN``. Any other ``CLASS``
        raises.
    count:
        How many elements to read.
    start:
        The first element to read, as an absolute position in *obj* — offsets
        are absolute, so this is what the enclosing level rebases against.
    """
    if isinstance(obj, h5py.Dataset):
        return _read_leaf(obj, count, start=start)
    cls = read_str_attr(obj, ATTR_CLASS)
    if cls == CLASS_STRING_VALUES:
        return _read_string(obj, count, start=start)
    if cls == CLASS_LIST_COLUMN:
        return read_list_column(obj, count, start=start)
    raise ConformanceError(f"{obj.name!r}: VALUES group has unexpected CLASS {cls!r}")


def _read_leaf(ds: Any, count: int, *, start: int = 0) -> list[Any]:
    """Read *count* leaf elements, turning fill values back into ``None``.

    Parameters
    ----------
    ds:
        The leaf ``VALUES`` dataset. Its datatype decides the decoding, and its
        fill value, when it declares one, decides which elements come back as
        ``None``.
    count:
        How many elements to read.
    start:
        The first element to read. Defaults to 0.
    """
    raw = ds[start : start + count]
    dtype = ds.dtype
    if is_bool_dtype(dtype):
        return list(decode_bool(raw))
    has_fill = ds.id.get_create_plist().fill_value_defined() == 2
    if FixedString.is_fixed_string(dtype):
        vals = FixedString.from_dtype(dtype).decode(raw)
        if has_fill:
            miss = missing.is_missing(raw, ds.fillvalue)
            return [None if miss[i] else vals[i] for i in range(count)]
        return list(vals)
    if has_fill:
        miss = missing.is_missing(raw, ds.fillvalue)
        return [None if miss[i] else raw[i] for i in range(count)]
    return list(raw)


def _read_string(sv_group: Any, count: int, *, start: int = 0) -> list[Any]:
    """Read *count* UTF-8 strings from a ``STRING_VALUES`` group.

    Parameters
    ----------
    sv_group:
        The ``STRING_VALUES`` group to read from. Only the span of ``CHARS``
        the wanted elements cover is read, and a ``MASK``, when present, says
        which of them come back as ``None``.
    count:
        How many elements to read. Zero or fewer reads nothing and touches no
        dataset.
    start:
        The first element to read. Defaults to 0.
    """
    if count <= 0:
        return []
    offs = sv_group[MEMBER_OFFSETS][start : start + count + 1]
    # As in read_list_column: CHARS offsets are absolute, the block is not.
    char_start, char_stop = int(offs[0]), int(offs[count])
    chars = sv_group[MEMBER_CHARS][char_start:char_stop]
    mask = None
    if MEMBER_MASK in sv_group:
        mask = decode_bool(sv_group[MEMBER_MASK][start : start + count])
    out: list[Any] = []
    for j in range(count):
        if mask is not None and not mask[j]:
            out.append(None)
        else:
            lo = int(offs[j]) - char_start
            hi = int(offs[j + 1]) - char_start
            out.append(bytes(chars[lo:hi]).decode("utf-8"))
    return out


# --------------------------------------------------------------------------- #
# Validation (H5Col consistency rules 10 and 11)
# --------------------------------------------------------------------------- #
def validate_list_column(level_group: Any, count: int) -> None:
    """Validate a list column subtree at *count* entries (recursively).

    Parameters
    ----------
    level_group:
        The list column group, or an inner nesting level of one.
    count:
        How many entries this level is expected to hold, which its ``OFFSETS``
        and ``MASK`` extents are checked against.
    """
    cls = read_str_attr(level_group, ATTR_CLASS)
    if cls != CLASS_LIST_COLUMN:
        raise ConformanceError(
            f"{level_group.name!r}: list column CLASS must be {CLASS_LIST_COLUMN!r}"
        )
    kind = read_str_attr(level_group, ATTR_KIND)
    if kind != KIND_OFFSETS:
        raise ConformanceError(
            f"{level_group.name!r}: list column KIND must be {KIND_OFFSETS!r}, "
            f"got {kind!r}"
        )
    allowed = {MEMBER_OFFSETS, MEMBER_VALUES, MEMBER_MASK}
    extra = set(level_group.keys()) - allowed
    if extra:
        raise ConformanceError(
            f"{level_group.name!r}: unexpected list column members {sorted(extra)}"
        )
    if MEMBER_OFFSETS not in level_group:
        raise ConformanceError(f"{level_group.name!r}: missing OFFSETS")
    if MEMBER_VALUES not in level_group:
        raise ConformanceError(f"{level_group.name!r}: missing VALUES")

    offs = level_group[MEMBER_OFFSETS]
    _check_offsets(offs, count, level_group.name)
    child_count = int(offs[count])
    if MEMBER_MASK in level_group:
        _check_mask_and_nulls(level_group[MEMBER_MASK], offs, count, level_group.name)
    _validate_values(level_group[MEMBER_VALUES], child_count)


def _check_offsets(ds: Any, count: int, where: str) -> None:
    """Check one ``OFFSETS`` dataset against the H5Col rules for it.

    Parameters
    ----------
    ds:
        The dataset to check. It must be rank-1 and uint64, extend to at least
        ``count + 1``, start at 0 and never decrease.
    count:
        How many entries the enclosing level holds. The first ``count + 1``
        offsets describe them and are the only ones looked at; anything in the
        reserved tail above them is left alone.
    where:
        The name of the group holding *ds*, used only to make the error
        messages specific.
    """
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
        raise ConformanceError(f"{where!r}/OFFSETS must be a rank-1 dataset")
    if not (ds.dtype.kind == "u" and ds.dtype.itemsize == 8):
        raise ConformanceError(f"{where!r}/OFFSETS must be uint64")
    if ds.shape[0] < count + 1:
        raise ConformanceError(
            f"{where!r}/OFFSETS extent {ds.shape[0]} < entry count + 1 ({count + 1})"
        )
    o = ds[0 : count + 1]
    if int(o[0]) != 0:
        raise ConformanceError(f"{where!r}/OFFSETS[0] must be 0")
    if count and bool(np.any(o[1:] < o[:-1])):
        raise ConformanceError(
            f"{where!r}/OFFSETS must be monotonically non-decreasing"
        )


def _check_mask_and_nulls(m: Any, offs: Any, count: int, where: str) -> None:
    """Check a ``MASK`` dataset, and that every null entry spans no elements.

    Parameters
    ----------
    m:
        The ``MASK`` dataset to check. It must be rank-1, of the H5Col boolean
        datatype, and extend to at least *count*.
    offs:
        The sibling ``OFFSETS`` dataset, read to confirm that a false mask
        entry has an empty span. Its own validity is checked elsewhere.
    count:
        How many entries to check. Zero returns at once, without reading
        either dataset.
    where:
        The name of the group holding *m*, used only to make the error
        messages specific.
    """
    if not isinstance(m, h5py.Dataset) or m.ndim != 1:
        raise ConformanceError(f"{where!r}/MASK must be a rank-1 dataset")
    if not is_bool_dtype(m.dtype):
        raise ConformanceError(f"{where!r}/MASK must be the H5Col boolean datatype")
    if m.shape[0] < count:
        raise ConformanceError(
            f"{where!r}/MASK extent {m.shape[0]} < entry count ({count})"
        )
    if count == 0:
        return
    mv = decode_bool(m[0:count])
    ov = offs[0 : count + 1]
    for i in range(count):
        if not mv[i] and int(ov[i + 1]) != int(ov[i]):
            raise ConformanceError(
                f"{where!r}: null entry {i} must have OFFSETS[i+1] == OFFSETS[i]"
            )


def _validate_values(obj: Any, count: int) -> None:
    """Validate a ``VALUES`` member, descending into a nested one.

    Parameters
    ----------
    obj:
        The ``VALUES`` member to validate: a leaf dataset, which must be
        rank-1, free of variable-length datatypes and long enough, or a group
        whose ``CLASS`` is ``STRING_VALUES`` or ``LIST_COLUMN``. Any other
        ``CLASS`` raises.
    count:
        How many elements this member is expected to hold, which its extent is
        checked against.
    """
    if isinstance(obj, h5py.Dataset):
        try:
            reject_vlen(obj.dtype)  # rule 11
        except SchemaError as exc:
            raise ConformanceError(str(exc)) from exc
        if obj.ndim != 1:
            raise ConformanceError(f"{obj.name!r}: leaf VALUES must be rank-1")
        if obj.shape[0] < count:
            raise ConformanceError(
                f"{obj.name!r}: leaf VALUES extent {obj.shape[0]} < element count "
                f"({count})"
            )
        return
    cls = read_str_attr(obj, ATTR_CLASS)
    if cls == CLASS_STRING_VALUES:
        _validate_string_values(obj, count)
    elif cls == CLASS_LIST_COLUMN:
        validate_list_column(obj, count)
    else:
        raise ConformanceError(
            f"{obj.name!r}: VALUES group has unexpected CLASS {cls!r}"
        )


def _validate_string_values(g: Any, count: int) -> None:
    """Validate a ``STRING_VALUES`` group and its members.

    Parameters
    ----------
    g:
        The group to check. It may hold nothing beyond ``OFFSETS``, ``CHARS``
        and ``MASK``; the first two are required, and ``CHARS`` must be a
        rank-1 uint8 dataset extending to at least the byte count its offsets
        describe.
    count:
        How many elements the group is expected to hold, which its ``OFFSETS``
        and ``MASK`` are checked against.
    """
    allowed = {MEMBER_OFFSETS, MEMBER_CHARS, MEMBER_MASK}
    extra = set(g.keys()) - allowed
    if extra:
        raise ConformanceError(
            f"{g.name!r}: unexpected STRING_VALUES members {sorted(extra)}"
        )
    if MEMBER_OFFSETS not in g or MEMBER_CHARS not in g:
        raise ConformanceError(f"{g.name!r}: STRING_VALUES needs OFFSETS and CHARS")
    offs = g[MEMBER_OFFSETS]
    _check_offsets(offs, count, g.name)
    chars = g[MEMBER_CHARS]
    if not (chars.dtype.kind == "u" and chars.dtype.itemsize == 1) or chars.ndim != 1:
        raise ConformanceError(f"{g.name!r}/CHARS must be a rank-1 uint8 dataset")
    nb = int(offs[count])
    if chars.shape[0] < nb:
        raise ConformanceError(
            f"{g.name!r}/CHARS extent {chars.shape[0]} < byte count ({nb})"
        )
    if MEMBER_MASK in g:
        _check_mask_and_nulls(g[MEMBER_MASK], offs, count, g.name)
