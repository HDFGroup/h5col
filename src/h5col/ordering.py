"""The H5Col canonical ordering: orderability and min/max under the defined order.

H5Col defines a total order for a fixed set of datatypes (spec, "Sorted-row
permutation index" / *Ordering*): integers arithmetically, floats by IEEE 754
over finite values and infinities (NaN unordered), booleans and enums by code,
and strings byte-wise over UTF-8 with trailing storage padding stripped.
``CHUNK_MINMAX`` and ``SORTED_ROWS`` indexes may only be built over orderable
datatypes; :func:`is_orderable` is that predicate.

NumPy's ``S``-dtype comparison is full-width ``memcmp`` over NUL-padded values,
which is order-equivalent to the spec's trailing-NUL-stripped byte-wise rule
(NUL sorts below every byte), so NULLTERM/NULLPAD fixed strings compare
natively. SPACEPAD strings need their trailing spaces stripped first
(:func:`normalize_strings`).
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
from h5py import h5t

from .exceptions import SchemaError


def is_orderable(dtype: Any) -> bool:
    """True if *dtype* has an H5Col-defined order.

    Excluded per the spec: object/region references, compound datatypes, array
    datatypes, and variable-length-array datatypes. Everything else the spec
    enumerates — integers, floats, booleans, strings (fixed and variable
    length), opaque, and enumerations — is orderable.
    """
    dt = np.dtype(dtype)
    if h5py.check_ref_dtype(dt) is not None:
        return False
    if dt.subdtype is not None or dt.names is not None:
        return False
    if h5py.check_string_dtype(dt) is not None:
        return True
    if h5py.check_vlen_dtype(dt) is not None:
        return False
    if h5py.check_enum_dtype(dt) is not None:
        return True
    if dt.kind in ("i", "u", "f", "b"):
        return True
    # Opaque values order byte-wise over their raw bytes (unstructured void).
    return dt.kind == "V"


def is_spacepad(dataset: Any) -> bool:
    """True if *dataset* stores fixed-length strings with space padding."""
    try:
        return bool(dataset.id.get_type().get_strpad() == h5t.STR_SPACEPAD)
    except Exception:
        return False


def normalize_strings(values: np.ndarray, *, spacepad: bool) -> np.ndarray:
    """Strip trailing storage padding so byte-wise comparison matches the spec.

    NUL padding needs no work (``memcmp`` order-equivalence above); space
    padding is stripped explicitly.
    """
    if spacepad:
        return np.char.rstrip(values, b" ")
    return values


def min_max(values: np.ndarray) -> tuple[Any, Any]:
    """Min and max of *values* under the H5Col order.

    The caller passes only orderable, non-missing, non-NaN elements — at least
    one. Flexible dtypes (fixed strings) cannot use NumPy reductions, so they
    sort instead.

    Raises
    ------
    SchemaError
        If *values* is empty.
    """
    if values.shape[0] == 0:
        raise SchemaError("min_max needs at least one element")
    if values.dtype.kind in ("S", "O", "V"):
        ordered = np.sort(values)
        return ordered[0], ordered[-1]
    return values.min(), values.max()
