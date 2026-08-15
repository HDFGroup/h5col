"""Opaque columns: raw bytes of a fixed width per row.

HDF5's opaque datatype (``H5T_OPAQUE``) stores a fixed number of bytes per
value with no interpretation attached — digests, ciphertext, packed records,
anything whose meaning lives outside the file. H5Col permits it as a column
datatype and gives it a sorting order and a search-index hash, but the
recommended fill-value table has no row for it, because no byte string can be
reserved on the grounds of being out of range.

This module supplies the one that H5Col writes by default, and the predicate
that tells a plain opaque datatype apart from the compound and array datatypes
h5py also reads through NumPy's ``V`` kind.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Opening bytes of an opaque column's recommended fill value: ASCII ``FILL``,
#: so the value announces itself in a hex dump rather than looking like data.
OPAQUE_FILL_MAGIC = b"FILL"


def is_opaque_dtype(dtype: Any) -> bool:
    """True for a plain opaque datatype — raw bytes of a fixed width.

    ``V`` is NumPy's void kind, which covers plain raw bytes, structured
    datatypes and fixed-shape sub-arrays alike. h5py reads three different HDF5
    datatypes through it — ``H5T_OPAQUE``, ``H5T_COMPOUND`` and ``H5T_ARRAY`` —
    so a column's kind alone does not say which arrived. Only the last two
    carry a structure, and those two structural fields are what separates them.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    dtype:
        Anything :func:`numpy.dtype` accepts.
    """
    resolved = np.dtype(dtype)
    return (
        resolved.kind == "V" and resolved.fields is None and resolved.subdtype is None
    )


def opaque_fill_bytes(nbytes: int) -> bytes:
    """H5Col's recommended fill value for an opaque column *nbytes* wide.

    Since no byte string can be reserved by being out of range, the next best
    thing is one that is vanishingly unlikely to occur: the ASCII ``FILL``
    marker, followed by rising byte values that wrap through zero.

    The rising tail is the part that earns its keep. A fill of all zeros or all
    ``0xFF`` would collide constantly — that is what zero padding, erased flash
    and uninitialized memory leave behind — while a counting sequence is
    something real data almost never is. An eight-byte column gets
    ``46 49 4c 4c 01 02 03 04``, which reads as ``FILL....`` in a hex dump.

    A column narrower than the marker gets as much of it as fits. Note that a
    one-byte opaque column has 256 possible values and this claims one of them,
    which is a real risk rather than a negligible one; such a column is better
    given a fill value chosen against its own data.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    nbytes:
        The column's fixed width. The result is exactly this long.
    """
    head = OPAQUE_FILL_MAGIC[:nbytes]
    return head + bytes((i + 1) % 256 for i in range(nbytes - len(head)))
