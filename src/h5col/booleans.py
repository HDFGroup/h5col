"""The H5Col boolean datatype.

HDF5 has no native boolean type, so H5Col fixes one: an HDF5 enumeration with
base ``H5T_STD_I8LE`` and exactly two members, ``FALSE`` = 0 and ``TRUE`` = 1.
Producers must write exactly that datatype; consumers must additionally accept
any one-byte-integer enumeration (either signedness) whose members are exactly
``FALSE`` = 0 and ``TRUE`` = 1.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from .exceptions import ConformanceError, SchemaError

#: The two members of the H5Col boolean enumeration.
BOOL_MEMBERS: dict[str, int] = {"FALSE": 0, "TRUE": 1}


def bool_dtype() -> np.dtype:
    """Return the H5Col boolean dtype (enum over little-endian signed int8).

    On little-endian platforms the ``<i1`` base is byte-identical to
    ``H5T_STD_I8LE``, which is what H5Col mandates.
    """
    return h5py.enum_dtype(dict(BOOL_MEMBERS), basetype=np.dtype("<i1"))


def is_bool_dtype(dtype: Any) -> bool:
    """Return True if *dtype* is an acceptable H5Col boolean datatype.

    Applies the consumer-lenient rule: any one-byte integer enumeration, of
    either signedness, whose members are exactly ``FALSE`` = 0 and ``TRUE`` = 1.
    Also accepts NumPy ``bool``: h5py normalizes its FALSE/TRUE-over-int8 enum
    (which *is* the H5Col boolean datatype on disk) back to ``bool`` on read.

    Parameters
    ----------
    dtype:
        Anything :func:`numpy.dtype` accepts, including a dtype carrying h5py
        enumeration metadata. A dtype that is not an enumeration at all is not
        an error; it simply answers False.
    """
    d = np.dtype(dtype)
    if d.kind == "b":
        return True
    members = h5py.check_enum_dtype(dtype)
    if members is None:
        return False
    if d.itemsize != 1 or d.kind not in ("i", "u"):
        return False
    return members == BOOL_MEMBERS


def encode_bool(values: Any) -> npt.NDArray[np.int8]:
    """Encode a boolean/0-1 array-like to an ``int8`` array of codes.

    Parameters
    ----------
    values:
        A sequence or array of Python bools, NumPy bools, or integers that are
        all 0 or 1. Any other integer is rejected rather than coerced, and
        floats such as ``1.5`` are rejected rather than truncated.

    Raises
    ------
    SchemaError
        If an input value is neither 0 nor 1 (an H5Col boolean column may hold
        only those two codes).
    """
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return arr.astype(np.int8)
    # Validate the *input* values, not the int8 cast: casting wraps (256 -> 0)
    # and truncates floats (1.5 -> 1), which would silently coerce non-conformant
    # input into a valid code.
    if not np.all((arr == 0) | (arr == 1)):
        raise SchemaError("boolean values must be 0/1 (or Python bools)")
    return arr.astype(np.int8)


def decode_bool(values: Any) -> npt.NDArray[np.bool_]:
    """Decode stored integer codes to a NumPy boolean array.

    Every code must be 0 (FALSE) or 1 (TRUE). A code outside that domain is a
    non-conformant boolean value; H5Col forbids interpreting it as either
    FALSE or TRUE, so this raises instead of silently coercing (NumPy maps
    every nonzero code to ``True``).

    Parameters
    ----------
    values:
        The stored codes, as read from a boolean column's dataset.

    Raises
    ------
    ConformanceError
        If any code is neither 0 nor 1.
    """
    arr = np.asarray(values)
    if not np.all((arr == 0) | (arr == 1)):
        raise ConformanceError("boolean column value must be 0 or 1")
    return arr.astype(np.bool_)
