"""Fill values and the canonical H5Col missing-value test.

A column marks a missing row with its HDF5 fill value. H5Col recommends a
per-datatype sentinel chosen to lie outside the column's logical value range, and
defines a single *canonical missing-value test* that both the fill-equality and
NaN cases reduce to.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from .exceptions import FillValueError

# H5Col "Recommended fill values" table, keyed by NumPy dtype name.
_RECOMMENDED: dict[str, Any] = {
    "int8": np.int8(-127),
    "int16": np.int16(-32767),
    "int32": np.int32(-2147483647),
    "int64": np.int64(-9223372036854775807),
    "uint8": np.uint8(255),
    "uint16": np.uint16(65535),
    "uint32": np.uint32(4294967295),
    "uint64": np.uint64(18446744073709551615),
    "float32": np.float32(9.9692099683868690e36),
    "float64": np.float64(9.9692099683868690e36),
}


def recommended_fill(dtype: Any) -> Any:
    """Return H5Col's recommended fill value for *dtype*.

    - Fixed- or variable-length string dtypes → ``b""``.
    - Enumerations with a ``MISSING`` member → the integer code of that member
      (the spec's enum fill convention).
    - Enumerations without a ``MISSING`` member, including the H5Col boolean
      datatype (which MUST NOT declare a fill value at all) → raises.
    - Integer and float families → the tabulated value for that width.
    - Anything else (e.g. ``float16``) → raises :class:`FillValueError`.

    Parameters
    ----------
    dtype:
        Anything :func:`numpy.dtype` accepts, including h5py string and
        enumeration dtypes, whose metadata decides which rule above applies.
    """
    if h5py.check_string_dtype(dtype) is not None:
        return b""
    members = h5py.check_enum_dtype(dtype)
    if members is not None:
        if "MISSING" in members:
            return np.dtype(dtype).type(members["MISSING"])
        raise FillValueError(
            f"enum dtype {np.dtype(dtype)!r} has no MISSING member; no "
            "recommended fill value (boolean columns declare none)"
        )
    name = np.dtype(dtype).name
    if name in _RECOMMENDED:
        return _RECOMMENDED[name]
    raise FillValueError(f"no recommended fill value for dtype {np.dtype(dtype)!r}")


def masked_to_none(values: Any) -> Any:
    """Rewrite a 1-D masked array to a list carrying ``None`` where it was masked.

    A masked element and a ``None`` say the same thing on append — this row is
    missing — so folding one into the other keeps a single missing-value path
    through every encoder. The payload under a mask is discarded deliberately:
    ``numpy.ma`` promises nothing about it, and in practice it holds whatever
    stale arithmetic left behind rather than the column's fill value.

    Anything that is not a 1-D masked array is returned unchanged, and a masked
    array with nothing masked yields its plain data, so the typed fast paths
    downstream are left undisturbed.

    Parameters
    ----------
    values:
        Usually a 1-D :class:`numpy.ma.MaskedArray`. Anything else, including a
        plain array or a list, is returned unchanged.
    """
    if not isinstance(values, np.ma.MaskedArray) or values.ndim != 1:
        return values
    # getmaskarray, not .mask: the latter is the scalar ``nomask`` when nothing
    # is masked, which cannot be zipped over.
    mask = np.ma.getmaskarray(values)
    if not mask.any():
        return values.data
    return [None if m else v for m, v in zip(mask, values.data, strict=True)]


def is_missing(values: Any, fill_value: Any) -> npt.NDArray[np.bool_]:
    """Apply the canonical missing-value test element-wise.

    ``missing(v, f) = isnan(f) ? isnan(v) : v == f`` — i.e. when the fill value is
    a NaN bit pattern the test is ``isnan(v)``; otherwise it is bit/value equality.

    Parameters
    ----------
    values:
        The stored values to test, as read from a column.
    fill_value:
        The column's declared fill value. A NaN may be given as a Python
        float, a NumPy scalar or a 0-d array; all three take the ``isnan``
        branch.
    """
    arr = np.asarray(values)
    fill = np.asarray(fill_value)
    # Normalize the fill first so a NaN delivered as a 0-d ndarray (not just a
    # Python/NumPy float scalar) still takes the isnan branch.
    fill_is_nan = (
        np.issubdtype(fill.dtype, np.floating)
        and fill.ndim == 0
        and bool(np.isnan(fill))
    )
    if fill_is_nan:
        return np.isnan(arr)
    return np.asarray(arr == fill_value, dtype=np.bool_)


def validate_fill_outside_range(
    fill: Any,
    valid_min: Any | None = None,
    valid_max: Any | None = None,
) -> None:
    """Check that *fill* lies strictly outside ``[valid_min, valid_max]``.

    Parameters
    ----------
    fill:
        The column's fill value.
    valid_min:
        Lower bound of the column's declared valid range, or None for
        unbounded below.
    valid_max:
        Upper bound, or None for unbounded above. With both bounds None there
        is nothing to check and the call succeeds.

    Raises
    ------
    FillValueError
        If *fill* falls inside the declared range, where a genuine value could
        collide with it.
    """
    if valid_min is None and valid_max is None:
        return
    ge_min = valid_min is None or fill >= valid_min
    le_max = valid_max is None or fill <= valid_max
    if ge_min and le_max:
        raise FillValueError(
            f"fill value {fill!r} must lie strictly outside "
            f"[{valid_min!r}, {valid_max!r}]"
        )
