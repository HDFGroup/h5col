"""User-friendly handling of fixed-length HDF5 string datatypes.

H5Col leans heavily on fixed-length HDF5 strings, which h5py exposes awkwardly
(as NumPy ``|S`` dtypes carrying opaque encoding metadata) and which silently
truncate oversized values on write. :class:`FixedString` makes them pleasant to
use from Python and NumPy, and — crucially — enforces the byte budget on encode,
raising :class:`~h5col.exceptions.OversizedStringError` instead of truncating.

The length of an HDF5 fixed-length string is a *byte* count, not a character
count. A single UTF-8 code point can occupy up to four bytes, so the enforced
limit is always measured on the encoded bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from .exceptions import OversizedStringError, SchemaError

_ENCODINGS = ("utf-8", "ascii")


def decoded_string_dtype(*, nullable: bool = False) -> np.dtype:
    """The NumPy dtype H5Col string values decode into.

    ``numpy.dtypes.StringDType`` (NumPy ≥ 2.0) holds real ``str`` values without
    the per-element Python object a ``dtype=object`` array needs, and still
    supports ``==``, ``sort``, ``unique`` and fancy indexing elementwise.

    Parameters
    ----------
    nullable:
        When True the dtype also accepts ``None``, for categorical columns whose
        missing rows have no label to decode to.
    """
    if nullable:
        return np.dtypes.StringDType(na_object=None)
    return np.dtypes.StringDType()


@dataclass(frozen=True)
class FixedString:
    """A fixed-length HDF5 string datatype of ``nbytes`` bytes.

    Parameters
    ----------
    nbytes:
        Storage width in bytes (``> 0``).
    encoding:
        ``"utf-8"`` (default) or ``"ascii"``.

    Raises
    ------
    SchemaError
        If *nbytes* is not a positive integer, or *encoding* is unsupported.
    """

    nbytes: int
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if not isinstance(self.nbytes, int) or self.nbytes <= 0:
            raise SchemaError(f"nbytes must be a positive integer, got {self.nbytes!r}")
        if self.encoding not in _ENCODINGS:
            raise SchemaError(
                f"encoding must be one of {_ENCODINGS}, got {self.encoding!r}"
            )

    @property
    def dtype(self) -> np.dtype:
        """The NumPy/h5py dtype for creating a dataset or attribute of this type."""
        return h5py.string_dtype(encoding=self.encoding, length=self.nbytes)

    # -- encoding (Python/NumPy -> stored bytes) ---------------------------- #
    def encode_scalar(self, value: object, *, index: int | None = None) -> bytes:
        """Encode a single value to bytes, enforcing the byte budget.

        Parameters
        ----------
        value:
            ``bytes`` is stored as given; anything else is passed through
            ``str()`` and encoded with this type's encoding.
        index:
            Position of the value in the array being written, carried into the
            error message so an oversized row can be located. None when the
            value did not come from an array.

        Raises
        ------
        OversizedStringError
            If the encoded form exceeds ``nbytes``. Nothing is truncated.
        """
        if isinstance(value, bytes):
            raw = value
        else:
            raw = str(value).encode(self.encoding)
        if len(raw) > self.nbytes:
            raise OversizedStringError(value, self.nbytes, len(raw), index=index)
        return raw

    def encode(self, values: Any) -> npt.NDArray[np.bytes_]:
        """Encode an array-like of strings to a ``|S{nbytes}`` array.

        Parameters
        ----------
        values:
            Any array-like of values :meth:`encode_scalar` accepts. The shape
            is preserved.

        Raises
        ------
        OversizedStringError
            On the first value whose encoding exceeds ``nbytes``, carrying its
            position. No value is ever truncated.
        """
        arr = np.asarray(values, dtype=object)
        flat = arr.ravel()
        encoded: list[bytes] = []
        for i, v in enumerate(flat):
            encoded.append(self.encode_scalar(v, index=i))
        # Build with the h5py string dtype (not a bare ``|S``) so the encoding
        # metadata rides along and h5py accepts the array on write.
        out = np.array(encoded, dtype=self.dtype)
        return out.reshape(arr.shape)

    # -- decoding (stored bytes -> Python str) ------------------------------ #
    def decode_scalar(self, raw: bytes | bytearray | np.bytes_) -> str:
        """Decode a single stored value to ``str`` (trailing NULs stripped).

        Parameters
        ----------
        raw:
            One stored value. HDF5 pads a fixed-length string to its full width
            with NUL bytes, which are stripped before decoding.
        """
        return bytes(raw).rstrip(b"\x00").decode(self.encoding)

    def decode(self, values: Any) -> npt.NDArray[Any]:
        """Decode an array-like of stored bytes to a NumPy string array.

        The result carries :func:`decoded_string_dtype`, which keeps the text in
        one compact arena instead of allocating a Python ``str`` object per
        element the way a ``dtype=object`` array does.

        NumPy's own ``S`` → ``StringDType`` cast replaces the per-element
        Python loop this used to run: it strips the trailing NULs HDF5 pads
        with and takes the bytes as UTF-8, which also covers ``ascii`` columns
        since ASCII is a subset.

        The cast copies eagerly but validates lazily. Bytes that are not valid
        UTF-8 — only reachable from a non-conformant producer, since
        :meth:`encode` validates on write — therefore raise
        ``UnicodeDecodeError`` when the offending value is read out of the
        array, not when the column is read.

        Parameters
        ----------
        values:
            An array-like of stored bytes, as read from a fixed-length string
            dataset. The shape is preserved.
        """
        return np.asarray(values).astype(decoded_string_dtype())

    # -- introspection ------------------------------------------------------ #
    @classmethod
    def from_dtype(cls, dtype: Any) -> FixedString:
        """Build a :class:`FixedString` from an existing fixed-length string dtype.

        Parameters
        ----------
        dtype:
            An h5py string dtype, normally taken from an existing dataset. Its
            length and encoding become the new instance's ``nbytes`` and
            ``encoding``; an absent encoding defaults to UTF-8.

        Raises
        ------
        SchemaError
            If *dtype* is not an HDF5 string dtype, or is a variable-length
            (not fixed-length) string dtype.
        """
        info = h5py.check_string_dtype(dtype)
        if info is None:
            raise SchemaError(f"{dtype!r} is not an HDF5 string dtype")
        if info.length is None:
            raise SchemaError(
                f"{dtype!r} is a variable-length string, not a fixed-length string"
            )
        return cls(nbytes=int(info.length), encoding=info.encoding or "utf-8")

    @staticmethod
    def is_fixed_string(dtype: Any) -> bool:
        """Return True if *dtype* is a fixed-length HDF5 string dtype.

        Parameters
        ----------
        dtype:
            Any dtype. One that is not a string dtype at all, and a
            variable-length string dtype, both answer False.
        """
        info = h5py.check_string_dtype(dtype)
        return info is not None and info.length is not None


def ascii_token_dtype(value: str) -> np.dtype:
    """Return a fixed-length ASCII string dtype sized to hold *value* plus a NUL.

    Used for H5Col reserved-token attributes (``CLASS``, ``VERSION``, ``KIND``),
    whose values are ASCII and are stored null-terminated/null-padded.

    Parameters
    ----------
    value:
        The token the dtype has to hold. Only its encoded length is used, so
        any token of the same length gives the same dtype.
    """
    nbytes = len(value.encode("ascii")) + 1
    return FixedString(nbytes=nbytes, encoding="ascii").dtype
