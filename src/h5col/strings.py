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

        Raises :class:`OversizedStringError` if the encoding exceeds ``nbytes``.
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

        No value is ever truncated: the first over-budget value raises
        :class:`OversizedStringError`.
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
        """Decode a single stored value to ``str`` (trailing NULs stripped)."""
        return bytes(raw).rstrip(b"\x00").decode(self.encoding)

    def decode(self, values: Any) -> npt.NDArray[np.object_]:
        """Decode an array-like of stored bytes to an object array of ``str``."""
        arr = np.asarray(values)
        flat = arr.ravel()
        decoded = [self.decode_scalar(v) for v in flat]
        return np.array(decoded, dtype=object).reshape(arr.shape)

    # -- introspection ------------------------------------------------------ #
    @classmethod
    def from_dtype(cls, dtype: Any) -> FixedString:
        """Build a :class:`FixedString` from an existing fixed-length string dtype.

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
        """Return True if *dtype* is a fixed-length HDF5 string dtype."""
        info = h5py.check_string_dtype(dtype)
        return info is not None and info.length is not None


def ascii_token_dtype(value: str) -> np.dtype:
    """Return a fixed-length ASCII string dtype sized to hold *value* plus a NUL.

    Used for H5Col reserved-token attributes (``CLASS``, ``VERSION``, ``KIND``),
    whose values are ASCII and are stored null-terminated/null-padded.
    """
    nbytes = len(value.encode("ascii")) + 1
    return FixedString(nbytes=nbytes, encoding="ascii").dtype
