"""Tests for the fixed-length string handler (h5col.strings)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col.exceptions import OversizedStringError, SchemaError
from h5col.strings import FixedString, ascii_token_dtype


def test_construction_validation() -> None:
    with pytest.raises(SchemaError):
        FixedString(0)
    with pytest.raises(SchemaError):
        FixedString(-1)
    with pytest.raises(SchemaError):
        FixedString(4, encoding="latin-1")


def test_dtype_is_fixed_length_string() -> None:
    fs = FixedString(4, encoding="utf-8")
    info = h5py.check_string_dtype(fs.dtype)
    assert info is not None
    assert info.length == 4
    assert info.encoding == "utf-8"


def test_encode_decode_roundtrip() -> None:
    fs = FixedString(4)
    encoded = fs.encode(["abcd", "ab"])
    assert encoded.dtype.kind == "S"
    assert encoded.dtype.itemsize == 4
    decoded = fs.decode(encoded)
    assert list(decoded) == ["abcd", "ab"]


def test_oversize_ascii_raises_not_truncates() -> None:
    fs = FixedString(4)
    with pytest.raises(OversizedStringError) as exc:
        fs.encode(["abcde"])
    assert exc.value.actual_bytes == 5
    assert exc.value.max_bytes == 4
    assert exc.value.index == 0


def test_limit_is_bytes_not_characters() -> None:
    fs = FixedString(4)
    # 'é' is two UTF-8 bytes: two fit, three do not.
    assert fs.encode(["éé"])[0] == "éé".encode()
    with pytest.raises(OversizedStringError) as exc:
        fs.encode(["ééé"])  # 6 bytes
    assert exc.value.actual_bytes == 6


def test_multibyte_is_never_split() -> None:
    # h5py would silently truncate "ééé" to 4 bytes (splitting a code point);
    # the handler must refuse instead.
    fs = FixedString(4)
    with pytest.raises(OversizedStringError):
        fs.encode(["ééé"])


def test_h5py_would_truncate_but_handler_prevents_it(h5file: h5py.File) -> None:
    fs = FixedString(4)
    # Writing pre-validated bytes stores them intact.
    d = h5file.create_dataset("s", shape=(1,), dtype=fs.dtype)
    d[...] = fs.encode(["abcd"])
    assert fs.decode(d[...])[0] == "abcd"

    # Direct str assignment of an oversize value is silently truncated by h5py —
    # exactly the behavior the handler exists to prevent.
    d[0] = "abcde"
    assert d[0] == b"abcd"


def test_decode_strips_trailing_nulls() -> None:
    fs = FixedString(4)
    raw = np.array([b"ab"], dtype="S4")
    assert fs.decode(raw)[0] == "ab"
    assert fs.decode_scalar(b"ab\x00\x00") == "ab"


def test_ascii_encoding_rejects_non_ascii() -> None:
    fs = FixedString(8, encoding="ascii")
    with pytest.raises(UnicodeEncodeError):
        fs.encode(["é"])


def test_from_dtype_roundtrip() -> None:
    fs = FixedString(11, encoding="ascii")
    back = FixedString.from_dtype(fs.dtype)
    assert back.nbytes == 11
    assert back.encoding == "ascii"


def test_from_dtype_rejects_vlen_and_nonstring() -> None:
    with pytest.raises(SchemaError):
        FixedString.from_dtype(h5py.string_dtype())  # variable-length
    with pytest.raises(SchemaError):
        FixedString.from_dtype(np.dtype("i4"))


def test_is_fixed_string() -> None:
    assert FixedString.is_fixed_string(FixedString(4).dtype)
    assert not FixedString.is_fixed_string(h5py.string_dtype())  # vlen
    assert not FixedString.is_fixed_string(np.dtype("i4"))


def test_ascii_token_dtype_sized_for_value_plus_nul() -> None:
    dt = ascii_token_dtype("COLUMN_TABLE")  # 12 chars
    info = h5py.check_string_dtype(dt)
    assert info is not None
    assert info.length == 13  # value + NUL
    assert info.encoding == "ascii"
