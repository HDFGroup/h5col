"""Tests for the H5Col boolean datatype (h5col.booleans)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col.booleans import (
    BOOL_MEMBERS,
    bool_dtype,
    decode_bool,
    encode_bool,
    is_bool_dtype,
)
from h5col.exceptions import ConformanceError, SchemaError


def test_bool_dtype_members_and_width() -> None:
    dt = bool_dtype()
    assert h5py.check_enum_dtype(dt) == {"FALSE": 0, "TRUE": 1}
    assert np.dtype(dt).itemsize == 1


def test_bool_dtype_base_is_std_i8le() -> None:
    tid = h5py.h5t.py_create(bool_dtype(), logical=True)
    base = tid.get_super()
    assert base.equal(h5py.h5t.STD_I8LE)


def test_is_bool_dtype_accepts_canonical() -> None:
    assert is_bool_dtype(bool_dtype())


def test_is_bool_dtype_lenient_accepts_uint8_base() -> None:
    # Consumer-lenient rule: unsigned one-byte base is also a boolean.
    dt = h5py.enum_dtype({"FALSE": 0, "TRUE": 1}, basetype=np.dtype("u1"))
    assert is_bool_dtype(dt)


def test_is_bool_dtype_rejects_other_member_sets() -> None:
    three = h5py.enum_dtype({"FALSE": 0, "TRUE": 1, "MISSING": 2}, basetype="i1")
    assert not is_bool_dtype(three)
    wrong = h5py.enum_dtype({"NO": 0, "YES": 1}, basetype="i1")
    assert not is_bool_dtype(wrong)


def test_is_bool_dtype_rejects_wrong_width_and_nonenum() -> None:
    wide = h5py.enum_dtype({"FALSE": 0, "TRUE": 1}, basetype="i2")
    assert not is_bool_dtype(wide)
    assert not is_bool_dtype(np.dtype("i1"))


def test_encode_from_python_bools() -> None:
    out = encode_bool([True, False, True])
    assert out.dtype == np.int8
    assert list(out) == [1, 0, 1]


def test_encode_from_ints_validates_domain() -> None:
    assert list(encode_bool([0, 1, 0])) == [0, 1, 0]
    with pytest.raises(SchemaError):
        encode_bool([0, 2])
    # Values that wrap into {0, 1} under int8 casting must still be rejected.
    for bad in ([256], [257], [-256], [-255]):
        with pytest.raises(SchemaError):
            encode_bool(bad)


def test_encode_rejects_nonintegral_floats() -> None:
    # Floats that truncate into {0, 1} under int8 casting must be rejected.
    for bad in ([1.5], [0.5], [np.nan]):
        with pytest.raises(SchemaError):
            encode_bool(bad)
    # Exact-valued floats remain acceptable.
    assert list(encode_bool([1.0, 0.0])) == [1, 0]


def test_decode_bool() -> None:
    assert list(decode_bool([0, 1, 0])) == [False, True, False]


def test_decode_bool_rejects_out_of_domain() -> None:
    # A non-conformant code must not be silently coerced to True.
    with pytest.raises(ConformanceError):
        decode_bool([2])
    with pytest.raises(ConformanceError):
        decode_bool([5, -1])


def test_roundtrip_through_hdf5(h5file: h5py.File) -> None:
    data = encode_bool([True, False, True, True])
    d = h5file.create_dataset("b", data=data, dtype=bool_dtype())
    # On disk it is the H5Col boolean datatype: enum over STD_I8LE.
    t = d.id.get_type()
    assert t.get_class() == 8  # H5T_ENUM
    assert t.get_super().equal(h5py.h5t.STD_I8LE)
    # h5py hands the column back as NumPy bool.
    assert is_bool_dtype(d.dtype)
    assert list(decode_bool(d[...])) == [True, False, True, True]


def test_members_constant() -> None:
    assert BOOL_MEMBERS == {"FALSE": 0, "TRUE": 1}
