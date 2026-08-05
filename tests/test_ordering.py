"""Tests for the H5Col canonical ordering (orderability and min/max)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import FixedString, bool_dtype
from h5col.exceptions import SchemaError
from h5col.ordering import is_orderable, min_max, normalize_strings


# --------------------------------------------------------------------------- #
# Orderability predicate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("i1"),
        np.dtype("u8"),
        np.dtype("f4"),
        np.dtype("f8"),
        np.bool_,
        bool_dtype(),
        h5py.enum_dtype({"A": 0, "B": 1}, basetype="i2"),
        FixedString(8).dtype,
        FixedString(4, encoding="ascii").dtype,
        h5py.string_dtype(),  # variable-length strings order byte-wise too
    ],
)
def test_orderable(dtype: object) -> None:
    assert is_orderable(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        h5py.ref_dtype,  # object references
        h5py.regionref_dtype,  # region references
        np.dtype([("a", "i4"), ("b", "f8")]),  # compound
        np.dtype(("f8", (3,))),  # array datatype
        h5py.vlen_dtype(np.dtype("i4")),  # variable-length array
        np.dtype("O"),  # bare object dtype: no defined order
        np.dtype("M8[s]"),  # datetime: not an H5Col datatype
    ],
)
def test_not_orderable(dtype: object) -> None:
    assert not is_orderable(dtype)


# --------------------------------------------------------------------------- #
# min/max under the defined order
# --------------------------------------------------------------------------- #
def test_min_max_integers() -> None:
    assert min_max(np.array([3, -7, 12], dtype="i4")) == (-7, 12)


def test_min_max_floats_with_infinities() -> None:
    vmin, vmax = min_max(np.array([0.5, -np.inf, np.inf, 2.0]))
    assert vmin == -np.inf and vmax == np.inf


def test_min_max_boolean_false_before_true() -> None:
    vmin, vmax = min_max(np.array([True, False, True]))
    assert (vmin, vmax) == (False, True)


def test_min_max_strings_bytewise() -> None:
    # Byte-wise UTF-8: all uppercase ASCII < lowercase, multibyte é sorts last.
    arr = np.array([b"b", b"AB", "é".encode()], dtype="S4")
    vmin, vmax = min_max(arr)
    assert vmin == b"AB"
    assert vmax == "é".encode()


def test_min_max_strings_prefix_rule() -> None:
    # Trailing-NUL-stripped rule: "ab" < "abc" (prefix sorts first).
    arr = np.array([b"abc", b"ab"], dtype="S4")
    assert min_max(arr) == (b"ab", b"abc")


def test_min_max_empty_raises() -> None:
    with pytest.raises(SchemaError):
        min_max(np.array([], dtype="f8"))


def test_normalize_strings_spacepad() -> None:
    arr = np.array([b"ab  ", b"a   "], dtype="S4")
    out = normalize_strings(arr, spacepad=True)
    assert out.tolist() == [b"ab", b"a"]
    # NUL padding needs no normalization (memcmp order-equivalence).
    same = normalize_strings(arr, spacepad=False)
    assert same is arr
