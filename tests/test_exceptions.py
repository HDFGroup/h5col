"""Tests for the H5Col exception hierarchy (h5col.exceptions)."""

from __future__ import annotations

from h5col.exceptions import (
    H5ColError,
    ObjectReferenceError,
    OversizedStringError,
    ReservedNameError,
    SchemaError,
)


def test_all_derive_from_base() -> None:
    for exc in (
        SchemaError,
        ReservedNameError,
        OversizedStringError,
        ObjectReferenceError,
    ):
        assert issubclass(exc, H5ColError)


def test_builtin_compatibility() -> None:
    assert issubclass(SchemaError, ValueError)
    assert issubclass(ReservedNameError, ValueError)
    assert issubclass(OversizedStringError, ValueError)


def test_object_reference_error_distinct_from_builtin() -> None:
    # Deliberately not the built-in ReferenceError.
    assert not issubclass(ObjectReferenceError, ReferenceError)


def test_oversized_string_error_carries_context() -> None:
    err = OversizedStringError("abcde", 4, 5, index=2)
    assert err.value == "abcde"
    assert err.max_bytes == 4
    assert err.actual_bytes == 5
    assert err.index == 2
    assert "index 2" in str(err)
