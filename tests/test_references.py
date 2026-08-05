"""Tests for the object-reference backend (h5col.references)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import references
from h5col.exceptions import ObjectReferenceError


def test_ref_dtype_and_detection() -> None:
    assert references.is_reference_dtype(references.ref_dtype())
    assert not references.is_reference_dtype(np.dtype("i4"))


def test_scalar_ref_roundtrip(h5file: h5py.File) -> None:
    a = h5file.create_dataset("a", data=np.arange(3))
    references.write_ref_attr(h5file, "r", a)
    resolved = references.resolve(h5file, h5file.attrs["r"])
    assert resolved.name == a.name


def test_array_ref_roundtrip(h5file: h5py.File) -> None:
    a = h5file.create_dataset("a", data=np.arange(3))
    b = h5file.create_dataset("b", data=np.arange(3))
    references.write_ref_array_attr(h5file, "refs", [a, b])
    stored = h5file.attrs["refs"]
    assert references.is_reference_dtype(stored.dtype)
    names = [references.resolve(h5file, r).name for r in stored]
    assert names == [a.name, b.name]


def test_make_ref_on_non_object_raises() -> None:
    with pytest.raises(ObjectReferenceError):
        references.make_ref(object())


def test_null_reference_detected_and_unresolvable(h5file: h5py.File) -> None:
    # An unwritten reference dataset yields null references.
    d = h5file.create_dataset("refs", shape=(2,), dtype=references.ref_dtype())
    null = d[0]
    assert references.is_null_ref(null)
    with pytest.raises(ObjectReferenceError):
        references.resolve(h5file, null)


def test_valid_ref_is_not_null(h5file: h5py.File) -> None:
    a = h5file.create_dataset("a", data=np.arange(3))
    assert not references.is_null_ref(references.make_ref(a))


def test_ondisk_type_tracks_deviation_d1(h5file: h5py.File) -> None:
    # Deviation D1 (docs/DEVIATIONS.md): h5py writes the deprecated
    # H5T_STD_REF_OBJ rather than the H5Col-mandated H5T_STD_REF. When h5py
    # gains H5T_STD_REF support, this assertion will fail and should be revisited.
    a = h5file.create_dataset("a", data=np.arange(3))
    references.write_ref_attr(h5file, "r", a)
    tid = h5file.attrs.get_id("r").get_type()
    assert tid.equal(h5py.h5t.STD_REF_OBJ)
