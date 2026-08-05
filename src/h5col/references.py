"""HDF5 object-reference backend for H5Col.

All object-reference creation and resolution in H5Col goes through this module,
so the on-disk reference representation can be changed in one place.

.. warning::

   H5Col mandates the unified ``H5T_STD_REF`` datatype (HDF5 1.12+) and forbids
   the deprecated ``H5T_STD_REF_OBJ``. h5py (as of 3.16) cannot create
   ``H5T_STD_REF``, so this backend currently writes ``H5T_STD_REF_OBJ``. This is
   a documented deviation (see ``docs/DEVIATIONS.md`` D1). The read side accepts
   either representation. A conformant backend can replace this module without
   changing any caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from .exceptions import ObjectReferenceError

# h5py object types that can be referenced.
_Referable = Any  # h5py.Group | h5py.Dataset | h5py.Datatype
_Location = Any  # h5py.File | h5py.Group


def ref_dtype() -> np.dtype:
    """Return the NumPy dtype used to store object references."""
    return h5py.ref_dtype


def is_reference_dtype(dtype: Any) -> bool:
    """Return True if *dtype* is an HDF5 object/region reference dtype."""
    return h5py.check_ref_dtype(dtype) is not None


def make_ref(obj: _Referable) -> h5py.Reference:
    """Create an object reference to an open HDF5 object.

    Raises
    ------
    ObjectReferenceError
        If *obj* exposes no ``ref`` (it is not a referable HDF5 object).
    """
    try:
        return obj.ref
    except AttributeError as exc:
        raise ObjectReferenceError(
            f"cannot create an object reference to {obj!r}"
        ) from exc


def is_null_ref(ref: Any) -> bool:
    """Return True if *ref* is a null object reference."""
    return not bool(ref)


def write_ref_attr(target: _Referable, name: str, obj: _Referable) -> None:
    """Write a scalar object-reference attribute onto *target*."""
    target.attrs.create(name, make_ref(obj), dtype=ref_dtype())


def write_ref_array_attr(
    target: _Referable, name: str, objs: Iterable[_Referable]
) -> None:
    """Write a 1-D object-reference array attribute onto *target*."""
    refs: npt.NDArray[Any] = np.array([make_ref(o) for o in objs], dtype=ref_dtype())
    target.attrs.create(name, refs)


def append_ref_to_array_attr(target: _Referable, name: str, obj: _Referable) -> None:
    """Append a reference to *obj* to a 1-D reference-array attribute on *target*.

    Creates the attribute when absent. HDF5 attributes cannot be resized in
    place, so an existing attribute is rewritten with the extended array. The
    new reference is created *before* the old attribute is touched, and the old
    array is restored if the rewrite fails, so a failed append cannot silently
    drop the existing references. (A hard crash between the delete and the
    create can still lose the attribute — HDF5 offers no atomic attribute
    rewrite.)
    """
    new_ref = make_ref(obj)
    old: list[Any] = []
    if name in target.attrs:
        old = list(target.attrs[name])
        del target.attrs[name]
    try:
        target.attrs.create(name, np.array([*old, new_ref], dtype=ref_dtype()))
    except BaseException:
        if old:
            try:
                target.attrs.create(name, np.array(old, dtype=ref_dtype()))
            except Exception:
                pass  # the original failure is the error that matters
        raise


def resolve(where: _Location, ref: Any) -> _Referable:
    """Dereference *ref* relative to file/group *where*.

    Raises :class:`ObjectReferenceError` for a null reference or a reference that
    does not resolve.
    """
    if is_null_ref(ref):
        raise ObjectReferenceError("cannot resolve a null object reference")
    try:
        return where[ref]
    except (KeyError, ValueError, TypeError) as exc:
        raise ObjectReferenceError(f"reference does not resolve: {ref!r}") from exc
