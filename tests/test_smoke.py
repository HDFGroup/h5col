"""Smoke tests validating the package and its runtime environment.

These also serve as executable preconditions for the H5Col features built in
later phases (e.g. HDF5 >= 1.12 is required for the ``H5T_STD_REF`` object
references the convention mandates).
"""

import h5py

import h5col


def test_h5col_imports() -> None:
    assert h5col.__version__


def test_hdf5_supports_standard_references() -> None:
    # H5T_STD_REF was introduced in HDF5 1.12.
    assert h5py.version.hdf5_version_tuple >= (1, 12, 0), (
        f"HDF5 {h5py.version.hdf5_version} is too old for H5T_STD_REF"
    )
