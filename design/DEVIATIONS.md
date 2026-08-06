# Known deviations from HEP001

This reference implementation aims for full HEP001 conformance. Where a
deviation exists because of a limitation in the underlying tooling, it is
recorded here with its rationale and the conditions under which it can be
removed.

## D1 — Object references use `H5T_STD_REF_OBJ`, not `H5T_STD_REF`

**Spec requirement.** HEP001 §"Object references" states that every HEP001
reference attribute (`INDEX_COLUMNS`, `CATEGORIES`, `SEARCH_INDEX_LIST`, and the
bitmap `VALUES` link) **MUST** use the unified `H5T_STD_REF` datatype introduced
in HDF5 1.12, and that producers **MUST NOT** write the deprecated
`H5T_STD_REF_OBJ`.

**Deviation.** H5Col currently writes `H5T_STD_REF_OBJ`.

**Cause.** h5py (verified with h5py 3.16.0 against HDF5 2.1.0) does not support
the unified `H5T_STD_REF` datatype. Its low-level API exposes only
`h5py.h5t.STD_REF_OBJ` and `h5py.h5t.STD_REF_DSETREG`; there is no `STD_REF`
constant and no wrapper for the HDF5 1.12 `H5Rcreate_object` API. An empirical
check confirmed that a reference written through h5py has a committed datatype
that `.equal(STD_REF_OBJ)` returns true for. Because CLAUDE.md establishes h5py
as the foundation of this package, H5Col cannot emit `H5T_STD_REF` without
bypassing h5py via direct `libhdf5` calls.

**Containment.** All reference creation and resolution is isolated behind the
`h5col.references` module with a pluggable backend. The current backend writes
`H5T_STD_REF_OBJ`; a future `H5T_STD_REF` backend (e.g. via a ctypes shim or a
future h5py release) can be swapped in without changing any caller. The read
side is lenient and accepts either reference datatype.

**Feedback.** This gap is worth raising upstream:

- *HEP001:* consider whether a conformant reference implementation is currently
  achievable with the dominant Python HDF5 library, and whether the spec should
  note the h5py limitation or offer a transition allowance.
- *h5py:* the unified `H5T_STD_REF` reference type (HDF5 ≥ 1.12) is not yet
  supported; adding it would let Python producers write conformant HEP001
  references.

**Removal condition.** Replace the backend once h5py can create `H5T_STD_REF`
references natively, or once a vetted low-level shim is implemented.
