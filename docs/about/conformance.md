# Conformance

`h5col` is the reference implementation of
[HEP001 — H5Col: Column-Oriented Tabular Data in HDF5](https://hdfalliance.github.io/heps/hep001/).
Being a reference implementation cuts two ways: the code aims to follow the
convention exactly, and where it cannot, it says so precisely. This page is
that record.

Tables are written with `VERSION = "1.0"`. On open, the major version is
checked and a table from a newer major raises
{class}`~h5col.VersionError`; {meth}`~h5col.Table.validate` checks the
convention's consistency rules on demand, structurally by default. With
`deep=True` it goes further, re-deriving every valid index of the
implemented families over the element datatypes its builders handle and
comparing contents; indexes of other kinds, or over datatypes the builders
cannot recompute, still receive the structural checks only.

## What is implemented

- Table groups with the full write protocol: creation from specs, appends
  with commit-last `NROWS` ordering, logical truncation, schema evolution
  via {meth}`~h5col.Table.add_column`, and validation.
- All scalar column families — numeric, fixed-length UTF-8 strings (with
  the no-silent-truncation guarantee), the boolean enumeration, and
  categorical columns with label datasets under `CATEGORIES`.
- Missing-value semantics: recommended per-datatype fills, the canonical
  missing-value test, and `valid_min`/`valid_max` enforcement against the
  fill.
- List columns in the offsets encoding, including string values
  (`STRING_VALUES`), nested lists, null-versus-empty distinction, and
  per-level masks.
- Per-column filter pipelines over the HDF5 filter ecosystem.
- Search indexes: `BITMAP`, `SORTED_ROWS`, and `CHUNK_MINMAX`, with the
  `GENERATION`-based validity protocol, index maintenance inside the write
  protocol (`maintain_indexes=True`), and refresh.
- The query layer with pyarrow-parity predicate syntax and three-valued
  missing-value semantics.

## Not implemented

The convention's fourth index family, `CHUNK_BLOOM` (per-chunk Bloom
filters for equality pruning on high-cardinality columns), is not built by
this package; requesting it raises {class}`~h5col.SchemaError`. Files
containing one are still safe to use — unknown index kinds are left
untouched and simply never consulted.

## Known deviation: object references

This is the one place the implementation knowingly departs from the convention.

HEP001 requires every reference attribute (`CATEGORIES`,
`SEARCH_INDEX_LIST`, `INDEX_COLUMNS`, and the bitmap `VALUES` link) to use
the unified `H5T_STD_REF` datatype introduced in HDF5 1.12, and forbids the
older `H5T_STD_REF_OBJ`. h5py — the foundation this package is built on —
cannot yet create `H5T_STD_REF` values, so `h5col` writes `H5T_STD_REF_OBJ`
instead.

The practical consequences are small, and in one respect the deviation is
the more portable choice: `H5T_STD_REF_OBJ` is readable by effectively every
HDF5 installation ever shipped. The deviation is also contained by design.
Every reference is created and resolved through one module,
`h5col.references`, whose read side accepts either datatype; when h5py
gains `H5T_STD_REF` support, swapping the write side requires no changes
anywhere else, and previously written files remain readable.

The engineering record of this decision, including the exact h5py versions
probed, is kept in the repository at
[`design/DEVIATIONS.md`](https://github.com/HDFGroup/h5col/blob/main/design/DEVIATIONS.md).

## Reading foreign files

Conformance checking is consumer-lenient where the convention says to be:
{meth}`Table.open <h5col.Table.open>` verifies the `CLASS` marker and the
version major, while {meth}`~h5col.Table.validate` performs the full rule
check when asked. Non-conformant structures fail with
{class}`~h5col.ConformanceError` and a message naming the violated rule
rather than with a generic h5py error.
