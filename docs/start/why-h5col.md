# Why H5Col?

If you work with columnar data, you very likely already use Apache Parquet or
Apache Arrow, and they are excellent at what they do. This page does not argue
otherwise. It answers a narrower question: why would a columnar table belong
inside an HDF5 file, and what do you actually gain when it does?

## A convention, not another file format

H5Col is not a competing container. It is a set of layout rules — an HDF5
Enhancement Proposal,
[HEP001](https://hdfalliance.github.io/heps/hep001/) — for arranging a table
inside an ordinary HDF5 file. The proposal describes itself as a
column-oriented storage layout in the spirit of Parquet, Arrow, and Feather
that lives natively as an HDF5 group. Each scalar column becomes its own
rank-1 dataset with its own datatype, chunking, compression, and fill value
(variable-length list columns become small groups of such datasets); a handful
of attributes tie the columns together into a table with a committed row
count.

That framing matters for adoption cost. An H5Col file is an HDF5 file. It opens
in h5py, `h5dump`, HDFView, MATLAB, and every C, Fortran, Java, Julia, or Rust
stack that already links against HDF5. Nothing about your tooling, your I/O
drivers, or your archival practices has to change.

## Tables rarely travel alone

In scientific and engineering work, a table is usually a companion to
something larger: an event catalog next to the images it was extracted from, a
calibration table next to the sensor arrays it corrects, quality flags next to
the model fields they annotate. Formats like Parquet handle the table and
leave everything else to sidecar files, which means the dataset you actually
care about becomes a directory of loosely coupled artifacts held together by a
naming scheme.

With H5Col, the table lives in the same file as the arrays, under the same
group tree, sharing one metadata discipline. There is one artifact to move,
checksum, and archive. And because HDF5 has first-class object references, the
table can point at other objects in the file — H5Col itself uses references to
link columns to their category labels and search indexes.

## Readable by tools that have never heard of it

This is the property that most distinguishes H5Col from a purpose-built format.
A Parquet file is opaque to anything that does not implement Parquet. An H5Col
table degrades gracefully: a reader that knows the convention gets the full
semantics (missing values, categorical labels, index acceleration), while a
reader that does not still sees perfectly usable datasets with descriptive
attributes. An analyst with a twenty-year-old Fortran code can read the columns
written yesterday.

## Per-column storage control, on an open filter ecosystem

Parquet offers a curated menu of encodings and compression codecs. HDF5's
filter pipeline is an open plugin architecture, and H5Col exposes it per
column: each column declares its own ordered pipeline (for example shuffle
followed by DEFLATE, or a single Zstandard stage via
[hdf5plugin](https://pypi.org/project/hdf5plugin/)), its own chunk shape, and
optionally a Fletcher-32 checksum. When you know your data — and in
instrument and simulation work you usually do — this control is worth real
storage and throughput. Domain-specific and lossy compressors that are
unlikely ever to appear in the Parquet specification are a filter plugin
away.

## Semantics the application would otherwise have to invent

Column metadata in most formats is a key–value bag whose meaning each tool
defines for itself. The H5Col convention assigns defined meanings: physical
units and the vocabulary they come from, valid ranges, per-column
descriptions, and above all a precise missing-value model. A missing row
stores the column's fill value; the convention recommends per-datatype
sentinels, permits NaN where that is the natural choice, and defines a single
canonical test that both cases reduce to. This implementation also enforces a
rule that anyone who has lost data to a database will appreciate: writing an
oversized value into a fixed-length string column raises an error. Nothing is
ever silently truncated.

## Row selection without a query engine

H5Col defines optional, persistent search indexes — bitmap indexes for
categorical and boolean columns, sorted-row permutations and per-chunk min/max
summaries for orderable ones — stored beside the data in the same file, with a
validity protocol so a stale index can never produce a wrong answer. On top of
them this package provides a small predicate API deliberately parallel to
pyarrow's:

```python
from h5col import field

hot = table.select((field("kind") == "automatic") & (field("t_air") > 30.0))
```

Selections follow the same three-valued logic as SQL and pyarrow, so
predicates over missing values behave the way an analyst expects, and
`explain()` shows exactly which index answered each part of the query. There
is no server, no sidecar index file, and no query engine to deploy.

## What Parquet and Arrow still do better

An honest comparison mandates this as well.

- Ecosystem breadth. DuckDB, Spark, Polars, pandas, and a long list of cloud
  services read Parquet natively. Nothing comparable exists for H5Col today,
  and this package is its first implementation.
- In-memory interchange. Arrow's zero-copy columnar memory model and its
  compute kernels are unmatched. H5Col is a storage convention, not a memory
  format.
- Query maturity. Parquet's row-group statistics, dictionary and bloom-filter
  pushdown are implemented in many mature engines. H5Col's indexes cover the
  common cases but are young by comparison.
- Nested data. H5Col supports variable-length list columns, including nested
  lists, but not struct or map types.

If your data is purely tabular and lives in a data-lake toolchain, Parquet is
the right default, and H5Col does not ask you to migrate.

## Where H5Col is the better fit

The picture inverts when HDF5 is already your world: missions and instruments
whose products are standardized on HDF5, HPC codes that write it natively,
archives with HDF5 format guarantees, and any dataset where tables and arrays
belong together in one self-describing file. In those settings H5Col adds the
columnar model (independent per-column storage, dictionary-encoded
categories, predicate-driven selection) without adding a second format to
your stack. The workflow it serves best is the one scientific data usually
follows: write once, read often.

The [quickstart](quickstart.md) builds a first table in a few minutes. The
[conformance page](../about/conformance.md) records exactly how this
implementation tracks the convention.
