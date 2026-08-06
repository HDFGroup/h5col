:::{container} h5col-hero
```{image} _static/logo/h5col-logo.png
:alt: h5col — big tables next to big arrays
:class: only-light
```
```{image} _static/logo/h5col-logo-dark.png
:alt: h5col — big tables next to big arrays
:class: only-dark
```
:::

# h5col

`h5col` stores column-oriented tables natively in HDF5 files. A table is an
ordinary HDF5 group. Each scalar column in it is an ordinary rank-1 HDF5
dataset. A variable-length list column is a small group of such HDF5 datasets.
On top of that deliberately simple layout, the H5Col convention specifies the
things a real tabular workload needs: a committed row count, per-column chunking
and compression, fixed-length strings that never truncate silently, boolean and
categorical columns, precise missing-value semantics, variable-length list
columns, and persistent search indexes that accelerate row selection.

The convention is an HDF5 Enhancement Proposal: [HEP001 — H5Col: Column-Oriented
Tabular Data in HDF5](https://hdfalliance.github.io/heps/hep001/). This package,
`h5col`, built on [h5py](https://docs.h5py.org/en/stable/), is its first
experimental implementation. It supports writing and reading the convention's
tables, and adds a small pyarrow-style query API on top:

```python
import h5py
from h5col import ColumnSpec, FixedString, Table, field

with h5py.File("observations.h5", "w") as f:
    table = Table.create(
        f.create_group("obs"),
        [
            ColumnSpec(name="station", dtype=FixedString(nbytes=8)),
            ColumnSpec(name="t_air", dtype="float32", units="degC"),
        ],
        title="Surface air temperature",
    )
    table.append({"station": ["KBOS", "KJFK", "KLGA"], "t_air": [21.5, 24.0, 23.1]})
    print(table.select(field("t_air") > 22.0).read(["station"]))
```

A file written this way needs nothing beyond HDF5 to be read. Every column is a
plain dataset that h5py, `h5dump`, HDFView, or any other HDF5 tool can open,
whether or not this package is installed. The convention's extra meaning is
carried in attributes that convention-aware readers understand and other
readers simply ignore.

## Where to start

- If you already use Parquet or Arrow and are wondering what this adds, start
  with [Why H5Col?](start/why-h5col.md), then work through the
  [quickstart](start/quickstart.md).
- The [user guide](guide/index.md) explains one concept per chapter, from the
  on-disk data model to search indexes.
- [Queries](queries/index.md) documents row selection: the predicate API, the
  complete syntax, and how indexes accelerate evaluation.
- The [examples](examples.md) are rendered Jupyter notebooks, including a
  real-data walkthrough with New York City taxi trips.
- The [API reference](api/index.md) documents every public name in the
  package.
- Adherence of this package to the H5Col (HEP001) specification is documented in
  the [conformance page](about/conformance.md).
- The package's license is [here](about/license.md).

:::{container} h5col-funding
Funding

The software initial development was funded by the U.S. Department of Energy,
Office of Science, Office of Fusion Energy Sciences, under Award Number
DE-SC0024442.
:::

```{toctree}
:hidden:

start/index
guide/index
queries/index
examples
api/index
about/index
```
