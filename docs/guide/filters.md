# Filters and storage

Column-oriented layouts and compression are natural partners. A column is a
long run of same-typed, often similar values, which is exactly what
compressors like. In H5Col each column dataset carries its own HDF5 filter
pipeline and its own chunk shape, so every column can be stored the way its
data deserves.

## Filter pipelines

A {class}`~h5col.FilterPipeline` is an ordered list of filters, mirroring
HDF5's own chunk filter pipeline: on write, each chunk passes through the
filters in order; on read, HDF5 reverses them transparently. Three filters
every HDF5 installation understands are built in:

```python
from h5col import ColumnSpec, Deflate, FilterPipeline, Fletcher32, Shuffle

ColumnSpec(
    name="fare_amount",
    dtype="float64",
    filters=FilterPipeline([Shuffle(), Deflate(5)]),
)
```

{func}`~h5col.Shuffle` reorders bytes so same-significance bytes group together
which typically improves compression ratio of numeric data.
{func}`~h5col.Deflate` is zlib/gzip compression at the given level.
{func}`~h5col.Fletcher32` appends a checksum to each chunk, so bit rot is
detected at read time. If using it, place it in the pipeline alongside the
others (recommendation is to be the last).

The pattern `Shuffle()` then `Deflate(...)` — or shuffle then any
general-purpose compressor — is the sensible default for numeric columns.

One nuance about ordering: the declared order (and any per-filter `optional`
flag) is honored exactly on numeric and boolean columns, which are created
through HDF5's low-level property-list path. Fixed-length string columns go
through h5py's high-level API instead, which normalizes the pipeline to
shuffle → compressor → checksum, supports one compressor stage, and ignores
per-filter optional flags. For the usual shuffle-then-compress pipelines the
two paths agree.

## The wider filter ecosystem

HDF5 filters are registered plugins, and the
[hdf5plugin](https://pypi.org/project/hdf5plugin/) package (a regular
dependency of `h5col`) provides the widely used modern ones. Its filter
objects drop straight into a pipeline:

```python
import hdf5plugin

ColumnSpec(
    name="total_amount",
    dtype="float64",
    filters=FilterPipeline([hdf5plugin.Zstd(clevel=5)]),
)
```

({func}`~h5col.from_hdf5plugin` is the explicit adapter behind that coercion.)
Beyond `hdf5plugin`'s catalog, any registered HDF5 filter, such as
domain-specific or lossy compressors, can be named directly by its plugin id
with {class}`~h5col.Filter`:

```python
Filter(plugin_id=32015, cd_values=(5,), name="zstd")
```

The built-in shuffle, DEFLATE, and Fletcher-32 are part of every HDF5 build, but
chunks written with a plugin filter need that plugin present at read time too.
Within Python this is handled by importing `h5col` which imports `hdf5plugin`.
Readers outside Python need the corresponding [HDF5 plugin
binaries](https://github.com/HDFGroup/hdf5_plugins). Choose plugin filters with
your consumers in mind.

## Chunking

Chunks are the unit of I/O, of filtering, and of
[chunk-level index pruning](indexes.md). Every column accepts a `chunks=`
row count in its spec.

When `chunks` is omitted, this implementation picks a chunk size aimed at
large tables: a few mebibytes per chunk (2–8 MiB, scaled to the dataset
chunk cache), so scans stream efficiently and the chunk B-tree stays small.
Two situations call for an explicit value:

- Small tables. A small dimension table with an auto-sized, million-element
  chunk allocates the whole chunk on disk regardless — a 265-row lookup
  table can cost megabytes. Set `chunks` near the expected row count; in the
  {doc}`taxi example <../notebooks/06_nyc_taxi>`, doing so on the zones
  table shrank the file from about 5.6 MB to 1.4 MB by releasing the
  over-allocated chunk space.
- Selective queries over big columns. A `CHUNK_MINMAX` index prunes whole
  chunks, so its resolution is the chunk size; the taxi trips table chunks
  its columns at 4,096 rows partly to make that pruning effective.

## Seeing what storage decisions cost

Storage questions deserve measurements, and h5py exposes them directly.
Logical versus stored bytes per column:

```python
for name in table.column_names:
    ds = table[name].dataset
    logical = ds.size * ds.dtype.itemsize
    stored = ds.id.get_storage_size()
    print(f"{name:24s} {logical / max(stored, 1):5.1f}x  {stored:>10,d} B stored")
```

On the 25,000-row taxi sample, shuffle-plus-DEFLATE pipelines compress the
trips table about 4.9× overall. Numbers like that are dataset-dependent —
measure your own.
