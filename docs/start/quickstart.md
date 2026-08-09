# Quickstart

This page builds a small table end to end: define columns, append rows, read
them back, select rows with a predicate, add a search index, and reopen the
file later. It assumes `h5col` is [installed](installation.md) and takes about
ten minutes.

One design decision is worth knowing up front: `h5col` never opens files. You
open an HDF5 file with h5py with whatever driver, page-buffering, or
cloud-optimized settings you need, and hand any {class}`h5py.Group` to `h5col`.
The package defines what happens inside that group, and h5py keeps defining
everything outside it.

## Create a table

A table is created from column specifications. Each
{class}`~h5col.ColumnSpec` names a column and describes its storage: the
datatype, an optional fill value that marks missing rows, optional valid
bounds, units, and a description.

```python
import h5py
import numpy as np

from h5col import ColumnSpec, FixedString, Table, field

columns = [
    ColumnSpec(
        name="station",
        dtype=FixedString(nbytes=8),
        description="Reporting station identifier",
    ),
    ColumnSpec(name="kind", categories=["manned", "automatic"]),
    ColumnSpec(name="t_air", dtype="float64", units="degC", fill_value=np.nan),
    ColumnSpec(name="samples", dtype="int32", fill_value=-1, valid_min=0),
]

f = h5py.File("quickstart.h5", "w")
table = Table.create(f.create_group("obs"), columns, title="Surface observations")
table.nrows
```

```text
0
```

Each spec shows a different capability. `station` is a fixed-length string
column with an 8-byte budget; writing a longer value will raise
{class}`~h5col.OversizedStringError` rather than truncate. `kind` is a
categorical column: the labels are declared once, and the column stores small
integer codes. `t_air` uses NaN as its fill value, so missing temperatures
behave the way NumPy and pandas users expect. `samples` marks missing rows
with the sentinel `-1` instead, and declares `valid_min=0` so the sentinel is
provably outside the valid range.

## Append rows

Rows are appended column-wise, as a mapping from column name to a sequence of
values. Every provided column must have the same length. Categorical columns
take labels, not codes. In any column, `None` marks a missing row — it is
stored as that column's fill value — and you may equally write the fill value
yourself, as the NaN in `t_air`'s last entry below does.

```python
table.append(
    {
        "station": ["KBOS", "KJFK", "KLGA", "KDCA"],
        "kind": ["manned", "automatic", "automatic", None],
        "t_air": [21.5, 24.0, 23.1, np.nan],
        "samples": [12, 60, 58, 60],
    }
)
table.nrows
```

```text
4
```

The append follows the convention's write protocol: column data is written and
flushed first, and the committed row count (the table's `NROWS` attribute) is
updated last, so a reader never sees a row count that points at unwritten
data. An unknown label in a categorical column, an oversized string, or
unequal column lengths all raise before anything is committed.

## Read columns back

Reading decodes each column to friendly values: fixed-length strings come back
as Python strings, categorical columns as their labels, booleans as NumPy
booleans.

```python
table.read(["station", "kind"])
```

```text
{'station': array(['KBOS', 'KJFK', 'KLGA', 'KDCA'], dtype=StringDType()),
 'kind': array(['manned', 'automatic', 'automatic', None],
               dtype=StringDType(na_object=None))}
```

Individual columns are reached by name. A categorical column also exposes its
raw integer codes and its label array:

```python
table["kind"].codes
```

```text
array([ 0,  1,  1, -1], dtype=int8)
```

```python
table["t_air"].is_missing()
```

```text
array([False, False, False,  True])
```

The last row's temperature is missing (it stored the NaN fill), and the same
row's `kind` was appended as `None`, so it reads back as `None`.

## Select rows

{func}`~h5col.field` builds predicates that combine with `&`, `|`, and `~`.
Because Python's bitwise operators bind more tightly than comparisons, always
parenthesize each comparison:

```python
sel = table.select((field("kind") == "automatic") & (field("t_air") > 22.0))
sel.count
```

```text
2
```

```python
sel.read(["station", "t_air"])
```

```text
{'station': array(['KJFK', 'KLGA'], dtype=StringDType()),
 't_air': array([24. , 23.1])}
```

A {class}`~h5col.Selection` is lazy — it evaluates once, on first use, and
`read()` materializes only the requested columns for the matching rows.

Missing values follow three-valued logic, exactly as in SQL and pyarrow: a
comparison against a missing value is unknown, and only rows where the whole
predicate is true are selected. Station KDCA, whose temperature is missing, is
not matched by `field("t_air") > 22.0` — and it would not be matched by the
negation either. To ask for missing rows explicitly:

```python
table.count(field("t_air").is_null())
```

```text
1
```

If you prefer pyarrow's tuple form, the same selection can be written as
`table.select([("kind", "==", "automatic"), ("t_air", ">", 22.0)])`.

## Accelerate with a search index

Queries work with no preparation — every predicate can be answered by scanning
the column. A search index makes selective queries cheaper, and it is stored
inside the file, next to the data:

```python
table.build_index("t_air", "SORTED_ROWS")
print(table.select(field("t_air") > 22.0).explain())
```

```text
QueryPlan: 2 / 4 rows matched
   AND-term 0:
      t_air > via sorted_rows
```

`explain()` reports how each part of the predicate was evaluated — here the
sorted-rows index answered the range predicate exactly, without scanning. With
`kind` left unindexed, a query on it would report `via scan`. The
[user guide](../guide/indexes.md) covers the three index families and how to
choose one.

## Appends invalidate, refreshing restores

By default, appending does not rewrite indexes — the hot write path stays
fast, and every index is left detectably stale. Queries notice and quietly
fall back to scanning, so results are always correct. When the writing is
done, one call rebuilds them:

```python
table.append(
    {
        "station": ["KIAD"],
        "kind": ["automatic"],
        "t_air": [25.9],
        "samples": [55],
    }
)
table.refresh_indexes()
```

```text
1
```

Appends that must keep indexes valid throughout (at a cost on the write path)
can pass `append(..., maintain_indexes=True)`.

## Reopen and validate

Everything written so far is just HDF5 in a group. Reopening is the same two
steps in reverse — open the file with h5py, then hand the group to
{meth}`Table.open <h5col.Table.open>`:

```python
f.close()

f = h5py.File("quickstart.h5", "r")
table = Table.open(f["obs"])
table.column_names
```

```text
['station', 'kind', 't_air', 'samples']
```

```python
table.validate()
f.close()
```

{meth}`~h5col.Table.validate` checks the convention's consistency rules and
raises on the first violation; with `deep=True` it also re-derives every valid
search index and compares contents.

## Where to go next

- The [user guide](../guide/index.md) treats each of these topics properly:
  the [data model](../guide/data-model.md),
  [column datatypes](../guide/column-types.md),
  [missing values](../guide/missing-values.md),
  [list columns](../guide/list-columns.md),
  [filters](../guide/filters.md), and
  [search indexes](../guide/indexes.md).
- [Queries](../queries/index.md) documents the full predicate syntax and the
  query planner.
- The {doc}`NYC taxi notebook <../notebooks/06_nyc_taxi>` applies all of it
  to a real dataset.
