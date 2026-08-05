# Search indexes

Every `h5col` query works with no index at all by scanning a compressed column
as the universal fallback. A search index is an optional, persistent
acceleration structure stored inside the file, next to the data it
summarizes. Indexes never change what a query returns, only how much data it
touches. The convention's validity protocol guarantees a stale index is
detected and ignored rather than believed.

## The three families

`CHUNK_MINMAX`
: Stores each chunk's minimum and maximum value, plus three small per-chunk
  counters (row, NaN, and fill counts). Given a range predicate, it answers
  with the chunks that might contain matches, and the query layer reads and
  verifies only those. Its answers are a superset by design, its cost is
  tiny (one entry per chunk), and it shines when values are clustered —
  timestamps in an append-ordered table being the classic case.
  Resolution equals the [chunk shape](filters.md), which is worth choosing
  deliberately on columns you will query.

`SORTED_ROWS`
: Stores a permutation of row numbers ordering the column's values. It is the
  right tool for selective range queries over unclustered data. Range and equality
  predicates are answered exactly, by binary search, no matter how the values
  are scattered. It costs one integer per row.

`BITMAP`
: Stores one bitset per distinct value: bit `i` of the bitset for value `v`
  says whether row `i` equals `v`. Equality and set-membership predicates are
  answered exactly by reading one bitset per queried value. It is built for
  low-cardinality columns (categoricals and booleans) where it is compact
  and effectively instant.

## Building an index

{meth}`Table.build_index <h5col.Table.build_index>` (an alias of
{meth}`~h5col.Table.add_search_index`, also available on
{class}`~h5col.Column`) builds one:

```python
table.build_index("payment_type", "BITMAP")
table.build_index("total_amount", "SORTED_ROWS")
table.build_index("tpep_pickup_datetime", "CHUNK_MINMAX")
```

With the kind omitted, the choice is automatic: `BITMAP` for boolean and
categorical columns, `CHUNK_MINMAX` for any other column this implementation can
index (a column whose datatype no family supports raises
{class}`~h5col.SchemaError` instead). Building an index is not a table mutation
because the data does not change, so existing indexes stay valid.

Each index dataset gets a readable default name, `<column>__<kind
lowercased>` (for example `total_amount__sorted_rows`), and an optional
`description`. The name carries no meaning. What binds an index to its
column is an object reference in the column's `SEARCH_INDEX_LIST`
attribute, so renaming an index dataset cannot quietly detach it.

## What is in the file

Index datasets live in the reserved `SEARCH_INDEXES` group under the table.
Each carries a `KIND` attribute naming its family, and two validity tokens:
`SOURCE_GENERATION` and `SOURCE_NROWS`, the table state it was built from.
The table itself carries a `GENERATION` counter, created with the first
index and incremented by every mutation of committed data.

An index is valid exactly when its tokens match the table's current `GENERATION`
and `NROWS`. That check, {meth}`Table.index_is_valid
<h5col.Table.index_is_valid>`, is cheap, needs no data reads, and is applied by
every consumer before trusting an index.

## Staleness is normal, and safe

By default, {meth}`~h5col.Table.append` and {meth}`~h5col.Table.truncate` do
not rewrite indexes. They bump `GENERATION`, which invalidates every index
detectably, and move on but the hot write path stays fast. Queries notice the
invalid tokens and fall back to scanning, so results stay correct. The plan
from {meth}`Selection.explain() <h5col.Selection.explain>` will simply show
`scan` where an index would have been used.

When a batch of writes is done, one call rebuilds everything:

```python
table.refresh_indexes()   # returns the number of indexes rebuilt
```

Writers that need indexes to remain valid through every append can pass
`append(..., maintain_indexes=True)`, trading write speed for always-fresh
indexes: supported indexes are rewritten inside the append protocol,
tokens-before-content, so even a crash mid-append cannot leave an index
believed-valid but wrong.

{meth}`Table.validate(deep=True) <h5col.Table.validate>` goes one step
further when you want proof: it re-derives every valid index from its column
and compares.

## Using indexes directly

The query layer is the intended consumer, but the wrappers
({class}`~h5col.ChunkMinMaxIndex`, {class}`~h5col.SortedRowsIndex`,
{class}`~h5col.BitmapIndex`) expose the underlying primitives:
`prune(op, value)` returns candidate chunk ids (a superset to verify), and
the `rows(...)`/`isin(...)` methods return exact matching row positions.
One contract detail matters to direct callers: a bitmap whose value
enumeration is not exhaustive (its `exhaustive` attribute is false — for
example, a float column held NaN when the index was built) answers `None`
for a value it cannot prove, and the caller must fall back to a scan.
Called on a stale index, all of these methods raise
{class}`~h5col.StaleIndexError` — the silent fallback to scanning happens
only in the query layer, which always has a correct alternative.

## Missing rows

Missing rows never match a comparison, and the index implementations
preserve that rule exactly. Fill values and NaNs are segregated when an
index is built, and `is_null()` / `is_valid()` predicates are answered from
the missing-value mask, not from indexes. You never need to think about
missing rows when deciding whether to index a column.

## Choosing, in one paragraph

Index the columns your predicates actually filter on. Use `BITMAP` where
the column is categorical or boolean; prefer `SORTED_ROWS` where range
queries are selective and the data is not sorted; add `CHUNK_MINMAX` where
data is naturally clustered (it is nearly free) or simply accept it as the
automatic default. The fourth family defined by the convention,
`CHUNK_BLOOM`, is not implemented in this package — see
[conformance](../about/conformance.md).
