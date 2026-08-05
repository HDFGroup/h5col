# Queries

`h5col` includes a small query layer for the question every table eventually
faces: which rows match? Predicates are written in a syntax deliberately
parallel to pyarrow's with {func}`~h5col.field`, and evaluated against the table
using any valid [search index](../guide/indexes.md) that applies or scanning
where none does. Results are exact either way, indexes only change the cost.

```{toctree}
:maxdepth: 1

syntax
```

## The selection model

{meth}`Table.select <h5col.Table.select>` builds a
{class}`~h5col.Selection` — a lazy handle that evaluates once, on first
use:

```python
sel = table.select((field("payment_type") == "Credit card") & (field("total_amount") > 100.0))

sel.count            # how many rows matched (also len(sel))
sel.row_positions    # their positions, as a sorted integer array
sel.read(["tpep_pickup_datetime", "total_amount"])
                     # {name: array} for the matching rows only
sel.explain()        # the QueryPlan that produced the answer
```

`read()` decodes values exactly like a full-column read and returns only the
matching rows of only the requested columns.

Worth knowing where the cost goes. Evaluating the predicate reads only the
chunks an index leaves in play, and materializing the result does the same:
when the matching rows are confined to a modest share of the column's chunks,
`read()` fetches just those chunks with coalesced block reads and skips the
rest. For example, on a four-million-row table, taking 500 clustered rows from a
compressed column that way runs about forty times faster than reading the
column through, and holds a fraction of the memory. When the matches instead
touch nearly every chunk there is nothing left to skip, so `read()` reads the
column straight through, which is the cheaper option at that point.

Two things fall outside that: narrowing `columns=` always reduces the work,
whatever the predicate matched, and list columns are still read in full and
then subset. If only the answer rather than the values is needed, `count()`
and `row_positions` stop before this stage entirely.

Two conveniences on {class}`~h5col.Table` cover the common cases:
`table.read(where=...)` reads matching rows directly (and with
`explain=True` returns a `(result, plan)` pair), and `table.count(where)`
counts without materializing anything.

## What can be passed as a predicate

Anything the [syntax reference](syntax.md) defines:

- an {class}`~h5col.Expression` built from {func}`~h5col.field`
  comparisons, combined with `&`, `|`, and `~`;
- a single tuple, `("total_amount", ">", 100.0)`;
- a list of tuples, meaning their conjunction (AND);
- a list of lists of tuples, meaning an OR of ANDs — pyarrow's filter
  format, accepted verbatim.

`None` selects every row.

## How a query is evaluated

The expression is first normalized to disjunctive normal form as an OR of
AND-terms. Each term's predicates are then planned independently, choosing
per predicate:

- a `BITMAP` index for equality and membership on categorical or boolean
  columns → exact row answers;
- a `SORTED_ROWS` index for comparisons on orderable columns → exact row
  answers by binary search;
- a `CHUNK_MINMAX` index for comparisons → candidate chunks, which the
  engine reads and verifies value by value;
- otherwise, a column scan.

Only indexes that pass the validity check are considered, so a stale index is
never consulted (see [staleness](../guide/indexes.md)).

Each predicate produces the set of rows it matched, and those sets are then
combined in two steps. Within a single AND-term, a row has to appear in every
predicate's set to survive, so the term keeps their intersection. Across
terms, a row needs to satisfy only one of them, so the query keeps their
union, counting a row once however many terms matched it.

Missing values are treated the same way whichever route a predicate took. A
missing row is not an ordinary value that happens to compare false. It is
unknown, and so it satisfies neither a comparison nor the negation of one.
Since the three index families and the scan all follow that same rule, the
plan the engine settles on changes only how long a query takes, never which
rows come back.

## Reading the plan

{meth}`Selection.explain() <h5col.Selection.explain>` returns a
{class}`~h5col.QueryPlan` whose string form shows the decision per
predicate:

```python
print(table.select((field("payment_type") == "Cash") & (field("total_amount") >= 20.0)).explain())
```

```text
QueryPlan: 1784 / 25000 rows matched
   AND-term 0:
      payment_type == via bitmap
      total_amount >= via sorted_rows
```

The method names it can report are `bitmap`, `sorted_rows`,
`chunk_minmax+verify`, `scan`, `presence` (for `is_null()`/`is_valid()`,
answered from the missing-value mask), and `categorical-empty` (an equality
against an unknown categorical label, answered as provably empty without
touching data or index). The plan is
a small dataclass — `nrows`, `matched`, and per-term leaf entries — so it
can be inspected programmatically as well as printed.

When a query is slower than expected, `explain()` answers the usual
questions directly: whether an index was used at all, whether it had gone
stale (`scan` where you expected an index), and how many candidate chunks a
`CHUNK_MINMAX` verification had to read.
