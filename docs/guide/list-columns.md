# List columns

Some columns do not hold one value per row. A list column stores a variable-length
sequence of values per row, including nested sequences, while keeping every
byte in ordinary, chunked, filterable HDF5 datasets.

## The offsets layout

A list column is a group (a direct child of the table group) with
`CLASS = "LIST_COLUMN"` and `KIND = "OFFSETS"` attributes. Its storage follows the same
offsets encoding used by Arrow's list arrays: all elements are stored
back-to-back in a values member, and a `uint64` `OFFSETS` dataset marks the
boundaries:row `i`'s elements are `VALUES[OFFSETS[i]:OFFSETS[i+1]]`, with
`OFFSETS[0] = 0` and the offsets non-decreasing. A nullable list column adds
a `MASK` dataset that records which rows are null.

A nullable list-of-strings column, for example, is laid out as:

```text
GROUP "tags"                         CLASS = "LIST_COLUMN"
│                                    KIND  = "OFFSETS"
├── DATASET "OFFSETS"   uint64       row boundaries into VALUES
├── DATASET "MASK"                   which rows are null lists
└── GROUP "VALUES"                   CLASS = "STRING_VALUES"
    ├── DATASET "OFFSETS"  uint64    per-string boundaries into CHARS
    ├── DATASET "CHARS"              the UTF-8 bytes, back to back
    └── DATASET "MASK"               which string elements are null
```

The values member takes one of three shapes, and this is where the layout
becomes recursive:

- a leaf dataset: a rank-1 dataset of any scalar column datatype
  (declared with {class}`~h5col.LeafValuesSpec`);
- a `STRING_VALUES` group: variable-length UTF-8 strings stored as an
  offsets-plus-bytes pair ({class}`~h5col.StringValuesSpec`), so even
  variable text lives in chunked, compressible datasets rather than on the
  HDF5 global heap;
- another list level ({class}`~h5col.NestedListSpec`), giving lists of
  lists to any depth.

Variable-length HDF5 datatypes are deliberately forbidden anywhere below a
list column; the offsets encoding exists precisely to avoid them.

## Declaring and writing

A list column is declared with {class}`~h5col.ListColumnSpec` alongside
ordinary columns:

```python
from h5col import LeafValuesSpec, ListColumnSpec, StringValuesSpec, Table

columns = [
    ListColumnSpec(name="depths", values=LeafValuesSpec(dtype="float32"), units="m"),
    ListColumnSpec(name="tags", values=StringValuesSpec(nullable=True), nullable=True),
]
table = Table.create(f.create_group("profiles"), columns)

table.append(
    {
        "depths": [[0.5, 1.0, 2.0], [], [0.7]],
        "tags": [["qc", "raw"], None, []],
    }
)
```

Rows are Python sequences; `None` writes a null row (permitted because `tags`
is nullable). Reading returns row-wise Python lists, with `None` for null
rows and decoded strings for string values:

```python
table["tags"].read()
```

```text
[['qc', 'raw'], None, []]
```

For deeper nesting, wrap another level — this declares `list<list<int32>>`:

```python
ListColumnSpec(
    name="pairs",
    values=NestedListSpec(values=LeafValuesSpec(dtype="int32"), nullable=True),
)
```

## Null, empty, and missing are three different things

The layout distinguishes states that formats without a mask conflate:

- A null row (`None`): the row has no value at all. Recorded in the
  top-level `MASK`; only nullable list columns can hold one. This is also
  what new rows become when a nullable list column is omitted from an
  append — and why a non-nullable list column must be provided in every
  append.
- An empty row (`[]`): the row's value is a sequence with zero elements —
  two equal consecutive offsets. Any list column can hold one.
- A missing element inside a row: leaf values reuse the ordinary
  [fill-value mechanism](missing-values.md), and string values use the
  `STRING_VALUES` group's own element mask.

{meth}`ListColumn.is_missing() <h5col.ListColumn.is_missing>` returns the
null-row mask.

## Storage control

Each piece of the layout accepts its own `chunks` and `filters`: on the
{class}`~h5col.ListColumnSpec` they apply to the top-level `OFFSETS`
dataset, and on a {class}`~h5col.LeafValuesSpec` to the element dataset. On
a {class}`~h5col.StringValuesSpec`, `filters` applies to the `CHARS` byte
buffer — usually the one worth compressing — while `chunks` sets the chunk
size of both the group's inner `OFFSETS` and `CHARS`. See
[filters](filters.md).

## Limits worth knowing

List columns are stored and appended like any other column, but they stand
outside two features: they cannot serve as a table's row-index column, and
they cannot carry [search indexes](indexes.md) — predicates on list columns
are not supported by the query layer. Where selection matters, keep the
selective keys in scalar columns beside the list.
