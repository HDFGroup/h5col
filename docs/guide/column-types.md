# Column datatypes

A scalar column holds one value per row, and H5Col defines four families for it:
numeric columns, fixed-length strings, booleans, and categoricals. Each family
maps to plain HDF5 datatypes, so files remain readable everywhere. What the
convention adds is precise semantics on top. Variable-length values per row are
the job of [list columns](list-columns.md), which have their own chapter.

Every family is declared through the same object, a {class}`~h5col.ColumnSpec`,
whose fields cover identity (`name`, `description`), the datatype, storage
(`chunks`, `filters` — see [filters](filters.md)), and value semantics
(`fill_value`, `valid_min`, `valid_max`, `units`, `units_vocabulary`).

## Numeric columns

The convention's core numeric set covers: `int8` through `int64`, their unsigned
counterparts, `float32`, and `float64`. They can be declared in several flavors:
`np.float64`, `"float64"`, or `np.dtype("<f8")`:

```python
ColumnSpec(name="t_air", dtype="float64", units="degC", valid_min=-90, valid_max=60)
```

Unless you pass `fill_value=`, the column receives the convention's recommended
per-`dtype` sentinel as its fill (the exact values are tabulated in [missing
values](missing-values.md)). A numeric dtype outside that set, e.g., `float16`,
has no recommended sentinel, so it is accepted only with an explicit
`fill_value=`. Creation raises a {class}`~h5col.FillValueError` without one.
`valid_min` and `valid_max` are stored as column attributes for consumers, and
they guard the fill: a fill value inside the declared valid range is rejected at
creation with {class}`~h5col.FillValueError`.

## Fixed-length strings

A string column declares its byte size with {class}`~h5col.FixedString`:

```python
ColumnSpec(name="station", dtype=FixedString(nbytes=8))
```

Values are UTF-8 encoded into exactly `nbytes` bytes of storage per row,
padded with NULs. The budget counts bytes, not characters — a
four-character string of non-ASCII text can need more than four bytes. The
guarantee that matters is that a value whose encoding exceeds the budget raises
{class}`~h5col.OversizedStringError` and names the row, instead of being
silently truncated. If the data can outgrow the column, you find out at write
time, not at analysis time.

Reading decodes back to Python strings (a NumPy object array). The default
fill value for a string column is the empty string, so an empty value reads
as missing by default; if empty strings are meaningful data in your model,
account for that when choosing the column's fill.

Why fixed-length rather than HDF5's variable-length strings? Variable-length
data is stored on the HDF5 file's global heap, outside the dataset's chunks.
This means chunk compression never touches the actual characters, and reads
scatter across the file. Fixed-length strings keep every byte inside the chunk
pipeline, where filters and contiguous reads work as intended. When string
lengths genuinely vary too much for a fixed length, a [list column with string
values](list-columns.md) provides variable-length text that still lives in
filterable chunks.

## Boolean columns

HDF5 has no native boolean type, so the convention fixes one: an enumeration
over signed 8-bit integers with exactly two members, `FALSE = 0` and
`TRUE = 1`. Declare it with {func}`~h5col.bool_dtype`:

```python
ColumnSpec(name="qc_passed", dtype=bool_dtype())
```

Booleans read back as a NumPy boolean array. The implementation is strict in
both directions: appending accepts Python/NumPy booleans or exact 0/1
integers ({class}`~h5col.SchemaError` otherwise), and reading a stored code
other than 0 or 1 raises {class}`~h5col.ConformanceError` rather than
guessing (NumPy would happily call every nonzero value true).

A boolean column declares no fill value because a boolean cannot be missing so
it must be supplied in every append, and it may not declare `valid_min` or
`valid_max`.

## Categorical columns

A categorical column stores small integer codes and keeps the label values
in a labels dataset under the table's `CATEGORIES` group. Declare one
by listing its categories; the code dtype is chosen automatically — the
smallest signed integer whose positive range covers the category count, so
up to 127 labels cost one byte per row:

```python
ColumnSpec(name="payment_type", categories=["Credit card", "Cash", "Dispute"])
```

Labels may be strings (stored as a fixed-length string dataset sized to the
longest label) or numbers. The column dataset carries a `CATEGORIES` object
reference to its labels dataset, so the association survives renames and
moves.

The API works in labels, not codes. {meth}`~h5col.Table.append` takes label
values. An unknown label raises {class}`~h5col.SchemaError`, and `None`
marks a missing row. {meth}`Column.read <h5col.Column.read>` returns
labels, with `None` where the code is the fill. The raw codes remain
available as {attr}`Column.codes <h5col.Column.codes>`, the labels as
{attr}`Column.categories <h5col.Column.categories>`, and an optional
`ordered` flag (for ordinal categories) round-trips through the spec and
{attr}`Column.ordered <h5col.Column.ordered>`.

The default fill code is `-1` for signed code dtypes (and the type maximum
for unsigned), which cannot collide with a valid code as long as the code
type leaves room for it. A fill that would collide, an unsigned code type
fully saturated by its categories, is rejected at creation, and an
explicit categorical fill must likewise lie outside `[0, ncategories)`.

Categoricals earn their keep twice: in storage, where a repeated 20-byte
label costs one byte per row, and in queries, where equality predicates
compare labels (`field("payment_type") == "Cash"`) and a
[bitmap index](indexes.md) answers them exactly without touching the column
data.
