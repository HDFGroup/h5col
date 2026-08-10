# Reading into Python

An H5Col table holds more than a plain NumPy array can express. Some rows are
missing. A categorical column stores small integer codes that stand for
labels. A list column holds a different number of values in every row. When you
read a table, all of that has to arrive in some Python object, and no single
object is the right answer for every purpose.

So `h5col` offers two, and it is worth knowing which one you want before you
start:

- {meth}`Table.read <h5col.Table.read>` gives a dictionary of NumPy arrays.
  It needs nothing beyond NumPy and it is the right default for almost
  everything.
- {meth}`Table.to_arrow <h5col.Table.to_arrow>` gives an Apache Arrow
  table. It is the form that carries everything the table holds, and it is the
  bridge to pandas, Polars, DuckDB and Parquet. It needs the optional `pyarrow`
  package.

This chapter goes through what each one hands back, column type by column type,
and where each falls short.

## Reading into NumPy

Take a small table of weather observations, with a missing temperature, a
missing station kind, and a mix of list values:

```python
table.append({
    "station": ["KBOS", "KJFK", "KLGA"],
    "t_air":   [21.5, None, 23.1],
    "kind":    ["manned", "automatic", None],
    "checked": [True, True, False],
    "samples": [[1.0, 2.0], None, []],
})
```

{meth}`~h5col.Table.read` returns one entry per column:

| Column type | What you get |
|---|---|
| numeric | a masked array of the column's own dtype |
| boolean | a masked array of `bool` |
| fixed-length string | a masked array of `numpy.dtypes.StringDType` |
| categorical | a masked array of the labels, not the codes |
| list column | a Python list, one entry per row |

Every column except a list column comes back as a
{class}`numpy.ma.MaskedArray`. That is the part most worth explaining.

## Why missing values arrive masked

A missing row is stored as the column's fill value: for `t_air` above, that is
`-999`. If reading simply handed you the stored numbers, you would get this:

```python
table["t_air"].read(masked=False)
```

```text
array([  21.5, -999. ,   23.1], dtype=float32)
```

Nothing in that array says the middle value is not a real measurement. Take the
average and you get `-318.1`, which is not a temperature anyone recorded. The
mistake is easy to make and produces a plausible-looking number, which is the
worst kind of mistake.

By default you get the same values with a mask alongside them:

```python
table["t_air"].read()
```

```text
masked_array(data=[21.5, --, 23.100000381469727],
             mask=[False,  True, False],
       fill_value=-999.0,
            dtype=float32)
```

Now the average is `22.3`, because NumPy skips the masked entry. Sums, counts,
minimums and the rest all behave the same way. You did not have to remember
anything, which is the point.

The mask itself is an ordinary boolean array, `True` where the row is missing:

```python
table["t_air"].read().mask
```

```text
array([False,  True, False])
```

It always agrees with {meth}`Column.is_missing() <h5col.Column.is_missing>`,
which is the same question asked directly.

### Every column is masked, even when it cannot be

Boolean columns cannot have missing rows because the H5Col convention does not
allow a fill value for them. And some columns simply have no missing rows in
them. Those still come back as masked arrays, with a mask that is `False`
everywhere.

This is deliberate. If the type of `result["checked"]` depended on whether that
particular column could be missing, then any code looping over the columns of a
table would have to check before it could do anything, and would break the
first time it met a table whose author made a different choice. One type per
kind of column is easier to write against than one type per column.

### Turning it off

Pass `masked=False` to get plain arrays:

```python
table.read(masked=False)
```

Missing rows then hold the fill value, with nothing to mark them, and it is
back to you to call {meth}`~h5col.Column.is_missing` and apply it. The keyword
works the same way on
{meth}`Column.read <h5col.Column.read>`,
{meth}`Column.read_rows <h5col.Column.read_rows>`,
{meth}`Table.read <h5col.Table.read>` and
{meth}`Selection.read <h5col.query.Selection.read>`.

### Two habits worth picking up

Use `.tolist()`, not `list()`. They differ, and only one of them does what
you probably want:

```python
table["t_air"].read().tolist()
```

```text
[21.5, None, 23.100000381469727]
```

```python
list(table["t_air"].read())
```

```text
[np.float32(21.5), masked, np.float32(23.1)]
```

`.tolist()` turns a missing row into `None`. Plain `list()` gives NumPy's
`masked` marker, which is not `None` and will not compare equal to it. If you
have code that tests `if value is None`, reach for `.tolist()`.

Know which NumPy functions keep the mask. A good many do not.
`np.concatenate`, `np.stack`, `np.append` and `np.where` all return something
that is still a masked array by type but has quietly lost its mask, and
`np.asarray` hands back the underlying values. NumPy provides masked versions —
`np.ma.concatenate` and friends — and those are the ones to use.

When the mask does get dropped, what you are left with is the stored values,
fill value and all. That is the same thing `masked=False` would have given you,
so a lost mask never leaves you worse off than not having asked for one. It is
still worth avoiding.

If plain fill values are deliberately needed, `.filled()` is the proper way to get them:

```python
table["t_air"].read().filled()
```

```text
array([  21.5, -999. ,   23.1], dtype=float32)
```

Default NumPy printing of masked array values shows all the sigits, whereas for
plain arrays it asjust based on the dtype. The numbers are identical and only
the display changed, but if you are showing values to somebody, rounding first
is worth the trouble. `.tolist()` behaves the same way.

One small oddity: for a column with no missing rows at all, the
`fill_value` shown in the array's display is a NumPy placeholder such as
`'N/A'` rather than the column's own. It is never used, `.filled()` on an
array with nothing masked returns the values untouched, so it is display noise
rather than anything to act on.

## Strings

A fixed-length string column reads back as real Python strings, held in a NumPy
array of `numpy.dtypes.StringDType`:

```python
table["station"].read().tolist()
```

```text
['KBOS', 'KJFK', 'KLGA']
```

`StringDType` keeps the text packed in one block of memory instead of building a
separate Python string object for every row. It supports comparing, sorting, and
finding unique values. However, one consequence of the NumPy implementation is
that the text is only checked for valid UTF-8 when reading specific values out,
not when the column is read into the array. If a string contains invalid
bytes, the error appears only when that particular value is accessed. String
columns written by `h5col` cannot get into this state because there are checks
on the way in.

## Categorical columns

You get labels, not codes:

```python
table["kind"].read().tolist()
```

```text
['manned', 'automatic', None]
```

The codes are still there if you want them, as
{attr}`Column.codes <h5col.Column.codes>`, and the label set as
{attr}`Column.categories <h5col.Column.categories>`.

There is one place where categoricals do not follow the general rule.
`.filled()` on other column types puts the column's fill value into the masked
slots; for a categorical there is no label to put there, because a missing row
has no category. `.tolist()` gives you `None`, which is the answer you want.

## List columns

A list column reads back as a plain Python list, one entry per row:

```python
table["samples"].read()
```

```text
[[np.float64(1.0), np.float64(2.0)], None, []]
```

There is no masked array here, and there cannot be. Rows hold different numbers
of values, and a NumPy array needs them all to be the same shape. A list column
already reports missing rows in the clearest way available: the row is
`None`. Note that `None` and `[]` mean different things. `None` is a row
with no value at all; `[]` is a row whose value is a list that happens to be
empty. The [list columns](list-columns.md) chapter goes into that distinction.

The values inside each row are NumPy scalars rather than plain Python numbers,
which is why the output above reads `np.float64(1.0)` and not `1.0`. They
compare and calculate exactly as expected.

These columns accept the `masked` keyword and ignore it, so you can pass it
across a whole table without special-casing.

## The complete picture: Arrow

Three things a table can hold have no NumPy equivalent at all:

- a missing value that is genuinely absent, rather than a particular number
  standing in for absence;
- a categorical column with its complete semantics — a small set of labels plus one
  code per row, rather than expanded to a full label for every row;
- a list column, with its own missing values at every level of nesting.

Arrow enables preserving the entire H5Col table semantics. {meth}`Table.to_arrow <h5col.Table.to_arrow>` gives you a
`pyarrow.Table`:

```python
table.to_arrow()
```

```text
station: large_string
t_air: float
kind: dictionary<values=string, indices=int8, ordered=0>
checked: bool
samples: large_list<item: double>
----
station: [["KBOS","KJFK","KLGA"]]
t_air: [[21.5,null,23.1]]
kind: [  -- dictionary: ["manned","automatic"]  -- indices: [0,1,null]]
checked: [[true,true,false]]
samples: [[[1,2],null,[]]]
```

The missing temperature is `null`, not `-999`. The `kind` column is still a
dictionary of two labels with one code per row. The `samples` column keeps its
rows and its missing row.

Each column's `units`, `description` and valid-range attributes travel along as
Arrow field metadata, under names beginning `h5col.`, and they survive being
written to Parquet and read back. So a table exported this way does not lose
the descriptions that made it understandable.

The most of the tabular ecosystem is just one call away from an Arrow table:
`.to_pandas()`, Polars, DuckDB, `pyarrow.parquet.write_table`. Arrow is also the
faster path for list columns, by a wide margin. `h5col` stores them in nearly
the layout Arrow uses. Where reading a list column into Python lists has to
build every row as an object, the Arrow export mostly hands the same blocks of
memory straight over. On a column of two hundred thousand rows that is roughly
twenty to thirty times faster.

`pyarrow` is not required to use `h5col`. Install it alongside if you want this
data export feature:

```bash
pip install h5col[arrow]
```

## Where these forms fall short

Worth knowing to avoid any surprises.

**An empty string reads as missing.** The default fill value for a string
column is the empty string, so a row genuinely containing `""` cannot be told
apart from a row with no value. If empty strings are real data in your model,
choose a different fill value when creating the column. The
[missing values](missing-values.md) chapter covers the choice.

**A mask can be lost quietly.** Covered above: several NumPy functions drop it
without complaint. The `np.ma` versions do not.

**List columns cannot carry a mask**, so a table read into NumPy is not
uniform: most columns are masked arrays, list columns are Python lists. Arrow
does not have this split.

**Reading a whole column reads the whole column.** Both `read()` and
`to_arrow()` bring every row into memory. To read part of a table, select rows
first with {meth}`Table.select <h5col.Table.select>` and read from the
selection, or use {meth}`Column.read_rows <h5col.Column.read_rows>`, both of
which fetch only the chunks they need. The [queries](../queries/index.md)
section covers selection properly.

**Arrow needs a dependency.** It is optional on purpose: a file written by
`h5col` can be read with nothing but HDF5, and requiring a large package for
the base case would undercut that.
