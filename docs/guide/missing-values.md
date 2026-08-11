# Missing values

Real tables have holes: the sensor did not report, the field was not filled
in, the join found no partner. H5Col gives missing values a precise,
file-level definition instead of leaving each application to invent one.

## The model

A missing row simply stores the column's fill value, the same fill value
HDF5 already defines as a dataset creation property. What the convention adds
is one test that every consumer applies:

```text
missing(v, fill) =  isnan(v)   if fill is NaN
                    v == fill  otherwise
```

The two cases exist because `NaN` does not compare equal to itself, so asking
`v == fill` would answer no for every row. Any other fill value does compare
equal, and the first line never applies. The package runs the test for you:
{meth}`Column.is_missing() <h5col.Column.is_missing>` returns the boolean
mask over the logical rows, and {func}`h5col.is_missing` is the same test as
a standalone function.

Reading a table applies this for you: every column but a list column comes back
as a masked array, so a missing row is skipped by an average rather than
counted as its fill value. [Reading into Python](reading-into-python.md)
explains the shape of everything a read hands back.

## Every column has a fill, chosen or recommended

When a {class}`~h5col.ColumnSpec` does not set `fill_value`, the column
receives the convention's recommended fill for its datatype:

| Column dtype | Recommended fill |
|---|---|
| `int8` / `int16` / `int32` / `int64` | `-127`, `-32767`, `-2147483647`, `-9223372036854775807` |
| `uint8` / `uint16` / `uint32` / `uint64` | the type maximum |
| `float32` / `float64` | `9.9692099683868690e+36` |
| fixed-length string | `""` (the empty string) |
| categorical codes | `-1` (signed) or the type maximum (unsigned) |

({func}`h5col.recommended_fill` returns the values in the first four rows.
The categorical default is chosen from the code dtype itself by
`h5col.categorical.default_categorical_fill`.)

Two of these deserve a note. The floating-point default is not `NaN` but the
value long used as the default fill in the netCDF world, kept here so files
behave consistently across that ecosystem. `NaN` remains available as an
explicit choice. And the empty string as the string fill means an empty value
reads as missing by default, so if empty strings are data in your model,
choose the column's semantics deliberately.

Boolean columns are the exception: a boolean cannot be missing, declares no
fill at all, and therefore must be provided in every append.

## Two ways to mark a missing number

For floating-point columns you will usually pick one of two styles.

Setting aside an ordinary number keeps a missing row separate from anything a
calculation could produce. A `NaN` that slips into an average turns the whole
answer into `NaN`, which tells you nothing about where it came from, whereas a
fill value that slips into arithmetic gives a number obviously outside the
range the data lives in. It is also the only option for integer columns, which
have no `NaN`:

```python
ColumnSpec(name="samples", dtype="int32", fill_value=-1, valid_min=0)
```

The `NaN` style is the natural one when the data flows to and from NumPy or
pandas, where `NaN` already means missing:

```python
ColumnSpec(name="t_air", dtype="float64", fill_value=np.nan, units="degC")
```

The {doc}`NYC taxi example <../notebooks/06_nyc_taxi>` uses both styles side
by side on real data.

## Valid ranges keep the fill honest

A fill value only works if no genuine value can equal it. Declaring
`valid_min`/`valid_max` makes that checkable. At creation time the fill must
lie strictly outside the declared range, or {class}`~h5col.FillValueError`
is raised. In the `samples` spec above, `-1` is provably not a measurement,
because measurements start at 0.

## Writing missing rows

There are three ways a missing row comes to exist:

- Append `None` in place of a value. Whatever the column's datatype, `None`
  is stored as that column's fill value, so a `None` in an integer column
  filled with `-1` becomes `-1`, and a `None` in a NaN-filled float column
  becomes `NaN`. In a categorical column it becomes the fill code and reads
  back as `None`.
- Write the fill value itself — `NaN` into a NaN-filled float column, `-1`
  into one filled with `-1`. This is equivalent to writing `None`, and is
  often the natural form when the data already arrives as a NumPy array.
- Omit the column from an append entirely. The column is extended and its new
  rows keep the fill value.

The exception is a column with no fill value to store. A boolean column declares
none so it must be supplied in every append, and a `None` in one raises
{class}`~h5col.SchemaError` rather than being coerced to `False`. Non-nullable
list columns must likewise always be provided.

## Missing values in queries

Selections treat missing rows with three-valued logic, exactly as SQL and
pyarrow do: a comparison with a missing value is neither true nor false but
unknown, and only rows whose whole predicate evaluates true are selected. The
consequence worth remembering is that `field("x") == 5` and its negation
together do not cover the missing rows — those match only
`field("x").is_null()`. The [queries](../queries/index.md) section defines the
semantics fully.

## List columns are different

A list column distinguishes a null list (no value for the row) from an empty
list (a value with zero elements) using an explicit `MASK` dataset, not a
fill value. The [list columns](list-columns.md) chapter explains that
in more detail.
