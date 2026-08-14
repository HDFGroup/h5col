# Writing an Arrow table into H5Col

Arrow is where a lot of tabular data already is. Parquet reads into it, DuckDB
and Polars hand it out, pandas converts to it, and a table that arrives over the
wire during interprocess communication is Arrow before it is anything else. So
the other direction matters as much as the [export to
Arrow](reading-into-python.md#reading-into-arrow).

To store a `pyarrow.Table` as an H5Col table is one call:

```python
import h5py
import h5col

with h5py.File("weather.h5", "w") as f:
    table = h5col.Table.from_arrow(f.create_group("observations"), arrow_tbl)
```

The rows are written batch by batch rather than all at once, so importing a
large table costs about one batch of memory.

## What cannot come across

Arrow's datatype system is richer than H5Col's, and the differences cannot be
silently handled. A timestamp column has no H5Col equivalent, and neither do
dates, times, durations, decimals, structs, maps, or unions.

None of these are approximated. Each is refused with an exception which
recommends an alternative:

```text
SchemaError: column 'observed_at': Arrow type timestamp[us] cannot be stored
in H5Col — H5Col has no datetime type; store the epoch offsets as an integer
column and describe the encoding in its attributes
```

Silently storing a timestamp as an integer would produce a file that reads back
as numbers no one can interpret. Converting the column yourself, and saying in
its `description` or some other attributes what the numbers mean, produces one
that survives the trip.

A fixed-size list is refused for a different reason, and it is worth knowing
which. HDF5 has the datatype for it: an array datatype (`H5T_ARRAY`) holds a
fixed count of elements per row. This is what Arrow's `fixed_size_list` is, and
the convention lets a column dataset carry any HDF5 datatype. However, h5py
cannot set a fill value for this datatype, which is how H5Col marks a missing
row. Converting such an Arrow column to an H5Col variable list type is left for
the user rather than silently done, because it drops a guarantee: a list column
does not fix its row lengths, so nothing afterwards holds the imported column to
a fixed count.

Binary columns are refused for a third reason. H5Col does store raw bytes with
an opaque (`H5T_OPAQUE`) fixed-width column, a fill-value rule, a sorting order,
and a hash for search indexes. What has no home is Arrow's variable-length
`binary`: with no fixed width, there is nothing to size a column to.
`fixed_size_binary` does carry a width and this package does not support it yet.

The same boundary runs the other way. A column whose datatype the Arrow export
has no type for — opaque, compound, array, complex — is refused by name when
you call `to_arrow`, rather than being handed to `pyarrow` to fail on. The data
is still there: read it with `read()`, or reach the stored values through the
column's `dataset`.

## Nulls have to become values

This is the part worth understanding before importing anything.

Arrow marks a missing value with a null: a bit in a separate buffer, holding no
opinion about what the value would have been. H5Col marks one with a [fill
value](missing-values.md) drawn from the column's own datatype domain. The two
models do not line up on their own, so every column that can hold a null needs a
fill value chosen for it.

`from_arrow` chooses the recommended fill for the column's datatype, and then
checks it against the data. If the fill already occurs as a real value, the
import is refused:

```text
SchemaError: column 'delta': the recommended fill value np.int8(-127) occurs
in the data, so those rows would read as missing; pass a ColumnSpec with a
fill_value the column does not contain
```

This check is worth the trouble it occasionally causes. Without it you get a
file that passes `validate(deep=True)` and reads back with rows quietly
missing that were never missing at all, the kind of error that surfaces
months later in someone's analysis.

The same rule applies inside a list column, at every level of nesting: a null
element becomes the leaf's fill value, so a leaf that already contains that
value is refused, too.

Two cases have no fill available at all:

- **Boolean columns.** H5Col forbids a boolean from declaring a fill value, so a
  boolean column holding nulls is refused. Drop the nulls, or import the column
  as an integer.
- **String columns containing an empty string.** The recommended fill for a
  string column *is* the empty string, so a column holding one needs a
  different fill named explicitly. That is what the specs below are for.

## Taking the specs into your own hands

Two things about a H5Col column have no Arrow equivalent whatsoever:
[chunking and filters](filters.md). Nothing in an Arrow schema says how the
data should be laid out on disk or what should compress it, and those are
exactly the decisions that make a stored column fast or slow to read.

So `from_arrow` on its own gives you an unfiltered table with default chunking.
To decide otherwise, ask what it would do, adjust, and hand it back:

```python
specs = h5col.specs_from_arrow(arrow_tbl)

for spec in specs:
    spec.chunks = 65_536
    spec.filters = h5col.FilterPipeline([h5col.Shuffle(), h5col.Deflate(4)])

table = h5col.Table.from_arrow(group, arrow_tbl, specs=specs)
```

Nothing is read from or written to a file by `specs_from_arrow`, so this is a
cheap thing to do first and examine its findings. It is also where you set a
fill value the data does not contain, widen a string column beyond what its
current values need, or fix anything else the inference got merely defensible
rather than right.

What the checks will not let you do is skip them. Supplying specs chooses the
fill value, it does not waive the check that the fill is absent from the data,
because that is the one importing mistake that produces a conformant file with
unreadable rows.

## What the inference reads from the data

Most of a column's spec comes from its Arrow type. Three things cannot:

- **String widths.** H5Col string columns are fixed-length; Arrow's are not.
  The column is scanned and sized to its widest value, which means a later
  `append` of a longer string raises rather than truncating. If you expect
  longer values later, set a wider budget in the spec.
- **Category labels.** A dictionary column's labels are collected across all
  its chunks, since two chunks of one Arrow column may carry different
  dictionaries. The dictionary's index type is kept as the code datatype, so a
  column that arrives with `int32` codes is stored with `int32` codes.
- **Which levels of a list column hold nulls.** A `MASK` dataset is created
  only at the levels that actually need one.

## Metadata

Field metadata beginning with `h5col.` is read back as the column's own
annotations: `units`, `units_vocabulary`, `description`, `valid_min`,
`valid_max` and `ordered`. These are the same keys the export writes, so a
table that made the trip in the other direction arrives with its descriptions
intact.

Any other field metadata is carried across as an ordinary HDF5 attribute on the
column, so a producer's own keys are kept rather than dropped. The one
restriction is that such a key may not be an H5Col reserved name, or one it
writes itself. Such filed metadata raises an
{exc}`~h5col.exceptions.ReservedNameError`.

Column names get the same treatment. A name Arrow permits but HDF5 cannot
store as a link, or one H5Col reserves, is refused rather than mangled into
something storable.

## The round trip

Export, import, and export again gives back an equal Arrow table — same types,
same values, same field metadata — for every column kind H5Col can store. The
test suite checks this per kind, which is the honest way to state what the two
directions guarantee together.

What it does not claim is that the *first* export is unchanged by the trip.
Arrow has types H5Col stores as something close but not identical: a `string`
column comes back as `large_string`, and a `list` as `large_list`, because
those are the layouts H5Col's own storage matches. The pair settles after one
hop, and stays there.

`pyarrow` is not required to use `h5col`. Install it alongside if you want to
import or export Arrow tables:

```bash
pip install h5col[arrow]
```
