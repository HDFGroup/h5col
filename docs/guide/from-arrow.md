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

Variable-length `binary` is refused for a third reason, and this one cannot be
worked around at all. Padding the blobs to a common width would not survive the
trip: no byte is safe to strip from the end of a blob, since any byte can
legitimately be part of one. What is missing is the length, and no padding
scheme carries it. Convert to `fixed_size_binary` if the values do share a
width — that maps exactly, as the next section explains.

## Raw bytes

`fixed_size_binary[n]` becomes an *opaque* column: `n` bytes per row, stored
back to back with no offsets, which is byte-for-byte what Arrow holds. The
buffer is handed over rather than converted, in both directions.

Opaque columns raise the missing-value question in its sharpest form. A fill
value has to be a value the column's data will not contain, and for raw bytes
there is no such value in principle — any byte string might be real data. H5Col
picks one that is merely very unlikely: the ASCII marker `FILL` followed by
rising byte values, so an eight-byte column's fill is

```text
46 49 4c 4c 01 02 03 04     "FILL...."
```

The rising tail is the part that earns its keep. All zeros and all `0xFF` are
what zero padding, erased flash and uninitialized memory leave behind, and a
fill made of either would collide constantly. A counting sequence is something
data almost never is.

Unlikely is not impossible, so the collision check applies here as everywhere
else: a column that does contain the pattern is refused, and you name a fill of
your own through the specs. One case deserves care — a one-byte opaque column
has only 256 possible values and the recommended fill claims one of them, which
is a real risk rather than a remote one.

What has no home is Arrow's variable-length `binary`. With no fixed width, there
is nothing to size a column to.

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
in the data, so those rows would read as missing; np.int8(-126) does not
occur — pass a spec with that as its fill_value if it lies outside the
column's logical range
```

This check is worth the trouble it occasionally causes. Without it you get a
file that passes `validate(deep=True)` and reads back with rows quietly
missing that were never missing at all, the kind of error that surfaces
months later in someone's analysis.

The suggested value is a genuine offer and not a decision made for you. Note
what it does *not* claim: a value absent from the data is not the same as a
value outside the column's logical range, which is what H5Col asks for. If
`-126` is a reading your instrument can produce, take a different one or widen
the datatype, which is what the convention prescribes for a column with no value
to spare. That is also what the message says when nothing near the limits of the
type is free:

```text
... nor is any other value near the limits of uint8; a column using its whole
datatype has no value left to mark absence, so widen the datatype, which is
what H5Col prescribes for this
```

Choosing a fill value on your behalf would be the wrong kind of helpful. The
recommended value is documented, so a reader knows what a `uint8` column's fill
is without opening the file; one picked per import from whatever the data
happened to contain is knowable only after the fact. And if a later `append`
brought in a value we had quietly reserved, those rows would start reading as
missing.

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

`specs_from_arrow` answers what the columns would look like, and stops there.
It does not check a fill value against the data, which is deliberate: setting
one is the way out of a collision, so getting hold of the specs cannot itself
be the thing that fails. A spec whose `fill_value` is unset means the
recommended value for its datatype, as it does anywhere else in the package.

What you cannot do is skip the check. It runs when the table is written,
whichever way the specs arrived, because a fill value that occurs in its own
column is the one importing mistake that produces a conformant file with
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
