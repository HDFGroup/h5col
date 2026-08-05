# The data model

Everything H5Col does rests on one idea: a table is an HDF5 group whose direct
children are its columns, each scalar column an ordinary rank-1 dataset,
each [list column](list-columns.md) a small group of such datasets. A small
set of attributes with reserved names gives the group its tabular meaning.
This chapter walks through that layout — what is in the file, and why.

## A table is a group

A group becomes a table by carrying the attribute `CLASS = "COLUMN_TABLE"`,
together with a `VERSION` string (this implementation writes `"1.0"`) and an
unsigned 64-bit `NROWS` — the committed row count. The table built in the
[quickstart](../start/quickstart.md) looks like this on disk:

```text
GROUP "obs"                          CLASS = "COLUMN_TABLE"
│                                    VERSION = "1.0"
│                                    NROWS = 5
│                                    TITLE = "Surface observations"
│                                    column-order = ["station", "kind",
│                                                    "t_air", "samples"]
│                                    GENERATION = 1   (arrives with the index)
├── DATASET "station"   shape (5,)   8-byte UTF-8 fixed string
├── DATASET "kind"      shape (5,)   int8 codes;  CATEGORIES → ref
├── DATASET "t_air"     shape (5,)   float64;  fill = NaN;  units = "degC"
├── DATASET "samples"   shape (5,)   int32;  fill = -1;  valid_min = 0
│
├── GROUP "CATEGORIES"
│   └── DATASET "kind__CATEGORIES"   the labels "manned", "automatic"
└── GROUP "SEARCH_INDEXES"
    └── DATASET "t_air__sorted_rows"    KIND = "SORTED_ROWS" + validity tokens
```

Nothing here is exotic HDF5. A tool with no knowledge of the convention sees a
group of one-dimensional datasets with descriptive attributes and can read
every value. A convention-aware reader additionally understands the row count,
the category labels, the missing-value rule, and the indexes.

Because a table is just a group, it can live anywhere in a file's hierarchy,
a file can hold any number of tables, and a table can sit beside arrays,
images, or other groups that have nothing to do with H5Col.

## The committed row count

`NROWS` is the table's logical length, and it is deliberately decoupled from
the physical extent of the column datasets. All column datasets must share one
extent (the equal-extent rule), and that extent may exceed `NROWS`; the rows
at positions `NROWS` and beyond are reserved storage that every consumer
ignores.

This decoupling is what makes writes safe. {meth}`~h5col.Table.append`
follows the convention's write protocol: extend the columns, write the new
values, flush them to the file, and only then update `NROWS` and flush again.
A reader, or a crash, can never observe a row count that points at
unwritten data. It also makes {meth}`~h5col.Table.truncate` cheap: shrinking
a table is a metadata operation that lowers `NROWS`, turning the tail rows
back into reserved storage without rewriting any column.

Two write-side rules follow from the same design. Rows are appended
column-wise, and every column supplied in one append must have the same
length. A column omitted from an append is extended and left at its fill
value, so its new rows read as missing — which is also why a column with no
fill value (a boolean column) must be supplied in every append.

## Column order

HDF5 groups do not preserve insertion order, so the table records its column
order in the `column-order` attribute. That name is not invented here: it is
borrowed, along with `_index`, `encoding-type`, and `encoding-version`, from
the way [AnnData](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)
encodes a data frame on disk. Reusing the spelling means a tool that already
understands AnnData files reads these four the same way.
{attr}`Table.column_names <h5col.Table.column_names>`
returns names in that order. A table may also designate one or more columns as
row-identifying via `index_columns=` at creation; the references land in the
`INDEX_COLUMNS` attribute, with the primary one named in `_index`.

## Table and column metadata

At the table level, `TITLE` and `description` carry human-readable context, and
`units_vocabulary` can name the convention that units strings follow. At the
column level, each dataset may carry `units`, `units_vocabulary`, `description`,
and the numeric bounds `valid_min` and `valid_max`. These are typed HDF5
attributes with meanings fixed by the convention which is what lets any
conforming reader interpret them without tool-specific configuration.

The fill value is not an attribute: it is the HDF5 dataset creation property,
read back through h5py as {attr}`h5py.Dataset.fillvalue` and exposed as
{attr}`Column.fill_value <h5col.Column.fill_value>`. The
[missing values](missing-values.md) chapter covers its semantics.

## The reserved side groups

Two reserved child groups support features that need storage of their own.
`CATEGORIES` holds one labels dataset per categorical column; the column's
integer codes point into it, and the column dataset carries a `CATEGORIES`
object reference to its labels (see [column datatypes](column-types.md)).
`SEARCH_INDEXES` holds the search-index datasets, each tagged with a `KIND`
attribute and validity tokens. A column lists its indexes in a
`SEARCH_INDEX_LIST` attribute of object references. A table that holds search
indexes also carries a `GENERATION` attribute: the counter each index's
validity tokens are checked against (see [search indexes](indexes.md)).

Both linkages are HDF5 object references, not name strings. Renaming or
moving things cannot silently break the association — the reference either
resolves or it does not.

## Reserved names

The convention reserves the uppercase names used above (`CLASS`, `NROWS`,
`CATEGORIES`, `SEARCH_INDEXES`, `OFFSETS`, `VALUES`, `MASK`, `CHARS`, and the
rest of the catalog) plus the lowercase `valid_min`/`valid_max`. A column may
not take any of these as its name — {meth}`~h5col.Table.create` raises
{class}`~h5col.ReservedNameError` if one tries. Names beginning with an
underscore are discouraged, since the convention uses `_index` and may reserve
similar names later.

## The h5py boundary

`h5col` deliberately does not open files. You create or open the file with h5py,
choosing whatever storage options the situation calls for — the core driver for
in-memory work, `ros3` or `fsspec`-backed access for cloud stores, page
buffering, or single-writer/multiple-reader (SWMR) access — and pass a
{class}`h5py.Group` to {meth}`Table.create <h5col.Table.create>` or
{meth}`Table.open <h5col.Table.open>`. Everything h5py can do with a file
remains available, because `h5col` only ever defines the contents of the HDF5
group you hand it.
