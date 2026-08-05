# API reference

Everything documented here is importable from the top-level `h5col` package,
which is the intended way to use the library. The pages group the public API
by topic; each entry's full documentation, including the exceptions it can
raise, is generated from the docstrings in the source.

```{toctree}
:maxdepth: 1

table
specs
types
filters
missing
indexes
query
exceptions
modules
```

## At a glance

Tables and columns:

```{eval-rst}
.. autosummary::

   h5col.Table
   h5col.Column
   h5col.ListColumn
```

Write-side specifications:

```{eval-rst}
.. autosummary::

   h5col.TableSpec
   h5col.ColumnSpec
   h5col.ListColumnSpec
   h5col.LeafValuesSpec
   h5col.StringValuesSpec
   h5col.NestedListSpec
```

Column datatypes:

```{eval-rst}
.. autosummary::

   h5col.FixedString
   h5col.ascii_token_dtype
   h5col.bool_dtype
   h5col.is_bool_dtype
   h5col.encode_bool
   h5col.decode_bool
```

Filters:

```{eval-rst}
.. autosummary::

   h5col.FilterPipeline
   h5col.Filter
   h5col.Deflate
   h5col.Shuffle
   h5col.Fletcher32
   h5col.from_hdf5plugin
```

Missing values:

```{eval-rst}
.. autosummary::

   h5col.recommended_fill
   h5col.is_missing
   h5col.validate_fill_outside_range
```

Search indexes:

```{eval-rst}
.. autosummary::

   h5col.SearchIndex
   h5col.ChunkMinMaxIndex
   h5col.SortedRowsIndex
   h5col.BitmapIndex
```

The query layer:

```{eval-rst}
.. autosummary::

   h5col.field
   h5col.Expression
   h5col.Selection
   h5col.QueryPlan
```

The exception family is on its [own page](exceptions.md), and the low-level
submodules (`h5col.indexes`, `h5col.lists`, `h5col.missing`,
`h5col.ordering`, `h5col.references`, `h5col.reserved`) on
[theirs](modules.md).
