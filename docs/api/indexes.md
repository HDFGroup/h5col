# Search indexes

Read-side wrappers over the index datasets stored under a table's
`SEARCH_INDEXES` group. They are returned by
{meth}`Table.build_index <h5col.Table.build_index>`,
{attr}`Table.search_indexes <h5col.Table.search_indexes>`, and
{attr}`Column.search_indexes <h5col.Column.search_indexes>`; the
[search indexes](../guide/indexes.md) chapter explains the three families
and the validity protocol.

```{eval-rst}
.. autoclass:: h5col.SearchIndex
   :members:

.. autoclass:: h5col.ChunkMinMaxIndex
   :members:
   :show-inheritance:

.. autoclass:: h5col.SortedRowsIndex
   :members:
   :show-inheritance:

.. autoclass:: h5col.BitmapIndex
   :members:
   :show-inheritance:
```
