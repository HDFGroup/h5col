# Tables and columns

The three classes below are the reading-and-writing heart of the package.
{class}`~h5col.Table` wraps a table group; its
{meth}`~h5col.Table.create` / {meth}`~h5col.Table.open` classmethods are the
two entry points, and its mapping-style access (`table["name"]`, `in`,
iteration) hands out the column wrappers.

The subscript, length and iteration behaviour is listed alongside the ordinary
methods: `table["name"]` hands out a column, and `column[17:98]` reads rows.

```{eval-rst}
.. autoclass:: h5col.Table
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__

.. autoclass:: h5col.Column
   :members:
   :special-members: __getitem__, __len__, __iter__

.. autoclass:: h5col.ListColumn
   :members:
   :special-members: __getitem__, __len__, __iter__
```

## Arrow interchange

{meth}`Table.to_arrow <h5col.Table.to_arrow>` exports a table and
{meth}`Table.from_arrow <h5col.Table.from_arrow>` imports one, deciding each
column's spec for itself. The function below returns those specs without
writing anything, so they can be looked at and adjusted first. It is also the
only way to set chunking and filters, which have no Arrow equivalent to be
inferred from. The [guide chapter](../guide/from-arrow.md) covers the whole
picture.

```{eval-rst}
.. autofunction:: h5col.specs_from_arrow
```
