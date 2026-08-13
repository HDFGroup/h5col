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
