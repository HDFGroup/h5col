# Tables and columns

The three classes below are the reading-and-writing heart of the package.
{class}`~h5col.Table` wraps a table group; its
{meth}`~h5col.Table.create` / {meth}`~h5col.Table.open` classmethods are the
two entry points, and its mapping-style access (`table["name"]`, `in`,
iteration) hands out the column wrappers.

```{eval-rst}
.. autoclass:: h5col.Table
   :members:

.. autoclass:: h5col.Column
   :members:

.. autoclass:: h5col.ListColumn
   :members:
```
