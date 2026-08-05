# Specifications

Specs are the write-side schema: validated (pydantic) descriptions of a
table and its columns, consumed by {meth}`Table.create <h5col.Table.create>`
and {meth}`Table.add_column <h5col.Table.add_column>`. Their fields are
introduced, with examples, in the
[column datatypes](../guide/column-types.md) and
[list columns](../guide/list-columns.md) chapters.

```{eval-rst}
.. autoclass:: h5col.TableSpec
   :members:

.. autoclass:: h5col.ColumnSpec
   :members:

.. autoclass:: h5col.ListColumnSpec
   :members:

.. autoclass:: h5col.LeafValuesSpec
   :members:

.. autoclass:: h5col.StringValuesSpec
   :members:

.. autoclass:: h5col.NestedListSpec
   :members:
```
