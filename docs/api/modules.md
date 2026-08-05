# Low-level modules

Six submodules are exported as public, stable entry points below the
class-based API. They operate directly on h5py objects — most take a table
group or a dataset rather than a {class}`~h5col.Table` — and exist for
tools that need the convention's mechanics without the wrappers: validators,
migration scripts, or readers in constrained environments. Most users never
need them.

## h5col.missing

```{eval-rst}
.. automodule:: h5col.missing
   :members:
   :no-index:
```

## h5col.ordering

```{eval-rst}
.. automodule:: h5col.ordering
   :members:
```

## h5col.references

```{eval-rst}
.. automodule:: h5col.references
   :members:
```

## h5col.reserved

```{eval-rst}
.. automodule:: h5col.reserved
   :members:
```

## h5col.lists

```{eval-rst}
.. automodule:: h5col.lists
   :members:
```

## h5col.indexes

```{eval-rst}
.. automodule:: h5col.indexes
   :members:
```
