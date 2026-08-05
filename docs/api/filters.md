# Filters

A column's storage pipeline, mirroring HDF5's chunk filter pipeline. The
[filters and storage](../guide/filters.md) chapter covers usage and the
`hdf5plugin` ecosystem.

```{eval-rst}
.. autoclass:: h5col.FilterPipeline
   :members:

.. autoclass:: h5col.Filter
   :members:

.. autofunction:: h5col.Deflate

.. autofunction:: h5col.Shuffle

.. autofunction:: h5col.Fletcher32

.. autofunction:: h5col.from_hdf5plugin
```
