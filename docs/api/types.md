# Column datatypes

Helpers for the two datatypes HDF5 does not hand to NumPy directly: the
fixed-length UTF-8 string and the H5Col boolean enumeration. The
[column datatypes](../guide/column-types.md) chapter explains when to reach
for each.

## Fixed-length strings

```{eval-rst}
.. autoclass:: h5col.FixedString
   :members:

.. autofunction:: h5col.ascii_token_dtype
```

## Booleans

```{eval-rst}
.. autofunction:: h5col.bool_dtype

.. autofunction:: h5col.is_bool_dtype

.. autofunction:: h5col.encode_bool

.. autofunction:: h5col.decode_bool
```
