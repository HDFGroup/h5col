# Exceptions

Every error the library raises deliberately derives from
{class}`~h5col.H5ColError`, so `except H5ColError` catches all of `h5col`'s
own diagnostics while letting genuine bugs surface. Several leaves also
derive from {class}`ValueError`, matching how standard Python code expects
bad values to fail.

```{eval-rst}
.. autoexception:: h5col.H5ColError
   :show-inheritance:

.. autoexception:: h5col.ConformanceError
   :show-inheritance:

.. autoexception:: h5col.SchemaError
   :show-inheritance:

.. autoexception:: h5col.ReservedNameError
   :show-inheritance:

.. autoexception:: h5col.OversizedStringError
   :show-inheritance:

.. autoexception:: h5col.FillValueError
   :show-inheritance:

.. autoexception:: h5col.FilterError
   :show-inheritance:

.. autoexception:: h5col.ObjectReferenceError
   :show-inheritance:

.. autoexception:: h5col.StaleIndexError
   :show-inheritance:

.. autoexception:: h5col.VersionError
   :show-inheritance:
```
