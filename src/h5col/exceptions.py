"""The ``h5col`` exception hierarchy.

Every error raised by ``h5col`` derives from :class:`H5ColError`. Where a failure
also fits a built-in category, the specific exception additionally inherits from
that built-in, so existing ``except ValueError`` handlers keep working.
"""

from __future__ import annotations

__all__ = [
    "H5ColError",
    "ConformanceError",
    "SchemaError",
    "ReservedNameError",
    "OversizedStringError",
    "FillValueError",
    "FilterError",
    "ObjectReferenceError",
    "StaleIndexError",
    "VersionError",
]


class H5ColError(Exception):
    """Base class for all ``h5col`` errors."""


class ConformanceError(H5ColError):
    """A group, dataset, or attribute violates H5Col when read."""


class SchemaError(H5ColError, ValueError):
    """An invalid table, column, or filter specification supplied by the caller."""


class ReservedNameError(SchemaError):
    """A H5Col reserved name was used where it is not permitted."""


class OversizedStringError(H5ColError, ValueError):
    """A string value's encoded byte length exceeds the fixed-length budget.

    H5Col forbids silent truncation of fixed-length string columns; H5Col
    raises this instead of writing a truncated value.
    """

    def __init__(
        self,
        value: object,
        max_bytes: int,
        actual_bytes: int,
        *,
        index: int | None = None,
    ) -> None:
        self.value = value
        self.max_bytes = max_bytes
        self.actual_bytes = actual_bytes
        self.index = index
        where = "" if index is None else f" at index {index}"
        super().__init__(
            f"string value{where} needs {actual_bytes} bytes but the column "
            f"allows at most {max_bytes}: {value!r}"
        )


class FillValueError(H5ColError, ValueError):
    """An invalid or inconsistent fill value / valid-range specification."""


class FilterError(H5ColError):
    """A problem building or applying a filter pipeline."""


class ObjectReferenceError(H5ColError):
    """A problem creating or resolving an HDF5 object reference.

    Named ``ObjectReferenceError`` to avoid shadowing the built-in
    :class:`ReferenceError`.
    """


class StaleIndexError(H5ColError):
    """A search index failed the H5Col validity check and cannot be used.

    Raised by index query primitives when the ``GENERATION``/``SOURCE_*`` token
    comparison fails; consumers must treat such an index as absent (fall back
    to a scan) rather than partially trust it.
    """


class VersionError(H5ColError):
    """The table's HEP001 ``VERSION`` major exceeds what this library supports."""
