"""A reference implementation of the H5Col convention for columnar tables in HDF5.

The H5Col convention stores a tabular dataset as an HDF5 group whose direct
children are rank-1 column datasets, identified by a ``CLASS="COLUMN_TABLE"``
attribute. The convention (HEP001) is specified at
https://hdfalliance.github.io/heps/hep001/.
"""

from __future__ import annotations

# Imported for its side effect: hdf5plugin registers its filter plugins
# (Zstandard, Blosc2, and the rest) with the HDF5 library, so a column
# compressed with one of them reads back without the caller importing
# hdf5plugin first. It is a required dependency, not an optional extra.
import hdf5plugin as _hdf5plugin  # noqa: F401

from . import indexes, lists, missing, opaque, ordering, references, reserved
from .arrow import specs_from_arrow
from .booleans import bool_dtype, decode_bool, encode_bool, is_bool_dtype
from .column import Column
from .exceptions import (
    ConformanceError,
    FillValueError,
    FilterError,
    H5ColError,
    ObjectReferenceError,
    OversizedStringError,
    ReservedNameError,
    SchemaError,
    StaleIndexError,
    VersionError,
)
from .filters import (
    Deflate,
    Filter,
    FilterPipeline,
    Fletcher32,
    Shuffle,
    from_hdf5plugin,
)
from .listcolumn import ListColumn
from .missing import is_missing, recommended_fill, validate_fill_outside_range
from .opaque import is_opaque_dtype, opaque_fill_bytes
from .query import Expression, QueryPlan, Selection, field
from .searchindex import (
    BitmapIndex,
    ChunkMinMaxIndex,
    SearchIndex,
    SortedRowsIndex,
)
from .specs import (
    ColumnSpec,
    LeafValuesSpec,
    ListColumnSpec,
    NestedListSpec,
    StringValuesSpec,
    TableSpec,
)
from .strings import FixedString, ascii_token_dtype
from .table import Table

# The only place the version is written. pyproject.toml reads it from here
# (dynamic = ["version"]), and docs/conf.py imports it. A trailing `.devN`
# means "working toward that release, not there yet"; drop it to release.
__version__ = "0.4.0"

__all__ = [
    "__version__",
    # submodules
    "indexes",
    "lists",
    "missing",
    "opaque",
    "ordering",
    "references",
    "reserved",
    # exceptions
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
    # strings
    "FixedString",
    "ascii_token_dtype",
    # booleans
    "bool_dtype",
    "is_bool_dtype",
    "encode_bool",
    "decode_bool",
    # opaque
    "is_opaque_dtype",
    "opaque_fill_bytes",
    # arrow interchange
    "specs_from_arrow",
    # filters
    "Filter",
    "FilterPipeline",
    "Deflate",
    "Shuffle",
    "Fletcher32",
    "from_hdf5plugin",
    # missing values
    "recommended_fill",
    "is_missing",
    "validate_fill_outside_range",
    # table / column / specs
    "Table",
    "Column",
    "ListColumn",
    "TableSpec",
    "ColumnSpec",
    "LeafValuesSpec",
    "StringValuesSpec",
    "NestedListSpec",
    "ListColumnSpec",
    # search indexes
    "SearchIndex",
    "ChunkMinMaxIndex",
    "SortedRowsIndex",
    "BitmapIndex",
    # query layer
    "field",
    "Expression",
    "Selection",
    "QueryPlan",
]
