"""H5Col reserved names, tokens, and name-validation helpers.

H5Col writes its reserved attribute and group names in fixed-length uppercase
ASCII (with a few lowercase exceptions borrowed from broader community practice
or from AnnData). This module centralizes those tokens so the rest of H5Col
never hard-codes a spelling, and provides validators for producer-chosen names.
"""

from __future__ import annotations

from .exceptions import ReservedNameError, SchemaError

# --------------------------------------------------------------------------- #
# CLASS attribute values (reserved tokens)
# --------------------------------------------------------------------------- #
CLASS_COLUMN_TABLE = "COLUMN_TABLE"
CLASS_LIST_COLUMN = "LIST_COLUMN"
CLASS_STRING_VALUES = "STRING_VALUES"

# --------------------------------------------------------------------------- #
# KIND attribute values
# --------------------------------------------------------------------------- #
KIND_OFFSETS = "OFFSETS"  # list-column storage method
KIND_CHUNK_MINMAX = "CHUNK_MINMAX"
KIND_SORTED_ROWS = "SORTED_ROWS"
KIND_BITMAP = "BITMAP"
KIND_CHUNK_BLOOM = "CHUNK_BLOOM"

SEARCH_INDEX_KINDS = frozenset(
    {KIND_CHUNK_MINMAX, KIND_SORTED_ROWS, KIND_BITMAP, KIND_CHUNK_BLOOM}
)

# --------------------------------------------------------------------------- #
# Reserved group (link) names directly under a table group
# --------------------------------------------------------------------------- #
GROUP_CATEGORIES = "CATEGORIES"
GROUP_SEARCH_INDEXES = "SEARCH_INDEXES"
RESERVED_GROUP_NAMES = frozenset({GROUP_CATEGORIES, GROUP_SEARCH_INDEXES})

# --------------------------------------------------------------------------- #
# Reserved dataset (member) names inside list-column / STRING_VALUES groups
# --------------------------------------------------------------------------- #
MEMBER_OFFSETS = "OFFSETS"
MEMBER_VALUES = "VALUES"
MEMBER_MASK = "MASK"
MEMBER_CHARS = "CHARS"
RESERVED_MEMBER_NAMES = frozenset(
    {MEMBER_OFFSETS, MEMBER_VALUES, MEMBER_MASK, MEMBER_CHARS}
)

# --------------------------------------------------------------------------- #
# Reserved attribute names
# --------------------------------------------------------------------------- #
# Uppercase H5Col attributes.
ATTR_CLASS = "CLASS"
ATTR_VERSION = "VERSION"
ATTR_NROWS = "NROWS"
ATTR_TITLE = "TITLE"
ATTR_INDEX_COLUMNS = "INDEX_COLUMNS"
ATTR_GENERATION = "GENERATION"
ATTR_KIND = "KIND"
ATTR_SEARCH_INDEX_LIST = "SEARCH_INDEX_LIST"
ATTR_CATEGORIES = "CATEGORIES"
ATTR_VALUES = "VALUES"
ATTR_SOURCE_GENERATION = "SOURCE_GENERATION"
ATTR_SOURCE_NROWS = "SOURCE_NROWS"
ATTR_MASK = "MASK"  # reserved on column datasets for a future revision

# Lowercase-by-exception attributes that still carry contractual meaning.
ATTR_VALID_MIN = "valid_min"
ATTR_VALID_MAX = "valid_max"

# Names borrowed verbatim from AnnData (lowercase, exact spelling).
ATTR_COLUMN_ORDER = "column-order"
ATTR_INDEX = "_index"
ATTR_ENCODING_TYPE = "encoding-type"
ATTR_ENCODING_VERSION = "encoding-version"
ATTR_ORDERED = "ordered"

# Per-search-index-family attributes (lowercase snake_case per naming rule 5;
# they are documented with their family and are not reserved-catalog names).
ATTR_NAN_TAIL_LENGTH = "nan_tail_length"
ATTR_FILL_TAIL_LENGTH = "fill_tail_length"
ATTR_EXHAUSTIVE = "exhaustive"

# Descriptive, non-contractual annotations (lowercase).
ATTR_UNITS = "units"
ATTR_UNITS_VOCABULARY = "units_vocabulary"
ATTR_DESCRIPTION = "description"

#: Attribute names H5Col treats as reserved (producers must not repurpose them).
RESERVED_ATTRIBUTE_NAMES = frozenset(
    {
        ATTR_CLASS,
        ATTR_VERSION,
        ATTR_NROWS,
        ATTR_TITLE,
        ATTR_INDEX_COLUMNS,
        ATTR_GENERATION,
        ATTR_KIND,
        ATTR_SEARCH_INDEX_LIST,
        ATTR_CATEGORIES,
        ATTR_VALUES,
        ATTR_SOURCE_GENERATION,
        ATTR_SOURCE_NROWS,
        ATTR_MASK,
        ATTR_VALID_MIN,
        ATTR_VALID_MAX,
    }
)

# Names a producer-chosen column name must never collide with: the whole H5Col
# reserved-name catalog — reserved group names, list-column member names, and
# reserved attribute names (H5Col "Column datasets / Required properties" and
# reserved-names rule 2).
_FORBIDDEN_COLUMN_NAMES = (
    RESERVED_GROUP_NAMES | RESERVED_MEMBER_NAMES | RESERVED_ATTRIBUTE_NAMES
)


def is_valid_link_name(name: object) -> bool:
    """Return True if *name* is usable as an HDF5 link name (UTF-8, no ``/``/NUL)."""
    return (
        isinstance(name, str)
        and len(name) > 0
        and "/" not in name
        and "\x00" not in name
    )


def validate_column_name(name: str) -> str:
    """Validate a producer-chosen column name and return it unchanged.

    Raises
    ------
    SchemaError
        If *name* is not a valid HDF5 link name.
    ReservedNameError
        If *name* collides with a H5Col reserved group, member, or attribute
        name.
    """
    if not is_valid_link_name(name):
        raise SchemaError(f"{name!r} is not a valid HDF5 link name for a column")
    if name in _FORBIDDEN_COLUMN_NAMES:
        raise ReservedNameError(
            f"{name!r} is a H5Col reserved name and cannot be used as a column name"
        )
    return name


def validate_index_dataset_name(name: str) -> str:
    """Validate a search-index dataset name and return it unchanged.

    Datasets in ``SEARCH_INDEXES`` may have any name — the spec assigns names
    no meaning — but the name must still be a single HDF5 link name, and
    reserved-names rule 2 forbids reusing any reserved name for a search-index
    dataset.

    Raises
    ------
    SchemaError
        If *name* is not a valid single HDF5 link name.
    ReservedNameError
        If *name* is a H5Col reserved name.
    """
    if not is_valid_link_name(name):
        raise SchemaError(
            f"{name!r} is not a valid HDF5 link name for a search-index dataset"
        )
    if name in _FORBIDDEN_COLUMN_NAMES:
        raise ReservedNameError(
            f"{name!r} is a H5Col reserved name and cannot name a search-index dataset"
        )
    return name


def is_discouraged_column_name(name: str) -> bool:
    """Return True for names H5Col says producers SHOULD avoid (leading ``_``)."""
    return name.startswith("_")
