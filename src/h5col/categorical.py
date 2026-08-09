"""Categorical column support: code datasets backed by a categories dataset.

A categorical column stores integer *codes*; each code is the zero-based position
of that row's label in a separate rank-1 *categories dataset* under the table
group's ``CATEGORIES`` subgroup. The column carries a scalar ``CATEGORIES`` object
reference to that dataset.

A missing category is the column's fill value, and nothing else. H5Col defines
the fill value as the sole missing-category marker and requires consumers to
confirm the ``H5D_FILL_VALUE_USER_DEFINED`` state before using it, so a code that
is neither the declared fill nor a valid ``[0, ncats)`` index is a malformed file
rather than another spelling of "missing".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import references
from .exceptions import ConformanceError, SchemaError
from .missing import masked_to_none
from .reserved import ATTR_CATEGORIES, ATTR_ORDERED
from .strings import FixedString

#: Default fill code for a *signed* categorical column.
DEFAULT_CATEGORICAL_FILL = -1


def default_categorical_fill(dtype: Any) -> int:
    """Default missing-code fill for a categorical column of *dtype*.

    Signed code dtypes use ``-1``; unsigned code dtypes use the type maximum
    (H5Col's recommended unsigned sentinel). Both lie outside ``[0, ncats)`` as
    long as the code type has room for a sentinel.
    """
    dt = np.dtype(dtype)
    if dt.kind == "u":
        return int(np.iinfo(dt).max)
    return DEFAULT_CATEGORICAL_FILL


def choose_code_dtype(ncats: int) -> np.dtype:
    """Pick the smallest signed integer code dtype that fits ``ncats`` labels.

    Signed so the default ``-1`` fill code is always representable and outside
    ``[0, ncats)``.
    """
    if ncats <= 127:
        return np.dtype("i1")
    if ncats <= 32767:
        return np.dtype("i2")
    if ncats <= 2_147_483_647:
        return np.dtype("i4")
    return np.dtype("i8")


def create_categories_dataset(
    cat_group: Any, name: str, categories: list[Any], ordered: bool | None = None
) -> Any:
    """Create a rank-1 categories dataset holding the label values."""
    cats = list(categories)
    if len(set(cats)) != len(cats):
        raise SchemaError("categories must be unique")
    if all(isinstance(c, str) for c in cats):
        max_bytes = max((len(c.encode("utf-8")) for c in cats), default=1)
        fs = FixedString(max(1, max_bytes))
        ds = cat_group.create_dataset(name, data=fs.encode(cats))
    else:
        ds = cat_group.create_dataset(name, data=np.asarray(cats))
    if ordered is not None:
        ds.attrs.create(ATTR_ORDERED, np.bool_(ordered))
    return ds


def load_category_labels(table_group: Any, code_dataset: Any) -> list[Any]:
    """Return the decoded category labels for a categorical column."""
    cat_ds = references.resolve(table_group, code_dataset.attrs[ATTR_CATEGORIES])
    raw = cat_ds[...]
    if FixedString.is_fixed_string(cat_ds.dtype):
        return list(FixedString.from_dtype(cat_ds.dtype).decode(raw))
    return list(raw)


def n_categories(table_group: Any, code_dataset: Any) -> int:
    """Return the number of categories, reading only the labels dataset extent.

    Cheaper than :func:`load_category_labels` (no reads/decoding of the label
    values) — used where only the count is needed, e.g. a ``repr``.
    """
    cat_ds = references.resolve(table_group, code_dataset.attrs[ATTR_CATEGORIES])
    return int(cat_ds.shape[0])


def user_fill_code(code_dataset: Any) -> int | None:
    """The column's declared fill code, or None when it declares no fill value.

    H5Col requires consumers to confirm the ``H5D_FILL_VALUE_USER_DEFINED`` state
    before treating a fill value as a missing-row marker. Skipping that check
    would read h5py's library default of ``0`` as a fill code, decoding the
    *first* category as missing on any column that declares no fill.
    """
    # H5D_FILL_VALUE_USER_DEFINED == 2 (matches Column._has_user_fill).
    if code_dataset.id.get_create_plist().fill_value_defined() != 2:
        return None
    return int(code_dataset.fillvalue)


def encode_labels(table_group: Any, code_dataset: Any, values: Any) -> npt.NDArray[Any]:
    """Map an array-like of labels to integer codes (``None`` -> the fill code).

    Raises
    ------
    SchemaError
        If a value is not one of the column's categories, or if a ``None`` is
        given for a column that declares no fill value and therefore has no way
        to represent a missing category.
    """
    labels = load_category_labels(table_group, code_dataset)
    index = {lab: i for i, lab in enumerate(labels)}
    fill = user_fill_code(code_dataset)
    # A masked element means the same as None here; normalize so one path
    # handles both (a MaskedConstant is unhashable and would blow up the
    # category lookup below).
    vals = list(masked_to_none(values))
    codes = np.empty(len(vals), dtype=code_dataset.dtype)
    for i, v in enumerate(vals):
        if v is None:
            if fill is None:
                raise SchemaError(
                    f"{code_dataset.name!r}: categorical column declares no fill "
                    "value, so a missing category cannot be represented"
                )
            codes[i] = fill
        elif v in index:
            codes[i] = index[v]
        else:
            raise SchemaError(f"unknown category {v!r}")
    return codes


def decode_codes(
    table_group: Any, code_dataset: Any, codes: Any
) -> npt.NDArray[np.object_]:
    """Map integer codes to labels; the declared fill code becomes ``None``.

    Raises
    ------
    ConformanceError
        If a code is neither the declared fill code nor a valid ``[0, ncats)``
        index. Decoding such a code to ``None`` would invent a missing value
        that the canonical missing-value test — and therefore
        :meth:`~h5col.Column.is_missing` and the query layer — does not see.
    """
    labels = load_category_labels(table_group, code_dataset)
    fill = user_fill_code(code_dataset)
    arr = np.asarray(codes)
    n = arr.shape[0]

    is_fill = np.zeros(n, dtype=np.bool_) if fill is None else np.asarray(arr == fill)
    in_range = (arr >= 0) & (arr < len(labels))
    bad = ~(is_fill | in_range)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ConformanceError(
            f"{code_dataset.name!r}: categorical code {int(arr[i])} (element {i} of "
            f"the block read) is neither the fill code nor a valid category index "
            f"[0, {len(labels)})"
        )

    out = np.full(n, None, dtype=object)
    present = ~is_fill
    out[present] = np.asarray(labels, dtype=object)[arr[present]]
    return out


def is_ordered(table_group: Any, code_dataset: Any) -> bool | None:
    """Return the categories dataset's ``ordered`` flag, or None if absent."""
    cat_ds = references.resolve(table_group, code_dataset.attrs[ATTR_CATEGORIES])
    if ATTR_ORDERED not in cat_ds.attrs:
        return None
    return bool(cat_ds.attrs[ATTR_ORDERED])
