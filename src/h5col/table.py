"""The Table class: create, write, and read a H5Col column-oriented table.

A table is an HDF5 group carrying ``CLASS="COLUMN_TABLE"``. This module builds
such groups from :class:`~h5col.specs.TableSpec` / :class:`~h5col.specs.ColumnSpec`,
appends rows following the H5Col write protocol (extend every column, write,
then commit ``NROWS`` last), reads them back, and validates conformance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import h5py
import numpy as np

from . import categorical, indexes, lists, query, references
from ._hdf5 import (
    create_column_dataset,
    extend_to,
    prepare_column_data,
    read_str_array_attr,
    read_str_attr,
    read_uint64_attr,
    substitute_fill_for_none,
    write_ascii_token_attr,
    write_uint64_attr,
    write_utf8_array_attr,
    write_utf8_attr,
)
from .booleans import bool_dtype, is_bool_dtype
from .column import Column
from .exceptions import ConformanceError, SchemaError, VersionError
from .listcolumn import ListColumn
from .missing import recommended_fill, validate_fill_outside_range
from .reserved import (
    ATTR_CATEGORIES,
    ATTR_CLASS,
    ATTR_COLUMN_ORDER,
    ATTR_DESCRIPTION,
    ATTR_ENCODING_TYPE,
    ATTR_ENCODING_VERSION,
    ATTR_GENERATION,
    ATTR_INDEX,
    ATTR_INDEX_COLUMNS,
    ATTR_NROWS,
    ATTR_TITLE,
    ATTR_UNITS,
    ATTR_UNITS_VOCABULARY,
    ATTR_VALID_MAX,
    ATTR_VALID_MIN,
    ATTR_VERSION,
    CLASS_COLUMN_TABLE,
    CLASS_LIST_COLUMN,
    GROUP_CATEGORIES,
    GROUP_SEARCH_INDEXES,
    KIND_BITMAP,
    KIND_CHUNK_MINMAX,
    KIND_SORTED_ROWS,
    MEMBER_MASK,
    validate_column_name,
)
from .searchindex import SearchIndex, wrap_index
from .specs import ColumnSpec, ListColumnSpec, TableSpec
from .strings import FixedString

#: HEP001 revision this implementation writes and the highest major it reads.
VERSION = "1.0"
SUPPORTED_MAJOR = 1

_SKIP_CHILDREN = frozenset({GROUP_CATEGORIES, GROUP_SEARCH_INDEXES})


class Table:
    """A H5Col column-oriented table backed by an HDF5 group."""

    def __init__(self, group: Any) -> None:
        self._group = group

    def __repr__(self) -> str:
        try:
            return f"<h5col.Table {self._group.name!r} nrows={self.nrows}>"
        except Exception:
            return "<h5col.Table (closed or invalid)>"

    # -- construction ------------------------------------------------------- #
    @staticmethod
    def is_table_group(group: Any) -> bool:
        """True if *group* is a H5Col table group (lenient CLASS check)."""
        return read_str_attr(group, ATTR_CLASS) == CLASS_COLUMN_TABLE

    @classmethod
    def open(cls, group: Any) -> Table:
        """Open an existing table group, checking its CLASS and VERSION major.

        Raises
        ------
        ConformanceError
            If *group* is not a H5Col table group, or its ``VERSION`` is missing
            or unparsable.
        VersionError
            If the table's ``VERSION`` major exceeds the supported major.
        """
        if not cls.is_table_group(group):
            raise ConformanceError(
                f"{group.name!r} is not a H5Col table group "
                f"(missing or wrong {ATTR_CLASS} attribute)"
            )
        table = cls(group)
        table._check_version()
        return table

    @classmethod
    def create(
        cls,
        group: Any,
        columns: TableSpec | Sequence[ColumnSpec | ListColumnSpec],
        *,
        title: str | None = None,
        description: str | None = None,
        index_columns: Sequence[str] | None = None,
        column_order: Sequence[str] | None = None,
        units_vocabulary: str | None = None,
        encoding_type: str | None = None,
        encoding_version: str | None = None,
        default_chunk_bytes: int | None = None,
    ) -> Table:
        """Create a new, empty table (``NROWS = 0``) with the given columns.

        ``default_chunk_bytes`` overrides the automatic (chunk-cache-scaled)
        target for columns that do not set an explicit ``chunks`` shape.

        Raises
        ------
        SchemaError
            If *group* is already a H5Col table group, a column spec is invalid,
            or an ``index_columns`` name is not among the declared columns.
        ReservedNameError
            If a column name is a H5Col reserved name.
        FillValueError
            If a column's fill value lies inside its declared valid range.
        """
        if cls.is_table_group(group):
            raise SchemaError(f"{group.name!r} is already a H5Col table group")

        if isinstance(columns, TableSpec):
            spec = columns
        else:
            spec = TableSpec(
                columns=list(columns),
                title=title,
                description=description,
                index_columns=list(index_columns) if index_columns else [],
                column_order=list(column_order) if column_order else None,
                units_vocabulary=units_vocabulary,
                encoding_type=encoding_type,
                encoding_version=encoding_version,
            )

        # Pre-flight: validate every column BEFORE writing the CLASS identifier,
        # so an invalid spec never leaves a group marked as a (broken) table
        # group (H5Col forbids CLASS="COLUMN_TABLE" on a non-conformant group).
        for col in spec.columns:
            cls._validate_column_spec(col)
        declared = {c.name for c in spec.columns}
        for ic in spec.index_columns:
            if ic not in declared:
                raise SchemaError(f"index column {ic!r} is not a declared column")

        # Mutate the group, rolling everything back on failure so a partially
        # built table is never left behind.
        written: list[str] = []
        created: list[str] = []
        cat_existed = GROUP_CATEGORIES in group
        try:
            write_ascii_token_attr(group, ATTR_CLASS, CLASS_COLUMN_TABLE)
            written.append(ATTR_CLASS)
            write_ascii_token_attr(group, ATTR_VERSION, VERSION)
            written.append(ATTR_VERSION)
            write_uint64_attr(group, ATTR_NROWS, 0)
            written.append(ATTR_NROWS)
            for attr, value in (
                (ATTR_TITLE, spec.title),
                (ATTR_DESCRIPTION, spec.description),
                (ATTR_UNITS_VOCABULARY, spec.units_vocabulary),
                (ATTR_ENCODING_TYPE, spec.encoding_type),
                (ATTR_ENCODING_VERSION, spec.encoding_version),
            ):
                if value is not None:
                    write_utf8_attr(group, attr, value)
                    written.append(attr)

            for col in spec.columns:
                cls._create_one_column(
                    group, col, default_chunk_bytes=default_chunk_bytes
                )
                created.append(col.name)

            if spec.columns:
                write_utf8_array_attr(group, ATTR_COLUMN_ORDER, spec.ordered_names)
                written.append(ATTR_COLUMN_ORDER)

            if spec.index_columns:
                references.write_ref_array_attr(
                    group, ATTR_INDEX_COLUMNS, [group[n] for n in spec.index_columns]
                )
                written.append(ATTR_INDEX_COLUMNS)
                write_utf8_attr(group, ATTR_INDEX, spec.index_columns[0])
                written.append(ATTR_INDEX)
        except BaseException:
            for n in created:
                if n in group:
                    del group[n]
            for a in written:
                if a in group.attrs:
                    del group.attrs[a]
            if not cat_existed and GROUP_CATEGORIES in group:
                del group[GROUP_CATEGORIES]
            raise

        return cls(group)

    @staticmethod
    def _validate_column_spec(col: ColumnSpec | ListColumnSpec) -> None:
        """Validate one column spec without touching the file."""
        if isinstance(col, ListColumnSpec):
            lists.validate_list_column_spec(col)
            return
        validate_column_name(col.name)
        if col.is_categorical:
            assert col.categories is not None
            ncats = len(col.categories)
            if len(set(col.categories)) != len(col.categories):
                raise SchemaError(
                    f"categorical column {col.name!r} has duplicate categories"
                )
            dtype = col.resolved_dtype()
            if ncats > 0 and int(np.iinfo(dtype).max) < ncats - 1:
                raise SchemaError(
                    f"categorical column {col.name!r} code dtype {dtype} cannot "
                    f"index {ncats} categories"
                )
            fill = (
                col.fill_value
                if col.fill_value is not None
                else categorical.default_categorical_fill(dtype)
            )
            try:
                cast_fill = int(np.asarray(fill, dtype=dtype))
            except (OverflowError, ValueError) as exc:
                raise SchemaError(
                    f"categorical column {col.name!r} fill {fill!r} is not "
                    f"representable in code dtype {dtype}"
                ) from exc
            if 0 <= cast_fill < ncats:
                raise SchemaError(
                    f"categorical column {col.name!r} fill {cast_fill} collides "
                    f"with a valid code [0, {ncats})"
                )
            validate_fill_outside_range(cast_fill, col.valid_min, col.valid_max)
            return
        if col.is_boolean:
            if col.fill_value is not None:
                raise SchemaError(
                    f"boolean column {col.name!r} must not declare a fill value"
                )
            if col.valid_min is not None or col.valid_max is not None:
                raise SchemaError(
                    f"boolean column {col.name!r} must not declare valid_min/valid_max"
                )
        else:
            fill = (
                col.fill_value
                if col.fill_value is not None
                else recommended_fill(col.resolved_dtype())
            )
            validate_fill_outside_range(fill, col.valid_min, col.valid_max)

    @staticmethod
    def _create_one_column(
        group: Any,
        col: ColumnSpec | ListColumnSpec,
        default_chunk_bytes: int | None = None,
    ) -> Any:
        if isinstance(col, ListColumnSpec):
            return lists.create_list_column(
                group, col, default_chunk_bytes=default_chunk_bytes
            )

        name = validate_column_name(col.name)
        dtype = col.resolved_dtype()

        if col.is_categorical:
            return Table._create_categorical_column(
                group, col, name, dtype, default_chunk_bytes=default_chunk_bytes
            )

        if col.is_boolean:
            if col.fill_value is not None:
                raise SchemaError(
                    f"boolean column {name!r} must not declare a fill value"
                )
            if col.valid_min is not None or col.valid_max is not None:
                raise SchemaError(
                    f"boolean column {name!r} must not declare valid_min/valid_max"
                )
            fill: Any = None
        else:
            fill = (
                col.fill_value
                if col.fill_value is not None
                else recommended_fill(dtype)
            )
            validate_fill_outside_range(fill, col.valid_min, col.valid_max)

        ds = create_column_dataset(
            group,
            name,
            dtype,
            chunks=col.chunks,
            fill_value=fill,
            filters=col.filters,
            default_chunk_bytes=default_chunk_bytes,
        )

        if col.valid_min is not None:
            ds.attrs.create(ATTR_VALID_MIN, np.asarray(col.valid_min, dtype=dtype))
        if col.valid_max is not None:
            ds.attrs.create(ATTR_VALID_MAX, np.asarray(col.valid_max, dtype=dtype))
        if col.units is not None:
            write_utf8_attr(ds, ATTR_UNITS, col.units)
        if col.units_vocabulary is not None:
            write_utf8_attr(ds, ATTR_UNITS_VOCABULARY, col.units_vocabulary)
        if col.description is not None:
            write_utf8_attr(ds, ATTR_DESCRIPTION, col.description)
        return ds

    @staticmethod
    def _create_categorical_column(
        group: Any,
        col: ColumnSpec,
        name: str,
        dtype: np.dtype,
        default_chunk_bytes: int | None = None,
    ) -> Any:
        assert col.categories is not None
        # Fill range, dtype fit, and representability are enforced by
        # _validate_column_spec, which always runs before creation.
        fill = (
            col.fill_value
            if col.fill_value is not None
            else categorical.default_categorical_fill(dtype)
        )
        ds = create_column_dataset(
            group,
            name,
            dtype,
            chunks=col.chunks,
            fill_value=np.asarray(fill, dtype=dtype),
            filters=col.filters,
            default_chunk_bytes=default_chunk_bytes,
        )

        cat_group = group.require_group(GROUP_CATEGORIES)
        cat_ds = categorical.create_categories_dataset(
            cat_group, f"{name}__CATEGORIES", col.categories, col.ordered
        )
        references.write_ref_attr(ds, ATTR_CATEGORIES, cat_ds)

        if col.valid_min is not None:
            ds.attrs.create(ATTR_VALID_MIN, np.asarray(col.valid_min, dtype=dtype))
        if col.valid_max is not None:
            ds.attrs.create(ATTR_VALID_MAX, np.asarray(col.valid_max, dtype=dtype))
        if col.units is not None:
            write_utf8_attr(ds, ATTR_UNITS, col.units)
        if col.units_vocabulary is not None:
            write_utf8_attr(ds, ATTR_UNITS_VOCABULARY, col.units_vocabulary)
        if col.description is not None:
            write_utf8_attr(ds, ATTR_DESCRIPTION, col.description)
        return ds

    @classmethod
    def from_arrays(
        cls,
        group: Any,
        arrays: Mapping[str, Any],
        *,
        specs: Sequence[ColumnSpec] | None = None,
        **table_kwargs: Any,
    ) -> Table:
        """Create a table from column arrays and write them in one call.

        When *specs* is omitted, a :class:`ColumnSpec` is inferred per array
        (boolean, fixed-string sized to the longest value, or the array dtype).
        The column order follows ``arrays``.
        """
        if specs is None:
            specs = [_infer_column_spec(name, arr) for name, arr in arrays.items()]
        table = cls.create(group, specs, **table_kwargs)
        table.append(arrays)
        return table

    # -- introspection ------------------------------------------------------ #
    @property
    def group(self) -> Any:
        """The underlying h5py Group backing the table."""
        return self._group

    @property
    def nrows(self) -> int:
        """The table's logical row count (its ``NROWS`` attribute).

        Raises
        ------
        ConformanceError
            If the group carries no ``NROWS`` attribute.
        """
        n = read_uint64_attr(self._group, ATTR_NROWS)
        if n is None:
            raise ConformanceError(f"table {self._group.name!r} has no NROWS attribute")
        return n

    @property
    def version(self) -> str | None:
        """The table's H5Col ``VERSION`` string, or None when absent."""
        return read_str_attr(self._group, ATTR_VERSION)

    @property
    def title(self) -> str | None:
        """The table's ``title`` attribute, or None when unset."""
        return read_str_attr(self._group, ATTR_TITLE)

    @property
    def description(self) -> str | None:
        """The table's ``description`` attribute, or None when unset."""
        return read_str_attr(self._group, ATTR_DESCRIPTION)

    @property
    def generation(self) -> int | None:
        """The table's ``GENERATION`` validity token, or None when absent.

        A table acquires ``GENERATION`` when its first search index is built
        and increments it on every subsequent mutation of committed data.
        """
        return indexes.table_generation(self._group)

    def _discover_columns(self) -> dict[str, Any]:
        """All columns by name: rank-1 datasets and ``CLASS=LIST_COLUMN`` groups."""
        found: dict[str, Any] = {}
        for name, obj in self._group.items():
            if name in _SKIP_CHILDREN:
                continue
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
                found[name] = obj
            elif isinstance(obj, h5py.Group):
                if read_str_attr(obj, ATTR_CLASS) == CLASS_LIST_COLUMN:
                    found[name] = obj
        return found

    def _scalar_column_datasets(self) -> dict[str, Any]:
        """Only the column *datasets* (excludes list column groups)."""
        return {
            n: o
            for n, o in self._discover_columns().items()
            if isinstance(o, h5py.Dataset)
        }

    def _wrap(self, obj: Any) -> Column | ListColumn:
        if isinstance(obj, h5py.Dataset):
            return Column(obj, self)
        return ListColumn(obj, self)

    @property
    def column_names(self) -> list[str]:
        """Column names in logical order (``column-order`` if present)."""
        order = read_str_array_attr(self._group, ATTR_COLUMN_ORDER)
        columns = self._discover_columns()
        if order is not None:
            return [n for n in order if n in columns]
        return sorted(columns)

    @property
    def index_columns(self) -> list[str]:
        """Names of the row-index columns, outermost first."""
        if ATTR_INDEX_COLUMNS not in self._group.attrs:
            return []
        refs = self._group.attrs[ATTR_INDEX_COLUMNS]
        names = []
        for r in refs:
            if references.is_null_ref(r):
                continue
            names.append(references.resolve(self._group, r).name.rsplit("/", 1)[-1])
        return names

    @property
    def columns(self) -> dict[str, Column | ListColumn]:
        """The table's columns by name, in column order, as wrapper objects."""
        cols = self._discover_columns()
        return {n: self._wrap(cols[n]) for n in self.column_names}

    def __getitem__(self, name: str) -> Column | ListColumn:
        cols = self._discover_columns()
        if name not in cols:
            raise KeyError(name)
        return self._wrap(cols[name])

    def __contains__(self, name: str) -> bool:
        return name in self._discover_columns()

    def __iter__(self) -> Iterator[str]:
        return iter(self.column_names)

    def __len__(self) -> int:
        return len(self._discover_columns())

    # -- writing ------------------------------------------------------------ #
    def append(
        self, data: Mapping[str, Any], *, maintain_indexes: bool = False
    ) -> None:
        """Append rows following the H5Col write protocol.

        Every provided column must supply the same number of rows ``K``. A scalar
        column absent from *data* is extended and left as its fill value
        (missing); a boolean column (which has no fill) must always be provided.
        A list column absent from *data* must be nullable — its new rows become
        null lists; a non-nullable list column must always be provided. ``NROWS``
        is committed last, then the file is flushed.

        ``None`` in a column's values marks that row as missing and is stored as
        the column's fill value (for a categorical column, its fill code). A
        column with no fill to store — a boolean, which H5Col forbids from
        declaring one — rejects ``None`` instead of coercing it.

        By default, search indexes are **not** maintained: the ``GENERATION``
        increment that publishes the append disables them, detectably, and
        :meth:`refresh_indexes` restores them later — this keeps the hot append
        path fast. With ``maintain_indexes=True``, every supported index is
        rewritten inside the append protocol (future-valued tokens before
        content) and remains valid after the commit; indexes this
        implementation cannot rebuild — unsupported kinds, element dtypes the
        builder does not handle, non-growable index datasets — are left
        entirely untouched, tokens included.

        Raises
        ------
        OversizedStringError
            If a fixed-length string value's encoding exceeds the column's byte
            budget (H5Col never silently truncates).
        SchemaError
            For unknown columns, values that are not 1-D, unequal column
            lengths, an omitted fill-less/boolean column, an omitted
            non-nullable list column, an unknown category label, or a ``None``
            in a column that declares no fill value.
        """
        cols = self._discover_columns()
        unknown = set(data) - set(cols)
        if unknown:
            raise SchemaError(f"unknown columns in append data: {sorted(unknown)}")
        scalar_ds = {n: o for n, o in cols.items() if isinstance(o, h5py.Dataset)}
        list_grps = {n: o for n, o in cols.items() if not isinstance(o, h5py.Dataset)}

        prepared: dict[str, np.ndarray] = {}
        list_rows: dict[str, list[Any]] = {}
        lengths: set[int] = set()
        for name, values in data.items():
            if name in scalar_ds:
                ds = scalar_ds[name]
                if ATTR_CATEGORIES in ds.attrs:
                    arr = categorical.encode_labels(self._group, ds, values)
                else:
                    # None means "this row is missing", so it becomes the
                    # column's fill value before encoding.
                    arr = prepare_column_data(
                        ds.dtype, substitute_fill_for_none(ds, values, name)
                    )
                if arr.ndim != 1:
                    raise SchemaError(
                        f"append values for column {name!r} must be a 1-D sequence, "
                        f"got {arr.ndim}-D"
                    )
                prepared[name] = arr
                lengths.add(arr.shape[0])
            else:
                rows = list(values)
                list_rows[name] = rows
                lengths.add(len(rows))
        if len(lengths) > 1:
            raise SchemaError(f"append columns have unequal lengths: {sorted(lengths)}")
        k = lengths.pop() if lengths else 0
        if k == 0:
            return

        # Validate before mutating. A scalar column absent from the append is
        # filled with its fill value, so a fill-less boolean column must be
        # provided. A list column absent from the append becomes null lists, so
        # it must be nullable (carry a top-level MASK).
        for name, ds in scalar_ds.items():
            has_fill = ds.id.get_create_plist().fill_value_defined() == 2
            if name not in prepared and not has_fill:
                raise SchemaError(
                    f"column {name!r} has no fill value and must be provided "
                    "in the append data"
                )
        for name, g in list_grps.items():
            if name not in list_rows:
                if MEMBER_MASK not in g:
                    raise SchemaError(
                        f"list column {name!r} is not nullable and must be provided "
                        "in the append data"
                    )
                list_rows[name] = [None] * k

        # Append-protocol step 1: read the pre-append state. The strict
        # (repairing) read keeps a malformed foreign GENERATION from being
        # incremented as-is, which could collide with an index's residue
        # tokens and spuriously validate it after the step-5 rewrite.
        n_old = self.nrows
        n_new = n_old + k
        g_old = indexes.mutation_generation(self._group)

        # Steps 2-3: extend + write every scalar column (equal-extent
        # invariant); resize leaves absent columns' new rows at their fill.
        for name, ds in scalar_ds.items():
            extend_to(ds, n_new)
            if name in prepared:
                ds[n_old:n_new] = prepared[name]

        # Encode + write every list column (leaf-first, top OFFSETS last).
        for name, g in list_grps.items():
            lists.append_list_column(g, list_rows[name], n_old)

        # Flush the new column data before committing NROWS, so a crash cannot
        # publish a larger NROWS that points at unwritten rows (H5Col ordering).
        self._group.file.flush()

        # Step 4: maintain the supported search indexes — future-valued tokens
        # first, then content, then flush (tokens-before-content rule).
        if maintain_indexes:
            if g_old is None and indexes.search_index_datasets(self._group):
                # A table with indexes must carry GENERATION (rule 12); repair.
                g_old = indexes.ensure_generation(self._group)
            if g_old is not None and indexes.append_refresh_indexes(
                self._group, g_old, n_new
            ):
                self._group.file.flush()

        # Step 5: bump GENERATION whenever the table carries it, so indexes
        # not maintained above fail the validity check.
        if g_old is not None:
            write_uint64_attr(self._group, ATTR_GENERATION, g_old + 1)

        # Step 6: commit NROWS last, then flush again.
        write_uint64_attr(self._group, ATTR_NROWS, n_new)
        self._group.file.flush()

    def truncate(self, nrows: int, *, maintain_indexes: bool = False) -> None:
        """Shrink the logical table to *nrows* rows (H5Col logical truncation).

        The truncation is logical: no column dataset changes extent, and the
        rows ``[nrows, old_NROWS)`` become reserved storage that consumers
        ignore. List columns need no extra writes — the smaller ``NROWS``
        bounds their offsets recursively. Reclaiming physical space would
        require rewriting each column to its new extent, which this
        implementation does not do.

        Index handling mirrors :meth:`append` (the spec applies the same
        steps 4-6 with the new row count): by default every search index is
        left detectably stale by the ``GENERATION`` bump; with
        ``maintain_indexes=True`` the supported indexes are rebuilt inside the
        protocol and remain valid after the commit.

        Truncating to the current row count is a no-op (nothing changes, so
        nothing is published); growing is an error — that is what
        :meth:`append` is for.

        Raises
        ------
        SchemaError
            If *nrows* is negative or greater than the current row count.
        """
        n_new = int(nrows)
        if n_new < 0:
            raise SchemaError(f"cannot truncate to a negative row count {n_new}")
        n_old = self.nrows
        if n_new > n_old:
            raise SchemaError(
                f"cannot truncate {n_old} rows to {n_new}; truncation only shrinks"
            )
        if n_new == n_old:
            return
        g_old = indexes.mutation_generation(self._group)

        # Step 4: maintain the supported search indexes — future-valued tokens
        # first, then content, then flush (tokens-before-content rule).
        if maintain_indexes:
            if g_old is None and indexes.search_index_datasets(self._group):
                # A table with indexes must carry GENERATION (rule 12); repair.
                g_old = indexes.ensure_generation(self._group)
            if g_old is not None and indexes.append_refresh_indexes(
                self._group, g_old, n_new
            ):
                self._group.file.flush()

        # Step 5: bump GENERATION whenever the table carries it, so indexes
        # not maintained above fail the validity check.
        if g_old is not None:
            write_uint64_attr(self._group, ATTR_GENERATION, g_old + 1)

        # Step 6: commit NROWS last, then flush.
        write_uint64_attr(self._group, ATTR_NROWS, n_new)
        self._group.file.flush()

    # -- reading ------------------------------------------------------------ #
    def read(
        self,
        columns: Sequence[str] | None = None,
        *,
        where: Any = None,
        explain: bool = False,
    ) -> Any:
        """Read columns (default all) as ``{name: array}`` over ``[0, NROWS)``.

        With ``where=`` (a query :class:`~h5col.query.Expression`, a
        ``List[Tuple]`` = AND, or a ``List[List[Tuple]]`` = OR-of-ANDs), only the
        matching rows are returned. With ``explain=True`` the return value is a
        ``(result, QueryPlan)`` pair.

        Raises
        ------
        KeyError
            If a requested column name is not a column of the table.
        SchemaError
            If ``where=`` is malformed or references an unknown column.
        """
        if where is not None or explain:
            sel = self.select(where)
            result = sel.read(columns)
            return (result, sel.explain()) if explain else result
        names = list(columns) if columns is not None else self.column_names
        cols = self.columns
        out: dict[str, Any] = {}
        for name in names:
            if name not in cols:
                raise KeyError(name)
            out[name] = cols[name].read()
        return out

    def select(self, where: Any = None) -> query.Selection:
        """Build a lazy :class:`~h5col.query.Selection` over the table.

        Accepts a query :class:`~h5col.query.Expression`, a ``List[Tuple]``
        (AND), or a ``List[List[Tuple]]`` (OR-of-ANDs, pyarrow DNF). ``None``
        selects every row.
        """
        return query.Selection(self, query._to_expression(where))

    def count(self, where: Any = None) -> int:
        """Number of rows matching *where* (no column materialization)."""
        return self.select(where).count

    def build_index(
        self,
        column: str,
        kind: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> SearchIndex:
        """Build a search index over *column* (alias of :meth:`add_search_index`)."""
        return self.add_search_index(column, kind, name=name, description=description)

    # -- search indexes ------------------------------------------------------ #
    @property
    def search_indexes(self) -> dict[str, SearchIndex]:
        """Every search-index dataset under ``SEARCH_INDEXES``, wrapped by kind."""
        return {
            name: wrap_index(ds, self)
            for name, ds in indexes.search_index_datasets(self._group).items()
        }

    def add_search_index(
        self,
        column: str,
        kind: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> SearchIndex:
        """Build a search index over *column* and link it to the column.

        With ``kind=None`` the family is picked automatically: ``BITMAP`` for
        boolean and categorical columns (low cardinality, exact equality
        answers), ``CHUNK_MINMAX`` for any other orderable column. Building an
        index over an unchanged table is not a mutation: ``GENERATION`` is
        created (``0``) if absent but never incremented. The default dataset
        name is ``<column>__<kind, lowercased>`` — a readable convention only;
        the linkage is the object reference in the column's
        ``SEARCH_INDEX_LIST``.

        Raises
        ------
        KeyError
            If *column* is not a column of the table.
        SchemaError
            If *column* is a list column, no index family applies to its dtype
            (``kind=None``), *kind* is unimplemented, or ``SEARCH_INDEXES``
            already holds a dataset of the chosen name.
        ReservedNameError
            If *name* is a H5Col reserved name.
        ConformanceError
            If the table carries no ``NROWS`` attribute.
        """
        cols = self._discover_columns()
        if column not in cols:
            raise KeyError(column)
        ds = cols[column]
        if not isinstance(ds, h5py.Dataset):
            raise SchemaError(
                f"search indexes over list columns are not permitted ({column!r})"
            )
        if kind is None:
            if not indexes.supported_index_dtype(ds.dtype):
                raise SchemaError(
                    f"no search-index family applies to column {column!r} with "
                    f"dtype {ds.dtype!r}"
                )
            if is_bool_dtype(ds.dtype) or ATTR_CATEGORIES in ds.attrs:
                kind = KIND_BITMAP
            else:
                kind = KIND_CHUNK_MINMAX
        if kind == KIND_CHUNK_MINMAX:
            index_ds = indexes.create_chunk_minmax(
                self._group, ds, name=name, description=description
            )
        elif kind == KIND_SORTED_ROWS:
            index_ds = indexes.create_sorted_rows(
                self._group, ds, name=name, description=description
            )
        elif kind == KIND_BITMAP:
            index_ds = indexes.create_bitmap(
                self._group, ds, name=name, description=description
            )
        else:
            raise SchemaError(
                f"search-index kind {kind!r} is not implemented "
                "(CHUNK_BLOOM is sub-phase 4c)"
            )
        return wrap_index(index_ds, self, ds)

    def refresh_indexes(self) -> int:
        """Rebuild every supported search index against the current table state.

        Restores indexes left stale by ``append(maintain_indexes=False)`` or by
        any other mutation. Returns the number of indexes refreshed; indexes of
        unsupported kinds are left untouched (and stay detectably stale).
        """
        return indexes.refresh_all_indexes(self._group)

    def index_is_valid(self, index: SearchIndex | Any) -> bool:
        """The H5Col consumer validity check for *index* (wrapper or dataset)."""
        ds = index.dataset if isinstance(index, SearchIndex) else index
        return indexes.index_is_valid(ds, self._group)

    # -- validation --------------------------------------------------------- #
    def _check_version(self) -> None:
        v = self.version
        if v is None:
            raise ConformanceError("table has no VERSION attribute")
        try:
            major = int(v.split(".")[0])
        except (ValueError, IndexError) as exc:
            raise ConformanceError(f"unparsable VERSION {v!r}") from exc
        if major > SUPPORTED_MAJOR:
            raise VersionError(
                f"table VERSION major {major} exceeds supported {SUPPORTED_MAJOR}"
            )

    def _check_nrows_attr(self) -> None:
        if ATTR_NROWS not in self._group.attrs:
            raise ConformanceError("table has no NROWS attribute")
        val = np.asarray(self._group.attrs[ATTR_NROWS])
        if val.shape != ():
            raise ConformanceError(
                f"NROWS must be a scalar attribute, got shape {val.shape}"
            )
        if not (val.dtype.kind == "u" and val.dtype.itemsize == 8):
            raise ConformanceError(f"NROWS must be uint64, got dtype {val.dtype}")

    def validate(self, *, deep: bool = False) -> None:
        """Check the H5Col consistency requirements, raising on any violation.

        ``deep=True`` additionally re-derives every *valid* search index from
        its column and compares (consistency rule 9's semantic half) — an
        O(index build) check; the default run is structural only. A stale index
        is never an error: the validity check disables it, as the spec intends.

        Raises
        ------
        ConformanceError
            On the first consistency violation found.
        VersionError
            If the table's ``VERSION`` major exceeds the supported major.
        """
        if not self.is_table_group(self._group):
            raise ConformanceError("missing/incorrect CLASS attribute")
        self._check_version()
        self._check_nrows_attr()
        nrows = self.nrows

        columns = self._discover_columns()
        datasets = {n: o for n, o in columns.items() if isinstance(o, h5py.Dataset)}
        list_grps = {
            n: o for n, o in columns.items() if not isinstance(o, h5py.Dataset)
        }

        # Rule 2 applies to column datasets only; list column members have their
        # own per-level extents (checked below).
        extents = {ds.shape[0] for ds in datasets.values()}
        if len(extents) > 1:
            raise ConformanceError(
                f"column datasets have unequal extents: {sorted(extents)}"
            )
        if extents and next(iter(extents)) < nrows:
            raise ConformanceError("a column extent is smaller than NROWS")

        # Rule 6: column-order lists every column (datasets AND list columns).
        order = read_str_array_attr(self._group, ATTR_COLUMN_ORDER)
        if order is not None:
            if sorted(order) != sorted(columns):
                raise ConformanceError(
                    "column-order does not list every column exactly once"
                )

        # Rules 10 and 11: every list column subtree is structurally conformant.
        for lg in list_grps.values():
            lists.validate_list_column(lg, nrows)

        for name, ds in datasets.items():
            user_fill = ds.id.get_create_plist().fill_value_defined() == 2
            if FixedString.is_fixed_string(ds.dtype) or ds.dtype.kind not in ("b",):
                if not user_fill and not _looks_boolean(ds):
                    raise ConformanceError(
                        f"non-boolean column {name!r} lacks a user-defined fill value"
                    )
            if _looks_boolean(ds) and user_fill:
                raise ConformanceError(
                    f"boolean column {name!r} must not declare a fill value"
                )

        # INDEX_COLUMNS: non-null references to direct-child column datasets
        # (compared by HDF5 path, not basename); _index must agree with the first.
        if ATTR_INDEX_COLUMNS in self._group.attrs:
            child_paths = {ds.name for ds in datasets.values()}
            resolved_names: list[str] = []
            for r in self._group.attrs[ATTR_INDEX_COLUMNS]:
                if references.is_null_ref(r):
                    raise ConformanceError("INDEX_COLUMNS contains a null reference")
                obj = references.resolve(self._group, r)
                if obj.name not in child_paths:
                    raise ConformanceError(
                        f"INDEX_COLUMNS entry {obj.name!r} is not a direct-child "
                        "column dataset"
                    )
                resolved_names.append(obj.name.rsplit("/", 1)[-1])
            idx = read_str_attr(self._group, ATTR_INDEX)
            if idx is not None and resolved_names and idx != resolved_names[0]:
                raise ConformanceError(
                    f"_index {idx!r} does not match INDEX_COLUMNS[0] "
                    f"{resolved_names[0]!r}"
                )

        # Categorical columns and the CATEGORIES subgroup (rules 5 and 8).
        cat_group = self._group.get(GROUP_CATEGORIES)
        referenced: set[str] = set()
        for name, ds in datasets.items():
            if ATTR_CATEGORIES not in ds.attrs:
                continue
            if ds.dtype.kind not in ("i", "u"):
                raise ConformanceError(
                    f"categorical column {name!r} must have an integer datatype"
                )
            ref = ds.attrs[ATTR_CATEGORIES]
            if references.is_null_ref(ref):
                raise ConformanceError(
                    f"categorical column {name!r} has a null CATEGORIES reference"
                )
            cat_ds = references.resolve(self._group, ref)
            if cat_group is None or not cat_ds.name.startswith(f"{cat_group.name}/"):
                raise ConformanceError(
                    f"categorical column {name!r} CATEGORIES does not resolve into "
                    "the CATEGORIES subgroup"
                )
            referenced.add(cat_ds.name)
            if ds.id.get_create_plist().fill_value_defined() == 2:
                fill = int(np.asarray(ds.fillvalue))
                ncats = cat_ds.shape[0]
                if 0 <= fill < ncats:
                    raise ConformanceError(
                        f"categorical column {name!r} fill {fill} collides with a "
                        f"valid code [0, {ncats})"
                    )
        if cat_group is not None:
            for cname, cobj in cat_group.items():
                if not isinstance(cobj, h5py.Dataset):
                    raise ConformanceError(
                        f"CATEGORIES contains a non-dataset object {cname!r}"
                    )
                if cobj.name not in referenced:
                    raise ConformanceError(
                        f"categories dataset {cname!r} is not referenced by any "
                        "categorical column"
                    )

        # Search indexes: rules 3, 4, 12, and rule 9 for every supported kind.
        indexes.validate_search_indexes(self._group, nrows, deep=deep)

    def add_column(
        self,
        spec: ColumnSpec | ListColumnSpec,
        *,
        default_chunk_bytes: int | None = None,
    ) -> Column | ListColumn:
        """Add a new column to an existing table (schema evolution).

        A scalar column is created and grown to the table's current extent; its
        existing rows read as its fill value (missing). A fill-less (boolean)
        column, or a list column (whose "missing" analogue is a null list that
        would need explicit backfilling), cannot represent pre-existing rows, so
        adding one to a table that already has rows is refused.

        Raises
        ------
        SchemaError
            If a column of that name already exists, the spec is invalid, or the
            column cannot backfill pre-existing rows (a boolean or list column on
            a non-empty table).
        """
        cols = self._discover_columns()
        if spec.name in cols:
            raise SchemaError(f"column {spec.name!r} already exists")
        self._validate_column_spec(spec)

        if isinstance(spec, ListColumnSpec):
            is_list = True
            cannot_backfill = True
        else:
            is_list = False
            cannot_backfill = spec.is_boolean
        if cannot_backfill and self.nrows > 0:
            kind = "list" if is_list else "boolean"
            raise SchemaError(
                f"cannot add {kind} column {spec.name!r} to a table with rows: "
                "it cannot represent the pre-existing rows as missing"
            )

        datasets = {n: o for n, o in cols.items() if isinstance(o, h5py.Dataset)}
        obj = self._create_one_column(
            self._group, spec, default_chunk_bytes=default_chunk_bytes
        )
        if not is_list:
            extent = next(iter({d.shape[0] for d in datasets.values()}), self.nrows)
            extend_to(obj, extent)
        order = read_str_array_attr(self._group, ATTR_COLUMN_ORDER) or list(cols)
        if ATTR_COLUMN_ORDER in self._group.attrs:
            del self._group.attrs[ATTR_COLUMN_ORDER]
        write_utf8_array_attr(self._group, ATTR_COLUMN_ORDER, [*order, spec.name])
        return self._wrap(obj)


def _looks_boolean(dataset: Any) -> bool:
    from .booleans import is_bool_dtype

    return is_bool_dtype(dataset.dtype)


def _infer_column_spec(name: str, arr: Any) -> ColumnSpec:
    a = np.asarray(arr)
    if a.dtype.kind == "b":
        return ColumnSpec(name=name, dtype=bool_dtype())
    if a.dtype.kind in ("U", "S") or (
        a.dtype.kind == "O" and all(isinstance(v, str | bytes) for v in a.ravel())
    ):
        vals = a.ravel().tolist()
        max_bytes = max(
            (len(v.encode("utf-8") if isinstance(v, str) else v) for v in vals),
            default=1,
        )
        return ColumnSpec(name=name, dtype=FixedString(max(1, max_bytes)))
    return ColumnSpec(name=name, dtype=a.dtype)
