"""Analyst-facing query layer (Appendix A, Layer 2).

Composes the Layer-1 index primitives into a pyarrow-parity selection API:
``field()`` expressions and DNF tuple filters, a lazy :class:`Selection`, and a
planner that skips I/O via valid indexes while always returning the same rows a
brute-force scan would.

**Correctness contract:** the result is defined by the data, never by an index.
Every index use is validity-gated; a stale, missing, or structurally unusable
index silently falls back to a scan. Missing-value semantics are three-valued
(Kleene), matching Arrow/pyarrow: a value predicate is UNKNOWN on a missing row,
``NOT`` never turns UNKNOWN into TRUE, and a row is selected if the whole
expression evaluates to TRUE. Under "select if TRUE" a leaf reduces to its set
of TRUE rows, an AND-term is the intersection of its leaves, and the DNF is the
union of its terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import categorical, indexes
from . import missing as _missing
from ._hdf5 import gather_rows as _gather
from .exceptions import ConformanceError, SchemaError, StaleIndexError
from .ordering import is_spacepad
from .searchindex import (
    BitmapIndex,
    ChunkMinMaxIndex,
    SortedRowsIndex,
    _canon_element,
    _encode_query_value,
)
from .strings import FixedString

if TYPE_CHECKING:
    from .column import Column
    from .table import Table

#: Predicate operators after normalization (``!=`` / ``not in`` become negated
#: ``==`` / ``in``; presence is ``is_null`` / ``is_valid``).
_RANGE_OPS = ("<", "<=", ">", ">=")
_ORDER_EQ_OPS = (*_RANGE_OPS, "==")
_VALUE_OPS = (*_ORDER_EQ_OPS, "in")
_PRESENCE_OPS = ("is_null", "is_valid")
_PRED_OPS = (*_VALUE_OPS, *_PRESENCE_OPS)

#: Operators accepted in pyarrow tuple filters.
_TUPLE_OPS = {"=", "==", "!=", "<", "<=", ">", ">=", "in", "not in"}

#: Guard against pathological DNF blow-up (deeply nested AND/OR/NOT).
_MAX_DNF_TERMS = 1024


# --------------------------------------------------------------------------- #
# Expression tree
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Pred:
    """A single-column predicate leaf (never negated; see :class:`_Not`)."""

    column: str
    op: str
    value: Any = None


@dataclass(frozen=True)
class _And:
    children: tuple[Any, ...]


@dataclass(frozen=True)
class _Or:
    children: tuple[Any, ...]


@dataclass(frozen=True)
class _Not:
    child: Any


def _as_node(obj: Any) -> Any:
    """Unwrap *obj* to the expression tree node it carries.

    Parameters
    ----------
    obj:
        The other operand of an ``&`` or ``|`` combination. It must be an
        :class:`Expression`; anything else — a bare :class:`Field`, a plain
        value — raises ``SchemaError`` rather than being coerced.
    """
    if isinstance(obj, Expression):
        return obj._node
    raise SchemaError(f"expected a query Expression, got {obj!r}")


class Expression:
    """A boolean combination of predicates, built by :func:`field` + ``& | ~``.

    Mirrors pyarrow: ``(field("a") > 1) & (field("b") == 2) | ~field("c").is_valid()``.
    """

    __slots__ = ("_node",)

    def __init__(self, node: Any) -> None:
        self._node = node

    def __and__(self, other: Any) -> Expression:
        return Expression(_And((self._node, _as_node(other))))

    def __or__(self, other: Any) -> Expression:
        return Expression(_Or((self._node, _as_node(other))))

    def __invert__(self) -> Expression:
        return Expression(_Not(self._node))

    def __repr__(self) -> str:
        return f"<h5col.Expression {_node_repr(self._node)}>"


class Field:
    """A column reference; comparisons return :class:`Expression` leaves."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __lt__(self, value: Any) -> Expression:
        return Expression(_Pred(self.name, "<", value))

    def __le__(self, value: Any) -> Expression:
        return Expression(_Pred(self.name, "<=", value))

    def __gt__(self, value: Any) -> Expression:
        return Expression(_Pred(self.name, ">", value))

    def __ge__(self, value: Any) -> Expression:
        return Expression(_Pred(self.name, ">=", value))

    def __eq__(self, value: Any) -> Expression:  # type: ignore[override]
        return Expression(_Pred(self.name, "==", value))

    def __ne__(self, value: Any) -> Expression:  # type: ignore[override]
        return Expression(_Not(_Pred(self.name, "==", value)))

    __hash__ = None  # type: ignore[assignment]

    def isin(self, values: Any) -> Expression:
        """A predicate matching rows whose value is in *values* (pyarrow ``in``).

        Parameters
        ----------
        values:
            Any iterable of values to match, in the column's decoded form. It
            is consumed immediately, so a generator is fine.
        """
        return Expression(_Pred(self.name, "in", tuple(values)))

    def is_null(self) -> Expression:
        """A predicate matching rows that are missing (pyarrow ``is_null``)."""
        return Expression(_Pred(self.name, "is_null"))

    def is_valid(self) -> Expression:
        """A predicate matching rows that are present (pyarrow ``is_valid``)."""
        return Expression(_Pred(self.name, "is_valid"))

    def __repr__(self) -> str:
        return f"<h5col.field {self.name!r}>"


def field(name: str) -> Field:
    """Reference a column by name for building query :class:`Expression` objects.

    Parameters
    ----------
    name:
        A column name. It is not checked here — a name that is not a column of
        the table raises when the expression is evaluated, not when it is
        built, so an expression can be assembled before a table is opened.
    """
    return Field(name)


def _node_repr(node: Any) -> str:
    """Render an expression tree node as compact text, for ``repr``.

    Parameters
    ----------
    node:
        A ``_Pred``, ``_Not``, ``_And`` or ``_Or`` node, whose children are
        rendered recursively. Anything else falls back to its ``repr``.
    """
    if isinstance(node, _Pred):
        if node.op in _PRESENCE_OPS:
            return f"{node.column}.{node.op}()"
        return f"{node.column} {node.op} {node.value!r}"
    if isinstance(node, _Not):
        return f"NOT({_node_repr(node.child)})"
    if isinstance(node, _And):
        return "(" + " AND ".join(_node_repr(c) for c in node.children) + ")"
    if isinstance(node, _Or):
        return "(" + " OR ".join(_node_repr(c) for c in node.children) + ")"
    return repr(node)


# --------------------------------------------------------------------------- #
# where= parsing (Expression | List[Tuple] | List[List[Tuple]])
# --------------------------------------------------------------------------- #
def _is_pred_tuple(obj: Any) -> bool:
    """True if *obj* has the shape of a pyarrow filter tuple.

    Parameters
    ----------
    obj:
        Any object. It is a predicate tuple if it is a 3-tuple whose first two
        elements are strings. The operator string itself is not checked here;
        :func:`_pred_tuple_to_node` does that.
    """
    return (
        isinstance(obj, tuple)
        and len(obj) == 3
        and isinstance(obj[0], str)
        and isinstance(obj[1], str)
    )


def _pred_tuple_to_node(t: tuple[Any, ...]) -> Any:
    """Convert one pyarrow filter tuple into an expression tree node.

    Parameters
    ----------
    t:
        A ``(column, operator, value)`` tuple, already shape-checked by
        :func:`_is_pred_tuple`. The operator must be one of :data:`_TUPLE_OPS`;
        ``!=`` and ``not in`` become negated ``==`` and ``in`` nodes, and the
        value of an ``in`` is consumed into a tuple straight away.
    """
    col, op, value = t
    if op not in _TUPLE_OPS:
        raise SchemaError(
            f"unknown tuple operator {op!r}; use one of {sorted(_TUPLE_OPS)}"
        )
    if op in ("=", "=="):
        return _Pred(col, "==", value)
    if op == "!=":
        return _Not(_Pred(col, "==", value))
    if op == "in":
        return _Pred(col, "in", tuple(value))
    if op == "not in":
        return _Not(_Pred(col, "in", tuple(value)))
    return _Pred(col, op, value)


def _to_expression(where: Any) -> Expression | None:
    """Normalize ``where=`` (any accepted shape) to an :class:`Expression`.

    Returns ``None`` for "match every row" (``where=None`` or empty).

    Parameters
    ----------
    where:
        The ``where=`` argument in any accepted shape: an :class:`Expression`,
        a single ``(column, op, value)`` tuple, a list of such tuples (ANDed),
        or a list of lists of them (an OR of ANDs). ``None`` and an empty list
        both mean every row. A bare :class:`Field` — a column reference that
        was never compared — and anything else raise ``SchemaError``.
    """
    if where is None:
        return None
    if isinstance(where, Expression):
        return where
    if isinstance(where, Field):
        raise SchemaError(
            "a bare field() is not a predicate; compare it, e.g. field('x') > 0"
        )
    if _is_pred_tuple(where):
        return Expression(_pred_tuple_to_node(where))
    if isinstance(where, (list, tuple)):
        items = list(where)
        if not items:
            return None
        if all(_is_pred_tuple(it) for it in items):
            return Expression(_And(tuple(_pred_tuple_to_node(it) for it in items)))
        if all(
            isinstance(it, (list, tuple)) and it and all(_is_pred_tuple(t) for t in it)
            for it in items
        ):
            terms = tuple(
                _And(tuple(_pred_tuple_to_node(t) for t in term)) for term in items
            )
            return Expression(_Or(terms))
        raise SchemaError(
            "where= list must be List[Tuple] (AND) or List[List[Tuple]] (OR of ANDs)"
        )
    raise SchemaError(f"cannot interpret where={where!r}")


# --------------------------------------------------------------------------- #
# DNF normalization
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Leaf:
    pred: _Pred
    negated: bool = False


def _dnf(node: Any, negated: bool = False) -> list[list[_Leaf]]:
    """Normalize to disjunctive normal form: a list of AND-terms of leaves.

    Parameters
    ----------
    node:
        The expression tree node to normalize: a ``_Pred``, ``_Not``, ``_And``
        or ``_Or``. Any other object raises ``SchemaError``.
    negated:
        True when *node* sits under an odd number of ``_Not`` nodes. The
        negation is pushed down through the recursion rather than rewritten
        into the tree, so a ``_Pred`` reached with this set becomes a negated
        ``_Leaf``. Callers normalizing a whole expression leave it at the
        default.
    """
    if isinstance(node, _Pred):
        return [[_Leaf(node, negated)]]
    if isinstance(node, _Not):
        return _dnf(node.child, not negated)
    if isinstance(node, _And):
        if not negated:
            terms: list[list[_Leaf]] = [[]]
            for child in node.children:
                child_terms = _dnf(child, False)
                terms = [t + ct for t in terms for ct in child_terms]
                if len(terms) > _MAX_DNF_TERMS:
                    raise SchemaError(
                        f"query expands to more than {_MAX_DNF_TERMS} DNF terms; "
                        "simplify the predicate"
                    )
            return terms
        # NOT(a AND b) = (NOT a) OR (NOT b)
        result: list[list[_Leaf]] = []
        for child in node.children:
            result += _dnf(child, True)
        return result
    if isinstance(node, _Or):
        if not negated:
            out: list[list[_Leaf]] = []
            for child in node.children:
                out += _dnf(child, False)
            return out
        # NOT(a OR b) = (NOT a) AND (NOT b)
        return _dnf(_And(tuple(_Not(c) for c in node.children)), False)
    raise SchemaError(f"unexpected query node {node!r}")


# --------------------------------------------------------------------------- #
# explain plan
# --------------------------------------------------------------------------- #
@dataclass
class LeafPlan:
    """How one predicate leaf was evaluated (a row in :class:`QueryPlan`)."""

    column: str
    op: str
    negated: bool
    # "sorted_rows" | "bitmap" | "chunk_minmax+verify" | "scan" | "presence"
    # | "categorical-empty"
    method: str
    note: str = ""


@dataclass
class TermPlan:
    """How one AND-term of the DNF was evaluated (its per-leaf plans)."""

    leaves: list[LeafPlan] = _dc_field(default_factory=list)


@dataclass
class QueryPlan:
    """Machine-readable explanation of how a :class:`Selection` was evaluated."""

    nrows: int
    terms: list[TermPlan] = _dc_field(default_factory=list)
    matched: int = -1

    def __str__(self) -> str:
        lines = [f"QueryPlan: {self.matched} / {self.nrows} rows matched"]
        for i, term in enumerate(self.terms):
            joiner = "OR " if i else "   "
            lines.append(f"{joiner}AND-term {i}:")
            for leaf in term.leaves:
                neg = "NOT " if leaf.negated else ""
                extra = f"  [{leaf.note}]" if leaf.note else ""
                lines.append(
                    f"      {neg}{leaf.column} {leaf.op} via {leaf.method}{extra}"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
def _column(table: Table, name: str) -> Column:
    """The scalar column *name* of *table*, for a predicate to be built on.

    Parameters
    ----------
    table:
        The table whose columns are looked in.
    name:
        The column name taken from a predicate leaf. A name the table does not
        have raises ``KeyError``; a name that resolves to a list column raises
        ``SchemaError``, since predicates on list columns are not supported.
    """
    from .column import Column
    from .listcolumn import ListColumn

    cols = table.columns
    if name not in cols:
        raise KeyError(name)
    col = cols[name]
    if isinstance(col, ListColumn):
        raise SchemaError(f"predicates on list columns are not supported ({name!r})")
    assert isinstance(col, Column)
    return col


def _valid_indexes(col: Column) -> list[Any]:
    """Valid, usable indexes for *col*; never raises.

    Resolving ``col.search_indexes`` itself can raise (e.g. a non-conformant
    ``SEARCH_INDEX_LIST``); per the query contract a broken linkage yields no
    usable index and the query silently scans, so the whole resolution is
    guarded, not just the per-index validity check.

    Parameters
    ----------
    col:
        The column whose search indexes are resolved and validity-checked. A
        column with no indexes, or one whose linkage or index datasets are
        broken, yields an empty list.
    """
    out: list[Any] = []
    try:
        candidates = col.search_indexes
    except Exception:
        return out
    for idx in candidates:
        try:
            if idx.is_valid:
                out.append(idx)
        except Exception:
            continue
    return out


def _first(indexes_list: list[Any], cls: type) -> Any | None:
    """The first index of type *cls* in *indexes_list*, or None if there is none.

    Parameters
    ----------
    indexes_list:
        Indexes to look through, in the order :func:`_valid_indexes` resolved
        them; when a column has several of one kind, the first one wins.
    cls:
        The index class to match, tested with ``isinstance``.
    """
    for idx in indexes_list:
        if isinstance(idx, cls):
            return idx
    return None


def _sorted_unique(rows: np.ndarray) -> np.ndarray:
    """Row positions as a sorted, duplicate-free ``int64`` array.

    Parameters
    ----------
    rows:
        Row positions as an index primitive returned them: any order, any
        integer dtype, duplicates allowed. They are cast to ``int64``.
    """
    return np.unique(np.asarray(rows, dtype=np.int64))


def _chunks_to_rows(col_ds: Any, chunk_ids: np.ndarray, nrows: int) -> np.ndarray:
    """Expand candidate chunk numbers into every row position they cover.

    Parameters
    ----------
    col_ds:
        The column dataset, whose chunk length sets the rows per chunk (its
        whole extent when the dataset is contiguous).
    chunk_ids:
        Chunk numbers to expand, used in the order given. A pruning index
        returns them ascending, which is what makes the result ascending too.
    nrows:
        The table's row count, which bounds the last chunk so that rows
        reserved above the table's extent are never included.
    """
    cl = indexes.source_chunk_len(col_ds, nrows)
    parts = [
        np.arange(c * cl, min(nrows, (c + 1) * cl), dtype=np.int64) for c in chunk_ids
    ]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


#: Gather only while the wanted rows are confined to at most this fraction of
#: the column's chunks. Beyond it a gather reads what a plain read would have
#: read anyway, and the per-run overhead makes it the slower of the two.
GATHER_CHUNK_FRACTION = 0.5


def _worth_gathering(col: Column, rows: np.ndarray, nrows: int) -> bool:
    """True if reading *rows* by gather should beat reading the whole column.

    Parameters
    ----------
    col:
        The column that would be read. A contiguous (unchunked) column has
        nothing to coalesce, so a gather never wins on one.
    rows:
        The wanted row positions. An empty selection is worth gathering, since
        it reads nothing at all.
    nrows:
        The table's row count, used to count the column's chunks. Zero rows
        means there is nothing to gather.
    """
    ds = col.dataset
    if nrows == 0 or ds.chunks is None:
        return False  # nothing to coalesce on a contiguous dataset
    if rows.size == 0:
        return True  # empty result, no read at all
    chunk_len = int(ds.chunks[0])
    total = -(-nrows // chunk_len)
    touched = np.unique(rows // chunk_len).size
    return touched <= total * GATHER_CHUNK_FRACTION


def _present_subset(col: Column, raw: np.ndarray) -> np.ndarray:
    """Present-row mask over a raw value subset (matches Column.is_missing).

    Parameters
    ----------
    col:
        The column the values came from. A boolean column, or one that
        declares no fill value, has no missing rows at all, so every row of
        the subset counts as present.
    raw:
        Raw stored values for the rows in question, as read from the column.
        The mask has this shape.
    """
    if col.is_boolean or not col._has_user_fill():
        return np.ones(raw.shape, dtype=np.bool_)
    return ~_missing.is_missing(raw, col.dataset.fillvalue)


#: Sentinel: a categorical label that has no code (matches nothing).
_NO_MATCH = object()

#: Exceptions from a primitive that mean "this index cannot answer" — the query
#: falls back to a scan. Covers a stale/structurally-invalid index AND the
#: low-level errors a token-valid-but-corrupt index dataset raises (e.g. a
#: CHUNK_MINMAX whose datatype is not the expected compound → IndexError). The
#: token check only compares SOURCE_* tokens, so structural corruption slips
#: past it; the contract is scan-fallback, never a crash.
_INDEX_UNUSABLE = (StaleIndexError, ConformanceError, IndexError, ValueError, TypeError)


def _index_query_value(col: Column, value: Any) -> Any:
    """Translate a query value into the domain the query primitives expect.

    Every categorical query value is a *label* (of any type — labels may be
    numeric, not only strings), so it must become its integer code before it is
    compared against the stored codes by any primitive or the verify path.
    Returns :data:`_NO_MATCH` for an unknown categorical label.

    Parameters
    ----------
    col:
        The column being queried. Only a categorical column changes the value;
        for every other column the value is handed back untouched.
    value:
        A single query value as the caller wrote it: a category label for a
        categorical column, otherwise a value in the column's own domain.
    """
    if col.is_categorical:
        code = _encode_for(col, value)
        return _NO_MATCH if code is None else code
    return value


def _encode_for(col: Column, value: Any) -> Any:
    """Encode a single query value, resolving categorical labels to codes.

    Categorical predicates are always expressed in *labels* (which may be
    numeric, not just strings); this maps the label to its integer code.
    Returns ``None`` when a categorical label is unknown (no code, no possible
    match); callers treat that as "matches nothing".

    Parameters
    ----------
    col:
        The column the value will be compared against. A categorical column
        maps the label to its code; every other column goes through
        ``_encode_query_value``, which canonicalizes the value for exact
        comparison against the stored values.
    value:
        A single query value: a category label for a categorical column,
        otherwise a value in the column's own domain.
    """
    if col.is_categorical:
        try:
            return int(
                categorical.encode_labels(col._table.group, col.dataset, [value])[0]
            )
        except SchemaError:
            return None
    return _encode_query_value(col.dataset, value)


#: Largest magnitude an integer keeps exactly in ``float64``. A comparison
#: against a bigger integer would round the value before comparing, so those
#: fall back to the Python path — the very case that makes the object domain
#: the exact one.
_EXACT_FLOAT_INT = 2**53


def _is_int(value: Any) -> bool:
    """True for a Python or NumPy integer.

    Parameters
    ----------
    value:
        The query value to test. A ``bool`` does not count as an integer here.
    """
    # bool is a subclass of int; it is handled by its own branch.
    return not isinstance(value, bool) and isinstance(value, int | np.integer)


def _is_real(value: Any) -> bool:
    """True for a Python or NumPy integer or float.

    Parameters
    ----------
    value:
        The query value to test. A ``bool`` does not count as a real number
        here, matching :func:`_is_int`.
    """
    return not isinstance(value, bool) and isinstance(
        value, int | float | np.integer | np.floating
    )


def _apply(arr: np.ndarray, op: str, v: Any) -> np.ndarray:
    """Apply one comparison operator element-wise in NumPy.

    Parameters
    ----------
    arr:
        The values to compare. The caller has already established that the
        comparison is exact for this dtype.
    op:
        One of ``==``, ``<``, ``<=``, ``>``, ``>=``. Anything other than the
        first four is taken to be ``>=``, so an operator outside the five is
        not rejected here — callers must not pass one.
    v:
        The right-hand value, already cast by the caller to a type that
        compares exactly against *arr*.
    """
    if op == "==":
        return np.asarray(arr == v, dtype=np.bool_)
    if op == "<":
        return np.asarray(arr < v, dtype=np.bool_)
    if op == "<=":
        return np.asarray(arr <= v, dtype=np.bool_)
    if op == ">":
        return np.asarray(arr > v, dtype=np.bool_)
    return np.asarray(arr >= v, dtype=np.bool_)


def _vector_compare(
    raw: np.ndarray, op: str, v: Any, *, spacepad: bool
) -> np.ndarray | None:
    """Mask for ``raw op v`` computed in NumPy, or None if not provably exact.

    None sends the caller to the per-element Python path, which is always
    exact but converts every stored value into a Python object first. Every
    branch here is one where the vectorized comparison is exact by
    construction, so the two paths cannot disagree.

    Parameters
    ----------
    raw:
        Raw stored values for the rows under consideration. Its dtype kind
        picks the branch; a kind none of them covers returns None.
    op:
        The comparison operator. Anything outside :data:`_ORDER_EQ_OPS` — that
        is, ``in`` — returns None.
    v:
        The right-hand query value, already encoded for the column by
        :func:`_encode_for`. A value whose type does not go with the column's
        dtype kind, such as a float bound on an integer column, returns None.
    spacepad:
        True when the column is a SPACEPAD fixed string, in which case
        trailing spaces are stripped from *raw* before comparing. It has no
        effect on any other dtype kind.
    """
    if op not in _ORDER_EQ_OPS:
        return None
    kind = raw.dtype.kind

    if kind == "b":
        if not isinstance(v, bool | np.bool_):
            return None
        return _apply(raw, op, np.bool_(v))

    if kind in "iu":
        if not _is_int(v):
            return None  # a float bound against an integer column
        info = np.iinfo(raw.dtype)
        iv = int(v)
        if iv < int(info.min) or iv > int(info.max):
            # Outside the column's representable range, so every stored value
            # answers the same way and no element needs looking at.
            below = iv < int(info.min)
            const = {
                "==": False,
                "<": not below,
                "<=": not below,
                ">": below,
                ">=": below,
            }[op]
            return np.full(raw.shape, const, dtype=np.bool_)
        return _apply(raw, op, raw.dtype.type(iv))

    if kind == "f":
        if not _is_real(v):
            return None
        if _is_int(v) and abs(int(v)) > _EXACT_FLOAT_INT:
            return None  # would round before comparing
        return _apply(raw, op, np.float64(v))

    if kind == "S":
        if not isinstance(v, bytes | np.bytes_):
            return None
        # Scalar comparison is safe at any value length (verified against the
        # Python semantics); only the ``in`` path has a width trap.
        left = np.strings.rstrip(raw, b" ") if spacepad else raw
        return _apply(left, op, bytes(v))

    return None


def _vector_isin(
    raw: np.ndarray, values: set[Any], *, spacepad: bool
) -> np.ndarray | None:
    """Mask for membership computed in NumPy, or None if not provably exact.

    Parameters
    ----------
    raw:
        Raw stored values for the rows under consideration. Its dtype kind
        picks the branch; a kind none of them covers returns None.
    values:
        The query values to match, already encoded for the column. An empty
        set matches nothing. Values that cannot equal any stored value — one
        outside an integer column's range, or a string wider than the column —
        are dropped here rather than passed to ``np.isin``, which would
        truncate a too-wide string to the column width and match wrong rows. A
        value whose type does not go with the column's dtype kind returns
        None.
    spacepad:
        True when the column is a SPACEPAD fixed string, in which case
        trailing spaces are stripped from *raw* before matching. It has no
        effect on any other dtype kind.
    """
    if not values:
        return np.zeros(raw.shape, dtype=np.bool_)
    kind = raw.dtype.kind

    if kind == "b":
        if not all(isinstance(v, bool | np.bool_) for v in values):
            return None
        return np.isin(raw, np.array([bool(v) for v in values], dtype=np.bool_))

    if kind in "iu":
        if not all(_is_int(v) for v in values):
            return None
        info = np.iinfo(raw.dtype)
        keep = [int(v) for v in values if int(info.min) <= int(v) <= int(info.max)]
        if not keep:
            return np.zeros(raw.shape, dtype=np.bool_)
        return np.isin(raw, np.array(keep, dtype=raw.dtype))

    if kind == "f":
        if not all(_is_real(v) for v in values):
            return None
        if any(_is_int(v) and abs(int(v)) > _EXACT_FLOAT_INT for v in values):
            return None
        return np.isin(raw, np.array([float(v) for v in values], dtype=np.float64))

    if kind == "S":
        if not all(isinstance(v, bytes | np.bytes_) for v in values):
            return None
        left = np.strings.rstrip(raw, b" ") if spacepad else raw
        width = left.dtype.itemsize
        # A value wider than the column cannot equal any stored value, and
        # must be dropped rather than passed to np.isin, which would silently
        # truncate it to the column width and match the wrong rows.
        keep_bytes = [bytes(v) for v in values if len(bytes(v)) <= width]
        if not keep_bytes:
            return np.zeros(raw.shape, dtype=np.bool_)
        return np.isin(left, np.array(keep_bytes, dtype=left.dtype))

    return None


def _compare_subset(col: Column, raw: np.ndarray, op: str, value: Any) -> np.ndarray:
    """Boolean mask of the comparison over a raw subset, in the exact domain.

    Only meaningful where the row is present; callers AND with the present mask.

    Parameters
    ----------
    col:
        The column the values came from. It supplies the string padding
        convention and, for a categorical column, the label-to-code encoding.
    raw:
        Raw stored values for the rows under consideration. The mask has this
        shape.
    op:
        One of the value operators: ``in``, or an order or equality operator.
        Any other operator raises ``SchemaError``.
    value:
        The query value, in the caller's own domain — it is encoded for the
        column here. A collection of values for ``in``, a single value
        otherwise. An unknown categorical label matches nothing under ``==``,
        is dropped from an ``in``, and raises ``SchemaError`` under an order
        operator, which has no answer for a label with no code.
    """
    spacepad = FixedString.is_fixed_string(col.dtype) and is_spacepad(col.dataset)

    if op == "in":
        encoded = set()
        for v in value:
            e = _encode_for(col, v)
            if e is not None:
                encoded.add(e)
        fast = _vector_isin(raw, encoded, spacepad=spacepad)
        if fast is not None:
            return fast
        canon = [_canon_element(x, spacepad=spacepad) for x in raw.tolist()]
        return np.array([c in encoded for c in canon], dtype=np.bool_)

    v = _encode_for(col, value)
    if v is None:  # unknown categorical label
        if op == "==":
            return np.zeros(raw.shape, dtype=np.bool_)
        raise SchemaError(f"unknown category {value!r} has no order for {op!r}")

    fast = _vector_compare(raw, op, v, spacepad=spacepad)
    if fast is not None:
        return fast

    canon = [_canon_element(x, spacepad=spacepad) for x in raw.tolist()]
    if op == "==":
        hit = [c == v for c in canon]
    elif op == "<":
        hit = [c < v for c in canon]
    elif op == "<=":
        hit = [c <= v for c in canon]
    elif op == ">":
        hit = [c > v for c in canon]
    elif op == ">=":
        hit = [c >= v for c in canon]
    else:
        raise SchemaError(f"unknown operator {op!r}")
    return np.array(hit, dtype=np.bool_)


def _verify_leaf(
    leaf: _Leaf, table: Table, candidates: np.ndarray | None, plan: LeafPlan
) -> np.ndarray:
    """TRUE rows for a leaf by reading values (scan/verify path, Kleene).

    Parameters
    ----------
    leaf:
        The predicate leaf to evaluate, with its negation flag.
    table:
        The table the leaf's column belongs to; it also supplies the row
        count.
    candidates:
        Row positions still in play from earlier leaves of the same AND-term,
        sorted ascending, or None when nothing has narrowed the term yet. Only
        these rows are read, so the result is always a subset of them. With
        None, a value leaf may first seed its own candidates from a valid
        CHUNK_MINMAX index, and otherwise every row is read.
    plan:
        The plan entry for this leaf. Its method, and any note, are filled in
        here; modified in place.
    """
    col = _column(table, leaf.pred.column)
    nrows = table.nrows
    op = leaf.pred.op

    if op in _PRESENCE_OPS:
        plan.method = "presence"
        rows = np.arange(nrows, dtype=np.int64) if candidates is None else candidates
        if col.is_boolean or not col._has_user_fill():
            present = np.ones(rows.shape, dtype=np.bool_)
        else:
            present = _present_subset(col, _gather(col.dataset, rows, nrows))
        true_mask = present if op == "is_valid" else ~present
        if leaf.negated:
            true_mask = ~true_mask
        return rows[true_mask]

    # value op: possibly seed candidates via a pruning index
    if candidates is None:
        cm = _first(_valid_indexes(col), ChunkMinMaxIndex)
        qv = _index_query_value(col, leaf.pred.value) if op in _ORDER_EQ_OPS else None
        if (
            cm is not None
            and not leaf.negated
            and op in _ORDER_EQ_OPS
            and qv is not _NO_MATCH
        ):
            try:
                chunk_ids = cm.prune(op, qv)
                rows = _chunks_to_rows(col.dataset, chunk_ids, nrows)
                plan.method = "chunk_minmax+verify"
                plan.note = f"{len(chunk_ids)} candidate chunks"
            except _INDEX_UNUSABLE as exc:
                rows = np.arange(nrows, dtype=np.int64)
                plan.method = "scan"
                plan.note = f"index unusable: {exc}"
        else:
            rows = np.arange(nrows, dtype=np.int64)
            plan.method = "scan"
    else:
        rows = candidates
        if not plan.method:
            plan.method = "scan"

    if rows.size == 0:
        return rows
    raw = _gather(col.dataset, rows, nrows)
    present = _present_subset(col, raw)
    cmp = _compare_subset(col, raw, op, leaf.pred.value)
    if leaf.negated:
        cmp = ~cmp
    return rows[present & cmp]


def _exact_leaf(leaf: _Leaf, col: Column, plan: LeafPlan) -> np.ndarray | None:
    """Exact TRUE rows via a valid index, or None to demote to verify.

    Parameters
    ----------
    leaf:
        The predicate leaf. A negated leaf, or one whose operator is not a
        value operator, cannot be answered exactly from an index and returns
        None straight away.
    col:
        The leaf's column, whose valid indexes are consulted. A column with no
        index that can answer the operator returns None.
    plan:
        The plan entry for this leaf. The index that answered it is recorded
        here; modified in place, and left untouched when None is returned.
    """
    if leaf.negated or leaf.pred.op not in _VALUE_OPS:
        return None
    idxs = _valid_indexes(col)
    op = leaf.pred.op
    sr = _first(idxs, SortedRowsIndex)
    bm = _first(idxs, BitmapIndex)
    try:
        if op in _ORDER_EQ_OPS:
            qv = _index_query_value(col, leaf.pred.value)
            if qv is _NO_MATCH:  # unknown categorical label
                if op == "==":
                    plan.method = "categorical-empty"
                    return np.empty(0, dtype=np.int64)
                # order op against an unknown label is caught eagerly in
                # _validate(); reaching here would be a bug.
                raise SchemaError(
                    f"unknown category {leaf.pred.value!r} has no order for {op!r}"
                )
            if sr is not None:
                rows = _sorted_unique(sr.rows(op, qv))
                plan.method = "sorted_rows"
                return rows
            if op == "==" and bm is not None:
                r = bm.rows(qv)
                if r is not None:
                    rows = _sorted_unique(r)
                    plan.method = "bitmap"
                    return rows
            return None
        # op == "in": translate every element to the primitive domain
        codes = [_index_query_value(col, v) for v in leaf.pred.value]
        usable = [c for c in codes if c is not _NO_MATCH]
        if bm is not None:
            r = bm.isin(usable)
            if r is not None:
                rows = _sorted_unique(r)
                plan.method = "bitmap"
                return rows
        if sr is not None:
            acc = np.zeros(col._table.nrows, dtype=np.bool_)
            for qv in usable:
                acc[sr.rows("==", qv)] = True
            plan.method = "sorted_rows"
            return np.flatnonzero(acc).astype(np.int64)
        return None
    except _INDEX_UNUSABLE:
        return None


def _term_rows(term: list[_Leaf], table: Table, term_plan: TermPlan) -> np.ndarray:
    """TRUE rows of one AND-term: the intersection of its leaves.

    Parameters
    ----------
    term:
        The leaves of the AND-term. They are reordered for evaluation — the
        ones an index answers exactly go first, so they narrow the survivors
        before any column is read — which the intersection makes harmless. An
        empty term narrows nothing and so matches every row.
    table:
        The table being queried; it supplies the columns and the row count.
    term_plan:
        The plan entry for this term. One :class:`LeafPlan` is appended for
        every leaf, in the order the leaves are given rather than the order
        they are evaluated; modified in place.
    """
    survivors: np.ndarray | None = None
    verify: list[tuple[_Leaf, LeafPlan]] = []

    # Exact leaves first: they narrow the survivor set with no column reads.
    for leaf in term:
        lp = LeafPlan(leaf.pred.column, leaf.pred.op, leaf.negated, "")
        term_plan.leaves.append(lp)
        col = _column(table, leaf.pred.column)
        rs = _exact_leaf(leaf, col, lp)
        if rs is not None:
            survivors = rs if survivors is None else np.intersect1d(survivors, rs)
            if survivors.size == 0:
                return np.empty(0, dtype=np.int64)
        else:
            verify.append((leaf, lp))

    for leaf, lp in verify:
        survivors = _verify_leaf(leaf, table, survivors, lp)
        if survivors.size == 0:
            return np.empty(0, dtype=np.int64)

    if survivors is None:
        return np.arange(table.nrows, dtype=np.int64)
    return survivors


def _validate(dnf: list[list[_Leaf]], table: Table) -> None:
    """Check every leaf of the query against the table before any row is read.

    Parameters
    ----------
    dnf:
        The normalized query: a list of AND-terms of leaves. Every leaf of
        every term is checked, whether or not evaluation would reach it, so
        the error a malformed query raises does not depend on leaf order or
        on which indexes happen to exist.
    table:
        The table the query runs against, used to resolve each leaf's column.
    """
    for term in dnf:
        for leaf in term:
            col = _column(table, leaf.pred.column)  # KeyError / list-column SchemaError
            if leaf.pred.op not in _PRED_OPS:
                raise SchemaError(f"unknown operator {leaf.pred.op!r}")
            if leaf.pred.op == "in" and not isinstance(
                leaf.pred.value, (list, tuple, set, frozenset, np.ndarray)
            ):
                raise SchemaError("'in' requires a collection of values")
            # An order comparison against an unknown categorical label has no
            # defined answer. Reject it EAGERLY here — before any per-term
            # short-circuiting — so the error is deterministic and independent
            # of leaf order and index configuration (planner == oracle).
            if (
                col.is_categorical
                and leaf.pred.op in _RANGE_OPS
                and _encode_for(col, leaf.pred.value) is None
            ):
                raise SchemaError(
                    f"unknown category {leaf.pred.value!r} has no order "
                    f"for {leaf.pred.op!r}"
                )


def _run(table: Table, expr: Expression | None) -> tuple[np.ndarray, QueryPlan]:
    """Evaluate a query, returning its matching rows and the plan it followed.

    The expression is normalized to DNF and validated whole before any row is
    read; the result is the union of the AND-terms, sorted and duplicate-free.

    Parameters
    ----------
    table:
        The table to query.
    expr:
        The query expression, or None to match every row of the table.
    """
    nrows = table.nrows
    plan = QueryPlan(nrows=nrows)
    if expr is None:
        rows = np.arange(nrows, dtype=np.int64)
        plan.matched = int(rows.size)
        return rows, plan
    dnf = _dnf(expr._node)
    _validate(dnf, table)
    result: np.ndarray | None = None
    for term in dnf:
        tp = TermPlan()
        plan.terms.append(tp)
        rows = _term_rows(term, table, tp)
        # Every _term_rows result is already sorted and duplicate-free, so a
        # lone AND-term is the answer as it stands. Folding it through
        # union1d would re-sort the entire result set to no purpose.
        result = rows if result is None else np.union1d(result, rows)
    if result is None:
        result = np.empty(0, dtype=np.int64)
    plan.matched = int(result.size)
    return result, plan


# --------------------------------------------------------------------------- #
# brute-force oracle (indexes ignored; independent semantics for tests)
# --------------------------------------------------------------------------- #
def _scan_leaf(leaf: _Leaf, table: Table) -> np.ndarray:
    """TRUE rows of one leaf, read from the values with no index consulted.

    Parameters
    ----------
    leaf:
        The predicate leaf to evaluate, with its negation flag.
    table:
        The table being queried. Its column is read whole and compared row by
        row in Python, which is the point of this path: it depends on nothing
        the planner depends on.
    """
    col = _column(table, leaf.pred.column)
    nrows = table.nrows
    op = leaf.pred.op
    present = ~col.is_missing()

    if op in _PRESENCE_OPS:
        base = present if op == "is_valid" else ~present
        if leaf.negated:
            base = ~base
        return np.flatnonzero(base).astype(np.int64)

    if col.is_categorical:
        # Categoricals live in the CODE domain for every op (labels — of any
        # type — map bijectively to codes); this mirrors the planner exactly.
        codes = np.asarray(col.codes[0:nrows]).tolist()
        if op == "in":
            targets = {
                c
                for c in (_encode_for(col, v) for v in leaf.pred.value)
                if c is not None
            }
            cmp = np.array([c in targets for c in codes], dtype=np.bool_)
        else:
            cv = _encode_for(col, leaf.pred.value)
            if (
                cv is None
            ):  # unknown label: == matches nothing, order rejected in _validate
                cmp = np.zeros(nrows, dtype=np.bool_)
            else:
                cmp = _py_compare(codes, op, cv)
    elif op == "in":
        vals = col.read(masked=False).tolist()
        targets = {_oracle_str(col, v) for v in leaf.pred.value}
        cmp = np.array([v in targets for v in vals], dtype=np.bool_)
    else:
        vals = col.read(masked=False).tolist()
        cmp = _py_compare(vals, op, _oracle_str(col, leaf.pred.value))

    if leaf.negated:
        cmp = ~cmp
    return np.flatnonzero(present & cmp).astype(np.int64)


def _oracle_str(col: Column, value: Any) -> Any:
    """Coerce a ``bytes`` query value to ``str`` for the oracle's string path.

    The oracle compares decoded ``str`` from ``col.read()``; the byte-wise
    planner accepts a ``bytes`` query value too, so the oracle must decode it
    to agree (fixed strings are UTF-8/ASCII, whose byte order equals codepoint
    order).

    Parameters
    ----------
    col:
        The column being compared. Only a string column changes the value.
    value:
        A single query value. A ``bytes``-like one is decoded; anything else,
        including a ``str``, passes through unchanged.
    """
    if col.is_string and isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8")
    return value


def _py_compare(values: list[Any], op: str, v: Any) -> np.ndarray:
    """Compare a list of values element-wise in the Python object domain.

    Parameters
    ----------
    values:
        One decoded value per row. Comparing them as Python objects keeps a
        mixed int/float comparison exact, which NumPy would not.
    op:
        One of ``==``, ``<``, ``<=``, ``>``, ``>=``. Any other operator raises
        ``SchemaError``.
    v:
        The right-hand query value, already put in the same domain as
        *values* by the caller.
    """
    if op == "==":
        hit = [x == v for x in values]
    elif op == "<":
        hit = [x < v for x in values]
    elif op == "<=":
        hit = [x <= v for x in values]
    elif op == ">":
        hit = [x > v for x in values]
    elif op == ">=":
        hit = [x >= v for x in values]
    else:
        raise SchemaError(f"unknown operator {op!r}")
    return np.array(hit, dtype=np.bool_)


def _scan_select(table: Table, expr: Expression | None) -> np.ndarray:
    """Brute-force reference: evaluate the expression with no index at all.

    Parameters
    ----------
    table:
        The table to query. Every column a leaf names is read whole.
    expr:
        The query expression, or None to match every row of the table.
    """
    nrows = table.nrows
    if expr is None:
        return np.arange(nrows, dtype=np.int64)
    dnf = _dnf(expr._node)
    _validate(dnf, table)
    result = np.empty(0, dtype=np.int64)
    for term in dnf:
        survivors: np.ndarray | None = None
        for leaf in term:
            rs = _scan_leaf(leaf, table)
            survivors = rs if survivors is None else np.intersect1d(survivors, rs)
        if survivors is None:
            survivors = np.arange(nrows, dtype=np.int64)
        result = np.union1d(result, survivors)
    return result


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
class Selection:
    """A lazy, composable row selection over a table.

    Built by :meth:`Table.select`; evaluates on first use and caches the result
    (a query, not a snapshot — re-evaluate a fresh Selection after a mutation).
    """

    def __init__(self, table: Table, expr: Expression | None) -> None:
        self._table = table
        self._expr = expr
        self._rows: np.ndarray | None = None
        self._plan: QueryPlan | None = None

    def _ensure(self) -> None:
        if self._rows is None:
            self._rows, self._plan = _run(self._table, self._expr)

    @property
    def row_positions(self) -> np.ndarray:
        """Sorted, unique ``int64`` positions of the matching rows.

        A :class:`Selection` is lazy: the query is validated and evaluated on
        first access (here, and via :attr:`count` / :meth:`read` / :meth:`explain`).

        Raises
        ------
        KeyError
            If the predicate references a column the table does not have.
        SchemaError
            If the predicate is malformed — a list-column predicate, an unknown
            operator, a non-collection ``in`` value, an order comparison against
            an unknown categorical label, or a predicate that expands to too many
            DNF terms.
        """
        self._ensure()
        assert self._rows is not None
        return self._rows

    @property
    def count(self) -> int:
        """The number of matching rows (materializes no column data)."""
        return int(self.row_positions.size)

    def __len__(self) -> int:
        return self.count

    def read(self, columns: Any = None, *, masked: bool = True) -> dict[str, Any]:
        """Materialize the selected rows as ``{name: values}``.

        Each value is a NumPy array for a scalar column and a Python ``list``
        (of per-row lists, ``None`` for a null row) for a list column.
        Evaluates the query on first use (see :attr:`row_positions`).

        A scalar column whose matching rows sit in at most
        :data:`GATHER_CHUNK_FRACTION` of its chunks is fetched with
        :meth:`Column.read_rows <h5col.Column.read_rows>`, reading those
        chunks and no others; otherwise it is read whole and then subset. A
        list column is always fetched with
        :meth:`ListColumn.read_rows <h5col.ListColumn.read_rows>`, which reads
        the span the matching rows cover and so is never wider than the whole
        column. The result is identical either way.

        Parameters
        ----------
        columns:
            Names to read, in the order given. None (the default) reads every
            column of the table.
        masked:
            As for :meth:`h5col.Table.read`: each scalar column comes back as a
            :class:`numpy.ma.MaskedArray` marking its missing rows unless False
            is passed. List columns accept it and ignore it.

        Raises
        ------
        KeyError
            If a requested column name is not a column of the table.
        SchemaError
            If the predicate is malformed (see :attr:`row_positions`).
        """
        from .column import Column  # local: .column imports this module's table

        rows = self.row_positions
        names = list(columns) if columns is not None else self._table.column_names
        cols = self._table.columns
        nrows = self._table.nrows
        out: dict[str, Any] = {}
        for name in names:
            if name not in cols:
                raise KeyError(name)
            col = cols[name]
            if not isinstance(col, Column):
                # A list column narrows to the span its matching rows cover,
                # which is never wider than reading it whole, so there is no
                # selectivity to weigh up first.
                out[name] = col.read_rows(rows)
                continue
            if _worth_gathering(col, rows, nrows):
                # Selective enough to be worth fetching only the wanted chunks.
                out[name] = col.read_rows(rows, masked=masked)
                continue
            out[name] = col.read(masked=masked)[rows]
        return out

    def to_arrow(self, columns: Any = None) -> Any:
        """Convert the selected rows to a :class:`pyarrow.Table`.

        As :meth:`h5col.Table.to_arrow`, restricted to the matching rows.

        .. versionadded:: 0.2.0

        Parameters
        ----------
        columns:
            Names to convert, in the order given. None (the default) converts
            every column of the table.
        """
        from . import arrow

        return arrow.table_arrow(self._table, columns, self.row_positions)

    def explain(self) -> QueryPlan:
        """The :class:`QueryPlan` describing how this selection was evaluated."""
        self._ensure()
        assert self._plan is not None
        return self._plan

    def __repr__(self) -> str:
        try:
            n = self.count if self._rows is not None else "?"
            return f"<h5col.Selection matched={n}>"
        except Exception:
            return "<h5col.Selection (unevaluated)>"
