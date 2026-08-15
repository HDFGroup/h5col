"""The vectorized comparison path must agree with the exact Python one.

`_compare_subset` computes a predicate in NumPy where that is provably exact
and falls back to per-element Python objects otherwise. These tests pin both
halves of that contract: the fast path never disagrees with the brute-force
oracle, and the guards really do decline the cases that would round.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from h5col import ColumnSpec, FixedString, Table, bool_dtype, field, query
from h5col.exceptions import SchemaError
from h5col.query import (
    _ORDER_EQ_OPS,
    _apply,
    _vector_compare,
    _vector_isin,
)

OPS = ["==", "<", "<=", ">", ">="]


def _build(h5file: h5py.File, name: str, spec: ColumnSpec, values: list) -> Table:
    table = Table.create(h5file.create_group(name), [spec])
    table.append({spec.name: values})
    return table


# --------------------------------------------------------------------------- #
# Fast path vs the brute-force oracle
# --------------------------------------------------------------------------- #
INT_VALUES = [-128, -1, 0, 1, 42, 126, 127]


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint16"])
@pytest.mark.parametrize("op", OPS)
def test_integer_scan_matches_oracle(h5file: h5py.File, dtype: str, op: str) -> None:
    info = np.iinfo(np.dtype(dtype))
    vals = [v for v in INT_VALUES if info.min <= v <= info.max]
    table = _build(h5file, "t", ColumnSpec(name="x", dtype=dtype, chunks=4), vals * 3)
    for probe in (*vals, int(info.min), int(info.max)):
        expr = query.Expression(query._Pred("x", op, probe))
        got = table.select(expr).row_positions.tolist()
        assert got == query._scan_select(table, expr).tolist(), (dtype, op, probe)


@pytest.mark.parametrize("op", OPS)
def test_float_scan_matches_oracle(h5file: h5py.File, op: str) -> None:
    vals = [-1.5, 0.0, 1.5, 2.25, 1e300, float("nan")]
    table = _build(
        h5file,
        "t",
        ColumnSpec(name="x", dtype="float64", chunks=4, fill_value=-9e99),
        vals * 2,
    )
    for probe in (-1.5, 0.0, 1.5, 2.0, 1e300, 3):
        expr = query.Expression(query._Pred("x", op, probe))
        got = table.select(expr).row_positions.tolist()
        assert got == query._scan_select(table, expr).tolist(), (op, probe)


@pytest.mark.parametrize("op", OPS)
def test_string_scan_matches_oracle(h5file: h5py.File, op: str) -> None:
    vals = ["", "a", "ab", "abc", "b", "z", "A", "é"]
    table = _build(
        h5file, "t", ColumnSpec(name="x", dtype=FixedString(nbytes=8), chunks=3), vals
    )
    for probe in ("", "a", "ab", "b", "é", "abcdefghij"):  # last is over-width
        expr = query.Expression(query._Pred("x", op, probe))
        got = table.select(expr).row_positions.tolist()
        assert got == query._scan_select(table, expr).tolist(), (op, probe)


@pytest.mark.parametrize("op", OPS)
def test_categorical_scan_matches_oracle(h5file: h5py.File, op: str) -> None:
    table = _build(
        h5file,
        "t",
        ColumnSpec(name="x", categories=["a", "b", "c"], chunks=3),
        ["a", "b", "c", None, "b"],
    )
    for probe in ("a", "b", "c"):
        expr = query.Expression(query._Pred("x", op, probe))
        got = table.select(expr).row_positions.tolist()
        assert got == query._scan_select(table, expr).tolist(), (op, probe)


def test_boolean_scan_matches_oracle(h5file: h5py.File) -> None:
    table = _build(
        h5file,
        "t",
        ColumnSpec(name="x", dtype=bool_dtype(), chunks=3),
        [True, False, True, True, False],
    )
    for probe in (True, False):
        expr = query.Expression(query._Pred("x", "==", probe))
        assert (
            table.select(expr).row_positions.tolist()
            == query._scan_select(table, expr).tolist()
        )


@pytest.mark.parametrize(
    ("dtype", "vals", "probes"),
    [
        ("int32", [1, 2, 3, 4], [(1, 3), (), (99,), (1, 99)]),
        ("float64", [1.0, 2.5, 3.0], [(1.0, 3.0), (2.5,), (9.0,)]),
    ],
)
def test_isin_matches_oracle(
    h5file: h5py.File, dtype: str, vals: list, probes: list
) -> None:
    table = _build(h5file, "t", ColumnSpec(name="x", dtype=dtype, chunks=2), vals)
    for probe in probes:
        expr = query.Expression(query._Pred("x", "in", probe))
        assert (
            table.select(expr).row_positions.tolist()
            == query._scan_select(table, expr).tolist()
        ), probe


def test_isin_string_ignores_over_width_value(h5file: h5py.File) -> None:
    # np.isin would truncate a too-long value to the column width and match
    # the wrong rows; it must match nothing instead.
    table = _build(
        h5file,
        "t",
        ColumnSpec(name="x", dtype=FixedString(nbytes=4), chunks=2),
        ["abcd", "ab", "zz"],
    )
    expr = query.Expression(query._Pred("x", "in", ("abcdefgh",)))
    assert table.select(expr).count == 0
    assert query._scan_select(table, expr).tolist() == []

    both = query.Expression(query._Pred("x", "in", ("abcdefgh", "zz")))
    assert table.select(both).row_positions.tolist() == [2]


# --------------------------------------------------------------------------- #
# The guards: these must decline rather than round
# --------------------------------------------------------------------------- #
def test_guard_declines_float_bound_on_integer_column() -> None:
    raw = np.array([1, 2, 3], dtype="int64")
    assert _vector_compare(raw, "<", 2.5, spacepad=False) is None


def test_guard_declines_huge_integer_against_float_column() -> None:
    raw = np.array([1.0, 2.0], dtype="float64")
    assert _vector_compare(raw, "<", 2**53 + 1, spacepad=False) is None
    # Just inside the exact range is handled in NumPy.
    assert _vector_compare(raw, "<", 2**53, spacepad=False) is not None


def test_guard_declines_unknown_operator() -> None:
    raw = np.array([1, 2], dtype="int64")
    assert _vector_compare(raw, "in", 1, spacepad=False) is None


def test_out_of_range_integer_short_circuits_to_a_constant() -> None:
    raw = np.array([1, 2, 3], dtype="int8")
    above = _vector_compare(raw, "<", 10_000, spacepad=False)
    below = _vector_compare(raw, "<", -10_000, spacepad=False)
    assert above is not None and above.all()
    assert below is not None and not below.any()
    eq = _vector_compare(raw, "==", 10_000, spacepad=False)
    assert eq is not None and not eq.any()


def test_uint64_beyond_float_precision_is_exact(h5file: h5py.File) -> None:
    # 2**63 + 1 and 2**63 + 2 are indistinguishable as float64; the integer
    # domain must still separate them.
    big = [2**63 + 1, 2**63 + 2, 2**63 + 3]
    table = _build(
        h5file, "t", ColumnSpec(name="x", dtype="uint64", chunks=2, fill_value=0), big
    )
    expr = query.Expression(query._Pred("x", "==", 2**63 + 2))
    assert table.select(expr).row_positions.tolist() == [1]
    assert query._scan_select(table, expr).tolist() == [1]

    gt = query.Expression(query._Pred("x", ">", 2**63 + 2))
    assert table.select(gt).row_positions.tolist() == [2]


def test_int64_extremes_are_exact(h5file: h5py.File) -> None:
    vals = [np.iinfo(np.int64).min + 1, -1, 0, np.iinfo(np.int64).max]
    table = _build(h5file, "t", ColumnSpec(name="x", dtype="int64", chunks=2), vals)
    for probe in vals:
        expr = query.Expression(query._Pred("x", "==", int(probe)))
        assert (
            table.select(expr).row_positions.tolist()
            == query._scan_select(table, expr).tolist()
        ), probe


def test_isin_guard_declines_mixed_types() -> None:
    raw = np.array([1, 2], dtype="int64")
    assert _vector_isin(raw, {1, 2.5}, spacepad=False) is None
    assert _vector_isin(raw, set(), spacepad=False) is not None  # empty -> all False


@pytest.mark.parametrize(
    "make_expr",
    [
        lambda: field("x") > 2,  # single AND-term: returned as-is
        lambda: (field("x") < 2) | (field("x") > 4),  # disjoint OR terms
        lambda: (field("x") > 1) | (field("x") > 3),  # overlapping OR terms
        lambda: (field("x") > 1) & (field("x") < 5),  # two leaves, one term
    ],
)
def test_result_rows_are_sorted_and_duplicate_free(
    h5file: h5py.File, make_expr
) -> None:
    # A single AND-term is returned without a union pass, which is only sound
    # while every term result is itself sorted and unique; overlapping OR
    # terms must still be merged without repeats.
    table = _build(
        h5file, "t", ColumnSpec(name="x", dtype="int32", chunks=2), [1, 2, 3, 4, 5, 6]
    )
    expr = make_expr()
    rows = table.select(expr).row_positions
    assert rows.tolist() == sorted(set(rows.tolist()))
    assert rows.tolist() == query._scan_select(table, expr).tolist()


def test_negated_leaf_matches_oracle(h5file: h5py.File) -> None:
    table = _build(
        h5file,
        "t",
        ColumnSpec(name="x", dtype="int32", chunks=3, fill_value=-1),
        [1, 2, None, 4, 5],
    )
    expr = ~(field("x") > 2)
    assert (
        table.select(expr).row_positions.tolist()
        == query._scan_select(table, expr).tolist()
    )


# --------------------------------------------------------------------------- #
# _apply enforces its operator set rather than trusting the caller
# --------------------------------------------------------------------------- #
def test_apply_answers_each_operator_it_claims() -> None:
    arr = np.array([1, 2, 3], dtype=np.int64)
    assert _apply(arr, "==", np.int64(2)).tolist() == [False, True, False]
    assert _apply(arr, "<", np.int64(2)).tolist() == [True, False, False]
    assert _apply(arr, "<=", np.int64(2)).tolist() == [True, True, False]
    assert _apply(arr, ">", np.int64(2)).tolist() == [False, False, True]
    assert _apply(arr, ">=", np.int64(2)).tolist() == [False, True, True]


def test_apply_refuses_an_operator_it_does_not_know() -> None:
    # Not reachable through the public API: _vector_compare is the only caller
    # and admits exactly _ORDER_EQ_OPS. What this pins is the contract — an
    # operator added to that set but not here used to be answered as ">=",
    # which returns the wrong rows and says nothing about it.
    with pytest.raises(SchemaError, match="unknown comparison operator"):
        _apply(np.array([1, 2, 3], dtype=np.int64), "!=", np.int64(2))


def test_the_caller_gate_and_apply_agree_on_the_operator_set() -> None:
    # The two must not drift: every operator _vector_compare lets through has
    # to be one _apply implements, and vice versa.
    arr = np.array([1, 2, 3], dtype=np.int64)
    for op in _ORDER_EQ_OPS:
        assert _vector_compare(arr, op, 2, spacepad=False) is not None
        assert _apply(arr, op, np.int64(2)) is not None
    for op in ("in", "!=", "is_null", ""):
        assert _vector_compare(arr, op, 2, spacepad=False) is None
        with pytest.raises(SchemaError):
            _apply(arr, op, np.int64(2))
