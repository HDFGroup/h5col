"""Tests for the analyst-facing query layer (phase 4d).

The central strategy is a *differential* one: the accelerated planner (which uses
whatever indexes exist) must return byte-identical rows to a brute-force scan
oracle that ignores every index. Kleene/missing semantics are additionally
pinned with known-answer tests so the oracle itself is checked against
hand-computed truth, not only against the planner.
"""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from h5col import (
    ColumnSpec,
    LeafValuesSpec,
    ListColumnSpec,
    SchemaError,
    Table,
    bool_dtype,
    field,
    query,
)
from h5col.strings import FixedString

# Base data: 20 rows, several columns, with deliberate missing rows.
#   x (i8, fill -1)  : rows 4 and 13 are missing (-1)
#   y (f8, fill NaN) : rows 2 and 9 are missing (NaN)
#   s (S6, fill '')  : row 7 is missing ('')
#   ok (bool)        : no missing (booleans declare no fill)
#   size (categorical, ordered S<M<L<XL): row 5 is missing (None)
_N = 20
_X = [i if i not in (4, 13) else -1 for i in range(_N)]
_Y = [float(i) if i not in (2, 9) else math.nan for i in range(_N)]
_S = ["" if i == 7 else f"r{i:02d}" for i in range(_N)]
_OK = [i % 3 == 0 for i in range(_N)]
_SIZES = ["S", "M", "L", "XL"]
_SIZE = [None if i == 5 else _SIZES[i % 4] for i in range(_N)]


def make_table(h5file: h5py.File, name: str = "t") -> Table:
    t = Table.create(
        h5file.create_group(name),
        [
            ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4),
            ColumnSpec(name="y", dtype="f8", fill_value=math.nan, chunks=4),
            ColumnSpec(name="s", dtype=FixedString(6), fill_value="", chunks=4),
            ColumnSpec(name="ok", dtype=bool_dtype(), chunks=4),
            ColumnSpec(name="size", categories=_SIZES, ordered=True, chunks=4),
        ],
    )
    t.append({"x": _X, "y": _Y, "s": _S, "ok": _OK, "size": _SIZE})
    return t


# Every index configuration under which the planner must equal the oracle.
def _cfg_none(t: Table) -> None:
    pass


def _cfg_minmax(t: Table) -> None:
    t.add_search_index("x", "CHUNK_MINMAX")
    t.add_search_index("y", "CHUNK_MINMAX")
    t.add_search_index("s", "CHUNK_MINMAX")


def _cfg_sorted(t: Table) -> None:
    t.add_search_index("x", "SORTED_ROWS")
    t.add_search_index("s", "SORTED_ROWS")
    t.add_search_index("size", "SORTED_ROWS")


def _cfg_bitmap(t: Table) -> None:
    t.add_search_index("ok", "BITMAP")
    t.add_search_index("size", "BITMAP")


def _cfg_mixed(t: Table) -> None:
    t.add_search_index("x", "SORTED_ROWS")
    t.add_search_index("x", "CHUNK_MINMAX")
    t.add_search_index("ok", "BITMAP")
    t.add_search_index("size", "BITMAP")
    t.add_search_index("s", "SORTED_ROWS")


_CONFIGS = [_cfg_none, _cfg_minmax, _cfg_sorted, _cfg_bitmap, _cfg_mixed]

# A broad predicate battery exercised against every config.
_EXPRS = [
    field("x") > 15,
    field("x") <= 3,
    field("x") == 7,
    field("x") != 7,
    (field("x") >= 10) & (field("x") < 15),
    (field("x") < 2) | (field("x") > 17),
    ~(field("x") > 10),
    field("x").isin([3, 7, 11, 4]),
    ~field("x").isin([3, 7, 11]),
    field("s") >= "r05",
    field("s") == "r03",
    field("s") != "r03",
    field("s").isin(["r01", "r02", "r19"]),
    field("ok") == True,  # noqa: E712
    field("ok") != True,  # noqa: E712
    ~(field("ok") == False),  # noqa: E712
    field("size") == "M",
    field("size").isin(["S", "L"]),
    field("size") < "L",
    field("size") != "M",
    field("y") > 5.0,
    field("y") >= 15.0,
    field("y") < 3.0,
    field("x").is_null(),
    field("x").is_valid(),
    field("y").is_null(),
    field("s").is_null(),
    ~field("x").is_valid(),
    (field("x") > 5) & (field("size") == "M"),
    (field("x") > 5) | field("y").is_null(),
    (field("size").isin(["S", "L"])) & ~(field("x") == 7),
    (field("x") < 3) | (field("size") == "XL") | field("s").is_null(),
]


class TestOracleDifferential:
    @pytest.mark.parametrize("cfg", _CONFIGS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("expr", _EXPRS, ids=lambda e: query._node_repr(e._node))
    def test_planner_matches_scan(self, h5file, cfg, expr) -> None:
        t = make_table(h5file)
        cfg(t)
        got = t.select(where=expr).row_positions
        oracle = query._scan_select(t, expr)
        assert got.tolist() == oracle.tolist()

    @pytest.mark.parametrize("expr", _EXPRS, ids=lambda e: query._node_repr(e._node))
    def test_stale_indexes_fall_back(self, h5file, expr) -> None:
        # Build every index, then append so all indexes go stale; the planner
        # must ignore their old content and still equal a full scan.
        t = make_table(h5file)
        _cfg_mixed(t)
        t.append(
            {
                "x": [100, -1, 102],
                "y": [math.nan, 1.0, 2.0],
                "s": ["z0", "", "z2"],
                "ok": [True, False, True],
                "size": ["XL", None, "S"],
            }
        )
        assert all(not si.is_valid for si in t.search_indexes.values())
        got = t.select(where=expr).row_positions
        oracle = query._scan_select(t, expr)
        assert got.tolist() == oracle.tolist()


class TestKnownAnswer:
    """Hand-computed expected rows — an independent check of the semantics."""

    def test_missing_excluded_by_positive_and_negated(self, h5file) -> None:
        t = make_table(h5file)
        # x missing at rows 4, 13.  x != 7 must exclude 7 AND the missing rows.
        rows = t.select(where=field("x") != 7).row_positions.tolist()
        assert 7 not in rows
        assert 4 not in rows and 13 not in rows
        # positive predicate also excludes missing
        assert 4 not in t.select(where=field("x") > -5).row_positions.tolist()

    def test_not_over_range_excludes_missing(self, h5file) -> None:
        t = make_table(h5file)
        rows = set(t.select(where=~(field("x") > 10)).row_positions.tolist())
        # present rows with x <= 10, i.e. 0..10 minus the missing row 4
        assert rows == {0, 1, 2, 3, 5, 6, 7, 8, 9, 10}

    def test_is_null_and_is_valid(self, h5file) -> None:
        t = make_table(h5file)
        assert t.select(where=field("x").is_null()).row_positions.tolist() == [4, 13]
        assert t.select(where=field("y").is_null()).row_positions.tolist() == [2, 9]
        assert t.select(where=field("s").is_null()).row_positions.tolist() == [7]
        valid_x = t.select(where=field("x").is_valid()).row_positions.tolist()
        assert 4 not in valid_x and 13 not in valid_x and len(valid_x) == 18
        # ~is_valid == is_null
        assert (
            t.select(where=~field("x").is_valid()).row_positions.tolist()
            == t.select(where=field("x").is_null()).row_positions.tolist()
        )

    def test_categorical_order_uses_codes(self, h5file) -> None:
        t = make_table(h5file)
        # S<M<L<XL. size<'L' -> {S, M}; row 5 missing excluded.
        rows = t.select(where=field("size") < "L").row_positions.tolist()
        sizes = {i: _SIZE[i] for i in rows}
        assert all(v in ("S", "M") for v in sizes.values())
        assert 5 not in rows

    def test_between_via_two_comparisons(self, h5file) -> None:
        t = make_table(h5file)
        rows = t.select(
            where=(field("x") >= 10) & (field("x") <= 12)
        ).row_positions.tolist()
        assert rows == [10, 11, 12]

    def test_nan_is_missing_not_matched(self, h5file) -> None:
        t = make_table(h5file)
        # rows 2, 9 are NaN; no value predicate matches them.
        assert 2 not in t.select(where=field("y") > -1e9).row_positions.tolist()
        assert 9 not in t.select(where=field("y") < 1e9).row_positions.tolist()


class TestInputForms:
    def test_three_forms_agree(self, h5file) -> None:
        t = make_table(h5file)
        expr = (field("x") > 5) & (field("s") <= "r10")
        tuples = [("x", ">", 5), ("s", "<=", "r10")]
        a = t.select(where=expr).row_positions
        b = t.select(where=tuples).row_positions
        assert a.tolist() == b.tolist()

    def test_dnf_list_of_lists_is_or(self, h5file) -> None:
        t = make_table(h5file)
        expr = (field("x") < 2) | (field("x") > 17)
        dnf = [[("x", "<", 2)], [("x", ">", 17)]]
        assert (
            t.select(where=dnf).row_positions.tolist()
            == t.select(where=expr).row_positions.tolist()
        )

    def test_tuple_not_in_and_ne(self, h5file) -> None:
        t = make_table(h5file)
        assert (
            t.select(where=[("x", "not in", [3, 7])]).row_positions.tolist()
            == t.select(where=~field("x").isin([3, 7])).row_positions.tolist()
        )
        assert (
            t.select(where=[("x", "!=", 7)]).row_positions.tolist()
            == t.select(where=field("x") != 7).row_positions.tolist()
        )
        assert (  # "=" is accepted as an alias of "=="
            t.select(where=[("x", "=", 7)]).row_positions.tolist()
            == t.select(where=field("x") == 7).row_positions.tolist()
        )

    def test_empty_where_selects_all(self, h5file) -> None:
        t = make_table(h5file)
        assert t.select(where=None).count == _N
        assert t.select(where=[]).count == _N
        assert t.count() == _N

    def test_de_morgan_equivalence(self, h5file) -> None:
        t = make_table(h5file)
        a = t.select(where=~((field("x") > 5) & (field("x") < 15))).row_positions
        b = t.select(where=(~(field("x") > 5)) | (~(field("x") < 15))).row_positions
        assert a.tolist() == b.tolist()


class TestExplain:
    def test_methods_reported(self, h5file) -> None:
        t = make_table(h5file)
        t.add_search_index("x", "SORTED_ROWS")
        t.add_search_index("ok", "BITMAP")
        _, plan = t.read(where=field("x") > 15, explain=True)
        assert plan.terms[0].leaves[0].method == "sorted_rows"
        assert plan.matched == 4
        _, plan2 = t.read(where=field("ok") == True, explain=True)  # noqa: E712
        assert plan2.terms[0].leaves[0].method == "bitmap"

    def test_scan_and_prune_reported(self, h5file) -> None:
        t = make_table(h5file)
        t.add_search_index("x", "CHUNK_MINMAX")
        plan = t.select(where=field("x") > 15).explain()
        assert plan.terms[0].leaves[0].method == "chunk_minmax+verify"
        # unindexed column -> scan
        plan2 = t.select(where=field("y") > 5.0).explain()
        assert plan2.terms[0].leaves[0].method == "scan"
        assert "matched" in str(plan2)

    def test_presence_method(self, h5file) -> None:
        t = make_table(h5file)
        plan = t.select(where=field("x").is_null()).explain()
        assert plan.terms[0].leaves[0].method == "presence"


class TestSelectionApi:
    def test_row_positions_sorted_unique_int64(self, h5file) -> None:
        t = make_table(h5file)
        rows = t.select(where=(field("x") < 3) | (field("x") < 5)).row_positions
        assert rows.dtype == np.int64
        assert rows.tolist() == sorted(set(rows.tolist()))

    def test_read_subset_columns(self, h5file) -> None:
        t = make_table(h5file)
        out = t.read(columns=["x", "s"], where=[("x", "in", [3, 8, 11])])
        assert out["x"].tolist() == [3, 8, 11]
        assert out["s"].tolist() == ["r03", "r08", "r11"]
        assert set(out.keys()) == {"x", "s"}

    def test_read_all_columns_default(self, h5file) -> None:
        t = make_table(h5file)
        out = t.select(where=field("x") == 3).read()
        assert set(out.keys()) == {"x", "y", "s", "ok", "size"}
        assert out["x"].tolist() == [3]

    def test_count_without_materializing(self, h5file) -> None:
        t = make_table(h5file)
        assert t.count(where=field("x") > 15) == 4

    def test_len_matches_count(self, h5file) -> None:
        t = make_table(h5file)
        sel = t.select(where=field("x") > 15)
        assert len(sel) == sel.count == 4

    def test_lazy_and_cached(self, h5file) -> None:
        t = make_table(h5file)
        sel = t.select(where=field("x") > 15)
        assert sel._rows is None  # not evaluated yet
        _ = sel.row_positions
        assert sel._rows is not None

    def test_read_list_column(self, h5file) -> None:
        t = Table.create(
            h5file.create_group("lt"),
            [
                ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=4),
                ListColumnSpec(
                    name="r", values=LeafValuesSpec(dtype="f4"), nullable=True
                ),
            ],
        )
        t.append({"x": [1, 2, 3, 4], "r": [[1.0], None, [3.0, 3.5], [4.0]]})
        out = t.read(columns=["x", "r"], where=[("x", ">", 1)])
        assert out["x"].tolist() == [2, 3, 4]
        assert out["r"][0] is None
        assert list(out["r"][1]) == [3.0, 3.5]


class TestErrors:
    def test_unknown_column_raises(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(KeyError):
            _ = t.select(where=field("nope") > 1).row_positions

    def test_predicate_on_list_column_rejected(self, h5file) -> None:
        t = Table.create(
            h5file.create_group("lt"),
            [
                ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=4),
                ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4")),
            ],
        )
        t.append({"x": [1, 2], "r": [[1.0], [2.0]]})
        with pytest.raises(SchemaError, match="list column"):
            _ = t.select(where=field("r") > 1).row_positions

    def test_unknown_tuple_operator(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="unknown tuple operator"):
            _ = t.select(where=[("x", "~=", 1)]).row_positions

    def test_bare_field_rejected(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="bare field"):
            _ = t.select(where=field("x"))

    def test_in_requires_collection(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="collection"):
            _ = t.select(
                where=query.Expression(query._Pred("x", "in", 5))
            ).row_positions

    def test_nan_query_value_raises(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="NaN"):
            _ = t.select(where=field("y") == math.nan).row_positions

    def test_unknown_category_equality_is_empty(self, h5file) -> None:
        t = make_table(h5file)
        assert t.select(where=field("size") == "ZZ").count == 0

    def test_unknown_category_range_raises(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="unknown category"):
            _ = t.select(where=field("size") < "ZZ").row_positions

    def test_dnf_term_guard(self, h5file) -> None:
        t = make_table(h5file)
        # A chain of ORed ANDs whose product explodes past the cap.
        expr = field("x") > 0
        for _ in range(12):
            expr = expr & ((field("x") > 1) | (field("x") < 2))
        with pytest.raises(SchemaError, match="DNF terms"):
            _ = t.select(where=expr).row_positions

    def test_malformed_where_type(self, h5file) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="cannot interpret where"):
            _ = t.select(where=42).row_positions


class TestBuildIndexAlias:
    def test_table_build_index(self, h5file) -> None:
        t = make_table(h5file)
        si = t.build_index("x", "SORTED_ROWS")
        assert si.is_valid
        assert "x__sorted_rows" in t.search_indexes

    def test_column_build_index(self, h5file) -> None:
        t = make_table(h5file)
        col = t.columns["ok"]
        si = col.build_index("BITMAP")
        assert si.is_valid


class TestQueriesNeverBuildIndexes:
    def test_select_does_not_create_indexes(self, h5file) -> None:
        t = make_table(h5file)
        before = set(t.search_indexes)
        _ = t.select(where=field("x") > 5).row_positions
        t.read(where=field("size") == "M")
        assert set(t.search_indexes) == before


class Test4dReviewRegressions:
    """Regressions for defects found by the phase-4d adversarial review."""

    def _numeric_cat(self, h5file) -> Table:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="cat", categories=[10, 20, 30], chunks=4)],
        )
        t.append({"cat": [10, 30, 20, 10]})  # codes 0, 2, 1, 0
        return t

    def test_numeric_categorical_label_equality(self, h5file) -> None:
        # A categorical LABEL may be numeric; == / in / != must translate the
        # label to its code, not compare the raw int against stored codes.
        t = self._numeric_cat(h5file)
        assert t.select(where=field("cat") == 10).row_positions.tolist() == [0, 3]
        assert t.select(where=field("cat").isin([10, 20])).row_positions.tolist() == [
            0,
            2,
            3,
        ]
        assert t.select(where=field("cat") != 10).row_positions.tolist() == [1, 2]

    @pytest.mark.parametrize("cfg", _CONFIGS, ids=lambda c: c.__name__)
    def test_numeric_categorical_matches_oracle(self, h5file, cfg) -> None:
        t = self._numeric_cat(h5file)
        # bitmap/sorted over the numeric-labeled categorical, all exercised
        t.add_search_index("cat", "BITMAP")
        t.add_search_index("cat", "SORTED_ROWS")
        for e in (field("cat") == 10, field("cat").isin([10, 30]), field("cat") < 30):
            got = t.select(where=e).row_positions
            assert got.tolist() == query._scan_select(t, e).tolist()

    def test_bytes_query_value_on_string_column(self, h5file) -> None:
        t = make_table(h5file)
        e = query.Expression(query._Pred("s", "==", b"r05"))
        assert t.select(where=e).row_positions.tolist() == [5]
        assert query._scan_select(t, e).tolist() == [5]  # oracle agrees

    def test_unknown_categorical_order_is_deterministic(self, h5file) -> None:
        # An order op against an unknown label must raise regardless of leaf
        # order or short-circuiting siblings (found: planner swallowed it).
        t = make_table(h5file)
        for e in (
            (field("x") > 100) & (field("size") >= "ZZZ"),
            (field("size") >= "ZZZ") & (field("x") > 100),
            ((field("x") > 100) & (field("size") >= "ZZZ")) | (field("x") == 3),
        ):
            with pytest.raises(SchemaError, match="unknown category"):
                _ = t.select(where=e).row_positions

    def test_malformed_search_index_list_falls_back_to_scan(self, h5file) -> None:
        # A non-1-D SEARCH_INDEX_LIST must not crash the query.
        from h5col.reserved import ATTR_SEARCH_INDEX_LIST

        t = make_table(h5file)
        t.add_search_index("x", "CHUNK_MINMAX")
        ds = t.columns["x"].dataset
        ref0 = np.asarray(ds.attrs[ATTR_SEARCH_INDEX_LIST]).reshape(-1)[0]
        del ds.attrs[ATTR_SEARCH_INDEX_LIST]
        ds.attrs.create(ATTR_SEARCH_INDEX_LIST, ref0, dtype=h5py.ref_dtype)  # scalar
        got = t.select(where=field("x") > 5).row_positions
        assert got.tolist() == query._scan_select(t, field("x") > 5).tolist()

    def test_structurally_broken_token_valid_index_falls_back(self, h5file) -> None:
        # A token-valid CHUNK_MINMAX whose datatype is not the expected compound
        # must degrade to a scan, not raise IndexError.
        from h5col._hdf5 import write_ascii_token_attr, write_uint64_attr
        from h5col.reserved import ATTR_SEARCH_INDEX_LIST, GROUP_SEARCH_INDEXES

        t = make_table(h5file)
        t.add_search_index("x", "CHUNK_MINMAX")
        si = t.group[GROUP_SEARCH_INDEXES]
        bad = si.create_dataset("x__bad", data=np.arange(3, dtype="i8"))
        write_ascii_token_attr(bad, "KIND", "CHUNK_MINMAX")
        write_uint64_attr(bad, "SOURCE_GENERATION", t.generation)
        write_uint64_attr(bad, "SOURCE_NROWS", t.nrows)
        ds = t.columns["x"].dataset
        del ds.attrs[ATTR_SEARCH_INDEX_LIST]
        ds.attrs.create(
            ATTR_SEARCH_INDEX_LIST, np.array([bad.ref], dtype=h5py.ref_dtype)
        )
        got = t.select(where=field("x") > 5).row_positions
        assert got.tolist() == query._scan_select(t, field("x") > 5).tolist()

    def test_explain_not_mislabeled_after_demotion(self, h5file) -> None:
        # A token-valid SORTED_ROWS broken structurally (ordered=false) must not
        # be reported as the method that answered a leaf that fell back to scan.
        from h5col.reserved import ATTR_ORDERED

        t = make_table(h5file)
        t.add_search_index("ok", "BITMAP")
        sr = t.add_search_index("x", "SORTED_ROWS")
        sr.dataset.attrs.modify(ATTR_ORDERED, np.False_)  # tokens untouched
        expr = (field("ok") == True) & (field("x") > 5)  # noqa: E712
        sel = t.select(where=expr)
        assert sel.row_positions.tolist() == query._scan_select(t, expr).tolist()
        x_leaf = next(
            leaf
            for term in sel.explain().terms
            for leaf in term.leaves
            if leaf.column == "x"
        )
        assert x_leaf.method != "sorted_rows"  # it actually scanned
