"""Tests for search indexes: validity tokens, CHUNK_MINMAX, pruning, validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from h5py import h5t

from h5col import (
    ChunkMinMaxIndex,
    ColumnSpec,
    ConformanceError,
    LeafValuesSpec,
    ListColumnSpec,
    ObjectReferenceError,
    SchemaError,
    SearchIndex,
    StaleIndexError,
    Table,
    bool_dtype,
    indexes,
    references,
)
from h5col._hdf5 import extend_to, write_ascii_token_attr, write_uint64_attr
from h5col.reserved import (
    ATTR_GENERATION,
    ATTR_KIND,
    ATTR_SEARCH_INDEX_LIST,
    ATTR_SOURCE_GENERATION,
    ATTR_SOURCE_NROWS,
    ATTR_VALUES,
    GROUP_SEARCH_INDEXES,
)
from h5col.strings import FixedString

CHUNK = 64  # rows per chunk in the test tables


def make_table(f: h5py.File, n: int = 500) -> Table:
    """A table with int/float/string/bool columns and deterministic data."""
    rng = np.random.default_rng(42)
    t = Table.create(
        f.create_group("t"),
        [
            ColumnSpec(name="ts", dtype="i8", chunks=CHUNK),
            ColumnSpec(name="temp", dtype="f8", chunks=CHUNK),
            ColumnSpec(name="tag", dtype=FixedString(8), chunks=CHUNK),
            ColumnSpec(name="ok", dtype=bool_dtype(), chunks=CHUNK),
        ],
    )
    temp = rng.normal(20.0, 5.0, n)
    temp[::37] = np.nan  # real NaN data points
    t.append(
        {
            "ts": np.arange(n, dtype="i8"),  # clustered / sorted
            "temp": temp,
            "tag": [f"tag{i % 23:03d}" for i in range(n)],
            "ok": rng.integers(0, 2, n).astype(bool),
        }
    )
    return t


def brute_candidates(
    values: np.ndarray, present: np.ndarray, op: str, value: Any, chunk_len: int
) -> set[int]:
    """Chunks holding >= 1 present element satisfying the predicate (oracle)."""
    ops = {
        "<": lambda v: v < value,
        "<=": lambda v: v <= value,
        ">": lambda v: v > value,
        ">=": lambda v: v >= value,
        "==": lambda v: v == value,
        "between": lambda v: (v >= value[0]) & (v <= value[1]),
    }
    out = set()
    for cid in range(math.ceil(len(values) / chunk_len)):
        s, e = cid * chunk_len, min(len(values), (cid + 1) * chunk_len)
        hit = ops[op](values[s:e]) & present[s:e]
        if np.any(hit):
            out.add(cid)
    return out


# --------------------------------------------------------------------------- #
# Validity-token lifecycle
# --------------------------------------------------------------------------- #
class TestValidityTokens:
    def test_generation_absent_until_first_index(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        assert t.generation is None
        t.add_search_index("ts")
        assert t.generation == 0

    def test_building_more_indexes_does_not_increment(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        t.add_search_index("temp")
        assert t.generation == 0  # building over an unchanged table: no bump

    def test_new_index_tokens_match_current_state(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        assert si.source_generation == t.generation == 0
        assert si.source_nrows == t.nrows
        assert si.is_valid
        assert t.index_is_valid(si)

    def test_append_without_maintenance_goes_stale(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        assert t.generation == 1
        assert not si.is_valid
        t.validate()  # a stale index is never a validation error

    def test_refresh_restores_validity(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        t.add_search_index("temp")
        t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        assert t.refresh_indexes() == 2
        assert all(si.is_valid for si in t.search_indexes.values())
        t.validate(deep=True)

    def test_maintained_append_stays_valid(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        t.append(
            {
                "ts": [1000, 1001],
                "temp": [1.0, 2.0],
                "tag": ["x", "y"],
                "ok": [True, False],
            },
            maintain_indexes=True,
        )
        assert t.generation == 1
        assert si.is_valid
        assert si.source_nrows == t.nrows
        t.validate(deep=True)

    def test_crash_window_between_generation_and_nrows(self, h5file: h5py.File) -> None:
        # Simulate a crash after step 5 (GENERATION) but before step 6 (NROWS):
        # every index is disabled, the table stays readable at the old NROWS,
        # and validation still passes.
        t = make_table(h5file)
        t.add_search_index("ts")
        nrows = t.nrows
        write_uint64_attr(t.group, ATTR_GENERATION, t.generation + 1)
        assert not any(si.is_valid for si in t.search_indexes.values())
        assert t.nrows == nrows
        assert t["ts"].read().shape == (nrows,)
        t.validate()

    def test_wrong_token_dtype_fails_check(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        ds = si.dataset
        del ds.attrs[ATTR_SOURCE_NROWS]
        ds.attrs.create(ATTR_SOURCE_NROWS, np.int32(t.nrows))  # wrong dtype
        assert not si.is_valid  # absent-or-wrong-dtype token fails the check

    def test_missing_token_fails_check(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del si.dataset.attrs[ATTR_SOURCE_GENERATION]
        assert not si.is_valid


# --------------------------------------------------------------------------- #
# CHUNK_MINMAX content
# --------------------------------------------------------------------------- #
class TestChunkMinMaxContent:
    def test_int_entries_match_brute_force(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        assert isinstance(si, ChunkMinMaxIndex)
        e = si.entries()
        assert len(e) == math.ceil(t.nrows / CHUNK)
        ts = np.arange(t.nrows)
        for cid in range(len(e)):
            s, stop = cid * CHUNK, min(t.nrows, (cid + 1) * CHUNK)
            assert e[cid]["min"] == ts[s:stop].min()
            assert e[cid]["max"] == ts[s:stop].max()
            assert e[cid]["nan_count"] == 0
            assert e[cid]["fill_count"] == 0
            assert e[cid]["n"] == stop - s

    def test_partial_last_chunk_n(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=100)  # 100 rows / 64 per chunk -> 2 chunks
        si = t.add_search_index("ts")
        e = si.entries()
        assert e["n"].tolist() == [64, 36]

    def test_float_nan_and_fill_counts(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("temp")
        temp = t["temp"].dataset[: t.nrows]
        e = si.entries()
        for cid in range(len(e)):
            s, stop = cid * CHUNK, min(t.nrows, (cid + 1) * CHUNK)
            chunk = temp[s:stop]
            assert e[cid]["nan_count"] == np.isnan(chunk).sum()
            finite = chunk[~np.isnan(chunk)]
            assert e[cid]["min"] == finite.min()
            assert e[cid]["max"] == finite.max()

    def test_nan_fill_column_fill_equals_nan_count(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="f8", fill_value=np.nan, chunks=8)],
        )
        vals = np.arange(20.0)
        vals[3] = np.nan
        vals[9:12] = np.nan
        t.append({"x": vals})
        e = t.add_search_index("x").entries()
        # NaN fill: every NaN element is missing by definition.
        assert np.array_equal(e["fill_count"], e["nan_count"])
        assert e["fill_count"].tolist() == [1, 3, 0]

    def test_placeholder_chunk_all_missing(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=4)],
        )
        t.append({"x": [5, 6, 7, 8, -1, -1, -1, -1, 9, 10, 11, 12]})
        e = t.add_search_index("x").entries()
        assert e[1]["fill_count"] == e[1]["n"] == 4
        assert e[1]["min"] == e[1]["max"] == -1  # placeholders = fill value

    def test_string_entries_bytewise(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("tag")
        e = si.entries()
        tags = [f"tag{i % 23:03d}".encode() for i in range(t.nrows)]
        for cid in range(len(e)):
            s, stop = cid * CHUNK, min(t.nrows, (cid + 1) * CHUNK)
            assert e[cid]["min"] == min(tags[s:stop])
            assert e[cid]["max"] == max(tags[s:stop])

    def test_boolean_minmax_field_is_enum_on_disk(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ok", "CHUNK_MINMAX")
        tid = si.dataset.id.get_type()
        # min/max fields carry the source element type: the H5Col boolean enum.
        assert tid.get_member_class(0) == h5t.ENUM
        assert tid.get_member_class(1) == h5t.ENUM
        e = si.entries()
        assert e[0]["min"] == np.False_ and e[0]["max"] == np.True_

    def test_categorical_codes_minmax(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="c", categories=["low", "mid", "high"], chunks=4)],
        )
        t.append({"c": ["low", "mid", None, "high", "low", "low", None, "mid"]})
        e = t.add_search_index("c", "CHUNK_MINMAX").entries()
        assert e[0]["min"] == 0 and e[0]["max"] == 2 and e[0]["fill_count"] == 1
        assert e[1]["min"] == 0 and e[1]["max"] == 1 and e[1]["fill_count"] == 1

    def test_contiguous_column_single_chunk(self, h5file: h5py.File) -> None:
        # A foreign (hand-written) table with a contiguous column: one entry.
        g = h5file.create_group("foreign")
        write_ascii_token_attr(g, "CLASS", "COLUMN_TABLE")
        write_ascii_token_attr(g, "VERSION", "1.0")
        write_uint64_attr(g, "NROWS", 10)
        ds = g.create_dataset("x", data=np.arange(10.0), fillvalue=-1.0)
        assert ds.chunks is None
        assert indexes.data_chunk_count(ds, 10) == 1
        index_ds = indexes.create_chunk_minmax(g, ds)
        e = index_ds[...]
        assert len(e) == 1
        assert e[0]["min"] == 0.0 and e[0]["max"] == 9.0 and e[0]["n"] == 10

    def test_empty_table_zero_entries(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"), [ColumnSpec(name="x", dtype="i4", chunks=4)]
        )
        si = t.add_search_index("x")
        assert si.dataset.shape == (0,)
        assert si.is_valid
        assert si.prune("==", 1).size == 0

    def test_index_dataset_is_chunked_and_unlimited(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        ds = t.add_search_index("ts").dataset
        assert ds.chunks is not None
        assert ds.maxshape == (None,)
        assert ds.chunks[0] == indexes.INDEX_CHUNK_BYTES // ds.dtype.itemsize


# --------------------------------------------------------------------------- #
# Pruning (the Layer-1 primitive)
# --------------------------------------------------------------------------- #
class TestPrune:
    @pytest.mark.parametrize(
        ("op", "value"),
        [
            ("<", 130),
            ("<=", 128),
            (">", 320),
            (">=", 448),
            ("==", 200),
            ("between", (100, 140)),
        ],
    )
    def test_clustered_int_prunes_exactly(
        self, h5file: h5py.File, op: str, value: Any
    ) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        ts = np.arange(t.nrows)
        expect = brute_candidates(ts, np.ones(t.nrows, bool), op, value, CHUNK)
        got = set(si.prune(op, value).tolist())
        assert expect <= got  # never excludes a matching chunk
        assert got == expect  # clustered data: minmax pruning is exact
        assert len(got) < si.n_chunks  # and it actually pruned something

    def test_random_float_superset_property(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("temp")
        temp = t["temp"].dataset[: t.nrows]
        present = ~np.isnan(temp)
        for op, value in [
            ("<", 15.0),
            (">", 30.0),
            ("==", 20.0),
            ("between", (18.0, 22.0)),
        ]:
            expect = brute_candidates(temp, present, op, value, CHUNK)
            got = set(si.prune(op, value).tolist())
            assert expect <= got

    def test_string_equality(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("tag")
        tags = np.array(
            [f"tag{i % 23:03d}".encode() for i in range(t.nrows)], dtype="S8"
        )
        expect = brute_candidates(tags, np.ones(t.nrows, bool), "==", b"tag003", CHUNK)
        got = set(si.prune("==", "tag003").tolist())
        assert expect <= got

    def test_string_value_wider_than_column(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("tag")
        assert si.prune("==", "zzzzzzzzzzzz").size == 0  # over-width: no chunk

    def test_placeholder_chunk_is_never_a_candidate(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=4)],
        )
        t.append({"x": [5, 6, 7, 8, -1, -1, -1, -1, 9, 10, 11, 12]})
        si = t.add_search_index("x")
        # The all-missing chunk's placeholder bounds (-1) would satisfy "< 100";
        # it must be excluded regardless.
        assert si.prune("<", 100).tolist() == [0, 2]
        # A predicate matching the fill value never surfaces missing rows.
        assert si.prune("==", -1).size == 0

    def test_nan_query_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("temp")
        with pytest.raises(SchemaError):
            si.prune("==", np.nan)

    def test_unknown_op_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        with pytest.raises(SchemaError):
            si.prune("!=", 5)

    def test_stale_index_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        with pytest.raises(StaleIndexError):
            si.prune("<", 10)

    def test_chunk_row_range(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=100)
        si = t.add_search_index("ts")
        assert si.chunk_row_range(0) == (0, 64)
        assert si.chunk_row_range(1) == (64, 100)  # clipped to NROWS
        with pytest.raises(IndexError):
            si.chunk_row_range(2)

    def test_boolean_prune(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="b", dtype=bool_dtype(), chunks=4)],
        )
        t.append({"b": [False] * 8 + [True] * 4})
        si = t.add_search_index("b", "CHUNK_MINMAX")
        assert si.prune("==", True).tolist() == [2]
        assert si.prune("==", False).tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# Validation: consistency rules 3, 4, 9, 12
# --------------------------------------------------------------------------- #
class TestValidateRules:
    def test_rule3_non_dataset_object(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        t.group[GROUP_SEARCH_INDEXES].create_group("intruder")
        with pytest.raises(ConformanceError, match="non-dataset"):
            t.validate()

    def test_rule3_kindless_unreferenced_dataset(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        t.group[GROUP_SEARCH_INDEXES].create_dataset("stray", data=[1, 2, 3])
        with pytest.raises(ConformanceError, match="carries no KIND"):
            t.validate()

    def test_rule4_dangling_reference(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del t.group[GROUP_SEARCH_INDEXES][si.name]  # ref in the column dangles
        with pytest.raises(ConformanceError, match="unlinked|does not resolve"):
            t.validate()

    def test_rule4_null_reference(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        ds = t["ts"].dataset
        del ds.attrs[ATTR_SEARCH_INDEX_LIST]
        ds.attrs.create(
            ATTR_SEARCH_INDEX_LIST,
            np.array([h5py.Reference()], dtype=h5py.ref_dtype),
        )
        with pytest.raises(ConformanceError, match="null"):
            t.validate()

    def test_rule4_target_without_kind(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        si_group = t.group[GROUP_SEARCH_INDEXES]
        # A KIND-less dataset that IS a legitimate accompanying dataset of a
        # (foreign) index, so rule 3 is satisfied...
        acc = si_group.create_dataset("acc_values", data=np.arange(3))
        foreign = si_group.create_dataset("future", data=np.zeros(4))
        write_ascii_token_attr(foreign, ATTR_KIND, "FANCY_FUTURE_INDEX")
        write_uint64_attr(foreign, ATTR_SOURCE_GENERATION, 0)
        write_uint64_attr(foreign, ATTR_SOURCE_NROWS, t.nrows)
        references.write_ref_attr(foreign, ATTR_VALUES, acc)
        # ...but a column's SEARCH_INDEX_LIST must reference KIND-tagged
        # search-index datasets only (rule 4).
        references.append_ref_to_array_attr(
            t["temp"].dataset, ATTR_SEARCH_INDEX_LIST, acc
        )
        with pytest.raises(ConformanceError, match="not a search-index dataset"):
            t.validate()

    def test_rule12_missing_generation(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        del t.group.attrs[ATTR_GENERATION]
        with pytest.raises(ConformanceError, match="GENERATION"):
            t.validate()

    def test_rule12_wrong_generation_dtype(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        del t.group.attrs[ATTR_GENERATION]
        t.group.attrs.create(ATTR_GENERATION, np.int32(0))
        with pytest.raises(ConformanceError, match="uint64"):
            t.validate()

    def test_rule12_missing_source_token(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del si.dataset.attrs[ATTR_SOURCE_NROWS]
        with pytest.raises(ConformanceError, match="SOURCE_NROWS"):
            t.validate()

    def test_unknown_kind_is_tolerated(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        si_group = t.group[GROUP_SEARCH_INDEXES]
        foreign = si_group.create_dataset("future", data=np.zeros(4))
        write_ascii_token_attr(foreign, ATTR_KIND, "FANCY_FUTURE_INDEX")
        write_uint64_attr(foreign, ATTR_SOURCE_GENERATION, 0)
        write_uint64_attr(foreign, ATTR_SOURCE_NROWS, t.nrows)
        t.validate()  # unknown KIND: ignore the index, don't reject the table
        wrapped = t.search_indexes["future"]
        assert type(wrapped) is SearchIndex
        assert wrapped.kind == "FANCY_FUTURE_INDEX"

    def test_rule9_structural_wrong_n_field(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        entry = si.dataset[0]
        entry["n"] = 63  # lie about the chunk coverage; tokens stay valid
        si.dataset[0] = entry
        with pytest.raises(ConformanceError, match="chunk coverage"):
            t.validate()

    def test_rule9_deep_detects_wrong_bounds(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        entry = si.dataset[0]
        entry["max"] = 7  # wrong bound; structurally fine, tokens valid
        si.dataset[0] = entry
        t.validate()  # structural pass cannot see it
        with pytest.raises(ConformanceError, match="deep"):
            t.validate(deep=True)

    def test_rule9_skipped_for_stale_index(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        entry = si.dataset[0]
        entry["n"] = 63
        si.dataset[0] = entry
        # Invalidate the index: rule 9 exempts it, so validate passes again.
        t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        t.validate(deep=True)

    def test_orphan_index_is_conformant(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        name = t.add_search_index("ts").name
        del t["ts"].dataset.attrs[ATTR_SEARCH_INDEX_LIST]
        # Table-wide discovery scans SEARCH_INDEX_LIST: nothing claims it.
        assert t.search_indexes[name].column is None
        t.validate()  # useless but not non-conformant


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #
class TestApi:
    def test_unknown_column_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        with pytest.raises(KeyError):
            t.add_search_index("nope")

    def test_list_column_refused(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))],
        )
        with pytest.raises(SchemaError, match="list column"):
            t.add_search_index("r")

    def test_unimplemented_kind_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="not implemented"):
            t.add_search_index("ts", "CHUNK_BLOOM")

    def test_duplicate_name_raises(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.add_search_index("ts")
        with pytest.raises(SchemaError, match="already contains"):
            t.add_search_index("ts")

    def test_custom_name_and_description(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts", name="zoning", description="built by tests")
        assert si.name == "zoning"
        assert si.description == "built by tests"

    def test_column_side_accessors(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        col = t["ts"]
        assert col.search_indexes == []
        si = col.add_search_index()
        assert isinstance(si, ChunkMinMaxIndex)
        got = col.search_indexes
        assert len(got) == 1 and got[0].name == si.name
        assert si.column.name == "ts"

    def test_reopen_readonly_discovery_and_prune(self, h5path: Path) -> None:
        with h5py.File(h5path, "w") as f:
            t = make_table(f)
            t.add_search_index("ts")
        with h5py.File(h5path, "r") as f:
            t = Table.open(f["t"])
            si = t.search_indexes["ts__chunk_minmax"]
            assert isinstance(si, ChunkMinMaxIndex)
            assert si.is_valid
            assert si.prune("<", CHUNK).tolist() == [0]
            t.validate(deep=True)

    def test_reopen_refresh_after_stale(self, h5path: Path) -> None:
        with h5py.File(h5path, "w") as f:
            t = make_table(f)
            t.add_search_index("ts")
            t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        with h5py.File(h5path, "r+") as f:
            t = Table.open(f["t"])
            assert not t.search_indexes["ts__chunk_minmax"].is_valid
            assert t.refresh_indexes() == 1
            assert t.search_indexes["ts__chunk_minmax"].is_valid
            t.validate(deep=True)

    def test_vlen_string_column_refused_by_builder(self, h5file: h5py.File) -> None:
        # The spec orders vlen strings, but this producer does not build
        # min/max over them (building an index is optional for producers).
        g = h5file.create_group("foreign")
        write_ascii_token_attr(g, "CLASS", "COLUMN_TABLE")
        write_ascii_token_attr(g, "VERSION", "1.0")
        write_uint64_attr(g, "NROWS", 2)
        ds = g.create_dataset("s", data=["a", "b"], dtype=h5py.string_dtype())
        with pytest.raises(SchemaError, match="CHUNK_MINMAX"):
            indexes.create_chunk_minmax(g, ds)


# --------------------------------------------------------------------------- #
# Regressions from the Phase 4a adversarial spec review
# --------------------------------------------------------------------------- #
def make_table_with_foreign_vlen_index(f: h5py.File) -> Table:
    """A conformant table: supported int column + a foreign (hand-built) valid
    CHUNK_MINMAX index over a vlen-string column this builder cannot rebuild."""
    t = Table.create(
        f.create_group("t"),
        [ColumnSpec(name="x", dtype="i8", chunks=4)],
    )
    s = t.group.create_dataset(
        "s",
        shape=(0,),
        dtype=h5py.string_dtype(),
        maxshape=(None,),
        chunks=(4,),
        fillvalue="",
    )
    del t.group.attrs["column-order"]  # 's' was added behind the Table API
    t.append({"x": [1, 2, 3, 4], "s": ["a", "b", "c", "d"]})
    t.add_search_index("x")  # also creates GENERATION
    # Hand-build the vlen index the way a foreign producer would.
    sdt = h5py.string_dtype()
    cdt = np.dtype(
        [
            ("min", sdt),
            ("max", sdt),
            ("nan_count", "<u8"),
            ("fill_count", "<u8"),
            ("n", "<u8"),
        ]
    )
    si_group = t.group[GROUP_SEARCH_INDEXES]
    idx = si_group.create_dataset(
        "s__chunk_minmax", shape=(1,), maxshape=(None,), chunks=(64,), dtype=cdt
    )
    entry = np.zeros(1, dtype=cdt)
    entry["min"], entry["max"], entry["n"] = b"a", b"d", 4
    idx[...] = entry
    write_ascii_token_attr(idx, ATTR_KIND, "CHUNK_MINMAX")
    write_uint64_attr(idx, ATTR_SOURCE_GENERATION, t.generation)
    write_uint64_attr(idx, ATTR_SOURCE_NROWS, t.nrows)
    references.append_ref_to_array_attr(s, ATTR_SEARCH_INDEX_LIST, idx)
    return t


class TestReviewRegressions:
    def test_maintained_append_skips_unrebuildable_index(
        self, h5file: h5py.File
    ) -> None:
        # A conformant foreign CHUNK_MINMAX over a vlen-string column must not
        # make the append fail — and, crucially, must keep its ORIGINAL tokens
        # (a failed maintenance attempt must never clobber a valid index).
        t = make_table_with_foreign_vlen_index(h5file)
        foreign = t.search_indexes["s__chunk_minmax"]
        assert foreign.is_valid
        t.append({"x": [5], "s": ["e"]}, maintain_indexes=True)
        assert t.nrows == 5  # the append committed
        assert t.search_indexes["x__chunk_minmax"].is_valid
        assert not foreign.is_valid  # stale via the GENERATION bump...
        assert foreign.source_generation == 0  # ...with its tokens untouched
        assert foreign.source_nrows == 4
        t.validate()

    def test_refresh_all_skips_unrebuildable_index(self, h5file: h5py.File) -> None:
        t = make_table_with_foreign_vlen_index(h5file)
        t.append({"x": [5], "s": ["e"]})  # both indexes go stale
        assert t.refresh_indexes() == 1  # only the supported one is rebuilt
        assert t.search_indexes["x__chunk_minmax"].is_valid
        assert not t.search_indexes["s__chunk_minmax"].is_valid

    def test_deep_validate_skips_unrebuildable_index(self, h5file: h5py.File) -> None:
        # validate(deep=True) must not reject a conformant table just because
        # this builder has no oracle for the index's element dtype.
        t = make_table_with_foreign_vlen_index(h5file)
        t.validate(deep=True)

    def test_maintained_append_skips_non_growable_index(
        self, h5file: h5py.File
    ) -> None:
        # A conformant fixed-shape (non-resizable) foreign index cannot be
        # grown; maintenance must skip it without touching its tokens.
        t = make_table(h5file, n=CHUNK)  # exactly one chunk
        si = t.add_search_index("ts")
        si_group = t.group[GROUP_SEARCH_INDEXES]
        fixed = si_group.create_dataset(
            "ts__fixed",
            shape=(1,),
            dtype=si.dataset.dtype,  # contiguous
        )
        fixed[...] = si.dataset[...]
        write_ascii_token_attr(fixed, ATTR_KIND, "CHUNK_MINMAX")
        write_uint64_attr(fixed, ATTR_SOURCE_GENERATION, t.generation)
        write_uint64_attr(fixed, ATTR_SOURCE_NROWS, t.nrows)
        references.append_ref_to_array_attr(
            t["ts"].dataset, ATTR_SEARCH_INDEX_LIST, fixed
        )
        rows = {"ts": [9999], "temp": [1.0], "tag": ["x"], "ok": [True]}
        t.append(rows, maintain_indexes=True)  # now needs 2 entries
        assert t.nrows == CHUNK + 1
        assert si.is_valid
        wrapped = t.search_indexes["ts__fixed"]
        assert not wrapped.is_valid
        assert wrapped.source_generation == 0  # tokens untouched

    def test_prune_float_query_on_int64_is_exact(self, h5file: h5py.File) -> None:
        # NumPy would promote the int64 bounds to float64 and round them
        # (2**60 + 13 -> 2**60), silently dropping the matching chunk.
        t = Table.create(
            h5file.create_group("t"), [ColumnSpec(name="ts", dtype="i8", chunks=4)]
        )
        base = 2**60
        t.append({"ts": [base + 1, base + 5, base + 9, base + 13]})
        si = t.add_search_index("ts")
        assert si.prune(">", float(base)).tolist() == [0]
        assert si.prune("==", base + 5).tolist() == [0]

    def test_prune_float_query_rounding_up_direction(self, h5file: h5py.File) -> None:
        # The other rounding direction: vmin = 2**60 + 255 rounds UP to
        # 2**60 + 256 under float64, defeating "<".
        t = Table.create(
            h5file.create_group("t"), [ColumnSpec(name="ts", dtype="i8", chunks=4)]
        )
        v = 2**60 + 255
        t.append({"ts": [v, v, v, v]})
        si = t.add_search_index("ts")
        assert si.prune("<", float(2**60 + 256)).tolist() == [0]

    def test_prune_big_int_query_on_float_column_is_exact(
        self, h5file: h5py.File
    ) -> None:
        # Promotion in the other direction: an int query beyond 2**53 would be
        # rounded to float64 by NumPy; Python compares int vs float exactly.
        t = Table.create(
            h5file.create_group("t"), [ColumnSpec(name="v", dtype="f8", chunks=4)]
        )
        t.append({"v": [float(2**53)] * 4})
        si = t.add_search_index("v")
        assert si.prune("<", 2**53 + 1).tolist() == [0]
        assert si.prune(">", 2**53 + 1).size == 0

    def test_add_search_index_rejects_bad_names(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        with pytest.raises(SchemaError, match="link name"):
            t.add_search_index("ts", name="a/b")  # would create a subgroup
        with pytest.raises(SchemaError, match="reserved"):
            t.add_search_index("ts", name="NROWS")  # reserved-names rule 2
        assert GROUP_SEARCH_INDEXES not in t.group  # nothing left behind
        t.validate()

    @pytest.mark.parametrize(
        "bad_kind",
        [
            np.uint64(7),  # not a string
            np.array([b"CHUNK_MINMAX"], dtype="S13"),  # not scalar
            "CHUNK_MINMAX",  # h5py writes this as variable-length UTF-8
        ],
    )
    def test_validate_flags_malformed_kind(
        self, h5file: h5py.File, bad_kind: Any
    ) -> None:
        # KIND must be a scalar fixed-length ASCII string; anything else is a
        # conformance violation, not something to launder through str().
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del si.dataset.attrs[ATTR_KIND]
        si.dataset.attrs.create(ATTR_KIND, bad_kind)
        with pytest.raises(ConformanceError, match="KIND"):
            t.validate()

    def test_malformed_kind_is_never_dispatched(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del si.dataset.attrs[ATTR_KIND]
        si.dataset.attrs.create(ATTR_KIND, np.array([b"CHUNK_MINMAX"], dtype="S13"))
        assert indexes.index_kind(si.dataset) is None  # no str() mangling
        assert type(t.search_indexes[si.name]) is SearchIndex  # not minmax
        assert t.refresh_indexes() == 0  # and no maintenance path touches it

    def test_validate_scalar_search_index_list(self, h5file: h5py.File) -> None:
        # SEARCH_INDEX_LIST is a 1-D array by definition; a scalar reference
        # attribute must be a ConformanceError, not a raw TypeError.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        ds = t["ts"].dataset
        del ds.attrs[ATTR_SEARCH_INDEX_LIST]
        references.write_ref_attr(ds, ATTR_SEARCH_INDEX_LIST, si.dataset)
        with pytest.raises(ConformanceError, match="1-D"):
            t.validate()

    def test_prune_refuses_undersized_valid_index(self, h5file: h5py.File) -> None:
        # A token-valid index that does not cover every data-bearing chunk
        # (rule-9 violation) must raise, not silently clip: the uncovered
        # chunks could hold matches.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        si.dataset.resize((1,))  # tokens still valid
        with pytest.raises(ConformanceError, match="data-bearing"):
            si.prune("<", 10)

    def test_list_column_must_not_carry_search_index_list(
        self, h5file: h5py.File
    ) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [ListColumnSpec(name="r", values=LeafValuesSpec(dtype="f4"))],
        )
        t["r"].group.attrs.create(
            ATTR_SEARCH_INDEX_LIST, np.array([], dtype=h5py.ref_dtype)
        )
        with pytest.raises(ConformanceError, match="list column"):
            t.validate()

    def test_append_ref_failure_leaves_existing_refs(self, h5file: h5py.File) -> None:
        # The new reference is created before the old attribute is touched, so
        # a failure cannot drop the column's existing SEARCH_INDEX_LIST.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        ds = t["ts"].dataset
        with pytest.raises(ObjectReferenceError):
            references.append_ref_to_array_attr(
                ds,
                ATTR_SEARCH_INDEX_LIST,
                object(),  # not referenceable
            )
        assert len(t["ts"].search_indexes) == 1
        assert t["ts"].search_indexes[0].name == si.name

    def test_crash_window_rebuilt_index_half(self, h5file: h5py.File) -> None:
        # The OTHER half of the GENERATION/NROWS crash window: an index rebuilt
        # with future-valued tokens (append step 4) before the crash must fail
        # the check via SOURCE_NROWS/SOURCE_GENERATION — this window is exactly
        # why SOURCE_NROWS exists.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        g_old, n_new = t.generation, t.nrows + 1
        for name in ("ts", "temp", "tag", "ok"):  # step 2: extend every column
            extend_to(t[name].dataset, n_new)
        assert indexes.append_refresh_indexes(t.group, g_old, n_new)
        # "Crash" here: neither GENERATION nor NROWS was committed.
        assert not si.is_valid
        assert si.source_nrows == n_new  # future-valued, detectably so
        t.validate()  # the table itself is still conformant at the old state

    def test_unlinked_ref_skipped_by_consumer_flagged_by_validate(
        self, h5file: h5py.File
    ) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del t.group[GROUP_SEARCH_INDEXES][si.name]
        # Consumer path: the index is simply absent.
        assert t["ts"].search_indexes == []
        # Auditor path: rule 4 violation.
        with pytest.raises(ConformanceError):
            t.validate()


# --------------------------------------------------------------------------- #
# Regressions from the resumed (minmax-dimension) adversarial review
# --------------------------------------------------------------------------- #
class TestResumedReviewRegressions:
    def test_shared_index_rejected(self, h5file: h5py.File) -> None:
        # Spec: a single search-index dataset MUST NOT cover multiple columns.
        # Before the fix, validate passed and prune silently answered from the
        # FIRST claiming column's data (a false negative for the second).
        t = make_table(h5file)
        si = t.add_search_index("ts")
        references.append_ref_to_array_attr(
            t["temp"].dataset, ATTR_SEARCH_INDEX_LIST, si.dataset
        )
        with pytest.raises(ConformanceError, match="exactly one column"):
            t.validate()
        # The table-wide scan refuses to pick a column rather than guess.
        with pytest.raises(ConformanceError, match="multiple columns"):
            indexes.find_index_column(t.group, si.dataset)
        # Once the index is stale, refresh must refuse to guess a column too
        # (a valid index is skipped without needing its column resolved).
        t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        with pytest.raises(ConformanceError, match="multiple columns"):
            t.refresh_indexes()

    def test_scalar_list_on_sibling_does_not_break_healthy_index(
        self, h5file: h5py.File
    ) -> None:
        # A malformed scalar SEARCH_INDEX_LIST on column X must not crash
        # operations on column Y's perfectly valid index with a raw TypeError.
        t = make_table(h5file)
        si = t.add_search_index("temp")
        x = t["ts"].dataset
        references.write_ref_attr(x, ATTR_SEARCH_INDEX_LIST, si.dataset)  # scalar
        assert si.prune(">", 30.0).size >= 0  # no TypeError
        assert si.column is not None and si.column.name == "temp"
        t.append({"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]})
        assert t.refresh_indexes() == 1  # skips the malformed column
        with pytest.raises(ConformanceError, match="1-D"):
            t.validate()  # the auditor still flags the malformed attribute

    def test_prune_non_ascii_query_on_ascii_column(self, h5file: h5py.File) -> None:
        # Spec: ASCII columns are ordered as if they were UTF-8, so a
        # non-ASCII query is a well-defined byte-wise comparison — every
        # ASCII string sorts below b"\xc3\xa9".
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="s", dtype=FixedString(6, encoding="ascii"), chunks=4)],
        )
        t.append({"s": ["abc", "xyz", "AAA", "zz"]})
        si = t.add_search_index("s")
        assert si.prune("<", "é").tolist() == [0]
        assert si.prune(">", "é").size == 0

    def test_prune_works_on_foreign_vlen_index(self, h5file: h5py.File) -> None:
        # A conformant foreign CHUNK_MINMAX over a vlen-string column is
        # queryable: bounds read back as bytes and compare byte-wise.
        t = make_table_with_foreign_vlen_index(h5file)
        (si,) = [x for x in t["s"].search_indexes if x.kind == "CHUNK_MINMAX"]
        assert isinstance(si, ChunkMinMaxIndex)
        assert si.prune("==", "c").tolist() == [0]  # within [a, d]
        assert si.prune("==", b"z").size == 0  # above max

    def test_search_indexes_as_dataset_flagged(self, h5file: h5py.File) -> None:
        # Misusing the reserved name for a dataset is a conformance violation,
        # not an AttributeError; the consumer property sees no indexes.
        t = Table.create(
            h5file.create_group("t"), [ColumnSpec(name="x", dtype="i4", chunks=4)]
        )
        t.group.create_dataset(GROUP_SEARCH_INDEXES, data=[1, 2, 3])
        assert t.search_indexes == {}
        with pytest.raises(ConformanceError, match="group"):
            t.validate()

    def test_minmax_values_attr_cannot_vouch(self, h5file: h5py.File) -> None:
        # Only kinds that DEFINE an accompanying dataset (BITMAP) — or unknown
        # kinds — may vouch for a KIND-less sibling; CHUNK_MINMAX has none.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        junk = t.group[GROUP_SEARCH_INDEXES].create_dataset("junk", data=[1, 2])
        references.write_ref_attr(si.dataset, ATTR_VALUES, junk)
        with pytest.raises(ConformanceError, match="carries no KIND"):
            t.validate()

    def test_wrong_dtype_generation_repaired_on_refresh(
        self, h5file: h5py.File
    ) -> None:
        # A foreign int32 GENERATION previously survived every producer write
        # (attrs.modify keeps the dtype), so refresh_indexes reported success
        # while the index stayed invalid forever.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del t.group.attrs[ATTR_GENERATION]
        t.group.attrs.create(ATTR_GENERATION, np.int32(0))
        assert not si.is_valid  # strict check: wrong dtype = absent
        assert t.refresh_indexes() == 1
        assert si.is_valid  # repaired (as uint64, with a safe spurious bump)
        gen = np.asarray(t.group.attrs[ATTR_GENERATION])
        assert gen.dtype == np.uint64 and int(gen) == 1
        t.validate(deep=True)

    def test_wrong_dtype_generation_repaired_on_append(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        si = t.add_search_index("ts")
        del t.group.attrs[ATTR_GENERATION]
        t.group.attrs.create(ATTR_GENERATION, np.int32(0))
        t.append(
            {"ts": [1000], "temp": [1.0], "tag": ["x"], "ok": [True]},
            maintain_indexes=True,
        )
        assert np.asarray(t.group.attrs[ATTR_GENERATION]).dtype == np.uint64
        assert si.is_valid
        t.validate(deep=True)

    def test_prune_huge_int_query_is_answered(self, h5file: h5py.File) -> None:
        # Python ints beyond the uint64/int64 machine range are well-defined
        # predicates, not errors: every uint64 value is < 2**64.
        t = Table.create(
            h5file.create_group("t"), [ColumnSpec(name="u", dtype="u8", chunks=4)]
        )
        t.append({"u": [2**63, 1, 2, 3, 2**64 - 2, 5, 6, 7]})
        si = t.add_search_index("u")
        assert si.prune("<", 2**64).tolist() == [0, 1]
        assert si.prune(">", -(2**70)).tolist() == [0, 1]
        assert si.prune("==", 2**64).size == 0

    def test_crash_window_source_nrows_only(self, h5file: h5py.File) -> None:
        # The window SOURCE_NROWS exists for: index rebuilt (step 4) AND
        # GENERATION committed (step 5), but NROWS (step 6) never written.
        # The rebuilt index matches on SOURCE_GENERATION and must be caught
        # by the SOURCE_NROWS comparison alone.
        t = make_table(h5file)
        si = t.add_search_index("ts")
        g_old, n_new = t.generation, t.nrows + 1
        for name in ("ts", "temp", "tag", "ok"):
            extend_to(t[name].dataset, n_new)
        assert indexes.append_refresh_indexes(t.group, g_old, n_new)
        write_uint64_attr(t.group, ATTR_GENERATION, g_old + 1)  # step 5
        # "Crash" before step 6: NROWS still old.
        assert si.source_generation == t.generation  # this token matches...
        assert si.source_nrows != t.nrows  # ...so only SOURCE_NROWS disables
        assert not si.is_valid
        t.validate()


# --------------------------------------------------------------------------- #
# Adversarial-review regressions (4b review, CHUNK_MINMAX-related)
# --------------------------------------------------------------------------- #
class Test4bReviewRegressions:
    def test_dtype_mismatched_foreign_minmax_not_maintained(
        self, h5file: h5py.File
    ) -> None:
        # Review: maintenance wrote recomputed int64 entries into a foreign
        # index with int32 min/max fields; HDF5 silently CLAMPED the bounds
        # (2**40 became INT32_MAX) and the corrupted index was stamped valid.
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=4)],
        )
        t.append({"x": [2**40, 5, 7]})
        g = t.group
        gen = indexes.ensure_generation(g)
        sig = g.require_group(GROUP_SEARCH_INDEXES)
        dt32 = np.dtype(
            [
                ("min", "<i4"),
                ("max", "<i4"),
                ("nan_count", "<u8"),
                ("fill_count", "<u8"),
                ("n", "<u8"),
            ]
        )
        mm = sig.create_dataset(
            "x__mm32", shape=(1,), maxshape=(None,), chunks=(64,), dtype=dt32
        )
        mm[0] = (5, 5, 0, 0, 3)
        write_ascii_token_attr(mm, ATTR_KIND, "CHUNK_MINMAX")
        write_uint64_attr(mm, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(mm, ATTR_SOURCE_NROWS, 3)
        references.append_ref_to_array_attr(g["x"], ATTR_SEARCH_INDEX_LIST, mm)

        t.append({"x": [9]}, maintain_indexes=True)
        assert not indexes.index_is_valid(mm, g)  # skipped, went stale
        assert int(mm[0]["max"]) == 5  # content untouched, no clamped garbage
        assert indexes._scalar_uint64(mm.attrs, ATTR_SOURCE_NROWS) == 3

    def test_minmax_over_unorderable_column_rejected(self, h5file: h5py.File) -> None:
        # Review: validate checked orderability for SORTED_ROWS but not for
        # CHUNK_MINMAX, accepting an index the spec forbids (compound columns
        # have no H5Col-defined order).
        t = Table.create(
            h5file.create_group("t"),
            [ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=8)],
        )
        t.append({"x": [1, 2]})
        g = t.group
        del g.attrs["column-order"]
        comp = np.dtype([("re", "<f8"), ("im", "<f8")])
        z = g.create_dataset(
            "z",
            shape=(2,),
            maxshape=(None,),
            chunks=(8,),
            dtype=comp,
            fillvalue=np.zeros((), dtype=comp)[()],
        )
        gen = indexes.ensure_generation(g)
        sig = g.require_group(GROUP_SEARCH_INDEXES)
        mmdt = np.dtype(
            [
                ("min", comp),
                ("max", comp),
                ("nan_count", "<u8"),
                ("fill_count", "<u8"),
                ("n", "<u8"),
            ]
        )
        mm = sig.create_dataset("z__mm", shape=(1,), dtype=mmdt)
        write_ascii_token_attr(mm, ATTR_KIND, "CHUNK_MINMAX")
        write_uint64_attr(mm, ATTR_SOURCE_GENERATION, gen)
        write_uint64_attr(mm, ATTR_SOURCE_NROWS, 2)
        references.append_ref_to_array_attr(z, ATTR_SEARCH_INDEX_LIST, mm)
        with pytest.raises(ConformanceError, match="order"):
            t.validate()
