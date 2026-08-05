"""Tests for Table.truncate(): logical truncation and its index protocol."""

from __future__ import annotations

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
    indexes,
)
from h5col._hdf5 import write_uint64_attr
from h5col.reserved import ATTR_GENERATION, ATTR_NROWS
from h5col.strings import FixedString


def make_table(f: h5py.File, n: int = 20) -> Table:
    t = Table.create(
        f.create_group("t"),
        [
            ColumnSpec(name="x", dtype="i8", fill_value=-1, chunks=8),
            ColumnSpec(name="s", dtype=FixedString(6), fill_value="", chunks=8),
            ColumnSpec(name="ok", dtype=bool_dtype(), chunks=8),
        ],
    )
    t.append(
        {
            "x": np.arange(n, dtype="i8"),
            "s": [f"row{i}" for i in range(n)],
            "ok": [i % 2 == 0 for i in range(n)],
        }
    )
    return t


class TestTruncateBasics:
    def test_shrinks_logically_not_physically(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=20)
        t.truncate(7)
        assert t.nrows == 7
        assert t.group["x"].shape[0] == 20  # extents untouched
        got = t.read()
        assert got["x"].tolist() == list(range(7))
        assert got["s"].tolist() == [f"row{i}" for i in range(7)]
        t.validate()

    def test_truncate_to_zero(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.truncate(0)
        assert t.nrows == 0
        assert t.read()["x"].tolist() == []
        t.validate()

    def test_grow_and_negative_rejected(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=5)
        with pytest.raises(SchemaError, match="only shrinks"):
            t.truncate(6)
        with pytest.raises(SchemaError, match="negative"):
            t.truncate(-1)
        assert t.nrows == 5

    def test_equal_is_a_noop(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=5)
        si = t.add_search_index("x")
        gen = t.generation
        t.truncate(5)
        assert t.nrows == 5
        assert t.generation == gen  # nothing changed, nothing published
        assert si.is_valid

    def test_no_generation_created_without_indexes(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.truncate(3)
        assert ATTR_GENERATION not in t.group.attrs

    def test_reappend_overwrites_reserved_tail(self, h5file: h5py.File) -> None:
        t = make_table(h5file, n=10)
        t.truncate(4)
        t.append({"x": [100, 101], "s": ["a", "b"], "ok": [True, False]})
        assert t.nrows == 6
        got = t.read()
        assert got["x"].tolist() == [0, 1, 2, 3, 100, 101]
        assert got["s"].tolist() == ["row0", "row1", "row2", "row3", "a", "b"]
        assert t.group["x"].shape[0] == 10  # never shrank
        t.validate()

    def test_list_column_truncates_logically(self, h5file: h5py.File) -> None:
        t = Table.create(
            h5file.create_group("t"),
            [
                ColumnSpec(name="x", dtype="i4", fill_value=-1, chunks=8),
                ListColumnSpec(
                    name="r", values=LeafValuesSpec(dtype="f4"), nullable=True
                ),
            ],
        )
        t.append({"x": [1, 2, 3], "r": [[1.0, 2.0], None, [3.0]]})
        t.truncate(2)
        got = t.read()["r"]
        assert len(got) == 2
        assert list(got[0]) == [1.0, 2.0] and got[1] is None
        t.validate()


class TestTruncateIndexProtocol:
    def _with_indexes(self, h5file: h5py.File) -> Table:
        t = make_table(h5file, n=20)
        t.add_search_index("x")  # CHUNK_MINMAX
        t.add_search_index("x", "SORTED_ROWS")
        t.add_search_index("ok")  # BITMAP
        return t

    def test_default_leaves_indexes_detectably_stale(self, h5file: h5py.File) -> None:
        t = self._with_indexes(h5file)
        gen = t.generation
        t.truncate(9)
        assert t.generation == gen + 1
        assert all(not si.is_valid for si in t.search_indexes.values())
        t.validate(deep=True)  # stale indexes are exempt, never an error
        assert t.refresh_indexes() == 3
        assert all(si.is_valid for si in t.search_indexes.values())
        t.validate(deep=True)

    def test_maintain_keeps_all_kinds_valid(self, h5file: h5py.File) -> None:
        t = self._with_indexes(h5file)
        t.truncate(9, maintain_indexes=True)
        assert all(si.is_valid for si in t.search_indexes.values())
        t.validate(deep=True)
        sr = t.search_indexes["x__sorted_rows"]
        assert sorted(sr.permutation().tolist()) == list(range(9))
        bm = t.search_indexes["ok__bitmap"]
        assert bm.rows(True).tolist() == [0, 2, 4, 6, 8]

    def test_crash_window_generation_written_nrows_not(self, h5file: h5py.File) -> None:
        # Simulate the crash between steps 5 and 6: GENERATION advanced,
        # NROWS still old. Every index must fail the validity check.
        t = self._with_indexes(h5file)
        gen = t.generation
        assert gen is not None
        write_uint64_attr(t.group, ATTR_GENERATION, gen + 1)
        assert all(not si.is_valid for si in t.search_indexes.values())
        t.validate(deep=True)

    def test_crash_window_maintained_tokens_before_commit(
        self, h5file: h5py.File
    ) -> None:
        # Simulate the crash inside step 4 of a maintained truncation: the
        # future-valued tokens are written but neither GENERATION nor NROWS
        # advanced. The index must fail the validity check on both counts.
        t = self._with_indexes(h5file)
        gen = t.generation
        assert gen is not None
        si = t.search_indexes["x__sorted_rows"]
        write_uint64_attr(si.dataset, "SOURCE_GENERATION", gen + 1)
        write_uint64_attr(si.dataset, "SOURCE_NROWS", 9)
        assert not si.is_valid
        t.validate(deep=True)

    def test_repairs_missing_generation_when_maintaining(
        self, h5file: h5py.File
    ) -> None:
        t = self._with_indexes(h5file)
        del t.group.attrs[ATTR_GENERATION]  # foreign tool dropped it
        t.truncate(9, maintain_indexes=True)
        # ensure_generation repaired GENERATION above every source token, the
        # maintained rebuild then wrote matching future-valued tokens.
        assert t.generation is not None
        assert all(si.is_valid for si in t.search_indexes.values())
        t.validate(deep=True)

    def test_truncate_then_data_change_detected(self, h5file: h5py.File) -> None:
        # A truncate that is later "undone" by an append of different data
        # must not resurrect the old index content.
        t = make_table(h5file, n=10)
        si = t.add_search_index("x", "SORTED_ROWS")
        t.truncate(5)
        t.append({"x": [50, 40, 30, 20, 10], "s": ["a"] * 5, "ok": [True] * 5})
        assert t.nrows == 10  # same NROWS as when the index was built
        assert not si.is_valid  # but two GENERATION bumps disable it
        assert t.nrows == si.source_nrows  # NROWS alone would falsely match


class TestTruncateNrowsAttr:
    def test_nrows_written_as_uint64(self, h5file: h5py.File) -> None:
        t = make_table(h5file)
        t.truncate(3)
        val = np.asarray(t.group.attrs[ATTR_NROWS])
        assert val.shape == () and val.dtype == np.uint64


class TestGenerationRepair:
    def test_absent_generation_repair_cannot_validate_stale_index(
        self, h5file: h5py.File
    ) -> None:
        # A foreign tool deletes GENERATION and mutates data behind the
        # index's back. The repair must pick a value above every existing
        # SOURCE_GENERATION, so the (now inaccurate) index stays disabled.
        t = make_table(h5file, n=6)
        si = t.add_search_index("x")
        assert si.source_generation == 0
        del t.group.attrs[ATTR_GENERATION]
        t.group["x"][0] = 999  # in-place mutation, undetectable by tokens
        gen = indexes.ensure_generation(t.group)
        assert gen == 1  # not the reused 0
        assert not si.is_valid

    def test_malformed_generation_repair_clears_all_tokens(
        self, h5file: h5py.File
    ) -> None:
        # The malformed old value may be LOWER than some index's token; the
        # repair must clear the tokens too, not just bump the old value.
        t = make_table(h5file, n=6)
        si = t.add_search_index("x")
        write_uint64_attr(si.dataset, "SOURCE_GENERATION", 7)
        del t.group.attrs[ATTR_GENERATION]
        t.group.attrs.create(ATTR_GENERATION, np.int32(3))
        gen = indexes.ensure_generation(t.group)
        assert gen == 8  # above both the malformed 3 and the token 7
        assert not si.is_valid

    def test_malformed_generation_never_incremented_verbatim(
        self, h5file: h5py.File
    ) -> None:
        # Review: append/truncate read g_old leniently, so a malformed
        # int32 GENERATION was incremented as-is; g_old + 1 could equal an
        # index's forged/residue SOURCE_GENERATION and, combined with a
        # matching SOURCE_NROWS, spuriously validate unverified content.
        t = make_table(h5file, n=4)
        si = t.add_search_index("x")
        del t.group.attrs[ATTR_GENERATION]
        t.group.attrs.create(ATTR_GENERATION, np.int32(0))
        write_uint64_attr(si.dataset, "SOURCE_GENERATION", 1)
        write_uint64_attr(si.dataset, "SOURCE_NROWS", 5)
        t.append({"x": [99], "s": ["z"], "ok": [True]})
        assert t.nrows == 5  # SOURCE_NROWS now matches...
        assert not t.index_is_valid(si)  # ...but the generation cannot

        # and the truncate path takes the same strict read
        del t.group.attrs[ATTR_GENERATION]
        t.group.attrs.create(ATTR_GENERATION, np.int32(0))
        gen_attr = indexes.mutation_generation(t.group)
        assert gen_attr is not None and gen_attr > 1  # repaired above tokens
