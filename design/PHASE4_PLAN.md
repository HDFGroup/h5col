# Phase 4 implementation plan — Search indexes

Status: **planned** (not started). Written 2026-07-11 after Phases 0–3 landed.
This document is the detailed design to pick up later; nothing here is committed
until the phase is built and passes the usual gate (tests + ruff + mypy) and an
adversarial H5Col spec-review.

Spec anchors (in `H5Col-convention.md`): §"Search indexes" 1353–1792;
§"Index validity tokens" 1826–1952; §"How consumers interpret NROWS" 1801–1824;
§"Appending rows" 1955–2044; §"Truncation" 2061–2085; §"In-place updates"
2086–2115; consistency rules 3, 4, 9, 12 (2125–2176).

---

## 1. Objective and scope

Search indexes are **derivative, recomputable** data that accelerate queries; a
conformant consumer may ignore all of them and still be correct (spec 1355–1358).
That framing sets the priorities for a reference implementation:

1. **Validity-token machinery** — `GENERATION` / `SOURCE_GENERATION` /
   `SOURCE_NROWS`, the write-ordering rules, and the consumer validity check.
   This is the backbone every index family depends on, and it changes the
   `append` protocol and `validate`. **Highest priority — build first.**
2. **Index construction + validation + linkage** — create an index over a
   column, link it via `SEARCH_INDEX_LIST`, keep it consistent across appends
   (or let it fall stale detectably), and validate all of it (rules 3, 4, 9, 12).
3. **Query execution using an index** — optional for conformance. A minimal
   "use the index to answer a predicate" path is valuable for a proof-of-concept
   and to *prove the indexes are correct*, but it is not required by the spec.

Per the project guidance ("search indexes belong in a later phase … their
support can be considered optional compared to other more important features"),
we **sub-phase** this and commit to `CHUNK_MINMAX` first; the other three
families are individually gated (see §10, and the open questions in §12).

Non-goals for Phase 4: query planner/optimizer, multi-column indexes (spec
forbids a single index dataset covering multiple columns, 1517–1519), indexes
over list columns (spec forbids this revision, 1250–1251), reference/compound/
array/vlen column indexing (spec excludes per family).

---

## 2. Where it plugs into the existing architecture

| Existing piece | Change in Phase 4 |
|---|---|
| `Table.append` | Add spec steps 4–5: for maintained indexes write future-valued tokens **then** content; write `GENERATION = g_old + 1` (only if the table carries `GENERATION`); `NROWS` commit stays last. Two `H5Fflush` points (after step 4, after step 6) already exist for the data path — extend to indexes. |
| `Table.validate` | Add rules 3, 4, 9 (partial), 12. New: `SEARCH_INDEXES` contents, `KIND` presence/absence, `SEARCH_INDEX_LIST` resolution, `GENERATION`/token datatypes. |
| `Table` | New API: `add_search_index(...)`, `search_indexes` discovery, `generation` property, and a consumer-side `index_is_valid(ds)` helper. |
| `Column` | New: `search_indexes` property (resolve `SEARCH_INDEX_LIST`); `add_search_index(...)` convenience. |
| `_hdf5.py` | Helpers for compound-dtype datasets, 2-D unlimited-first-dim index datasets, `uint16`/`uint32` scalar attrs. |
| `reserved.py` | Already has `GROUP_SEARCH_INDEXES`, `ATTR_GENERATION`, `ATTR_SOURCE_GENERATION`, `ATTR_SOURCE_NROWS`, `ATTR_SEARCH_INDEX_LIST`, `ATTR_VALUES`, `ATTR_KIND`, `SEARCH_INDEX_KINDS`, `KIND_*`. Add: `ATTR_ORDERED` (have it), `nan_tail_length`, `fill_tail_length`, `exhaustive`, `k`, `m_bits`, `hash_family`, `seed` tokens; `HASH_FAMILY_MURMUR3`. |

New modules:

- `src/h5col/ordering.py` — the H5Col canonical **order** for a column dtype
  (used by `CHUNK_MINMAX` and `SORTED_ROWS`): byte-wise string comparison with
  padding stripped, IEEE-754 with NaN handling, enum-by-code, boolean-by-code.
  Also the **orderability predicate** (which dtypes may carry these indexes).
- `src/h5col/canonical.py` — the Bloom **canonical byte representation** per
  dtype (little-endian ints/floats, UTF-8 strings padding-stripped, bool 0x00/
  0x01, enum-by-code), plus negative-zero normalization and NaN exclusion.
- `src/h5col/hashing.py` — `MurmurHash3_x64_128` (seeded) + Kirsch–Mitzenmacher
  double hashing. See §9 for the dependency decision.
- `src/h5col/indexes.py` — the index-family engine: spec types, build, append/
  refresh, validate, and (optional) query. Mirrors `lists.py`'s file-driven
  style so indexes work on reopened tables.
- `src/h5col/searchindex.py` — read-friendly `SearchIndex` wrapper classes
  (`ChunkMinMaxIndex`, `SortedRowsIndex`, `BitmapIndex`, `ChunkBloomIndex`),
  each exposing `.kind`, `.is_valid`, `.column`, and family-specific accessors.

Import DAG stays acyclic: `ordering`/`canonical`/`hashing` depend only on
primitives; `indexes` depends on those + `_hdf5`/`references`/`reserved`;
`table` depends on `indexes`/`searchindex`.

---

## 3. Validity tokens — the backbone (build in sub-phase 4a)

Data model (spec 1836–1905):

- `GENERATION` — scalar `uint64` on the **table group**. REQUIRED iff the table
  has ≥1 search-index dataset; MAY be absent otherwise. Compared by **equality
  only** (never ordered). Initial value RECOMMENDED `0`.
- `SOURCE_GENERATION`, `SOURCE_NROWS` — scalar `uint64` on **each** index dataset.
- Consumer check: `SOURCE_GENERATION == GENERATION AND SOURCE_NROWS == NROWS`;
  if either fails, or any of the four attrs is absent/wrong-dtype, treat the
  index **as absent** (not an error, never reject the table).

Producer obligations (spec 1842–1863):

- Increment `GENERATION` by 1 on any op that (1) modifies an element in
  `[0, NROWS)` of any column (incl. list-column members) or (2) commits a change
  to `NROWS` (append/truncate). Schema ops (add/remove/rename column) do **not**
  require an increment (index linkage is by object reference).

Implementation:

- `Table` gains `generation -> int | None` and an internal `_bump_generation()`
  used by `append`/`truncate`/in-place update. The bump is a no-op (attribute
  omitted) when the table has no indexes — but once any index exists, every
  mutating op must maintain it. Cleanest rule: **`GENERATION` is created the
  first time an index is added**, and thereafter maintained.
- `index_is_valid(index_ds) -> bool`: reads the four attrs, checks dtypes are
  scalar `uint64`, returns the equality check. Public and used by every consumer
  path.

Write-ordering (spec 1907–1940) — two opposite orders:

- **NROWS-gated mutations (append, truncation):** for each maintained index,
  write the *future* `SOURCE_GENERATION = g_old + 1` and `SOURCE_NROWS = N_new`
  **before** rewriting the index content; then (step 5) write
  `GENERATION = g_old + 1`; then (step 6) commit `NROWS`. A crash mid-rebuild
  leaves the index failing the check (its `SOURCE_GENERATION` names a generation
  the table does not yet have). `H5Fflush` after step 4 and after step 6.
- **Immediately-visible mutations (in-place update):** commit
  `GENERATION += 1` **and flush** *before* touching column data (disabling all
  indexes up front), then rewrite each index's content, then rewrite that
  index's tokens to the current values.

`append` integration (extends the existing method):

```
1. read N_old, g_old(=generation or None)
2. extend + write column data + list columns          (already implemented)
3. flush                                               (already implemented)
4. if any maintained index: for each, write future tokens then rebuild/append content; flush
5. if table carries GENERATION: write GENERATION = g_old + 1
6. write NROWS = N_new; flush                          (already implemented)
```

Indexes the producer chooses **not** to maintain are simply left untouched — the
validity check disables them. This is the escape hatch that keeps `append` cheap
when a user has not asked for index maintenance: default `append` may leave
indexes stale (documented), or we expose `append(..., maintain_indexes=True)`.
**Decision needed (see §12 Q3).**

Consistency rule 12: when ≥1 index dataset exists, the table MUST carry
`GENERATION` (uint64) and every index MUST carry `SOURCE_GENERATION` +
`SOURCE_NROWS` (uint64). `validate` enforces this.

---

## 4. `SEARCH_INDEXES` group, linkage, and discovery (sub-phase 4a)

- `SEARCH_INDEXES` is a direct child group of the table group holding **every**
  index dataset + their accompanying datasets and **no other objects**
  (rule 3, spec 1362–1370). Datasets may have **any** name — linkage is by
  object reference, never by parsed name (spec 1384–1397). Our writer will use
  readable `<col>__<kind>` names as a convention but the reader/validator MUST
  NOT depend on them.
- A dataset is an **index** iff it carries `KIND`; an **accompanying** dataset
  (e.g. a `BITMAP`'s `VALUES`) MUST NOT carry `KIND` (rule 3).
- Column linkage: `SEARCH_INDEX_LIST` on the *column* dataset — 1-D array of
  object references into `SEARCH_INDEXES`. No back-pointer from index to column;
  to find an index's column, scan columns for the one whose `SEARCH_INDEX_LIST`
  references it (spec 1372–1382).
- Discovery: `Table.search_indexes -> dict[str, SearchIndex]` enumerates
  `SEARCH_INDEXES` children with a `KIND`, wraps each. `Column.search_indexes`
  resolves that column's `SEARCH_INDEX_LIST`.
- `_discover_columns` already skips `SEARCH_INDEXES` (in `_SKIP_CHILDREN`) — good.

Validate additions:
- Rule 3: every dataset in `SEARCH_INDEXES` either carries `KIND` (index) or is
  referenced as an accompanying dataset (e.g. a bitmap `VALUES`); no other
  objects; accompanying datasets carry no `KIND`.
- Rule 4: every ref in a column's `SEARCH_INDEX_LIST` resolves to a `KIND`-tagged
  dataset under `SEARCH_INDEXES`.
- Rule 9: an index whose validity check **passes** must correctly describe rows
  `[0, NROWS)`; tail entries (≥ NROWS) may exist and are ignored. Full semantic
  re-derivation is expensive — see §12 Q4 for how deep `validate` goes.

---

## 5. Shared ordering + orderability (`ordering.py`, sub-phase 4a)

Used by `CHUNK_MINMAX` and `SORTED_ROWS`. H5Col-defined order (spec 1549–1604):

- Signed/unsigned ints: arithmetic order.
- Floats: IEEE-754 over finite + ±inf; `-0.0 == +0.0`; **NaN unordered** →
  goes to a NaN tail (SORTED_ROWS) / excluded from min/max (CHUNK_MINMAX).
- Boolean: by code, FALSE(0) < TRUE(1).
- Strings (fixed + vlen): **byte-wise** over UTF-8, NUL/space trailing padding
  stripped, no normalization, no BOM; ASCII treated as UTF-8.
- Opaque: raw-byte lexicographic (tag excluded).
- Enum: by underlying integer code.

Orderability predicate — **excluded**: object/region refs, compound, array,
vlen-array (spec 1586–1593). `CHUNK_MINMAX` and `SORTED_ROWS` MUST refuse these.

Missing handling threads through: `min`/`max` computed over non-missing,
orderable elements only, using the canonical missing test already in
`missing.is_missing`.

---

## 6. `CHUNK_MINMAX` (sub-phase 4a — the first committed family)

Spec 1453–1527.

- **Shape:** 1-D, length = data-bearing chunk count of the source column:
  `ceil(NROWS / chunk_len)` for chunked, `1` for contiguous, `0` when
  `NROWS = 0`. Tail-only chunks (`[NROWS, extent)`) are not indexed.
- **Datatype:** compound, fields in order:
  `min`, `max` (source element dtype), `nan_count` (uint64), `fill_count`
  (uint64), `n` (uint64).
- **Semantics:** per chunk, `min`/`max` over non-missing + orderable elements;
  floats exclude NaN and non-NaN-fill matches; other types exclude fill matches.
  Empty chunk (`fill_count + nan_count == n` for floats, `fill_count == n`
  otherwise) → set `min = max = fill` as placeholders and mark via the counts;
  consumers must not use placeholders. `n` = logical rows in the chunk (last
  data-bearing chunk may be < chunk_len).
- **Attrs:** `KIND="CHUNK_MINMAX"` (+ common tokens; + optional `description`).
- **Build:** read the source column's chunk_len from its DCPL; iterate chunks
  over `[0, NROWS)`; compute the five fields; write the compound array.
- **Append/refresh:** `CHUNK_MINMAX` supports **incremental** append — recompute
  only the last previously-partial chunk + new chunks (spec 1985 allows append).
  Simpler first cut: rebuild all data-bearing chunks (correct, O(NROWS)); note
  the incremental optimization as a follow-up.
- **Query (optional):** range/equality predicate → list of candidate chunks to
  scan (prune chunks whose `[min,max]` can't overlap; never prune placeholder
  chunks). This is the cheapest family to demonstrate end-to-end.

h5py notes: compound with the column's element dtype as `min`/`max` fields; build
the numpy structured dtype via `np.dtype([("min", elt), ("max", elt),
("nan_count","u8"), ("fill_count","u8"), ("n","u8")])`. For a fixed-string
column, `elt` is the h5py string dtype — verify structured-dtype + h5py write
works for string fields (probe before building; likely needs the h5py string
dtype inside the structured dtype).

---

## 7. `SORTED_ROWS` (sub-phase 4b)

Spec 1529–1619.

- **Shape:** 1-D, length `NROWS`; **Datatype:** unsigned int wide enough for all
  rows (default `uint64`).
- **Semantics:** element `i` = row `r` at rank `i` under the H5Col order; ties
  broken by increasing `r` (total, deterministic). Non-NaN-fill rows go to a
  **fill tail** (increasing `r`) immediately before the **NaN tail** (increasing
  `r`). When fill is NaN, all missing rows are NaN rows → NaN tail;
  `fill_tail_length = 0`.
- **Attrs:** `KIND="SORTED_ROWS"`; `nan_tail_length`, `fill_tail_length`
  (uint64, both required); `ordered` (bool, MUST be true).
- **Build:** stable argsort under the ordering key, partitioning out
  fill-tail / NaN-tail rows first. Use a key function from `ordering.py`
  (byte-wise for strings — argsort on the raw padded-stripped bytes).
- **Append/refresh:** no efficient incremental update — **rebuild** (spec 1987).
- **Query (optional):** binary search for equality/range → contiguous slice of
  the permutation (excluding tails).

---

## 8. `BITMAP` (sub-phase 4b) and `CHUNK_BLOOM` (sub-phase 4c)

### BITMAP — spec 1622–1672

- **Shape:** 2-D `(K, ceil(NROWS/8))`, `K` = distinct values indexed.
- **Datatype:** `uint8`; bit `r%8` of byte `r/8` of row `k` set iff value at row
  `r` equals the `k`-th indexed value. Trailing pad bits (`≥ NROWS`) MUST be 0.
- **Accompanying `VALUES` dataset:** sibling 1-D dataset (source dtype) holding
  the `K` values; linked from the bitmap by a scalar `VALUES` object-ref attr;
  MUST NOT carry `KIND`.
- **Attrs:** `KIND="BITMAP"`, `VALUES` (ref), `ordered` (bool), `exhaustive`
  (bool). `exhaustive=true` ⇒ values enumerate every distinct non-missing value
  in `[0, NROWS)` (query value absent ⇒ provably zero rows).
- **Build:** natural fit for categorical/boolean/low-cardinality columns; `K`
  from distinct values (or category set). Pack bits little-endian within bytes.
- **Append/refresh:** rebuild (K and packing shift with NROWS).

### CHUNK_BLOOM — spec 1675–1791 (most complex; sub-phase 4c)

- **Shape:** 2-D `(n_chunks, m_bytes)` (n_chunks like `CHUNK_MINMAX`).
- **Datatype:** `uint8`; each row = one chunk's packed Bloom bits; bit `g` at
  bit `g%8` of byte `g/8`; `m_bits = 8*m_bytes`.
- **Hashing:** Kirsch–Mitzenmacher double hashing
  `h_i = (h_a + i*h_b) mod m_bits`, `i=0..k-1`; `h_a`=low64, `h_b`=high64 of a
  single seeded `MurmurHash3_x64_128` over the value's **canonical byte
  representation** (`canonical.py`). NaN never inserted/queried; `-0.0`
  normalized to `+0.0`; missing values excluded.
- **Attrs:** `KIND="CHUNK_BLOOM"`, `k` (uint16), `m_bits` (uint64),
  `hash_family="murmur3_x64_128_double"` (ASCII), `seed` (uint32, default 0).
- **Build:** choose `m_bits`/`k` from a target false-positive rate + expected
  distinct-per-chunk (expose as params). Per chunk, insert each non-missing
  value's canonical bytes.
- **Query (optional):** for an equality predicate, test the value's bits per
  chunk → candidate chunk list (Bloom gives no-false-negatives).

The interop risk here is the **byte-exact** hash. Cross-check `MurmurHash3` and
the canonical encoding against a known-answer vector (see §9, §11).

---

## 9. Dependencies: MurmurHash3

`CHUNK_BLOOM` requires `MurmurHash3_x64_128` seeded with a 32-bit seed and the
exact canonical byte encoding. Options, in order of preference:

1. **`mmh3`** (conda-forge, widely used C++ MurmurHash3 binding). Exposes
   `mmh3.hash128(bytes, seed, x64=True, signed=False)` → the 128-bit value; split
   into low/high 64. Pin it and add a known-answer test. **Recommended** — avoids
   a hand-rolled hash whose bugs would silently break interop.
2. Vendored pure-Python MurmurHash3_x64_128 (~60 lines) if we want zero runtime
   deps for the Bloom path. More risk; needs the same KAT.

Decision: default to `mmh3` behind `hashing.py` so the rest of the code is
agnostic; if the user prefers no new dependency, vendor it. This only affects
sub-phase 4c and can be deferred with the Bloom family itself.

---

## 10. Sub-phasing and deliverables

| Sub-phase | Deliverable | Committed? |
|---|---|---|
| **4a** | Validity tokens (`GENERATION`/`SOURCE_*`, `index_is_valid`, write-ordering in `append`), `SEARCH_INDEXES` group + linkage + discovery, `ordering.py`, rules 3/4/12 in `validate`, **`CHUNK_MINMAX`** build/append/validate/query. | Yes — first commit of the phase. |
| **4b** | `SORTED_ROWS` and `BITMAP` (build/validate; rebuild-on-append), rule 9 depth for these. | Yes — second commit. |
| **4c** | `CHUNK_BLOOM` + `canonical.py` + `hashing.py` (+ `mmh3`), KAT for hashing. | Optional/gated — third commit or deferred. |

Each sub-phase follows the established loop: probe h5py unknowns → implement +
tests → gate (pytest/ruff/mypy) → adversarial H5Col spec-review (find→verify) →
fix + regression tests → commit.

Rationale for the order: 4a delivers the machinery + one useful family and is the
part with real cross-cutting changes (append/validate). 4b adds two
straightforward families. 4c is isolated (its own hashing/canonical modules) and
carries the interop risk, so it is last and separately gated.

---

## 11. Testing strategy

- **Round-trip per family:** build over a known column, reopen from disk,
  re-derive the answer a brute-force scan gives, assert equality (the index must
  agree with a naive scan on random predicates).
- **Validity tokens:** append without maintaining an index ⇒ `index_is_valid`
  false; maintain ⇒ true; simulate a crash between `GENERATION` and `NROWS`
  (write `GENERATION` but not `NROWS`) ⇒ all indexes invalid; assert `validate`
  still passes (stale index ≠ error) and reads are correct at old `NROWS`.
- **Ordering:** byte-wise string order incl. multibyte UTF-8 and padding
  stripping; float NaN/`-0.0`; enum-by-code; boolean order.
- **CHUNK_MINMAX:** placeholder chunks (all-missing), partial last chunk `n`,
  `nan_count` vs `fill_count` overlap when fill is NaN, contiguous vs chunked.
- **SORTED_ROWS:** tie-breaking by `r`; fill tail + NaN tail lengths and
  disjointness; fill == NaN case.
- **BITMAP:** trailing pad bits zeroed; `exhaustive`/`ordered` semantics; K from
  categorical category set.
- **CHUNK_BLOOM:** **known-answer test** for `MurmurHash3_x64_128` (fixed input/
  seed → fixed 128-bit output) and for the canonical encoding of each dtype; no
  false negatives over a random workload; NaN never inserted.
- **validate:** rules 3 (foreign object in `SEARCH_INDEXES`, accompanying
  dataset carrying `KIND`), 4 (dangling `SEARCH_INDEX_LIST` ref), 12 (index
  present but `GENERATION`/tokens missing or wrong dtype).

---

## 12. Open design questions (resolve before/at the start of the phase)

- **Q1 — Which families to commit?** 4a (`CHUNK_MINMAX`) is in scope for sure.
  Are `SORTED_ROWS`/`BITMAP` (4b) wanted now, and is `CHUNK_BLOOM` (4c) in scope
  this phase or deferred to a later revision? (The spec makes all four optional
  for consumers; the project plan calls index support "optional compared to
  other features".)
- **Q2 — Query execution? [RESOLVED 2026-07-11]** Yes — build the analyst-facing
  query layer (details in Appendix A):
  - **Scope:** Layer 1 primitives (per family) **+** Layer 2 native selection via
    tuple filters. No string/expression query language in v1.
  - **Result forms:** all three — `t.select([...]) -> row positions`,
    `t.read(where=[...], columns=[...]) -> dict`, and a lazy `Selection` object
    (the composable core; the other two are conveniences over it).
  - **Predicate scope (v1):** comparisons, `==`, `in`, between — combined with
    **AND** only. OR / negation deferred.
  - **Auto index build:** `t.build_index(col)` auto-picks the family
    (`CHUNK_MINMAX` for orderable numeric, `BITMAP` for low-cardinality
    categorical/boolean, `CHUNK_BLOOM` for high-cardinality); explicit `kind=`
    overrides.
  - **Observability:** include `explain=True` (index used, chunks pruned, rows
    scanned).
  - **Bridges:** build the `h5col ↔ pyarrow` bridge (Appendix B); defer the
    pandas convenience wrappers (`df.h5col.save` / `read_pandas`) to later.
- **Q3 — Index maintenance on append.** Default `append` behavior when indexes
  exist: (a) auto-maintain all indexes every append (simple, correct, can be
  slow — full rebuild for SORTED_ROWS/BITMAP); (b) leave them stale by default
  and expose `refresh_indexes()` / `append(maintain_indexes=True)`; (c)
  auto-maintain only the incrementally-cheap families (`CHUNK_MINMAX`,
  `CHUNK_BLOOM`) and mark the rest stale. Recommendation: (b) — explicit, matches
  the spec's "indexes the producer does not maintain MAY be left untouched", and
  keeps the hot append path fast. `validate` and `index_is_valid` make staleness
  safe and visible.
- **Q4 — `validate` depth for rule 9.** Cheap structural validation (shape,
  dtype, tokens, `KIND`) always; full semantic re-derivation (does the index
  actually match the column?) only under an opt-in `validate(deep=True)`, since
  it is O(index build). Recommendation: structural by default, `deep=` for the
  semantic check — reuse the build functions as the oracle.
- **Q5 — MurmurHash3 dependency** (`mmh3` vs vendored). Only affects 4c. See §9.
- **Q6 — Truncation / in-place update.** Phase 4 must at least *define*
  `GENERATION` behavior for these (rule: any `[0,NROWS)` element change or NROWS
  change bumps `GENERATION`). Do we implement `truncate()` / in-place `update()`
  in this phase (they are separate spec sections 2061–2115) or keep them as a
  small Phase 4.5? Recommendation: implement `truncate()` here (it's short and
  interacts with tokens); defer general in-place update unless wanted.

---

## 13. Risks

- **Bloom interop** is the sharpest risk: any deviation in `MurmurHash3`, seed
  handling, bit packing, or canonical bytes silently breaks cross-implementation
  reads. Mitigation: KAT vectors, `uint8` storage (no byte-swap), freeze
  `hash_family`.
- **Compound dtype with string fields** (`CHUNK_MINMAX` over a fixed-string
  column) — verify h5py writes a structured dtype whose `min`/`max` are h5py
  string dtypes; probe first, and if awkward, restrict `CHUNK_MINMAX` string
  support or store min/max via the fixed-string handler.
- **Append cost** if indexes auto-maintain (rebuild for SORTED_ROWS/BITMAP).
  Addressed by Q3 recommendation (b).
- **`GENERATION` lifecycle** — subtle: created on first index add, maintained on
  every subsequent mutation. A missed bump would let a stale index look valid.
  Covered by the crash-window tests in §11.

---

## Appendix A — Analyst-facing exposure: query layers & Layer-1 performance

How an analyst actually *uses* the indexes, and how to build the primitives so
the acceleration is real. **Decisions settled (2026-07-11, see §12 Q2):** ship
Layer 1 primitives + Layer 2 tuple-filter selection; expose all three result
forms (row positions / materialized dict / lazy `Selection`); v1 predicates are
comparisons + `==` + `in`/between combined with AND only; `build_index` auto-
picks the family; `explain=True` is included; build the Arrow bridge (Appendix B)
and defer the pandas convenience wrappers.

### The correctness contract that anchors everything

> **The query result is defined by the data, never by the index. Indexes only
> change *speed*.**

Three invariants the API must enforce invisibly:

- **Validity-gated** — every index use first passes the `GENERATION`/`SOURCE_*`
  check; a stale index is silently ignored and the query falls back to a scan.
- **Exact vs. pruning** — `SORTED_ROWS`/`BITMAP` give the *exact* matching rows;
  `CHUNK_MINMAX`/`CHUNK_BLOOM` give only *candidate chunks* (a superset) that
  must be read and re-tested. The analyst gets the exact answer either way.
- **Scan fallback** — a predicate on an unindexed column just scans; the query
  always works, an index only makes it faster.

This is also the test strategy: the accelerated path must return byte-identical
results to a brute-force scan.

### Three exposure layers

1. **Layer 1 — primitives** (per family): predicate → candidate rows/chunks.
   As implemented (4a/4b): `ChunkMinMaxIndex.prune(op, value)`,
   `SortedRowsIndex.rows(op, value)`, `BitmapIndex.rows(value)` /
   `BitmapIndex.isin(values)` (`ChunkBloomIndex.candidate_chunks(value)` deferred
   with 4c). The substrate *and* the test oracle. **Essential regardless of
   what's above.**
2. **Layer 2 — native selection** composing primitives transparently. Built in
   4d with full pyarrow parity (see docs/PHASE4D_PLAN.md): DNF tuple filters
   `t.read(where=[("temp_c", ">", 30.0)], columns=[...])` /
   `t.select(where=...) -> Selection` (`.row_positions`), and fluent `field()`
   expressions with `& | ~`. AND/OR/NOT all shipped in v1 (Kleene semantics).
3. **Layer 3 — bridges & sugar**: `to_pandas`/Arrow (Appendix B); optionally a
   string query language (deferred — the `field()` expression API shipped in 4d).

Predicate → index:

| Predicate | Index | Exact or pruning |
|---|---|---|
| range (`< <= > >=`, between) | `SORTED_ROWS` | exact |
| range | `CHUNK_MINMAX` | pruning (chunks) |
| equality (`==`), low-cardinality | `BITMAP` | exact |
| equality | `SORTED_ROWS` | exact |
| equality, high-cardinality | `CHUNK_BLOOM` | pruning (chunks) |
| set (`in […]`) | `BITMAP` OR / repeated equality | exact / pruning |
| no valid index | — | full scan |

### Layer-1 performance model

Layer 1 is the **decision** engine, not the work engine. The gain is **data
skipping** — the I/O and decompression it lets you *avoid* — not Layer 1's own
CPU, which runs over small index structures (KB–few MB), not the GB column. A
query's cost is roughly `read index (tiny) + read & decompress the SURVIVING
chunks (dominant) + vectorized verify`. Layer 1 shrinks the middle term.

Two levers, neither of which is Layer 1's raw speed:

1. **Selectivity** — a tight candidate set. For pruning indexes this is mostly a
   *data-layout* property: `CHUNK_MINMAX` is a zone map that prunes ~everything
   on a clustered/sorted column (timestamps, monotonic ids) and *nothing* on
   random data — so writing data clustered by the filter columns is the biggest
   lever. `CHUNK_BLOOM` selectivity is the `m_bits`/`k` false-positive tuning.
2. **Read shape** — turn candidate rows into **chunk-aligned, coalesced** reads
   (`pos // chunk_len` → distinct chunk ids → merge adjacent → contiguous
   hyperslab reads; in the cloud, merge candidate-chunk byte ranges into a few
   large GETs via `get_chunk_info`). HDF5's I/O granularity is the chunk;
   scattered per-row fancy-indexing defeats the purpose.

Techniques to keep Layer 1 itself cheap:

- **Vectorize in NumPy — no Python per-row/per-chunk loops.** `CHUNK_MINMAX`: one
  vectorized min/max comparison → chunk mask. `BITMAP`: packed bitwise AND/OR to
  combine predicates, `unpackbits` only at the end. `CHUNK_BLOOM`: precompute the
  value's `k` bit positions once, vectorized bit-test across all chunks.
  `SORTED_ROWS`: `np.searchsorted` for range bounds, then **sort the returned
  positions before gathering** to coalesce reads (caveat: the permutation stores
  positions, not values, so each search probe dereferences `column[perm[mid]]` —
  `O(log N)` scattered reads; cache them).
- **Read each (small) index once**; adequate `rdcc` chunk cache so verifying
  adjacent candidates hits cache (ties into the cache-aware chunking work).
- **Cost-model gate** — use the index's own cheap stats (`BITMAP` popcount = exact
  match count; `CHUNK_MINMAX` survivor estimate) to choose index-vs-scan and skip
  the index for non-selective predicates (a 60%-selectivity filter gains nothing
  from pruning).
- **Parallel verify** of surviving chunks is embarrassingly parallel (HDF5/NumPy
  release the GIL) — nice-to-have, not v1.

Reference-implementation stance: absolute throughput is bounded by NumPy + h5py,
so the goal is that the architecture **doesn't preclude the fast path** —
produce chunk-aligned candidates, vectorize, and expose selectivity stats.
Micro-optimizing the index evaluation itself would be optimizing the already-
negligible term.

---

## Appendix B — Arrow & pandas interop

Goal: give a data analyst `df.h5col.save(...)` + `h5col.read_pandas(...)`
ergonomics, and a broad ecosystem, without H5Col becoming a query engine.

### Mechanism reality — pandas has no I/O-format plugin

`read_parquet`/`to_parquet`/… are hardcoded; the only in-format pluggability is
an *engine*. There is **no registry** that makes `pd.read_h5col` a first-class
method. (For contrast: pandas' only entry-point plugin is the *plotting* backend;
**xarray** *does* have a real `xarray.backends` entry-point system.) So:

| Side | Idiomatic, supported | Notes |
|---|---|---|
| write | `df.h5col.save(path, …)` via `@pd.api.extensions.register_dataframe_accessor("h5col")` | ✅ sanctioned |
| read | `h5col.read_pandas(path, columns=…, where=…)` | ✅ ordinary function (lives on `h5col`, not `pd`) |
| exact names | `h5col.register_pandas()` → `pd.read_h5col`, `df.to_h5col` | ⚠️ monkeypatch; unsupported/fragile; explicit opt-in only |

### The real work is the type mapping

| pandas | → H5Col | Friction |
|---|---|---|
| int/float/bool (numpy) | numeric / boolean column | clean |
| nullable `Int64`/`Float64`/`boolean` | numeric + fill; boolean can't be missing | **nullable boolean has no home** (the `MASK`-on-column mechanism is reserved/future) |
| `object`/`string[…]` | `FixedString(max_bytes)` | **variable-length strings** are the main mismatch (H5Col scalar strings are fixed-length) |
| `category` (ordered) | categorical column | clean, 1:1 (ordered preserved) |
| `datetime64[ns]`, tz | `int64` epoch + unit/tz attribute | persist unit + tz in attrs |
| `object` of lists/dicts | list column / two aligned list columns | H5Col's strength, but untyped in pandas → heuristic or opt-in schema |
| index / MultiIndex | `INDEX_COLUMNS` (+ `_index`, `column-order`) | maps well; RangeIndex → drop & regenerate |
| non-string / MultiIndex column names | UTF-8 link names | stringify / flatten |

### Recommended architecture: go through Arrow

Build an **`h5col ↔ pyarrow`** bridge and let pandas fall out of it:

- `h5col.to_arrow(table, columns=…, where=…) -> pa.Table`; `from_arrow(pa_table)`.
- H5Col list columns *already* use the Arrow offsets layout; strings/categoricals
  map to Arrow `large_string`/`dictionary` → **faithful** round-trip (especially
  lists and variable-length strings, sidestepping the fixed-length compromise).
- pandas comes for free (`pa.Table.to_pandas()` / `ArrowDtype`), and so do
  **polars / DuckDB / DataFusion**. `df.h5col.save` / `read_pandas` become thin
  wrappers over the Arrow bridge.

### Two payoffs that make it a "backend", not an exporter

1. **Predicate + column pushdown** — `read_pandas(path, columns=[…], where=[…])`
   using the Phase-4 indexes: the H5Col analog of `read_parquet(filters=…,
   columns=…)`. This is the reason to be a backend.
2. **Out-of-core reality** — pandas is eager/in-memory, so a full read defeats
   H5Col's point; lean on `columns=`/`where=` subsetting and an optional
   `chunksize=` iterator, and point truly big scans at polars/DuckDB over the
   Arrow bridge.

### Sequencing

**Decided (2026-07-11):** build the `h5col ↔ pyarrow` bridge; defer the
pandas-specific convenience wrappers (`df.h5col.save` / `read_pandas`) for later.
The **Arrow bridge is independent of Phase 4** — it can land earlier and
immediately gives a faithful DataFrame (and polars/DuckDB) round-trip for the
columns we already support. Index **pushdown** (`where=`) slots into `to_arrow`
once Phase 4's indexes exist; the deferred pandas wrappers then layer on top.

