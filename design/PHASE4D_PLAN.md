# Phase 4d implementation plan — Analyst-facing query layer (Appendix A, Layer 2)

Completes the committed Q2 scope (PHASE4_PLAN.md §12 Q2, RESOLVED 2026-07-11):
Layer 1 primitives shipped in 4a/4b; this phase builds **Layer 2 — native
selection composing those primitives transparently**, plus `explain`. It comes
*before* phase 5 (docs) because it is committed phase-4 scope and adds the
largest remaining slice of public API — the surface phase 5 must document.

## 1. Objective and scope

Give an analyst a correct, index-accelerated way to select and read row subsets
with predicate filters, without exposing the index machinery.

**In scope (v1) — decisions locked 2026-07-13:**
- **pyarrow-parity syntax** (see §3). Two accepted forms:
  - **DNF tuple filters** (parquet/pyarrow idiom): `List[Tuple]` = AND;
    `List[List[Tuple]]` = OR-of-ANDs. Tuple ops: `= == != < > <= >= in "not in"`.
  - **fluent expressions**: `field("x") > 30`, combined with `& | ~`, plus
    `.isin(iterable)`, `.is_null()`, `.is_valid()` — pyarrow's exact spelling.
- **Full boolean algebra: AND, OR, NOT** (user override of the AND-only
  recommendation — see the "expanded scope" note below). Both input forms
  normalize to one internal boolean tree.
- **No `between`** (write `(field("x") >= lo) & (field("x") <= hi)`, as in
  pyarrow). Presence only via `is_null`/`is_valid` (pyarrow has no tuple-form
  null predicate) — the old `missing`/`present` ops are dropped.
- **Three-valued (Kleene) missing-value semantics** — see §4a. Load-bearing
  correctness decision; it also *matches Arrow/pyarrow's own null propagation*,
  so results agree with pyarrow, not merely with SQL.
- Three result forms, all over one lazy core:
  - lazy `Selection` (the composable core),
  - row positions (`Selection.row_positions` / `Table.select(...).row_positions`),
  - materialized dict (`Selection.read(columns=...)` / `Table.read(where=..., columns=...)`).
- `explain` — the DNF plan, index used per leaf, chunks pruned, rows scanned,
  exact-vs-pruning, scan-forced branches flagged, final count.
- Automatic planning via DNF: each AND-term uses exact index → exact rows;
  pruning index → candidate chunks → read+verify; no valid index → scan+verify;
  disjuncts are unioned.

**Expanded scope note (user decision 2026-07-13):** OR/NOT/`!=` were recommended
*out* of v1 because they break monotone data-skipping (OR widens; an unindexed
disjunct forces a full column scan), force full three-valued logic, require a
predicate tree (Layer-3 structure), make acceleration family-dependent
(BITMAP-OR is cheap, SORTED_ROWS/CHUNK_MINMAX are not), and multiply the test
surface with no conformance payoff (HEP001 mandates no query language). The user
accepted these costs; this plan contains them via one explicit semantics
(§4a), a DNF planner that degrades *honestly* and reports it in `explain`, and a
term-count guard against DNF blow-up.

**Out of scope (deferred, stated explicitly):**
- String/expression query language / parser (Layer 3 sugar) — the combinators
  build the tree in Python; no text DSL.
- Arrow/pandas bridges (Appendix B — separate phase).
- Auto-*building* indexes from inside a query (see §7 — a firm stance, not a gap).
- CHUNK_BLOOM planning paths (4c, deferred) — the planner treats a bloom index
  the same as "no exact index for this leaf": such a column simply scans or uses
  its other indexes.

## 2. The correctness contract (the whole point)

> The query result is defined by the data, never by the index. Indexes only
> change speed.

Three invariants enforced invisibly (Appendix A):
- **Validity-gated** — every index use passes the `GENERATION`/`SOURCE_*` check
  first (the primitives already raise `StaleIndexError`; the planner catches it
  and falls back to a scan, never surfacing it to the analyst).
- **Exact vs. pruning** — `SORTED_ROWS`/`BITMAP` yield exact rows; `CHUNK_MINMAX`
  yields candidate *chunks* that must be read and re-tested. Analyst gets the
  exact answer either way.
- **Scan fallback** — an unindexed (or stale-indexed, or bitmap-non-exhaustive-miss)
  predicate just scans. The query always works; an index only makes it faster.

**Test oracle:** a private brute-force `_scan_select(where)` that ignores every
index and computes the mask by reading full columns and applying the predicates
in the exact-comparison domain. Every query test asserts
`select(where).row_positions == _scan_select(where)` — byte-identical — across
{no index, valid index per family, deliberately-staled index} configurations.
This is the central test strategy, not an afterthought.

## 3. Public API (proposed)

```python
# lazy core
sel = t.select(where=[("temp_c", ">", 30.0), ("station", "==", "A17")])  # AND
sel.row_positions          # -> np.ndarray[int64], sorted ascending, unique
sel.count                  # -> int (len without materializing columns)
len(sel)                   # == sel.count
sel.read(columns=None)     # -> {name: array} for the surviving rows (all cols by default)
sel.explain()              # -> QueryPlan (machine-readable + pretty __str__)

# conveniences over the core
t.read(where=[...], columns=[...])   # == t.select(where=...).read(columns=...)
t.select(where=[...]).row_positions  # the "row positions" form

# pyarrow DNF tuple form — outer list = OR, inner lists = AND
#   (temp_c > 30 AND station == 'A17') OR (qc is null)
t.select(where=[[("temp_c", ">", 30.0), ("station", "==", "A17")],
                [("qc", "not in", [0, 1])]])

# pyarrow fluent form — field() + operator overloading, exact pyarrow spelling
from h5col import field
expr = (field("temp_c") > 30.0) & (field("station") == "A17") | ~field("qc").is_valid()
t.select(where=expr)                 # where= accepts an Expression, a List[Tuple], or List[List[Tuple]]
field("x").isin([1, 2, 3]); field("x").is_null(); field("x").is_valid()
```

`where=` accepts three shapes, all normalized to one internal boolean tree:
an `Expression` (from `field()`), a `List[Tuple]` (AND), or a `List[List[Tuple]]`
(OR-of-ANDs, pyarrow DNF). Tuple ops: `= == != < > <= >= in "not in"` (no
`between`; presence via the fluent `is_null()`/`is_valid()` — pyarrow has no
tuple null op). `field()` returns an `Expression` overloading `< <= > >= == !=`,
`& | ~`, and exposing `.isin()`, `.is_null()`, `.is_valid()`. Values are encoded
through the existing `_encode_query_value` (searchindex.py) — string→UTF-8, exact
int/float, NaN→SchemaError, categorical label→code — so leaf semantics match the
Layer-1 primitives exactly.

## 4a. Missing-value semantics — three-valued (Kleene/SQL)

The load-bearing decision the expanded scope forces. A leaf predicate evaluates
per row to one of **TRUE / FALSE / UNKNOWN**:

- A **value** leaf (`< <= > >= == != in "not in"`) is **UNKNOWN** on a *missing*
  row, else the ordinary boolean over the present value. (So `== v` and `!= v`
  are *both* UNKNOWN on missing rows — `!=` never resurrects missing rows.)
- `is_valid()` is TRUE on present rows, FALSE on missing; `is_null()` is the
  inverse. These are the *only* leaves that are ever TRUE/FALSE (never UNKNOWN)
  on a missing row — the only way to select on missingness.
- Kleene combinators: `NOT UNKNOWN = UNKNOWN`; `TRUE OR UNKNOWN = TRUE`;
  `FALSE AND UNKNOWN = FALSE`; etc.
- **A row is selected iff the whole expression evaluates to TRUE** (UNKNOWN and
  FALSE are both rejected).

This is **exactly Arrow/pyarrow's null propagation** — a comparison on a null
input yields null and the scanner drops the row; `~null` is still null, still
dropped — so H5Col query results agree with pyarrow row-for-row, not merely with
SQL. Consequences that keep it coherent: every value predicate excludes missing
rows regardless of negation; De Morgan holds under Kleene; the brute-force oracle
implements *exactly* this table over full-column reads, so the accelerated path
is tested against it. (Two-valued semantics — missing→FALSE, so `NOT` flips
missing to matching — was rejected as surprising *and* as diverging from
pyarrow; revisit only on explicit request.)

## 4. The planner (`src/h5col/query.py`)

Over `nrows = t.nrows`, for the internal predicate **tree** (parsed from an
`Expression`, a `List[Tuple]`, or a `List[List[Tuple]]` — all three converge here):

1. **Normalize** the tree: push `Not` down to leaves via De Morgan (a negated
   `And`/`Or` flips; `Not(Not x) = x`), turning it into **DNF** — a disjunction
   of AND-terms of (possibly negated) leaves. (`List[List[Tuple]]` input is
   already DNF.) A **term-count guard** aborts with a clear `SchemaError` if DNF
   expansion exceeds a cap (pathological nesting), rather than blowing up memory.
2. **Validate** every leaf: column exists (`KeyError`), not a list column
   (`SchemaError`), op known (`SchemaError`), value shape matches the op; encode
   value(s) via `_encode_query_value`.
3. **Plan each AND-term** with monotone data-skipping (this is where indexes pay
   off, and only AND-terms have the narrowing property):
   - Classify each leaf by the best *valid* index on its column
     (`Column.search_indexes` filtered by `is_valid`; bound wrappers, so a
     non-conformant shared index still answers for its own column):
     - **exact** — `SORTED_ROWS` (range/eq) or `BITMAP` (eq/`in`) → exact TRUE-row
       set. BITMAP `rows`/`isin` returning `None` (non-exhaustive miss) demotes
       the leaf to *scan*. A **negated** leaf is exact only on an *exhaustive*
       BITMAP (complement within present rows); otherwise it is a scan leaf.
     - **pruning** — `CHUNK_MINMAX` (range/eq) → candidate chunk ids.
     - **scan** — no valid usable index (incl. `is_null`/`is_valid`, `!=`/`not in`
       without an exhaustive bitmap, and negated ranges).
   - Intersect all **exact** leaves' TRUE-row sets first (no column reads) →
     `survivors` (or "all rows" if the term has no exact leaf).
   - For each **pruning**/**scan** leaf, verify the Kleene truth value *only for
     `survivors`*, reading those rows' values **chunk-aligned and coalesced**
     (`pos // chunk_len` → distinct chunk ids → merged contiguous hyperslabs);
     keep rows where the leaf is TRUE. Pruning indexes first restrict which
     chunks are candidates.
4. **Union** the AND-terms' survivor sets (OR). *Honesty about cost:* a term
   containing a scan leaf reads that column over its (possibly un-narrowed)
   candidate set; OR cannot share narrowing across terms. `explain` flags every
   such scan-forced branch.
5. **Return** the union as sorted unique int64 row positions.

Exactness/verify rules:
- Exact leaves are **trusted** once validity-gated (no re-verify) — their
  contract; the oracle tests guarantee it.
- Pruning/scan leaves are **always verified** against real values under the
  Kleene table (§4a).
- Every value comparison uses the object-domain / byte-wise discipline already
  proven for the primitives (no NumPy float promotion of int64/uint64; trailing
  NUL/space handling for fixed strings; categorical via codes).

`Selection` holds `(table, expr)`, computes `row_positions` lazily and caches it;
`read`/`count`/`explain` build on it. It captures `nrows`/`generation` at first
evaluation only for `explain` reporting — results are recomputed from current
data if re-evaluated after a mutation (documented: a `Selection` is a query, not
a snapshot).

## 5. `explain`

`sel.explain()` returns a `QueryPlan` dataclass (pretty `__str__`):
- the normalized **DNF structure** (the AND-terms and their union).
- per leaf: column, op, value; resolution (`index=<name> kind=<KIND>` / `scan`);
  exact-or-pruning; negated?; for pruning: `chunks_pruned/chunks_total`; rows
  verified. **Scan-forced branches are flagged** so lost acceleration is visible.
- overall: `nrows`, final `matched` count, evaluation order.

`t.read(where=..., explain=True)` / `t.select(where=..., explain=True)` return
`(result, QueryPlan)`; `Selection.explain()` is always available regardless.

## 6. `build_index` (Q2 deliverable) — already satisfied, plus an alias

Q2 asked for `t.build_index(col)` auto-picking the family. `add_search_index(col,
kind=None)` already auto-picks (`BITMAP` for bool/categorical, `CHUNK_MINMAX`
otherwise; `CHUNK_BLOOM` deferred). Deliverable: add `Table.build_index` /
`Column.build_index` as a thin alias for the analyst vocabulary, documented as
the same operation. No new behavior.

## 7. Firm stance: queries never build indexes

A query only *reads* existing valid indexes and otherwise scans. It never
creates or refreshes an index — building mutates the file, bumps nothing but is
an explicit act with cost, and silent index creation during a read would violate
least-surprise and the write-once/read-often model. `explain` may *suggest*
"column X scanned N rows; consider `build_index('X')`", but never acts.

## 8. Files

- **new** `src/h5col/query.py` — `field()` → `Expression` (overloads
  `< <= > >= == != & | ~`, `.isin()`, `.is_null()`, `.is_valid()`); internal
  `And`/`Or`/`Not`/leaf nodes; parsers for `List[Tuple]` and `List[List[Tuple]]`;
  DNF normalization + term-count guard; Kleene evaluator; `Selection`;
  `QueryPlan`; the planner; chunk-aligned coalesced reader; brute-force
  `_scan_select` oracle (Kleene over full-column reads).
- `src/h5col/table.py` — `select(where=...)`, extend `read(where=..., columns=...,
  explain=...)`, `count(where=...)` (optional), `build_index` alias.
- `src/h5col/column.py` — `build_index` alias (optional).
- `src/h5col/__init__.py` — export `Selection`, `QueryPlan`, `field` (internal
  `And`/`Or`/`Not` nodes stay private).
- `tests/test_query.py` — oracle differential tests (central), all three input
  forms (Expression / List[Tuple] / List[List[Tuple]]) agree, explain tests,
  stale-index fallback, AND/OR/NOT combinations, three-valued/missing/NaN
  semantics (esp. `!=`/`not in` and `NOT` vs missing) **cross-checked to match
  pyarrow's null propagation**, De Morgan equivalences, string/categorical/bool
  predicates, `in`/`not in`, `is_null`/`is_valid`, negated-leaf-on-exhaustive-
  bitmap exact path, empty result, empty `where` (all rows), unindexed column,
  DNF term-count guard, list-column rejection, shared-index-on-non-conformant-file.

## 9. Deliverables / sub-steps

1. Detail confirmed (this doc) + reconcile Appendix A primitive names in
   PHASE4_PLAN.md (`.search()`→`.rows()`, `.rows_equal()`→`.rows()`/`.isin()`).
2. Implement `query.py` (planner + Selection + QueryPlan + oracle).
3. Wire Table (`select`, `read(where=)`, `explain`, `build_index` alias).
4. Tests — oracle-differential first, then explain/edge cases.
5. Gate: `pixi run -e dev pytest` + `ruff check` + `ruff format --check` + `mypy`.
6. Adversarial review workflow (find→verify) focused on the correctness contract:
   exact-comparison discipline, missing-row exclusion, stale/None fallback, AND
   intersection correctness, chunk-aligned verify reads, list-column/shared-index
   edge cases.
7. Fix confirmed findings; commit on explicit user request.

## 10. Design decisions — RESOLVED 2026-07-13

- **Q(4d)-A — `select()` return + `explain` delivery. [lazy Selection]**
  `select()` returns the lazy `Selection`; row positions via `.row_positions`;
  `explain=True` returns `(result, QueryPlan)`, `Selection.explain()` always
  available.
- **Q(4d)-B — presence predicates. [via `is_null`/`is_valid`]** Fluent-form only
  (pyarrow parity — no tuple null op). The earlier `missing`/`present` ops are
  dropped.
- **Q(4d)-C — `!=` / OR / negation. [ALL INCLUDED — user override]** Full boolean
  algebra in v1 against the recommendation to defer; costs accepted and contained
  per the expanded-scope note (§1) and the Kleene semantics (§4a).
- **Q(4d)-D — missing-value semantics. [Kleene three-valued]** §4a: value leaves
  are UNKNOWN on missing rows, select-iff-TRUE. Matches Arrow/pyarrow null
  propagation. Two-valued rejected.
- **Q(4d)-E — repr/nrows polish placement. [stays in phase 5]** Unrelated surface.
- **Q(4d)-F — syntax fidelity. [FULL PYARROW PARITY]** DNF tuple filters
  (`List[Tuple]` = AND, `List[List[Tuple]]` = OR) + fluent `field()` with
  `& | ~`, `.isin()`, `.is_null()`, `.is_valid()`. No `between`; ops
  `= == != < > <= >= in "not in"`.
