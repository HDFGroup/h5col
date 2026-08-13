# Changelog

All notable changes to H5Col are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-11

### Added

- **Columns read by subscript.** `column[17:98]`, `column[-1]`,
  `column[[3, 1, 3]]` and `column[mask]` are `read_rows` with its defaults, so
  a row selection reads the way it does in h5py and NumPy. An integer key
  returns that row's value on its own — `numpy.ma.masked` when the row is
  missing — and every other key returns an array. Subscript has nowhere to put
  a keyword, so it always decodes and always masks; `read` and `read_rows`
  remain the forms that take `masked=False`.

  `len(column)` is the row count and iterating a column yields its decoded
  rows, reading the column once rather than once per row. Note that having a
  length makes a column with no rows falsy, as it does for a list or an h5py
  dataset, so `if column:` now asks whether the column has rows rather than
  whether the object exists.

  List columns accept the same keys, and gained `read_rows` to match.

  Note that `column[...]` and `column.dataset[...]` are not the same read. The
  second goes straight to h5py, so it skips decoding, ignores missing values,
  and can return rows above `NROWS` that `truncate` left behind as reserved
  storage.

- **List columns read only the rows asked for.** A list column's rows are
  reached through its `OFFSETS`, so reading a range means looking up where
  that range's values start and end and narrowing the child read to that span
  — recursively, so a nested list or a `STRING_VALUES` group narrows too.
  Scattered positions are served from the range that spans them, which is
  never wider than the column itself: rows near each other cost almost
  nothing, and rows at opposite ends cost what reading the column costs.

  Reading 50 rows of a 200,000-row list column pulls 251 values rather than a
  million, taking 1.5 ms rather than 79 ms. A single row costs 6 values.
  `Selection.read` no longer sends list columns down the read-whole path, so a
  query matching a few rows of a table with a list column went from 84 ms to
  1.6 ms, of which the list column is now a rounding error.

  `to_arrow` is unchanged and still reads the whole column: it saves no
  reading but skips building a Python list per row, which suits wanting most
  of a large column, where a range read suits wanting a small part of one.

- **Row selections take slices, boolean masks and negative positions.**
  `Column.read_rows`, `Column.to_arrow` and everything built on them accept a
  slice (`read_rows(slice(17, 98))`), a boolean array with one entry per row
  (`read_rows(col.is_missing())`), and positions counting back from the end
  (`read_rows([-1, -2])`) alongside the integer sequences they already took.

  A slice is read as a single hyperslab rather than going through the gather
  path, which had to sort the positions and scatter the result back into the
  caller's order — work that is pure overhead when the positions were a range
  to begin with. Reading a million contiguous rows from a two-million-row
  column drops from 6.9 ms to 1.3 ms, which is what the same read costs in
  h5py directly. Asking for half a column is now cheaper than asking for all
  of it; previously it was three times more expensive.

- Example notebook [Reading part of a
  table](https://hdfgroup.github.io/h5col/notebooks/08_reading_rows.html):
  subscript and `read_rows` over a 200,000-row table, with every HDF5 read
  counted so the saving is shown rather than asserted. Scalar columns fetch the
  chunks their rows land in — two rows at opposite ends cost two chunks, not
  the span; list columns are served from the range that spans the wanted rows,
  where reading fifty of them costs 251 values against a million. Also covers
  masked results from subscript, `masked=False` via `read_rows`, and the
  difference between `column[...]` and `column.dataset[...]`.

- **Every parameter of every public callable is documented.** An audit of the
  source found 114 public functions and methods whose docstring left at least
  one parameter of its signature undescribed — 207 parameters in all, of which
  86 were on callables the API reference renders. `Table.create` documented
  none of its ten; `Table.read` and `Column.read_rows` documented `masked` but
  not the parameters that decide what gets read. All of them now have a
  `Parameters` section covering the whole signature.

  Ruff's D417, which catches a `Parameters` section that covers only part of a
  signature, is switched off by the numpy docstring convention. It is now named
  explicitly in the lint configuration so it runs, and the gate fails if a
  parameter description goes missing again.

### Changed

- The version is declared in one place. `pyproject.toml` now takes it from
  `src/h5col/__init__.py` rather than repeating it, so the two cannot drift.
  A `release-check` workflow runs on `v*` tags and fails when the
  commit a tag points at does not report the version the tag claims, or when a
  release tag carries a pre-release version.
- The missing-value documentation no longer uses "sentinel". H5Col's marker for
  a missing row is simply the column's fill value. The reference pages keep the
  convention's own term, "the canonical missing-value test".
- The [Reading into
  Python](https://hdfgroup.github.io/h5col/guide/reading-into-python.html)
  chapter gained a "Reading part of a column" section, so row selection is no
  longer buried in a list of limitations, and its Arrow half is now titled to
  match its NumPy half rather than reading as the chapter's conclusion.

### Fixed

- A boolean mask passed to `read_rows` selected the wrong rows. The mask was
  cast to an integer dtype, turning it into a run of ones and zeros, so a mask
  marking rows 3, 7 and 9 read rows 1 and 0 over and over. A mask now selects
  the rows it marks, and one of the wrong length raises `IndexError` rather
  than being reinterpreted.
- Non-integer row positions were truncated silently — `read_rows([1.5])` read
  row 1. They now raise `TypeError`.

## [0.2.0] - 2026-08-10

### Added

- **Arrow export** — `Table.to_arrow()`, `Column.to_arrow()` and
  `Selection.to_arrow()`, behind the optional `pyarrow` dependency
  (`pip install h5col[arrow]`). This is the one representation that carries the
  whole H5Col data model, because NumPy has no type for three of the things
  H5Col stores:

  - missing rows become real Arrow nulls, so the fill value never reaches a
    consumer as data;
  - a categorical becomes a `DictionaryArray` of exactly the codes and labels
    already on disk, rather than being expanded to one label per row;
  - a list column keeps its nulls at every level of nesting, including an inner
    null no top-level mask can express.

  Numeric columns hand their data buffer to Arrow unchanged. Fixed-length
  string columns are converted to `large_string` — a fixed-width column has no
  offsets to lend — with the offsets computed by array arithmetic rather than a
  Python loop. Each column's `units`, `description`, `valid_min`, `valid_max`
  and (for categoricals) `ordered` attributes ride along as Arrow field
  metadata under an `h5col.` prefix, and survive a Parquet round trip.

  List columns are the case Arrow fits best, because H5Col already stores them
  in Arrow's layout. `OFFSETS` (uint64) is reinterpreted as Arrow's int64
  offsets without touching the memory, and a `STRING_VALUES` group's `CHARS`
  goes across as the string payload. Only the null masks are converted, from
  H5Col's byte per row to Arrow's bit. Against the Python `read()` at 200,000
  rows this is 19x for `list<float64>`, 33x for `list<string>` and 26x for
  `list<list<float64>>`.

- User guide chapter [Reading into
  Python](https://hdfgroup.github.io/h5col/guide/reading-into-python.html),
  covering what each column type hands back from `read()` and `to_arrow()`, why
  missing values arrive masked, `.tolist()` versus `list()`, which NumPy
  functions drop a mask, and where each form falls short.
- Example notebook [Exporting to
  Arrow](https://hdfgroup.github.io/h5col/notebooks/07_arrow_export.html):
  `to_arrow()` end to end — real nulls in place of fill values, categoricals as
  Arrow dictionaries, list columns with their nesting and null-versus-empty
  distinction intact, column attributes carried as field metadata, and the hop
  to pandas and Parquet.

### Changed

- **`read()` now returns masked arrays by default.** `Column.read`,
  `Column.read_rows`, `Table.read` and `Selection.read` take `masked=True`,
  and every scalar column comes back as a `numpy.ma.MaskedArray` whose mask
  marks its missing rows. Previously a missing row was handed back as the
  column's fill value with nothing to distinguish it from data, so a mean over
  a column with missing rows was silently wrong. Pass `masked=False` for the
  previous behaviour, unchanged.

  - Uniform across scalar columns: boolean columns, which H5Col forbids from
    declaring a fill, and columns that declare none still come back masked with
    an all-False mask, so code written over the returned dict never has to
    branch on whether a given column can be missing.
  - List columns are unchanged — ragged, so they cannot be masked arrays — and
    already spell a null row `None`. They accept `masked=` and ignore it.
  - `fill_value` is the column's own decoded fill value, so `read().filled()`
    reproduces `read(masked=False)` and any operation that drops the mask
    degrades to the previous behaviour rather than to NumPy's defaults
    (999999 for an int8 column, the string `N/A` for a string one).
  - Note that `list(...)` over a masked array yields `numpy.ma.masked` where
    `.tolist()` yields `None`, and that `np.concatenate`, `np.stack` and
    friends drop the mask silently — use the `np.ma.*` equivalents.
  - NumPy prints a masked array by spelling out each value rather than
    formatting for the dtype, so a `float32` column that displayed as
    `[21.4 27.9]` now displays as `[21.399999618530273 ...]`. The values are
    unchanged.

- String and categorical columns now decode into NumPy 2's
  `numpy.dtypes.StringDType` instead of a `dtype=object` array of Python
  strings. Categorical columns use its nullable form, since a missing row has
  no label to carry; categories whose labels are not strings keep an object
  array. For 400,000 short strings the decoded column costs 6.4 MB rather than
  25.8 MB of resident memory, and `Column.read` on that column drops from
  23 ms to 2 ms.
- `Column.categories` returns a `StringDType` array for string labels.
- The `S` to `StringDType` cast validates lazily, so a non-conformant
  producer's invalid UTF-8 now raises `UnicodeDecodeError` when the offending
  value is read out of the array rather than when the column is read.
- Minimum NumPy raised to 2.0, which introduced `StringDType`; the conda
  dependency on HDF5 raised to 2.1.
- The documentation workflow builds only when something the site is actually
  built from changes, rather than on every push.
- GitHub Actions updated to versions running on Node 24, clearing the Node 20
  deprecation warnings.

### Fixed

- Categorical decoding invented missing values. `decode_codes` read the fill
  value without confirming the `H5D_FILL_VALUE_USER_DEFINED` state, and mapped
  any code outside `[0, ncategories)` to `None`. A single column could then
  contradict itself — `read()` reporting a row as `None` while `is_missing()`
  and the query layer reported it present — on a file that passes
  `validate(deep=True)`. An unindexable code now raises `ConformanceError`, and
  a column declaring no fill value no longer has h5py's library default read as
  one.
- A `numpy.ma.MaskedArray` passed to `append()` had its mask ignored, so a
  masked row was written as whatever value sat beneath the mask. For a
  fixed-length string column that produced the literal characters `--`; for a
  numeric column it silently stored stale data as though it were real. A masked
  element now means the same as `None` on write.

## [0.1.0] - 2026-08-05

### Added

- **Tables** backed by HDF5 groups — create, open, append, read, and
  `truncate`, tracking the logical row count (`NROWS`).
- **Column datatypes** — numeric, fixed-length strings (oversized values raise
  `OversizedStringError`; never silently truncated), the H5Col boolean enum,
  and categorical columns (integer codes with a label mapping).
- **List columns** — `LIST_COLUMN` / `STRING_VALUES` with null masks and
  arbitrary nesting.
- **Filter pipelines** — `Filter` / `FilterPipeline` and the built-in
  `Deflate` / `Shuffle` / `Fletcher32`, plus adaptation of `hdf5plugin`
  filters, applied in declared order through the dataset-creation property list.
- **Search indexes** with validity tokens (`GENERATION` / `SOURCE_*`) —
  `CHUNK_MINMAX`, `SORTED_ROWS`, and `BITMAP`: build, refresh, and query
  primitives.
- **Query layer** — `field()` expressions and pyarrow-style DNF tuple filters,
  `Table.select` / `read(where=)` / `count`, with three-valued (Kleene)
  missing-value semantics matching Arrow, and an `explain` plan.
- **Conformance validation** — `validate()` and the semantic `validate(deep=True)`.
- Friendly `FixedString` handler for HDF5 fixed-length strings, and a full
  H5Col exception family rooted at `H5ColError`.
- `Column.read_rows(rows)` reads just the given rows, decoded, using
  coalesced chunk-aligned block reads. Rows may be in any order and may
  repeat.
- Documentation site under `docs/` (Sphinx, Markdown via MyST, pydata theme),
  published to GitHub Pages: getting-started pages, a user guide, the query
  syntax reference, the rendered example notebooks, and an API reference. The
  theme is restyled with self-hosted IBM Plex Sans/Mono (SIL OFL 1.1) and a
  deep-teal palette tuned for both light and dark schemes.
- Project logo, applied across the README, the documentation navbar and
  landing page, the favicon and Apple touch icon, and the `og:image` used for
  link previews.
- GitHub workflows: `ci.yml` runs the gate (tests, lint, format check, type
  check) on Linux and macOS; `docs.yml` builds the documentation on every push
  and pull request and deploys it to GitHub Pages from `main`.
- CI runs the gate on Windows as well as Linux and macOS, and the pixi lockfile
  pins `win-64` alongside `linux-64` and `osx-arm64`.
- CI and documentation build badges in the README.

## Known limitations

- `CHUNK_BLOOM` search indexes are deferred.
- Object references are written as `H5T_STD_REF_OBJ` rather than the
  convention's `H5T_STD_REF`, because h5py cannot yet create the unified type.
