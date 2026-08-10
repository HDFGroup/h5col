# Changelog

All notable changes to H5Col are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/).

## [0.1.0]

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
- Documentation site under `docs/` (Sphinx, Markdown via MyST, pydata theme),
  published to GitHub Pages: getting-started pages, a user guide, the query
  syntax reference, the rendered example notebooks, and an API reference. The
  theme is restyled with self-hosted IBM Plex Sans/Mono (SIL OFL 1.1) and a
  deep-teal palette tuned for both light and dark schemes.
- Project logo, applied across the README, the documentation navbar and
  landing page, the favicon and Apple touch icon, and the `og:image` used for
  link previews.
- GitHub workflows: `ci.yml` runs the gate (tests, lint, format check, type
  check) on Linux, macOS, and Windows; `docs.yml` builds the documentation on
  every push and pull request and deploys it to GitHub Pages from `main`.
- `Column.read_rows(rows)` reads just the given rows, decoded, using
  coalesced chunk-aligned block reads. Rows may be in any order and may
  repeat.
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

### Changed

- String and categorical columns now decode into NumPy 2's
  `numpy.dtypes.StringDType` instead of a `dtype=object` array of Python
  strings. Categorical columns use its nullable form, since a missing row has
  no label to carry; categories whose labels are not strings keep an object
  array. For 400,000 short strings the decoded column costs 6.4 MB rather than
  25.8 MB of resident memory, and `Column.read` on that column drops from
  23 ms to 2 ms.
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
  - `fill_value` is the column's own decoded sentinel, so
    `read().filled()` reproduces `read(masked=False)` and any operation that
    drops the mask degrades to the previous behaviour rather than to NumPy's
    defaults (999999 for an int8 column, the string `N/A` for a string one).
  - Note that `list(...)` over a masked array yields `numpy.ma.masked` where
    `.tolist()` yields `None`, and that `np.concatenate`, `np.stack` and
    friends drop the mask silently — use the `np.ma.*` equivalents.
- `Column.categories` returns a `StringDType` array for string labels.
- The `S` to `StringDType` cast validates lazily, so a non-conformant producer's
  invalid UTF-8 now raises `UnicodeDecodeError` when the offending value is read
  out of the array rather than when the column is read.
- Minimum NumPy raised to 2.0, which introduced `StringDType`.

### Known limitations

- `CHUNK_BLOOM` search indexes are deferred.
- Object references are written as `H5T_STD_REF_OBJ` rather than the
  convention's `H5T_STD_REF`, because h5py cannot yet create the unified type.
