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

### Known limitations

- `CHUNK_BLOOM` search indexes are deferred.
- Object references are written as `H5T_STD_REF_OBJ` rather than the
  convention's `H5T_STD_REF`, because h5py cannot yet create the unified type.
