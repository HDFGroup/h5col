# Examples

The examples are executable Jupyter notebooks, committed with their outputs,
so every page below shows real results. They live in the repository's
[`examples/`](https://github.com/HDFGroup/h5col/tree/main/examples)
directory; run them yourself with the `examples` pixi environment
(`pixi run -e examples jupyter lab`).

Each notebook is self-contained. Most write their files to the system
temporary directory; the taxi notebook builds its file from a small committed
sample of the public NYC TLC trip data, so it too runs offline. The Arrow
notebook needs the optional `pyarrow` dependency, which the `examples`
environment already has.

| Notebook | Covers |
|---|---|
| {doc}`Quickstart <notebooks/01_quickstart>` | Create → append → read → validate, and reopen from disk. |
| {doc}`Column types <notebooks/02_column_types>` | Fixed-length strings (no silent truncation), categoricals, booleans, valid ranges, and missing values. |
| {doc}`List columns <notebooks/03_list_columns>` | `list<float>`, `list<str>`, nested lists, null versus empty, and the offsets layout. |
| {doc}`Filters and storage <notebooks/04_filters_and_storage>` | Per-column filter pipelines, `hdf5plugin` codecs, compression ratios, and the raw HDF5 layout. |
| {doc}`JSON logs <notebooks/05_json_logs>` | Modeling semi-structured log records with columns and list columns. |
| {doc}`NYC taxi trips <notebooks/06_nyc_taxi>` | A real dataset end to end: categoricals, both missing-value styles, a datetime codec, filters, and indexed queries with `explain()`. |
| {doc}`Exporting to Arrow <notebooks/07_arrow_export>` | `to_arrow()`: real nulls instead of fill values, categoricals as dictionaries, list columns intact, column attributes carried as field metadata, and the hop to pandas and Parquet. |

```{toctree}
:hidden:
:maxdepth: 1

notebooks/01_quickstart
notebooks/02_column_types
notebooks/03_list_columns
notebooks/04_filters_and_storage
notebooks/05_json_logs
notebooks/06_nyc_taxi
notebooks/07_arrow_export
```
