# h5col example notebooks

Runnable, pre-executed Jupyter notebooks that tour the `h5col` API. They are a
good way to explore what the package does, from a first table to a real
dataset.

| Notebook | Covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Create → append → read → validate, and reopen from disk. |
| [`02_column_types.ipynb`](02_column_types.ipynb) | Fixed-length strings (no silent truncation), categorical columns, booleans, valid ranges, and missing values / fill values. |
| [`03_list_columns.ipynb`](03_list_columns.ipynb) | Variable-length list columns: `list<float>`, `list<str>`, nested `list<list<int>>`, null-vs-empty, and the offsets layout. |
| [`04_filters_and_storage.ipynb`](04_filters_and_storage.ipynb) | Per-column filter pipelines (shuffle + gzip, `hdf5plugin` Zstd), compression ratios, and a walk of the raw HDF5 layout. |
| [`05_json_logs.ipynb`](05_json_logs.ipynb) | Modeling semi-structured JSON log records as H5Col columns and list columns. |
| [`06_nyc_taxi.ipynb`](06_nyc_taxi.ipynb) | A real dataset (NYC yellow-taxi trips): categoricals, missing values, an `int64` datetime codec, filters, and the query layer driven by `BITMAP` / `SORTED_ROWS` / `CHUNK_MINMAX` indexes. |
| [`07_arrow_export.ipynb`](07_arrow_export.ipynb) | `to_arrow()`: missing values as real nulls, categoricals as Arrow dictionaries, list columns with their nesting intact, column attributes as field metadata, and the hop to pandas and Parquet. Needs the optional `pyarrow` dependency. |

## Running them

The notebook tooling lives in a dedicated pixi environment named `examples`
(kept out of the lean `dev` test/lint environment):

```bash
# from the h5col/ directory
pixi run -e examples jupyter lab      # then open a notebook
```

Or execute one headless to refresh its outputs:

```bash
pixi run -e examples jupyter nbconvert --to notebook --execute --inplace examples/01_quickstart.ipynb
```

Most notebooks write their HDF5 file to the system temp directory, so running
them leaves nothing behind in the repo.

### The NYC taxi example (`06_nyc_taxi.ipynb`)

This one uses the [`taxi/`](taxi) helper package and a small committed sample of
the public NYC TLC data ([`taxi/data/`](taxi/data)), so it runs offline. It
writes `taxi/nyc_taxi.h5` (git-ignored). For a real-scale run, download a full
month first — the helpers fetch it into a git-ignored cache:

```bash
pixi run -e examples python -m examples.taxi.fetch 2024-01
```

