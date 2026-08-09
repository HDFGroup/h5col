<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="docs/_static/logo/h5col-logo-dark.png">
    <img src="docs/_static/logo/h5col-logo.png" alt="h5col — big tables next to big arrays"
         width="480">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/HDFGroup/h5col/actions/workflows/ci.yml"><img
     src="https://github.com/HDFGroup/h5col/actions/workflows/ci.yml/badge.svg?branch=main"
     alt="CI: tests, lint, format, and type checks on Linux, macOS, and Windows"></a>
  <a href="https://github.com/HDFGroup/h5col/actions/workflows/docs.yml"><img
     src="https://github.com/HDFGroup/h5col/actions/workflows/docs.yml/badge.svg?branch=main"
     alt="Docs: documentation site build and GitHub Pages deployment"></a>
</p>

# h5col

A reference implementation and proof-of-concept for **H5Col — Column-Oriented
Tabular Data in HDF5**. `h5col` reads and writes column-oriented tables that
live natively as HDF5 groups: per-column datatypes and fill/missing values,
fixed-length strings, boolean and categorical columns, nested list columns,
per-column filter pipelines, and query-accelerating search indexes — with a
small pyarrow-style predicate API for selecting rows.

The convention (proposal id HEP001) is specified at
https://hdfalliance.github.io/heps/hep001/.

Opening HDF5 files is a standard `h5py` operation and is intentionally left
outside `h5col`, so every storage option (drivers, cloud-optimized settings, etc.)
remains available to callers.

> **Status:** functional (read/write, list columns, filters, search indexes, and
> the query layer are implemented and tested). The reference API is stable and
> documented at https://hdfgroup.github.io/h5col/. Not yet released to PyPI or
> conda-forge — install from this repository.

## Requirements

- Python ≥ 3.11
- `h5py` ≥ 3.11, built against **HDF5 ≥ 2.1** (the version this project pins)
- `numpy` ≥ 2.0 (for `StringDType`), `pydantic` ≥ 2.5, `hdf5plugin` ≥ 4.0

## Installation (from the repository)

Clone the repository, then use whichever tool you prefer. All commands are run
from the `h5col/` project directory.

### pixi (recommended)

[pixi](https://pixi.sh) manages the conda-based environments and installs
`h5col` editable into them automatically:

```bash
pixi install                     # solve + create the default environment
pixi run python -c "import h5col; print(h5col.__version__)"
```

### pip

```bash
pip install .                    # or: pip install -e .   (editable)
```

The dependencies install from PyPI; the `h5py` wheels bundle a suitable HDF5
build. Use a virtual environment.

### conda / mamba

`h5col` is not on conda-forge yet, so install the native dependencies from
conda-forge and then the package itself with pip:

```bash
conda create -n h5col -c conda-forge "python>=3.11" "h5py>=3.11" "hdf5>=2.1" \
    "numpy>=2.0" "hdf5plugin>=4.0" "pydantic>=2.5"
conda activate h5col
pip install .                    # from the repository root
```

## pixi environments

Four environments are defined (all in one solve group, so they share a
consistent dependency set):

| Environment | Adds | Use it for |
|---|---|---|
| `default`  | runtime deps + `h5col` (editable) | using the library |
| `dev`      | `pytest`, `ruff`, `mypy` | running tests, linting, type-checking |
| `examples` | JupyterLab, `nbconvert`, `pandas` | the example notebooks |
| `docs`     | Sphinx, `myst-nb`, theme | building the documentation site |

Select one with `pixi run -e <env> ...`.

## Running the checks

Predefined `pixi` tasks (run them in the `dev` environment):

```bash
pixi run -e dev test          # pytest
pixi run -e dev lint          # ruff check src tests
pixi run -e dev typecheck     # mypy src
pixi run -e dev format        # ruff format src tests  (rewrites files)
pixi run -e dev format-check  # ruff format --check src tests  (as CI runs it)
```

## Running the examples

The [`examples/`](examples/) directory holds runnable notebooks
(`01_quickstart` … `06_nyc_taxi`); see [`examples/README.md`](examples/README.md)
for a guide. Open them in JupyterLab:

```bash
pixi run -e examples jupyter lab
```

Or execute one headless:

```bash
pixi run -e examples jupyter nbconvert --to notebook --execute \
    examples/01_quickstart.ipynb
```

## Documentation

The documentation — a user guide, the query-syntax reference, the rendered
example notebooks, and the API reference — is published at
https://hdfgroup.github.io/h5col/. To build it locally:

```bash
pixi run docs         # build into docs/_build/html (warnings are errors)
pixi run docs-live    # live-reloading preview while editing
```

---

The software initial development was funded by the U.S. Department of Energy,
Office of Science, Office of Fusion Energy Sciences, under Award Number DE-SC0024442.
