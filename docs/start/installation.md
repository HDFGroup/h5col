# Installation

`h5col` is not yet published to PyPI or conda-forge, so today it installs from
its repository at
[github.com/HDFGroup/h5col](https://github.com/HDFGroup/h5col). The package is
pure Python; all of its binary needs are covered by its dependencies.

## Requirements

- Python ≥ 3.11
- [h5py](https://docs.h5py.org/en/stable/) ≥ 3.11, built against HDF5 ≥ 2.1.0
- [NumPy](https://numpy.org) ≥ 2.0 (string columns decode into its
  `StringDType`)
- [hdf5plugin](https://pypi.org/project/hdf5plugin/) ≥ 4.0
- [pydantic](https://docs.pydantic.dev) ≥ 2.5

`hdf5plugin` is a regular runtime dependency, not an extra: importing it
registers the widely used compression filters (Zstandard, Blosc2, and
others) with HDF5. `h5col` imports it for you, so columns compressed with
those filters read and write without any additional setup.

## With pixi

[pixi](https://pixi.sh) is the environment manager the project itself uses.
It creates the conda-based environment and installs `h5col` into it in editable
mode:

```bash
git clone https://github.com/HDFGroup/h5col.git
cd h5col
pixi install
pixi run python -c "import h5col; print(h5col.__version__)"
```

## With pip

Inside a virtual environment, either install straight from the repository:

```bash
pip install git+https://github.com/HDFGroup/h5col.git
```

or from a clone, which is the better choice if you want the examples and
tests:

```bash
git clone https://github.com/HDFGroup/h5col.git
cd h5col
pip install .
```

The dependencies install from PyPI; the h5py wheels bundle a suitable HDF5
library, so no system HDF5 is required.

## With conda or mamba

Install the compiled dependencies from conda-forge first, then the package
itself with pip:

```bash
conda create -n h5col -c conda-forge "python>=3.11" "h5py>=3.11" "hdf5>=2.1" \
    "numpy>=2.0" "hdf5plugin>=4.0" "pydantic>=2.5"
conda activate h5col
pip install git+https://github.com/HDFGroup/h5col.git
```

## For development

Clone the repository and use the pixi environments described in
[Development](../about/contributing.md); they pin the toolchain used by the
test suite, the linters, and this documentation.
