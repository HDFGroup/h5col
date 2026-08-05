# Development

The repository at
[github.com/HDFGroup/h5col](https://github.com/HDFGroup/h5col) is a
conventional `src`-layout Python project managed with
[pixi](https://pixi.sh). Everything below assumes a clone and runs from the
project directory.

## Environments

Four pixi environments share one solve group, so they resolve to a single
consistent dependency set. `h5col` itself is installed editable into each.

| Environment | Adds | Use it for |
|---|---|---|
| `default` | the runtime dependencies | using the library |
| `dev` | `pytest`, `ruff`, `mypy` | tests, linting, type checking |
| `examples` | JupyterLab, `nbconvert`, `pandas`, `pyarrow` | the example notebooks |
| `docs` | Sphinx, `myst-nb`, the theme | this documentation |

## The gate

Four checks must pass before a change lands, and CI runs exactly these:

```bash
pixi run -e dev test          # pytest
pixi run -e dev lint          # ruff check src tests
pixi run -e dev format-check  # ruff format --check src tests
pixi run -e dev typecheck     # mypy src (strict)
```

Tests live in `tests/`, named by the feature they codify; they are written
alongside the code, and the suite doubles as the executable form of the
convention's rules. Docstrings follow the numpy convention and are enforced
by ruff's pydocstyle rules.

## Documentation

The documentation source is `docs/`, written in Markdown (MyST) and built
with Sphinx:

```bash
pixi run docs        # build into docs/_build/html; warnings are errors
pixi run docs-live   # rebuild-and-reload preview while editing
```

The example notebooks render into the site from `examples/` (via the
`docs/notebooks` symlink) using their committed outputs; building the docs
never executes them. After changing a notebook, refresh its outputs
headlessly and commit the result:

```bash
pixi run -e examples jupyter nbconvert --to notebook --execute --inplace \
    examples/01_quickstart.ipynb
```

Publishing is automatic: the `docs.yml` GitHub workflow builds the site on
every pull request and every push to `main`, and deploys it to GitHub Pages
on the pushes to `main`.
