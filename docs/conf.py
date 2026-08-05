"""Sphinx configuration for the H5Col documentation site."""

from __future__ import annotations

import h5col

# -- Project -----------------------------------------------------------------
project = "h5col"
author = "The HDF Group"
copyright = "2026, The HDF Group"  # noqa: A001 (Sphinx's required name)
version = h5col.__version__
release = h5col.__version__

# -- General -----------------------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

exclude_patterns = [
    "_build",
    ".DS_Store",
    # Copied verbatim as static assets, never parsed as source documents.
    "_static",
    # `notebooks/` is a symlink to the repository's `examples/` directory; its
    # README addresses the repository view, not the rendered site.
    "notebooks/README.md",
]

# -- MyST (Markdown) and notebooks --------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# The example notebooks are committed pre-executed; render their stored
# outputs instead of re-executing them at build time.
nb_execution_mode = "off"

# -- API reference ------------------------------------------------------------
autodoc_member_order = "bysource"
autosummary_generate = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# -- Code copy button ----------------------------------------------------------
# Strip prompts so copied snippets paste runnable.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML ----------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "h5col"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_favicon = "_static/logo/favicon-64.png"
html_baseurl = "https://hdfgroup.github.io/h5col/"
# Sphinx does not expose html_baseurl to templates; layout.html needs it to
# build the absolute og:image URL that link unfurling requires.
html_context = {"baseurl": html_baseurl}
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "github_url": "https://github.com/HDFGroup/h5col",
    "external_links": [
        {
            "name": "HEP001 (the convention)",
            "url": "https://hdfalliance.github.io/heps/hep001/",
        },
    ],
    # The mark alone in the navbar: it is legible at 34 px and reads on both
    # the light and the dark ground, which the full lockup does not. The
    # wordmark beside it comes from `text`.
    # NOTE: the theme references the logo by basename under `_static/`, so this
    # file must sit at the top of `_static/` — a path like `logo/mark.png`
    # renders as a broken image.
    "logo": {
        "text": "h5col",
        "image_light": "h5col-mark.png",
        "image_dark": "h5col-mark.png",
        "alt_text": "h5col",
    },
    "navbar_align": "left",
    "header_links_before_dropdown": 7,
    "show_toc_level": 2,
}
