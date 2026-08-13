"""Sphinx configuration for the H5Col documentation site."""

from __future__ import annotations

import re
from pathlib import Path

import h5col

_REPO = Path(__file__).resolve().parent.parent


def _latest_release() -> str:
    """The newest released version, from the topmost CHANGELOG.md heading.

    Not the same thing as ``h5col.__version__``: the site is published from
    ``main``, which carries a ``.devN`` version between releases, so the version
    these pages were built from is usually ahead of the newest release.

    Taken from the changelog rather than from git tags because the docs
    workflow checks out shallowly, leaving no tags to read — a tag-derived
    lookup would come back empty in the one place it matters. ``[Unreleased]``
    is skipped: the pattern requires a version number.
    """
    changelog = _REPO / "CHANGELOG.md"
    for line in changelog.read_text(encoding="utf-8").splitlines():
        found = re.match(r"^##\s*\[(\d+\.\d+\.\d+)\]", line)
        if found:
            return found.group(1)
    raise RuntimeError(f"no released version heading found in {changelog}")


def _check_pinned_install_version(latest: str) -> None:
    """Fail the build if the installation page pins a stale version.

    The install commands name a tag so that following the documentation gets
    you a release rather than whatever ``main`` happens to be. That pin cannot
    be substituted in — MyST does not expand substitutions inside a code fence
    — so it is written out and checked here instead.
    """
    page = _REPO / "docs" / "start" / "installation.md"
    pinned = set(re.findall(r"@v(\d+\.\d+\.\d+)", page.read_text(encoding="utf-8")))
    stale = pinned - {latest}
    if stale:
        raise RuntimeError(
            f"{page} pins {sorted(stale)} but the latest release is {latest}; "
            f"update the install commands"
        )


# -- Project -----------------------------------------------------------------
project = "h5col"
author = "The HDF Group"
copyright = "2026, The HDF Group"  # noqa: A001 (Sphinx's required name)
version = h5col.__version__
release = h5col.__version__
latest_release = _latest_release()
_check_pinned_install_version(latest_release)

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
html_context = {"baseurl": html_baseurl, "latest_release": latest_release}
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "github_url": "https://github.com/HDFGroup/h5col",
    # Just the identifier: the theme appends an external-link glyph, so the
    # off-site jump is already signalled, and the landing page introduces
    # HEP001 by its full title.
    "external_links": [
        {
            "name": "HEP001",
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
    # The release number sits before the theme and icon controls, so it is
    # visible on every page without a banner taking a strip off the top of
    # each one.
    "navbar_end": ["latest-release", "theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 7,
    "show_toc_level": 2,
}
