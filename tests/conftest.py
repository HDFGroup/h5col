"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import pytest


@pytest.fixture
def h5path(tmp_path: Path) -> Path:
    return tmp_path / "test.h5"


@pytest.fixture
def h5file(h5path: Path) -> Iterator[h5py.File]:
    with h5py.File(h5path, "w") as f:
        yield f
