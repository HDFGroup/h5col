"""Download-on-demand for the NYC TLC yellow-taxi data.

The full monthly parquet files (~3M rows, tens of MB) are **not** committed to
the repository. This module fetches them on demand into a git-ignored cache
directory so the notebook can be re-run at real scale. For the committed,
offline sample see :mod:`~examples.taxi.make_sample`.

Data source: NYC Taxi & Limousine Commission, served from CloudFront. The
records are open data published by the City of New York.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

#: Base URL of the TLC trip-record CloudFront distribution.
_TRIP_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
_MISC_BASE = "https://d37ci6vzurychx.cloudfront.net/misc"

#: Git-ignored cache directory (next to this file).
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* unless it already exists; return *dest*."""
    if dest.exists():
        print(f"cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed, trusted TLC host)
    tmp.replace(dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def yellow_tripdata(year_month: str) -> Path:
    """Fetch one month of yellow-taxi trip records.

    Parameters
    ----------
    year_month : str
        Month to fetch, formatted ``"YYYY-MM"`` (e.g. ``"2024-01"``).

    Returns
    -------
    pathlib.Path
        Path to the cached parquet file.
    """
    name = f"yellow_tripdata_{year_month}.parquet"
    return _download(f"{_TRIP_BASE}/{name}", CACHE_DIR / name)


def zone_lookup() -> Path:
    """Fetch the taxi zone lookup table (``taxi_zone_lookup.csv``)."""
    name = "taxi_zone_lookup.csv"
    return _download(f"{_MISC_BASE}/{name}", CACHE_DIR / name)


if __name__ == "__main__":
    import sys

    ym = sys.argv[1] if len(sys.argv) > 1 else "2024-01"
    yellow_tripdata(ym)
    zone_lookup()
