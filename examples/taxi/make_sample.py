"""Build the committed, offline sample from a downloaded month.

The sample is deterministic (no random sampling): the first ``HEAD_ROWS`` rows
of the month (clean, roughly time-ordered early-January trips — ideal for the
``CHUNK_MINMAX`` datetime-range demo) concatenated with ``NULL_ROWS`` rows drawn
from the file's block of records that carry missing values (so the notebook can
demonstrate H5Col's fill-value handling). The full zone lookup is copied
verbatim.

Run once to (re)generate the files under ``taxi/data/``::

    pixi run -e examples python -m examples.taxi.make_sample 2024-01
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import fetch

#: Clean, time-ordered rows taken from the head of the month.
HEAD_ROWS = 24_000
#: Rows taken from the missing-value block (all share nulls across columns).
NULL_ROWS = 1_000

#: Committed sample directory (next to this file).
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PARQUET = DATA_DIR / "yellow_sample.parquet"
ZONE_CSV = DATA_DIR / "taxi_zone_lookup.csv"


def build_sample(year_month: str = "2024-01") -> Path:
    """Slice the deterministic sample from the cached month and write it.

    Parameters
    ----------
    year_month : str
        Month to sample, ``"YYYY-MM"``. Must already be cached (or downloadable)
        via :func:`examples.taxi.fetch.yellow_tripdata`.

    Returns
    -------
    pathlib.Path
        Path to the written sample parquet file.
    """
    src = fetch.yellow_tripdata(year_month)
    table = pq.read_table(src)

    head = table.slice(0, HEAD_ROWS)

    # First index whose RatecodeID is null marks the start of the missing block.
    null_mask = pc.is_null(table["RatecodeID"])
    first_null = pc.index(null_mask, True).as_py()
    if first_null < 0:
        null_block = table.slice(0, 0)  # month has no missing rows
    else:
        null_block = table.slice(first_null, NULL_ROWS)

    sample = pa.concat_tables([head, null_block])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(sample, SAMPLE_PARQUET, compression="zstd")
    shutil.copyfile(fetch.zone_lookup(), ZONE_CSV)

    print(
        f"wrote {SAMPLE_PARQUET.name}: {sample.num_rows} rows "
        f"({SAMPLE_PARQUET.stat().st_size / 1e6:.2f} MB)"
    )
    print(f"wrote {ZONE_CSV.name}")
    return SAMPLE_PARQUET


if __name__ == "__main__":
    ym = sys.argv[1] if len(sys.argv) > 1 else "2024-01"
    build_sample(ym)
