"""Read taxi parquet/CSV and write the ``trips`` and ``zones`` H5Col tables.

Both tables are written into a single HDF5 file under root groups ``trips`` and
``zones``. After the data is appended, three search indexes are built to
demonstrate the query planner:

* ``CHUNK_MINMAX`` on ``tpep_pickup_datetime`` — range pruning by chunk;
* ``SORTED_ROWS`` on ``total_amount`` — exact range answers;
* ``BITMAP`` on ``payment_type`` — exact categorical equality/``isin``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from h5col import Table

from . import make_sample
from .schema import (
    PAYMENT_TYPES,
    encode_datetime,
    trips_spec,
    zones_spec,
)

#: Money/float columns copied straight across (NaN marks missing).
_FLOAT_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
]

#: Default output path for the built file.
DEFAULT_OUT = Path(__file__).resolve().parent / "nyc_taxi.h5"


def _labels(series: pd.Series, fn: Callable[[Any], Any]) -> list[Any]:
    """Map a source series to categorical labels, NaN -> ``None`` (missing)."""
    return [None if pd.isna(v) else fn(v) for v in series]


def _prep_trips(df: pd.DataFrame) -> dict[str, Any]:
    """Prepare the ``trips`` append payload from the source dataframe."""
    data: dict[str, Any] = {
        "tpep_pickup_datetime": encode_datetime(df["tpep_pickup_datetime"].to_numpy()),
        "tpep_dropoff_datetime": encode_datetime(
            df["tpep_dropoff_datetime"].to_numpy()
        ),
        "PULocationID": df["PULocationID"].to_numpy(np.int32),
        "DOLocationID": df["DOLocationID"].to_numpy(np.int32),
        # Sentinel-style missing: nulls become the -1 fill (valid_min=0).
        "passenger_count": df["passenger_count"].fillna(-1).astype(np.int64).to_numpy(),
        # NaN-style missing: pandas nulls are already NaN.
        "congestion_surcharge": df["congestion_surcharge"].to_numpy(np.float64),
        "airport_fee": df["Airport_fee"].to_numpy(np.float64),
        # Categoricals (labels; None marks missing).
        "VendorID": _labels(df["VendorID"], int),
        "RatecodeID": _labels(df["RatecodeID"], int),
        "store_and_fwd_flag": _labels(df["store_and_fwd_flag"], str),
        "payment_type": _labels(df["payment_type"], lambda v: PAYMENT_TYPES[int(v)]),
    }
    for col in _FLOAT_COLUMNS:
        data[col] = df[col].to_numpy(np.float64)
    return data


def _prep_zones(zdf: pd.DataFrame) -> dict[str, Any]:
    """Prepare the ``zones`` append payload (fixed-string columns, ``""`` fill)."""
    return {
        "LocationID": zdf["LocationID"].to_numpy(np.int32),
        "Borough": zdf["Borough"].fillna("").astype(str).tolist(),
        "Zone": zdf["Zone"].fillna("").astype(str).tolist(),
        "service_zone": zdf["service_zone"].fillna("").astype(str).tolist(),
    }


def build(
    out: Path | str = DEFAULT_OUT,
    *,
    sample_parquet: Path | None = None,
    zone_csv: Path | None = None,
) -> Path:
    """Build the two-table H5Col file and return its path.

    Parameters
    ----------
    out : pathlib.Path or str
        Output HDF5 path (overwritten).
    sample_parquet, zone_csv : pathlib.Path, optional
        Source files; default to the committed sample under ``taxi/data/``.
        Pass a cached full-month parquet (see :mod:`examples.taxi.fetch`) for a
        real-scale run.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    sample_parquet = sample_parquet or make_sample.SAMPLE_PARQUET
    zone_csv = zone_csv or make_sample.ZONE_CSV
    out = Path(out)

    df = pq.read_table(sample_parquet).to_pandas()
    zdf = pd.read_csv(zone_csv)

    with h5py.File(out, "w") as f:
        trips = Table.create(f.create_group("trips"), trips_spec())
        trips.append(_prep_trips(df))
        trips.build_index("tpep_pickup_datetime", "CHUNK_MINMAX")
        trips.build_index("total_amount", "SORTED_ROWS")
        trips.build_index("payment_type", "BITMAP")

        zones = Table.create(f.create_group("zones"), zones_spec())
        zones.append(_prep_zones(zdf))

    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {df.shape[0]} trips")
    return out


if __name__ == "__main__":
    build()
