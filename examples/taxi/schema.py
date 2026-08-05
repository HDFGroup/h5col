"""H5Col table schemas for the NYC taxi example, plus the datetime codec.

Two tables are declared:

``trips``
    One row per taxi trip. Exercises numeric and string **categoricals**
    (``VendorID``, ``RatecodeID``, ``payment_type``, ``store_and_fwd_flag``),
    **missing values** in both the sentinel style (``passenger_count`` uses a
    ``-1`` fill with ``valid_min=0``) and the NaN style (nullable money columns
    use a NaN fill), plain integer columns (``PULocationID``/``DOLocationID``),
    and **compressed** float columns (Shuffle + Deflate).

``zones``
    The taxi-zone lookup dimension, using fixed-length UTF-8 string columns.

HDF5 has no native datetime type and the H5Col convention does not define one,
so the two timestamp columns are stored as ``int64`` seconds with a CF-style
``units`` attribute. The TLC timestamps carry no timezone; they are local NYC
wall-clock, and the codec preserves them verbatim (see :data:`TIME_UNITS`).
"""

from __future__ import annotations

import numpy as np

from h5col import ColumnSpec, Deflate, FilterPipeline, FixedString, Shuffle, TableSpec

# --------------------------------------------------------------------------- #
# Datetime codec
# --------------------------------------------------------------------------- #
#: CF/UDUNITS-style units string recorded on the stored timestamp columns. The
#: TLC values are naive local (America/New_York) wall-clock; they are stored as
#: seconds since the epoch *without* a timezone shift, so this string documents
#: the reference instant only, not a UTC guarantee.
TIME_UNITS = "seconds since 1970-01-01 00:00:00"


def encode_datetime(values: np.ndarray) -> np.ndarray:
    """Encode a ``datetime64`` array to ``int64`` seconds for storage."""
    return values.astype("datetime64[s]").astype("int64")


def decode_datetime(codes: np.ndarray) -> np.ndarray:
    """Decode stored ``int64`` seconds back to a ``datetime64[s]`` array."""
    return np.asarray(codes, dtype="int64").astype("datetime64[s]")


# --------------------------------------------------------------------------- #
# Categorical domains (raw source code -> stored label)
# --------------------------------------------------------------------------- #
#: TLC payment-type code -> human label. Stored as a *string* categorical so
#: queries read naturally, e.g. ``field("payment_type") == "Credit card"``.
PAYMENT_TYPES: dict[int, str] = {
    0: "Not specified",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}
PAYMENT_LABELS = list(PAYMENT_TYPES.values())

#: Vendor codes present in the data, kept as a *numeric* categorical.
VENDOR_IDS = [1, 2, 6]

#: Rate-code domain (numeric categorical); 99 is the TLC "unknown" code.
RATE_CODES = [1, 2, 3, 4, 5, 6, 99]

#: Store-and-forward flag domain (string categorical); rows may be missing.
STORE_FWD_FLAGS = ["N", "Y"]

# --------------------------------------------------------------------------- #
# Filter pipelines
# --------------------------------------------------------------------------- #
#: Byte-shuffle then Deflate — strong on the numeric columns.
_NUMERIC_FILTERS = FilterPipeline([Shuffle(), Deflate(5)])

#: Trip columns are chunked small so CHUNK_MINMAX pruning is visible on 25k rows.
_TRIP_CHUNK = 4096


def trips_spec() -> TableSpec:
    """Return the :class:`~h5col.TableSpec` for the ``trips`` table."""
    money = dict(dtype=np.float64, chunks=_TRIP_CHUNK, filters=_NUMERIC_FILTERS)
    return TableSpec(
        title="NYC yellow-taxi trips (2024-01 sample)",
        description=(
            "Sample of NYC TLC yellow-taxi trip records. Timestamps are stored "
            f"as int64 with units '{TIME_UNITS}' (local NYC wall-clock)."
        ),
        columns=[
            ColumnSpec(
                name="VendorID",
                categories=VENDOR_IDS,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                description="Provider that recorded the trip.",
            ),
            ColumnSpec(
                name="tpep_pickup_datetime",
                dtype=np.int64,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                units=TIME_UNITS,
                description="Meter engaged (trip start).",
            ),
            ColumnSpec(
                name="tpep_dropoff_datetime",
                dtype=np.int64,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                units=TIME_UNITS,
                description="Meter disengaged (trip end).",
            ),
            ColumnSpec(
                name="passenger_count",
                dtype=np.int64,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                fill_value=-1,
                valid_min=0,
                description="Driver-entered passenger count; -1 fill = missing.",
            ),
            ColumnSpec(
                name="trip_distance",
                units="mile",
                **money,
                description="Elapsed trip distance.",
            ),
            ColumnSpec(
                name="RatecodeID",
                categories=RATE_CODES,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                description="Final rate code; 99 = unknown. May be missing.",
            ),
            ColumnSpec(
                name="store_and_fwd_flag",
                categories=STORE_FWD_FLAGS,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                description="Store-and-forward flag (Y/N). May be missing.",
            ),
            ColumnSpec(
                name="PULocationID",
                dtype=np.int32,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                description="Pickup TLC zone (see zones table).",
            ),
            ColumnSpec(
                name="DOLocationID",
                dtype=np.int32,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                description="Dropoff TLC zone (see zones table).",
            ),
            ColumnSpec(
                name="payment_type",
                categories=PAYMENT_LABELS,
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                description="How the passenger paid.",
            ),
            ColumnSpec(
                name="fare_amount", units="USD", **money, description="Meter fare."
            ),
            ColumnSpec(
                name="extra",
                units="USD",
                **money,
                description="Misc. extras and surcharges.",
            ),
            ColumnSpec(name="mta_tax", units="USD", **money, description="MTA tax."),
            ColumnSpec(
                name="tip_amount",
                units="USD",
                **money,
                description="Tip (credit-card tips only).",
            ),
            ColumnSpec(
                name="tolls_amount", units="USD", **money, description="Tolls paid."
            ),
            ColumnSpec(
                name="improvement_surcharge",
                units="USD",
                **money,
                description="Improvement surcharge.",
            ),
            ColumnSpec(
                name="total_amount",
                units="USD",
                **money,
                description="Total charged to passenger.",
            ),
            ColumnSpec(
                name="congestion_surcharge",
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                dtype=np.float64,
                fill_value=np.nan,
                units="USD",
                description="Congestion surcharge; NaN fill = missing.",
            ),
            ColumnSpec(
                name="airport_fee",
                chunks=_TRIP_CHUNK,
                filters=_NUMERIC_FILTERS,
                dtype=np.float64,
                fill_value=np.nan,
                units="USD",
                description="Airport pickup fee; NaN fill = missing.",
            ),
        ],
    )


def zones_spec() -> TableSpec:
    """Return the :class:`~h5col.TableSpec` for the ``zones`` dimension table."""
    return TableSpec(
        title="NYC TLC taxi zone lookup",
        description="LocationID -> borough / zone / service zone.",
        columns=[
            # Small dimension table: chunk to the row count so the single chunk
            # is not over-allocated (auto-sizing targets large tables).
            ColumnSpec(
                name="LocationID",
                dtype=np.int32,
                chunks=512,
                description="TLC taxi-zone identifier.",
            ),
            ColumnSpec(
                name="Borough",
                dtype=FixedString(nbytes=20),
                chunks=512,
                filters=_NUMERIC_FILTERS,
                description="Borough name.",
            ),
            ColumnSpec(
                name="Zone",
                dtype=FixedString(nbytes=64),
                chunks=512,
                filters=_NUMERIC_FILTERS,
                description="Zone name.",
            ),
            ColumnSpec(
                name="service_zone",
                dtype=FixedString(nbytes=16),
                chunks=512,
                filters=_NUMERIC_FILTERS,
                description="Service zone.",
            ),
        ],
    )
