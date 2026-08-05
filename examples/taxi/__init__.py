"""Helpers for the NYC taxi H5Col example (notebook ``06_nyc_taxi.ipynb``).

This subpackage turns the public NYC Taxi & Limousine Commission (TLC) yellow
trip records into two H5Col tables:

* :mod:`~examples.taxi.fetch` downloads a monthly parquet file (and the taxi
  zone lookup) on demand into a git-ignored cache — used for real-scale runs.
* :mod:`~examples.taxi.make_sample` slices a small, deterministic, committed
  sample so the notebook runs offline.
* :mod:`~examples.taxi.schema` declares the ``trips`` and ``zones`` table
  schemas and the datetime codec.
* :mod:`~examples.taxi.build` reads parquet and writes the H5Col file.

The TLC trip-record data is published by NYC as open data.
See https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
"""
