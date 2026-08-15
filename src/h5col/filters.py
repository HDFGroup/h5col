"""An H5Col filter API modeled as an ordered pipeline of filter plugins.

H5Col columns benefit from per-column filter pipelines. Rather than exposing
h5py's high-level filter keywords (which allow only a single plugin compressor),
H5Col represents a pipeline as an ordered list of :class:`Filter` entries and
applies them through the HDF5 dataset-creation property list, mimicking the HDF5
filter pipeline itself. Filters from the ``hdf5plugin`` package are accepted
directly and adapted transparently.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from h5py import h5z

from .exceptions import FilterError

# Built-in HDF5 filter identifiers.
FILTER_DEFLATE = int(h5z.FILTER_DEFLATE)
FILTER_SHUFFLE = int(h5z.FILTER_SHUFFLE)
FILTER_FLETCHER32 = int(h5z.FILTER_FLETCHER32)
FILTER_SZIP = int(h5z.FILTER_SZIP)
FILTER_NBIT = int(h5z.FILTER_NBIT)
FILTER_SCALEOFFSET = int(h5z.FILTER_SCALEOFFSET)

_FLAG_OPTIONAL = int(h5z.FLAG_OPTIONAL)


@dataclass(frozen=True)
class Filter:
    """One entry of a filter pipeline.

    Parameters
    ----------
    plugin_id:
        The registered HDF5 filter plugin identifier. A filter plugin is the
        piece of software that connects the HDF5 library to the code that
        actually filters a chunk's bytes.
    cd_values:
        Client data (the filter's unsigned-int parameters), in order.
    optional:
        If True, HDF5 may skip the filter for a chunk it cannot process instead
        of failing the write (the ``H5Z_FLAG_OPTIONAL`` flag).
    name:
        Human-readable label (informational only).
    """

    plugin_id: int
    cd_values: tuple[int, ...] = ()
    optional: bool = False
    name: str = ""

    @property
    def flags(self) -> int:
        """The HDF5 filter flags for this entry."""
        return _FLAG_OPTIONAL if self.optional else 0


# --------------------------------------------------------------------------- #
# Built-in filter constructors
# --------------------------------------------------------------------------- #
def Deflate(level: int = 4) -> Filter:
    """The built-in deflate compression filter (HDF5's ``H5Z_FILTER_DEFLATE``).

    Parameters
    ----------
    level:
        Compression level, ``0``–``9`` (default ``4``).

    Raises
    ------
    FilterError
        If *level* is outside ``0``–``9``.
    """
    if not 0 <= level <= 9:
        raise FilterError(f"deflate level must be in 0..9, got {level}")
    return Filter(FILTER_DEFLATE, (int(level),), name="deflate")


def Shuffle() -> Filter:
    """The built-in byte-shuffle filter."""
    return Filter(FILTER_SHUFFLE, (), name="shuffle")


def Fletcher32() -> Filter:
    """The built-in Fletcher-32 checksum filter."""
    return Filter(FILTER_FLETCHER32, (), name="fletcher32")


def from_hdf5plugin(obj: Any) -> Filter:
    """Adapt an ``hdf5plugin`` filter instance to a :class:`Filter`.

    ``hdf5plugin`` filter objects behave like mappings of h5py create_dataset
    keyword arguments (``compression`` = filter id, ``compression_opts`` =
    client data) and expose a ``filter_id`` attribute.

    Parameters
    ----------
    obj:
        An ``hdf5plugin`` filter instance such as ``hdf5plugin.Zstd()``.
        Anything else that converts to such a mapping and carries a filter id
        also works; the class itself is not required.

    Raises
    ------
    FilterError
        If *obj* cannot be interpreted as an ``hdf5plugin`` filter or carries no
        filter id.
    """
    try:
        mapping = dict(obj)
    except (TypeError, ValueError) as exc:
        raise FilterError(f"cannot interpret {obj!r} as an hdf5plugin filter") from exc
    fid = mapping.get("compression", getattr(obj, "filter_id", None))
    if fid is None:
        raise FilterError(f"{obj!r} has no filter id")
    opts = mapping.get("compression_opts") or ()
    return Filter(
        int(fid),
        tuple(int(x) for x in opts),
        name=type(obj).__name__.lower(),
    )


def _coerce(obj: Any) -> Filter:
    """Return *obj* as a :class:`Filter`, adapting an ``hdf5plugin`` filter.

    Parameters
    ----------
    obj:
        One pipeline entry. A :class:`Filter` is returned unchanged. An
        ``hdf5plugin`` filter instance goes through :func:`from_hdf5plugin`;
        it is recognized by having a ``filter_id`` attribute, or by being
        mapping-like (having ``keys``) with a ``compression`` key. Anything
        else raises :class:`FilterError`.
    """
    if isinstance(obj, Filter):
        return obj
    # hdf5plugin filter instance (mapping-like with a filter id).
    if hasattr(obj, "filter_id") or (hasattr(obj, "keys") and "compression" in obj):
        return from_hdf5plugin(obj)
    raise FilterError(f"cannot interpret {obj!r} as a filter")


@dataclass(frozen=True)
class FilterPipeline(Sequence[Filter]):
    """An ordered, immutable sequence of :class:`Filter` entries.

    Accepts :class:`Filter` instances and ``hdf5plugin`` filter objects, adapting
    the latter automatically.

    Raises
    ------
    FilterError
        If an entry is neither a :class:`Filter` nor an ``hdf5plugin`` filter.
    """

    filters: tuple[Filter, ...] = field(default_factory=tuple)

    def __init__(self, filters: Iterable[Any] = ()) -> None:
        object.__setattr__(self, "filters", tuple(_coerce(f) for f in filters))

    def __getitem__(self, index: Any) -> Any:
        """One filter by position, or a tuple of them for a slice.

        Parameters
        ----------
        index:
            An integer position in pipeline order, or a slice.
        """
        return self.filters[index]

    def __len__(self) -> int:
        return len(self.filters)

    def __iter__(self) -> Iterator[Filter]:
        return iter(self.filters)

    def apply(self, dcpl: Any) -> None:
        """Add every filter, in order, to an HDF5 dataset-creation property list.

        Parameters
        ----------
        dcpl:
            A dataset-creation property list, modified in place. Declaration
            order is pipeline order, so filters are added exactly as listed.
        """
        for f in self.filters:
            dcpl.set_filter(f.plugin_id, f.flags, f.cd_values)

    def to_h5py_kwargs(self) -> dict[str, Any]:
        """Map the pipeline to h5py high-level ``create_dataset`` keyword arguments.

        h5py's high-level dataset API is the only creation path that sets fill
        values correctly for every dtype (its low-level ``set_fill_value`` is
        broken for fixed-length strings), so column creation goes through it. The
        builtin shuffle/fletcher32 filters map to their boolean keywords and the
        single remaining compressor maps to ``compression`` / ``compression_opts``.

        Raises :class:`FilterError` if the pipeline needs more than one
        compressor filter, which the high-level API cannot express (combine them
        into a Blosc/Blosc2 meta-compressor, or drive the low-level DCPL via
        :meth:`apply`).
        """
        kwargs: dict[str, Any] = {}
        compressors: list[Filter] = []
        for f in self.filters:
            if f.plugin_id == FILTER_SHUFFLE:
                kwargs["shuffle"] = True
            elif f.plugin_id == FILTER_FLETCHER32:
                kwargs["fletcher32"] = True
            else:
                compressors.append(f)
        if len(compressors) > 1:
            names = [c.name or c.plugin_id for c in compressors]
            raise FilterError(
                "h5py's high-level dataset API supports only one compressor "
                f"filter; this pipeline has {len(compressors)}: {names}"
            )
        if compressors:
            c = compressors[0]
            if c.plugin_id == FILTER_DEFLATE:
                kwargs["compression"] = "gzip"  # h5py's spelling for deflate
                kwargs["compression_opts"] = int(c.cd_values[0]) if c.cd_values else 4
            else:
                kwargs["compression"] = c.plugin_id
                kwargs["compression_opts"] = c.cd_values or None
        return kwargs
