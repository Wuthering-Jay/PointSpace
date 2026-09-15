"""Shared LAS/LAZ output helpers for utility scripts.

laspy can read COPC but cannot serialize its hierarchy VLRs. Any utility that
changes points or attributes necessarily writes a conventional LAS/LAZ file,
so COPC-only records must be removed before calling ``LasData.write``.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Union


_COPC_WARNING_EMITTED = False


def _is_copc_record(vlr) -> bool:
    return str(getattr(vlr, "user_id", "")).strip().lower() == "copc"


def strip_copc_metadata(las) -> bool:
    """Convert a loaded COPC header to a standard LAS/LAZ header in-place.

    Only COPC info/hierarchy records are removed. CRS records, ExtraBytes,
    point order and all point attributes are preserved.
    """

    return strip_copc_header_metadata(las.header)


def strip_copc_header_metadata(header) -> bool:
    """Remove only COPC indexing records from a LAS header in-place."""
    original_vlr_count = len(header.vlrs)
    header.vlrs = type(header.vlrs)(
        vlr for vlr in header.vlrs if not _is_copc_record(vlr)
    )

    original_evlr_count = len(header.evlrs or ())
    if header.evlrs is not None:
        header.evlrs = type(header.evlrs)(
            vlr for vlr in header.evlrs if not _is_copc_record(vlr)
        )
    return (
        len(header.vlrs) != original_vlr_count
        or len(header.evlrs or ()) != original_evlr_count
    )


def standard_header_copy(header):
    """Return an independent conventional LAS header preserving its schema."""
    output_header = deepcopy(header)
    strip_copc_header_metadata(output_header)
    return output_header


def write_standard_las(
    las,
    output_path: Union[str, Path],
    *,
    warn_on_copc: bool = True,
) -> bool:
    """Write LAS/LAZ, transparently converting unsupported COPC metadata.

    Returns ``True`` when the input header contained COPC hierarchy records.
    The output extension still controls compression in laspy.
    """

    global _COPC_WARNING_EMITTED
    output_path = Path(output_path)
    converted = strip_copc_metadata(las)
    if converted and warn_on_copc and not _COPC_WARNING_EMITTED:
        logging.getLogger(__name__).warning(
            "COPC input is being written as standard LAS/LAZ because laspy "
            "does not support COPC writing. CRS, point order and attributes "
            "are preserved; COPC hierarchy indexing is removed. First output: %s",
            output_path,
        )
        _COPC_WARNING_EMITTED = True
    las.write(output_path)
    return converted
