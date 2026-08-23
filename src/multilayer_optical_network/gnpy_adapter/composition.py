"""Compose end-to-end GSNR from per-OMS ``1/gsnr_lin`` increments plus an
analytic endpoint (terminal-ROADM add/drop + tx_osnr) noise term.

Why this module exists: GNPy propagation dominates cold QoT workloads (~16
propagations/service, ~75% of wall time). Rather than re-propagate a full path
every time, ``adapter._propagate_loading(capture_increments=True)`` captures
each OMS's own noise contribution ONCE, inside a real multi-OMS propagation
(by differencing accumulated ``1/gsnr_lin`` at OMS boundaries). This module
recomposes GSNR for any path built from cached per-OMS increments:

    N_total(slot) = N_endpoint(slot) + sum_i increment_i(slot)
    GSNR_dB(slot) = -10 * log10(N_total(slot))

An interior (express) ROADM contributes no add/drop penalty and preserves
``1/gsnr_lin`` exactly (verified by element trace), so summing per-OMS
increments is exact -- UNLIKE running one OMS as a standalone lightpath via
``harvest_qot``, which runs it as a COMPLETE lightpath (its own endpoint add/
drop + tx_osnr chain) and so double-counts that endpoint chain once per hop
when several such single-OMS harvests are summed (-1.23 dB at 2 hops, -1.64 dB
at 3 -- the false start ``adapter._propagate_loading``'s capture path exists
to avoid; see its docstring and the composition design doc).

This module has NO dependency on a live gnpy propagation: ``oms_fingerprint``,
``endpoint_noise_lin``, and ``endpoint_key`` are pure functions of the
``OpticalNetworkModel``; ``compose_gsnr_db`` is pure arithmetic over already-
captured numbers. The capture half (harvesting increments and the endpoint
term from a real propagation) lives in ``adapter.py``'s ``_propagate_loading``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from ..model.assets import Direction
from ..model.optical_network import OpticalNetworkModel
from .adapter import _oms_fingerprint_parts
from .synthesize import SI_TX_OSNR_DB
from .translate import reverse_oms_sequence

__all__ = [
    "SI_TX_OSNR_DB",
    "oms_fingerprint",
    "endpoint_noise_lin",
    "endpoint_key",
    "compose_gsnr_db",
]

# gnpy's add_drop_osnr/tx_osnr reference bandwidth (Hz). Penalties declared at
# this bandwidth are renormalised to the mode's symbol rate via gnpy's own
# `snr_sum` factor, baud_rate / _ENDPOINT_REF_BW_HZ -- see `endpoint_noise_lin`.
_ENDPOINT_REF_BW_HZ = 12.5e9


def oms_fingerprint(model: OpticalNetworkModel, oms_id: str) -> tuple:
    """Physical fingerprint of ONE OMS.

    Emits EXACTLY the element tuples ``adapter._path_physical_fingerprint``
    emits for this OMS between its ``("oms",)`` markers, via the shared
    ``_oms_fingerprint_parts`` helper, so the two representations can never
    drift apart -- a change to what counts as GSNR-relevant physics is made
    once, in that helper, and both callers pick it up automatically.

    Content-addressed and direction-free: OMS objects are already directional
    (a bidirectional span registers two separate OMS with their own amp/fiber
    chains -- see ``testing.add_bidir_span``), so an asymmetric
    ``apply_nf_delta`` on one direction's amp changes only that direction's
    fingerprint, with no explicit invalidation logic — the HarvestCache
    property this module's cache keys are meant to preserve."""
    return tuple(_oms_fingerprint_parts(model, oms_id))


def _terminal_roadm_ids(
    model: OpticalNetworkModel, oms_sequence: Tuple[str, ...], direction: Direction,
) -> Tuple[str, str]:
    """(add-side, drop-side) terminal ROADM ids for *oms_sequence* under
    *direction*.

    Mirrors the S4-4 terminal-ROADM identification ``adapter._apply_penalties``
    and ``adapter._path_physical_fingerprint`` already use: the add-side ROADM
    is the first (resolved) OMS's own leading element (``roadm_<src>``, the
    importer's ``_add_directed_oms`` convention puts it at ``elements[0]``);
    the drop-side ROADM is ``roadm_<dst>`` of the last (resolved) OMS -- it is
    nobody's OMS chain member, appended separately at propagation time
    (S4-4), so it is derived from ``dst_node_id`` here rather than read off
    any OMS's ``elements``."""
    if direction == Direction.BACKWARD:
        seq = reverse_oms_sequence(model, oms_sequence)
        if seq is None:
            raise ValueError(
                f"backward endpoint term requires a paired reverse OMS for every "
                f"leg of {oms_sequence!r}, but none was found."
            )
    else:
        seq = oms_sequence
    first_oms = model.get_oms(seq[0])
    add_id = next(el for el in first_oms.elements if model.has_roadm(el))
    drop_id = f"roadm_{model.get_oms(seq[-1]).dst_node_id}"
    return add_id, drop_id


def endpoint_noise_lin(
    model: OpticalNetworkModel,
    oms_sequence: Tuple[str, ...],
    direction: Direction,
    *,
    baud_rate: float,
    tx_osnr_db: float,
) -> float:
    """Analytic terminal-ROADM add/drop + tx_osnr noise, in linear units at the
    *baud_rate* reference:

        N_endpoint = [db2lin(-(add_drop_add + lin2db(2)))
                    + db2lin(-(add_drop_drop + lin2db(2)))
                    + db2lin(-tx_osnr)] * (baud_rate / 12.5e9)

    Each terminal ROADM contributes its OWN ``add_drop_osnr_db`` (the
    per-instance penalty), so a heterogeneous-ROADM path (different add/drop
    budgets at the two ends) is handled correctly, not averaged or assumed
    uniform. The ``+ lin2db(2)`` one-sided correction mirrors
    ``adapter._apply_penalties``: gnpy's ``add_drop_osnr`` is the COMBINED
    add+drop budget for a full add+drop cycle, but a TERMINAL ROADM incurs
    only one side of it (add OR drop, never both) -- see that function's
    docstring for the numeric derivation gnpy verifies to 4 decimal places.
    Computed rather than fitted: a curve fit against composed GSNR silently
    absorbed this exact renormalisation factor (11x too large at 87.5 GBaud)
    because it was fitting a value already expressed at the 12.5 GHz
    reference, not the symbol rate.

    *tx_osnr_db* is a caller-supplied parameter, deliberately NOT defaulted to
    ``SI_TX_OSNR_DB`` (this module's re-export of the equipment SI block's
    declared 40): a caller composing GSNR for a path that was actually
    propagated (``adapter._propagate_loading``) must pass that propagation's
    own ``float(si.tx_osnr[i])`` instead, since ``translate.build_si_for_loading``
    carries its own independent 35 dB default that this adapter never
    overrides -- ``SI_TX_OSNR_DB`` is real for gnpy's own path_request_run
    computation path, but not for anything this adapter itself propagates."""
    from gnpy.core.utils import db2lin, lin2db

    add_id, drop_id = _terminal_roadm_ids(model, oms_sequence, direction)
    penalties_noise_lin = 0.0
    for rid in (add_id, drop_id):
        r = model._roadms.get(rid)
        if r is not None:
            penalties_noise_lin += db2lin(-(r.add_drop_osnr_db + lin2db(2.0)))
    penalties_noise_lin += db2lin(-tx_osnr_db)
    return penalties_noise_lin * (baud_rate / _ENDPOINT_REF_BW_HZ)


def endpoint_key(
    model: OpticalNetworkModel,
    oms_sequence: Tuple[str, ...],
    direction: Direction,
    *,
    baud_rate: float,
    tx_osnr_db: float,
) -> tuple:
    """Content-addressed key for the endpoint term: the two terminal ROADMs'
    own ``add_drop_osnr_db`` plus *baud_rate* and *tx_osnr_db* -- the exact set
    of inputs ``endpoint_noise_lin`` reads, and nothing else. A terminal
    ROADM's ``add_drop_osnr_db`` changing (a per-instance ROADM edit) splits
    this key on its own, content-addressed the same way
    ``_path_physical_fingerprint``'s keys split on a degraded span.

    *tx_osnr_db* is a required caller-supplied parameter, mirroring
    ``endpoint_noise_lin``'s signature exactly -- NOT defaulted to
    ``SI_TX_OSNR_DB``. A caller keying a real propagation's endpoint term must
    pass that propagation's own ``float(si.tx_osnr[i])`` (see
    ``endpoint_noise_lin``'s docstring for why: ``translate.build_si_for_loading``
    carries an independent 35 dB default that this adapter never overrides, so
    ``SI_TX_OSNR_DB`` -- 40, the equipment SI block's declared value -- is not
    what a real propagation's SI actually carries). Hardcoding
    ``SI_TX_OSNR_DB`` here would silently alias every real tx_osnr value under
    a key claiming a tx_osnr no propagation in this adapter ever uses."""
    add_id, drop_id = _terminal_roadm_ids(model, oms_sequence, direction)
    add_r = model._roadms.get(add_id)
    drop_r = model._roadms.get(drop_id)
    return (
        add_r.add_drop_osnr_db if add_r is not None else None,
        drop_r.add_drop_osnr_db if drop_r is not None else None,
        baud_rate,
        tx_osnr_db,
    )


def compose_gsnr_db(
    increments: Sequence[Optional[float]], endpoint_lin: Optional[float], slot: int,
) -> Optional[float]:
    """Compose end-to-end GSNR (dB) at grid *slot* from per-OMS ``1/gsnr_lin``
    increments (one per hop, already indexed to *slot* by the caller) plus the
    analytic *endpoint_lin* noise term (also already indexed to *slot* by the
    caller -- it is typically constant across slots, but this function does
    not re-derive that indexing, only consumes it):

        N_total(slot) = endpoint_lin + sum(increments)
        GSNR_dB(slot) = -10 * log10(N_total(slot))

    Returns ``None`` if *endpoint_lin* or any increment is missing (e.g. a
    band-edge-filtered carrier absent from one hop's captured comb) rather
    than silently composing a partial, wrong total -- a missing hop is a
    "cannot compose" situation, not a zero contribution. *slot* is not read
    by the arithmetic itself (both other arguments are already sliced to it
    by the caller) but is kept as an explicit parameter so a call site stays
    self-documenting about which grid slot it is composing."""
    if endpoint_lin is None or any(v is None for v in increments):
        return None
    total = endpoint_lin + sum(increments)
    return -10.0 * math.log10(total)
