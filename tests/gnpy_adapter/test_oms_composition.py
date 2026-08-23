"""Per-OMS increment extraction: difference accumulated 1/gsnr_lin at OMS boundaries
INSIDE one propagation.

This is the route the 2026-08-20 investigation identified as correct, and it exists
specifically to avoid the false start recorded in that document: deriving an OMS's
noise by calling harvest_qot on a single-OMS path runs it as a COMPLETE standalone
lightpath (transmitter, add ROADM, span chain, terminal drop ROADM -- the S4-4 rule at
adapter.py:303-309), so summing k of them counts the endpoint chain k times (-1.23 dB
at 2 hops, -1.64 dB at 3). Correcting with add_drop_osnr recovers ~0.1 dB. Difference
inside a propagation; never fit a constant."""
import math

import pytest

from multilayer_optical_network.gnpy_adapter.adapter import (
    _apply_penalties, _extract_gsnr_osnr, _propagate_loading,
)
from multilayer_optical_network.gnpy_adapter.composition import (
    compose_gsnr_db, endpoint_noise_lin, oms_fingerprint,
)
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_network.gnpy_adapter.translate import reverse_oms_sequence
from multilayer_optical_network.model.assets import Direction
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.testing import add_bidir_span, new_model

MODE = "400G"
# Central grid slots, well away from the C-band edges the default topology's
# amp/ROADM passbands filter -- see harvest_qot's "may be missing entries at
# the band edge" note.
_SLOTS = (18, 19, 20, 21, 22)


def _multi_oms_model():
    """A-B-C-D: three chained bidirectional spans, so the forward OMS sequence
    walks an interior (express) ROADM at B and at C -- the case the composition
    identity depends on ("an express ROADM preserves 1/gsnr_lin exactly")."""
    m = new_model()
    seq = (
        add_bidir_span(m, "A", "B", "oms1"),
        add_bidir_span(m, "B", "C", "oms2"),
        add_bidir_span(m, "C", "D", "oms3"),
    )
    return m, seq


def _reverse_of(model, oms_id):
    seq = reverse_oms_sequence(model, (oms_id,))
    assert seq is not None
    return seq[0]


def _bump_nf(model, oms_id, delta_db):
    """Mutate one amplifier's NF on *oms_id* only (its paired reverse OMS has
    its own, untouched, amplifiers)."""
    oms = model.get_oms(oms_id)
    amp_id = next(el for el in oms.elements if el in model._amplifiers)
    model.apply_nf_delta(amp_id, delta_db)


def _first_roadm(model, seq):
    oms = model.get_oms(seq[0])
    return next(el for el in oms.elements if el in model._roadms)


def _last_roadm(model, seq):
    return f"roadm_{model.get_oms(seq[-1]).dst_node_id}"


def _full_comb_loading(grid):
    return LoadingState(tuple(
        Channel(grid.freq(s), grid.spacing_hz, None, MODE) for s in _SLOTS
    ))


def _harvest_with_increments(model, oms_sequence, direction, mode_id):
    """Thin wrapper mirroring harvest_qot's per-slot extraction, but calling
    _propagate_loading with capture_increments=True and returning the captured
    per-OMS increments and endpoint term alongside the usual slot -> QoTState
    vector -- so the self-consistency test below can verify that increments
    harvested from a propagation reproduce that SAME propagation's own GSNR."""
    grid = SpectrumGrid.default()
    mode = model.modes.get(mode_id)
    loading = _full_comb_loading(grid)

    pr = _propagate_loading(model, oms_sequence, direction, loading, mode,
                            probe_idx=0, capture_increments=True)

    vec: dict[int, QoTState] = {}
    for i, freq_hz in enumerate(pr.si.frequency):
        gsnr_db, osnr_db = _extract_gsnr_osnr(pr.si, i)
        gsnr_db, osnr_db = _apply_penalties(
            pr.si, i, pr.uids_list, pr.elements, pr.roadm_propagated,
            pr.baud_rate, gsnr_db, osnr_db)
        slot = grid.slot_of(float(freq_hz))
        vec[slot] = QoTState(
            gsnr_db=gsnr_db, osnr_db=osnr_db,
            margin_db=gsnr_db - mode.required_gsnr_db - model.design_margin_db,
            limiting_element_id=None,
        )
    return vec, pr.oms_increments, pr.endpoint_lin


def test_increments_sum_to_the_propagated_gsnr_on_the_same_path():
    """Self-consistency, the tightest possible check: the increments harvested FROM a
    propagation must, when summed and combined with the endpoint term, reproduce that
    same propagation's per-slot GSNR to within floating-point noise. Error here is a
    bookkeeping bug, not model error -- there is no approximation on one path."""
    model, seq = _multi_oms_model()
    vec, incs, endp = _harvest_with_increments(model, seq, Direction.FORWARD, MODE)
    assert vec                                    # sanity: something propagated
    for slot, state in vec.items():
        composed = compose_gsnr_db(
            [incs[o][slot] for o in seq], endp[slot], slot)
        assert composed == pytest.approx(state.gsnr_db, abs=1e-6), slot


def test_oms_fingerprint_splits_on_asymmetric_degradation():
    """Content-addressed keying, the HarvestCache property: an asymmetric
    inject_degradation must give the forward OMS and its paired reverse OMS DIFFERENT
    keys, so the table re-splits the directions on its own with no invalidation logic."""
    model, seq = _multi_oms_model()
    fwd = oms_fingerprint(model, seq[0])
    rev = oms_fingerprint(model, _reverse_of(model, seq[0]))
    assert fwd == rev                                   # symmetric to start with
    _bump_nf(model, seq[0], 2.0)                         # one direction only
    assert oms_fingerprint(model, seq[0]) != fwd
    assert oms_fingerprint(model, _reverse_of(model, seq[0])) == rev


def test_endpoint_term_is_analytic_not_fitted():
    """The endpoint term is db2lin(-(add_drop_osnr + lin2db(2))) per TERMINAL ROADM
    plus db2lin(-tx_osnr), renormalised from gnpy's 12.5 GHz reference to the symbol
    rate (x7.0 at 87.5 GBaud -- the factor the earlier curve fit was absorbing). Assert
    it against the closed form, and against the gap between a propagation's last
    pre-penalty snapshot and its returned GSNR."""
    model, seq = _multi_oms_model()
    got = endpoint_noise_lin(model, seq, Direction.FORWARD,
                             baud_rate=87.5e9, tx_osnr_db=40.0)
    add_drop = model._roadms[_first_roadm(model, seq)].add_drop_osnr_db
    drop = model._roadms[_last_roadm(model, seq)].add_drop_osnr_db
    expect = (10 ** (-(add_drop + 10 * math.log10(2.0)) / 10)
              + 10 ** (-(drop + 10 * math.log10(2.0)) / 10)
              + 10 ** (-40.0 / 10)) * (87.5e9 / 12.5e9)
    assert got == pytest.approx(expect, rel=1e-9)

    # Cross-check against a REAL propagation, independent of the increment
    # bookkeeping above: the gap between _propagate_loading's own pre-penalty
    # final_gsnr_db and _apply_penalties's penalty-applied GSNR, for one probe
    # carrier, must be explained EXACTLY by this same analytic endpoint term --
    # evaluated at the tx_osnr ACTUALLY baked into the propagated SI
    # (build_si_for_loading's own default, 35 dB; NOT the equipment SI block's
    # declared 40, which this direct-propagation adapter never reads).
    mode = model.modes.get(MODE)
    grid = SpectrumGrid.default()
    loading = _full_comb_loading(grid)
    probe_idx = 0
    pr = _propagate_loading(model, seq, Direction.FORWARD, loading, mode, probe_idx)
    penalized_gsnr_db, _ = _apply_penalties(
        pr.si, probe_idx, pr.uids_list, pr.elements, pr.roadm_propagated,
        pr.baud_rate, pr.final_gsnr_db, pr.final_osnr_db)
    raw_inv_lin = 10 ** (-pr.final_gsnr_db / 10)
    penalized_inv_lin = 10 ** (-penalized_gsnr_db / 10)
    real_tx_osnr_db = float(pr.si.tx_osnr[probe_idx])
    endpoint_at_real_tx_osnr = endpoint_noise_lin(
        model, seq, Direction.FORWARD, baud_rate=87.5e9, tx_osnr_db=real_tx_osnr_db)
    assert (penalized_inv_lin - raw_inv_lin) == pytest.approx(
        endpoint_at_real_tx_osnr, rel=1e-6)
