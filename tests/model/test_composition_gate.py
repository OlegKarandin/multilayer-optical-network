"""The safety valves around composed mode selection. Composition is an OPTIMISTIC
approximation; every one of these gates exists so an uncalibrated or out-of-range
question falls back to an exact propagation rather than getting a guessed answer."""
from unittest.mock import patch

import pytest

from multilayer_optical_network.gnpy_adapter.adapter import compute_qot
from multilayer_optical_network.gnpy_adapter.composition import (
    COMPOSITION_ERROR_BOUND_DB, MAX_COMPOSED_HOPS, oms_fingerprint,
)
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_network.model.allocation import (
    _best_feasible_mode, make_adapter_evaluator,
)
from multilayer_optical_network.model.assets import (
    Amplifier, Direction, Fiber, OMS, ROADM, Transceiver,
)
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.qot_results import (
    HarvestCache, IncrementCache, QoTResultStore,
)
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.testing import add_bidir_span, new_model

MODE = "400G"
GRID = SpectrumGrid.default()


class _FakeComposeQot:
    """A QotEvaluator whose `__call__` and `compose_gsnr` return deliberately
    DIFFERENT, easily distinguished GSNR values, so a test can tell which path
    `_best_feasible_mode` actually took just from the returned number."""

    def __init__(self, exact_gsnr: float, composed_gsnr):
        self.exact_gsnr = exact_gsnr
        self.composed_gsnr = composed_gsnr
        self.call_count = 0
        self.compose_count = 0

    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        self.call_count += 1
        return QoTState(gsnr_db=self.exact_gsnr, osnr_db=30.0, margin_db=0.0)

    def compose_gsnr(self, oms_sequence, direction, mode_id, slot):
        self.compose_count += 1
        return self.composed_gsnr


def _loading_with_probe(mode_id: str, probe_slot: int, slots=None) -> LoadingState:
    """Probe channel FIRST (mirrors allocation._build_loading's convention, which
    is what `_composed_worse_direction`'s "first mode_id match" probe-selection
    relies on), followed by the given *slots* (default: every grid slot -- a full
    comb)."""
    if slots is None:
        slots = range(GRID.num_slots)
    probe = Channel(GRID.freq(probe_slot), GRID.spacing_hz, None, mode_id)
    others = tuple(
        Channel(GRID.freq(s), GRID.spacing_hz, None, mode_id)
        for s in slots if s != probe_slot
    )
    return LoadingState((probe,) + others)


def _multi_oms_model():
    """A-B-C-D: three chained bidirectional spans, so the forward OMS sequence
    walks an interior (express) ROADM at B and C -- exercises a genuine
    multi-hop composition, not a degenerate single-hop one where "reverse chain"
    and "forward chain reversed" would coincide."""
    m = new_model()
    seq = (
        add_bidir_span(m, "A", "B", "oms1"),
        add_bidir_span(m, "B", "C", "oms2"),
        add_bidir_span(m, "C", "D", "oms3"),
    )
    return m, seq


def _bump_nf(model, oms_id, delta_db):
    """Mutate one amplifier's NF on *oms_id* only -- its paired reverse OMS has
    its own, untouched, amplifiers."""
    oms = model.get_oms(oms_id)
    amp_id = next(el for el in oms.elements if el in model._amplifiers)
    model.apply_nf_delta(amp_id, delta_db)


def _one_directional_model():
    """A single forward-only OMS (A -> B) with NO paired reverse (unlike every
    fixture built via `add_bidir_span`), so `reverse_oms_sequence` returns
    `None` for it -- the leg composition must refuse to guess for."""
    m = new_model()
    for node in ("A", "B"):
        m.add_roadm(ROADM(id=f"roadm_{node}"))
        m.add_transceiver(Transceiver(id=f"trx_{node}", site=node))
    m.add_amplifier(Amplifier(id="boost1", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    m.add_fiber(Fiber(id="f1", a_end="roadm_A", z_end="pre1",
                      length_km=80.0, type_variety="SSMF"))
    m.add_amplifier(Amplifier(id="pre1", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    m.add_oms(OMS(id="oms1", src_node_id="A", dst_node_id="B",
                  elements=("roadm_A", "boost1", "f1", "pre1")))
    return m


def test_actual_fill_policy_never_reaches_composition():
    """Hard policy gate. Under FillPolicy.ACTUAL the increments depend on the
    loading and the table is meaningless. _best_feasible_mode cannot see
    fill_policy -- only the resulting LoadingState -- so it detects a full comb
    the same way make_adapter_evaluator already does (len(slots) ==
    grid.num_slots) and composes only then."""
    n = new_model()
    qot = _FakeComposeQot(exact_gsnr=8.0, composed_gsnr=20.0)
    subset_loading = _loading_with_probe(MODE, 20, slots=(18, 19, 20, 21, 22))
    mode, gsnr = _best_feasible_mode(n, qot, ("oms1",), subset_loading, MODE)
    assert gsnr == 8.0                    # the EXACT value, not the composed one
    assert qot.compose_count == 0         # composition was never even attempted


def test_uncalibrated_oms_forces_an_exact_propagation():
    """A model with a fresh OMS (no increment recorded for its fingerprint) must
    NOT produce a composed answer. Counts propagations to prove one actually
    ran."""
    n = new_model()
    add_bidir_span(n, "A", "B", "oms1")
    qot = make_adapter_evaluator(n, QoTResultStore(), harvest_cache=HarvestCache(),
                                 increment_cache=IncrementCache())
    loading = _loading_with_probe(MODE, 20)

    import multilayer_optical_network.model.allocation as alloc
    with patch.object(alloc, "harvest_qot", wraps=alloc.harvest_qot) as spy:
        mode, gsnr = _best_feasible_mode(n, qot, ("oms1",), loading, MODE)
    assert spy.call_count == 1            # a REAL propagation ran -- nothing was cached
    assert mode is not None


def test_path_longer_than_the_calibrated_range_forces_exact():
    """MAX_COMPOSED_HOPS is a measured bound (Task A4), not an extrapolation. A
    longer path falls through rather than trusting the trend past where it was
    measured."""
    n = new_model()
    qot = make_adapter_evaluator(n, QoTResultStore(), increment_cache=IncrementCache())
    long_seq = tuple(f"leg-{i}" for i in range(MAX_COMPOSED_HOPS + 1))
    assert qot.compose_gsnr(long_seq, Direction.FORWARD, MODE, 20) is None


def test_backward_composition_walks_the_reverse_oms_chain():
    """CLAUDE.md's per-direction contract, and the S4-2/S4-3 rule the adapter
    already enforces for propagation: a backward composition must sum the
    REVERSE OMS chain, never the forward chain summed in reverse. Assert that
    an asymmetric NF bump on one direction changes exactly one direction's
    composed GSNR."""
    model, seq = _multi_oms_model()
    # add_bidir_span builds physically SYMMETRIC pairs (same NF/gain both ways),
    # so every forward OMS's fingerprint starts IDENTICAL to its own reverse
    # pair's -- IncrementCache would alias forward and reverse entries onto the
    # SAME cache slot for oms2/oms3 (unaffected by the bump below), and merely
    # re-harvesting forward would then overwrite those SHARED slots with a
    # position-in-chain-dependent value that differs by a few thousandths of a
    # dB from the original capture (composition's own measured, ~1.3%-per-hop
    # optimism -- see composition.py's module docstring) -- noise unrelated to
    # what this test checks. A tiny uniform pre-desync (every FORWARD OMS only)
    # makes every forward fingerprint permanently distinct from its reverse
    # pair's, so the two directions' cache entries never alias and the ONLY
    # thing that can move bwd_after is composition actually reading the wrong
    # (forward) chain for a backward query.
    for oms_id in seq:
        _bump_nf(model, oms_id, 0.05)
    qot = make_adapter_evaluator(model, QoTResultStore(), harvest_cache=HarvestCache(),
                                 increment_cache=IncrementCache())
    loading = _loading_with_probe(MODE, 20)
    slot = 20

    # Prime the increment table for both directions.
    qot(oms_sequence=seq, direction=Direction.FORWARD, mode_id=MODE, loading=loading)
    qot(oms_sequence=seq, direction=Direction.BACKWARD, mode_id=MODE, loading=loading)

    fwd_before = qot.compose_gsnr(seq, Direction.FORWARD, MODE, slot)
    bwd_before = qot.compose_gsnr(seq, Direction.BACKWARD, MODE, slot)
    assert fwd_before is not None
    assert bwd_before is not None

    _bump_nf(model, seq[0], 2.0)          # forward "oms1" only; "oms1_rev" untouched

    # Re-calibrate the forward direction under the new physics: a stale cached
    # increment for the OLD fingerprint is simply never looked up again
    # (content-addressed, no invalidation) -- a fresh harvest is required to
    # populate an entry under the NEW fingerprint.
    qot(oms_sequence=seq, direction=Direction.FORWARD, mode_id=MODE, loading=loading)

    fwd_after = qot.compose_gsnr(seq, Direction.FORWARD, MODE, slot)
    bwd_after = qot.compose_gsnr(seq, Direction.BACKWARD, MODE, slot)

    assert abs(fwd_after - fwd_before) > 1e-4     # forward direction moved
    assert bwd_after == pytest.approx(bwd_before, abs=1e-9)  # backward untouched --
    # proves composition walked "oms1_rev"/"oms2_rev"/"oms3_rev" (its own,
    # separately-tracked reverse chain), not the forward chain re-read backwards.


def test_leg_without_a_paired_reverse_oms_falls_through_to_exact():
    """reverse_oms_sequence returns None -- fall through, so the adapter raises
    its existing typed error rather than composition inventing a number."""
    model = _one_directional_model()
    ic = IncrementCache()
    ic.put(oms_fingerprint(model, "oms1"), {s: 1e-3 for s in range(GRID.num_slots)})
    qot = make_adapter_evaluator(model, QoTResultStore(), increment_cache=ic)

    # Direct unit check: compose_gsnr itself must refuse to guess.
    assert qot.compose_gsnr(("oms1",), Direction.BACKWARD, MODE, 20) is None

    # End-to-end: _best_feasible_mode falls through to the exact path, which
    # raises the SAME typed error compute_qot already raises for a leg with no
    # paired reverse -- composition must not swallow or mask it.
    loading = _loading_with_probe(MODE, 20)
    with pytest.raises(ValueError):
        _best_feasible_mode(model, qot, ("oms1",), loading, MODE)


def test_composed_selection_is_genuinely_feasible():
    """The safety theorem, empirically. For every mode selected from a composed
    GSNR, an exact propagation must confirm actual_gsnr >= required_gsnr_db with
    NO margin:
        composed >= required + design_margin
        actual   >= composed - COMPOSITION_ERROR_BOUND_DB
                 >= required + design_margin - bound  >  required
    holds only while design_margin_db > COMPOSITION_ERROR_BOUND_DB -- assert
    that relation explicitly too, so a future margin reduction fails loudly
    here."""
    model, seq = _multi_oms_model()
    model.design_margin_db = 0.5
    assert model.design_margin_db > COMPOSITION_ERROR_BOUND_DB

    qot = make_adapter_evaluator(model, QoTResultStore(), harvest_cache=HarvestCache(),
                                 increment_cache=IncrementCache())
    loading = _loading_with_probe(MODE, 20)
    slot = 20

    # Prime both directions' increment tables.
    qot(oms_sequence=seq, direction=Direction.FORWARD, mode_id=MODE, loading=loading)
    qot(oms_sequence=seq, direction=Direction.BACKWARD, mode_id=MODE, loading=loading)

    mode, composed_gsnr = _best_feasible_mode(model, qot, seq, loading, MODE)
    assert mode is not None
    assert composed_gsnr >= mode.required_gsnr_db + model.design_margin_db

    # Independent ground truth: a REAL, uncomposed propagation for both
    # directions of this exact path/loading.
    store = QoTResultStore()
    fwd, _ = compute_qot(model=model, store=store, oms_sequence=seq,
                         direction=Direction.FORWARD, mode_id=MODE, loading=loading,
                         center_freq_hz=GRID.freq(slot))
    bwd, _ = compute_qot(model=model, store=store, oms_sequence=seq,
                         direction=Direction.BACKWARD, mode_id=MODE, loading=loading,
                         center_freq_hz=GRID.freq(slot))
    actual_gsnr = min(fwd.gsnr_db, bwd.gsnr_db)

    assert actual_gsnr >= mode.required_gsnr_db          # genuinely feasible
    assert actual_gsnr >= composed_gsnr - COMPOSITION_ERROR_BOUND_DB
