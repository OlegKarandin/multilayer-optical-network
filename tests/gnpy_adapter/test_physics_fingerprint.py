"""Finding A: the harvest key is content-addressed on PHYSICS ONLY.

A bidirectional link is two physically distinct fiber runs, but on an
undamaged span they carry identical parameters and GNPy is deterministic, so
forward and backward propagate to the same GSNR exactly (delta 0.00e+00,
docs/2026-08-20-allocation-qot-performance-findings.md section 4). Keying the
harvest on physics instead of ids collapses those two propagations into one.

CLAUDE.md's per-direction contract survives untouched, and this file proves
it: an asymmetric impairment makes the two directions' physical parts diverge
and the key splits again with no invalidation logic.
"""
from multilayer_optical_network.model.assets import (
    Amplifier, Direction, Fiber, FiberType, OMS, ROADM, Transceiver,
    TransceiverMode,
)
from multilayer_optical_network.model.modes import ModeRegistry
from multilayer_optical_network.model.network import NetworkModel
from multilayer_optical_network.gnpy_adapter.adapter import (
    _path_physical_fingerprint, harvest_cache_key,
)
from tests.gnpy_adapter.test_reverse_oms import MODE, _line_ab_model


def test_symmetric_span_shares_one_harvest_key_across_directions():
    model = _line_ab_model()
    fwd = harvest_cache_key(model, ("oms_A_B",), Direction.FORWARD, MODE)
    bwd = harvest_cache_key(model, ("oms_A_B",), Direction.BACKWARD, MODE)
    assert fwd == bwd, (
        "an undamaged bidirectional link is physically symmetric, so both "
        "directions must share one harvest entry")


def test_asymmetric_nf_delta_resplits_the_harvest_key():
    model = _line_ab_model()
    fwd0 = harvest_cache_key(model, ("oms_A_B",), Direction.FORWARD, MODE)
    model.apply_nf_delta("amp_B_A_0", 5.0)     # reverse chain only
    fwd1 = harvest_cache_key(model, ("oms_A_B",), Direction.FORWARD, MODE)
    bwd1 = harvest_cache_key(model, ("oms_A_B",), Direction.BACKWARD, MODE)
    assert fwd1 == fwd0, "a reverse-chain impairment must not move forward's key"
    assert bwd1 != fwd1, (
        "an asymmetric impairment must re-split the two directions' keys with "
        "no explicit invalidation -- the per-direction contract, preserved")


def test_forward_key_still_misses_on_each_physics_input():
    """Fingerprint completeness: omitting a GSNR input means a hit returns a
    confident wrong number (see QoTCache's docstring)."""
    base = harvest_cache_key(_line_ab_model(), ("oms_A_B",),
                             Direction.FORWARD, MODE)

    m = _line_ab_model()
    m.apply_nf_delta("amp_A_B_0", 3.0)
    assert harvest_cache_key(m, ("oms_A_B",), Direction.FORWARD, MODE) != base

    m = _line_ab_model()
    m.apply_loss_delta("fiber_A_B_0", 2.0)
    assert harvest_cache_key(m, ("oms_A_B",), Direction.FORWARD, MODE) != base

    m = _line_ab_model()
    # _line_ab_model() registers only MODE, so a second mode is not available
    # from the registry to derive "other" from. harvest_cache_key never
    # validates mode_id against the registry -- it is embedded verbatim in
    # the returned tuple -- so a literal distinct id exercises the same
    # "mode differs -> key differs" contract without needing one registered.
    other = "800G@8.0dB"
    assert other != MODE
    assert harvest_cache_key(m, ("oms_A_B",), Direction.FORWARD, other) != base


def test_fingerprint_carries_no_asset_or_node_ids():
    """The projection is the whole point: ids in the key are what stopped the
    two directions from aliasing."""
    model = _line_ab_model()
    fp = _path_physical_fingerprint(model, ("oms_A_B",), Direction.FORWARD)
    flat = [x for part in fp for x in part if isinstance(x, str)]
    for token in ("oms_A_B", "amp_A_B_0", "fiber_A_B_0", "roadm_A", "roadm_B",
                  "A", "B"):
        assert token not in flat, f"identity leaked into the fingerprint: {token}"


def _unpaired_model():
    """A model with oms_A_B and NO oms_B_A. Built by hand: the importer always
    creates both directions (_add_directed_oms), which is exactly why this
    case needs an explicit guard rather than a fixture."""
    mode = TransceiverMode(id=MODE, bitrate_gbps=400.0, required_gsnr_db=7.1,
                           symbol_rate_baud=87.5e9, channel_spacing_hz=100e9)
    m = NetworkModel(modes=ModeRegistry([mode]))
    m.register_fiber_type(FiberType(type_variety="SSMF", loss_coef_db_per_km=0.2))
    m.add_roadm(ROADM(id="roadm_A"))
    m.add_roadm(ROADM(id="roadm_B"))
    m.add_transceiver(Transceiver(id="trx_A", site="A"))
    m.add_transceiver(Transceiver(id="trx_B", site="B"))
    m.add_fiber(Fiber(id="fiber_A_B_0", a_end="A", z_end="B", length_km=80.0,
                      type_variety="SSMF"))
    m.add_amplifier(Amplifier(id="amp_A_B_0", type_variety="advanced_toy",
                              gain_db=16.0, nf_db=5.5))
    m.add_oms(OMS(id="oms_A_B", src_node_id="A", dst_node_id="B",
                  elements=("roadm_A", "fiber_A_B_0", "amp_A_B_0")))
    return m


def test_unpaired_reverse_never_aliases_forward():
    """With `direction` gone from the harvest key, the old silent fall-back to
    the forward sequence would serve a forward answer to a request
    _propagate_loading is required to reject."""
    m = _unpaired_model()
    fwd = harvest_cache_key(m, ("oms_A_B",), Direction.FORWARD, MODE)
    bwd = harvest_cache_key(m, ("oms_A_B",), Direction.BACKWARD, MODE)
    assert bwd != fwd
