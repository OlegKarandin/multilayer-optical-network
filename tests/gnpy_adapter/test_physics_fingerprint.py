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
import dataclasses

import pytest

from multilayer_optical_network.model.assets import (
    Amplifier, Direction, Fiber, FiberType, OMS, ROADM, Transceiver,
    TransceiverMode,
)
from multilayer_optical_network.model.modes import ModeRegistry
from multilayer_optical_network.model.network import NetworkModel
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.topology_import import model_from_abstract_graph
from multilayer_optical_network.gnpy_adapter.adapter import (
    _path_physical_fingerprint, harvest_cache_key, harvest_qot,
)
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
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


# ---------------------------------------------------------------------------
# Whole-branch review, Finding 3(a): the aliasing constraint, proven against
# real GNPy on two DIFFERENT paths.
#
# Global Constraints: "where a cache key is widened to alias two requests, a
# test must prove the two requests produce identical physics."
# test_per_direction.py covers forward-vs-backward of ONE path. Dropping
# `oms_sequence` widens the key further -- two DISTINCT, non-overlapping OMS
# chains alias whenever their physical parameters match -- and nothing proved
# GNPy actually returns the same numbers for that case. Every assertion above
# is on key equality alone; this one propagates.
# ---------------------------------------------------------------------------

def _twin_links_model():
    """Two disjoint links, A->B and C->D, built from identical span geometry:
    physically identical element chains that share no asset, no node, no OMS."""
    mode = TransceiverMode(id=MODE, bitrate_gbps=400.0, required_gsnr_db=7.1,
                           symbol_rate_baud=87.5e9, channel_spacing_hz=100e9)
    graph = {
        "nodes": [{"id": n} for n in ("A", "B", "C", "D")],
        "edges": [
            {"src": "A", "dst": "B", "length_km": 160.0,
             "span_lengths_km": [80.0, 80.0]},
            {"src": "C", "dst": "D", "length_km": 160.0,
             "span_lengths_km": [80.0, 80.0]},
        ],
    }
    return model_from_abstract_graph(graph, modes=ModeRegistry([mode]))


def _full_comb(grid):
    return LoadingState(tuple(
        Channel(grid.freq(s), grid.spacing_hz, None, MODE)
        for s in range(grid.num_slots)))


def test_two_distinct_but_identical_paths_really_do_propagate_identically():
    """The alias the widened key asserts, checked against GNPy itself: two
    different OMS chains sharing one harvest entry must be handed the same
    numbers they would each have computed on their own."""
    m = _twin_links_model()
    ab, cd = ("oms_A_B",), ("oms_C_D",)
    assert ab != cd and not set(m.get_oms(ab[0]).elements) & set(m.get_oms(cd[0]).elements)
    assert (harvest_cache_key(m, ab, Direction.FORWARD, MODE)
            == harvest_cache_key(m, cd, Direction.FORWARD, MODE)), (
        "physically identical distinct paths must share one harvest entry")

    grid = SpectrumGrid.default()
    comb = _full_comb(grid)
    vec_ab = harvest_qot(m, ab, Direction.FORWARD, MODE, comb)
    vec_cd = harvest_qot(m, cd, Direction.FORWARD, MODE, comb)

    assert vec_ab and set(vec_ab) == set(vec_cd)
    for slot in sorted(vec_ab):
        assert vec_ab[slot].gsnr_db == pytest.approx(vec_cd[slot].gsnr_db,
                                                     rel=0, abs=1e-12), (
            f"slot {slot}: aliased paths disagree on GSNR")
        assert vec_ab[slot].osnr_db == pytest.approx(vec_cd[slot].osnr_db,
                                                     rel=0, abs=1e-12), (
            f"slot {slot}: aliased paths disagree on OSNR")


def test_a_real_physical_difference_still_splits_the_two_paths():
    """The other half of the alias: it is keyed on physics, so a difference in
    physics must both re-split the key and move the propagated numbers."""
    m = _twin_links_model()
    ab, cd = ("oms_A_B",), ("oms_C_D",)
    m.apply_nf_delta("amp_C_D_0", 4.0)
    assert (harvest_cache_key(m, ab, Direction.FORWARD, MODE)
            != harvest_cache_key(m, cd, Direction.FORWARD, MODE))

    grid = SpectrumGrid.default()
    comb = _full_comb(grid)
    vec_ab = harvest_qot(m, ab, Direction.FORWARD, MODE, comb)
    vec_cd = harvest_qot(m, cd, Direction.FORWARD, MODE, comb)
    slot = sorted(vec_ab)[len(vec_ab) // 2]
    assert vec_cd[slot].gsnr_db < vec_ab[slot].gsnr_db - 0.1


# ---------------------------------------------------------------------------
# Whole-branch review, Finding 3(b): completeness guard on the projection.
#
# `_path_physical_fingerprint` used to embed the whole frozen dataclass, so a
# new physics field was picked up automatically. It now projects field by
# field, and a field added to any of these dataclasses without a matching edit
# to the projection would silently drop out of the key -- a hit that returns a
# confident wrong number. These two tests fail in exactly that case.
# ---------------------------------------------------------------------------

# {dataclass: (fields the projection embeds, fields legitimately excluded)}.
# The only legitimate exclusions are IDENTITY: the whole point of the
# projection is that no asset/node id reaches the key (see
# test_fingerprint_carries_no_asset_or_node_ids). `Fiber.a_end`/`z_end` are
# element ids too -- the chain's ORDER already comes from OMS.elements, which
# the fingerprint walks. Everything else is physics and must be embedded.
_PROJECTION = {
    Amplifier: ({"type_variety", "gain_db", "nf_db", "tilt_db"}, {"id"}),
    Fiber: ({"length_km", "extra_loss_db", "type_variety"},
            {"id", "a_end", "z_end"}),
    FiberType: ({"type_variety", "loss_coef_db_per_km", "dispersion",
                 "effective_area", "pmd_coef"}, set()),
    ROADM: ({"target_pch_out_db", "add_drop_osnr_db"}, {"id"}),
}


@pytest.mark.parametrize("cls", list(_PROJECTION), ids=lambda c: c.__name__)
def test_fingerprint_projection_covers_every_field(cls):
    """Add a physics field to one of these dataclasses and this fails until the
    field is also added to `_path_physical_fingerprint` (and to `_PROJECTION`
    below it, whose entries the next test proves are really embedded)."""
    projected, excluded = _PROJECTION[cls]
    declared = {f.name for f in dataclasses.fields(cls)}
    assert projected | excluded == declared, (
        f"{cls.__name__}: fields unaccounted for by the physics fingerprint: "
        f"{sorted(declared - (projected | excluded))} -- add each to "
        f"_path_physical_fingerprint (physics) or to the excluded set (identity)")
    assert not (projected & excluded)


def _perturb(value):
    return value + 1.0 if isinstance(value, (int, float)) else f"{value}-x"


_FP_ARGS = ("oms_A_B",), Direction.FORWARD


def _mutate(m, cls, name):
    if cls is Amplifier:
        a = m._amplifiers["amp_A_B_0"]
        m._amplifiers[a.id] = dataclasses.replace(
            a, **{name: _perturb(getattr(a, name))})
    elif cls is ROADM:
        r = m._roadms["roadm_A"]
        m._roadms[r.id] = dataclasses.replace(
            r, **{name: _perturb(getattr(r, name))})
    elif cls is FiberType:
        # Re-stored under the ORIGINAL registry key, so the fiber's lookup still
        # resolves and the ONLY thing that moves is the stored type's own field.
        ft = m._fiber_types["SSMF"]
        m._fiber_types["SSMF"] = dataclasses.replace(
            ft, **{name: _perturb(getattr(ft, name))})
    elif cls is Fiber:
        f = m._fibers["fiber_A_B_0"]
        if name == "type_variety":
            # A second, NUMERICALLY IDENTICAL type under a different name: the
            # only difference is which type the fiber names, isolating
            # Fiber.type_variety from every FiberType field.
            twin = dataclasses.replace(m._fiber_types["SSMF"],
                                       type_variety="SSMF-twin")
            m.register_fiber_type(twin)
            m._fibers[f.id] = dataclasses.replace(f, type_variety="SSMF-twin")
        else:
            m._fibers[f.id] = dataclasses.replace(
                f, **{name: _perturb(getattr(f, name))})
    else:                                          # pragma: no cover - guard
        raise AssertionError(f"no mutator for {cls!r}")


@pytest.mark.parametrize(
    "cls,name",
    [(c, n) for c, (proj, _) in _PROJECTION.items() for n in sorted(proj)],
    ids=lambda v: v.__name__ if isinstance(v, type) else v,
)
def test_every_projected_field_actually_moves_the_fingerprint(cls, name):
    """The behavioural half of the guard: `_PROJECTION` is a claim about the
    projection code, so each field it lists must demonstrably be in the key.
    Listing a field there without wiring it into `_path_physical_fingerprint`
    fails here rather than passing the completeness test vacuously."""
    base = _path_physical_fingerprint(_line_ab_model(), *_FP_ARGS)
    m = _line_ab_model()
    _mutate(m, cls, name)
    assert _path_physical_fingerprint(m, *_FP_ARGS) != base, (
        f"{cls.__name__}.{name} is listed as projected but does not reach the "
        f"physics fingerprint")
