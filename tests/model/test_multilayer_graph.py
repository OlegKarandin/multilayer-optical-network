# tests/model/test_multilayer_graph.py
"""Layered auxiliary graph: existing lightpaths -> LPE edges (residual,
margin-gated); free wavelengths -> WLE edges driven from the OMS bitmask."""
from multilayer_optical_network.model.assets import FiberType, Fiber, Amplifier, OMS, ROADM, Lightpath, TransceiverMode
from multilayer_optical_network.model.ip_assets import Router, IPLink
from multilayer_optical_network.model.modes import ModeRegistry
from multilayer_optical_network.model.network import NetworkModel
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.multilayer_graph import (
    build_layered_graph, ACCESS, WLIN, WLOUT, lpe_edges, wle_count_on_layer, place_demands,
)
from multilayer_optical_network.model.topology_import import model_from_abstract_graph
from multilayer_optical_network.model import multilayer_graph as mg


class _ConstQot:
    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        return QoTState(gsnr_db=16.0, osnr_db=30.0, margin_db=0.0)


def _line3() -> NetworkModel:
    """Importer-built 0-1-2 line: both directed OMS per fiber, so a 0->2 demand
    routes 0->1->2 and its new lightpath must chain oms_0_1 then oms_1_2."""
    graph = {
        "nodes": [{"id": i} for i in range(3)],
        "edges": [
            {"src": 0, "dst": 1, "length_km": 80.0},
            {"src": 1, "dst": 2, "length_km": 80.0},
        ],
    }
    return model_from_abstract_graph(graph, modes=ModeRegistry([
        TransceiverMode(id="400G", bitrate_gbps=400.0, required_gsnr_db=7.1,
                        symbol_rate_baud=87.5e9, channel_spacing_hz=100e9)]))


def test_new_runs_are_direction_contiguous_through_intermediate_node():
    """Regression: place_demands must not emit a new lightpath whose oms_sequence
    walks OMS against their travel direction (multilayer_graph:174 added each OMS
    in BOTH directions, so a 0->2 route could pick oms_1_0 for the 0->1 hop)."""
    m = _line3()
    g = build_layered_graph(m)
    cands = place_demands(m, g, _ConstQot(), src="0", dst="2",
                          demand_gbps=100.0, policy="new_only", k=8)
    assert cands
    for p in cands:
        for run in p.new_lightpaths:
            seq = [m.get_oms(o) for o in run.oms_sequence]
            assert seq[0].src_node_id == run.src_node
            assert seq[-1].dst_node_id == run.dst_node
            for x, y in zip(seq, seq[1:]):
                assert x.dst_node_id == y.src_node_id, run.oms_sequence


def _one_lightpath_model(margin_db: float = 3.0) -> NetworkModel:
    n = NetworkModel(modes=ModeRegistry([
        TransceiverMode(id="100G", bitrate_gbps=100.0, required_gsnr_db=12.0,
                        symbol_rate_baud=32e9, channel_spacing_hz=100e9)]))
    n.register_fiber_type(FiberType("SSMF", 0.2))
    for a in ("a1", "a2"):
        n.add_amplifier(Amplifier(id=a, type_variety="advanced_toy", gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber("fAB", "a1", "a2", 80.0, "SSMF"))
    for node in ("A", "B"):
        n.add_roadm(ROADM(id=f"roadm_{node}"))
    n.add_oms(OMS("oms-AB", "A", "B", ("roadm_A", "a1", "fAB", "a2")))
    # Lightpath on slot 20 (193.4 THz on the default grid).
    n.add_lightpath(Lightpath("lp-AB", ("oms-AB",), "100G", 193.4e12))
    n.set_qot_state("lp-AB", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=margin_db))
    n.add_router(Router("R1", "A"))
    n.add_router(Router("R2", "B"))
    n.add_ip_link(IPLink("ip-AB", "R1", "R2", "lp-AB"))
    return n


def test_existing_lightpath_becomes_lpe_with_residual():
    n = _one_lightpath_model()
    g = build_layered_graph(n)
    edges = lpe_edges(g)
    assert len(edges) == 1
    (u, v, data), = edges
    assert u == (ACCESS, "A") and v == (ACCESS, "B")
    assert data["lightpath_id"] == "lp-AB"
    assert data["residual_gbps"] == 100.0     # 100G mode, no load


def test_margin_negative_lightpath_has_no_lpe_edge():
    n = _one_lightpath_model(margin_db=-1.0)   # down -> capacity 0
    g = build_layered_graph(n)
    assert lpe_edges(g) == []


def test_wle_present_only_on_free_slots():
    n = _one_lightpath_model()
    SpectrumGrid.default()
    g = build_layered_graph(n)
    # slot 20 is occupied by lp-AB on oms-AB -> no WLE on layer 20
    assert wle_count_on_layer(g, "oms-AB", 20) == 0
    # slot 0 is free -> one WLE on layer 0 for oms-AB (its A->B direction only)
    assert wle_count_on_layer(g, "oms-AB", 0) == 1


def test_forbidden_asset_drops_lpe_and_wle():
    n = _one_lightpath_model()
    g = build_layered_graph(n, forbidden_assets=frozenset({"fAB"}))
    assert lpe_edges(g) == []                  # lightpath crosses fAB
    assert wle_count_on_layer(g, "oms-AB", 0) == 0   # OMS pruned entirely


# ---------------------------------------------------------------------------
# place_demands tests
# ---------------------------------------------------------------------------

from multilayer_optical_network.model.assets import FiberType as _FT, Fiber as _F, Amplifier as _A, OMS as _O, Lightpath as _L
from multilayer_optical_network.model.ip_assets import Router as _R, IPLink as _I, Service


class FakeQot:
    def __init__(self, gsnr): self._g = gsnr
    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        return QoTState(gsnr_db=self._g, osnr_db=30.0, margin_db=0.0)


def test_groom_only_reuses_existing_lightpath():
    n = _one_lightpath_model()
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=40.0, policy="groom_only")
    assert res, "expected at least one placement"
    assert res[0].reused_lightpaths == ("lp-AB",)
    assert res[0].new_lightpaths == ()
    assert res[0].restored_gbps == 40.0          # fits in 100G residual
    assert res[0].shortfall_gbps == 0.0


def test_groom_only_degrades_to_bottleneck_residual():
    n = _one_lightpath_model()
    # load 70G onto the IP link via a background service so residual is 30G < demand
    n.add_service(Service("s-load", "R1", "R2", 70.0, working_path=("ip-AB",)))
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=40.0, policy="groom_only")
    assert res[0].restored_gbps == 30.0
    assert res[0].shortfall_gbps == 10.0


def test_new_only_lights_new_lightpath_ignoring_existing():
    n = _one_lightpath_model()
    # new_only drops LPE (grooming) edges, so the A->B demand cannot reuse lp-AB
    # and must light a FRESH A->B lightpath on a free slot.
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only")
    assert res
    assert res[0].reused_lightpaths == ()
    assert len(res[0].new_lightpaths) == 1
    assert res[0].new_lightpaths[0].oms_sequence == ("oms-AB",)
    assert res[0].restored_gbps == 100.0


def _symmetric_two_node() -> NetworkModel:
    """A<->B with BOTH directed OMS, as a real importer builds — so a B->A demand
    routes over the B->A OMS rather than traversing an A->B OMS backwards."""
    n = NetworkModel(modes=ModeRegistry([_TM_helper()]))
    n.register_fiber_type(_FT("SSMF", 0.2))
    for a in ("s1", "s2", "s3", "s4"):
        n.add_amplifier(_A(a, "advanced_toy", 20.0, 5.5))
    n.add_fiber(_F("fAB", "s1", "s2", 80.0, "SSMF"))
    n.add_fiber(_F("fBA", "s3", "s4", 80.0, "SSMF"))
    for node in ("A", "B"):
        n.add_roadm(ROADM(id=f"roadm_{node}"))
    n.add_oms(_O("oms-AB", "A", "B", ("roadm_A", "s1", "fAB", "s2")))
    n.add_oms(_O("oms-BA", "B", "A", ("roadm_B", "s3", "fBA", "s4")))
    return n


def test_new_run_records_travel_endpoints():
    """A new run records its travel endpoints (src_node/dst_node), which drive
    provisioning. A B->A demand lights a lightpath over the B->A OMS."""
    n = _symmetric_two_node()
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="B", dst="A",
                        demand_gbps=100.0, policy="new_only")
    run = res[0].new_lightpaths[0]
    assert run.oms_sequence == ("oms-BA",)     # travels the B->A OMS
    assert run.src_node == "B"
    assert run.dst_node == "A"


def test_groom_only_empty_when_no_existing_path():
    n = _one_lightpath_model()
    g = build_layered_graph(n)
    assert place_demands(n, g, FakeQot(15.0), src="B", dst="A",
                         demand_gbps=10.0, policy="groom_only") == []


def _groom_plus_gap_model() -> NetworkModel:
    """A->M has an existing lightpath (lp-AM); M->B has free spectrum but NO
    existing lightpath. Demand A->B must groom A->M then light a new M->B
    lightpath -> a hybrid placement."""
    n = NetworkModel(modes=ModeRegistry([
        _TM_helper()]))
    n.register_fiber_type(_FT("SSMF", 0.2))
    for a in ("m1", "m2", "n1", "n2"):
        n.add_amplifier(_A(a, "advanced_toy", 20.0, 5.5))
    n.add_fiber(_F("fAM", "m1", "m2", 60.0, "SSMF"))
    n.add_fiber(_F("fMB", "n1", "n2", 60.0, "SSMF"))
    n.add_oms(_O("oms-AM", "A", "M", ("m1", "fAM", "m2")))
    n.add_oms(_O("oms-MB", "M", "B", ("n1", "fMB", "n2")))
    n.add_lightpath(_L("lp-AM", ("oms-AM",), "100G", 193.4e12))
    n.set_qot_state("lp-AM", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    n.add_router(_R("RA", "A"))
    n.add_router(_R("RM", "M"))
    n.add_ip_link(_I("ip-AM", "RA", "RM", "lp-AM"))
    return n


def _TM_helper():
    from multilayer_optical_network.model.assets import TransceiverMode
    return TransceiverMode(id="100G", bitrate_gbps=100.0, required_gsnr_db=12.0,
                           symbol_rate_baud=32e9, channel_spacing_hz=100e9)


def _cheap_route_plus_distinct_route() -> NetworkModel:
    """A cheap 1-hop A->B route and a distinct 2-hop A->C->B route. Driven with a
    wide grid, the cheap route's lambda-variants exceed the raw-path budget, so a
    naive (per-emission) budget never reaches the strictly-more-expensive 2-hop
    route."""
    n = NetworkModel(modes=ModeRegistry([_TM_helper()]))
    n.register_fiber_type(_FT("SSMF", 0.2))
    for a in ("aba", "abz", "aca", "acz", "cba", "cbz"):
        n.add_amplifier(_A(a, "advanced_toy", 20.0, 5.5))
    n.add_fiber(_F("fab", "aba", "abz", 80.0, "SSMF"))
    n.add_fiber(_F("fac", "aca", "acz", 80.0, "SSMF"))
    n.add_fiber(_F("fcb", "cba", "cbz", 80.0, "SSMF"))
    n.add_oms(_O("oms-AB", "A", "B", ("aba", "fab", "abz")))
    n.add_oms(_O("oms-AC", "A", "C", ("aca", "fac", "acz")))
    n.add_oms(_O("oms-CB", "C", "B", ("cba", "fcb", "cbz")))
    return n


def test_new_only_budget_not_starved_by_wavelength_variants():
    """The distinct 2-hop route must be reachable even though >_PATH_BUDGET
    lambda-variants of the cheaper 1-hop route precede it in weight order (they
    would exhaust a raw-per-emission budget before the distinct route is seen)."""
    from multilayer_optical_network.model.spectrum import SpectrumGrid
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=80)
    n = _cheap_route_plus_distinct_route()
    g = build_layered_graph(n, grid=grid)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only", grid=grid)
    routes = {p.new_lightpaths[0].oms_sequence for p in res if p.new_lightpaths}
    assert ("oms-AC", "oms-CB") in routes, routes


def _two_parallel_oms_model() -> NetworkModel:
    """Two parallel A->B OMS (oms-AB-1, oms-AB-2) on physically-separate fibers,
    no existing lightpath. A new A->B lightpath can be lit on EITHER fiber, so
    both routes must be enumerable. On a plain nx.DiGraph the two WLE edges share
    the ordered vertex pair ((WL,A,lam)->(WL,B,lam)) per slot and the second
    overwrites the first (S7-13), collapsing them to one route."""
    n = NetworkModel(modes=ModeRegistry([_TM_helper()]))
    n.register_fiber_type(_FT("SSMF", 0.2))
    for a in ("p1a", "p1z", "p2a", "p2z"):
        n.add_amplifier(_A(a, "advanced_toy", 20.0, 5.5))
    n.add_fiber(_F("fab1", "p1a", "p1z", 80.0, "SSMF"))
    n.add_fiber(_F("fab2", "p2a", "p2z", 80.0, "SSMF"))
    n.add_oms(_O("oms-AB-1", "A", "B", ("p1a", "fab1", "p1z")))
    n.add_oms(_O("oms-AB-2", "A", "B", ("p2a", "fab2", "p2z")))
    return n


def test_parallel_oms_both_routes_enumerable():
    """S7-13: both parallel A->B fibers must surface as distinct new-lightpath
    routes; the DiGraph WL-layer overwrite silently kept only the last-added one."""
    n = _two_parallel_oms_model()
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only")
    routes = {p.new_lightpaths[0].oms_sequence for p in res if p.new_lightpaths}
    assert routes == {("oms-AB-1",), ("oms-AB-2",)}, routes


def test_wle_count_counts_parallel_oms_per_layer():
    """Each parallel OMS contributes its own WLE edge per free slot (one, in its
    A->B direction); the DiGraph overwrite would collapse them to a single edge."""
    n = _two_parallel_oms_model()
    g = build_layered_graph(n)
    assert wle_count_on_layer(g, "oms-AB-1", 0) == 1
    assert wle_count_on_layer(g, "oms-AB-2", 0) == 1


def _f(slot: int) -> float:
    """Center frequency of a default-grid slot."""
    return SpectrumGrid.default().freq(slot)


def _line4_model() -> NetworkModel:
    """Importer-built A-B-C-D line, one directed OMS per hop in each direction.
    Used for the incomparable-signatures case (spec §3.1's counter-example)."""
    graph = {
        "nodes": [{"id": n} for n in ("A", "B", "C", "D")],
        "edges": [
            {"src": "A", "dst": "B", "length_km": 80.0},
            {"src": "B", "dst": "C", "length_km": 80.0},
            {"src": "C", "dst": "D", "length_km": 80.0},
        ],
    }
    return model_from_abstract_graph(graph, modes=ModeRegistry([
        TransceiverMode(id="100G", bitrate_gbps=100.0, required_gsnr_db=12.0,
                        symbol_rate_baud=32e9, channel_spacing_hz=100e9)]))


# ---------------------------------------------------------------------------
# Node-split + dominance-maximal wavelength layers. Each optical node has a
# WLin/WLout port pair per LAYER joined by an EXPRESS edge, so a segmented
# placement terminates at (WLin,n,c) and re-originates at (WLout,n,c) — distinct
# vertices — and can ride ONE wavelength. A layer is built per ⊆-MAXIMAL free-slot
# signature, not per slot: if every OMS free on slot A is also free on slot B, every
# route liftable onto layer A lifts onto layer B at identical cost (all WLE weigh
# _W_WLE), so layer A contributes nothing and is dropped.
# ---------------------------------------------------------------------------

def _wle_classes(g):
    return sorted({d["class_id"] for _, _, d in g.edges(data=True)
                   if d.get("kind") == "WLE"})


def _class_sigs(g):
    return {c.oms_ids for c in g.graph["slot_classes"]}


def test_empty_network_builds_exactly_one_layer():
    """Every slot is free on every OMS, so all signatures are equal AND maximal:
    one class, one layer."""
    n = _two_parallel_oms_model()
    g = build_layered_graph(n)
    assert _wle_classes(g) == [0]
    (cls,) = g.graph["slot_classes"]
    assert cls.oms_ids == {"oms-AB-1", "oms-AB-2"}
    assert len(cls.slots) == SpectrumGrid.default().num_slots


def test_dominated_signature_is_dropped():
    """Slot 0 lit on oms-AB-1 -> signature(0)={oms-AB-2}; every other slot is free on
    both -> signature={oms-AB-1,oms-AB-2}. {oms-AB-2} is a strict subset, so slot 0's
    class is DOMINATED and dropped: 2 candidate classes -> 1 layer. (The old cap
    heuristic built both.) The low slot is not lost — it is handed out at assignment
    time by first_fit_slot; see test_assignment_uses_a_dominated_low_slot."""
    n = _two_parallel_oms_model()
    n.add_lightpath(_L("lp0", ("oms-AB-1",), "100G", 191.4e12))     # slot 0
    n.set_qot_state("lp0", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    g = build_layered_graph(n)
    assert _wle_classes(g) == [0]
    assert _class_sigs(g) == {frozenset({"oms-AB-1", "oms-AB-2"})}
    assert wle_count_on_layer(g, "oms-AB-1", 0) == 1
    assert wle_count_on_layer(g, "oms-AB-2", 0) == 1


def test_incomparable_signatures_are_all_kept():
    """The rule refuses to over-merge. Three OMS, no universally-free slot:
    signatures {BC,CD}, {AB,BC}, {AB,CD} are pairwise incomparable, so all three
    layers survive — A->C exists only on the second, B->D only on the first."""
    n = _line4_model()
    # slot 0 lit on AB, slot 1 lit on CD, slot 2 lit on BC; slots >=3 lit everywhere
    n.add_lightpath(_L("l0", ("oms_A_B",), "100G", _f(0)))
    n.add_lightpath(_L("l1", ("oms_C_D",), "100G", _f(1)))
    n.add_lightpath(_L("l2", ("oms_B_C",), "100G", _f(2)))
    for lp in ("l0", "l1", "l2"):
        n.set_qot_state(lp, QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    for s in range(3, SpectrumGrid.default().num_slots):
        n.add_lightpath(_L(f"lx{s}", ("oms_A_B", "oms_B_C", "oms_C_D"), "100G", _f(s)))
        n.set_qot_state(f"lx{s}", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    g = build_layered_graph(n)
    assert _class_sigs(g) == {
        frozenset({"oms_B_A", "oms_B_C", "oms_C_B", "oms_C_D", "oms_D_C"}),
        frozenset({"oms_A_B", "oms_B_A", "oms_C_B", "oms_C_D", "oms_D_C"}),
        frozenset({"oms_A_B", "oms_B_A", "oms_B_C", "oms_C_B", "oms_D_C"}),
    }


def test_forbidding_an_oms_changes_which_signatures_are_maximal():
    """Signatures are over NON-FORBIDDEN OMS. Forbid oms-AB-1 and slot 0's signature
    becomes {oms-AB-2} — now equal to every other slot's, hence maximal again and
    merged into one class rather than dropped."""
    n = _two_parallel_oms_model()
    n.add_lightpath(_L("lp0", ("oms-AB-1",), "100G", 191.4e12))     # slot 0
    n.set_qot_state("lp0", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    g = build_layered_graph(n, forbidden_assets=frozenset({"oms-AB-1"}))
    assert _class_sigs(g) == {frozenset({"oms-AB-2"})}
    (cls,) = g.graph["slot_classes"]
    assert 0 in cls.slots            # the low slot is inside the surviving class


def test_fully_lit_oms_contributes_no_layer():
    """A slot free on NO non-forbidden OMS has an empty signature and is discarded
    before maximality is tested — an empty class would add vertices and no routes."""
    n = _one_lightpath_model()      # single oms-AB, slot 20 lit
    g = build_layered_graph(n)
    assert _class_sigs(g) == {frozenset({"oms-AB"})}
    assert all(20 not in c.slots for c in g.graph["slot_classes"])


def test_pass_through_node_has_express_edge():
    """Node-split: a node that both receives and forwards on a layer (a pass-through
    like C on the A->C->B route) gets an EXPRESS edge (WLin,C,0)->(WLout,C,0). Pure
    endpoints (A source-only, B sink-only) get none."""
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=80)
    n = _cheap_route_plus_distinct_route()
    g = build_layered_graph(n, grid=grid)
    assert g.has_edge((WLIN, "C", 0), (WLOUT, "C", 0))     # pass-through C
    assert not g.has_edge((WLIN, "A", 0), (WLOUT, "A", 0))  # source-only
    assert not g.has_edge((WLIN, "B", 0), (WLOUT, "B", 0))  # sink-only


def test_through_lightpath_is_single_run_across_two_oms():
    """Continuity via EXPRESS: an A->B demand can be one through-lightpath spanning
    (oms-AC, oms-CB) on a single wavelength — a single run, C optically bypassed."""
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=80)
    n = _cheap_route_plus_distinct_route()
    g = build_layered_graph(n, grid=grid)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only", grid=grid)
    single = {p.new_lightpaths[0].oms_sequence
              for p in res if len(p.new_lightpaths) == 1}
    assert ("oms-AC", "oms-CB") in single


def test_segmented_two_run_placement_shares_one_wavelength():
    """The node-split's payoff: a demand served by TWO separate lightpaths meeting at
    a regen node (A->C then C->B) is enumerable on a SINGLE wavelength. The two runs
    are OMS-disjoint, so accept-time assignment gives them the SAME slot 0 (they do
    not contend); only runs sharing an OMS are forced apart."""
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=80)
    n = _cheap_route_plus_distinct_route()
    g = build_layered_graph(n, grid=grid)
    assert _wle_classes(g) == [0]                       # one layer on empty
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only", grid=grid)
    two_run = [p for p in res if len(p.new_lightpaths) == 2]
    assert two_run, "expected a segmented A->C + C->B placement on one wavelength"
    assert {r.lam for r in two_run[0].new_lightpaths} == {0}


def test_demand_still_placeable_after_merging():
    """Completeness: with slot 0 lit on one A->B fiber, an A->B demand still places on
    BOTH fibers even though slot 0's layer was dropped as dominated."""
    n = _two_parallel_oms_model()
    n.add_lightpath(_L("lp0", ("oms-AB-1",), "100G", 191.4e12))     # slot 0
    n.set_qot_state("lp0", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only")
    routes = {p.new_lightpaths[0].oms_sequence for p in res if p.new_lightpaths}
    assert ("oms-AB-1",) in routes and ("oms-AB-2",) in routes, routes


def test_assignment_uses_a_dominated_low_slot():
    """Slot 0's layer is dropped, but a route over oms-AB-2 (where slot 0 IS free)
    must still be ASSIGNED slot 0 — dominance removes enumeration duplicates, it must
    not cost spectrum."""
    n = _two_parallel_oms_model()
    n.add_lightpath(_L("lp0", ("oms-AB-1",), "100G", 191.4e12))     # slot 0
    n.set_qot_state("lp0", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="new_only")
    by_route = {p.new_lightpaths[0].oms_sequence: p.new_lightpaths[0].lam
                for p in res if p.new_lightpaths}
    assert by_route[("oms-AB-2",)] == 0     # lowest free slot on that fiber
    assert by_route[("oms-AB-1",)] == 1     # slot 0 is lit there


# ---------------------------------------------------------------------------
# O2: hoist the offered-load map + min-over-bound-links residual
# ---------------------------------------------------------------------------

def _multi_link_lightpath_model() -> NetworkModel:
    """One lightpath (lp-AB) bound to TWO IP links (ip-AB loaded 60G, ip-AB2
    loaded 20G), both reading the 100G mode. residual_gbps per link:
    ip-AB -> 40, ip-AB2 -> 80. `max` reports 80 (the healthy link); `min`
    reports 40 (the bottleneck link a groom is actually limited by)."""
    n = _one_lightpath_model()               # lp-AB, ip-AB (R1->R2)
    n.add_router(_R("R3", "A"))
    n.add_router(_R("R4", "B"))
    n.add_ip_link(_I("ip-AB2", "R3", "R4", "lp-AB"))  # 2nd link on same lightpath
    n.add_service(Service("s1", "R1", "R2", 60.0, working_path=("ip-AB",)))
    n.add_service(Service("s2", "R3", "R4", 20.0, working_path=("ip-AB2",)))
    return n


def test_residual_is_min_over_bound_ip_links_not_max():
    """A lightpath serving two IP links, one more loaded than the other, reports
    the BOTTLENECK residual (min), not the healthiest link's headroom (max) —
    max would overstate capacity the groom can't actually use."""
    n = _multi_link_lightpath_model()
    g = build_layered_graph(n)
    (u, v, data), = lpe_edges(g)
    assert data["residual_gbps"] == 40.0     # min(40, 80), not max


def _two_ip_bound_lightpaths_model() -> NetworkModel:
    """Two lightpaths, each bound to its own IP link, so both exercise the
    load-map branch of _residual_gbps (the no-IP-link branch skips the map)."""
    n = _one_lightpath_model()               # oms-AB, lp-AB, ip-AB
    for a in ("c1", "c2"):
        n.add_amplifier(_A(a, "advanced_toy", 20.0, 5.5))
    n.add_fiber(_F("fCD", "c1", "c2", 80.0, "SSMF"))
    n.add_oms(_O("oms-CD", "C", "D", ("c1", "fCD", "c2")))
    n.add_lightpath(_L("lp-CD", ("oms-CD",), "100G", 193.4e12))
    n.set_qot_state("lp-CD", QoTState(gsnr_db=15.0, osnr_db=30.0, margin_db=3.0))
    n.add_router(_R("R3", "C"))
    n.add_router(_R("R4", "D"))
    n.add_ip_link(_I("ip-CD", "R3", "R4", "lp-CD"))
    return n


def test_offered_load_map_built_once_per_graph_build(monkeypatch):
    """S5-8/S7-8: the offered-load map is built ONCE per build_layered_graph,
    not rebuilt inside the per-lightpath loop (O(L·S) -> O(L+S))."""
    from multilayer_optical_network.model import ip_routing
    n = _two_ip_bound_lightpaths_model()
    calls = {"n": 0}
    real = ip_routing.offered_load_per_link

    def counting(model):
        calls["n"] += 1
        return real(model)

    monkeypatch.setattr(ip_routing, "offered_load_per_link", counting)
    build_layered_graph(n)
    assert calls["n"] == 1                   # once, despite two lightpaths


def test_groom_or_new_finds_hybrid_groom_plus_new():
    n = _groom_plus_gap_model()
    g = build_layered_graph(n)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="B",
                        demand_gbps=100.0, policy="groom_or_new")
    hybrids = [p for p in res if p.reused_lightpaths and p.new_lightpaths]
    assert hybrids, "expected a hybrid (groom A->M + new M->B)"
    h = hybrids[0]
    assert h.reused_lightpaths == ("lp-AM",)
    assert h.new_lightpaths[0].oms_sequence == ("oms-MB",)
    assert h.restored_gbps == 100.0


def test_residual_gbps_treats_unseeded_qot_as_zero_not_a_crash():
    """Regression for the audit's Critical finding: a lightpath with a bound
    IP link but NO recorded QoT state (the state left by provision_lightpath,
    a live single-op tool that never seeds or recomputes QoT) must read as
    zero residual capacity, not raise LookupError -- consistent with the
    no-IP-link branch three lines above in the same function."""
    n = NetworkModel(modes=ModeRegistry([
        TransceiverMode(id="100G", bitrate_gbps=100.0, required_gsnr_db=12.0,
                        symbol_rate_baud=32e9, channel_spacing_hz=100e9)]))
    n.register_fiber_type(FiberType("SSMF", 0.2))
    for a in ("a1", "a2"):
        n.add_amplifier(Amplifier(id=a, type_variety="advanced_toy", gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber("fAB", "a1", "a2", 80.0, "SSMF"))
    for node in ("A", "B"):
        n.add_roadm(ROADM(id=f"roadm_{node}"))
    n.add_oms(OMS("oms-AB", "A", "B", ("roadm_A", "a1", "fAB", "a2")))
    n.add_lightpath(Lightpath("lp-AB", ("oms-AB",), "100G", 193.4e12))
    # Deliberately NO n.set_qot_state(...) call -- the unseeded state.
    n.add_router(Router("R1", "A"))
    n.add_router(Router("R2", "B"))
    n.add_ip_link(IPLink("ip-AB", "R1", "R2", "lp-AB"))

    g = build_layered_graph(n)   # must not raise LookupError
    assert lpe_edges(g) == []    # zero residual -> no LPE edge, same as margin<0


# ---------------------------------------------------------------------------
# min_residual_gbps tests (Task 4: capacity-filtered LPE edges)
# ---------------------------------------------------------------------------


def test_min_residual_prunes_a_lightpath_that_cannot_carry_the_demand():
    """_W_LPE is the cheapest weight in the graph, so Yen's returns a groom
    first whatever its residual. Capacity must be checked DURING routing, not
    clamped after it."""
    n = _one_lightpath_model()
    n.add_service(Service("s-load", "R1", "R2", 70.0, working_path=("ip-AB",)))
    g = build_layered_graph(n, min_residual_gbps=40.0)   # 30G residual < 40G
    assert lpe_edges(g) == []


def test_min_residual_keeps_a_lightpath_that_can_carry_the_demand():
    n = _one_lightpath_model()
    n.add_service(Service("s-load", "R1", "R2", 70.0, working_path=("ip-AB",)))
    g = build_layered_graph(n, min_residual_gbps=30.0)   # exactly enough
    assert [d["lightpath_id"] for _, _, d in lpe_edges(g)] == ["lp-AB"]


def test_min_residual_defaults_to_todays_behaviour():
    """route_service and restoration want degraded options; the default must
    not take them away."""
    n = _one_lightpath_model()
    n.add_service(Service("s-load", "R1", "R2", 70.0, working_path=("ip-AB",)))
    g = build_layered_graph(n)
    assert [d["lightpath_id"] for _, _, d in lpe_edges(g)] == ["lp-AB"]


# ---------------------------------------------------------------------------
# B2: accept-time slot assignment
# ---------------------------------------------------------------------------

def _shared_oms_two_run_model() -> NetworkModel:
    """A-B-C-D line where a two-run placement (A->C then C->D) and a single-run
    placement (A->D) both exist, and the two runs of a segmented A->B / B->D placement
    share oms-BC. Built through the importer so both directed OMS exist per hop."""
    graph = {
        "nodes": [{"id": n} for n in ("A", "B", "C", "D")],
        "edges": [
            {"src": "A", "dst": "B", "length_km": 80.0},
            {"src": "B", "dst": "C", "length_km": 80.0},
            {"src": "C", "dst": "D", "length_km": 80.0},
        ],
    }
    return model_from_abstract_graph(graph, modes=ModeRegistry([
        TransceiverMode(id="100G", bitrate_gbps=100.0, required_gsnr_db=12.0,
                        symbol_rate_baud=32e9, channel_spacing_hz=100e9)]))


def test_sibling_runs_sharing_an_oms_get_distinct_slots():
    """Two runs in ONE placement can legitimately share a physical OMS (the
    WLin/WLout split lets a later run re-enter a span an earlier one used). They are
    different lightpaths, so they must NOT be assigned the same slot on that span —
    accept-time assignment threads a placement-local extra_state to force them apart."""
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=80)
    n = _shared_oms_two_run_model()
    g = build_layered_graph(n, grid=grid)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="D",
                        demand_gbps=100.0, policy="new_only", grid=grid)
    two_run = [p for p in res if len(p.new_lightpaths) == 2]
    assert two_run, "expected a two-run placement sharing an OMS"
    for p in two_run:
        a, b = p.new_lightpaths
        if set(a.oms_sequence) & set(b.oms_sequence):
            assert a.lam != b.lam, (a.oms_sequence, b.oms_sequence, a.lam)


def test_no_returned_placement_double_books_a_slot_on_one_oms():
    """Whatever comes back must be internally conflict-free: no two runs of one
    placement may hold the same slot on a shared OMS."""
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=4)
    n = _shared_oms_two_run_model()
    g = build_layered_graph(n, grid=grid)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="D",
                        demand_gbps=100.0, policy="new_only", grid=grid)
    assert res
    for p in res:
        booked = set()
        for r in p.new_lightpaths:
            for oms_id in r.oms_sequence:
                assert (oms_id, r.lam) not in booked, (p, oms_id, r.lam)
                booked.add((oms_id, r.lam))


def test_placement_rejected_for_want_of_a_slot_does_not_consume_budget(monkeypatch):
    """A route with no assignable slot is infeasible, not 'examined'. If it burned a
    _PATH_BUDGET slot, a saturated corridor could starve the frontier of the
    structurally distinct routes that ARE placeable.

    Forced by shrinking the grid to ONE slot: any two-run placement sharing an OMS then
    has no second slot and must be rejected -- while the single-run route still comes
    back, proving the frontier was not starved. `_PATH_BUDGET` is patched down to 2 so
    the starvation would be visible if rejections consumed it."""
    monkeypatch.setattr(mg, "_PATH_BUDGET", 2)
    grid = SpectrumGrid(anchor_hz=191.4e12, spacing_hz=100e9, num_slots=1)
    n = _shared_oms_two_run_model()
    g = build_layered_graph(n, grid=grid)
    res = place_demands(n, g, FakeQot(15.0), src="A", dst="D",
                        demand_gbps=100.0, policy="new_only", grid=grid)
    assert any(len(p.new_lightpaths) == 1 for p in res), res
    for p in res:
        booked = set()
        for r in p.new_lightpaths:
            for oms_id in r.oms_sequence:
                assert (oms_id, r.lam) not in booked
                booked.add((oms_id, r.lam))
