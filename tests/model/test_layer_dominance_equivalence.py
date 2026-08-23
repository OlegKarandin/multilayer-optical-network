"""The central dominance property: merging layers must not lose a route.

`maximal_slot_classes` keeps only ⊆-maximal free-slot signatures. The claim is that a
dominated layer contributes no route a maximal layer doesn't — so the STRUCTURAL route
set (reused lightpath ids + new OMS sequences, wavelength-blind) enumerated over the
merged graph must equal the one enumerated over a graph with one layer per free slot.
The unmerged reference is obtained by monkeypatching the selector, so both sides run
the identical enumeration code and only the layer construction differs."""
import json

import pytest

from multilayer_optical_network.data import reference_topology
from multilayer_optical_network.model import multilayer_graph as mg
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.multilayer_graph import (
    SlotClass, build_layered_graph, place_demands,
)
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.topology_import import model_from_abstract_graph


class _FlatQot:
    """Route-agnostic GSNR: this test is about the ROUTE SET, not physics."""
    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        return QoTState(gsnr_db=30.0, osnr_db=40.0, margin_db=10.0)


def _all_slot_classes(non_forbidden, spectrum, grid):
    """Reference layer selection: one class per slot with a non-empty signature, no
    maximality filter. This is the pre-dominance behaviour, minus the cap."""
    out = []
    for lam in range(grid.num_slots):
        sig = frozenset(o.id for o in non_forbidden
                        if not ((spectrum.get(o.id, 0) >> lam) & 1))
        if sig:
            out.append(SlotClass(class_id=len(out), slots=(lam,), oms_ids=sig))
    return tuple(out)


def _german17():
    graph = json.loads(
        reference_topology("german_17").read_text(encoding="utf-8"))["graph"]
    return model_from_abstract_graph(graph, modes=default_modes())


def _route_set(model, *, src, dst, policy, forbidden, grid):
    g = build_layered_graph(model, forbidden_assets=forbidden, grid=grid)
    res = place_demands(model, g, _FlatQot(), src=src, dst=dst,
                        demand_gbps=100.0, policy=policy, k=20, grid=grid)
    return {(p.reused_lightpaths,
             tuple(r.oms_sequence for r in p.new_lightpaths)) for p in res}


PAIRS = [("0", "5"), ("1", "9"), ("3", "12"), ("7", "16"), ("2", "11")]


@pytest.mark.parametrize("src,dst", PAIRS)
@pytest.mark.parametrize("policy", ["groom_or_new", "new_only"])
@pytest.mark.parametrize("forbidden", [frozenset(), frozenset({"oms_0_3"})])
def test_merged_route_set_equals_unmerged(monkeypatch, src, dst, policy, forbidden):
    model = _german17()
    grid = SpectrumGrid.default()
    merged = _route_set(model, src=src, dst=dst, policy=policy,
                        forbidden=forbidden, grid=grid)
    monkeypatch.setattr(mg, "maximal_slot_classes", _all_slot_classes)
    unmerged = _route_set(model, src=src, dst=dst, policy=policy,
                          forbidden=forbidden, grid=grid)
    assert merged == unmerged, sorted(unmerged - merged)


def test_merged_graph_is_much_smaller():
    """The measured payoff on the packaged topology: one maximal layer instead of the
    cap heuristic's several, and a graph that stays flat as the network fills."""
    model = _german17()
    g = build_layered_graph(model)
    assert len(g.graph["slot_classes"]) == 1
    assert g.number_of_nodes() < 60      # was 175 under the cap heuristic


def _fill_first_fit(model, grid, *, fraction: float) -> None:
    """Light lightpaths on first-fit slots along random-ish 2-hop routes until
    `fraction` of (OMS, slot) pairs are occupied. Deterministic: OMS are walked in
    model order, no RNG.

    Genuinely 2-hop, chaining each OMS `a` with the first continuator OMS `b`
    sharing its dst node (`by_src[a.dst_node_id]`), because a 1-hop version (one
    lightpath per single OMS, round-robin) provably CANNOT ever produce more than
    one maximal signature here: first-fit on an independent single-OMS state always
    fills each OMS's slots contiguously from 0, so every OMS's free set is a simple
    suffix {k, k+1, ..., num_slots-1} for its own fill count k. Free-sets that are
    all suffixes of the same axis are totally ordered by inclusion (a chain), and a
    chain's dominance-maximal filter always collapses to exactly one top element —
    confirmed empirically on this topology up to fraction=0.995, still 1 class, see
    task-B4-report.md. Sharing OMS across 2-hop routes breaks that independence
    (a slot free on `a` alone can still be blocked by `b`'s occupancy), which is
    what makes several incomparable signatures possible, matching this function's
    own original docstring text (2-hop) that the literal 1-hop code contradicted."""
    from multilayer_optical_network.model.assets import Lightpath
    from multilayer_optical_network.model.spectrum import (
        build_spectrum_state, first_fit_slot,
    )
    oms_list = model.list_oms()
    oms_ids = [o.id for o in oms_list]
    by_src: dict = {}
    for o in oms_list:
        by_src.setdefault(o.src_node_id, []).append(o)
    target = int(len(oms_ids) * grid.num_slots * fraction)
    placed = 0
    i = 0
    mode = model.modes.list()[0].id
    while placed < target and i < len(oms_ids) * grid.num_slots:
        a = oms_list[i % len(oms_ids)]
        continuators = by_src.get(a.dst_node_id, [])
        b = next((o for o in continuators if o.id != a.id), None)
        seq = (a.id, b.id) if b is not None else (a.id,)
        slot = first_fit_slot(build_spectrum_state(model, grid), seq, grid)
        i += 1
        if slot is None:
            continue
        lp_id = f"fill{i}"
        model.add_lightpath(Lightpath(lp_id, seq, mode, grid.freq(slot)))
        model.set_qot_state(lp_id, QoTState(gsnr_db=30.0, osnr_db=40.0, margin_db=10.0))
        placed += 1


def test_saturated_state_keeps_several_maximal_layers_and_loses_no_route(monkeypatch):
    """With no universally-free slot the lattice has no top element and several
    signatures are maximal. Assert more than one layer is built AND the route set is
    still complete — the merge must degrade to keeping layers, never to dropping
    routes. This is the regime where _RAW_PATH_CAP and the λ-blind dedup key (both
    retained, both idle in the common case) become live again. fraction=0.75 (the
    plan's original value) is enough once `_fill_first_fit` is genuinely 2-hop (46
    maximal layers observed on this topology) — see that helper's docstring for why
    the literal 1-hop code the task brief gave cannot ever satisfy this assertion,
    at any fraction."""
    model = _german17()
    grid = SpectrumGrid.default()
    _fill_first_fit(model, grid, fraction=0.75)
    g = build_layered_graph(model, grid=grid)
    assert len(g.graph["slot_classes"]) > 1
    merged = _route_set(model, src=PAIRS[0][0], dst=PAIRS[0][1],
                        policy="new_only", forbidden=frozenset(), grid=grid)
    monkeypatch.setattr(mg, "maximal_slot_classes", _all_slot_classes)
    unmerged = _route_set(model, src=PAIRS[0][0], dst=PAIRS[0][1],
                          policy="new_only", forbidden=frozenset(), grid=grid)
    assert merged == unmerged, sorted(unmerged - merged)
