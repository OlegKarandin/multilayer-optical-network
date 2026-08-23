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
