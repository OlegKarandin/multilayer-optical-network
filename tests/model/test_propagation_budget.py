"""Deterministic propagation budget for the german_17 packer.

The packer's wall-clock is dominated by GNPy propagation (88-92 % of `_pack`,
docs/2026-08-20-allocation-qot-performance-findings.md). Wall-clock is not
assertable in CI, but the NUMBER OF PROPAGATIONS is: it is a pure function of
the model, the frozen demand set and the cache keying, none of which involve
timing or floating-point luck.

So this test stubs the adapter's two propagation entry points with counting
fakes -- no GNPy, milliseconds to run -- and pins the count. Every caching or
frontier change shows up here as a hard before/after number.

Both stubs return a constant, comfortably-feasible GSNR so every candidate
route survives the physics filter. That maximizes candidate exploration, which
makes the count a stable upper bound and keeps it independent of the physics
numbers themselves (which the pinned gnpy version, not this test, owns).
"""
import json

from multilayer_optical_network.data import reference_topology
from multilayer_optical_network.model import allocation
from multilayer_optical_network.model.allocation import (
    make_adapter_evaluator, solve_allocation_model,
)
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.qot_results import (
    HarvestCache, QoTCache, QoTResultStore,
)
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.topology_import import model_from_abstract_graph
from tests.conftest import FIXTURES_DIR

# Propagations `solve_allocation_model` performs on german_17 with the frozen
# 22-demand set. UPDATE DELIBERATELY, in the commit that changes it, and say
# why in the commit message. A surprise change here is the point of the test.
#
# 352 -> 179 (Task 3, perf(qot): physics-only harvest key): harvest_cache_key
# dropped oms_sequence/direction, so a symmetric span's forward and backward
# requests now alias to one harvest. Not an exact halving: solving 2X + Y =
# 352 (old) and X + Y = 179 (new) gives X = 173 path/mode combos that were
# queried in both directions and now alias 2:1, and Y = 6 combos genuinely
# queried in only one direction during this solve (no counterpart request
# exists to alias against) -- verified by instrumenting harvest_cache_key
# call sites during a real solve_allocation_model run on this fixture.
PROPAGATION_BUDGET = 179

_STUB_GSNR_DB = 30.0


def _stub_state(model, mode_id):
    m = model.modes.get(mode_id)
    return QoTState(gsnr_db=_STUB_GSNR_DB, osnr_db=_STUB_GSNR_DB,
                    margin_db=_STUB_GSNR_DB - m.required_gsnr_db,
                    limiting_element_id=None)


def _german_17():
    graph = json.loads(
        reference_topology("german_17").read_text(encoding="utf-8"))["graph"]
    return model_from_abstract_graph(graph, modes=default_modes())


def _frozen_demands():
    path = FIXTURES_DIR / "german_17_demands_seed0.json"
    return json.loads(path.read_text(encoding="utf-8"))["demands"]


def test_german_17_pack_propagation_budget(monkeypatch):
    """Pin the propagation count of a full 22-demand pack."""
    calls = {"harvest": 0, "compute": 0}
    grid = SpectrumGrid.default()

    def _harvest(model, oms_sequence, direction, mode_id, full_comb):
        calls["harvest"] += 1
        state = _stub_state(model, mode_id)
        return {s: state for s in range(grid.num_slots)}

    def _compute(**kw):
        calls["compute"] += 1
        return _stub_state(kw["model"], kw["mode_id"]), ""

    # allocation.py binds both names at import time (allocation.py:41), so the
    # patch must land on the allocation module, not on gnpy_adapter.adapter.
    monkeypatch.setattr(allocation, "harvest_qot", _harvest)
    monkeypatch.setattr(allocation, "compute_qot", _compute)

    model = _german_17()
    demands = _frozen_demands()
    inventory = {r.site: 10 ** 6 for r in model.list_routers()}
    qot = make_adapter_evaluator(model, QoTResultStore(), cache=QoTCache(),
                                 harvest_cache=HarvestCache())

    result, _work = solve_allocation_model(model, qot, demands, inventory)

    assert result.placements, "the harness must actually place demands"
    total = calls["harvest"] + calls["compute"]
    assert total == PROPAGATION_BUDGET, (
        f"propagation count changed: {total} "
        f"(harvest={calls['harvest']}, compute={calls['compute']}); "
        f"budget={PROPAGATION_BUDGET}. If this change is intended, update "
        f"PROPAGATION_BUDGET in this file and say why in the commit message.")
