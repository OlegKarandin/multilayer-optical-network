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
    HarvestCache, IncrementCache, QoTCache, QoTResultStore,
)
from multilayer_optical_network.model.spectrum import SpectrumGrid
from multilayer_optical_network.model.topology_import import model_from_abstract_graph
from tests.conftest import FIXTURES_DIR

# Propagations `solve_allocation_model` performs on german_17 with the frozen
# 22-demand set. UPDATE DELIBERATELY, in the commit that changes it, and say
# why in the commit message. A surprise change here is the point of the test.
#
# The full trail across the six perf tasks is 352 -> 352 -> 179 -> 179 -> 69 ->
# 19: only Task 3, Task 5 and Task A5 (below) move the number. Task 2 (wire a
# shared HarvestCache into the CLI evaluator) and Task 4 (gate grooming on
# whether the lightpath can carry the demand) deliberately leave it UNCHANGED
# -- Task 2 moves the cache into the production wiring this test already
# instrumented by hand, and Task 4's capacity filter narrows the graph without
# removing any path/mode combo german_17's demand set actually probes. Both
# are no-ops here by design, not missed wins; a reader diffing the commit
# range should expect two flat steps.
#
# 352 -> 179 (Task 3, perf(qot): physics-only harvest key): harvest_cache_key
# dropped oms_sequence/direction, so a symmetric span's forward and backward
# requests now alias to one harvest. Not an exact halving: solving 2X + Y =
# 352 (old) and X + Y = 179 (new) gives X = 173 path/mode combos that were
# queried in both directions and now alias 2:1, and Y = 6 combos genuinely
# queried in only one direction during this solve (no counterpart request
# exists to alias against) -- verified by instrumenting harvest_cache_key
# call sites during a real solve_allocation_model run on this fixture.
#
# 179 -> 69 (Task 5, perf(alloc): stop the frontier at the first full-rate
# candidate): _pack's unprotected branch (~70% of the frozen 22-demand set)
# now passes stop_when=lambda p: p.shortfall_gbps <= 0.0 into the harvest, so
# enumeration stops at the first candidate that carries the demand in full
# instead of exhausting the whole k-best frontier across both groom_or_new and
# new_only. Protected demands are unaffected (stop_when=None there --
# disjoint_pairs needs the full frontier).
#
# 69 -> 19 (Task A5, perf(qot): compose path GSNR from per-OMS increments under
# FillPolicy.FULL): `_best_feasible_mode` now tries composition (summing
# cached per-OMS 1/gsnr_lin increments -- gnpy_adapter/composition.py) before
# falling through to a fresh propagation. Wiring `increment_cache` into THIS
# test's evaluator (mirroring `cache`/`harvest_cache` -- see
# `make_adapter_evaluator`) lets a K-hop harvest calibrate K per-OMS entries in
# one propagation, and any LATER candidate route that reuses any of those OMS
# -- even a totally different path, a different demand, a different direction
# -- composes its GSNR from the cached table instead of propagating again.
# ~72.5% fewer propagations (69 -> 19, a ~3.6x reduction) on this frozen
# 22-demand german_17 mesh, where many k-best candidate routes and disjoint
# working/protection pairs share OMS segments. At the time this was measured
# it was a STANDALONE proof that the mechanism works, not yet production
# wiring: `build_cli.py` deliberately did not pass `increment_cache`, because
# composition is only SAFE once `design_margin_db > composition.
# COMPOSITION_ERROR_BOUND_DB` (see `tests/model/test_composition_gate.py`'s
# `test_composed_selection_is_genuinely_feasible`). Task A7 flipped the
# model's default margin to 0.5 (satisfying that condition), and Task A8 (see
# `tests/test_build_cli.py`'s `test_cli_wires_an_increment_cache_into_the_
# evaluator`) wired `build_cli.py` to pass `increment_cache` in production.
# This test's physics are entirely stubbed (`_STUB_GSNR_DB`), so that safety
# gate is moot here -- it exercises the counting mechanism only.
#
# 19 -> 28 (Task A6, feat(qot): verify the committed placement exactly and
# watchdog the composition bound): allocation._pack now calls
# objective.verify_and_reseed once per demand, right after its winning
# placement is accepted, which re-propagates EXACTLY (both directions) every
# new lightpath run whose NewLightpathRun.gsnr_estimated is True (i.e. every
# run this frozen solve actually composed rather than propagated). Of this
# solve's 23 total new lightpath runs (working + protection legs across all
# 22 demands), 18 are composed and therefore verified -- 36 additional exact
# calls in the naive worst case, but the verify calls route through the SAME
# shared `harvest_cache` the rest of the solve already populated, and most of
# them land on a fingerprint some earlier candidate-scoring or calibration
# harvest already cached (a symmetric span's forward/backward alias to one
# entry -- `harvest_cache_key`'s own docstring). Only 9 of the 36 are genuine
# cache misses, so the net is +9 (19 -> 28), not +36. This test's physics stub
# makes composed-vs-exact disagree on almost every verified run (the stub's
# exact path always returns a flat `_STUB_GSNR_DB`, while the composed path
# runs REAL endpoint-noise physics over near-zero stubbed increments -- two
# unrelated numbers by construction here), so `AllocationResult.violations`
# is non-empty on this fixture; harmless for THIS test (it only pins the
# propagation count), but not representative of a real composed/exact gap.
PROPAGATION_BUDGET = 28

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

    def _harvest(model, oms_sequence, direction, mode_id, full_comb, *,
                 capture_increments=False):
        calls["harvest"] += 1
        state = _stub_state(model, mode_id)
        vec = {s: state for s in range(grid.num_slots)}
        if capture_increments:
            # Near-zero-noise per-OMS increments (comfortably feasible, same
            # spirit as _STUB_GSNR_DB -- see the module docstring): the point
            # of this stub is to prove composition SKIPS a propagation when
            # it fires, not to reproduce _STUB_GSNR_DB's exact number.
            increments = {oms_id: {s: 0.0 for s in range(grid.num_slots)}
                         for oms_id in oms_sequence}
            return vec, increments
        return vec

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
                                 harvest_cache=HarvestCache(),
                                 increment_cache=IncrementCache())

    result, _work = solve_allocation_model(model, qot, demands, inventory)

    assert result.placements, "the harness must actually place demands"
    total = calls["harvest"] + calls["compute"]
    assert total == PROPAGATION_BUDGET, (
        f"propagation count changed: {total} "
        f"(harvest={calls['harvest']}, compute={calls['compute']}); "
        f"budget={PROPAGATION_BUDGET}. If this change is intended, update "
        f"PROPAGATION_BUDGET in this file and say why in the commit message.")
