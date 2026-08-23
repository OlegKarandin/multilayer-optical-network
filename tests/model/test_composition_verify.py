"""Task A6: the exact verify pass + composition-error watchdog that runs once
per accepted placement in allocation._pack, right after a NewLightpathRun with
gsnr_estimated=True (a COMPOSED GSNR -- see gnpy_adapter/composition.py and
Task A5's allocation._best_feasible_mode) is provisioned+seeded on the winning
candidate. objective.verify_and_reseed re-propagates such a run exactly, seeds
`work`'s stored QoTState with the exact (never composed) value, and raises a
typed ViolationType.COMPOSITION_ERROR if the measured error ever exceeds
COMPOSITION_ERROR_BOUND_DB.

No real GNPy propagation is exercised here (that is test_composition_gate.py's
job, against a real synthesizable topology). This file uses a fake QotEvaluator
whose __call__ (exact) and compose_gsnr (composed) return deliberately
different, easily distinguished GSNR values -- mirroring test_composition_gate
.py's _FakeComposeQot -- so every assertion can tell which path a number came
from just by comparing it against the two fixed constants.
"""
import pytest

from multilayer_optical_network.gnpy_adapter.composition import COMPOSITION_ERROR_BOUND_DB
from multilayer_optical_network.model import objective as objective_mod
from multilayer_optical_network.model import views as views_mod
from multilayer_optical_network.model.allocation import solve_allocation, solve_allocation_model
from multilayer_optical_network.model.assets import (
    Amplifier, Fiber, FiberType, OMS, ROADM, TransceiverMode,
)
from multilayer_optical_network.model.ip_assets import Router
from multilayer_optical_network.model.modes import ModeRegistry
from multilayer_optical_network.model.network import NetworkModel
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.solvers import SolverStatus
from multilayer_optical_network.model.validate import ViolationType
from multilayer_optical_network.model.violations import CompositionErrorViolation

REQUIRED_GSNR_DB = 5.0
BITRATE_GBPS = 100.0


def _modes() -> ModeRegistry:
    return ModeRegistry([
        TransceiverMode(id="100G", bitrate_gbps=BITRATE_GBPS,
                        required_gsnr_db=REQUIRED_GSNR_DB,
                        symbol_rate_baud=87.5e9, channel_spacing_hz=100e9),
    ])


def _one_route_model() -> NetworkModel:
    """A single A->Z OMS route -- one candidate, one accepted new lightpath
    run per demand. Physical assets are real (spectrum/layered-graph
    bookkeeping touches them); GSNR itself never comes from GNPy in this file
    -- the fake QotEvaluator answers every QoT question."""
    n = NetworkModel(modes=_modes())
    n.register_fiber_type(FiberType("SSMF", 0.2))
    n.design_margin_db = 1.0  # > COMPOSITION_ERROR_BOUND_DB, the safe convention
    for a in ("a1", "a2"):
        n.add_amplifier(Amplifier(id=a, type_variety="advanced_toy", gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber("f1", "a1", "a2", 80.0, "SSMF"))
    for node in ("A", "Z"):
        n.add_roadm(ROADM(id=f"roadm_{node}"))
    n.add_oms(OMS("oms1", "A", "Z", ("roadm_A", "a1", "f1", "a2")))
    n.add_router(Router(id="r_A", site="A"))
    n.add_router(Router(id="r_Z", site="Z"))
    return n


def _two_route_model() -> NetworkModel:
    """Two parallel A->Z OMS routes, so place_demands' k-best frontier has more
    than one candidate per demand -- the fixture test_verify_pass_costs_two_
    propagations_per_new_run_not_per_candidate needs to prove the exact-verify
    hook does NOT scale with candidates examined."""
    n = NetworkModel(modes=_modes())
    n.register_fiber_type(FiberType("SSMF", 0.2))
    n.design_margin_db = 1.0
    for a in ("aN1", "aN2", "aS1", "aS2"):
        n.add_amplifier(Amplifier(id=a, type_variety="advanced_toy", gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber("fN", "aN1", "aN2", 80.0, "SSMF"))
    n.add_fiber(Fiber("fS", "aS1", "aS2", 120.0, "SSMF"))
    for node in ("A", "Z"):
        n.add_roadm(ROADM(id=f"roadm_{node}"))
    n.add_oms(OMS("oms-north", "A", "Z", ("roadm_A", "aN1", "fN", "aN2")))
    n.add_oms(OMS("oms-south", "A", "Z", ("roadm_A", "aS1", "fS", "aS2")))
    n.add_router(Router(id="r_A", site="A"))
    n.add_router(Router(id="r_Z", site="Z"))
    return n


class _CountingComposeQot:
    """A QotEvaluator whose exact (`__call__`) and composed (`compose_gsnr`)
    answers are deliberately different constants, with both call counts
    tracked -- lets a test assert exactly how many of each kind of question
    was asked, and which value ended up where."""

    def __init__(self, exact_gsnr: float, composed_gsnr: float):
        self.exact_gsnr = exact_gsnr
        self.composed_gsnr = composed_gsnr
        self.call_count = 0
        self.compose_count = 0

    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        self.call_count += 1
        return QoTState(gsnr_db=self.exact_gsnr, osnr_db=self.exact_gsnr, margin_db=0.0)

    def compose_gsnr(self, oms_sequence, direction, mode_id, slot):
        self.compose_count += 1
        return self.composed_gsnr


class _FakeQotNoCompose:
    """A QotEvaluator with no `compose_gsnr` at all -- every real test fake in
    the suite looks like this; `_best_feasible_mode` must fall through to its
    exact path (see composition_gate's own module docstring), so a run built
    from this evaluator must carry `gsnr_estimated=False`."""

    def __init__(self, gsnr: float):
        self.gsnr = gsnr

    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        return QoTState(gsnr_db=self.gsnr, osnr_db=self.gsnr, margin_db=0.0)


# composed - exact = 0.1 dB: comfortably under the real COMPOSITION_ERROR_BOUND_DB
# (0.23) so tests 1/3/4 (which don't monkeypatch the bound) never trip the
# watchdog by accident; test 2 shrinks the bound below 0.1 to force it.
COMPOSED_GSNR = 20.0
EXACT_GSNR = 19.9
assert (COMPOSED_GSNR - EXACT_GSNR) < COMPOSITION_ERROR_BOUND_DB


def test_committed_lightpath_carries_an_exact_margin_not_a_composed_one():
    """QoTState.margin_db is persisted, surfaced to agents, and summed into
    evaluate_objective's total_margin. A composed margin is optimistic by up to the
    declared bound; the committed value must be the exact one."""
    model = _one_route_model()
    qot = _CountingComposeQot(exact_gsnr=EXACT_GSNR, composed_gsnr=COMPOSED_GSNR)
    demands = [{"id": "d1", "src": "A", "dst": "Z", "demand_gbps": BITRATE_GBPS}]

    result, work = solve_allocation_model(model, qot, demands, {"A": 1, "Z": 1})

    assert result.status is SolverStatus.SOLUTION
    run = result.placements[0].new_lightpaths[0]
    assert run.gsnr_estimated is True
    assert run.gsnr_db == pytest.approx(COMPOSED_GSNR)   # the candidate's own record IS composed

    lp = work.list_lightpaths()[0]
    qs = work.get_qot_state(lp.id)
    expected_margin = EXACT_GSNR - REQUIRED_GSNR_DB - work.design_margin_db
    assert qs.gsnr_db == pytest.approx(EXACT_GSNR)
    assert qs.gsnr_db != pytest.approx(COMPOSED_GSNR)
    assert qs.margin_db == pytest.approx(expected_margin)

    obj = objective_mod.evaluate_objective(work)
    assert obj.total_margin == pytest.approx(expected_margin)


def test_watchdog_raises_a_typed_violation_when_the_bound_is_exceeded(monkeypatch):
    """Continuous audit: comparing composed against exact on the placement actually
    committed is how the measured bound stays honest as topologies change. Force it by
    monkeypatching COMPOSITION_ERROR_BOUND_DB down to something tiny."""
    tiny_bound = 0.01
    assert tiny_bound < abs(COMPOSED_GSNR - EXACT_GSNR)  # the 0.1 dB gap must trip it
    # objective.py binds COMPOSITION_ERROR_BOUND_DB as a bare module-global
    # (imported once from gnpy_adapter.composition), so the patch must land on
    # the objective module -- verify_and_reseed looks the name up there at
    # call time, matching this repo's existing monkeypatch convention (e.g.
    # test_propagation_budget.py patches allocation.harvest_qot the same way).
    monkeypatch.setattr(objective_mod, "COMPOSITION_ERROR_BOUND_DB", tiny_bound)

    model = _one_route_model()
    qot = _CountingComposeQot(exact_gsnr=EXACT_GSNR, composed_gsnr=COMPOSED_GSNR)
    demands = [{"id": "d1", "src": "A", "dst": "Z", "demand_gbps": BITRATE_GBPS}]

    result, work = solve_allocation_model(model, qot, demands, {"A": 1, "Z": 1})

    assert result.status is SolverStatus.SOLUTION
    assert len(result.violations) == 1
    v = result.violations[0]
    assert v.type is ViolationType.COMPOSITION_ERROR
    assert v.transient is False
    assert v.state_index == 0

    lp = work.list_lightpaths()[0]
    assert v.asset_id == lp.id

    assert v.detail["composed_gsnr_db"] == pytest.approx(COMPOSED_GSNR)
    assert v.detail["exact_gsnr_db"] == pytest.approx(EXACT_GSNR)
    assert v.detail["error_db"] == pytest.approx(COMPOSED_GSNR - EXACT_GSNR)
    assert v.detail["bound_db"] == pytest.approx(tiny_bound)

    # The watchdog firing does not change the re-seed decision: the stored
    # margin is still the EXACT one, bound-exceeded or not.
    qs = work.get_qot_state(lp.id)
    assert qs.gsnr_db == pytest.approx(EXACT_GSNR)

    # violations.py/views.py stay in lock-step with validate.py's ViolationType
    # (the same drift guard test_views.py's cross-validation exercises for
    # every OTHER violation type -- see views._violation_dict's docstring).
    flat = views_mod._violation_dict(v)
    assert "detail" not in flat
    instance = CompositionErrorViolation.model_validate(flat)
    assert instance.type == "composition_error"
    assert set(flat) == set(CompositionErrorViolation.model_fields)


def test_estimated_flag_marks_composed_runs_in_tool_results():
    """Spec 8.2's open question, answered: an agent reading a route_service result
    must be able to tell a composed (optimistic by up to the bound) GSNR from an exact
    one. route_service is read-only and never reaches the verify pass, so without this
    flag every restoration candidate would report an unqualified optimistic number."""
    demands = [{"id": "d1", "src": "A", "dst": "Z", "demand_gbps": BITRATE_GBPS}]

    composed_model = _one_route_model()
    composed_qot = _CountingComposeQot(exact_gsnr=EXACT_GSNR, composed_gsnr=COMPOSED_GSNR)
    composed_result = solve_allocation(composed_model, composed_qot, demands, {"A": 1, "Z": 1})
    composed_dict = views_mod.allocation_result_dict(composed_result)
    composed_run = composed_dict["placements"][0]["new_lightpaths"][0]
    assert composed_run["gsnr_estimated"] is True
    assert composed_run["gsnr_db"] == pytest.approx(COMPOSED_GSNR)

    exact_model = _one_route_model()
    exact_qot = _FakeQotNoCompose(gsnr=EXACT_GSNR)
    exact_result = solve_allocation(exact_model, exact_qot, demands, {"A": 1, "Z": 1})
    exact_dict = views_mod.allocation_result_dict(exact_result)
    exact_run = exact_dict["placements"][0]["new_lightpaths"][0]
    assert exact_run["gsnr_estimated"] is False
    assert exact_run["gsnr_db"] == pytest.approx(EXACT_GSNR)


def test_verify_pass_costs_two_propagations_per_new_run_not_per_candidate():
    """The hook is on the ACCEPTED placement in _pack, not on apply_candidate --
    verifying every scored candidate clone would cost more propagations than the search
    saved. Count them."""
    model = _two_route_model()
    qot = _CountingComposeQot(exact_gsnr=EXACT_GSNR, composed_gsnr=COMPOSED_GSNR)
    # A single PROTECTED demand: unlike the unprotected branch (which stops at
    # the first full-rate candidate -- Task 5's perf fix), the protected branch
    # always explores the WHOLE k-best frontier (disjoint_pairs searches WITHIN
    # the candidate set -- see _pack's own comment), so both of this model's
    # routes get scored via composition before working+protection (one new run
    # each) are accepted.
    demands = [{"id": "d1", "src": "A", "dst": "Z", "demand_gbps": BITRATE_GBPS,
               "protected": True}]

    result, work = solve_allocation_model(model, qot, demands, {"A": 2, "Z": 2})

    assert result.status is SolverStatus.SOLUTION
    total_new_runs = sum(len(p.new_lightpaths) + len(p.protection_new)
                         for p in result.placements)
    assert total_new_runs == 2   # one working + one protection new run
    assert all(r.gsnr_estimated
              for p in result.placements
              for r in (*p.new_lightpaths, *p.protection_new))

    # The full k-best frontier means place_demands scored MORE than one
    # candidate via composition -- compose_gsnr fired more times than the
    # exact evaluator did.
    assert qot.compose_count > qot.call_count

    # Yet the EXACT evaluator was only ever called by the post-accept verify
    # pass: 2 propagations (forward + backward) per accepted new run, never
    # once during candidate scoring/search of the frontier above.
    assert qot.call_count == 2 * total_new_runs


def test_ordering_fix_survives_a_later_demands_cross_lightpath_invalidation():
    """The brief flagged _pack's verify-then-refresh ordering as subtle:
    the per-iteration corrective re-seed loop blindly replays the WHOLE
    accumulated all_seeded list, including entries from demands placed in
    EARLIER iterations. Without refreshing an earlier demand's all_seeded
    entry right after ITS OWN verify_and_reseed call, a LATER demand that
    shares an OMS with it (and so invalidates its QoT via
    NetworkModel._invalidate_qot_sharing_oms when its own new lightpath is
    provisioned) would have that later iteration's corrective re-seed loop
    replay the STALE COMPOSED value still sitting in all_seeded, silently
    reverting demand 1's exact correction.

    Two full-mode demands on the single-OMS fixture: demand 2 cannot groom
    onto demand 1's (fully consumed) lightpath, so it must light its own new
    lightpath -- on the SAME (only) OMS, guaranteeing demand 2's provisioning
    invalidates demand 1's already-verified-and-corrected QoT. Asserting on
    BOTH lightpaths' FINAL stored QoTState (after the whole _pack run,
    including demand 2's own corrective re-seed and verify pass) proves the
    refresh-after-verify step actually survives this end-to-end, not just in
    isolation."""
    model = _one_route_model()
    qot = _CountingComposeQot(exact_gsnr=EXACT_GSNR, composed_gsnr=COMPOSED_GSNR)
    demands = [
        {"id": "d1", "src": "A", "dst": "Z", "demand_gbps": BITRATE_GBPS},
        {"id": "d2", "src": "A", "dst": "Z", "demand_gbps": BITRATE_GBPS},
    ]

    result, work = solve_allocation_model(model, qot, demands, {"A": 2, "Z": 2})

    assert result.status is SolverStatus.SOLUTION
    assert len(result.placements) == 2
    lp1_run = result.placements[0].new_lightpaths[0]
    lp2_run = result.placements[1].new_lightpaths[0]
    # Both new runs must land on the SAME (only) OMS for the cross-lightpath
    # invalidation this test targets to actually fire -- assert it rather
    # than assume it, so a future placement-engine change that stops sharing
    # the OMS fails loudly here instead of silently making this test vacuous.
    assert lp1_run.oms_sequence == lp2_run.oms_sequence == ("oms1",)
    assert lp1_run.gsnr_estimated and lp2_run.gsnr_estimated

    lightpaths = work.list_lightpaths()
    assert len(lightpaths) == 2
    for lp in lightpaths:
        qs = work.get_qot_state(lp.id)
        assert qs.gsnr_db == pytest.approx(EXACT_GSNR), (
            f"{lp.id}'s final stored QoTState is not the exact value -- "
            "a later demand's cross-lightpath invalidation + corrective "
            "re-seed replayed a stale composed value over an earlier "
            "demand's verify_and_reseed correction")
        assert qs.gsnr_db != pytest.approx(COMPOSED_GSNR)
