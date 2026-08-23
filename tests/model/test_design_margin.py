# tests/model/test_design_margin.py
"""design_margin_db: an explicit modelling-uncertainty margin.

The model carried NO design margin before this: `sys_margins: 2` in the synthesized
equipment SI block (gnpy_adapter/synthesize.py:213) is declared and never read, because
the adapter propagates elements directly and bypasses gnpy's request.py, where that key
is consumed. Real line systems carry 1-3 dB for aging, ripple drift, connector
degradation and GN-model error, so this is a fidelity fix in its own right -- and it is
what makes per-OMS composition (a bounded, OPTIMISTIC approximation) safe to ship."""
import pytest

from multilayer_optical_network.model.allocation import _best_feasible_mode
from multilayer_optical_network.model.assets import (
    Amplifier, Fiber, FiberType, Lightpath, OMS, ROADM, TransceiverMode,
)
from multilayer_optical_network.model.ip_assets import IPLink, Router
from multilayer_optical_network.model.modes import ModeRegistry
from multilayer_optical_network.model.network import NetworkModel
from multilayer_optical_network.gnpy_adapter.composition import COMPOSITION_ERROR_BOUND_DB
from multilayer_optical_network.model.optical_network import DEFAULT_DESIGN_MARGIN_DB
from multilayer_optical_network.model.plan import Plan
from multilayer_optical_network.model.qot import QoTState
from multilayer_optical_network.model.qot_results import QoTResultStore
from multilayer_optical_network.model.validate import ViolationType, validate_plan


_MODES = ModeRegistry([
    TransceiverMode(id="hi", bitrate_gbps=400.0, required_gsnr_db=10.0,
                    symbol_rate_baud=87.5e9, channel_spacing_hz=100e9),
    TransceiverMode(id="lo", bitrate_gbps=200.0, required_gsnr_db=7.0,
                    symbol_rate_baud=87.5e9, channel_spacing_hz=100e9),
])


class _Qot:
    def __init__(self, gsnr): self.gsnr = gsnr
    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        return QoTState(gsnr_db=self.gsnr, osnr_db=30.0, margin_db=0.0)


def _model_with_one_lightpath(*, design_margin_db: float, mode_id: str,
                               gsnr_db: float) -> NetworkModel:
    """Modelled line-for-line on test_multilayer_graph.py's `_one_lightpath_model`:
    same ROADM-terminated single span, plus a Router/IPLink pair so
    `ip_link_capacity_gbps` has something to read."""
    n = NetworkModel(modes=_MODES, design_margin_db=design_margin_db)
    n.register_fiber_type(FiberType("SSMF", 0.2))
    for a in ("a1", "a2"):
        n.add_amplifier(Amplifier(id=a, type_variety="advanced_toy", gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber("fAB", "a1", "a2", 80.0, "SSMF"))
    for node in ("A", "B"):
        n.add_roadm(ROADM(id=f"roadm_{node}"))
    n.add_oms(OMS("oms-AB", "A", "B", ("roadm_A", "a1", "fAB", "a2")))
    n.add_lightpath(Lightpath("lp-AB", ("oms-AB",), mode_id, 193.4e12))
    n.set_qot_state("lp-AB", QoTState(
        gsnr_db=gsnr_db, osnr_db=30.0,
        margin_db=gsnr_db - _MODES.get(mode_id).required_gsnr_db - design_margin_db))
    n.add_router(Router("R1", "A"))
    n.add_router(Router("R2", "B"))
    n.add_ip_link(IPLink("ip-AB", "R1", "R2", "lp-AB"))
    return n


def test_default_is_zero_point_five_after_task_a7():
    """Task A7 flipped DEFAULT_DESIGN_MARGIN_DB from 0.0 (behaviour-preserving,
    Tasks A1-A6) to 0.5 (the reviewed, capacity-moving change) -- see
    optical_network.py's rationale comment above the constant.

    Also asserts the safety-invariant guard that constant's own comment
    promises: DEFAULT_DESIGN_MARGIN_DB must stay strictly above
    COMPOSITION_ERROR_BOUND_DB, or Task A5's composed-selection safety theorem
    (test_composition_gate.py's test_composed_selection_is_genuinely_feasible)
    no longer holds -- unlike that test, which only checks a locally-set
    model.design_margin_db = 0.5, this checks the actual SHIPPED default."""
    assert DEFAULT_DESIGN_MARGIN_DB == 0.5
    assert NetworkModel(modes=_MODES).design_margin_db == 0.5
    assert DEFAULT_DESIGN_MARGIN_DB > COMPOSITION_ERROR_BOUND_DB


def test_clone_preserves_design_margin():
    """clone() rebuilds the model via `type(self)(modes=..., grid=...)`; a constructor
    param not threaded there is silently reset to the default on every snapshot,
    branch and restore -- the exact failure clone()'s own docstring warns about."""
    m = NetworkModel(modes=_MODES, design_margin_db=1.25)
    assert m.clone().design_margin_db == 1.25


def test_mode_selection_respects_the_margin():
    """GSNR 10.4 dB clears `hi` (10.0) bare, but not with a 0.5 dB design margin --
    the selection must downshift to `lo` rather than pick a mode the line cannot
    actually carry under the declared uncertainty."""
    bare = NetworkModel(modes=_MODES, design_margin_db=0.0)
    guarded = NetworkModel(modes=_MODES, design_margin_db=0.5)
    q = _Qot(10.4)
    assert _best_feasible_mode(bare, q, ("oms-x",), None, "hi")[0].id == "hi"
    assert _best_feasible_mode(guarded, q, ("oms-x",), None, "hi")[0].id == "lo"


def test_no_feasible_mode_under_a_large_margin():
    m = NetworkModel(modes=_MODES, design_margin_db=6.0)
    mode, gsnr, composed = _best_feasible_mode(m, _Qot(10.4), ("oms-x",), None, "hi")
    assert mode is None and gsnr == 10.4
    assert composed is False   # _Qot has no compose_gsnr -- always the exact path


def test_margin_gate_drops_ip_capacity_to_zero():
    """CLAUDE.md's layer-consistency rule: a mode made infeasible by the design margin
    must take the bound IP link DOWN (capacity 0), not leave it at nominal line rate."""
    m = _model_with_one_lightpath(design_margin_db=0.0, mode_id="hi", gsnr_db=10.2)
    assert m.ip_link_capacity_gbps("ip-AB") == 400.0
    m2 = _model_with_one_lightpath(design_margin_db=0.5, mode_id="hi", gsnr_db=10.2)
    assert m2.ip_link_capacity_gbps("ip-AB") == 0.0


def test_mode_infeasible_violation_reports_raw_threshold_and_margin_separately():
    """`required_gsnr_db` in the violation detail stays the TRANSCEIVER threshold and
    `deficit_db` stays measured against it; the design margin is its own field. Baking
    the margin into the reported threshold would make the violation lie about the
    hardware."""
    m = _model_with_one_lightpath(design_margin_db=0.5, mode_id="hi", gsnr_db=10.2)
    report = validate_plan(m, Plan(ops=()), store=QoTResultStore())
    v, = [x for x in report.violations if x.type is ViolationType.MODE_INFEASIBLE]
    assert v.detail["required_gsnr_db"] == 10.0
    assert v.detail["design_margin_db"] == 0.5
    assert v.detail["deficit_db"] == pytest.approx(-0.2)
    assert v.detail["margin_db"] == pytest.approx(-0.3)
