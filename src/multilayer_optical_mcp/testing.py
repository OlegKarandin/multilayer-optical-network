"""Shared synthesizable topology helpers for the Phase 7 tests.

Phase 7 was written before correctness batches C1-C6 landed. Those batches made
GNPy synthesis strict in three ways the plan's toy fixtures (a bare amp+fiber OMS)
no longer satisfy:
  * S3-11: every OMS endpoint must resolve to a registered ``roadm_<node>`` with a
    launch transceiver — a ROADM-less endpoint raises.
  * S4-2/S4-3: gated (worse-direction) QoT needs a *paired reverse OMS*
    ``(dst,src)`` for every forward ``(src,dst)`` leg, or backward propagation
    raises.
So any Phase 7 test whose `validate_plan`/recompute path runs real GNPy must build
a bidirectional, ROADM-terminated span. `add_bidir_span` centralizes that; the
forward OMS id is the caller-supplied `oms_id` so the plan's `"omsAB"`-style
references stay intact.
"""
from __future__ import annotations

from multilayer_optical_mcp.gnpy_adapter.loading import LoadingState
from multilayer_optical_mcp.model.assets import (
    Amplifier, Direction, Fiber, FiberType, Lightpath, OMS, ROADM,
    TransceiverMode, Transceiver,
)
from multilayer_optical_mcp.model.ip_assets import IPLink, Router
from multilayer_optical_mcp.model.modes import ModeRegistry, default_modes
from multilayer_optical_mcp.model.network import NetworkModel
from multilayer_optical_mcp.model.qot import QoTState
from multilayer_optical_mcp.model.scenario import build_operating_network
from multilayer_optical_mcp.model.topology_import import model_from_abstract_graph

# 400G needs 10 dB, 200G needs 7 dB; a single 80 km advanced-amp span delivers
# ~18.3 dB GSNR, so both modes sit comfortably above threshold — steady-state
# findings are driven by capacity, not by a marginal QoT.
MODES = ModeRegistry([
    TransceiverMode(id="400G", bitrate_gbps=400.0, required_gsnr_db=10.0,
                    symbol_rate_baud=87.5e9, channel_spacing_hz=100e9),
    TransceiverMode(id="200G", bitrate_gbps=200.0, required_gsnr_db=7.0,
                    symbol_rate_baud=43.75e9, channel_spacing_hz=100e9),
])


def new_model(modes: ModeRegistry | None = None) -> NetworkModel:
    m = NetworkModel(modes=modes or MODES)
    m.register_fiber_type(FiberType(type_variety="SSMF", loss_coef_db_per_km=0.2))
    return m


def _ensure_site(m: NetworkModel, node: str) -> None:
    rid = f"roadm_{node}"
    if rid not in m._roadms:
        m.add_roadm(ROADM(id=rid))
        m.add_transceiver(Transceiver(id=f"trx_{node}", site=node))


def _one_dir(m: NetworkModel, src: str, dst: str, tag: str) -> str:
    """One directed span in importer shape: roadm_<src> -> booster -> fiber ->
    preamp. Matches `topology_import._add_directed_oms` — the OMS `elements`
    start at the source ROADM (the drop `roadm_<dst>` is omitted, recovered by
    `terminal_roadm_id`). Link-level disjointness still compares SPANS, not the
    shared endpoint ROADM, because `path_basis_keys` excludes each path's own
    endpoint ROADM (see `exposure.path_endpoint_exclusions`). Returns the OMS id
    (== tag)."""
    m.add_amplifier(Amplifier(id=f"boost_{tag}", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    m.add_fiber(Fiber(id=f"f_{tag}", a_end=f"roadm_{src}", z_end=f"pre_{tag}",
                      length_km=80.0, type_variety="SSMF"))
    m.add_amplifier(Amplifier(id=f"pre_{tag}", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    m.add_oms(OMS(id=tag, src_node_id=src, dst_node_id=dst,
                  elements=(f"roadm_{src}", f"boost_{tag}", f"f_{tag}", f"pre_{tag}")))
    return tag


def add_bidir_span(m: NetworkModel, src: str, dst: str, oms_id: str) -> str:
    """Add a synthesizable bidirectional span. The FORWARD OMS is named `oms_id`;
    the paired reverse OMS (needed for backward QoT) is `oms_id + "_rev"`. Returns
    the forward OMS id."""
    _ensure_site(m, src)
    _ensure_site(m, dst)
    _one_dir(m, src, dst, oms_id)
    _one_dir(m, dst, src, oms_id + "_rev")
    return oms_id


# --------------------------------------------------------------------------
# Fixtures promoted from tests/gnpy_adapter/{test_compute_qot,
# test_recompute_under_loading,test_per_path_comb}.py and tests/test_state_file.py.
#
# These were originally private helpers cross-imported between test files (a
# simulator-side test file importing from another simulator-side test file, or
# from a server-side test file). That cross-import is fine only as long as
# every importer stays inside this one repo; a later phase splits this repo
# into a "simulator" library repo and a separate "MCP server" repo, carrying
# forward only tests/test_server*.py, tests/e2e/, tests/conftest.py, and
# tests/__init__.py. Anything a test_server*.py file needs must therefore come
# from shipped package code (here), not from another tests/*.py module that
# won't exist on the other side of the split. The originating test files still
# import these same names from here — this module is the single canonical
# copy, not a fork.
# --------------------------------------------------------------------------


def _toy_model(roll_off: float = 0.15) -> NetworkModel:
    """Symmetric two-span A<->Z toy topology (see
    tests/gnpy_adapter/test_compute_qot.py's ground-truth header for the pinned
    GSNR). Both directions terminate at a ROADM (S3-11 Option B) with their own
    physically separate reverse OMS, so backward QoT walks its own amp chain
    rather than a reversed copy of the forward element list."""
    reg = ModeRegistry([
        TransceiverMode(
            id="400G@7.1dB", bitrate_gbps=400.0, required_gsnr_db=7.1,
            symbol_rate_baud=87.5e9, channel_spacing_hz=100e9, roll_off=roll_off,
        ),
        TransceiverMode(
            id="800G@15.1dB", bitrate_gbps=800.0, required_gsnr_db=15.1,
            symbol_rate_baud=87.5e9, channel_spacing_hz=100e9, roll_off=roll_off,
        ),
        TransceiverMode(
            id="impossible", bitrate_gbps=1.0, required_gsnr_db=100.0,
            symbol_rate_baud=87.5e9, channel_spacing_hz=100e9, roll_off=roll_off,
        ),
    ])
    n = NetworkModel(modes=reg)
    n.register_fiber_type(FiberType(type_variety="SSMF", loss_coef_db_per_km=0.2))
    n.add_roadm(ROADM(id="roadm_A", target_pch_out_db=-20.0))
    n.add_transceiver(Transceiver(id="trx_A", site="A"))
    n.add_amplifier(Amplifier(
        id="booster A", type_variety="advanced_toy", gain_db=20.0, nf_db=5.5,
    ))
    n.add_fiber(Fiber(
        id="east fiber A to ILA", a_end="roadm_A", z_end="east edfa in ILA",
        length_km=80.0, type_variety="SSMF",
    ))
    n.add_amplifier(Amplifier(
        id="east edfa in ILA", type_variety="advanced_toy", gain_db=20.0, nf_db=5.5,
    ))
    n.add_fiber(Fiber(
        id="east fiber ILA to Z", a_end="east edfa in ILA", z_end="east edfa at Z",
        length_km=80.0, type_variety="SSMF",
    ))
    n.add_amplifier(Amplifier(
        id="east edfa at Z", type_variety="advanced_toy", gain_db=20.0, nf_db=5.5,
    ))
    n.add_oms(OMS(
        id="oms-AZ", src_node_id="A", dst_node_id="Z",
        elements=(
            "roadm_A", "booster A", "east fiber A to ILA", "east edfa in ILA",
            "east fiber ILA to Z", "east edfa at Z",
        ),
    ))
    # Physically separate reverse OMS (Z -> A): backward QoT walks its own amp
    # chain and add-side ROADM, not a reversed copy of the forward element list.
    n.add_roadm(ROADM(id="roadm_Z", target_pch_out_db=-20.0))
    n.add_transceiver(Transceiver(id="trx_Z", site="Z"))
    n.add_amplifier(Amplifier(id="booster Z", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber(id="west fiber Z to ILA", a_end="roadm_Z",
                      z_end="west edfa in ILA", length_km=80.0, type_variety="SSMF"))
    n.add_amplifier(Amplifier(id="west edfa in ILA", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_fiber(Fiber(id="west fiber ILA to A", a_end="west edfa in ILA",
                      z_end="west edfa at A", length_km=80.0, type_variety="SSMF"))
    n.add_amplifier(Amplifier(id="west edfa at A", type_variety="advanced_toy",
                              gain_db=20.0, nf_db=5.5))
    n.add_oms(OMS(
        id="oms-ZA", src_node_id="Z", dst_node_id="A",
        elements=(
            "roadm_Z", "booster Z", "west fiber Z to ILA", "west edfa in ILA",
            "west fiber ILA to A", "west edfa at A",
        ),
    ))
    return n


def _model_with_lightpath() -> NetworkModel:
    """`_toy_model()` plus a committed lightpath (lp1, oms-AZ @ 193.4 THz) and
    its bound IP link (R1 -> R2 via ip1)."""
    n = _toy_model()
    n.add_lightpath(Lightpath(id="lp1", oms_sequence=("oms-AZ",),
                              mode_id="400G@7.1dB", center_freq_hz=193.4e12))
    n.add_router(Router(id="R1", site="A"))
    n.add_router(Router(id="R2", site="Z"))
    n.add_ip_link(IPLink(id="ip1", a_router="R1", z_router="R2",
                         lightpath_id="lp1"))
    return n


MODE = "400G@7.1dB"
ROUTE1 = ("oms_A_M", "oms_M_Z")
ROUTE2 = ("oms_A_N", "oms_N_Z")


def _diamond_model() -> NetworkModel:
    """A-M-Z (ROUTE1) and A-N-Z (ROUTE2): two node-disjoint routes sharing only
    the add/drop ROADMs at A and Z. Every OMS has a unique (src,dst)."""
    mode = TransceiverMode(id=MODE, bitrate_gbps=400.0, required_gsnr_db=7.1,
                           symbol_rate_baud=87.5e9, channel_spacing_hz=100e9)
    graph = {
        "nodes": [{"id": "A"}, {"id": "M"}, {"id": "N"}, {"id": "Z"}],
        "edges": [
            {"src": "A", "dst": "M", "length_km": 80.0},
            {"src": "M", "dst": "Z", "length_km": 80.0},
            {"src": "A", "dst": "N", "length_km": 80.0},
            {"src": "N", "dst": "Z", "length_km": 80.0},
        ],
    }
    return model_from_abstract_graph(graph, modes=ModeRegistry([mode]))


TOPOLOGY = {
    "graph": {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {"src": "a", "dst": "b", "length_km": 80.0},
            {"src": "b", "dst": "c", "length_km": 80.0},
            {"src": "c", "dst": "a", "length_km": 80.0},
        ],
    },
    "srlgs": [{"id": "srlg_ab", "asset_ids": ["roadm_a", "roadm_b"]}],
}


class ConstQot:
    """Route-agnostic high GSNR: any path clears the top mode's threshold."""
    def __call__(self, *, oms_sequence, direction, mode_id, loading):
        return QoTState(gsnr_db=16.0, osnr_db=30.0, margin_db=0.0)


def _fake_settle(qot):
    """A GNPy-free stand-in for the real `recompute_qot_under_loading` settle
    pass, seeding every lightpath's QoT from `qot` instead."""
    def _settle(work):
        for lp in work.list_lightpaths():
            work.set_qot_state(lp.id, qot(oms_sequence=lp.oms_sequence,
                                          direction=Direction.FORWARD,
                                          mode_id=lp.mode_id,
                                          loading=LoadingState.empty()))
    return _settle


def _bare() -> NetworkModel:
    """`TOPOLOGY`'s 3-node ring, with no lightpaths/services placed yet."""
    modes = default_modes()
    return model_from_abstract_graph(TOPOLOGY["graph"], modes=modes)


def _built() -> NetworkModel:
    """A REAL packer-built operating network on `TOPOLOGY`'s 3-node ring,
    GNPy-free (see `ConstQot`/`_fake_settle`)."""
    qot = ConstQot()
    res = build_operating_network(_bare(), seed=0, qot=qot, target_mean_util=0.5,
                                  max_util_cap=0.95, max_iters=6,
                                  settle=_fake_settle(qot))
    assert res.model.list_services(), "fixture must actually place something"
    return res.model
