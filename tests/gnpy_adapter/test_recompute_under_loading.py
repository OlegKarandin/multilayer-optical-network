from multilayer_optical_network.model.qot_results import QoTResultStore
from multilayer_optical_network.gnpy_adapter.loading import Channel, LoadingState
from multilayer_optical_network.gnpy_adapter.adapter import (
    recompute_qot_under_loading,
    unattributed_channel_freqs_hz,
)
from multilayer_optical_network.testing import _model_with_lightpath


def test_unattributed_channel_freqs_hz_direct():
    n = _model_with_lightpath()  # lp1 committed on oms-AZ @ 193.4 THz
    committed_and_new = LoadingState(channels=(
        Channel(193.4e12, 100e9, None, "400G@7.1dB"),  # committed (lp1)
        Channel(193.2e12, 100e9, None, "400G@7.1dB"),  # not committed anywhere
    ))
    assert unattributed_channel_freqs_hz(n, committed_and_new) == (193.2e12,)
    # A loading that only restates what's already committed reports none.
    committed_only = LoadingState(channels=(
        Channel(193.4e12, 100e9, None, "400G@7.1dB"),
    ))
    assert unattributed_channel_freqs_hz(n, committed_only) == ()


def test_recompute_writes_state_and_returns_result_ids():
    n = _model_with_lightpath(); store = QoTResultStore()
    loading = LoadingState(channels=(
        Channel(193.4e12, 100e9, None, "400G@7.1dB"),
    ))
    results = recompute_qot_under_loading(model=n, store=store, loading=loading)
    state, rid = results["lp1"]
    # Recorded on the model.
    assert n.get_qot_state("lp1") == state
    # Breakdown reachable from the store.
    bd = store.get(rid)
    assert bd.snapshots
    # And capacity derives correctly.
    cap = n.ip_link_capacity_gbps("ip1")
    assert cap == (400.0 if state.mode_feasible else 0.0)
