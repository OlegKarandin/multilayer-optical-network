"""Step C2 (S4-6 + S8-5): recompute builds each lightpath's interferer comb from
its OWN OMS, not a global concat of every committed channel.

Two consequences of the old global comb are fixed:
  - over-count: a channel on a disjoint fiber was counted as a co-propagating
    interferer, inflating NLI on paths it never shares;
  - malformed NLI: two lightpaths reusing a wavelength on disjoint OMS produced
    two carriers at the same frequency in one SpectralInformation
    (slot_width = f[1]-f[0] = 0).
"""
from multilayer_optical_network.model.assets import Lightpath
from multilayer_optical_network.model.qot_results import QoTResultStore
from multilayer_optical_network.model.whatif import loading_from_model
from multilayer_optical_network.gnpy_adapter.adapter import recompute_qot_under_loading
from multilayer_optical_network.testing import MODE, ROUTE1, ROUTE2, _diamond_model


def _gsnr_of_lp1(add_disjoint_lp2: bool, lp2_freq_hz: float = 193.5e12) -> float:
    model = _diamond_model()
    model.add_lightpath(Lightpath(id="lp1", oms_sequence=ROUTE1,
                                  mode_id=MODE, center_freq_hz=193.4e12))
    if add_disjoint_lp2:
        model.add_lightpath(Lightpath(id="lp2", oms_sequence=ROUTE2,
                                      mode_id=MODE, center_freq_hz=lp2_freq_hz))
    recompute_qot_under_loading(model=model, store=QoTResultStore(),
                                loading=loading_from_model(model))
    return model.get_qot_state("lp1").gsnr_db


def test_disjoint_fiber_lightpath_is_not_an_interferer():
    # 193.7 THz keeps lp2 clear of the single-carrier dummy (probe + 100 GHz =
    # 193.5), so the old global comb's phantom interferer is genuinely exercised.
    g_alone = _gsnr_of_lp1(add_disjoint_lp2=False)
    g_with = _gsnr_of_lp1(add_disjoint_lp2=True, lp2_freq_hz=193.7e12)
    assert abs(g_with - g_alone) < 1e-6, (
        f"a lightpath on a disjoint fiber must not change lp1's QoT: "
        f"alone={g_alone:.6f} with-disjoint={g_with:.6f} dB"
    )


def test_wavelength_reuse_on_disjoint_fiber_gives_clean_single_carrier():
    # lp2 reuses lp1's exact wavelength on the disjoint route -> the old global
    # concat would emit two 193.4 THz carriers (malformed NLI). Per-path comb
    # keeps lp1 a clean single carrier equal to the isolated case.
    g_alone = _gsnr_of_lp1(add_disjoint_lp2=False)
    g_reuse = _gsnr_of_lp1(add_disjoint_lp2=True, lp2_freq_hz=193.4e12)
    assert abs(g_reuse - g_alone) < 1e-6, (
        f"wavelength reuse on a disjoint fiber must not corrupt lp1's QoT: "
        f"alone={g_alone:.6f} reuse={g_reuse:.6f} dB"
    )
