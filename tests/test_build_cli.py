"""The offline operating-network builder CLI.

build_operating_network is monkeypatched throughout: this file tests the CLI's
wiring and its gating, not the packer (which tests/model/test_scenario.py
covers) and not GNPy (which would make these tests minutes long).
"""
import json
import sys
from pathlib import Path

import pytest

from multilayer_optical_network import build_cli
from multilayer_optical_network.data import reference_topology
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.scenario import ScenarioReport, ScenarioResult
from multilayer_optical_network.model.solvers import SolverStatus
from multilayer_optical_network.model.topology_import import model_from_abstract_graph
from multilayer_optical_network.state_file import load_model_from_state_file

TOPOLOGY = {
    "graph": {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {"src": "a", "dst": "b", "length_km": 80.0},
            {"src": "b", "dst": "c", "length_km": 80.0},
            {"src": "c", "dst": "a", "length_km": 80.0},
        ],
    },
}


def _report(**over):
    base = dict(status=SolverStatus.SOLUTION, achieved_mean_util=0.4,
                achieved_max_util=0.6, n_demands=4, total_offered_gbps=400.0,
                transponders_used=4, unplaced_count=0, scale=800.0, limit="none",
                unplaced_reasons={})
    base.update(over)
    return ScenarioReport(**base)


def _patch_build(monkeypatch, report):
    """Stand in for the real (minutes-to-hours, GNPy-driven) build."""
    modes = default_modes()
    model = model_from_abstract_graph(TOPOLOGY["graph"], modes=modes)

    def fake_build(_model, **_kw):
        return ScenarioResult(model, [], report, None)

    monkeypatch.setattr(build_cli, "build_operating_network", fake_build)


@pytest.fixture
def topo(tmp_path: Path) -> Path:
    p = tmp_path / "topo.json"
    p.write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    return p


def _run(monkeypatch, topo: Path, out: Path, *extra):
    monkeypatch.setattr(sys, "argv", ["multilayer-optical-network-build",
                                      "--topology", str(topo), "--out", str(out),
                                      *extra])
    build_cli.main()


def test_build_writes_a_state_file_the_server_can_load(monkeypatch, topo, tmp_path):
    _patch_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["format_version"] == 1
    assert doc["meta"]["topology_fingerprint"].startswith("sha256:")
    assert doc["meta"]["params"]["seed"] == 0
    assert doc["meta"]["report"]["limit"] == "none"
    # The whole point: the artifact round-trips back through the server's loader.
    modes = default_modes()
    assert load_model_from_state_file(topo, out, modes=modes) is not None


def test_build_refuses_to_write_when_no_demands_were_generated(monkeypatch, topo, tmp_path, capsys):
    # The silent-failure mode: gravity demands quantize to zero on a large
    # sparse topology, and the build "succeeds" with an empty network.
    _patch_build(monkeypatch, _report(n_demands=0))
    out = tmp_path / "state.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, topo, out)
    assert exc.value.code != 0
    assert not out.exists()
    err = capsys.readouterr().err
    assert "--pair-density" in err and "--unit-gbps" in err


def test_build_refuses_to_write_on_no_solution(monkeypatch, topo, tmp_path):
    _patch_build(monkeypatch, _report(status=SolverStatus.NO_SOLUTION))
    out = tmp_path / "state.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, topo, out)
    assert exc.value.code != 0
    assert not out.exists()


def test_build_warns_but_still_writes_when_some_demands_are_unplaced(
        monkeypatch, topo, tmp_path, capsys):
    # A partially-placed operating network is legitimate -- warn, don't fail.
    _patch_build(monkeypatch, _report(
        status=SolverStatus.PARTIAL, unplaced_count=8, limit="no_disjoint_pair",
        unplaced_reasons={"no disjoint feasible pair": 8}))
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)
    assert out.exists()
    err = capsys.readouterr().err
    assert "no disjoint feasible pair" in err
    assert "8" in err


def test_build_records_the_parameters_it_was_given(monkeypatch, topo, tmp_path):
    _patch_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out, "--seed", "7", "--pair-density", "0.02")
    params = json.loads(out.read_text(encoding="utf-8"))["meta"]["params"]
    assert params["seed"] == 7
    assert params["pair_density"] == 0.02


# --- Task A2: --design-margin-db threads into model construction + meta.params ---

def test_design_margin_db_flag_is_set_on_the_model_before_the_build_runs(
        monkeypatch, topo, tmp_path):
    """The model handed to build_operating_network must already carry the
    flag's value -- design_margin_db affects mode feasibility during the
    build itself, not just what gets recorded afterward."""
    modes = default_modes()
    fresh_model = model_from_abstract_graph(TOPOLOGY["graph"], modes=modes)
    seen = {}

    def fake_build(_model, **_kw):
        seen["design_margin_db"] = _model.design_margin_db
        return ScenarioResult(fresh_model, [], _report(), None)

    monkeypatch.setattr(build_cli, "build_operating_network", fake_build)
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out, "--design-margin-db", "1.25")
    assert seen["design_margin_db"] == 1.25


def test_design_margin_db_defaults_to_the_model_default_when_omitted(
        monkeypatch, topo, tmp_path):
    from multilayer_optical_network.model.optical_network import DEFAULT_DESIGN_MARGIN_DB

    modes = default_modes()
    fresh_model = model_from_abstract_graph(TOPOLOGY["graph"], modes=modes)
    seen = {}

    def fake_build(_model, **_kw):
        seen["design_margin_db"] = _model.design_margin_db
        return ScenarioResult(fresh_model, [], _report(), None)

    monkeypatch.setattr(build_cli, "build_operating_network", fake_build)
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)
    assert seen["design_margin_db"] == DEFAULT_DESIGN_MARGIN_DB


def test_design_margin_db_flag_is_recorded_in_meta_params(monkeypatch, topo, tmp_path):
    _patch_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out, "--design-margin-db", "1.25")
    params = json.loads(out.read_text(encoding="utf-8"))["meta"]["params"]
    assert params["design_margin_db"] == 1.25


def test_design_margin_db_is_none_in_meta_params_when_flag_omitted(
        monkeypatch, topo, tmp_path):
    _patch_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)
    params = json.loads(out.read_text(encoding="utf-8"))["meta"]["params"]
    assert params["design_margin_db"] is None


# --- Important #2: --out is validated before the (expensive) build runs ---

def test_out_parent_directory_must_exist_before_the_build_runs(
        monkeypatch, topo, tmp_path, capsys):
    def fake_build(_model, **_kw):
        raise AssertionError("build must not run when --out is invalid")
    monkeypatch.setattr(build_cli, "build_operating_network", fake_build)

    out = tmp_path / "does_not_exist" / "state.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, topo, out)
    assert exc.value.code != 0
    assert not out.exists()
    assert "does not exist" in capsys.readouterr().err


def test_out_in_an_existing_directory_proceeds_to_the_build(monkeypatch, topo, tmp_path):
    # The companion positive case: a valid --out must not be rejected.
    _patch_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)
    assert out.exists()


# --- Important #3: the CLI's flag -> builder-keyword wiring ---

def _capture_build(monkeypatch, report):
    """Like _patch_build, but records every keyword build_operating_network
    was actually called with, so a test can assert the flag -> kwarg mapping
    directly instead of just asserting the build "succeeded"."""
    modes = default_modes()
    model = model_from_abstract_graph(TOPOLOGY["graph"], modes=modes)
    captured: dict = {}

    def fake_build(_model, **kw):
        captured.update(kw)
        return ScenarioResult(model, [], report, None)

    monkeypatch.setattr(build_cli, "build_operating_network", fake_build)
    return captured


def test_flags_reach_the_correct_builder_keywords(monkeypatch, topo, tmp_path):
    captured = _capture_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out,
         "--seed", "3",
         "--target-mean-util", "0.55",
         "--max-util-cap", "0.8",
         "--pair-density", "0.1",
         "--unit-gbps", "50",
         "--protected-fraction", "0.6",
         "--max-iters", "9",
         "--protection-basis", "srlg",
         "--protection-level", "srlg",
         "--protection-best-effort")

    assert captured["seed"] == 3
    assert captured["target_mean_util"] == 0.55
    assert captured["max_util_cap"] == 0.8
    assert captured["pair_density"] == 0.1
    assert captured["unit_gbps"] == 50.0
    assert captured["protected_fraction"] == 0.6
    assert captured["max_iters"] == 9
    assert captured["protection_constraints"] == {
        "best_effort": True, "basis": "srlg", "level": "srlg"}


def test_protection_constraints_flows_through_to_the_builder(monkeypatch, topo, tmp_path):
    captured = _capture_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out,
         "--protection-basis", "risk_group",
         "--protection-level", "risk_group",
         "--protection-best-effort")
    assert captured["protection_constraints"] == {
        "best_effort": True, "basis": "risk_group", "level": "risk_group"}


def test_protection_constraints_is_none_without_any_protection_flag(
        monkeypatch, topo, tmp_path):
    captured = _capture_build(monkeypatch, _report())
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)
    assert captured["protection_constraints"] is None


def test_protection_constraints_direct_from_flags():
    args = build_cli._parser().parse_args([
        "--topology", "t.json", "--out", "o.json",
        "--protection-basis", "srlg", "--protection-level", "link",
        "--protection-best-effort"])
    assert build_cli._protection_constraints(args) == {
        "best_effort": True, "basis": "srlg", "level": "link"}


def test_protection_constraints_direct_none_when_no_flags_set():
    args = build_cli._parser().parse_args(["--topology", "t.json", "--out", "o.json"])
    assert build_cli._protection_constraints(args) is None


def test_protection_constraints_direct_best_effort_alone():
    # best_effort=True with no basis/level: still non-None, and the dict must
    # not silently gain unrequested basis/level keys.
    args = build_cli._parser().parse_args([
        "--topology", "t.json", "--out", "o.json", "--protection-best-effort"])
    assert build_cli._protection_constraints(args) == {"best_effort": True}


# --- Minor #7: NO_SOLUTION gate ordering and remediation hints ---

def test_no_solution_gate_is_checked_before_the_zero_demands_gate(
        monkeypatch, topo, tmp_path, capsys):
    # The <2-site early return in scenario.py reports BOTH n_demands=0 and
    # NO_SOLUTION with limit="none" -- the no_solution message (which can name
    # the site-count issue) must win over the pair-density/unit-gbps advice,
    # which cannot help a one-site topology.
    _patch_build(monkeypatch, _report(
        status=SolverStatus.NO_SOLUTION, n_demands=0, limit="none"))
    out = tmp_path / "state.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, topo, out)
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "no_solution" in err
    assert "--pair-density" not in err


def test_no_solution_gate_names_a_remediation_flag_for_a_known_limit(
        monkeypatch, topo, tmp_path, capsys):
    _patch_build(monkeypatch, _report(
        status=SolverStatus.NO_SOLUTION, limit="no_disjoint_pair"))
    out = tmp_path / "state.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, topo, out)
    assert exc.value.code != 0
    assert "--protection-best-effort" in capsys.readouterr().err


def test_unplaced_warning_names_a_remediation_flag_for_a_known_limit(
        monkeypatch, topo, tmp_path, capsys):
    _patch_build(monkeypatch, _report(
        status=SolverStatus.PARTIAL, unplaced_count=3, limit="max_util_cap",
        unplaced_reasons={"utilization cap": 3}))
    out = tmp_path / "state.json"
    _run(monkeypatch, topo, out)
    err = capsys.readouterr().err
    assert "--max-util-cap" in err or "--target-mean-util" in err


# --- Task 2: the CLI must wire a HarvestCache into the evaluator ---

def test_cli_wires_a_harvest_cache_into_the_evaluator(monkeypatch, tmp_path):
    """FillPolicy.FULL's harvest fast path is gated on `harvest_cache is not
    None` (allocation.make_adapter_evaluator). Without one, every candidate
    lambda re-propagates the same path through compute_qot -- the probe sits
    first in the loading tuple, so `_cache_key` differs per probe slot and the
    QoTCache cannot collapse them. The CLI must pass one."""
    from multilayer_optical_network import build_cli
    from multilayer_optical_network.model.qot_results import HarvestCache

    seen = {}
    real = build_cli.make_adapter_evaluator

    def _spy(model, store, **kw):
        seen.update(kw)
        return real(model, store, **kw)

    monkeypatch.setattr(build_cli, "make_adapter_evaluator", _spy)
    monkeypatch.setattr(build_cli, "build_operating_network",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))

    out = tmp_path / "state.json"
    monkeypatch.setattr(
        "sys.argv",
        ["multilayer-optical-network-build",
         "--topology", str(reference_topology("german_17")),
         "--out", str(out)])
    with pytest.raises(SystemExit):
        build_cli.main()

    assert isinstance(seen.get("harvest_cache"), HarvestCache), (
        f"CLI must pass a HarvestCache to make_adapter_evaluator; got {seen!r}")


# --- Task A8: the CLI must wire an IncrementCache into the evaluator ---

def test_cli_wires_an_increment_cache_into_the_evaluator(monkeypatch, tmp_path):
    """Composition (`AdapterEvaluator.compose_gsnr`) is gated on `increment_cache
    is not None` (allocation.make_adapter_evaluator's docstring). Without one,
    `_best_feasible_mode` always falls through to exact propagation and the
    ~3.6x propagation-count reduction tests/model/test_propagation_budget.py
    measures (69 -> 19 on the frozen german_17 fixture) never reaches
    production -- exactly the gap this test closes. Safe to wire
    unconditionally here: model.design_margin_db defaults to 0.5 dB, above
    composition.COMPOSITION_ERROR_BOUND_DB (0.23 dB), and every composed run
    is re-verified exactly on accept (objective.verify_and_reseed) regardless
    of the margin in effect."""
    from multilayer_optical_network import build_cli
    from multilayer_optical_network.model.qot_results import IncrementCache

    seen = {}
    real = build_cli.make_adapter_evaluator

    def _spy(model, store, **kw):
        seen.update(kw)
        return real(model, store, **kw)

    monkeypatch.setattr(build_cli, "make_adapter_evaluator", _spy)
    monkeypatch.setattr(build_cli, "build_operating_network",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))

    out = tmp_path / "state.json"
    monkeypatch.setattr(
        "sys.argv",
        ["multilayer-optical-network-build",
         "--topology", str(reference_topology("german_17")),
         "--out", str(out)])
    with pytest.raises(SystemExit):
        build_cli.main()

    assert isinstance(seen.get("increment_cache"), IncrementCache), (
        f"CLI must pass an IncrementCache to make_adapter_evaluator; got {seen!r}")
